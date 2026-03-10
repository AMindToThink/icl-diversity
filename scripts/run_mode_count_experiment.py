"""Run the mode count experiment: vary m and measure ICL diversity metrics.

For each mode count m, generates n total responses (fixed across all m) using
format-based modes, computes ICL diversity metrics, and saves results to JSON.

Usage:
    # GPT-2, full range with 50 outer draws
    uv run python scripts/run_mode_count_experiment.py \
        --mode-counts 1 2 3 4 5 6 7 8 9 10 \
        --n-responses 12 \
        --n-permutations 20 \
        --n-mode-draws 50 \
        --base-model gpt2 \
        --device cuda:1 \
        --batch-size 8 \
        --output results/mode_count/gpt2_50seeds.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity import compute_icl_diversity_metrics
from icl_diversity.mode_count_scenarios import (
    PROMPT,
    MODE_NAMES,
    generate_mode_count_responses,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "results" / "mode_count" / "gpt2_50seeds.json"
)


def estimate_tokens(responses: list[str], tokenizer: Any) -> int:
    """Estimate total tokens for the full concatenated context."""
    labels = [chr(ord("A") + i) if i < 26 else f"R{i}" for i in range(len(responses))]
    parts = [PROMPT + "\n\n"]
    for label, resp in zip(labels, responses):
        parts.append(f"Response {label}: {resp}\n\n")
    context = "".join(parts)
    return len(tokenizer.encode(context))


def run_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    mode_counts: list[int],
    n_responses: int,
    n_permutations: int,
    n_mode_draws: int,
    batch_size: int,
    base_model_name: str,
) -> dict[str, Any]:
    """Run mode count experiment across all (m, outer_seed) combinations.

    For each (m, outer_seed):
    - Select m modes and generate n responses (OUTER randomness: mode selection)
    - Call compute_icl_diversity_metrics with n_permutations (INNER randomness: ordering)
    - Store core's full metrics dict directly

    Aggregate across outer draws for error bars (mean ± SD).
    """
    # Generate deterministic outer seeds
    seed_rng = random.Random(42)
    outer_seeds = [seed_rng.randint(0, 2**31) for _ in range(n_mode_draws)]

    results: dict[str, Any] = {
        "experiment": "mode_count",
        "base_model": base_model_name,
        "n_permutations": n_permutations,
        "n_responses": n_responses,
        "n_mode_draws": n_mode_draws,
        "outer_seeds": outer_seeds,
        "mode_names": MODE_NAMES,
        "prompt": PROMPT,
        "runs": [],
    }

    total_runs = len(mode_counts) * n_mode_draws
    pbar = tqdm(total=total_runs, desc="mode count runs")

    for m in mode_counts:
        for draw_idx, outer_seed in enumerate(outer_seeds):
            t0 = time.time()

            # OUTER randomness: select modes and generate responses
            responses, modes_used = generate_mode_count_responses(
                m, n=n_responses, seed=outer_seed,
            )

            # INNER randomness: permutation averaging handled by core
            metrics = compute_icl_diversity_metrics(
                model,
                tokenizer,
                PROMPT,
                responses,
                n_permutations=n_permutations,
                seed=outer_seed,
                batch_size=batch_size,
            )

            elapsed = time.time() - t0
            n_tokens = estimate_tokens(responses, tokenizer)

            run_entry = {
                "m": m,
                "draw_idx": draw_idx,
                "outer_seed": outer_seed,
                "n_responses": n_responses,
                "n_tokens_estimated": n_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "modes_used": modes_used,
                # All metrics from core — single source of truth
                **{k: v for k, v in metrics.items()},
            }
            results["runs"].append(run_entry)

            pbar.set_postfix(
                m=m,
                draw=f"{draw_idx + 1}/{n_mode_draws}",
                E=f"{metrics['excess_entropy_E']:.1f}",
            )
            pbar.update(1)

    pbar.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mode count experiment for ICL diversity metric"
    )
    parser.add_argument(
        "--mode-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Mode counts to test (default: 1 2 3 4 5 6 7 8 9 10)",
    )
    parser.add_argument(
        "--n-responses",
        type=int,
        default=12,
        help="Total responses per run, fixed across all m (default: 12)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=20,
        help="Number of permutations for averaging (default: 20)",
    )
    parser.add_argument(
        "--n-mode-draws",
        type=int,
        default=50,
        help="Number of outer draws (mode selection seeds) for error bars (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPU parallelism (default: 8)",
    )
    parser.add_argument(
        "--base-model",
        default="gpt2",
        help="HuggingFace model ID (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Device: cpu, cuda, cuda:0, or auto (multi-GPU)',
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Model dtype: float16, bfloat16, float32",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Resolve dtype
    torch_dtype = None
    if args.torch_dtype is not None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(args.torch_dtype)
        if torch_dtype is None:
            print(f"Unknown dtype: {args.torch_dtype}")
            sys.exit(1)

    # Load model
    use_device_map = args.device == "auto"
    print(f"Loading {args.base_model} (dtype={args.torch_dtype}, device={args.device})...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    load_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if use_device_map:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if not use_device_map and args.device != "cpu":
        model = model.to(args.device)
    model.eval()

    # Validate mode counts
    for m in args.mode_counts:
        if m < 1 or m > 50:
            print(f"Invalid mode count: {m}. Must be 1-50.")
            sys.exit(1)

    # Check context length
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx:
        print(f"Model max context length: {max_ctx} tokens")
        check_seeds = [42, 137, 256, 0, 999]
        violations = []
        for m in args.mode_counts:
            worst_tokens = 0
            for cs in check_seeds:
                responses, _ = generate_mode_count_responses(m, n=args.n_responses, seed=cs)
                n_tokens = estimate_tokens(responses, tokenizer)
                worst_tokens = max(worst_tokens, n_tokens)
            ok = worst_tokens < max_ctx
            print(f"  m={m:3d}, n={args.n_responses:3d}, ~{worst_tokens:5d} tokens (worst-case) — {'OK' if ok else 'EXCEEDS CONTEXT'}")
            if not ok:
                violations.append((m, worst_tokens))
        if violations:
            print(f"\nERROR: {len(violations)} mode count(s) exceed the model's {max_ctx}-token context window:")
            for m, n_tok in violations:
                print(f"  m={m}: ~{n_tok} tokens")
            print("Reduce --n-responses, remove large mode counts, or use a model with a longer context window.")
            sys.exit(1)

    print(f"\nRunning experiment: mode_counts={args.mode_counts}, "
          f"n_responses={args.n_responses}, "
          f"n_permutations={args.n_permutations}, "
          f"n_mode_draws={args.n_mode_draws}, batch_size={args.batch_size}")

    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        mode_counts=args.mode_counts,
        n_responses=args.n_responses,
        n_permutations=args.n_permutations,
        n_mode_draws=args.n_mode_draws,
        batch_size=args.batch_size,
        base_model_name=args.base_model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
