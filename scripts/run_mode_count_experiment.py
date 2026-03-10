"""Run the mode count experiment: vary m and measure ICL diversity metrics.

For each mode count m, generates n_per_mode * m responses using format-based
modes, computes ICL diversity metrics, and saves results to JSON.

Usage:
    # GPT-2, small mode counts (fits in 1024-token context)
    uv run python scripts/run_mode_count_experiment.py \
        --mode-counts 1 2 3 5 10 \
        --n-responses-per-mode 4 \
        --n-permutations 20 \
        --base-model gpt2 \
        --device cuda:0 \
        --batch-size 8 \
        --output results/mode_count/gpt2.json

    # Qwen 2.5-32B, full range (needs 2 GPUs)
    uv run python scripts/run_mode_count_experiment.py \
        --mode-counts 1 2 3 5 10 15 25 50 \
        --n-responses-per-mode 4 \
        --n-permutations 20 \
        --base-model Qwen/Qwen2.5-32B \
        --device auto --torch-dtype float16 \
        --batch-size 8 \
        --output results/mode_count/qwen2.5-32b.json
"""

import argparse
import json
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
    get_format_modes,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "results" / "mode_count" / "gpt2.json"
)

SEEDS = [42, 137, 256]


def estimate_tokens(responses: list[str], tokenizer: Any) -> int:
    """Estimate total tokens for the full concatenated context."""
    # Build the full context string the same way the metric does:
    # prompt + "Response A: ... Response B: ..." etc.
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
    n_per_mode: int,
    n_permutations: int,
    n_seeds: int,
    batch_size: int,
    base_model_name: str,
) -> dict[str, Any]:
    """Run mode count experiment across all (m, seed) combinations."""
    results: dict[str, Any] = {
        "experiment": "mode_count",
        "base_model": base_model_name,
        "n_permutations": n_permutations,
        "n_responses_per_mode": n_per_mode,
        "seeds": SEEDS[:n_seeds],
        "mode_names": MODE_NAMES,
        "prompt": PROMPT,
        "runs": [],
    }

    total_runs = len(mode_counts) * n_seeds
    pbar = tqdm(total=total_runs, desc="mode count runs")

    for m in mode_counts:
        for seed_idx in range(n_seeds):
            seed = SEEDS[seed_idx]
            t0 = time.time()

            # Generate responses (seed controls both mode selection and generation)
            responses, modes_used = generate_mode_count_responses(
                m, n_per_mode=n_per_mode, seed=seed,
            )
            n_total = len(responses)

            # Check token count
            n_tokens = estimate_tokens(responses, tokenizer)

            # Compute metrics
            metrics = compute_icl_diversity_metrics(
                model,
                tokenizer,
                PROMPT,
                responses,
                n_permutations=n_permutations,
                seed=seed,
                batch_size=batch_size,
            )

            elapsed = time.time() - t0

            run_entry = {
                "m": m,
                "seed": seed,
                "n_responses": n_total,
                "n_tokens_estimated": n_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "modes_used": modes_used,
                **metrics,
            }
            results["runs"].append(run_entry)

            pbar.set_postfix(m=m, seed=seed, E=f"{metrics['excess_entropy_E']:.2f}")
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
        default=[1, 2, 3, 5, 10],
        help="Mode counts to test (default: 1 2 3 5 10)",
    )
    parser.add_argument(
        "--n-responses-per-mode",
        type=int,
        default=4,
        help="Responses per mode (default: 4, total = m * this)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=20,
        help="Number of permutations for averaging (default: 20)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds for error bars (default: 3)",
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

    # Check context length for each m — fail fast if any exceed the limit
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx:
        print(f"Model max context length: {max_ctx} tokens")
        violations = []
        for m in args.mode_counts:
            responses, _ = generate_mode_count_responses(m, n_per_mode=args.n_responses_per_mode, seed=42)
            n_tokens = estimate_tokens(responses, tokenizer)
            ok = n_tokens < max_ctx
            print(f"  m={m:3d}, n={len(responses):3d}, ~{n_tokens:5d} tokens — {'OK' if ok else 'EXCEEDS CONTEXT'}")
            if not ok:
                violations.append((m, n_tokens))
        if violations:
            print(f"\nERROR: {len(violations)} mode count(s) exceed the model's {max_ctx}-token context window:")
            for m, n_tok in violations:
                print(f"  m={m}: ~{n_tok} tokens")
            print("Reduce --n-responses-per-mode, remove large mode counts, or use a model with a longer context window.")
            sys.exit(1)

    print(f"\nRunning experiment: mode_counts={args.mode_counts}, "
          f"n_per_mode={args.n_responses_per_mode}, "
          f"n_permutations={args.n_permutations}, "
          f"n_seeds={args.n_seeds}, batch_size={args.batch_size}")

    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        mode_counts=args.mode_counts,
        n_per_mode=args.n_responses_per_mode,
        n_permutations=args.n_permutations,
        n_seeds=args.n_seeds,
        batch_size=args.batch_size,
        base_model_name=args.base_model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
