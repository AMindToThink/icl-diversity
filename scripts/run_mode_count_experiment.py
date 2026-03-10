"""Run the mode count experiment: vary m and measure ICL diversity metrics.

For each mode count m, generates n total responses (fixed across all m) using
format-based modes, computes ICL diversity metrics, and saves results to JSON.

Usage:
    # GPT-2, small mode counts (fits in 1024-token context)
    uv run python scripts/run_mode_count_experiment.py \
        --mode-counts 1 2 3 5 10 \
        --n-responses 20 \
        --n-permutations 20 \
        --base-model gpt2 \
        --device cuda:0 \
        --batch-size 8 \
        --output results/mode_count/gpt2.json

    # Qwen 2.5-32B, full range (needs 2 GPUs)
    uv run python scripts/run_mode_count_experiment.py \
        --mode-counts 1 2 3 5 10 15 25 50 \
        --n-responses 20 \
        --n-permutations 20 \
        --base-model Qwen/Qwen2.5-32B \
        --device auto --torch-dtype float16 \
        --batch-size 8 \
        --output results/mode_count/qwen2.5-32b.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
    n_responses: int,
    n_permutations: int,
    n_seeds: int,
    batch_size: int,
    base_model_name: str,
) -> dict[str, Any]:
    """Run mode count experiment across all (m, seed) combinations.

    For each (m, seed), runs n_permutations trials. Each trial independently
    selects m random modes, generates n_responses total responses, and computes
    a single-pass a_k curve. The a_k curves are then averaged across trials,
    and aggregate metrics (E, C, D) are computed from the averaged curve.

    All m values produce the same number of responses (n_responses), so a_k
    curves have the same x-axis length for direct comparison.

    This design means that permutation averaging captures both response-ordering
    variance AND mode-selection variance.
    """
    results: dict[str, Any] = {
        "experiment": "mode_count",
        "base_model": base_model_name,
        "n_permutations": n_permutations,
        "n_responses": n_responses,
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

            # Generate sub-seeds for each permutation from the main seed
            rng = random.Random(seed)
            perm_seeds = [rng.randint(0, 2**31) for _ in range(n_permutations)]

            per_perm_curves: list[list[float]] = []
            per_perm_byte_counts: list[list[float]] = []
            all_modes_used: list[list[str]] = []

            perm_pbar = tqdm(
                perm_seeds, desc="  permutations", leave=False,
            )
            for perm_seed in perm_pbar:
                # Each permutation: fresh mode selection + fresh responses
                responses, modes_used = generate_mode_count_responses(
                    m, n=n_responses, seed=perm_seed,
                )
                all_modes_used.append(modes_used)

                # Compute with n_permutations=1 (ordering is already randomized
                # by generate_mode_count_responses via _generate_high_diversity_responses)
                metrics = compute_icl_diversity_metrics(
                    model,
                    tokenizer,
                    PROMPT,
                    responses,
                    n_permutations=1,
                    seed=perm_seed,
                    batch_size=batch_size,
                )
                per_perm_curves.append(metrics["a_k_curve"])
                per_perm_byte_counts.append(metrics["a_k_byte_counts"])

            perm_pbar.close()

            # Aggregate across permutations
            n_responses = len(per_perm_curves[0])
            avg_curve = np.mean(per_perm_curves, axis=0).tolist()
            avg_byte_counts = np.mean(per_perm_byte_counts, axis=0).tolist()
            avg_curve_per_byte = [
                t / b if b > 0 else 0.0
                for t, b in zip(avg_curve, avg_byte_counts)
            ]

            # Compute aggregate metrics from averaged curve
            # E = a_1 - a_n (total bits)
            excess_entropy_E = avg_curve[0] - avg_curve[-1]
            # E_rate = E / mean_byte_length (per-byte)
            mean_byte_length = float(np.mean(avg_byte_counts))
            E_rate = excess_entropy_E / mean_byte_length if mean_byte_length > 0 else 0.0

            # Unconditional surprise = a_1 (first point, no conditioning)
            unconditional_mean = float(np.mean([c[0] for c in per_perm_curves]))
            conditional_mean = float(np.mean([c[-1] for c in per_perm_curves]))

            # Coherence C
            coherence_C = 1.0 - conditional_mean / unconditional_mean if unconditional_mean > 0 else 0.0
            diversity_score_D = coherence_C * excess_entropy_E
            D_rate = coherence_C * E_rate

            # Unique modes seen across all permutations
            all_unique_modes = sorted(set(
                name for modes in all_modes_used for name in modes
            ))

            elapsed = time.time() - t0
            n_tokens = estimate_tokens(
                generate_mode_count_responses(m, n=n_responses, seed=perm_seeds[0])[0],
                tokenizer,
            )

            run_entry = {
                "m": m,
                "seed": seed,
                "n_responses": n_responses,
                "n_tokens_estimated": n_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "modes_used_unique": all_unique_modes,
                "n_unique_mode_sets": len(set(tuple(m) for m in all_modes_used)),
                "a_k_curve": avg_curve,
                "a_k_curve_per_byte": avg_curve_per_byte,
                "a_k_byte_counts": avg_byte_counts,
                "per_permutation_a_k_curves": per_perm_curves,
                "per_permutation_byte_counts": per_perm_byte_counts,
                "excess_entropy_E": excess_entropy_E,
                "E_rate": E_rate,
                "coherence_C": coherence_C,
                "diversity_score_D": diversity_score_D,
                "diversity_score_D_rate": D_rate,
                "mean_byte_length": mean_byte_length,
            }
            results["runs"].append(run_entry)

            pbar.set_postfix(m=m, seed=seed, E=f"{excess_entropy_E:.2f}")
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
        "--n-responses",
        type=int,
        default=20,
        help="Total responses per run, fixed across all m (default: 20)",
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

    # Check context length — since n is fixed across all m, different mode
    # selections produce different-length text. Check a few seeds per m to
    # find worst-case token counts.
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
          f"n_seeds={args.n_seeds}, batch_size={args.batch_size}")

    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        mode_counts=args.mode_counts,
        n_responses=args.n_responses,
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
