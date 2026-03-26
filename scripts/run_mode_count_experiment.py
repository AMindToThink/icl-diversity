"""Run the mode count experiment: vary m and measure ICL diversity metrics.

For each mode count m, runs N fully independent draws. Each draw independently
selects m modes, generates n responses, shuffles the response order, and computes
metrics via compute_icl_diversity_metrics. All metrics come from core.py (single
source of truth). Error bars are computed across the N independent draws.

Usage:
    # GPT-2, 1000 independent draws per m
    uv run python scripts/run_mode_count_experiment.py \
        --mode-counts 1 2 3 4 5 6 7 8 9 10 \
        --n-responses 12 \
        --n-draws 1000 \
        --base-model gpt2 \
        --device cuda:1 \
        --batch-size 32 \
        --wandb \
        --output results/mode_count/gpt2_independent.json
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

from icl_diversity import APIModel, compute_icl_diversity_metrics
from icl_diversity.mode_count_scenarios import (
    PROMPT,
    MODE_NAMES,
    generate_mode_count_responses,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "mode_count"
    / "gpt2_1k_draws.json"
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
    n_draws: int,
    batch_size: int,
    base_model_name: str,
    use_wandb: bool = False,
) -> dict[str, Any]:
    """Run mode count experiment with fully independent draws.

    For each (m, draw_seed):
    - Select m modes and generate n responses (fresh each draw)
    - Shuffle response order (removes cycling-order artifact)
    - Call compute_icl_diversity_metrics with n_permutations=1
    - Store core's full metrics dict

    All draws are statistically independent. Error bars (mean ± SD)
    across draws give honest uncertainty estimates.
    """
    # Generate deterministic, unique seeds for all draws
    seed_rng = random.Random(42)
    draw_seeds = [seed_rng.randint(0, 2**31) for _ in range(n_draws)]

    results: dict[str, Any] = {
        "experiment": "mode_count",
        "base_model": base_model_name,
        "n_responses": n_responses,
        "n_draws": n_draws,
        "draw_seeds": draw_seeds,
        "mode_names": MODE_NAMES,
        "prompt": PROMPT,
        "runs": [],
    }

    total_runs = len(mode_counts) * n_draws
    pbar = tqdm(total=total_runs, desc="mode count runs")

    for m in mode_counts:
        for draw_idx, draw_seed in enumerate(draw_seeds):
            t0 = time.time()

            # Fresh mode selection + response generation
            responses, modes_used = generate_mode_count_responses(
                m, n=n_responses, seed=draw_seed,
            )

            # Shuffle with a derived seed to decouple from generation RNG
            shuffle_rng = random.Random(draw_seed ^ 0x5A5A5A5A)
            shuffle_rng.shuffle(responses)

            # n_permutations=1: each draw is already a fresh random ordering
            # (shuffled above). Permutation averaging is unnecessary because
            # statistical power comes from the outer loop (n_draws independent
            # draws per m), not from inner permutation averaging. Each run's
            # a_k_curve is therefore a single-ordering curve (noisy per-run,
            # but the mean across 1000 draws is smooth).
            metrics = compute_icl_diversity_metrics(
                model,
                tokenizer,
                PROMPT,
                responses,
                n_permutations=1,
                seed=draw_seed,
                batch_size=batch_size,
            )

            elapsed = time.time() - t0
            n_tokens = estimate_tokens(responses, tokenizer)

            run_entry = {
                "m": m,
                "draw_idx": draw_idx,
                "draw_seed": draw_seed,
                "n_responses": n_responses,
                "n_tokens_estimated": n_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "modes_used": modes_used,
                # All metrics from core — single source of truth
                **{k: v for k, v in metrics.items()},
            }
            results["runs"].append(run_entry)

            if use_wandb:
                import wandb
                wandb.log({
                    "m": m,
                    "draw_idx": draw_idx,
                    "E": metrics["excess_entropy_E"],
                    "C": metrics["coherence_C"],
                    "D": metrics["diversity_score_D"],
                    "sigma_l": metrics["coherence_spread_sigma"],
                    "a_n": metrics["a_k_curve"][-1],
                    "elapsed": elapsed,
                })

            pbar.set_postfix(
                m=m,
                draw=f"{draw_idx + 1}/{n_draws}",
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
        "--n-draws",
        type=int,
        default=1000,
        help="Number of fully independent draws per m (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU parallelism (default: 32)",
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
        "--provider",
        choices=["local", "together", "fireworks"],
        default="local",
        help="Model provider: local (HuggingFace), together, or fireworks (default: local)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for provider (default: uses env var)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging (requires wandb login)",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.15,
        help="Max fraction of GPU memory to use (default: 0.15 → ~7GB on 46GB GPU)",
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

    # Set GPU memory cap before loading model
    if args.provider == "local" and args.device.startswith("cuda") and args.device != "auto":
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device_idx)
        print(f"GPU memory cap: {args.gpu_memory_fraction:.0%} of device {device_idx}")

    # Initialize W&B if requested
    if args.wandb:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")

        import wandb
        wandb.init(
            project="icl-diversity",
            config={
                "experiment": "mode_count",
                "base_model": args.base_model,
                "provider": args.provider,
                "mode_counts": args.mode_counts,
                "n_responses": args.n_responses,
                "n_draws": args.n_draws,
                "batch_size": args.batch_size,
                "device": args.device,
                "gpu_memory_fraction": args.gpu_memory_fraction,
            },
        )

    if args.provider != "local":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")

        print(f"Using API model: {args.base_model} via {args.provider}")
        model = APIModel(
            model_name=args.base_model,
            provider=args.provider,
            api_key=args.api_key,
            max_concurrent_requests=args.max_concurrent,
        )
        tokenizer = model.tokenizer
    else:
        # Load local model
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
            print(
                f"  m={m:3d}, n={args.n_responses:3d}, ~{worst_tokens:5d} tokens "
                f"(worst-case) — {'OK' if ok else 'EXCEEDS CONTEXT'}"
            )
            if not ok:
                violations.append((m, worst_tokens))
        if violations:
            print(
                f"\nERROR: {len(violations)} mode count(s) exceed the model's "
                f"{max_ctx}-token context window:"
            )
            for m, n_tok in violations:
                print(f"  m={m}: ~{n_tok} tokens")
            print(
                "Reduce --n-responses, remove large mode counts, or use a model "
                "with a longer context window."
            )
            sys.exit(1)

    print(
        f"\nRunning experiment: mode_counts={args.mode_counts}, "
        f"n_responses={args.n_responses}, "
        f"n_draws={args.n_draws}, batch_size={args.batch_size}"
    )

    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        mode_counts=args.mode_counts,
        n_responses=args.n_responses,
        n_draws=args.n_draws,
        batch_size=args.batch_size,
        base_model_name=args.base_model,
        use_wandb=args.wandb,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
