"""Run mode count experiment across a temperature grid.

Reuses the mode count experiment infrastructure (m modes, n responses, N
independent draws) but sweeps temperature. This gives much more statistical
power than the 5-scenario temperature experiment because we have 10 mode
counts x 200 draws = 2000 runs per temperature.

Statistical Hypotheses
======================

**H1' — Within-m Variance Reduction**
  H0: std_D(T, m) = std_D(1.0, m)  for all m, for T > 1.
  H1': std_D(T, m) < std_D(1.0, m)  (temperature reduces draw-to-draw noise).
  Test: For each T > 1, paired Wilcoxon signed-rank across the 10 mode counts
      comparing std(D) at T vs std(D) at T=1.0.

**H2' — Mode Count Ordering Preservation**
  H0: Spearman rho between mean_D(m) at T and mean_D(m) at T=1.0 = 0.
  H1': rho > 0  (mode count ordering is preserved across temperatures).
  Test: Spearman rank correlation on the 10 mode-count means. With 10 items
      (vs 5 scenarios), this test has much more power.

**H3' — Discriminability (Signal-to-Noise)**
  H0: Cohen's d between adjacent mode counts is the same at T > 1 and T = 1.0.
  H1': Cohen's d increases at T > 1  (noise shrinks faster than signal, so
      adjacent mode counts become easier to distinguish).
  Test: Compute Cohen's d = (mean_D(m+1) - mean_D(m)) / pooled_std for each
      adjacent pair. Compare median Cohen's d at T vs T=1.0.

**H4' — Sample Efficiency**
  H0: The number of draws needed to correctly rank all 10 mode counts is
      the same at T > 1 and T = 1.0.
  H1': Fewer draws are needed at T > 1  (the practical payoff).
  Test: Subsample n draws from 200, compute Spearman rho of subsampled
      mean_D(m) vs ground truth (T=1.0, all draws). Bootstrap comparison of
      rho distributions at T > 1 vs T = 1.0 via Mann-Whitney U.

Usage:
    uv run python scripts/run_mode_count_temperature.py --device cuda:1
    uv run python scripts/run_mode_count_temperature.py --device cuda:1 --temperatures 0.5,1.0,1.5,2.0 --n-draws 200
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

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "results" / "mode_count_temperature"
)
DEFAULT_TEMPERATURES = [0.5, 1.0, 1.5, 2.0, 3.0]


def estimate_tokens(responses: list[str], tokenizer: Any) -> int:
    """Estimate total tokens for the full concatenated context."""
    labels = [chr(ord("A") + i) if i < 26 else f"R{i}" for i in range(len(responses))]
    parts = [PROMPT + "\n\n"]
    for label, resp in zip(labels, responses):
        parts.append(f"Response {label}: {resp}\n\n")
    context = "".join(parts)
    return len(tokenizer.encode(context))


def run_single_temperature(
    model: torch.nn.Module,
    tokenizer: Any,
    mode_counts: list[int],
    n_responses: int,
    n_draws: int,
    batch_size: int,
    temperature: float,
    draw_seeds: list[int],
) -> list[dict[str, Any]]:
    """Run mode count experiment at one temperature. Returns list of run entries."""
    runs: list[dict[str, Any]] = []
    total_runs = len(mode_counts) * n_draws
    pbar = tqdm(total=total_runs, desc=f"  T={temperature}", leave=False)

    for m in mode_counts:
        for draw_idx, draw_seed in enumerate(draw_seeds):
            t0 = time.time()

            responses, modes_used = generate_mode_count_responses(
                m, n=n_responses, seed=draw_seed,
            )

            shuffle_rng = random.Random(draw_seed ^ 0x5A5A5A5A)
            shuffle_rng.shuffle(responses)

            metrics = compute_icl_diversity_metrics(
                model,
                tokenizer,
                PROMPT,
                responses,
                n_permutations=1,
                seed=draw_seed,
                batch_size=batch_size,
                temperature=temperature,
            )

            elapsed = time.time() - t0

            run_entry = {
                "m": m,
                "draw_idx": draw_idx,
                "draw_seed": draw_seed,
                "temperature": temperature,
                "modes_used": modes_used,
                "elapsed_seconds": round(elapsed, 2),
                **{k: v for k, v in metrics.items()},
            }
            runs.append(run_entry)

            pbar.set_postfix(m=m, draw=f"{draw_idx + 1}/{n_draws}")
            pbar.update(1)

    pbar.close()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mode count experiment across temperature grid"
    )
    parser.add_argument(
        "--mode-counts", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Mode counts to test (default: 1..10)",
    )
    parser.add_argument(
        "--n-responses", type=int, default=12,
        help="Responses per run (default: 12)",
    )
    parser.add_argument(
        "--n-draws", type=int, default=200,
        help="Independent draws per mode count (default: 200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for GPU parallelism (default: 32)",
    )
    parser.add_argument(
        "--base-model", default="gpt2",
        help="HuggingFace model ID (default: gpt2)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: cpu, cuda, cuda:0, or auto",
    )
    parser.add_argument(
        "--torch-dtype", default=None,
        help="Model dtype: float16, bfloat16, float32",
    )
    parser.add_argument(
        "--temperatures",
        default=",".join(str(t) for t in DEFAULT_TEMPERATURES),
        help=f"Comma-separated temperatures (default: {','.join(str(t) for t in DEFAULT_TEMPERATURES)})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=0.15,
        help="Max fraction of GPU memory to use (default: 0.15)",
    )
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]

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

    # Set GPU memory cap
    if args.device.startswith("cuda") and args.device != "auto":
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device_idx)
        print(f"GPU memory cap: {args.gpu_memory_fraction:.0%} of device {device_idx}")

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

    # Generate deterministic seeds shared across all temperatures
    seed_rng = random.Random(42)
    draw_seeds = [seed_rng.randint(0, 2**31) for _ in range(args.n_draws)]

    # Context length check
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx:
        for m in args.mode_counts:
            responses, _ = generate_mode_count_responses(m, n=args.n_responses, seed=42)
            n_tokens = estimate_tokens(responses, tokenizer)
            if n_tokens >= max_ctx:
                print(f"ERROR: m={m} needs ~{n_tokens} tokens, model max is {max_ctx}")
                sys.exit(1)

    print(
        f"\nMode counts: {args.mode_counts}, n_responses={args.n_responses}, "
        f"n_draws={args.n_draws}, temperatures={temperatures}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for temp in temperatures:
        print(f"\n=== Temperature: {temp} ===")
        runs = run_single_temperature(
            model, tokenizer, args.mode_counts, args.n_responses,
            args.n_draws, args.batch_size, temp, draw_seeds,
        )

        result = {
            "experiment": "mode_count_temperature",
            "base_model": args.base_model,
            "temperature": temp,
            "n_responses": args.n_responses,
            "n_draws": args.n_draws,
            "mode_counts": args.mode_counts,
            "draw_seeds": draw_seeds,
            "mode_names": MODE_NAMES,
            "prompt": PROMPT,
            "runs": runs,
        }

        out_path = args.output_dir / f"T_{temp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
