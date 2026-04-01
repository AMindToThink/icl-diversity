"""Estimate Tinker API cost for running ICL diversity metrics on Tevet datasets.

Reads the Tevet CSVs, tokenizes each sample using the target model's tokenizer
(via Tinker's get_tokenizer()), and computes:
  - Total tokens per dataset (unconditional + progressive × n_permutations)
  - Total API calls
  - Estimated cost at the configured price per million tokens

Usage:
    uv run python scripts/estimate_tinker_cost.py \
        --base-model Qwen/Qwen3-30B-A3B-Base \
        --n-permutations 50 \
        --datasets con_test_200 mcdiv_nuggets

    # See all available datasets
    uv run python scripts/estimate_tinker_cost.py --list-datasets
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_diversity.core import format_conditioning_context  # noqa: E402

# Tinker prefill pricing (USD per million tokens) as of 2026-03-30.
# Source: https://thinkingmachines.ai/tinker/
# compute_logprobs is a prefill-only operation.
TINKER_PRICES: dict[str, float] = {
    "meta-llama/Llama-3.2-1B": 0.03,
    "meta-llama/Llama-3.2-3B": 0.06,
    "meta-llama/Llama-3.1-8B": 0.13,
    "Qwen/Qwen3-8B-Base": 0.13,
    "Qwen/Qwen3-30B-A3B-Base": 0.12,
    "meta-llama/Llama-3.1-70B": 0.70,
}

# Dataset templates: same as compute_icl_metrics_for_tevet.py
EXPERIMENT_TEMPLATES: dict[str, dict[str, str]] = {
    "con_test_200": {
        "story_gen": "conTest/con_test_200_with_hds_story_gen.csv",
        "resp_gen": "conTest/con_test_200_with_hds_resp_gen.csv",
        "prompt_gen": "conTest/con_test_200_with_hds_prompt_gen.csv",
    },
    "dec_test_200": {
        "story_gen": "decTest/dec_test_200_with_hds_story_gen.csv",
        "resp_gen": "decTest/dec_test_200_with_hds_resp_gen.csv",
        "prompt_gen": "decTest/dec_test_200_with_hds_prompt_gen.csv",
    },
    "dec_test_1000": {
        "story_gen": "decTest/dec_test_1000_no_hds_story_gen.csv",
        "resp_gen": "decTest/dec_test_1000_no_hds_resp_gen.csv",
        "prompt_gen": "decTest/dec_test_1000_no_hds_prompt_gen.csv",
    },
    "mcdiv_nuggets": {
        "story_gen": "McDiv_nuggets/mcdiv_nuggets_no_hds_story_gen.csv",
        "resp_gen": "McDiv_nuggets/mcdiv_nuggets_no_hds_resp_gen.csv",
        "prompt_gen": "McDiv_nuggets/mcdiv_nuggets_no_hds_prompt_gen.csv",
    },
    "mcdiv_nuggets_with_hds": {
        "story_gen": "McDiv_nuggets/mcdiv_nuggets_200_with_hds_story_gen.csv",
        "resp_gen": "McDiv_nuggets/mcdiv_nuggets_200_with_hds_resp_gen.csv",
        "prompt_gen": "McDiv_nuggets/mcdiv_nuggets_200_with_hds_prompt_gen.csv",
    },
}


@dataclass
class DatasetEstimate:
    name: str
    sub_exp: str
    n_samples: int
    n_skipped: int
    unconditional_tokens: int  # total across all samples
    progressive_tokens: int  # total for ONE permutation across all samples
    n_permutations: int
    api_calls: int = 0
    total_tokens: int = 0

    def compute(self) -> None:
        n = self.n_samples
        self.api_calls = n * (5 + self.n_permutations)  # 5 uncond + n_perm progressive
        self.total_tokens = (
            self.unconditional_tokens
            + self.progressive_tokens * self.n_permutations
        )


def get_response_columns(fieldnames: list[str]) -> list[str]:
    resp_cols = []
    for f in fieldnames:
        if f.startswith("resp_") and f.replace("resp_", "").isdigit():
            resp_cols.append(f)
    return sorted(resp_cols, key=lambda x: int(x.replace("resp_", "")))


def estimate_csv(
    csv_path: Path,
    tokenizer: AutoTokenizer,
    n_permutations: int,
    format_mode: str,
    max_tokens: int,
) -> tuple[int, int, int, int]:
    """Estimate tokens for one CSV file.

    Returns (n_samples, n_skipped, unconditional_tokens, progressive_tokens_per_perm).
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        resp_cols = get_response_columns(reader.fieldnames or [])
        if not resp_cols:
            raise ValueError(f"No resp_* columns in {csv_path}")

        n_samples = 0
        n_skipped = 0
        uncond_total = 0
        prog_total = 0

        for row in reader:
            context = row.get("context", row.get("prompt", ""))
            responses = [row[c] for c in resp_cols]

            # Count tokens for full progressive context (all responses concatenated)
            prefix, target = format_conditioning_context(
                prompt=context,
                previous_responses=responses[:-1],
                current_response=responses[-1],
                format_mode=format_mode,
            )
            prog_tokens = len(tokenizer.encode(prefix + target))

            if prog_tokens > max_tokens:
                n_skipped += 1
                continue

            n_samples += 1
            prog_total += prog_tokens

            # Count tokens for unconditional (each response scored independently)
            for resp in responses:
                prefix_u, target_u = format_conditioning_context(
                    prompt=context,
                    previous_responses=[],
                    current_response=resp,
                    format_mode=format_mode,
                )
                uncond_total += len(tokenizer.encode(prefix_u + target_u))

    return n_samples, n_skipped, uncond_total, prog_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Tinker API cost for ICL diversity on Tevet datasets"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Base",
        help="Model to estimate for (used for tokenizer and pricing lookup)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=50,
        help="Number of permutations (default: 50)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=f"Datasets to estimate. Choices: {list(EXPERIMENT_TEMPLATES.keys())}. "
        f"Default: all.",
    )
    parser.add_argument(
        "--format-mode",
        type=str,
        default="completion",
        choices=["instruct", "completion"],
        help="Response formatting mode (default: completion)",
    )
    parser.add_argument(
        "--price-per-million-tokens",
        type=float,
        default=None,
        help="Override price per million tokens (default: lookup from model)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max context length; samples exceeding this are skipped (default: 32768)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to diversity-eval/data/with_metrics/ (default: auto-detect)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for name, sub_exps in EXPERIMENT_TEMPLATES.items():
            print(f"  {name}: {', '.join(sub_exps.keys())}")
        return

    # Resolve pricing
    if args.price_per_million_tokens is not None:
        price_per_m = args.price_per_million_tokens
    elif args.base_model in TINKER_PRICES:
        price_per_m = TINKER_PRICES[args.base_model]
    else:
        print(
            f"No pricing for {args.base_model}. Use --price-per-million-tokens. "
            f"Known models: {list(TINKER_PRICES.keys())}"
        )
        sys.exit(1)

    datasets = args.datasets or list(EXPERIMENT_TEMPLATES.keys())
    for d in datasets:
        if d not in EXPERIMENT_TEMPLATES:
            print(f"Unknown dataset: {d}. Choices: {list(EXPERIMENT_TEMPLATES.keys())}")
            sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try worktree first, fall back to main repo
        data_dir = PROJECT_ROOT / "diversity-eval" / "data" / "with_metrics"
        if not data_dir.exists():
            main_repo = PROJECT_ROOT.parent.parent.parent
            data_dir = main_repo / "diversity-eval" / "data" / "with_metrics"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Use --data-dir to specify the path to diversity-eval/data/with_metrics/")
        sys.exit(1)

    # Estimate each dataset
    estimates: list[DatasetEstimate] = []
    grand_tokens = 0
    grand_calls = 0
    grand_samples = 0
    grand_skipped = 0

    for ds_name in datasets:
        sub_exps = EXPERIMENT_TEMPLATES[ds_name]
        for sub_exp, rel_path in sub_exps.items():
            csv_path = data_dir / rel_path
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found, skipping")
                continue

            n_samples, n_skipped, uncond_tok, prog_tok = estimate_csv(
                csv_path, tokenizer, args.n_permutations, args.format_mode, args.max_tokens
            )

            est = DatasetEstimate(
                name=ds_name,
                sub_exp=sub_exp,
                n_samples=n_samples,
                n_skipped=n_skipped,
                unconditional_tokens=uncond_tok,
                progressive_tokens=prog_tok,
                n_permutations=args.n_permutations,
            )
            est.compute()
            estimates.append(est)

            grand_tokens += est.total_tokens
            grand_calls += est.api_calls
            grand_samples += est.n_samples
            grand_skipped += est.n_skipped

    # Print results
    print(f"\n{'='*80}")
    print(f"Tinker API Cost Estimate")
    print(f"  Model: {args.base_model}")
    print(f"  Price: ${price_per_m:.2f} / million tokens (prefill)")
    print(f"  Permutations: {args.n_permutations}")
    print(f"  Format mode: {args.format_mode}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"{'='*80}\n")

    print(f"{'Dataset':<30s} {'Sub-exp':<12s} {'Samples':>8s} {'Skip':>5s} "
          f"{'API calls':>10s} {'Tokens (M)':>11s} {'Cost':>8s}")
    print("-" * 90)

    for est in estimates:
        cost = est.total_tokens / 1_000_000 * price_per_m
        print(
            f"{est.name:<30s} {est.sub_exp:<12s} {est.n_samples:>8d} "
            f"{est.n_skipped:>5d} {est.api_calls:>10,d} "
            f"{est.total_tokens / 1_000_000:>11.2f} ${cost:>7.4f}"
        )

    grand_cost = grand_tokens / 1_000_000 * price_per_m
    print("-" * 90)
    print(
        f"{'TOTAL':<30s} {'':12s} {grand_samples:>8d} "
        f"{grand_skipped:>5d} {grand_calls:>10,d} "
        f"{grand_tokens / 1_000_000:>11.2f} ${grand_cost:>7.4f}"
    )
    print(f"\nEstimated total cost: ${grand_cost:.4f}")


if __name__ == "__main__":
    main()
