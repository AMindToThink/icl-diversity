"""
Bridge script: compute ICL diversity metrics for Tevet's diversity-eval CSVs.

Reads CSV files from diversity-eval/data/with_metrics/, computes ICL diversity
metrics (E, E_rate, C, D, D_rate) for each row's response set, and writes
the metrics back as new columns. Also saves per-row a_k curves and
per-permutation data to a sidecar JSON for later plotting.

Usage:
    # Process all with_metrics CSVs
    uv run python scripts/compute_icl_metrics_for_tevet.py

    # Process specific CSV(s)
    uv run python scripts/compute_icl_metrics_for_tevet.py \
        --input diversity-eval/data/with_metrics/conTest/con_test_200_with_hds_story_gen.csv

    # Use a different base model
    uv run python scripts/compute_icl_metrics_for_tevet.py --base-model gpt2-medium --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Add project root to path so we can import icl_diversity
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_diversity.core import compute_icl_diversity_metrics  # noqa: E402
from icl_diversity.core import format_conditioning_context  # noqa: E402

logger = logging.getLogger(__name__)

# ICL metric columns to add to CSVs
ICL_METRIC_COLUMNS = [
    "metric_icl_E",
    "metric_icl_E_rate",
    "metric_icl_C",
    "metric_icl_D",
    "metric_icl_D_rate",
]

# Mapping from column name to key in compute_icl_diversity_metrics() output
METRIC_KEY_MAP = {
    "metric_icl_E": "excess_entropy_E",
    "metric_icl_E_rate": "excess_entropy_E_rate",
    "metric_icl_C": "coherence_C",
    "metric_icl_D": "diversity_score_D",
    "metric_icl_D_rate": "diversity_score_D_rate",
}


@dataclass
class ProcessingStats:
    """Track what happened during processing for end-of-file reporting."""

    computed: int = 0
    cached: int = 0
    skipped_token_limit: int = 0
    skipped_error: int = 0
    token_counts: list[int] = field(default_factory=list)
    error_sample_ids: list[str] = field(default_factory=list)
    token_limit_sample_ids: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.computed + self.cached + self.skipped_token_limit + self.skipped_error
        )

    @property
    def total_skipped(self) -> int:
        return self.skipped_token_limit + self.skipped_error

    @property
    def skip_fraction(self) -> float:
        return self.total_skipped / self.total if self.total > 0 else 0.0


def find_all_with_metrics_csvs() -> list[Path]:
    """Find all CSV files under diversity-eval/data/with_metrics/."""
    data_dir = PROJECT_ROOT / "diversity-eval" / "data" / "with_metrics"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} not found. Run data download first:\n"
            "  cd diversity-eval && python -c 'import utils; utils.download_and_place_data()'"
        )
    return sorted(data_dir.rglob("*.csv"))


def get_response_columns(fieldnames: list[str]) -> list[str]:
    """Extract resp_0, resp_1, ... columns in order."""
    resp_cols = []
    for f in fieldnames:
        if f.startswith("resp_") and f.replace("resp_", "").isdigit():
            resp_cols.append(f)
    return sorted(resp_cols, key=lambda x: int(x.replace("resp_", "")))


def load_sidecar(sidecar_path: Path) -> dict:
    """Load existing sidecar JSON cache, or return empty dict."""
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            return json.load(f)
    return {}


def save_sidecar(sidecar_path: Path, data: dict) -> None:
    """Save sidecar JSON data."""
    with open(sidecar_path, "w") as f:
        json.dump(data, f, indent=2)


def count_tokens_for_full_context(
    tokenizer: AutoTokenizer,
    context: str,
    responses: list[str],
) -> int:
    """Count tokens for the full concatenated context using the real formatting.

    Uses format_conditioning_context (the same function the metric computation
    uses) to build the exact text that would be fed to the model, then counts
    tokens. This ensures the check matches reality.
    """
    # Build the full context as format_conditioning_context would for the last response
    prefix, target = format_conditioning_context(
        prompt=context,
        previous_responses=responses[:-1],
        current_response=responses[-1],
    )
    full_text = prefix + target
    return len(tokenizer.encode(full_text))


def process_csv(
    csv_path: Path,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    n_permutations: int,
    batch_size: int,
    max_tokens: int,
    force: bool = False,
    max_skip_fraction: float = 0.1,
) -> ProcessingStats:
    """Process a single CSV file: compute ICL metrics and add columns.

    Returns ProcessingStats so the caller can inspect what happened.
    """
    logger.info(f"Processing {csv_path}")
    stats = ProcessingStats()

    # Sidecar JSON for a_k curves and per-permutation data
    sidecar_path = csv_path.with_suffix(".icl_curves.json")
    sidecar_data = load_sidecar(sidecar_path)

    # Read existing CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    resp_cols = get_response_columns(fieldnames)
    n_responses = len(resp_cols)
    logger.info(f"  {len(rows)} rows, {n_responses} responses per row")

    # Check if ICL metrics already computed (all columns present and non-empty)
    if not force and all(col in fieldnames for col in ICL_METRIC_COLUMNS):
        # Check if first row has values
        if rows and all(rows[0].get(col, "") != "" for col in ICL_METRIC_COLUMNS):
            logger.info(
                "  ICL metrics already present, skipping (use --force to recompute)"
            )
            return stats

    # Add new columns to fieldnames if not present
    new_fieldnames = list(fieldnames)
    for col in ICL_METRIC_COLUMNS:
        if col not in new_fieldnames:
            new_fieldnames.append(col)

    # --- Pre-scan: count tokens for all rows to report distribution ---
    logger.info("  Pre-scanning token lengths...")
    row_token_counts: list[tuple[str, int]] = []
    for row in rows:
        context = row.get("context", "")
        responses = [row[col] for col in resp_cols]
        token_count = count_tokens_for_full_context(tokenizer, context, responses)
        row_token_counts.append((row["sample_id"], token_count))

    all_counts = [tc for _, tc in row_token_counts]
    n_over = sum(1 for tc in all_counts if tc > max_tokens)
    logger.info(
        f"  Token lengths: min={min(all_counts)}, "
        f"median={sorted(all_counts)[len(all_counts) // 2]}, "
        f"max={max(all_counts)}, "
        f"over {max_tokens} limit: {n_over}/{len(all_counts)} "
        f"({100 * n_over / len(all_counts):.1f}%)"
    )
    if n_over > 0:
        # Show the worst offenders
        over_limit = [(sid, tc) for sid, tc in row_token_counts if tc > max_tokens]
        over_limit.sort(key=lambda x: -x[1])
        for sid, tc in over_limit[:5]:
            logger.warning(f"    Over limit: {sid} = {tc} tokens")
        if len(over_limit) > 5:
            logger.warning(f"    ... and {len(over_limit) - 5} more")

    # --- Main processing loop ---
    for row, (sample_id, token_count) in tqdm(
        zip(rows, row_token_counts),
        desc=csv_path.name,
        unit="set",
        total=len(rows),
    ):
        stats.token_counts.append(token_count)

        # Check sidecar cache
        if not force and sample_id in sidecar_data:
            cached_metrics = sidecar_data[sample_id].get("metrics", {})
            for col, key in METRIC_KEY_MAP.items():
                row[col] = f"{cached_metrics.get(key, 0.0):.6f}"
            stats.cached += 1
            continue

        # Extract context and responses
        context = row.get("context", "")
        responses = [row[col] for col in resp_cols]

        # Check token length
        if token_count > max_tokens:
            for col in ICL_METRIC_COLUMNS:
                row[col] = ""
            stats.skipped_token_limit += 1
            stats.token_limit_sample_ids.append(sample_id)
            # Record skip in sidecar so we know why it's missing
            sidecar_data[sample_id] = {
                "skipped": True,
                "reason": "token_limit",
                "token_count": token_count,
                "max_tokens": max_tokens,
            }
            continue

        # Compute ICL diversity metrics
        try:
            metrics = compute_icl_diversity_metrics(
                model=model,
                tokenizer=tokenizer,
                prompt=context,
                responses=responses,
                n_permutations=n_permutations,
                batch_size=batch_size,
            )
        except Exception:
            logger.error(
                f"  Error computing metrics for {sample_id}:\n{traceback.format_exc()}"
            )
            for col in ICL_METRIC_COLUMNS:
                row[col] = ""
            stats.skipped_error += 1
            stats.error_sample_ids.append(sample_id)
            sidecar_data[sample_id] = {
                "skipped": True,
                "reason": "error",
                "error": traceback.format_exc(),
            }
            continue

        # Write metric values to row
        for col, key in METRIC_KEY_MAP.items():
            row[col] = f"{metrics[key]:.6f}"

        # Save to sidecar cache
        sidecar_data[sample_id] = {
            "metrics": {
                key: metrics[key]
                for key in [
                    "excess_entropy_E",
                    "excess_entropy_E_rate",
                    "coherence_C",
                    "diversity_score_D",
                    "diversity_score_D_rate",
                    "coherence_spread_sigma",
                    "mean_byte_length",
                    "is_monotone",
                ]
            },
            "token_count": token_count,
            "a_k_curve": metrics["a_k_curve"],
            "a_k_curve_per_byte": metrics["a_k_curve_per_byte"],
            "a_k_byte_counts": metrics["a_k_byte_counts"],
            "unconditional_surprises": metrics["unconditional_surprises"],
            "unconditional_total_bits": metrics["unconditional_total_bits"],
            "per_permutation_a_k_curves": metrics.get("per_permutation_a_k_curves"),
            "per_permutation_byte_counts": metrics.get("per_permutation_byte_counts"),
        }
        stats.computed += 1

    # --- End-of-file summary ---
    logger.info(
        f"  Summary: {stats.computed} computed, {stats.cached} cached, "
        f"{stats.skipped_token_limit} skipped (token limit), "
        f"{stats.skipped_error} skipped (error)"
    )
    if stats.total_skipped > 0:
        logger.warning(
            f"  Skip rate: {stats.total_skipped}/{stats.total} "
            f"({100 * stats.skip_fraction:.1f}%)"
        )

    if stats.skip_fraction > max_skip_fraction:
        logger.error(
            f"  ABORTING write for {csv_path.name}: skip rate "
            f"{100 * stats.skip_fraction:.1f}% exceeds threshold "
            f"{100 * max_skip_fraction:.0f}%. "
            f"Sidecar saved (for debugging) but CSV NOT modified. "
            f"Use --max-skip-fraction to raise the threshold if this is expected."
        )
        # Still save sidecar for debugging, but don't corrupt the CSV
        save_sidecar(sidecar_path, sidecar_data)
        return stats

    # Write updated CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save sidecar
    save_sidecar(sidecar_path, sidecar_data)
    logger.info(f"  Wrote {csv_path.name} and {sidecar_path.name}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ICL diversity metrics for Tevet's diversity-eval CSVs"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="*",
        help="Specific CSV file(s) to process. Default: all with_metrics CSVs.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID for the base model (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'cuda:0', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: float32)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=50,
        help="Number of permutations for averaging (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for forward passes (default: 8)",
    )
    parser.add_argument(
        "--max-skip-fraction",
        type=float,
        default=0.1,
        help="Abort CSV write if skip rate exceeds this fraction (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even if already present",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    logger.info(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device == "auto" else None,
    )
    if args.device != "auto":
        model = model.to(device)
    model.eval()

    # Get max token length from model config
    max_tokens = getattr(model.config, "max_position_embeddings", 1024)
    logger.info(f"Max context length: {max_tokens} tokens")

    # Find CSVs to process
    if args.input:
        csv_paths = [Path(p) for p in args.input]
        for p in csv_paths:
            if not p.exists():
                logger.error(f"File not found: {p}")
                sys.exit(1)
    else:
        csv_paths = find_all_with_metrics_csvs()

    logger.info(f"Processing {len(csv_paths)} CSV files")

    # --- Process all CSVs and collect stats ---
    all_stats: dict[str, ProcessingStats] = {}
    for csv_path in csv_paths:
        stats = process_csv(
            csv_path=csv_path,
            model=model,
            tokenizer=tokenizer,
            n_permutations=args.n_permutations,
            batch_size=args.batch_size,
            max_tokens=max_tokens,
            force=args.force,
            max_skip_fraction=args.max_skip_fraction,
        )
        all_stats[csv_path.name] = stats

    # --- Final summary across all files ---
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    total_computed = sum(s.computed for s in all_stats.values())
    total_cached = sum(s.cached for s in all_stats.values())
    total_skipped = sum(s.total_skipped for s in all_stats.values())
    total_rows = sum(s.total for s in all_stats.values())
    logger.info(f"  Total: {total_rows} rows across {len(csv_paths)} files")
    logger.info(
        f"  Computed: {total_computed}, Cached: {total_cached}, "
        f"Skipped: {total_skipped}"
    )
    if total_skipped > 0:
        logger.warning(
            f"  Overall skip rate: {total_skipped}/{total_rows} "
            f"({100 * total_skipped / total_rows:.1f}%)"
        )
        # Per-file breakdown for files with skips
        for name, stats in all_stats.items():
            if stats.total_skipped > 0:
                logger.warning(
                    f"    {name}: {stats.skipped_token_limit} token limit, "
                    f"{stats.skipped_error} errors"
                )


if __name__ == "__main__":
    main()
