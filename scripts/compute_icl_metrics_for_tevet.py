"""
Bridge script: compute ICL diversity metrics for Tevet's diversity-eval CSVs.

Reads CSV files from diversity-eval/data/with_metrics/, computes ICL diversity
metrics (E, E_rate, C, D, D_rate) for each row's response set, and writes
enriched copies to results/tevet/<run-tag>/ (keeping the submodule clean).

Each run is identified by a run tag (auto-derived from model name or specified
via --run-tag). Column names are suffixed with the tag (e.g. metric_icl_E_gpt2)
so multiple model runs coexist in the same CSV. Sidecar JSONs carry the tag
in their filename and include run_config metadata.

Usage:
    # GPT-2 (default tag: gpt2)
    uv run python scripts/compute_icl_metrics_for_tevet.py --device cuda:0 --batch-size 8

    # Qwen 2.5 (auto-tag: qwen25)
    uv run python scripts/compute_icl_metrics_for_tevet.py \
        --base-model Qwen/Qwen2.5-32B --device auto --torch-dtype float16 \
        --n-permutations 20

    # Custom tag
    uv run python scripts/compute_icl_metrics_for_tevet.py --run-tag gpt2_50perm

    # Migrate existing sidecar data (no GPU needed)
    uv run python scripts/compute_icl_metrics_for_tevet.py --migrate-from-sidecars
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Add project root to path so we can import icl_diversity
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_diversity.core import FormatMode  # noqa: E402
from icl_diversity.core import compute_icl_diversity_metrics  # noqa: E402
from icl_diversity.core import format_conditioning_context  # noqa: E402

logger = logging.getLogger(__name__)


def dedup_rows(rows: list[dict]) -> list[dict]:
    """Remove rows whose sample_id appears with conflicting label_value.

    Some McDiv_nuggets CSVs contain the same sample_id with both label=0
    and label=1. This removes ALL rows for such conflicting IDs (keeping
    neither), since the correct label is ambiguous.

    Returns the filtered list (may be shorter than input).
    """
    if not rows or "sample_id" not in rows[0] or "label_value" not in rows[0]:
        return rows

    # Collect labels per sample_id
    labels_per_id: dict[str, set[str]] = {}
    for row in rows:
        sid = row["sample_id"]
        labels_per_id.setdefault(sid, set()).add(row["label_value"])

    conflicting = {sid for sid, labels in labels_per_id.items() if len(labels) > 1}
    if not conflicting:
        return rows

    logger.info(f"  Dedup: removing {len(conflicting)} sample_ids with conflicting labels")
    return [row for row in rows if row["sample_id"] not in conflicting]


# Base ICL metric names (before tag suffix)
ICL_METRIC_BASES = ["metric_icl_E", "metric_icl_E_rate", "metric_icl_C", "metric_icl_D", "metric_icl_D_rate"]

# Mapping from base column name to key in compute_icl_diversity_metrics() output
METRIC_KEY_MAP = {
    "metric_icl_E": "excess_entropy_E",
    "metric_icl_E_rate": "excess_entropy_E_rate",
    "metric_icl_C": "coherence_C",
    "metric_icl_D": "diversity_score_D",
    "metric_icl_D_rate": "diversity_score_D_rate",
}

# Tevet experiment JSON templates: maps experiment name → (class_name, {sub_exp: relative_csv_path})
EXPERIMENT_TEMPLATES: dict[str, tuple[str, dict[str, str]]] = {
    "con_test_200": ("ConTest", {
        "story_gen": "conTest/con_test_200_with_hds_story_gen.csv",
        "resp_gen": "conTest/con_test_200_with_hds_resp_gen.csv",
        "prompt_gen": "conTest/con_test_200_with_hds_prompt_gen.csv",
    }),
    "dec_test_200": ("DecTest", {
        "story_gen": "decTest/dec_test_200_with_hds_story_gen.csv",
        "resp_gen": "decTest/dec_test_200_with_hds_resp_gen.csv",
        "prompt_gen": "decTest/dec_test_200_with_hds_prompt_gen.csv",
    }),
    "dec_test_1000": ("DecTest", {
        "story_gen": "decTest/dec_test_1000_no_hds_story_gen.csv",
        "resp_gen": "decTest/dec_test_1000_no_hds_resp_gen.csv",
        "prompt_gen": "decTest/dec_test_1000_no_hds_prompt_gen.csv",
    }),
    "mcdiv_all": ("ConTest", {
        "story_gen": "McDiv/mcdiv_all_no_hds_story_gen.csv",
        "resp_gen": "McDiv/mcdiv_all_no_hds_resp_gen.csv",
        "prompt_gen": "McDiv/mcdiv_all_no_hds_prompt_gen.csv",
    }),
    "mcdiv_nuggets": ("ConTest", {
        "story_gen": "McDiv_nuggets/mcdiv_nuggets_no_hds_story_gen.csv",
        "resp_gen": "McDiv_nuggets/mcdiv_nuggets_no_hds_resp_gen.csv",
        "prompt_gen": "McDiv_nuggets/mcdiv_nuggets_no_hds_prompt_gen.csv",
    }),
    "mcdiv_nuggets_with_hds": ("ConTest", {
        "story_gen": "McDiv_nuggets/mcdiv_nuggets_200_with_hds_story_gen.csv",
        "resp_gen": "McDiv_nuggets/mcdiv_nuggets_200_with_hds_resp_gen.csv",
        "prompt_gen": "McDiv_nuggets/mcdiv_nuggets_200_with_hds_prompt_gen.csv",
    }),
}


def derive_run_tag(model_name: str) -> str:
    """Derive a short run tag from a model name.

    Examples:
        gpt2 → gpt2
        gpt2-medium → gpt2_medium
        Qwen/Qwen2.5-32B → qwen25
        meta-llama/Llama-3.1-8B → llama31
    """
    # Take the last component (after /)
    short = model_name.split("/")[-1].lower()
    # Strip common prefixes/suffixes
    short = re.sub(r"[-_](base|instruct|chat|hf)$", "", short)
    # Extract model family + version
    m = re.match(r"(gpt2|llama|qwen|mistral|phi|gemma)[-_.]?(\d+)?\.?(\d+)?", short)
    if m:
        family = m.group(1)
        major = m.group(2) or ""
        minor = m.group(3) or ""
        tag = family + major + minor
    else:
        # Fallback: alphanumeric only
        tag = re.sub(r"[^a-z0-9]", "_", short).strip("_")
    return tag


def tagged_columns(tag: str) -> list[str]:
    """Return ICL metric column names with run tag suffix."""
    return [f"{base}_{tag}" for base in ICL_METRIC_BASES]


def tagged_std_columns(tag: str) -> list[str]:
    """Return ICL metric _std column names with run tag suffix."""
    return [f"{base}_{tag}_std" for base in ICL_METRIC_BASES]


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
    format_mode: FormatMode = "instruct",
) -> int:
    """Count tokens for the full concatenated context using the real formatting."""
    prefix, target = format_conditioning_context(
        prompt=context,
        previous_responses=responses[:-1],
        current_response=responses[-1],
        format_mode=format_mode,
    )
    full_text = prefix + target
    return len(tokenizer.encode(full_text))


def compute_per_permutation_metrics(
    per_perm_curves: list[list[float]],
    per_perm_byte_counts: list[list[int]] | None,
    unconditional_per_byte: list[float],
) -> dict[str, float]:
    """Compute std of each metric across permutation curves.

    Returns dict with keys like 'excess_entropy_E_std', etc.
    """
    if not per_perm_curves or len(per_perm_curves) < 2:
        return {f"{key}_std": 0.0 for key in METRIC_KEY_MAP.values()}

    n_perm = len(per_perm_curves)

    # Mean unconditional surprise (constant across permutations)
    mean_h = sum(unconditional_per_byte) / len(unconditional_per_byte)
    coherence_C = 2.0 ** (-mean_h)

    e_vals: list[float] = []
    e_rate_vals: list[float] = []
    d_vals: list[float] = []
    d_rate_vals: list[float] = []
    c_vals: list[float] = []  # C is constant, but include for completeness

    for i in range(n_perm):
        curve_bits = per_perm_curves[i]
        a_n = curve_bits[-1]
        # E in total bits
        e = sum(a_k - a_n for a_k in curve_bits)

        # E_rate: need per-byte curve
        if per_perm_byte_counts and per_perm_byte_counts[i]:
            byte_counts = per_perm_byte_counts[i]
            curve_per_byte = [
                t / b if b > 0 else 0.0 for t, b in zip(curve_bits, byte_counts)
            ]
            a_n_rate = curve_per_byte[-1]
            e_rate = sum(a_k - a_n_rate for a_k in curve_per_byte)
        else:
            e_rate = 0.0

        e_vals.append(e)
        e_rate_vals.append(e_rate)
        c_vals.append(coherence_C)
        d_vals.append(coherence_C * e)
        d_rate_vals.append(coherence_C * e_rate)

    return {
        "excess_entropy_E_std": float(np.std(e_vals, ddof=1)) if n_perm > 1 else 0.0,
        "excess_entropy_E_rate_std": float(np.std(e_rate_vals, ddof=1)) if n_perm > 1 else 0.0,
        "coherence_C_std": 0.0,  # C doesn't vary with permutations
        "diversity_score_D_std": float(np.std(d_vals, ddof=1)) if n_perm > 1 else 0.0,
        "diversity_score_D_rate_std": float(np.std(d_rate_vals, ddof=1)) if n_perm > 1 else 0.0,
    }


def sidecar_path_for_tag(csv_path: Path, tag: str) -> Path:
    """Return sidecar JSON path with run tag: foo.icl_curves.<tag>.json"""
    return csv_path.with_suffix(f".icl_curves.{tag}.json")


def mean_curves_path_for_tag(csv_path: Path, tag: str) -> Path:
    """Return lightweight mean curves path: foo.icl_mean_curves.<tag>.json

    This file is small (~200-400 bytes per sample) and intended to be
    committed to git, unlike the full sidecar which is gitignored.
    """
    return csv_path.with_suffix(f".icl_mean_curves.{tag}.json")


def output_csv_path(source_csv: Path, output_dir: Path, source_data_dir: Path) -> Path:
    """Map a source CSV path to its output location under output_dir.

    Preserves the subdirectory structure (e.g. McDiv/foo.csv → output_dir/McDiv/foo.csv).
    """
    rel = source_csv.relative_to(source_data_dir)
    return output_dir / rel


def process_csv(
    csv_path: Path,
    output_csv: Path,
    model: PreTrainedModel | None,
    tokenizer: AutoTokenizer | None,
    tag: str,
    run_config: dict,
    n_permutations: int,
    batch_size: int,
    max_tokens: int,
    force: bool = False,
    max_skip_fraction: float = 0.1,
    format_mode: FormatMode = "instruct",
    dedup: bool = False,
) -> ProcessingStats:
    """Process a single CSV file: compute ICL metrics and write to output location.

    If model is None, uses sidecar cache only (for migration).
    """
    logger.info(f"Processing {csv_path}")
    stats = ProcessingStats()

    # Sidecar paths: tag-specific sidecar at output location
    output_sidecar = sidecar_path_for_tag(output_csv, tag)
    sidecar_data = load_sidecar(output_sidecar)

    # Also check legacy sidecar (untagged, in source location) for migration
    legacy_sidecar_path = csv_path.with_suffix(".icl_curves.json")
    legacy_sidecar = load_sidecar(legacy_sidecar_path) if legacy_sidecar_path.exists() else {}

    # Read source CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if dedup:
        rows = dedup_rows(rows)

    resp_cols = get_response_columns(fieldnames)
    n_responses = len(resp_cols)
    logger.info(f"  {len(rows)} rows, {n_responses} responses per row")

    # Build tagged column names
    metric_cols = tagged_columns(tag)
    std_cols = tagged_std_columns(tag)
    all_new_cols = metric_cols + std_cols

    # Check if tagged metrics already computed in output
    if not force and output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            existing_reader = csv.DictReader(f)
            existing_fields = list(existing_reader.fieldnames)
            existing_rows = list(existing_reader)
        if all(col in existing_fields for col in metric_cols):
            if existing_rows and all(existing_rows[0].get(col, "") != "" for col in metric_cols):
                logger.info(
                    "  Tagged metrics already present in output, skipping (use --force to recompute)"
                )
                return stats

    # Strip any old untagged ICL columns from fieldnames (clean migration)
    old_icl_cols = [f for f in fieldnames if f.startswith("metric_icl_") and f not in all_new_cols]
    clean_fieldnames = [f for f in fieldnames if f not in old_icl_cols]

    # Build output fieldnames: existing non-ICL + new tagged columns
    new_fieldnames = list(clean_fieldnames)
    for col in all_new_cols:
        if col not in new_fieldnames:
            new_fieldnames.append(col)

    # Pre-scan token lengths if we have a tokenizer
    row_token_counts: list[tuple[str, int]] = []
    if tokenizer is not None:
        logger.info("  Pre-scanning token lengths...")
        for row in rows:
            context = row.get("context", "")
            responses = [row[col] for col in resp_cols]
            token_count = count_tokens_for_full_context(tokenizer, context, responses, format_mode)
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
    else:
        # Migration mode: no token counting
        row_token_counts = [(row["sample_id"], 0) for row in rows]

    # --- Main processing loop ---
    for row, (sample_id, token_count) in tqdm(
        zip(rows, row_token_counts),
        desc=csv_path.name,
        unit="set",
        total=len(rows),
    ):
        stats.token_counts.append(token_count)

        # Check tagged sidecar cache first
        cached_entry = sidecar_data.get(sample_id)
        if cached_entry is None:
            # Fall back to legacy sidecar
            cached_entry = legacy_sidecar.get(sample_id)

        if not force and cached_entry is not None and not cached_entry.get("skipped", False):
            cached_metrics = cached_entry.get("metrics", {})
            for base, key in METRIC_KEY_MAP.items():
                col = f"{base}_{tag}"
                row[col] = f"{cached_metrics.get(key, 0.0):.6f}"

            # Compute std from cached per-permutation curves
            std_metrics = compute_per_permutation_metrics(
                cached_entry.get("per_permutation_a_k_curves", []),
                cached_entry.get("per_permutation_byte_counts"),
                cached_entry.get("unconditional_surprises", []),
            )
            for base, key in METRIC_KEY_MAP.items():
                std_col = f"{base}_{tag}_std"
                row[std_col] = f"{std_metrics.get(key + '_std', 0.0):.6f}"

            # Copy to tagged sidecar if from legacy
            if sample_id not in sidecar_data:
                sidecar_data[sample_id] = cached_entry
            stats.cached += 1
            continue

        # Skip if entry was previously skipped
        if cached_entry is not None and cached_entry.get("skipped", False):
            for col in all_new_cols:
                row[col] = ""
            if cached_entry.get("reason") == "token_limit":
                stats.skipped_token_limit += 1
            else:
                stats.skipped_error += 1
            if sample_id not in sidecar_data:
                sidecar_data[sample_id] = cached_entry
            stats.cached += 1
            continue

        # Need model for fresh computation
        if model is None or tokenizer is None:
            logger.warning(f"  No model loaded, skipping {sample_id} (no cache)")
            for col in all_new_cols:
                row[col] = ""
            stats.skipped_error += 1
            continue

        # Extract context and responses
        context = row.get("context", "")
        responses = [row[col] for col in resp_cols]

        # Check token length
        if token_count > max_tokens:
            for col in all_new_cols:
                row[col] = ""
            stats.skipped_token_limit += 1
            stats.token_limit_sample_ids.append(sample_id)
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
                format_mode=format_mode,
            )
        except Exception:
            logger.error(
                f"  Error computing metrics for {sample_id}:\n{traceback.format_exc()}"
            )
            for col in all_new_cols:
                row[col] = ""
            stats.skipped_error += 1
            stats.error_sample_ids.append(sample_id)
            sidecar_data[sample_id] = {
                "skipped": True,
                "reason": "error",
                "error": traceback.format_exc(),
            }
            continue

        # Write metric values to row (tagged columns)
        for base, key in METRIC_KEY_MAP.items():
            col = f"{base}_{tag}"
            row[col] = f"{metrics[key]:.6f}"

        # Compute and write std columns
        std_metrics = compute_per_permutation_metrics(
            metrics.get("per_permutation_a_k_curves", []),
            metrics.get("per_permutation_byte_counts"),
            metrics.get("unconditional_surprises", []),
        )
        for base, key in METRIC_KEY_MAP.items():
            std_col = f"{base}_{tag}_std"
            row[std_col] = f"{std_metrics.get(key + '_std', 0.0):.6f}"

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
            "std_metrics": std_metrics,
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
        output_sidecar.parent.mkdir(parents=True, exist_ok=True)
        save_sidecar(output_sidecar, sidecar_data)
        return stats

    # Write output CSV (clean copy with tagged columns, old untagged ICL columns removed)
    # Remove old untagged ICL values from rows
    for row in rows:
        for old_col in old_icl_cols:
            row.pop(old_col, None)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Save tagged sidecar with run_config
    sidecar_data["__run_config__"] = run_config
    save_sidecar(output_sidecar, sidecar_data)
    logger.info(f"  Wrote {output_csv} and {output_sidecar.name}")

    # Save lightweight mean curves (committed to git, not gitignored)
    output_mean_curves = mean_curves_path_for_tag(output_csv, tag)
    mean_curves_data: dict[str, dict] = {}
    for sample_id, entry in sidecar_data.items():
        if sample_id.startswith("__") or not isinstance(entry, dict):
            continue
        if entry.get("skipped", False):
            continue
        mean_curves_data[sample_id] = {
            "a_k_curve": entry.get("a_k_curve"),
            "a_k_byte_counts": entry.get("a_k_byte_counts"),
            "unconditional_surprises": entry.get("unconditional_surprises"),
            "unconditional_total_bits": entry.get("unconditional_total_bits"),
        }
    save_sidecar(output_mean_curves, mean_curves_data)
    logger.info(f"  Wrote mean curves: {output_mean_curves.name}")

    return stats


def generate_experiment_jsons(output_dir: Path, tag: str) -> list[Path]:
    """Generate Tevet-compatible experiment JSON files pointing to our output CSVs.

    Paths are relative to the diversity-eval/ directory (where run_experiments.py runs).
    """
    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Compute relative path from diversity-eval/ to output_dir
    diversity_eval_dir = PROJECT_ROOT / "diversity-eval"
    try:
        rel_to_de = output_dir.relative_to(diversity_eval_dir)
        prefix = str(rel_to_de)
    except ValueError:
        # output_dir is outside diversity-eval, use relative path
        prefix = str(Path("..") / output_dir.relative_to(PROJECT_ROOT))

    generated: list[Path] = []
    for exp_name, (class_name, sub_exps) in EXPERIMENT_TEMPLATES.items():
        exp_json = {
            "global_config": {"class_name": class_name},
            "experiments": {
                sub_name: f"{prefix}/{csv_rel}"
                for sub_name, csv_rel in sub_exps.items()
            },
        }
        out_path = experiments_dir / f"{exp_name}.json"
        with open(out_path, "w") as f:
            json.dump(exp_json, f, indent=2)
        generated.append(out_path)
        logger.info(f"  Generated {out_path}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ICL diversity metrics for Tevet's diversity-eval CSVs"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="*",
        help="Specific source CSV file(s) to process. Default: all with_metrics CSVs.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID for the base model (default: gpt2)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag for column names and filenames (default: auto from model name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/tevet/<run-tag>)",
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
    parser.add_argument(
        "--format-mode",
        type=str,
        default="instruct",
        choices=["instruct", "completion"],
        help="Response formatting mode: 'instruct' (default, Response A: ...) or "
             "'completion' (1. {prompt}{completion}..., for story completions)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Remove samples with conflicting labels (same sample_id, different label_value). "
             "Appends '-dedup' to the run tag.",
    )
    parser.add_argument(
        "--migrate-from-sidecars",
        action="store_true",
        help="Migrate existing untagged sidecar data to new tagged format (no GPU needed)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Derive run tag
    tag = args.run_tag or derive_run_tag(args.base_model)
    if args.dedup and not tag.endswith("-dedup"):
        tag += "-dedup"
    logger.info(f"Run tag: {tag}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "tevet" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Source data directory
    source_data_dir = PROJECT_ROOT / "diversity-eval" / "data" / "with_metrics"

    # Run config metadata
    run_config = {
        "base_model": args.base_model,
        "run_tag": tag,
        "n_permutations": args.n_permutations,
        "torch_dtype": args.torch_dtype,
        "batch_size": args.batch_size,
        "format_mode": args.format_mode,
        "dedup": args.dedup,
    }

    model = None
    tokenizer = None
    max_tokens = 1024

    if not args.migrate_from_sidecars:
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

        max_tokens = getattr(model.config, "max_position_embeddings", 1024)
        logger.info(f"Max context length: {max_tokens} tokens")
        run_config["max_tokens"] = max_tokens
    else:
        logger.info("Migration mode: using cached sidecar data only")

    # Find source CSVs to process
    if args.input:
        csv_paths = [Path(p).resolve() for p in args.input]
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
        out_csv = output_csv_path(csv_path, output_dir, source_data_dir)
        stats = process_csv(
            csv_path=csv_path,
            output_csv=out_csv,
            model=model,
            tokenizer=tokenizer,
            tag=tag,
            run_config=run_config,
            n_permutations=args.n_permutations,
            batch_size=args.batch_size,
            max_tokens=max_tokens,
            force=args.force,
            max_skip_fraction=args.max_skip_fraction,
            format_mode=args.format_mode,
            dedup=args.dedup,
        )
        all_stats[csv_path.name] = stats

    # --- Generate experiment JSONs ---
    logger.info("Generating experiment JSONs...")
    generate_experiment_jsons(output_dir, tag)

    # --- Save run config ---
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    logger.info(f"Saved {config_path}")

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
        for name, stats in all_stats.items():
            if stats.total_skipped > 0:
                logger.warning(
                    f"    {name}: {stats.skipped_token_limit} token limit, "
                    f"{stats.skipped_error} errors"
                )

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Experiment JSONs: {output_dir / 'experiments'}")
    logger.info(
        f"To run Tevet's pipeline:\n"
        f"  cd diversity-eval && python run_experiments.py "
        f"--input_json {output_dir / 'experiments'}"
    )


if __name__ == "__main__":
    main()
