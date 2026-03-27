"""Compute Tevet's baseline metric scores (ρ, OCA) for comparison.

Reads the pre-computed baseline metrics from the diversity-eval CSVs
(metric_averaged_distinct_ngrams, metric_bert_score, etc.) and computes
Spearman ρ and OCA against labels, matching Tevet's evaluation protocol.

Data provenance:
- Input: diversity-eval/data/with_metrics/ (Tevet's original CSVs)
- Baseline metrics computed by Tevet's run_metrics.py
- Evaluation: Spearman ρ + OCA via Tevet's utils.optimal_classification_accuracy

Usage:
    uv run python scripts/compute_tevet_baselines.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "diversity-eval"))
from utils import optimal_classification_accuracy

DATA_DIR = PROJECT_ROOT / "diversity-eval" / "data" / "with_metrics"

BASELINE_METRICS = [
    "metric_averaged_distinct_ngrams",
    "metric_bert_score",
    "metric_bert_sts",
    "metric_sent_bert",
    "metric_averaged_cosine_similarity",
]

DISPLAY_NAMES = {
    "metric_averaged_distinct_ngrams": "distinct-n",
    "metric_bert_score": "BERTScore",
    "metric_bert_sts": "BERTsts",
    "metric_sent_bert": "SentBERT",
    "metric_averaged_cosine_similarity": "cos-sim",
}

# Tevet's 6 experiments (from data/experiments/*.json)
EXPERIMENTS = [
    ("ConTest (200, with_hds)", "ConTest", "conTest", "with_hds"),
    ("McDiv_nuggets (with_hds)", "ConTest", "McDiv_nuggets", "with_hds"),
    ("McDiv_nuggets (no_hds)", "ConTest", "McDiv_nuggets", "no_hds"),
    ("McDiv (no_hds)", "ConTest", "McDiv", "no_hds"),
    ("DecTest (1000, no_hds)", "DecTest", "decTest", "no_hds"),
    ("DecTest (200, with_hds)", "DecTest", "decTest", "with_hds"),
]

TASKS = ["prompt_gen", "resp_gen", "story_gen"]


def find_csv(dataset_dir: str, subset: str, task: str) -> Path | None:
    """Find the CSV matching the experiment config."""
    base = DATA_DIR / dataset_dir
    for p in base.glob("*.csv"):
        if task in p.name and subset in p.name:
            return p
    return None


def main() -> None:
    for exp_name, test_class, dataset_dir, subset in EXPERIMENTS:
        print(f"\n{'=' * 90}")
        print(f"  {exp_name}  (test: {test_class})")
        print(f"{'=' * 90}")

        is_binary = test_class == "ConTest"

        if is_binary:
            header = f"  {'Task':<15s} {'Metric':<20s} {'Spearman ρ':>12s} {'OCA':>8s}"
        else:
            header = f"  {'Task':<15s} {'Metric':<20s} {'Spearman ρ':>12s}"
        print(header)
        print(f"  {'-'*15} {'-'*20} {'-'*12}" + (f" {'-'*8}" if is_binary else ""))

        for task in TASKS:
            csv_path = find_csv(dataset_dir, subset, task)
            if csv_path is None:
                print(f"  {task:<15s} (CSV not found)")
                continue

            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            labels = [float(r["label_value"]) for r in rows]
            available = [m for m in BASELINE_METRICS if m in rows[0]]

            for i, metric_col in enumerate(available):
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r[metric_col]))
                    except (ValueError, KeyError):
                        vals.append(float("nan"))

                valid = [(v, l) for v, l in zip(vals, labels)
                         if not np.isnan(v) and not np.isnan(l)]
                if len(valid) < 5:
                    continue

                v_scores, v_labels = zip(*valid)
                rho, _ = spearmanr(v_scores, v_labels)
                display = DISPLAY_NAMES.get(metric_col, metric_col)
                task_col = task if i == 0 else ""

                if is_binary:
                    unique_labels = set(v_labels)
                    if unique_labels == {0.0, 1.0}:
                        high = [s for s, l in zip(v_scores, v_labels) if l == 1.0]
                        low = [s for s, l in zip(v_scores, v_labels) if l == 0.0]
                        oca, _ = optimal_classification_accuracy(high, low)
                        print(f"  {task_col:<15s} {display:<20s} {rho:>+12.3f} {oca:>8.3f}")
                    else:
                        print(f"  {task_col:<15s} {display:<20s} {rho:>+12.3f}      —")
                else:
                    print(f"  {task_col:<15s} {display:<20s} {rho:>+12.3f}")


if __name__ == "__main__":
    main()
