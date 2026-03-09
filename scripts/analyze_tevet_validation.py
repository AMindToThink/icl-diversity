"""
Analyze Tevet diversity-eval results for ICL diversity metric validation.

Parses the results JSON and CSVs from Tevet's experiment pipeline,
compares ICL metrics (E, D) against existing baselines, and tests
each hypothesis from hypotheses/tevet_validation.md.

Usage:
    # After running compute_icl_metrics_for_tevet.py and run_experiments.py:
    uv run python scripts/analyze_tevet_validation.py --run-tag gpt2

    # Direct CSV analysis (no run_experiments.py needed):
    uv run python scripts/analyze_tevet_validation.py --run-tag gpt2

    # Analyze all available run tags:
    uv run python scripts/analyze_tevet_validation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"
DIVERSITY_EVAL_DIR = PROJECT_ROOT / "diversity-eval"
TEVET_RESULTS_DIR = DIVERSITY_EVAL_DIR / "results"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "tevet_validation"

# Baseline results from Tevet paper (Table 2 & 4, approximate)
PAPER_BASELINES = {
    "decTest": {
        "story_gen": {
            "metric_averaged_distinct_ngrams": {"spearman": 0.91},
            "metric_averaged_cosine_similarity": {"spearman": 0.81},
            "metric_bert_score": {"spearman": 0.63},
            "metric_sent_bert": {"spearman": 0.74},
        },
        "resp_gen": {
            "metric_averaged_distinct_ngrams": {"spearman": 0.89},
            "metric_sent_bert": {"spearman": 0.81},
        },
        "prompt_gen": {
            "metric_averaged_distinct_ngrams": {"spearman": 0.76},
            "metric_sent_bert": {"spearman": 0.71},
        },
    },
    "conTest": {
        "story_gen": {
            "metric_averaged_distinct_ngrams": {"spearman": 0.57, "oca": 0.70},
            "metric_sent_bert": {"spearman": 0.67, "oca": 0.90},
        },
        "resp_gen": {
            "metric_averaged_distinct_ngrams": {"spearman": 0.47, "oca": 0.69},
            "metric_sent_bert": {"spearman": 0.52, "oca": 0.79},
        },
        "prompt_gen": {
            "metric_averaged_distinct_ngrams": {"spearman": 0.33, "oca": 0.62},
            "metric_sent_bert": {"spearman": 0.44, "oca": 0.73},
        },
    },
}


def find_run_tags() -> list[str]:
    """Find available run tags under results/tevet/."""
    if not RESULTS_BASE.exists():
        return []
    return sorted(
        d.name for d in RESULTS_BASE.iterdir()
        if d.is_dir() and (d / "run_config.json").exists()
    )


def icl_metric_names(tag: str) -> list[str]:
    """Return tagged ICL metric column names."""
    bases = ["metric_icl_E", "metric_icl_E_rate", "metric_icl_C", "metric_icl_D", "metric_icl_D_rate"]
    return [f"{b}_{tag}" for b in bases]


def load_experiment_results(results_dir: Path) -> dict:
    """Load results.json from each experiment directory."""
    all_results = {}
    for results_json in results_dir.rglob("results.json"):
        exp_name = results_json.parent.name
        with open(results_json) as f:
            all_results[exp_name] = json.load(f)
    return all_results


def load_csv_metrics(csv_path: Path) -> dict[str, list[float]]:
    """Load metric columns and labels from a CSV."""
    data: dict[str, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in reader.fieldnames:
                if field.startswith("metric_") or field.startswith("label_"):
                    if field not in data:
                        data[field] = []
                    try:
                        val = float(row[field]) if row[field] != "" else np.nan
                    except (ValueError, TypeError):
                        val = row[field]
                    data[field].append(val)
    return data


def print_results_table(all_results: dict) -> None:
    """Print a summary table of all experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    for exp_name, results in sorted(all_results.items()):
        print(f"\n{'─' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"{'─' * 60}")

        for sub_exp, metrics in sorted(results.items()):
            print(f"\n  {sub_exp}:")
            score_types = set()
            for m_results in metrics.values():
                score_types.update(m_results.keys())
            score_types = sorted(score_types)

            header = f"    {'Metric':<45s}"
            for st in score_types:
                header += f" {st:>12s}"
            print(header)
            print(f"    {'─' * (45 + 13 * len(score_types))}")

            for metric_name, m_results in sorted(metrics.items()):
                row = f"    {metric_name:<45s}"
                for st in score_types:
                    val = m_results.get(st, float("nan"))
                    row += f" {val:>12.4f}"
                print(row)


def test_hypotheses(all_results: dict, tag: str) -> None:
    """Test each hypothesis against the experiment results."""
    print("\n" + "=" * 80)
    print(f"HYPOTHESIS TESTING [{tag}]")
    print("=" * 80)

    e_col = f"metric_icl_E_{tag}"
    d_col = f"metric_icl_D_{tag}"

    # Collect ConTest results
    contest_results = {}
    for exp_name, results in all_results.items():
        if "con_test" in exp_name.lower():
            contest_results[exp_name] = results

    if contest_results:
        print(
            "\n--- H1: E detects content diversity better than n-gram metrics (ConTest) ---"
        )
        for exp_name, results in contest_results.items():
            for sub_exp, metrics in results.items():
                e_spearman = metrics.get(e_col, {}).get("spearman_cor", None)
                e_oca = metrics.get(e_col, {}).get("oca", None)
                dn_spearman = metrics.get("metric_averaged_distinct_ngrams", {}).get(
                    "spearman_cor", None
                )
                dn_oca = metrics.get("metric_averaged_distinct_ngrams", {}).get(
                    "oca", None
                )

                if e_spearman is not None:
                    passed = e_spearman > 0.5
                    print(
                        f"  {exp_name}/{sub_exp}: E ρ={e_spearman:.3f} "
                        f"(target > 0.5: {'PASS' if passed else 'FAIL'})"
                    )
                    if dn_spearman is not None:
                        print(f"    vs distinct-n ρ={dn_spearman:.3f}")
                if e_oca is not None:
                    passed = e_oca > 0.75
                    print(
                        f"  {exp_name}/{sub_exp}: E OCA={e_oca:.3f} "
                        f"(target > 0.75: {'PASS' if passed else 'FAIL'})"
                    )
                    if dn_oca is not None:
                        print(f"    vs distinct-n OCA={dn_oca:.3f}")

        print("\n--- H2: D outperforms E on ConTest ---")
        for exp_name, results in contest_results.items():
            for sub_exp, metrics in results.items():
                e_oca = metrics.get(e_col, {}).get("oca", None)
                d_oca = metrics.get(d_col, {}).get("oca", None)
                if e_oca is not None and d_oca is not None:
                    diff = d_oca - e_oca
                    print(
                        f"  {exp_name}/{sub_exp}: D OCA={d_oca:.3f}, "
                        f"E OCA={e_oca:.3f}, diff={diff:+.3f}"
                    )

    # H3: E on DecTest
    dectest_results = {}
    for exp_name, results in all_results.items():
        if "dec_test" in exp_name.lower():
            dectest_results[exp_name] = results

    if dectest_results:
        print("\n--- H3: E is competitive on DecTest ---")
        for exp_name, results in dectest_results.items():
            for sub_exp, metrics in results.items():
                e_spearman = metrics.get(e_col, {}).get("spearman_cor", None)
                dn_spearman = metrics.get("metric_averaged_distinct_ngrams", {}).get(
                    "spearman_cor", None
                )
                if e_spearman is not None:
                    passed = e_spearman > 0.5
                    print(
                        f"  {exp_name}/{sub_exp}: E ρ={e_spearman:.3f} "
                        f"(target > 0.5: {'PASS' if passed else 'FAIL'})"
                    )
                    if dn_spearman is not None:
                        print(f"    vs distinct-n ρ={dn_spearman:.3f}")

    # H4: E on McDiv_nuggets
    nuggets_results = {}
    for exp_name, results in all_results.items():
        if "nuggets" in exp_name.lower():
            nuggets_results[exp_name] = results

    if nuggets_results:
        print("\n--- H4: E excels on McDiv_nuggets ---")
        for exp_name, results in nuggets_results.items():
            for sub_exp, metrics in results.items():
                e_spearman = metrics.get(e_col, {}).get("spearman_cor", None)
                e_oca = metrics.get(e_col, {}).get("oca", None)
                dn_spearman = metrics.get("metric_averaged_distinct_ngrams", {}).get(
                    "spearman_cor", None
                )
                dn_oca = metrics.get("metric_averaged_distinct_ngrams", {}).get(
                    "oca", None
                )
                if e_spearman is not None and dn_spearman is not None:
                    print(
                        f"  {exp_name}/{sub_exp}: E ρ={e_spearman:.3f} "
                        f"vs distinct-n ρ={dn_spearman:.3f}"
                    )
                if e_oca is not None and dn_oca is not None:
                    print(
                        f"  {exp_name}/{sub_exp}: E OCA={e_oca:.3f} "
                        f"vs distinct-n OCA={dn_oca:.3f}"
                    )


def compute_metric_correlations(data_dir: Path, tag: str, output_dir: Path) -> None:
    """H5: Compute correlations between E and other metrics."""
    print(f"\n--- H5: E captures a different signal than existing metrics [{tag}] ---")

    e_col = f"metric_icl_E_{tag}"

    all_e = []
    all_dn = []
    all_sb = []

    for csv_path in sorted(data_dir.rglob("*.csv")):
        data = load_csv_metrics(csv_path)
        e_vals = data.get(e_col)
        dn_vals = data.get("metric_averaged_distinct_ngrams")
        sb_vals = data.get("metric_sent_bert")

        if e_vals is None or dn_vals is None:
            continue

        valid_idx = [
            i
            for i in range(len(e_vals))
            if not (np.isnan(e_vals[i]) if isinstance(e_vals[i], float) else False)
            and not (np.isnan(dn_vals[i]) if isinstance(dn_vals[i], float) else False)
        ]
        if len(valid_idx) < 10:
            continue

        e_valid = [e_vals[i] for i in valid_idx]
        dn_valid = [dn_vals[i] for i in valid_idx]
        all_e.extend(e_valid)
        all_dn.extend(dn_valid)

        if sb_vals is not None:
            sb_valid = [sb_vals[i] for i in valid_idx]
            all_sb.extend(sb_valid)

        r_dn, _ = pearsonr(e_valid, dn_valid)
        print(f"  {csv_path.name}: r(E, distinct-n) = {r_dn:.3f} (n={len(valid_idx)})")

        if sb_vals is not None:
            sb_valid = [sb_vals[i] for i in valid_idx]
            r_sb, _ = pearsonr(e_valid, sb_valid)
            print(f"  {csv_path.name}: r(E, sent-BERT) = {r_sb:.3f}")

    if len(all_e) > 10 and len(all_dn) > 10:
        r_overall, _ = pearsonr(all_e, all_dn)
        print(f"\n  Overall r(E, distinct-n) = {r_overall:.3f} (n={len(all_e)})")
        passed = abs(r_overall) < 0.5
        print(f"  Target |r| < 0.5: {'PASS' if passed else 'FAIL'}")

    if all_e and all_dn:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"H5: ICL E vs Existing Metrics [{tag}]", fontsize=14)

        axes[0].scatter(all_dn, all_e, alpha=0.2, s=5)
        axes[0].set_xlabel("Averaged Distinct N-grams")
        axes[0].set_ylabel("ICL E (excess entropy)")
        if len(all_e) > 1:
            r, _ = pearsonr(all_dn, all_e)
            axes[0].set_title(f"E vs Distinct-n (r={r:.3f})")

        if all_sb:
            axes[1].scatter(all_sb, all_e[: len(all_sb)], alpha=0.2, s=5)
            axes[1].set_xlabel("Sent-BERT")
            axes[1].set_ylabel("ICL E (excess entropy)")
            if len(all_sb) > 1:
                r, _ = pearsonr(all_sb, all_e[: len(all_sb)])
                axes[1].set_title(f"E vs Sent-BERT (r={r:.3f})")

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"metric_correlations_{tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")


def plot_metric_distributions(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot distributions of ICL metrics for high vs low diversity groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    contest_dir = data_dir / "conTest"
    if not contest_dir.exists():
        return

    e_col = f"metric_icl_E_{tag}"
    d_col = f"metric_icl_D_{tag}"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"ICL Metric Distributions: High vs Low Content Diversity (ConTest) [{tag}]",
        fontsize=14,
    )

    for col_idx, csv_path in enumerate(sorted(contest_dir.glob("*.csv"))[:3]):
        data = load_csv_metrics(csv_path)
        labels = data.get("label_value", [])
        task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]

        for row_idx, (metric_name, metric_label) in enumerate(
            [(e_col, "E"), (d_col, "D")]
        ):
            ax = axes[row_idx][col_idx]
            vals = data.get(metric_name, [])
            if not vals:
                continue

            high = [v for v, lab in zip(vals, labels) if lab == 1.0 and not np.isnan(v)]
            low = [v for v, lab in zip(vals, labels) if lab == 0.0 and not np.isnan(v)]

            if high and low:
                hist_range = (min(high + low), max(high + low))
                ax.hist(
                    high,
                    bins=20,
                    alpha=0.5,
                    label=f"Diverse (n={len(high)})",
                    range=hist_range,
                )
                ax.hist(
                    low,
                    bins=20,
                    alpha=0.5,
                    label=f"Constant (n={len(low)})",
                    range=hist_range,
                )
                ax.legend()

            ax.set_title(f"{metric_label} — {task}")
            ax.set_xlabel(f"{metric_label} value")
            ax.set_ylabel("Count")

    plt.tight_layout()
    out_path = output_dir / f"contest_metric_distributions_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_dectest_e_vs_temperature(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Scatter plot of E vs temperature for DecTest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dectest_dir = data_dir / "decTest"
    if not dectest_dir.exists():
        return

    e_col = f"metric_icl_E_{tag}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"DecTest: ICL E vs Temperature [{tag}]", fontsize=14)

    for ax_idx, csv_path in enumerate(sorted(dectest_dir.glob("*1000*.csv"))[:3]):
        data = load_csv_metrics(csv_path)
        temps = data.get("label_value", [])
        e_vals = data.get(e_col, [])
        task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]

        if not temps or not e_vals:
            continue

        valid = [(t, e) for t, e in zip(temps, e_vals) if not np.isnan(e)]
        if not valid:
            continue

        ts, es = zip(*valid)
        ax = axes[ax_idx]
        ax.scatter(ts, es, alpha=0.3, s=10)
        rho, _ = spearmanr(ts, es)
        ax.set_title(f"{task}\nSpearman ρ = {rho:.3f}")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("E (excess entropy)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"dectest_E_vs_temperature_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def analyze_tag(tag: str, output_dir: Path) -> None:
    """Run full analysis for a single run tag."""
    data_dir = RESULTS_BASE / tag
    if not data_dir.exists():
        logger.error(f"No data found for tag '{tag}' at {data_dir}")
        return

    print(f"\n{'#' * 80}")
    print(f"# ANALYSIS FOR RUN TAG: {tag}")
    print(f"{'#' * 80}")

    # Try loading Tevet experiment results
    all_results = load_experiment_results(TEVET_RESULTS_DIR)
    if all_results:
        print_results_table(all_results)
        test_hypotheses(all_results, tag)
    else:
        print("No Tevet experiment results found. Run the pipeline first:")
        print(f"  cd diversity-eval && python run_experiments.py --input_json {data_dir / 'experiments'}")
        print("\nFalling back to direct CSV analysis...")

    # Direct CSV analysis (works without run_experiments.py)
    compute_metric_correlations(data_dir, tag, output_dir)
    plot_metric_distributions(data_dir, tag, output_dir)
    plot_dectest_e_vs_temperature(data_dir, tag, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Tevet validation results")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag to analyze. Default: analyze all available tags.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_tag:
        tags = [args.run_tag]
    else:
        tags = find_run_tags()
        if not tags:
            logger.error("No run tags found. Run compute_icl_metrics_for_tevet.py first.")
            return
        logger.info(f"Found run tags: {tags}")

    for tag in tags:
        analyze_tag(tag, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Plots saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
