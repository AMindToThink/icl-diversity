"""
Plot a_k curves from Tevet diversity-eval ICL metric sidecar JSON files.

Generates:
1. Per-task summary: mean a_k curves for high vs low diversity (ConTest)
2. Temperature sweep: a_k curves at representative temperatures (DecTest)
3. Example curves: individual high-D, low-D, and edge-case response sets

Usage:
    # Plot from a specific run tag
    uv run python scripts/plot_tevet_ak_curves.py --run-tag gpt2

    # Default: auto-detect run tags under results/tevet/
    uv run python scripts/plot_tevet_ak_curves.py

    # Custom output directory
    uv run python scripts/plot_tevet_ak_curves.py --run-tag gpt2 --output-dir figures/tevet_gpt2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "tevet_validation"

TASK_LABELS = {
    "story_gen": "storyGen (ROC Stories)",
    "resp_gen": "respGen (Reddit Dialog)",
    "prompt_gen": "promptGen (GPT-2 Completions)",
}


def find_run_tags() -> list[str]:
    """Find available run tags under results/tevet/."""
    if not RESULTS_BASE.exists():
        return []
    return sorted(
        d.name for d in RESULTS_BASE.iterdir()
        if d.is_dir() and (d / "run_config.json").exists()
    )


def load_csv_and_sidecar(
    csv_path: Path,
    tag: str,
) -> tuple[list[dict], dict] | None:
    """Load CSV rows and corresponding sidecar/mean-curves JSON.

    Tries the full sidecar first (*.icl_curves.{tag}.json), then falls
    back to the lightweight mean curves (*.icl_mean_curves.{tag}.json).
    """
    sidecar_path = csv_path.with_suffix(f".icl_curves.{tag}.json")
    mean_curves_path = csv_path.with_suffix(f".icl_mean_curves.{tag}.json")

    curves_path = None
    if sidecar_path.exists():
        curves_path = sidecar_path
    elif mean_curves_path.exists():
        curves_path = mean_curves_path
    else:
        logger.warning(f"No sidecar/mean-curves found for {csv_path.name} (tag={tag})")
        return None

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(curves_path) as f:
        sidecar = json.load(f)

    return rows, sidecar


def get_task_name_from_filename(filename: str) -> str:
    """Extract task name (story_gen, resp_gen, prompt_gen) from CSV filename."""
    for task in ["story_gen", "resp_gen", "prompt_gen"]:
        if task in filename:
            return task
    return "unknown"


def plot_mean_with_bands(
    ax: plt.Axes,
    curves: list[list[float]],
    label: str,
    color: str,
) -> None:
    """Plot mean curve with ±1 SD (light) and ±1 SEM (dark) bands."""
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c
    n = np.sum(~np.isnan(padded), axis=0)
    mean_curve = np.nanmean(padded, axis=0)
    std_curve = np.nanstd(padded, axis=0)
    sem_curve = std_curve / np.sqrt(np.maximum(n, 1))
    x = np.arange(1, max_len + 1)
    ax.plot(x, mean_curve, label=f"{label} (n={len(curves)})", color=color)
    ax.fill_between(
        x, mean_curve - std_curve, mean_curve + std_curve,
        alpha=0.1, color=color, label="  ±1 SD",
    )
    ax.fill_between(
        x, mean_curve - sem_curve, mean_curve + sem_curve,
        alpha=0.3, color=color, label="  ±1 SEM",
    )


def plot_contest_summary(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot mean a_k curves for high vs low diversity per task (ConTest)."""
    contest_dir = data_dir / "conTest"
    if not contest_dir.exists():
        logger.warning("No conTest data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"ConTest: Mean a_k Curves — High vs Low Content Diversity [{tag}]", fontsize=14
    )

    for ax_idx, csv_path in enumerate(sorted(contest_dir.glob("*.csv"))):
        result = load_csv_and_sidecar(csv_path, tag)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        high_curves, low_curves = [], []
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar or sid.startswith("__"):
                continue
            entry = sidecar[sid]
            if entry.get("skipped"):
                continue
            curve = entry.get("a_k_curve")
            if curve is None:
                continue
            label = float(row["label_value"])
            if label == 1.0:
                high_curves.append(curve)
            else:
                low_curves.append(curve)

        ax = axes[ax_idx] if ax_idx < 3 else axes[-1]
        task_label = TASK_LABELS.get(task, task)

        for curves, label, color in [
            (high_curves, "High diversity", "tab:red"),
            (low_curves, "Low diversity", "tab:blue"),
        ]:
            if not curves:
                continue
            plot_mean_with_bands(ax, curves, label, color)

        ax.set_xlabel("Response index k")
        ax.set_ylabel("a_k (total bits)")
        ax.set_title(task_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"contest_ak_curves_summary_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_dectest_temperature_sweep(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot a_k curves at representative temperatures (DecTest)."""
    dectest_dir = data_dir / "decTest"
    if not dectest_dir.exists():
        logger.warning("No decTest data found")
        return

    target_temps = [0.3, 0.5, 0.8, 1.0, 1.2]

    for suffix, glob_pattern, title_extra in [
        ("", "*1000*.csv", ""),
        ("_200", "*200*.csv", " (200, with HDS)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"DecTest{title_extra}: Mean a_k Curves by Temperature [{tag}]", fontsize=14
        )

        for ax_idx, csv_path in enumerate(sorted(dectest_dir.glob(glob_pattern))[:3]):
            result = load_csv_and_sidecar(csv_path, tag)
            if result is None:
                continue
            rows, sidecar = result
            task = get_task_name_from_filename(csv_path.name)

            temp_curves: dict[float, list] = {}
            for row in rows:
                sid = row["sample_id"]
                if sid not in sidecar or sid.startswith("__"):
                    continue
                entry = sidecar[sid]
                if entry.get("skipped"):
                    continue
                curve = entry.get("a_k_curve")
                if curve is None:
                    continue
                temp = float(row["label_value"])
                temp_curves.setdefault(temp, []).append(curve)

            ax = axes[ax_idx] if ax_idx < 3 else axes[-1]
            task_label = TASK_LABELS.get(task, task)
            cmap = plt.cm.coolwarm

            available_temps = sorted(temp_curves.keys())
            if not available_temps:
                continue
            plot_temps = []
            for target in target_temps:
                closest = min(available_temps, key=lambda t: abs(t - target))
                if closest not in plot_temps:
                    plot_temps.append(closest)

            for i, temp in enumerate(sorted(plot_temps)):
                curves = temp_curves[temp]
                max_len = max(len(c) for c in curves)
                padded = np.full((len(curves), max_len), np.nan)
                for j, c in enumerate(curves):
                    padded[j, : len(c)] = c
                mean_curve = np.nanmean(padded, axis=0)
                x = np.arange(1, max_len + 1)
                color = cmap(i / max(len(plot_temps) - 1, 1))
                ax.plot(x, mean_curve, label=f"τ={temp:.1f} (n={len(curves)})", color=color)

            ax.set_xlabel("Response index k")
            ax.set_ylabel("a_k (total bits)")
            ax.set_title(task_label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"dectest{suffix}_ak_curves_temperature_{tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")


def plot_mcdiv_nuggets_summary(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot mean a_k curves for high vs low diversity (McDiv_nuggets)."""
    nuggets_dir = data_dir / "McDiv_nuggets"
    if not nuggets_dir.exists():
        logger.warning("No McDiv_nuggets data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"McDiv_nuggets: Mean a_k Curves — High vs Low Content Diversity [{tag}]",
        fontsize=14,
    )

    csv_files = sorted(nuggets_dir.glob("*.csv"))
    for ax_idx, csv_path in enumerate(csv_files[:3]):
        result = load_csv_and_sidecar(csv_path, tag)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        high_curves, low_curves = [], []
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar or sid.startswith("__"):
                continue
            entry = sidecar[sid]
            if entry.get("skipped"):
                continue
            curve = entry.get("a_k_curve")
            if curve is None:
                continue
            label = float(row["label_value"])
            if label == 1.0:
                high_curves.append(curve)
            else:
                low_curves.append(curve)

        ax = axes[ax_idx] if ax_idx < 3 else axes[-1]
        task_label = TASK_LABELS.get(task, task)

        for curves, label, color in [
            (high_curves, "High diversity", "tab:red"),
            (low_curves, "Low diversity", "tab:blue"),
        ]:
            if not curves:
                continue
            plot_mean_with_bands(ax, curves, label, color)

        ax.set_xlabel("Response index k")
        ax.set_ylabel("a_k (total bits)")
        ax.set_title(task_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"mcdiv_nuggets_ak_curves_summary_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_example_curves(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot individual a_k curves for interesting examples."""
    contest_dir = data_dir / "conTest"
    if not contest_dir.exists():
        return

    csv_path = next(contest_dir.glob("*story_gen*.csv"), None)
    if csv_path is None:
        return

    result = load_csv_and_sidecar(csv_path, tag)
    if result is None:
        return
    rows, sidecar = result

    examples = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in sidecar or sid.startswith("__"):
            continue
        entry = sidecar[sid]
        if entry.get("skipped"):
            continue
        metrics = entry.get("metrics", {})
        e_val = metrics.get("excess_entropy_E", 0)
        label = float(row["label_value"])
        per_perm = entry.get("per_permutation_a_k_curves")
        mean_curve = entry.get("a_k_curve")
        if mean_curve is None:
            continue
        examples.append(
            {
                "sample_id": sid,
                "E": e_val,
                "label": label,
                "mean_curve": mean_curve,
                "per_perm_curves": per_perm,
            }
        )

    if not examples:
        return

    examples.sort(key=lambda x: x["E"])
    picks = {
        "Lowest E (low diversity)": examples[0],
        "Median E": examples[len(examples) // 2],
        "Highest E (high diversity)": examples[-1],
    }

    high_label_low_e = [e for e in examples if e["label"] == 1.0]
    low_label_high_e = [e for e in examples if e["label"] == 0.0]
    if high_label_low_e:
        picks["High-label, lowest E"] = high_label_low_e[0]
    if low_label_high_e:
        picks["Low-label, highest E"] = low_label_high_e[-1]

    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 5))
    fig.suptitle(f"Example a_k Curves (ConTest storyGen) [{tag}]", fontsize=14)

    for ax_idx, (title, ex) in enumerate(picks.items()):
        ax = axes[ax_idx] if len(picks) > 1 else axes
        curve = ex["mean_curve"]
        x = np.arange(1, len(curve) + 1)
        ax.plot(x, curve, "k-", linewidth=2, label="Mean")

        if ex["per_perm_curves"]:
            for pc in ex["per_perm_curves"][:20]:
                ax.plot(
                    np.arange(1, len(pc) + 1),
                    pc,
                    alpha=0.1,
                    color="gray",
                )

        label_str = "diverse" if ex["label"] == 1.0 else "constant"
        ax.set_title(f"{title}\nE={ex['E']:.2f}, label={label_str}")
        ax.set_xlabel("Response index k")
        ax.set_ylabel("a_k (total bits)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"example_ak_curves_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_per_byte_summary(
    data_dir: Path,
    tag: str,
    output_dir: Path,
    subdir: str,
    title_prefix: str,
    out_filename: str,
) -> None:
    """Plot mean per-byte a_k curves for high vs low diversity."""
    target_dir = data_dir / subdir
    if not target_dir.exists():
        logger.warning(f"No {subdir} data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"{title_prefix}: Mean a_k Curves (per-byte) — High vs Low [{tag}]",
        fontsize=14,
    )

    csv_files = sorted(target_dir.glob("*.csv"))
    for ax_idx, csv_path in enumerate(csv_files[:3]):
        result = load_csv_and_sidecar(csv_path, tag)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        high_curves, low_curves = [], []
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar or sid.startswith("__"):
                continue
            entry = sidecar[sid]
            if entry.get("skipped"):
                continue
            curve = entry.get("a_k_curve_per_byte")
            if curve is None:
                continue
            label = float(row["label_value"])
            if label == 1.0:
                high_curves.append(curve)
            else:
                low_curves.append(curve)

        ax = axes[ax_idx] if ax_idx < 3 else axes[-1]
        task_label = TASK_LABELS.get(task, task)

        for curves, label, color in [
            (high_curves, "High diversity", "tab:red"),
            (low_curves, "Low diversity", "tab:blue"),
        ]:
            if not curves:
                continue
            plot_mean_with_bands(ax, curves, label, color)

        ax.set_xlabel("Response index k")
        ax.set_ylabel("a_k (bits/byte)")
        ax.set_title(task_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{out_filename}_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_contest_per_byte(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot per-byte a_k curves for ConTest."""
    _plot_per_byte_summary(
        data_dir, tag, output_dir,
        subdir="conTest",
        title_prefix="ConTest",
        out_filename="contest_ak_curves_per_byte",
    )


def plot_mcdiv_nuggets_per_byte(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Plot per-byte a_k curves for McDiv_nuggets."""
    _plot_per_byte_summary(
        data_dir, tag, output_dir,
        subdir="McDiv_nuggets",
        title_prefix="McDiv_nuggets",
        out_filename="mcdiv_nuggets_ak_curves_per_byte",
    )


def plot_e_rate_summary(data_dir: Path, tag: str, output_dir: Path) -> None:
    """Print and plot E vs E_rate comparison across all ConTest datasets."""
    e_col = f"metric_icl_E_{tag}"
    e_rate_col = f"metric_icl_E_rate_{tag}"

    rows_out: list[dict] = []
    for subdir in ["conTest", "McDiv_nuggets"]:
        target_dir = data_dir / subdir
        if not target_dir.exists():
            continue
        for csv_path in sorted(target_dir.glob("*.csv")):
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            task = get_task_name_from_filename(csv_path.name)
            for metric_name, metric_col in [("E", e_col), ("E_rate", e_rate_col)]:
                high = [
                    float(r[metric_col]) for r in rows
                    if r["label_value"] == "1.0" and r.get(metric_col, "") != ""
                ]
                low = [
                    float(r[metric_col]) for r in rows
                    if r["label_value"] == "0.0" and r.get(metric_col, "") != ""
                ]
                if high and low:
                    rows_out.append({
                        "dataset": subdir,
                        "task": task,
                        "metric": metric_name,
                        "high_mean": np.mean(high),
                        "low_mean": np.mean(low),
                        "diff": np.mean(high) - np.mean(low),
                        "n_high": len(high),
                        "n_low": len(low),
                    })

    if not rows_out:
        return

    # Print table
    print(f"\n{'=' * 90}")
    print(f"E vs E_rate Summary [{tag}]")
    print(f"{'=' * 90}")
    print(f"{'Dataset':<16s} {'Task':<12s} {'Metric':<8s} {'High mean':>10s} {'Low mean':>10s} {'Diff':>10s}")
    print(f"{'-' * 90}")
    for r in rows_out:
        print(
            f"{r['dataset']:<16s} {r['task']:<12s} {r['metric']:<8s} "
            f"{r['high_mean']:>10.3f} {r['low_mean']:>10.3f} {r['diff']:>+10.3f}"
        )

    # Bar chart: diff (high - low) for E and E_rate side by side
    datasets = sorted(set((r["dataset"], r["task"]) for r in rows_out))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.35

    e_diffs = []
    e_rate_diffs = []
    labels = []
    for ds, task in datasets:
        e_row = next((r for r in rows_out if r["dataset"] == ds and r["task"] == task and r["metric"] == "E"), None)
        er_row = next((r for r in rows_out if r["dataset"] == ds and r["task"] == task and r["metric"] == "E_rate"), None)
        e_diffs.append(e_row["diff"] if e_row else 0)
        e_rate_diffs.append(er_row["diff"] if er_row else 0)
        labels.append(f"{ds}\n{task}")

    ax.bar(x - width / 2, e_diffs, width, label="E (total bits)", color="tab:blue")
    ax.bar(x + width / 2, e_rate_diffs, width, label="E_rate (bits/byte)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean(high diversity) − Mean(low diversity)")
    ax.set_title(f"E vs E_rate: Separation Between Diverse and Constant [{tag}]")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / f"e_vs_e_rate_summary_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a_k curves from Tevet sidecar JSONs"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag to plot. Default: plot all available tags.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which tags to plot
    if args.run_tag:
        tags = [args.run_tag]
    else:
        tags = find_run_tags()
        if not tags:
            logger.error("No run tags found under results/tevet/. Run compute_icl_metrics_for_tevet.py first.")
            return
        logger.info(f"Found run tags: {tags}")

    for tag in tags:
        data_dir = RESULTS_BASE / tag
        logger.info(f"Plotting for tag: {tag}")

        plot_contest_summary(data_dir, tag, output_dir)
        plot_contest_per_byte(data_dir, tag, output_dir)
        plot_dectest_temperature_sweep(data_dir, tag, output_dir)
        plot_mcdiv_nuggets_summary(data_dir, tag, output_dir)
        plot_mcdiv_nuggets_per_byte(data_dir, tag, output_dir)
        plot_example_curves(data_dir, tag, output_dir)
        plot_e_rate_summary(data_dir, tag, output_dir)

    logger.info("All plots generated!")


if __name__ == "__main__":
    main()
