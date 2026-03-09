"""
Plot a_k curves from Tevet diversity-eval ICL metric sidecar JSON files.

Generates:
1. Per-task summary: mean a_k curves for high vs low diversity (ConTest)
2. Temperature sweep: a_k curves at representative temperatures (DecTest)
3. Example curves: individual high-D, low-D, and edge-case response sets

Usage:
    uv run python scripts/plot_tevet_ak_curves.py
    uv run python scripts/plot_tevet_ak_curves.py --output-dir figures/tevet_validation
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
DATA_DIR = PROJECT_ROOT / "diversity-eval" / "data" / "with_metrics"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "tevet_validation"

TASK_LABELS = {
    "story_gen": "storyGen (ROC Stories)",
    "resp_gen": "respGen (Reddit Dialog)",
    "prompt_gen": "promptGen (GPT-2 Completions)",
}


def load_csv_and_sidecar(
    csv_path: Path,
) -> tuple[list[dict], dict] | None:
    """Load CSV rows and corresponding sidecar JSON."""
    sidecar_path = csv_path.with_suffix(".icl_curves.json")
    if not sidecar_path.exists():
        logger.warning(f"No sidecar found for {csv_path.name}")
        return None

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    return rows, sidecar


def get_task_name_from_filename(filename: str) -> str:
    """Extract task name (story_gen, resp_gen, prompt_gen) from CSV filename."""
    for task in ["story_gen", "resp_gen", "prompt_gen"]:
        if task in filename:
            return task
    return "unknown"


def plot_contest_summary(output_dir: Path) -> None:
    """Plot mean a_k curves for high vs low diversity per task (ConTest)."""
    contest_dir = DATA_DIR / "conTest"
    if not contest_dir.exists():
        logger.warning("No conTest data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "ConTest: Mean a_k Curves — High vs Low Content Diversity", fontsize=14
    )

    for ax_idx, csv_path in enumerate(sorted(contest_dir.glob("*.csv"))):
        result = load_csv_and_sidecar(csv_path)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        high_curves, low_curves = [], []
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar:
                continue
            curve = sidecar[sid].get("a_k_curve_per_byte")
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
            # Pad curves to same length with NaN for averaging
            max_len = max(len(c) for c in curves)
            padded = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                padded[i, : len(c)] = c
            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            x = np.arange(1, max_len + 1)
            ax.plot(x, mean_curve, label=f"{label} (n={len(curves)})", color=color)
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Response index k")
        ax.set_ylabel("a_k (bits/byte)")
        ax.set_title(task_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "contest_ak_curves_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_dectest_temperature_sweep(output_dir: Path) -> None:
    """Plot a_k curves at representative temperatures (DecTest)."""
    dectest_dir = DATA_DIR / "decTest"
    if not dectest_dir.exists():
        logger.warning("No decTest data found")
        return

    # Use the 1000-sample no-hds files for richer data
    target_temps = [0.3, 0.5, 0.8, 1.0, 1.2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DecTest: Mean a_k Curves by Temperature", fontsize=14)

    for ax_idx, csv_path in enumerate(sorted(dectest_dir.glob("*1000*.csv"))):
        result = load_csv_and_sidecar(csv_path)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        # Group curves by temperature
        temp_curves: dict[float, list] = {}
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar:
                continue
            curve = sidecar[sid].get("a_k_curve_per_byte")
            if curve is None:
                continue
            temp = float(row["label_value"])
            temp_curves.setdefault(temp, []).append(curve)

        ax = axes[ax_idx] if ax_idx < 3 else axes[-1]
        task_label = TASK_LABELS.get(task, task)
        cmap = plt.cm.coolwarm

        # Find closest available temperatures to targets
        available_temps = sorted(temp_curves.keys())
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
        ax.set_ylabel("a_k (bits/byte)")
        ax.set_title(task_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "dectest_ak_curves_temperature.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

    # Also try with 200-sample with_hds files
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        "DecTest (200, with HDS): Mean a_k Curves by Temperature", fontsize=14
    )

    for ax_idx, csv_path in enumerate(sorted(dectest_dir.glob("*200*.csv"))):
        result = load_csv_and_sidecar(csv_path)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        temp_curves = {}
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar:
                continue
            curve = sidecar[sid].get("a_k_curve_per_byte")
            if curve is None:
                continue
            temp = float(row["label_value"])
            temp_curves.setdefault(temp, []).append(curve)

        ax = axes2[ax_idx] if ax_idx < 3 else axes2[-1]
        task_label = TASK_LABELS.get(task, task)
        cmap = plt.cm.coolwarm

        available_temps = sorted(temp_curves.keys())
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
        ax.set_ylabel("a_k (bits/byte)")
        ax.set_title(task_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = output_dir / "dectest_200_ak_curves_temperature.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved {out_path2}")


def plot_mcdiv_nuggets_summary(output_dir: Path) -> None:
    """Plot mean a_k curves for high vs low diversity (McDiv_nuggets)."""
    nuggets_dir = DATA_DIR / "McDiv_nuggets"
    if not nuggets_dir.exists():
        logger.warning("No McDiv_nuggets data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "McDiv_nuggets: Mean a_k Curves — High vs Low Content Diversity", fontsize=14
    )

    csv_files = sorted(nuggets_dir.glob("*.csv"))
    for ax_idx, csv_path in enumerate(csv_files[:3]):
        result = load_csv_and_sidecar(csv_path)
        if result is None:
            continue
        rows, sidecar = result
        task = get_task_name_from_filename(csv_path.name)

        high_curves, low_curves = [], []
        for row in rows:
            sid = row["sample_id"]
            if sid not in sidecar:
                continue
            curve = sidecar[sid].get("a_k_curve_per_byte")
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
            max_len = max(len(c) for c in curves)
            padded = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                padded[i, : len(c)] = c
            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            x = np.arange(1, max_len + 1)
            ax.plot(x, mean_curve, label=f"{label} (n={len(curves)})", color=color)
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Response index k")
        ax.set_ylabel("a_k (bits/byte)")
        ax.set_title(task_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "mcdiv_nuggets_ak_curves_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_example_curves(output_dir: Path) -> None:
    """Plot individual a_k curves for interesting examples (high-D, low-D, edge cases)."""
    contest_dir = DATA_DIR / "conTest"
    if not contest_dir.exists():
        return

    # Pick story_gen as representative task
    csv_path = next(contest_dir.glob("*story_gen*.csv"), None)
    if csv_path is None:
        return

    result = load_csv_and_sidecar(csv_path)
    if result is None:
        return
    rows, sidecar = result

    # Collect (sample_id, E, label, curves) for rows that have sidecar data
    examples = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in sidecar:
            continue
        metrics = sidecar[sid].get("metrics", {})
        e_val = metrics.get("excess_entropy_E", 0)
        label = float(row["label_value"])
        per_perm = sidecar[sid].get("per_permutation_a_k_curves")
        mean_curve = sidecar[sid].get("a_k_curve_per_byte")
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

    # Sort by E and pick interesting examples
    examples.sort(key=lambda x: x["E"])
    picks = {
        "Lowest E (low diversity)": examples[0],
        "Median E": examples[len(examples) // 2],
        "Highest E (high diversity)": examples[-1],
    }

    # Also find a misclassification edge case if possible
    high_label_low_e = [e for e in examples if e["label"] == 1.0]
    low_label_high_e = [e for e in examples if e["label"] == 0.0]
    if high_label_low_e:
        picks["High-label, lowest E"] = high_label_low_e[0]
    if low_label_high_e:
        picks["Low-label, highest E"] = low_label_high_e[-1]

    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 5))
    fig.suptitle("Example a_k Curves (ConTest storyGen)", fontsize=14)

    for ax_idx, (title, ex) in enumerate(picks.items()):
        ax = axes[ax_idx] if len(picks) > 1 else axes
        curve = ex["mean_curve"]
        x = np.arange(1, len(curve) + 1)
        ax.plot(x, curve, "k-", linewidth=2, label="Mean")

        # Plot per-permutation curves if available
        if ex["per_perm_curves"]:
            for pc in ex["per_perm_curves"][:20]:  # Limit to 20 for clarity
                ax.plot(
                    np.arange(1, len(pc) + 1),
                    # Normalize to per-byte if these are total bits
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
    out_path = output_dir / "example_ak_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a_k curves from Tevet sidecar JSONs"
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

    plot_contest_summary(output_dir)
    plot_dectest_temperature_sweep(output_dir)
    plot_mcdiv_nuggets_summary(output_dir)
    plot_example_curves(output_dir)

    logger.info("All plots generated!")


if __name__ == "__main__":
    main()
