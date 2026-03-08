"""Generate a_k curve plots from saved scenario metrics JSON.

Reads results/scenario_metrics.json (produced by run_scenarios.py) and
generates per-scenario and combined a_k curve plots. When multiple input
files are provided, generates side-by-side comparison plots.

Usage:
    uv run scripts/plot_ak_curves.py
    uv run scripts/plot_ak_curves.py --input results/my_run.json
    uv run scripts/plot_ak_curves.py --input results/scenario_metrics.json results/scenario_metrics_qwen.json --output-dir figures/comparison
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    }
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Human-readable scenario names for plot titles
SCENARIO_TITLES = {
    "pure_noise": "Pure noise",
    "multi_incoherent": "Multi incoherent",
    "multi_mode": "Multi mode (3 modes)",
    "one_mode": "One mode (paraphrase)",
    "mixed": "Mixed coherent+incoherent",
}

# Canonical scenario order for consistent plot layout
SCENARIO_ORDER = ["pure_noise", "multi_incoherent", "multi_mode", "one_mode", "mixed"]

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "results" / "scenario_metrics.json"
)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_single_scenario(
    scenario_name: str,
    metrics_list: list[dict[str, Any]],
    ax: plt.Axes,
) -> None:
    """Plot a_k curves for one scenario on the given axes.

    Uses per-byte normalized curves for human-readable y-axis.
    Falls back to a_k_curve if a_k_curve_per_byte is not available
    (backward compat with old JSON files).
    """
    curve_key = (
        "a_k_curve_per_byte" if "a_k_curve_per_byte" in metrics_list[0] else "a_k_curve"
    )
    n_responses = len(metrics_list[0][curve_key])

    for i, m in enumerate(metrics_list):
        curve = m[curve_key]
        k = np.arange(1, len(curve) + 1)
        label = m.get("prompt_label", f"Prompt {i}")
        color = COLORS[i % len(COLORS)]
        ax.plot(
            k, curve, marker="o", markersize=4, linewidth=1.5, color=color, label=label
        )

        # Show per-permutation curves as faint lines
        if m.get("per_permutation_a_k_curves") is not None:
            perm_byte_counts = m.get("per_permutation_byte_counts")
            for j, perm_curve in enumerate(m["per_permutation_a_k_curves"]):
                if perm_byte_counts is not None:
                    # Convert total bits to per-byte for plotting
                    perm_per_byte = [
                        t / b if b > 0 else 0.0
                        for t, b in zip(perm_curve, perm_byte_counts[j])
                    ]
                else:
                    perm_per_byte = perm_curve
                ax.plot(k, perm_per_byte, linewidth=0.5, alpha=0.25, color=color)

    ax.set_title(scenario_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("k (response index)")
    ax.set_ylabel("$a_k$ (bits/byte)")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(range(1, n_responses + 1))


def generate_plots(data: dict[str, Any], figures_dir: Path) -> None:
    """Create individual plots per scenario + one combined overview."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    scenarios = data["scenarios"]

    # Individual scenario plots
    for key, metrics_list in scenarios.items():
        title = SCENARIO_TITLES.get(key, key)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_single_scenario(title, metrics_list, ax)
        fig.tight_layout()
        path = figures_dir / f"ak_curve_{key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Combined 2x3 grid (5 scenarios + info panel)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, (key, metrics_list) in enumerate(scenarios.items()):
        title = SCENARIO_TITLES.get(key, key)
        plot_single_scenario(title, metrics_list, axes_flat[idx])

    # Info panel in the 6th subplot
    ax_extra = axes_flat[5]
    ax_extra.axis("off")
    summary_text = (
        f"ICL Diversity: $a_k$ Curves\n"
        f"─────────────────────\n"
        f"Base model: {data.get('base_model', '?')}\n"
        f"n_permutations: {data.get('n_permutations', '?')}\n"
        f"seed: {data.get('seed', '?')}\n"
        f"n_responses: {data.get('n_responses', '?')}\n\n"
        f"Bold lines: averaged $a_k$\n"
        f"Faint lines: per-permutation $a_k$"
    )
    ax_extra.text(
        0.1,
        0.5,
        summary_text,
        transform=ax_extra.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )

    fig.suptitle(
        "Progressive Conditional Surprise Curves by Scenario",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = figures_dir / "ak_curves_all_scenarios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_comparison_plots(
    datasets: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Generate side-by-side comparison plots for multiple models."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(datasets)
    model_names = [d.get("base_model", f"Model {i}") for i, d in enumerate(datasets)]

    # Collect all scenario keys across datasets (use canonical order)
    all_keys = []
    for key in SCENARIO_ORDER:
        if any(key in d.get("scenarios", {}) for d in datasets):
            all_keys.append(key)

    # Per-scenario comparison: 1 row x N models columns
    for key in all_keys:
        title = SCENARIO_TITLES.get(key, key)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

        # Compute shared y-axis limits across models for this scenario (per-byte)
        y_min, y_max = float("inf"), float("-inf")
        for data in datasets:
            if key in data.get("scenarios", {}):
                for m in data["scenarios"][key]:
                    ck = (
                        "a_k_curve_per_byte"
                        if "a_k_curve_per_byte" in m
                        else "a_k_curve"
                    )
                    curve = m[ck]
                    y_min = min(y_min, min(curve))
                    y_max = max(y_max, max(curve))
                    if m.get("per_permutation_a_k_curves") is not None:
                        pbc = m.get("per_permutation_byte_counts")
                        for j, pc in enumerate(m["per_permutation_a_k_curves"]):
                            if pbc is not None:
                                pb = [
                                    t / b if b > 0 else 0.0 for t, b in zip(pc, pbc[j])
                                ]
                            else:
                                pb = pc
                            y_min = min(y_min, min(pb))
                            y_max = max(y_max, max(pb))
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        y_range = (y_min - y_pad, y_max + y_pad)

        for col, data in enumerate(datasets):
            ax = axes[0, col]
            if key in data.get("scenarios", {}):
                plot_single_scenario(title, data["scenarios"][key], ax)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(
                f"{title}\n({model_names[col]})", fontsize=11, fontweight="bold"
            )
            ax.set_ylim(y_range)

        fig.tight_layout()
        path = figures_dir / f"comparison_ak_curve_{key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Combined overview: scenarios (rows) x models (columns)
    n_scenarios = len(all_keys)
    fig, axes = plt.subplots(
        n_scenarios, n_models, figsize=(7 * n_models, 4 * n_scenarios), squeeze=False
    )

    for row, key in enumerate(all_keys):
        title = SCENARIO_TITLES.get(key, key)

        # Shared y-axis per scenario row (per-byte)
        y_min, y_max = float("inf"), float("-inf")
        for data in datasets:
            if key in data.get("scenarios", {}):
                for m in data["scenarios"][key]:
                    ck = (
                        "a_k_curve_per_byte"
                        if "a_k_curve_per_byte" in m
                        else "a_k_curve"
                    )
                    curve = m[ck]
                    y_min = min(y_min, min(curve))
                    y_max = max(y_max, max(curve))
                    if m.get("per_permutation_a_k_curves") is not None:
                        pbc = m.get("per_permutation_byte_counts")
                        for j, pc in enumerate(m["per_permutation_a_k_curves"]):
                            if pbc is not None:
                                pb = [
                                    t / b if b > 0 else 0.0 for t, b in zip(pc, pbc[j])
                                ]
                            else:
                                pb = pc
                            y_min = min(y_min, min(pb))
                            y_max = max(y_max, max(pb))
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        y_range = (y_min - y_pad, y_max + y_pad)

        for col, data in enumerate(datasets):
            ax = axes[row, col]
            if key in data.get("scenarios", {}):
                plot_single_scenario(title, data["scenarios"][key], ax)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_ylim(y_range)
            if row == 0:
                ax.set_title(
                    f"{model_names[col]}\n{title}", fontsize=11, fontweight="bold"
                )
            else:
                ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Progressive Conditional Surprise Curves: Model Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = figures_dir / "comparison_ak_curves_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a_k curves from scenario metrics JSON"
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT],
        help="Input JSON path(s). Multiple files generate comparison plots.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURES_DIR, help="Output figures directory"
    )
    args = parser.parse_args()

    if len(args.input) == 1:
        # Single file: existing behavior
        print(f"Loading metrics from: {args.input[0]}")
        with open(args.input[0]) as f:
            data = json.load(f)
        print("Generating plots...")
        generate_plots(data, args.output_dir)
    else:
        # Multiple files: comparison mode
        datasets = []
        for path in args.input:
            print(f"Loading metrics from: {path}")
            with open(path) as f:
                datasets.append(json.load(f))
        print(f"Generating comparison plots for {len(datasets)} models...")
        generate_comparison_plots(datasets, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
