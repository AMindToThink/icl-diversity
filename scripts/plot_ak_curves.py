"""Generate a_k curve plots from saved scenario metrics JSON.

Reads results/scenario_metrics.json (produced by run_scenarios.py) and
generates per-scenario and combined a_k curve plots.

Usage:
    uv run scripts/plot_ak_curves.py
    uv run scripts/plot_ak_curves.py --input results/my_run.json
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

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "results" / "scenario_metrics.json"
)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_single_scenario(
    scenario_name: str,
    metrics_list: list[dict[str, Any]],
    ax: plt.Axes,
) -> None:
    """Plot a_k curves for one scenario on the given axes."""
    n_responses = len(metrics_list[0]["a_k_curve"])

    for i, m in enumerate(metrics_list):
        curve = m["a_k_curve"]
        k = np.arange(1, len(curve) + 1)
        label = m.get("prompt_label", f"Prompt {i}")
        color = COLORS[i % len(COLORS)]
        ax.plot(
            k, curve, marker="o", markersize=4, linewidth=1.5, color=color, label=label
        )

        # Show per-permutation curves as faint lines
        if m.get("per_permutation_a_k_curves") is not None:
            for perm_curve in m["per_permutation_a_k_curves"]:
                ax.plot(k, perm_curve, linewidth=0.5, alpha=0.25, color=color)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a_k curves from scenario metrics JSON"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Input JSON path"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURES_DIR, help="Output figures directory"
    )
    args = parser.parse_args()

    print(f"Loading metrics from: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    print("Generating plots...")
    generate_plots(data, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
