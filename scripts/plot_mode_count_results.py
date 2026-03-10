"""Plot mode count experiment results.

Generates several diagnostic plots:
1. a_k curves by m (subplots showing curve shape transition)
2. Overlay plot (all m values on same axes)
3. E vs m (does excess entropy increase with mode count?)
4. a_∞ vs m (does asymptote rise with mode count?)
5. Summary metrics panel

Usage:
    uv run python scripts/plot_mode_count_results.py
    uv run python scripts/plot_mode_count_results.py --input results/mode_count/gpt2.json
    uv run python scripts/plot_mode_count_results.py --input results/mode_count/gpt2.json results/mode_count/qwen2.5-32b.json
"""

import argparse
import json
from collections import defaultdict
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

# Colormap for different m values
M_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    5: "#d62728",
    10: "#9467bd",
    15: "#8c564b",
    25: "#e377c2",
    50: "#7f7f7f",
}

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "results" / "mode_count" / "gpt2.json"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "mode_count"


def load_data(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def group_runs_by_m(data: dict[str, Any]) -> dict[int, list[dict]]:
    """Group runs by mode count m."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in data["runs"]:
        grouped[run["m"]].append(run)
    return dict(sorted(grouped.items()))


def get_color(m: int) -> str:
    if m in M_COLORS:
        return M_COLORS[m]
    # Fallback for unlisted m values
    cmap = plt.cm.viridis
    return cmap(m / 50)


def plot_ak_curves_by_m(grouped: dict[int, list[dict]], figures_dir: Path, model_name: str) -> None:
    """Plot a_k curves as subplots, one per m value."""
    m_values = sorted(grouped.keys())
    n_plots = len(m_values)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, m in enumerate(m_values):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        runs = grouped[m]
        color = get_color(m)

        for run in runs:
            curve_key = "a_k_curve"
            curve = run[curve_key]
            k = np.arange(1, len(curve) + 1)
            ax.plot(k, curve, marker="o", markersize=3, linewidth=1.2, color=color,
                    alpha=0.6, label=f"seed={run['seed']}")

            # Per-permutation curves
            if run.get("per_permutation_a_k_curves"):
                perm_byte_counts = run.get("per_permutation_byte_counts")
                for j, pc in enumerate(run["per_permutation_a_k_curves"]):
                    if perm_byte_counts is not None:
                        pb = [t / b if b > 0 else 0.0 for t, b in zip(pc, perm_byte_counts[j])]
                    else:
                        pb = pc
                    ax.plot(k, pb, linewidth=0.3, alpha=0.15, color=color)

        ax.set_title(f"m = {m} ({m * runs[0].get('n_responses', 0) // m} resp/mode)", fontweight="bold")
        ax.set_xlabel("k")
        ax.set_ylabel("$a_k$ (bits)")
        ax.legend(fontsize=7, loc="best")

    # Hide unused subplots
    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"a_k Curves by Mode Count — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = figures_dir / "ak_curves_by_m.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ak_overlay(grouped: dict[int, list[dict]], figures_dir: Path, model_name: str) -> None:
    """Overlay all m values on same axes (averaged across seeds)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for m, runs in sorted(grouped.items()):
        curve_key = "a_k_curve"
        curves = [run[curve_key] for run in runs]
        # Average across seeds
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)
        k = np.arange(1, max_len + 1)

        color = get_color(m)
        ax.plot(k, mean_curve, marker="o", markersize=4, linewidth=2, color=color, label=f"m = {m}")
        ax.fill_between(k, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=color)

    ax.set_xlabel("k (response index)", fontsize=12)
    ax.set_ylabel("$a_k$ (bits)", fontsize=12)
    ax.set_title(f"a_k Curves Overlay by Mode Count — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    path = figures_dir / "ak_curves_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metrics_vs_m(grouped: dict[int, list[dict]], figures_dir: Path, model_name: str) -> None:
    """Plot E, D, C, a_∞ vs m with error bars."""
    m_values = []
    E_means, E_stds = [], []
    E_rate_means, E_rate_stds = [], []
    D_means, D_stds = [], []
    C_means, C_stds = [], []
    a_inf_means, a_inf_stds = [], []

    for m, runs in sorted(grouped.items()):
        m_values.append(m)
        Es = [r["excess_entropy_E"] for r in runs]
        E_means.append(np.mean(Es))
        E_stds.append(np.std(Es))

        E_rates = [r.get("E_rate", r.get("excess_entropy_E")) for r in runs]
        E_rate_means.append(np.mean(E_rates))
        E_rate_stds.append(np.std(E_rates))

        Ds = [r["diversity_score_D"] for r in runs]
        D_means.append(np.mean(Ds))
        D_stds.append(np.std(Ds))

        Cs = [r["coherence_C"] for r in runs]
        C_means.append(np.mean(Cs))
        C_stds.append(np.std(Cs))

        # a_∞ approximation: last point of a_k curve
        curve_key = "a_k_curve"
        a_infs = [r[curve_key][-1] for r in runs]
        a_inf_means.append(np.mean(a_infs))
        a_inf_stds.append(np.std(a_infs))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # E vs m
    ax = axes[0, 0]
    ax.errorbar(m_values, E_means, yerr=E_stds, marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("E (excess entropy, bits)")
    ax.set_title("E vs m", fontweight="bold")

    # E_rate vs m
    ax = axes[0, 1]
    ax.errorbar(m_values, E_rate_means, yerr=E_rate_stds, marker="s", capsize=4, linewidth=2, color="#ff7f0e")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("E_rate (bits/byte)")
    ax.set_title("E_rate vs m", fontweight="bold")

    # D vs m
    ax = axes[0, 2]
    ax.errorbar(m_values, D_means, yerr=D_stds, marker="^", capsize=4, linewidth=2, color="#2ca02c")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("D (diversity score)")
    ax.set_title("D vs m", fontweight="bold")

    # C vs m
    ax = axes[1, 0]
    ax.errorbar(m_values, C_means, yerr=C_stds, marker="d", capsize=4, linewidth=2, color="#d62728")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("C (coherence)")
    ax.set_title("C vs m", fontweight="bold")

    # a_∞ vs m
    ax = axes[1, 1]
    ax.errorbar(m_values, a_inf_means, yerr=a_inf_stds, marker="v", capsize=4, linewidth=2, color="#9467bd")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("$a_n$ (last curve point, bits)")
    ax.set_title("$a_\\infty$ approx vs m", fontweight="bold")

    # Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for i, m in enumerate(m_values):
        table_data.append([
            str(m),
            f"{E_means[i]:.2f}±{E_stds[i]:.2f}",
            f"{D_means[i]:.2f}±{D_stds[i]:.2f}",
            f"{C_means[i]:.3f}±{C_stds[i]:.3f}",
            f"{a_inf_means[i]:.3f}±{a_inf_stds[i]:.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["m", "E", "D", "C", "a_∞"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Summary", fontweight="bold")

    fig.suptitle(f"ICL Diversity Metrics vs Mode Count — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = figures_dir / "metrics_vs_m.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison(datasets: list[tuple[str, dict]], figures_dir: Path) -> None:
    """Compare mode count results across multiple models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = [
        ("excess_entropy_E", "E (excess entropy, bits)"),
        ("diversity_score_D", "D (diversity score)"),
        ("coherence_C", "C (coherence)"),
    ]

    for col, (metric_key, ylabel) in enumerate(metrics_to_plot):
        ax = axes[col]
        for model_name, data in datasets:
            grouped = group_runs_by_m(data)
            m_vals, means, stds = [], [], []
            for m, runs in sorted(grouped.items()):
                vals = [r[metric_key] for r in runs]
                m_vals.append(m)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(m_vals, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=model_name)
        ax.set_xlabel("m (mode count)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip(), fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Mode Count: Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = figures_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mode count experiment results")
    parser.add_argument(
        "--input", type=Path, nargs="+", default=[DEFAULT_INPUT],
        help="Input JSON path(s). Multiple files generate comparison plots.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURES_DIR,
        help="Output figures directory",
    )
    args = parser.parse_args()
    figures_dir = args.output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    if len(args.input) == 1:
        data = load_data(args.input[0])
        model_name = data.get("base_model", "unknown")
        grouped = group_runs_by_m(data)
        print(f"Loaded {len(data['runs'])} runs for {model_name}")
        print(f"Mode counts: {sorted(grouped.keys())}")

        plot_ak_curves_by_m(grouped, figures_dir, model_name)
        plot_ak_overlay(grouped, figures_dir, model_name)
        plot_metrics_vs_m(grouped, figures_dir, model_name)
    else:
        datasets = []
        for path in args.input:
            data = load_data(path)
            model_name = data.get("base_model", f"Model")
            datasets.append((model_name, data))
            grouped = group_runs_by_m(data)
            sub_dir = figures_dir / model_name.replace("/", "_")
            sub_dir.mkdir(parents=True, exist_ok=True)
            plot_ak_curves_by_m(grouped, sub_dir, model_name)
            plot_ak_overlay(grouped, sub_dir, model_name)
            plot_metrics_vs_m(grouped, sub_dir, model_name)

        plot_comparison(datasets, figures_dir)

    print("Done!")


if __name__ == "__main__":
    main()
