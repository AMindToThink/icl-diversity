"""Plot mode count experiment results.

Generates several diagnostic plots:
1. a_k curves by m (subplots showing curve shape transition)
2. Overlay plot (all m values on same axes)
3. Metrics panel: E, E_sig, D, D_sig, C, a_n, σ_ℓ
4. Comparison across models (when multiple inputs given)

Usage:
    uv run python scripts/plot_mode_count_results.py
    uv run python scripts/plot_mode_count_results.py --input results/mode_count/gpt2_50seeds.json
    uv run python scripts/plot_mode_count_results.py --input results/mode_count/gpt2_50seeds.json results/mode_count/qwen2.5-32b.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fit_ak_curves import fit_sigmoid

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    }
)

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "results" / "mode_count" / "gpt2_1k_draws.json"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "mode_count"


def _get_color_for_m(m: int, m_values: list[int]) -> str:
    """Get a color for mode count m using a perceptually uniform colormap."""
    if len(m_values) <= 1:
        return "#1f77b4"
    cmap = plt.cm.viridis
    idx = m_values.index(m)
    return cmap(idx / (len(m_values) - 1))


def load_data(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def group_runs_by_m(data: dict[str, Any]) -> dict[int, list[dict]]:
    """Group runs by mode count m."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in data["runs"]:
        grouped[run["m"]].append(run)
    return dict(sorted(grouped.items()))


def plot_ak_curves_by_m(grouped: dict[int, list[dict]], figures_dir: Path, model_name: str) -> None:
    """Plot a_k curves as subplots, one per m value."""
    m_values = sorted(grouped.keys())
    n_plots = len(m_values)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, m in enumerate(m_values):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        runs = grouped[m]
        color = _get_color_for_m(m, m_values)

        # Plot a sample of individual runs (up to 5 for readability)
        sample_runs = runs[:5]
        for run in sample_runs:
            curve = run["a_k_curve"]
            k = np.arange(1, len(curve) + 1)
            ax.plot(k, curve, marker="o", markersize=2, linewidth=0.8, color=color, alpha=0.4)

        # Plot mean ± SD
        curves = np.array([r["a_k_curve"] for r in runs])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        k = np.arange(1, len(mean_curve) + 1)
        ax.plot(k, mean_curve, marker="o", markersize=4, linewidth=2, color=color, label="mean")
        ax.fill_between(k, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=color)

        n_resp = runs[0].get("n_responses", 0)
        ax.set_title(f"m = {m} ({n_resp // m} resp/mode)", fontweight="bold")
        ax.set_xlabel("k")
        ax.set_ylabel("$a_k$ (bits)")

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
    """Overlay all m values on same axes (averaged across draws)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    m_values = sorted(grouped.keys())

    for m, runs in sorted(grouped.items()):
        curves = np.array([r["a_k_curve"] for r in runs])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        k = np.arange(1, len(mean_curve) + 1)

        color = _get_color_for_m(m, m_values)
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
    """Plot E, E_sig, E_fit, D variants, C, a_n, σ_ℓ vs m with error bars."""
    m_values = []
    E_means, E_stds = [], []
    E_sig_means, E_sig_stds = [], []
    E_fit_vals = []
    D_means, D_stds = [], []
    D_sig_means, D_sig_stds = [], []
    C_means, C_stds = [], []
    a_n_means, a_n_stds = [], []
    sigma_means, sigma_stds = [], []

    for m, runs in sorted(grouped.items()):
        m_values.append(m)
        Es = [r["excess_entropy_E"] for r in runs]
        E_means.append(np.mean(Es))
        E_stds.append(np.std(Es))

        # Fit sigmoid to the mean curve (1 fit per m, not per run)
        curves = np.array([r["a_k_curve"] for r in runs])
        mean_curve = np.mean(curves, axis=0)
        k = np.arange(1, len(mean_curve) + 1, dtype=float)
        fit_params, success = fit_sigmoid(k, mean_curve)

        if success:
            a_inf = fit_params["a_inf"]
            # E_sig: discrete sum using fitted a_inf as floor, per run
            E_sigs = [float(sum(a_k - a_inf for a_k in r["a_k_curve"])) for r in runs]
            D_sigs = [r["coherence_C"] * e_sig for r, e_sig in zip(runs, E_sigs)]
            E_sig_means.append(np.mean(E_sigs))
            E_sig_stds.append(np.std(E_sigs))
            D_sig_means.append(np.mean(D_sigs))
            D_sig_stds.append(np.std(D_sigs))
            # E_fit: single value from integral of mean curve's fit
            E_fit_vals.append(fit_params["E_fit"])
        else:
            E_sig_means.append(np.nan)
            E_sig_stds.append(np.nan)
            D_sig_means.append(np.nan)
            D_sig_stds.append(np.nan)
            E_fit_vals.append(np.nan)

        Cs = [r["coherence_C"] for r in runs]
        C_means.append(np.mean(Cs))
        C_stds.append(np.std(Cs))

        Ds = [r["diversity_score_D"] for r in runs]
        D_means.append(np.mean(Ds))
        D_stds.append(np.std(Ds))

        # a_n: last point of a_k curve
        a_ns = [r["a_k_curve"][-1] for r in runs]
        a_n_means.append(np.mean(a_ns))
        a_n_stds.append(np.std(a_ns))

        # σ_ℓ: coherence spread from core
        sigmas = [r["coherence_spread_sigma"] for r in runs]
        sigma_means.append(np.mean(sigmas))
        sigma_stds.append(np.std(sigmas))

    # D_fit: single value per m (E_fit * mean C)
    D_fit_vals = [e * c for e, c in zip(E_fit_vals, C_means)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # E vs m
    ax = axes[0, 0]
    ax.errorbar(m_values, E_means, yerr=E_stds, marker="o", capsize=4, linewidth=2, label="$E$ (using $a_n$)")
    ax.errorbar(m_values, E_sig_means, yerr=E_sig_stds, marker="s", capsize=4, linewidth=2,
                color="#ff7f0e", label="$E_{sig}$ (using $a_\\infty^{fit}$)")
    ax.plot(m_values, E_fit_vals, marker="^", linewidth=2,
            color="#d62728", label="$E_{fit}$ ($\\int_1^\\infty$ of mean curve)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("E (excess entropy, bits)")
    ax.set_title("E vs m", fontweight="bold")
    ax.legend(fontsize=8)

    # D vs m
    ax = axes[0, 1]
    ax.errorbar(m_values, D_means, yerr=D_stds, marker="o", capsize=4, linewidth=2,
                color="#2ca02c", label="$D$ (using $a_n$)")
    ax.errorbar(m_values, D_sig_means, yerr=D_sig_stds, marker="s", capsize=4, linewidth=2,
                color="#d62728", label="$D_{sig}$ (using $a_\\infty^{fit}$)")
    ax.plot(m_values, D_fit_vals, marker="^", linewidth=2,
            color="#9467bd", label="$D_{fit}$ ($\\int_1^\\infty$ of mean curve)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("D (diversity score, bits)")
    ax.set_title("D vs m", fontweight="bold")
    ax.legend(fontsize=8)

    # C vs m
    ax = axes[0, 2]
    ax.errorbar(m_values, C_means, yerr=C_stds, marker="d", capsize=4, linewidth=2, color="#9467bd")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("C (coherence)")
    ax.set_title("C vs m (sanity check: should be flat)", fontweight="bold")

    # σ_ℓ vs m
    ax = axes[0, 3]
    ax.errorbar(m_values, sigma_means, yerr=sigma_stds, marker="^", capsize=4, linewidth=2, color="#e377c2")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("$\\sigma_\\ell$ (bits/byte)")
    ax.set_title("$\\sigma_\\ell$ (coherence spread) vs m", fontweight="bold")

    # a_n vs m
    ax = axes[1, 0]
    ax.errorbar(m_values, a_n_means, yerr=a_n_stds, marker="v", capsize=4, linewidth=2, color="#8c564b")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("$a_n$ (last curve point, bits)")
    ax.set_title("$a_n$ vs m", fontweight="bold")

    # Summary table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for i, m in enumerate(m_values):
        table_data.append([
            str(m),
            f"{E_means[i]:.0f}±{E_stds[i]:.0f}",
            f"{E_sig_means[i]:.0f}±{E_sig_stds[i]:.0f}",
            f"{E_fit_vals[i]:.0f}",
            f"{D_means[i]:.0f}±{D_stds[i]:.0f}",
            f"{C_means[i]:.3f}±{C_stds[i]:.3f}",
            f"{sigma_means[i]:.3f}±{sigma_stds[i]:.3f}",
            f"{a_n_means[i]:.0f}±{a_n_stds[i]:.0f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["m", "E", "E_sig", "E_fit", "D", "C", "σ_ℓ", "a_n"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.3, 1.5)
    ax.set_title("Summary", fontweight="bold")

    # Hide unused subplots
    axes[1, 2].set_visible(False)
    axes[1, 3].set_visible(False)

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
            model_name = data.get("base_model", "Model")
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
