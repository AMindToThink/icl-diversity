"""Fit sigmoid model to a_k curves and extract parameters.

Fits a_k = a_inf + alpha / (1 + exp(beta * (k - k0)))
to each a_k curve from the mode count experiment.

This parameterization:
- a_inf: asymptotic surprise (floor)
- alpha: total drop from initial to floor
- beta: steepness of transition
- k0: inflection point (approximate mode count diagnostic)

When k0 < 1, the sigmoid degenerates to exponential decay.

Usage:
    uv run python scripts/fit_ak_curves.py
    uv run python scripts/fit_ak_curves.py --input results/mode_count/gpt2.json
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    }
)

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "results" / "mode_count" / "gpt2.json"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "mode_count"


def sigmoid_ak(k: np.ndarray, a_inf: float, alpha: float, beta: float, k0: float) -> np.ndarray:
    """Sigmoid model: a_k = a_inf + alpha / (1 + exp(beta * (k - k0)))"""
    return a_inf + alpha / (1.0 + np.exp(beta * (k - k0)))


def fit_sigmoid(k: np.ndarray, curve: np.ndarray) -> tuple[dict[str, float], bool]:
    """Fit sigmoid to a single a_k curve.

    Returns (params_dict, success).
    """
    # Initial guesses
    a_inf_guess = curve[-1]
    alpha_guess = curve[0] - curve[-1]
    beta_guess = 1.0
    k0_guess = len(curve) / 2.0
    p0 = [a_inf_guess, max(alpha_guess, 0.01), beta_guess, k0_guess]

    # Bounds: a_inf >= 0, alpha > 0, beta > 0, k0 can be anything
    bounds = ([0, 0, 0.01, -10], [np.inf, np.inf, 50, len(curve) * 3])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(sigmoid_ak, k, curve, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        residuals = curve - sigmoid_ak(k, *popt)
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return {
            "a_inf": float(popt[0]),
            "alpha": float(popt[1]),
            "beta": float(popt[2]),
            "k0": float(popt[3]),
            "a_inf_se": float(perr[0]),
            "alpha_se": float(perr[1]),
            "beta_se": float(perr[2]),
            "k0_se": float(perr[3]),
            "rmse": rmse,
            "r_squared": float(1 - np.sum(residuals**2) / np.sum((curve - np.mean(curve))**2)),
        }, True
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}, False


def fit_all_runs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Fit sigmoid to every run in the dataset."""
    results = []
    for run in data["runs"]:
        curve_key = "a_k_curve_per_byte" if "a_k_curve_per_byte" in run else "a_k_curve"
        curve = np.array(run[curve_key])
        k = np.arange(1, len(curve) + 1, dtype=float)

        fit_params, success = fit_sigmoid(k, curve)

        entry = {
            "m": run["m"],
            "seed": run["seed"],
            "n_responses": run["n_responses"],
            "a_n_raw": float(curve[-1]),
            "a_1_raw": float(curve[0]),
            "fit_success": success,
            **fit_params,
        }
        results.append(entry)
    return results


def plot_fits(data: dict[str, Any], fit_results: list[dict], figures_dir: Path) -> None:
    """Plot observed a_k curves with sigmoid fits overlaid."""
    grouped: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for run, fit in zip(data["runs"], fit_results):
        grouped[run["m"]].append((run, fit))

    m_values = sorted(grouped.keys())
    ncols = min(3, len(m_values))
    nrows = (len(m_values) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, m in enumerate(m_values):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        pairs = grouped[m]

        for run, fit in pairs:
            curve_key = "a_k_curve_per_byte" if "a_k_curve_per_byte" in run else "a_k_curve"
            curve = np.array(run[curve_key])
            k = np.arange(1, len(curve) + 1)

            ax.plot(k, curve, "o", markersize=4, alpha=0.6, label=f"seed={run['seed']}")

            if fit.get("fit_success"):
                k_fine = np.linspace(0.5, len(curve) + 0.5, 100)
                fitted = sigmoid_ak(k_fine, fit["a_inf"], fit["alpha"], fit["beta"], fit["k0"])
                ax.plot(k_fine, fitted, "--", linewidth=1.5, alpha=0.7,
                        label=f"fit: k₀={fit['k0']:.1f}, R²={fit['r_squared']:.3f}")

        ax.set_title(f"m = {m}", fontweight="bold")
        ax.set_xlabel("k")
        ax.set_ylabel("$a_k$ (bits/byte)")
        ax.legend(fontsize=7, loc="best")

    for idx in range(len(m_values), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    model_name = data.get("base_model", "unknown")
    fig.suptitle(f"Sigmoid Fits to a_k Curves — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = figures_dir / "sigmoid_fits.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fit_params_vs_m(fit_results: list[dict], figures_dir: Path, model_name: str) -> None:
    """Plot fitted parameters vs m."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in fit_results:
        if r.get("fit_success"):
            grouped[r["m"]].append(r)

    m_vals, k0_means, k0_stds = [], [], []
    a_inf_fit_means, a_inf_fit_stds = [], []
    a_n_raw_means, a_n_raw_stds = [], []
    r2_means = []

    for m in sorted(grouped.keys()):
        fits = grouped[m]
        m_vals.append(m)
        k0s = [f["k0"] for f in fits]
        k0_means.append(np.mean(k0s))
        k0_stds.append(np.std(k0s))
        a_infs = [f["a_inf"] for f in fits]
        a_inf_fit_means.append(np.mean(a_infs))
        a_inf_fit_stds.append(np.std(a_infs))
        a_ns = [f["a_n_raw"] for f in fits]
        a_n_raw_means.append(np.mean(a_ns))
        a_n_raw_stds.append(np.std(a_ns))
        r2s = [f["r_squared"] for f in fits]
        r2_means.append(np.mean(r2s))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # k0 vs m
    ax = axes[0, 0]
    ax.errorbar(m_vals, k0_means, yerr=k0_stds, marker="o", capsize=4, linewidth=2)
    ax.plot(m_vals, m_vals, "--", color="gray", alpha=0.5, label="k₀ = m (identity)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("k₀ (inflection point)")
    ax.set_title("k₀ vs m", fontweight="bold")
    ax.legend()

    # a_inf (fit) vs a_n (raw)
    ax = axes[0, 1]
    ax.errorbar(m_vals, a_inf_fit_means, yerr=a_inf_fit_stds, marker="o", capsize=4,
                linewidth=2, label="$a_\\infty$ (fit)")
    ax.errorbar(m_vals, a_n_raw_means, yerr=a_n_raw_stds, marker="s", capsize=4,
                linewidth=2, label="$a_n$ (raw last point)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("bits/byte")
    ax.set_title("$a_\\infty$ fit vs $a_n$ raw", fontweight="bold")
    ax.legend()

    # R² vs m
    ax = axes[1, 0]
    ax.bar(m_vals, r2_means, color="#2ca02c", alpha=0.7)
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("R²")
    ax.set_title("Sigmoid fit quality", fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Parameter table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for i, m in enumerate(m_vals):
        fits = grouped[m]
        alpha_mean = np.mean([f["alpha"] for f in fits])
        beta_mean = np.mean([f["beta"] for f in fits])
        table_data.append([
            str(m),
            f"{k0_means[i]:.2f}",
            f"{a_inf_fit_means[i]:.3f}",
            f"{alpha_mean:.2f}",
            f"{beta_mean:.2f}",
            f"{r2_means[i]:.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["m", "k₀", "a_∞", "α", "β", "R²"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Fit Parameters", fontweight="bold")

    fig.suptitle(f"Sigmoid Fit Parameters vs Mode Count — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = figures_dir / "fit_params_vs_m.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit sigmoid to a_k curves from mode count experiment")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSON path")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR, help="Output figures directory")
    parser.add_argument("--save-fits", type=Path, default=None, help="Save fit results to JSON")
    args = parser.parse_args()

    figures_dir = args.output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.input}")
    data = load_data(args.input)
    model_name = data.get("base_model", "unknown")

    print("Fitting sigmoid to all a_k curves...")
    fit_results = fit_all_runs(data)

    n_success = sum(1 for r in fit_results if r.get("fit_success"))
    print(f"  {n_success}/{len(fit_results)} fits successful")

    # Print summary table
    print(f"\n{'m':>4} {'seed':>6} {'k0':>8} {'a_inf':>8} {'a_n':>8} {'alpha':>8} {'beta':>8} {'R2':>8}")
    print("-" * 60)
    for r in fit_results:
        if r.get("fit_success"):
            print(f"{r['m']:>4} {r['seed']:>6} {r['k0']:>8.2f} {r['a_inf']:>8.3f} "
                  f"{r['a_n_raw']:>8.3f} {r['alpha']:>8.2f} {r['beta']:>8.2f} {r['r_squared']:>8.3f}")
        else:
            print(f"{r['m']:>4} {r['seed']:>6}  FAILED: {r.get('error', 'unknown')}")

    plot_fits(data, fit_results, figures_dir)
    plot_fit_params_vs_m(fit_results, figures_dir, model_name)

    if args.save_fits:
        args.save_fits.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_fits, "w") as f:
            json.dump(fit_results, f, indent=2)
        print(f"\nFit results saved to: {args.save_fits}")

    print("Done!")


def load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
