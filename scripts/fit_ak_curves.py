"""Fit sigmoid model to a_k curves and extract parameters.

Fits a_k = a_inf + alpha / (1 + exp(beta * (k - k0)))
to each a_k curve from the mode count experiment.

This parameterization:
- a_inf: asymptotic surprise (floor)
- alpha: total drop from initial to floor
- beta: steepness of transition
- k0: inflection point (approximate mode count diagnostic)

E_fit = ∫₁^∞ (a_k - a_∞) dk, computed by numerically integrating the fitted
sigmoid. This captures tail structure beyond k=n that the discrete sum misses,
and smooths over noise in individual a_k values.

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
from scipy.integrate import quad
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

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "results" / "mode_count" / "gpt2_1k_draws.json"
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

        a_inf, alpha, beta, k0 = popt

        # E_fit = ∫₁^∞ (a_k - a_∞) dk via numerical integration of the fitted curve
        E_fit, _ = quad(lambda k: sigmoid_ak(np.asarray(k), *popt) - a_inf, 1, np.inf)

        return {
            "a_inf": float(a_inf),
            "alpha": float(alpha),
            "beta": float(beta),
            "k0": float(k0),
            "E_fit": E_fit,
            "a_inf_se": float(perr[0]),
            "alpha_se": float(perr[1]),
            "beta_se": float(perr[2]),
            "k0_se": float(perr[3]),
            "rmse": rmse,
            "r_squared": float(1 - np.sum(residuals**2) / np.sum((curve - np.mean(curve))**2)),
        }, True
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}, False


def fit_mean_curves(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Fit sigmoid to the mean a_k curve for each mode count m."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in data["runs"]:
        grouped[run["m"]].append(run)

    results = []
    for m in sorted(grouped.keys()):
        runs = grouped[m]
        curves = np.array([r["a_k_curve"] for r in runs])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        k = np.arange(1, len(mean_curve) + 1, dtype=float)

        fit_params, success = fit_sigmoid(k, mean_curve)

        entry = {
            "m": m,
            "n_draws": len(runs),
            "n_responses": runs[0]["n_responses"],
            "a_n_mean": float(mean_curve[-1]),
            "a_n_std": float(std_curve[-1]),
            "a_1_mean": float(mean_curve[0]),
            "fit_success": success,
            **fit_params,
        }
        results.append(entry)
    return results


def plot_fits(data: dict[str, Any], fit_results: list[dict], figures_dir: Path) -> None:
    """Plot mean a_k curve + sigmoid fit + ±1 SD band per m panel."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in data["runs"]:
        grouped[run["m"]].append(run)
    fit_by_m = {r["m"]: r for r in fit_results}

    m_values = sorted(grouped.keys())
    ncols = min(3, len(m_values))
    nrows = (len(m_values) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, m in enumerate(m_values):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        runs = grouped[m]

        # Compute mean and SD across all draws
        curves = np.array([np.array(run["a_k_curve"]) for run in runs])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        k = np.arange(1, len(mean_curve) + 1)

        # Plot mean curve + SD band
        color = "#1f77b4"
        ax.plot(k, mean_curve, "o-", markersize=5, linewidth=2, color=color, label="mean $a_k$")
        ax.fill_between(k, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color=color, label="±1 SD")

        # Overlay sigmoid fit from pre-computed results
        fit = fit_by_m.get(m)
        if fit and fit.get("fit_success"):
            k_fine = np.linspace(0.5, len(mean_curve) + 0.5, 100)
            fitted = sigmoid_ak(k_fine, fit["a_inf"], fit["alpha"],
                                fit["beta"], fit["k0"])
            ax.plot(k_fine, fitted, "--", linewidth=2, color="#ff7f0e",
                    label=f"sigmoid: k₀={fit['k0']:.1f}, R²={fit['r_squared']:.3f}")

        ax.set_title(f"m = {m} (n={len(runs)} draws)", fontweight="bold")
        ax.set_xlabel("k")
        ax.set_ylabel("$a_k$ (bits)")
        ax.legend(fontsize=8, loc="best")

    for idx in range(len(m_values), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    model_name = data.get("base_model", "unknown")
    fig.suptitle(f"Sigmoid Fits to Mean a_k Curves — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = figures_dir / "sigmoid_fits.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fit_params_vs_m(fit_results: list[dict], figures_dir: Path, model_name: str) -> None:
    """Plot fitted parameters vs m."""
    fits = [r for r in fit_results if r.get("fit_success")]

    m_vals = [f["m"] for f in fits]
    k0_vals = [f["k0"] for f in fits]
    a_inf_vals = [f["a_inf"] for f in fits]
    a_n_means = [f["a_n_mean"] for f in fits]
    a_n_stds = [f["a_n_std"] for f in fits]
    r2_vals = [f["r_squared"] for f in fits]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # k0 vs m
    ax = axes[0, 0]
    ax.plot(m_vals, k0_vals, marker="o", linewidth=2)
    ax.plot(m_vals, m_vals, "--", color="gray", alpha=0.5, label="k₀ = m (identity)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("k₀ (inflection point)")
    ax.set_title("k₀ vs m", fontweight="bold")
    ax.legend()

    # a_inf (fit) vs a_n (raw)
    ax = axes[0, 1]
    ax.plot(m_vals, a_inf_vals, marker="o", linewidth=2, label="$a_\\infty$ (fit)")
    ax.errorbar(m_vals, a_n_means, yerr=a_n_stds, marker="s", capsize=4,
                linewidth=2, label="$a_n$ (mean ± SD)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("bits")
    ax.set_title("$a_\\infty$ fit vs $a_n$ raw", fontweight="bold")
    ax.legend()

    # R² vs m
    ax = axes[1, 0]
    ax.bar(m_vals, r2_vals, color="#2ca02c", alpha=0.7)
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("R²")
    ax.set_title("Sigmoid fit quality", fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Parameter table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for f in fits:
        table_data.append([
            str(f["m"]),
            f"{f['k0']:.2f}",
            f"{f['a_inf']:.3f}",
            f"{f['alpha']:.2f}",
            f"{f['beta']:.2f}",
            f"{f['E_fit']:.1f}",
            f"{f['r_squared']:.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["m", "k₀", "a_∞", "α", "β", "E_fit", "R²"],
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

    print("Fitting sigmoid to mean a_k curves (one per m)...")
    fit_results = fit_mean_curves(data)

    n_success = sum(1 for r in fit_results if r.get("fit_success"))
    print(f"  {n_success}/{len(fit_results)} fits successful")

    # Print summary table
    print(f"\n{'m':>4} {'draws':>6} {'k0':>8} {'a_inf':>8} {'a_n':>8} {'alpha':>8} {'beta':>8} {'E_fit':>8} {'R2':>8}")
    print("-" * 72)
    for r in fit_results:
        if r.get("fit_success"):
            print(f"{r['m']:>4} {r['n_draws']:>6} {r['k0']:>8.2f} {r['a_inf']:>8.3f} "
                  f"{r['a_n_mean']:>8.3f} {r['alpha']:>8.2f} {r['beta']:>8.2f} "
                  f"{r['E_fit']:>8.1f} {r['r_squared']:>8.3f}")
        else:
            print(f"{r['m']:>4} {r['n_draws']:>6}  FAILED: {r.get('error', 'unknown')}")

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
