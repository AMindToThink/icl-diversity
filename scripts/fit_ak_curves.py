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


def exponential_ak(k: np.ndarray, a_inf: float, alpha: float, beta: float) -> np.ndarray:
    """Exponential decay: a_k = a_inf + alpha * exp(-beta * (k - 1))

    3 parameters (vs sigmoid's 4) — better constrained for monotone decay.
    E_fit has closed form: E_fit = alpha / beta.
    """
    return a_inf + alpha * np.exp(-beta * (k - 1))


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


def bootstrap_fit(
    curves: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Bootstrap confidence intervals for sigmoid fit parameters.

    Resamples draws with replacement, computes mean curve per resample,
    fits sigmoid, and collects parameter distributions.

    Returns dict mapping parameter name to array of bootstrapped values.
    Failed fits are excluded.
    """
    rng = np.random.default_rng(seed)
    n_draws = curves.shape[0]
    k = np.arange(1, curves.shape[1] + 1, dtype=float)

    param_names = ["a_inf", "alpha", "beta", "k0", "E_fit", "r_squared"]
    boot_params: dict[str, list[float]] = {name: [] for name in param_names}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_draws, size=n_draws)
        mean_curve = np.mean(curves[indices], axis=0)
        fit_params, success = fit_sigmoid(k, mean_curve)
        if success:
            for name in param_names:
                boot_params[name].append(fit_params[name])

    return {name: np.array(vals) for name, vals in boot_params.items()}


def fit_mean_curves(data: dict[str, Any], n_bootstrap: int = 1000) -> list[dict[str, Any]]:
    """Fit sigmoid to the mean a_k curve for each mode count m.

    Includes bootstrap confidence intervals for all fit parameters.
    """
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

        # Bootstrap CIs
        boot = bootstrap_fit(curves, n_bootstrap=n_bootstrap)

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
        # Add bootstrap SEM and 95% CI for each parameter
        for name in ["a_inf", "alpha", "beta", "k0", "E_fit"]:
            vals = boot[name]
            if len(vals) > 0:
                entry[f"{name}_boot_sem"] = float(np.std(vals))
                entry[f"{name}_boot_ci_lo"] = float(np.percentile(vals, 2.5))
                entry[f"{name}_boot_ci_hi"] = float(np.percentile(vals, 97.5))
            else:
                entry[f"{name}_boot_sem"] = np.nan
                entry[f"{name}_boot_ci_lo"] = np.nan
                entry[f"{name}_boot_ci_hi"] = np.nan

        results.append(entry)
    return results


def fit_exponential(k: np.ndarray, curve: np.ndarray) -> tuple[dict[str, float], bool]:
    """Fit exponential decay to a single a_k curve.

    Model: a_k = a_inf + alpha * exp(-beta * (k - 1))
    E_fit = alpha / beta (closed-form integral from 1 to infinity).

    Returns (params_dict, success).
    """
    a_inf_guess = curve[-1]
    alpha_guess = max(curve[0] - curve[-1], 0.01)
    beta_guess = 0.5
    p0 = [a_inf_guess, alpha_guess, beta_guess]

    bounds = ([0, 0, 0.001], [np.inf, np.inf, 50])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(exponential_ak, k, curve, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        residuals = curve - exponential_ak(k, *popt)
        rmse = float(np.sqrt(np.mean(residuals**2)))

        a_inf, alpha, beta = popt
        E_fit = float(alpha / beta)

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((curve - np.mean(curve))**2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "a_inf": float(a_inf),
            "alpha": float(alpha),
            "beta": float(beta),
            "E_fit": E_fit,
            "a_inf_se": float(perr[0]),
            "alpha_se": float(perr[1]),
            "beta_se": float(perr[2]),
            "rmse": rmse,
            "r_squared": r_squared,
        }, True
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}, False


def bootstrap_fit_exponential(
    curves: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Bootstrap confidence intervals for exponential fit parameters."""
    rng = np.random.default_rng(seed)
    n_draws = curves.shape[0]
    k = np.arange(1, curves.shape[1] + 1, dtype=float)

    param_names = ["a_inf", "alpha", "beta", "E_fit", "r_squared"]
    boot_params: dict[str, list[float]] = {name: [] for name in param_names}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_draws, size=n_draws)
        mean_curve = np.mean(curves[indices], axis=0)
        fit_params, success = fit_exponential(k, mean_curve)
        if success:
            for name in param_names:
                boot_params[name].append(fit_params[name])

    return {name: np.array(vals) for name, vals in boot_params.items()}


def fit_mean_curves_exponential(data: dict[str, Any], n_bootstrap: int = 1000) -> list[dict[str, Any]]:
    """Fit exponential decay to the mean a_k curve for each mode count m.

    Includes bootstrap confidence intervals for all fit parameters.
    """
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

        fit_params, success = fit_exponential(k, mean_curve)

        boot = bootstrap_fit_exponential(curves, n_bootstrap=n_bootstrap)

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
        for name in ["a_inf", "alpha", "beta", "E_fit"]:
            vals = boot[name]
            if len(vals) > 0:
                entry[f"{name}_boot_sem"] = float(np.std(vals))
                entry[f"{name}_boot_ci_lo"] = float(np.percentile(vals, 2.5))
                entry[f"{name}_boot_ci_hi"] = float(np.percentile(vals, 97.5))
            else:
                entry[f"{name}_boot_sem"] = np.nan
                entry[f"{name}_boot_ci_lo"] = np.nan
                entry[f"{name}_boot_ci_hi"] = np.nan

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

        # Compute mean and SEM across all draws
        curves = np.array([np.array(run["a_k_curve"]) for run in runs])
        mean_curve = np.mean(curves, axis=0)
        sem_curve = np.std(curves, axis=0) / np.sqrt(len(runs))
        k = np.arange(1, len(mean_curve) + 1)

        # Plot mean curve + SEM band
        color = "#1f77b4"
        ax.plot(k, mean_curve, "o-", markersize=5, linewidth=2, color=color, label="mean $a_k$")
        ax.fill_between(k, mean_curve - sem_curve, mean_curve + sem_curve,
                        alpha=0.3, color=color, label="±1 SEM")

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
    k0_sems = [f.get("k0_boot_sem", 0) for f in fits]
    a_inf_vals = [f["a_inf"] for f in fits]
    a_inf_sems = [f.get("a_inf_boot_sem", 0) for f in fits]
    a_n_means = [f["a_n_mean"] for f in fits]
    a_n_sems = [f["a_n_std"] / np.sqrt(f["n_draws"]) for f in fits]
    E_fit_vals = [f["E_fit"] for f in fits]
    E_fit_sems = [f.get("E_fit_boot_sem", 0) for f in fits]
    r2_vals = [f["r_squared"] for f in fits]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # k0 vs m
    ax = axes[0, 0]
    ax.errorbar(m_vals, k0_vals, yerr=k0_sems, marker="o", capsize=4, linewidth=2)
    ax.plot(m_vals, m_vals, "--", color="gray", alpha=0.5, label="k₀ = m (identity)")
    ax.set_xlabel("m (mode count)")
    ax.set_ylabel("k₀ (inflection point)")
    ax.set_title("k₀ vs m (bootstrap SEM)", fontweight="bold")
    ax.legend()

    # a_inf (fit) vs a_n (raw)
    ax = axes[0, 1]
    ax.errorbar(m_vals, a_inf_vals, yerr=a_inf_sems, marker="o", capsize=4,
                linewidth=2, label="$a_\\infty$ (fit ± boot SEM)")
    ax.errorbar(m_vals, a_n_means, yerr=a_n_sems, marker="s", capsize=4,
                linewidth=2, label="$a_n$ (mean ± SEM)")
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
        k0_ci = f"[{f.get('k0_boot_ci_lo', 0):.1f}, {f.get('k0_boot_ci_hi', 0):.1f}]"
        table_data.append([
            str(f["m"]),
            f"{f['k0']:.2f}",
            k0_ci,
            f"{f['E_fit']:.1f}±{f.get('E_fit_boot_sem', 0):.1f}",
            f"{f['r_squared']:.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["m", "k₀", "k₀ 95% CI", "E_fit", "R²"],
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
    print(f"\n{'m':>4} {'draws':>6} {'k0':>8} {'k0_CI':>18} {'E_fit':>8} {'E_fit_CI':>18} {'R2':>8}")
    print("-" * 80)
    for r in fit_results:
        if r.get("fit_success"):
            k0_ci = f"[{r.get('k0_boot_ci_lo', 0):.1f}, {r.get('k0_boot_ci_hi', 0):.1f}]"
            ef_ci = f"[{r.get('E_fit_boot_ci_lo', 0):.0f}, {r.get('E_fit_boot_ci_hi', 0):.0f}]"
            print(f"{r['m']:>4} {r['n_draws']:>6} {r['k0']:>8.2f} {k0_ci:>18} "
                  f"{r['E_fit']:>8.1f} {ef_ci:>18} {r['r_squared']:>8.3f}")
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
