"""Check whether C × a_∞ tracks mode count in the Regime 1 experiments.

Loads the existing mode_count JSON results (GPT-2 and Qwen2.5-3B, 1k draws)
and computes Spearman ρ(metric, mode_count) for various metric candidates.

Usage:
    uv run python scripts/check_ainf_vs_mode_count.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "mode_count"


def load_fits(path: Path) -> dict[int, dict]:
    """Load *_fits.json keyed by mode count m."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(entry["m"]): entry for entry in data}


def analyze_mode_count_file(path: Path) -> None:
    """Load mode count results and print Spearman correlations."""
    with open(path) as f:
        raw = json.load(f)

    # Handle both formats: list of entries or dict with 'runs' key
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict) and "runs" in raw:
        entries = raw["runs"]
    else:
        raise ValueError(f"Unexpected JSON format in {path}")

    # Load fits file for E_fit and a_inf_fit (from mean-curve exponential fit)
    fits_path = path.with_name(path.name.replace("_1k_draws.json", "_fits.json"))
    fits = load_fits(fits_path)

    print(f"\n{'=' * 70}")
    print(f"  {path.name}")
    if fits:
        print(f"  (with fits from {fits_path.name})")
    print(f"{'=' * 70}")

    mode_counts: list[float] = []
    metrics: dict[str, list[float]] = {
        "a_n (last point)": [],
        "a_1 (unconditional)": [],
        "C": [],
        "E_disc": [],
        "C × a_n": [],
        "D_disc = C × E_disc": [],
    }

    for entry in entries:
        m = entry.get("m", entry.get("mode_count"))
        curve = entry.get("a_k_curve", entry.get("mean_curve"))

        if m is None or curve is None:
            continue

        # Flat structure (mode_count JSONs) or nested metrics dict
        C = entry.get("coherence_C")
        E = entry.get("excess_entropy_E")
        if C is None or E is None:
            m_dict = entry.get("metrics", {})
            C = C or m_dict.get("coherence_C")
            E = E or m_dict.get("excess_entropy_E")
        if C is None or E is None:
            continue

        mode_counts.append(float(m))
        metrics["a_n (last point)"].append(curve[-1])
        metrics["a_1 (unconditional)"].append(curve[0])
        metrics["C"].append(C)
        metrics["E_disc"].append(E)
        metrics["C × a_n"].append(C * curve[-1])
        metrics["D_disc = C × E_disc"].append(C * E)

    # Add fit-based metrics from the fits file (one value per m)
    if fits:
        # For per-run metrics, look up the fit for that run's m
        e_fit_vals: list[float] = []
        a_inf_fit_vals: list[float] = []
        c_a_inf_fit_vals: list[float] = []
        d_fit_vals: list[float] = []
        for i, m in enumerate(mode_counts):
            fit = fits.get(int(m))
            if fit and fit.get("fit_success"):
                e_fit = fit.get("E_fit", fit["alpha"] / fit["beta"])
                e_fit_vals.append(e_fit)
                a_inf_fit_vals.append(fit["a_inf"])
                c_a_inf_fit_vals.append(metrics["C"][i] * fit["a_inf"])
                d_fit_vals.append(metrics["C"][i] * e_fit)
            else:
                e_fit_vals.append(float("nan"))
                a_inf_fit_vals.append(float("nan"))
                c_a_inf_fit_vals.append(float("nan"))
                d_fit_vals.append(float("nan"))
        metrics["E_fit"] = e_fit_vals
        metrics["a_inf_fit"] = a_inf_fit_vals
        metrics["C × a_inf_fit"] = c_a_inf_fit_vals
        metrics["D_fit = C × E_fit"] = d_fit_vals

    mc = np.array(mode_counts)
    unique_m = sorted(set(mode_counts))
    print(f"  n = {len(mc)}, mode counts: {unique_m}")
    print(f"\n  {'Metric':<25s} {'ρ(metric, m)':>15s} {'p-value':>12s}")
    print(f"  {'-'*25} {'-'*15} {'-'*12}")

    for name, vals in metrics.items():
        rho, p = spearmanr(mc, vals)
        print(f"  {name:<25s} {rho:>+15.4f} {p:>12.2e}")

    # Also print mean values per mode count
    print(f"\n  {'m':>3s} {'C×a_n':>10s} {'D=C×E':>10s} {'a_n':>10s} {'E':>10s} {'C':>8s}")
    print(f"  {'---':>3s} {'----------':>10s} {'----------':>10s} {'----------':>10s} {'----------':>10s} {'--------':>8s}")
    for m in unique_m:
        idx = [i for i, v in enumerate(mode_counts) if v == m]
        ca = np.mean([metrics["C × a_n"][i] for i in idx])
        d = np.mean([metrics["D = C × E"][i] for i in idx])
        an = np.mean([metrics["a_n (last point)"][i] for i in idx])
        e = np.mean([metrics["E"][i] for i in idx])
        c = np.mean([metrics["C"][i] for i in idx])
        print(f"  {int(m):>3d} {ca:>10.3f} {d:>10.3f} {an:>10.3f} {e:>10.3f} {c:>8.4f}")


def main() -> None:
    for name in ["gpt2_1k_draws.json", "qwen2.5-3b_1k_draws.json"]:
        path = RESULTS_DIR / name
        if path.exists():
            analyze_mode_count_file(path)
        else:
            print(f"Not found: {path}")


if __name__ == "__main__":
    main()
