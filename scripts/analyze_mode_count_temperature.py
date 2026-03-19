"""Analyze mode count x temperature experiment results.

Reads T_*.json from results/mode_count_temperature/ and tests four hypotheses:

H1' — Within-m Variance Reduction
  std(D) across draws decreases at T > 1 for each mode count m.
  Test: paired Wilcoxon signed-rank across 10 mode counts.

H2' — Mode Count Ordering Preservation
  mean_D(m) ordering is preserved across temperatures.
  Test: Spearman rho on 10 mode-count means vs T=1.0.

H3' — Discriminability (Signal-to-Noise)
  Cohen's d between adjacent mode counts increases at T > 1.
  Test: compare median Cohen's d at T vs T=1.0.

H4' — Sample Efficiency
  T > 1 needs fewer draws to correctly rank all 10 mode counts.
  Test: subsample draws, compute Spearman rho vs ground truth,
  Mann-Whitney U on bootstrap rho distributions.

Usage:
    uv run python scripts/analyze_mode_count_temperature.py
    uv run python scripts/analyze_mode_count_temperature.py --input-dir results/mode_count_temperature/ --output-dir figures/mode_count_temperature/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Import sigmoid fitting from the existing fit_ak_curves script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ak_curves import fit_mean_curves

DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parent.parent / "results" / "mode_count_temperature"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "figures" / "mode_count_temperature"
)


def load_all_results(input_dir: Path) -> dict[float, dict[str, Any]]:
    """Load all T_*.json files, keyed by temperature."""
    results: dict[float, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("T_*.json")):
        with open(path) as f:
            data = json.load(f)
        temp = float(data["temperature"])
        results[temp] = data
    return results


def compute_D_fit_per_m(data: dict[str, Any], n_bootstrap: int = 200) -> dict[int, float]:
    """Compute D_fit = C_mean * E_fit per mode count m using fit_mean_curves.

    Delegates to fit_mean_curves from fit_ak_curves.py, which averages a_k
    curves across draws per m, fits one sigmoid, and integrates 1..inf.
    """
    # fit_mean_curves expects each run to have "n_responses"; patch if missing
    n_responses = data.get("n_responses")
    if n_responses is not None:
        for run in data["runs"]:
            if "n_responses" not in run:
                run["n_responses"] = n_responses

    fit_results = fit_mean_curves(data, n_bootstrap=n_bootstrap)

    # Also compute mean C per m
    grouped: dict[int, list[float]] = defaultdict(list)
    for run in data["runs"]:
        grouped[int(run["m"])].append(float(run["coherence_C"]))

    d_fit_by_m: dict[int, float] = {}
    for fit in fit_results:
        m = fit["m"]
        if not fit.get("fit_success"):
            print(f"    WARNING: sigmoid fit failed for m={m}")
            continue
        mean_C = float(np.mean(grouped[m]))
        d_fit_by_m[m] = mean_C * fit["E_fit"]

    return d_fit_by_m


def group_D_raw_by_m(data: dict[str, Any]) -> dict[int, list[float]]:
    """Extract raw per-draw D values grouped by mode count m.

    Uses the truncated E = sum(a_k - a_n) from compute_icl_diversity_metrics.
    Used for variance/std calculations where we need per-draw distributions.
    """
    by_m: dict[int, list[float]] = defaultdict(list)
    for run in data["runs"]:
        by_m[int(run["m"])].append(float(run["diversity_score_D"]))
    return dict(by_m)


def h1_variance_reduction(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H1': Within-m variance reduction at T > 1."""
    temperatures = sorted(all_results.keys())
    if 1.0 not in all_results:
        print("  WARNING: T=1.0 not in results, skipping H1'")
        return

    ref_by_m = group_D_raw_by_m(all_results[1.0])
    mode_counts = sorted(ref_by_m.keys())
    ref_stds = [float(np.std(ref_by_m[m])) for m in mode_counts]

    # Plot: std(D) vs m, one line per temperature
    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in temperatures:
        by_m = group_D_raw_by_m(all_results[temp])
        stds = [float(np.std(by_m[m])) for m in mode_counts]
        ax.plot(mode_counts, stds, "o-", label=f"T={temp}", markersize=4)

    ax.set_xlabel("Number of modes m")
    ax.set_ylabel("std(D) across draws")
    ax.set_title("H1': Within-Mode-Count Variance vs Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "h1_variance_by_mode_count.png", dpi=150)
    plt.close(fig)
    print("  Saved h1_variance_by_mode_count.png")

    # Formal test: paired Wilcoxon across mode counts
    print("\n  H1' formal tests (paired Wilcoxon, one-sided: std(D, T) < std(D, 1.0))")
    print(f"  {'T':>5} | {'mean std':>9} | {'ref std':>9} | {'W stat':>8} | {'p-value':>8} | reject?")
    print(f"  {'---':>5}-+-{'---':>9}-+-{'---':>9}-+-{'---':>8}-+-{'---':>8}-+--------")

    h1_results: list[dict[str, Any]] = []
    for temp in sorted(t for t in temperatures if t > 1.0):
        by_m = group_D_raw_by_m(all_results[temp])
        test_stds = [float(np.std(by_m[m])) for m in mode_counts]
        diffs = [r - t for r, t in zip(ref_stds, test_stds)]
        if all(d == 0.0 for d in diffs):
            print(f"  {temp:5.1f} | {'N/A':>9} | {'N/A':>9} | {'N/A':>8} | {'N/A':>8} | no diffs")
            continue
        w_result = stats.wilcoxon(ref_stds, test_stds, alternative="greater")
        w_stat = float(w_result.statistic)
        p_val = float(w_result.pvalue)
        reject = p_val < 0.05
        print(
            f"  {temp:5.1f} | {np.mean(test_stds):9.2f} | {np.mean(ref_stds):9.2f} | "
            f"{w_stat:8.1f} | {p_val:8.4f} | {'YES' if reject else 'no'}"
        )
        h1_results.append({
            "temperature": temp,
            "mean_std_D": float(np.mean(test_stds)),
            "ref_mean_std_D": float(np.mean(ref_stds)),
            "wilcoxon_W": w_stat,
            "p_value": p_val,
            "reject_H0_alpha_0.05": reject,
        })

    with open(output_dir / "h1_variance_reduction.json", "w") as f:
        json.dump(h1_results, f, indent=2)


def h2_ordering_preservation(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H2': Mode count ordering preserved across temperatures.

    Uses sigmoid-fitted D_fit (via fit_mean_curves) for mean values,
    so the ordering reflects the properly integrated E rather than the
    truncated finite-sample E = sum(a_k - a_n).
    """
    temperatures = sorted(all_results.keys())
    if 1.0 not in all_results:
        print("  WARNING: T=1.0 not in results, skipping H2'")
        return

    print("  Fitting sigmoids for D_fit per (temperature, m)...")
    d_fit_all: dict[float, dict[int, float]] = {}
    for temp in temperatures:
        d_fit_all[temp] = compute_D_fit_per_m(all_results[temp])

    ref_d_fit = d_fit_all[1.0]
    mode_counts = sorted(ref_d_fit.keys())
    ref_means = [ref_d_fit[m] for m in mode_counts]

    # Plot: D_fit vs m at each temperature
    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in temperatures:
        d_fit = d_fit_all[temp]
        means = [d_fit.get(m, 0.0) for m in mode_counts]
        ax.plot(mode_counts, means, "o-", label=f"T={temp}", markersize=5)

    ax.set_xlabel("Number of modes m")
    ax.set_ylabel("$D_{fit}$ = C * E_fit (sigmoid-integrated)")
    ax.set_title("H2': Mode Count vs Diversity at Each Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "h2_mean_D_vs_m.png", dpi=150)
    plt.close(fig)
    print("  Saved h2_mean_D_vs_m.png")

    # Formal test: Spearman rho at each T vs T=1.0
    print("\n  H2' Spearman rho of D_fit(m) at T vs T=1.0 (10 mode counts)")
    print(f"  {'T':>5} | {'rho':>6} | {'p-value':>8} | reject?")
    print(f"  {'---':>5}-+-{'---':>6}-+-{'---':>8}-+--------")

    h2_results: list[dict[str, Any]] = []
    for temp in temperatures:
        d_fit = d_fit_all[temp]
        means = [d_fit.get(m, 0.0) for m in mode_counts]
        rho_result = stats.spearmanr(ref_means, means)
        rho = float(rho_result.statistic)
        p_val = float(rho_result.pvalue)
        reject = p_val < 0.05 and rho > 0
        print(f"  {temp:5.1f} | {rho:6.3f} | {p_val:8.4f} | {'YES' if reject else 'no'}")
        h2_results.append({
            "temperature": temp, "spearman_rho": rho,
            "p_value": p_val, "reject_H0_alpha_0.05": reject,
        })

    with open(output_dir / "h2_ordering_preservation.json", "w") as f:
        json.dump(h2_results, f, indent=2)


def h3_discriminability(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H3': Cohen's d between adjacent mode counts at each temperature."""
    temperatures = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    print("\n  H3' median Cohen's d between adjacent mode counts")
    print(f"  {'T':>5} | {'median d':>9} | adjacent d values")
    print(f"  {'---':>5}-+-{'---':>9}-+-------------------")

    h3_results: list[dict[str, Any]] = []
    for temp in temperatures:
        by_m = group_D_raw_by_m(all_results[temp])
        mode_counts = sorted(by_m.keys())

        cohens_ds: list[float] = []
        pairs: list[str] = []
        for i in range(len(mode_counts) - 1):
            m1, m2 = mode_counts[i], mode_counts[i + 1]
            d1 = np.array(by_m[m1])
            d2 = np.array(by_m[m2])
            pooled_std = float(np.sqrt((np.var(d1, ddof=1) + np.var(d2, ddof=1)) / 2))
            if pooled_std > 0:
                d = float((np.mean(d2) - np.mean(d1)) / pooled_std)
            else:
                d = 0.0
            cohens_ds.append(d)
            pairs.append(f"{m1}-{m2}")

        ax.plot(
            [f"{mode_counts[i]}-{mode_counts[i+1]}" for i in range(len(mode_counts) - 1)],
            cohens_ds, "o-", label=f"T={temp}", markersize=4,
        )

        median_d = float(np.median(cohens_ds))
        d_str = ", ".join(f"{d:.2f}" for d in cohens_ds)
        print(f"  {temp:5.1f} | {median_d:9.3f} | {d_str}")
        h3_results.append({
            "temperature": temp,
            "median_cohens_d": median_d,
            "cohens_d_per_pair": dict(zip(pairs, cohens_ds)),
        })

    ax.set_xlabel("Adjacent mode count pair")
    ax.set_ylabel("Cohen's d")
    ax.set_title("H3': Discriminability Between Adjacent Mode Counts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="d=0.8 (large)")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "h3_cohens_d.png", dpi=150)
    plt.close(fig)
    print("  Saved h3_cohens_d.png")

    with open(output_dir / "h3_discriminability.json", "w") as f:
        json.dump(h3_results, f, indent=2)


def h4_sample_efficiency(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H4': T > 1 needs fewer draws to rank mode counts correctly."""
    if 1.0 not in all_results:
        print("  WARNING: T=1.0 not in results, skipping H4'")
        return

    ref_by_m = group_D_raw_by_m(all_results[1.0])
    mode_counts = sorted(ref_by_m.keys())
    ref_means = np.array([float(np.mean(ref_by_m[m])) for m in mode_counts])

    subsample_sizes = [5, 10, 20, 50, 100]
    test_temperatures = [t for t in sorted(all_results.keys()) if t >= 1.0]
    n_bootstrap = 200
    rng = np.random.RandomState(42)

    rho_distributions: dict[float, dict[int, list[float]]] = {}

    fig, ax = plt.subplots(figsize=(8, 5))

    for temp in test_temperatures:
        by_m = group_D_raw_by_m(all_results[temp])
        rho_distributions[temp] = {}

        mean_rhos: list[float] = []
        std_rhos: list[float] = []

        for n_sub in subsample_sizes:
            rhos: list[float] = []
            for _ in range(n_bootstrap):
                sub_means: list[float] = []
                for m in mode_counts:
                    ds = by_m[m]
                    n_avail = len(ds)
                    if n_avail <= n_sub:
                        sub_means.append(float(np.mean(ds)))
                    else:
                        idx = rng.choice(n_avail, size=n_sub, replace=False)
                        sub_means.append(float(np.mean([ds[i] for i in idx])))
                rho = float(stats.spearmanr(ref_means, sub_means).statistic)
                rhos.append(rho)

            rho_distributions[temp][n_sub] = rhos
            mean_rhos.append(float(np.mean(rhos)))
            std_rhos.append(float(np.std(rhos)))

        ax.errorbar(
            subsample_sizes, mean_rhos, yerr=std_rhos,
            fmt="o-", label=f"T={temp}", capsize=3,
        )

    ax.set_xlabel("Number of draws (subsampled)")
    ax.set_ylabel("Spearman rho vs ground truth (T=1.0, all draws)")
    ax.set_title("H4': Sample Efficiency (10 mode counts)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "h4_sample_efficiency.png", dpi=150)
    plt.close(fig)
    print("  Saved h4_sample_efficiency.png")

    # Delta plot
    if 1.0 in rho_distributions:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for temp in [t for t in test_temperatures if t != 1.0]:
            if temp not in rho_distributions:
                continue
            mean_deltas: list[float] = []
            ci_lo: list[float] = []
            ci_hi: list[float] = []
            for n_sub in subsample_sizes:
                rhos_t1 = np.array(rho_distributions[1.0][n_sub])
                rhos_alt = np.array(rho_distributions[temp][n_sub])
                deltas = rhos_alt - rhos_t1
                mean_deltas.append(float(np.mean(deltas)))
                ci_lo.append(float(np.percentile(deltas, 2.5)))
                ci_hi.append(float(np.percentile(deltas, 97.5)))
            ax2.plot(subsample_sizes, mean_deltas, "o-", label=f"T={temp} - T=1.0")
            ax2.fill_between(subsample_sizes, ci_lo, ci_hi, alpha=0.2)

        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Number of draws (subsampled)")
        ax2.set_ylabel("Delta Spearman rho (T>1 minus T=1.0)")
        ax2.set_title("H4': Ranking Advantage of Higher Temperature (95% CI)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(output_dir / "h4_sample_efficiency_delta.png", dpi=150)
        plt.close(fig2)
        print("  Saved h4_sample_efficiency_delta.png")

    # Formal test
    for t_alt in [1.5, 2.0, 3.0]:
        if 1.0 not in rho_distributions or t_alt not in rho_distributions:
            continue
        print(f"\n  H4' formal: T={t_alt} vs T=1.0 (Mann-Whitney U, one-sided)")
        print(f"  {'n_draw':>6} | {'mean rho T=1':>12} | {f'mean rho T={t_alt}':>12} | {'U':>8} | {'p-value':>8} | reject?")
        print(f"  {'---':>6}-+-{'---':>12}-+-{'---':>12}-+-{'---':>8}-+-{'---':>8}-+--------")

        h4_results: list[dict[str, Any]] = []
        for n_sub in subsample_sizes:
            rhos_t1 = rho_distributions[1.0][n_sub]
            rhos_alt = rho_distributions[t_alt][n_sub]
            u_result = stats.mannwhitneyu(rhos_alt, rhos_t1, alternative="greater")
            u_stat = float(u_result.statistic)
            p_val = float(u_result.pvalue)
            reject = p_val < 0.05
            print(
                f"  {n_sub:6d} | {np.mean(rhos_t1):12.4f} | {np.mean(rhos_alt):12.4f} | "
                f"{u_stat:8.1f} | {p_val:8.4f} | {'YES' if reject else 'no'}"
            )
            h4_results.append({
                "n_draws": n_sub,
                "mean_rho_T1": float(np.mean(rhos_t1)),
                f"mean_rho_T{t_alt}": float(np.mean(rhos_alt)),
                "mann_whitney_U": u_stat,
                "p_value": p_val,
                "reject_H0_alpha_0.05": reject,
            })

        with open(output_dir / f"h4_sample_efficiency_T{t_alt}.json", "w") as f:
            json.dump(h4_results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze mode count x temperature experiment"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_results = load_all_results(args.input_dir)
    if not all_results:
        print(f"No T_*.json files found in {args.input_dir}")
        return
    print(f"Loaded temperatures: {sorted(all_results.keys())}")

    # Count runs
    for temp, data in sorted(all_results.items()):
        n_runs = len(data["runs"])
        by_m = group_D_raw_by_m(data)
        n_m = len(by_m)
        draws_per_m = n_runs // n_m if n_m > 0 else 0
        print(f"  T={temp}: {n_runs} runs ({n_m} mode counts x {draws_per_m} draws)")

    print("\n--- H1': Within-m Variance Reduction ---")
    h1_variance_reduction(all_results, args.output_dir)

    print("\n--- H2': Ordering Preservation ---")
    h2_ordering_preservation(all_results, args.output_dir)

    print("\n--- H3': Discriminability ---")
    h3_discriminability(all_results, args.output_dir)

    print("\n--- H4': Sample Efficiency ---")
    h4_sample_efficiency(all_results, args.output_dir)

    print("\nAll analysis complete!")


if __name__ == "__main__":
    main()
