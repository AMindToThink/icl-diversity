"""Analyze temperature experiment results and produce plots.

Reads JSON files from results/temperature_experiments/ and generates:
1. Variance reduction: std(D) vs T per scenario
2. Ranking preservation: Kendall tau of scenario rankings vs T=1.0
3. Permutation efficiency: Kendall tau vs n_permutations at T=1.0 vs T=2.0
4. Bonus: a_k curves for multi_mode scenario at all temperatures

Statistical Hypotheses
======================

**H1 — Variance Reduction (Experiment 1)**
  H0: sigma_D(T) = sigma_D(1.0) for all T > 1  (temperature does not reduce
      permutation variance of the diversity score D).
  H1: sigma_D(T) < sigma_D(1.0) for T > 1  (higher temperature reduces
      permutation variance).
  Test: paired one-sided Wilcoxon signed-rank across (scenario, prompt) pairs,
      comparing per-permutation std(D) at T vs T=1.0.

**H2 — Ranking Preservation (Experiment 2)**
  H0: tau(T, 1.0) = 0  (scenario ranking at temperature T is unrelated to
      the calibrated ranking at T=1.0).
  H1: tau(T, 1.0) > 0  (rankings are positively correlated — temperature
      shifts absolute D values but preserves ordinal structure).
  Test: Kendall tau between scenario mean-D rankings at each T vs T=1.0,
      with exact permutation p-value.

**H3 — Permutation Efficiency (Experiment 3)**
  H0: tau_sub(T=2, n) = tau_sub(T=1, n)  (at a fixed subsample size n,
      T=2.0 recovers the ground-truth ranking no better than T=1.0).
  H1: tau_sub(T=2, n) > tau_sub(T=1, n)  (T=2.0 needs fewer permutations
      to match T=1.0's ranking accuracy — the practical payoff of variance
      reduction).
  Test: bootstrap comparison — at each subsample size n, compare the
      distribution of Kendall tau across bootstrap resamples at T=2 vs T=1
      using a Mann-Whitney U one-sided test.

Usage:
    uv run python scripts/analyze_temperature.py
    uv run python scripts/analyze_temperature.py --input-dir results/temperature_experiments/ --output-dir figures/temperature/
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "temperature_experiments"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "temperature"


def load_all_results(input_dir: Path) -> dict[float, dict[str, Any]]:
    """Load all T_*.json files, keyed by temperature."""
    results: dict[float, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("T_*.json")):
        with open(path) as f:
            data = json.load(f)
        temp = float(data["temperature"])
        results[temp] = data
    return results


def compute_per_permutation_D(
    per_perm_curves: list[list[float]],
    per_perm_byte_counts: list[list[int]],
    unconditional_per_byte: list[float],
) -> list[float]:
    """Compute D for each permutation from its a_k curve."""
    ds: list[float] = []
    n = len(unconditional_per_byte)
    mean_h = sum(unconditional_per_byte) / n
    C = 2.0 ** (-mean_h)

    for curve_tb, byte_counts in zip(per_perm_curves, per_perm_byte_counts):
        # E = sum(a_k - a_n) in total bits
        a_n = curve_tb[-1]
        E = sum(a_k - a_n for a_k in curve_tb)
        D = C * E
        ds.append(D)
    return ds


def _collect_per_prompt_std_D(
    all_results: dict[float, dict[str, Any]],
    temperature: float,
) -> list[float]:
    """Collect std(D) for each (scenario, prompt) pair at a given temperature."""
    stds: list[float] = []
    data = all_results[temperature]
    for scenario_entries in data["scenarios"].values():
        for entry in scenario_entries:
            per_perm_curves = entry.get("per_permutation_a_k_curves")
            per_perm_byte_counts = entry.get("per_permutation_byte_counts")
            if per_perm_curves is None:
                continue
            ds = compute_per_permutation_D(
                per_perm_curves,
                per_perm_byte_counts,
                entry["unconditional_surprises"],
            )
            stds.append(float(np.std(ds)))
    return stds


def experiment1_variance_reduction(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H1: Temperature T>1 reduces permutation variance of D.

    H0: sigma_D(T) = sigma_D(1.0) for T > 1.
    H1: sigma_D(T) < sigma_D(1.0) for T > 1.
    Test: paired one-sided Wilcoxon signed-rank across (scenario, prompt) pairs.
    """
    temperatures = sorted(all_results.keys())
    scenario_names = list(all_results[temperatures[0]]["scenarios"].keys())

    # --- Plot 1: std(D) vs T, one line per scenario ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for scenario_name in scenario_names:
        stds: list[float] = []
        for temp in temperatures:
            scenario_entries = all_results[temp]["scenarios"].get(scenario_name, [])
            all_D_stds: list[float] = []
            for entry in scenario_entries:
                per_perm_curves = entry.get("per_permutation_a_k_curves")
                per_perm_byte_counts = entry.get("per_permutation_byte_counts")
                if per_perm_curves is None:
                    continue
                ds = compute_per_permutation_D(
                    per_perm_curves,
                    per_perm_byte_counts,
                    entry["unconditional_surprises"],
                )
                all_D_stds.append(float(np.std(ds)))
            stds.append(float(np.mean(all_D_stds)) if all_D_stds else 0.0)
        ax.plot(temperatures, stds, "o-", label=scenario_name)

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("std(D) across permutations")
    ax.set_title("H1: Variance Reduction — std(D) vs Temperature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_reduction_std_D.png", dpi=150)
    plt.close(fig)
    print("  Saved variance_reduction_std_D.png")

    # --- Plot 2: mean per-position a_k std vs T ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for scenario_name in scenario_names:
        mean_pos_stds: list[float] = []
        for temp in temperatures:
            scenario_entries = all_results[temp]["scenarios"].get(scenario_name, [])
            all_pos_stds: list[float] = []
            for entry in scenario_entries:
                per_perm_curves = entry.get("per_permutation_a_k_curves")
                if per_perm_curves is None:
                    continue
                curves_arr = np.array(per_perm_curves)  # (n_perms, n_responses)
                pos_std = float(np.mean(np.std(curves_arr, axis=0)))
                all_pos_stds.append(pos_std)
            mean_pos_stds.append(float(np.mean(all_pos_stds)) if all_pos_stds else 0.0)
        ax2.plot(temperatures, mean_pos_stds, "o-", label=scenario_name)

    ax2.set_xlabel("Temperature T")
    ax2.set_ylabel("Mean per-position a_k std")
    ax2.set_title("H1: Per-Position a_k Variance vs Temperature")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "variance_reduction_ak_std.png", dpi=150)
    plt.close(fig2)
    print("  Saved variance_reduction_ak_std.png")

    # --- Formal hypothesis test: paired Wilcoxon for each T > 1 vs T=1 ---
    if 1.0 not in all_results:
        print("  WARNING: T=1.0 not in results, skipping H1 formal test")
        return

    ref_stds = _collect_per_prompt_std_D(all_results, 1.0)
    print("\n  H1 formal tests (paired Wilcoxon, one-sided: sigma_D(T) < sigma_D(1.0))")
    print(f"  {'T':>5} | {'mean std(D)':>11} | {'ref std(D)':>10} | {'W stat':>8} | {'p-value':>8} | reject H0?")
    print(f"  {'---':>5}-+-{'---':>11}-+-{'---':>10}-+-{'---':>8}-+-{'---':>8}-+----------")

    h1_results: list[dict[str, Any]] = []
    for temp in sorted(t for t in temperatures if t > 1.0):
        test_stds = _collect_per_prompt_std_D(all_results, temp)
        if len(test_stds) != len(ref_stds):
            print(f"  {temp:5.1f} | SKIP (mismatched prompt counts)")
            continue
        # Paired differences: ref - test (positive if T reduces variance)
        diffs = [r - t for r, t in zip(ref_stds, test_stds)]
        if all(d == 0.0 for d in diffs):
            print(f"  {temp:5.1f} | {np.mean(test_stds):11.4f} | {np.mean(ref_stds):10.4f} | {'N/A':>8} | {'N/A':>8} | no diffs")
            continue
        w_result = stats.wilcoxon(ref_stds, test_stds, alternative="greater")
        w_stat = float(w_result.statistic)
        p_val = float(w_result.pvalue)
        reject = p_val < 0.05
        print(
            f"  {temp:5.1f} | {np.mean(test_stds):11.4f} | {np.mean(ref_stds):10.4f} | "
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


def experiment2_ranking_preservation(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H2: Scenario rankings at T != 1.0 are positively correlated with T=1.0.

    H0: tau(T, 1.0) = 0  (no correlation).
    H1: tau(T, 1.0) > 0  (rankings preserved).
    Test: Kendall tau with exact permutation p-value.
    """
    temperatures = sorted(all_results.keys())
    if 1.0 not in all_results:
        print("  WARNING: T=1.0 not in results, skipping ranking preservation")
        return

    scenario_names = list(all_results[temperatures[0]]["scenarios"].keys())

    def get_mean_D(data: dict[str, Any], scenario: str) -> float:
        entries = data["scenarios"].get(scenario, [])
        ds = [e["diversity_score_D"] for e in entries]
        return float(np.mean(ds)) if ds else 0.0

    # Reference ranking at T=1.0
    ref_ds = [get_mean_D(all_results[1.0], s) for s in scenario_names]
    ref_ranking = list(np.argsort(ref_ds))

    rows: list[dict[str, Any]] = []
    for temp in temperatures:
        ds = [get_mean_D(all_results[temp], s) for s in scenario_names]
        ranking = list(np.argsort(ds))
        tau_result = stats.kendalltau(ref_ranking, ranking)
        tau = float(tau_result.statistic)
        pvalue = float(tau_result.pvalue)
        ranked_scenarios = [scenario_names[i] for i in np.argsort(ds)]
        reject = pvalue < 0.05 and tau > 0
        rows.append({
            "temperature": temp,
            "kendall_tau": tau,
            "p_value": pvalue,
            "reject_H0_alpha_0.05": reject,
            "scenario_ranking": ranked_scenarios,
            "mean_D_per_scenario": {s: get_mean_D(all_results[temp], s) for s in scenario_names},
        })

    # Save table
    table_path = output_dir / "h2_ranking_preservation.json"
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)
    print("  Saved h2_ranking_preservation.json")

    # Print table
    print("\n  H2: Kendall tau of scenario rankings vs T=1.0")
    print("  H0: tau = 0 (no correlation)  |  H1: tau > 0 (rankings preserved)")
    print(f"  {'T':>5} | {'tau':>6} | {'p-value':>8} | {'reject?':>7} | Ranking (low D → high D)")
    print(f"  {'---':>5}-+-{'---':>6}-+-{'---':>8}-+-{'---':>7}-+-------------------------")
    for row in rows:
        ranking_str = " < ".join(row["scenario_ranking"])
        print(
            f"  {row['temperature']:5.1f} | {row['kendall_tau']:6.3f} | "
            f"{row['p_value']:8.4f} | {'YES' if row['reject_H0_alpha_0.05'] else 'no':>7} | "
            f"{ranking_str}"
        )


def _collect_per_prompt_entries(
    data: dict[str, Any],
) -> list[tuple[str, int, dict[str, Any]]]:
    """Flatten all (scenario, prompt_idx, entry) from a temperature result."""
    entries: list[tuple[str, int, dict[str, Any]]] = []
    for scenario_name, scenario_entries in data["scenarios"].items():
        for i, entry in enumerate(scenario_entries):
            entries.append((scenario_name, i, entry))
    return entries


def experiment3_permutation_efficiency(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """H3: T=2.0 with n permutations recovers ground-truth D ranking better than T=1.0.

    Ranks all per-prompt D values (not per-scenario means) to get enough items
    for meaningful correlation. Uses Spearman rho on 25 prompt-level D values.

    H0: rho_sub(T=2, n) = rho_sub(T=1, n)  (no advantage).
    H1: rho_sub(T=2, n) > rho_sub(T=1, n)  (T=2 more efficient).
    Test: at each subsample size, Mann-Whitney U one-sided test on bootstrap
        Spearman rho distributions.
    """
    if 1.0 not in all_results:
        print("  WARNING: T=1.0 not in results, skipping permutation efficiency")
        return

    # Build ground truth: per-prompt D at T=1.0 with all permutations
    ref_entries = _collect_per_prompt_entries(all_results[1.0])
    ref_ds = np.array([e["diversity_score_D"] for _, _, e in ref_entries])
    n_prompts = len(ref_ds)
    print(f"  Ranking {n_prompts} per-prompt D values (not {len(all_results[1.0]['scenarios'])} scenario means)")

    subsample_sizes = [3, 5, 10, 20, 50]
    test_temperatures = [t for t in [1.0, 1.5, 2.0] if t in all_results]
    n_bootstrap = 200
    rng = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Store bootstrap rho distributions for formal testing
    rho_distributions: dict[float, dict[int, list[float]]] = {}

    for temp in test_temperatures:
        entries = _collect_per_prompt_entries(all_results[temp])
        # Pre-compute per-permutation D for each prompt
        all_perm_ds_per_prompt: list[list[float]] = []
        for _, _, entry in entries:
            per_perm_curves = entry.get("per_permutation_a_k_curves")
            per_perm_byte_counts = entry.get("per_permutation_byte_counts")
            if per_perm_curves is None:
                all_perm_ds_per_prompt.append([entry["diversity_score_D"]])
            else:
                all_perm_ds_per_prompt.append(
                    compute_per_permutation_D(
                        per_perm_curves,
                        per_perm_byte_counts,
                        entry["unconditional_surprises"],
                    )
                )

        mean_rhos: list[float] = []
        std_rhos: list[float] = []
        rho_distributions[temp] = {}

        for n_sub in subsample_sizes:
            rhos: list[float] = []
            for _ in range(n_bootstrap):
                ds = np.zeros(n_prompts)
                for j, perm_ds in enumerate(all_perm_ds_per_prompt):
                    n_avail = len(perm_ds)
                    if n_avail <= n_sub:
                        ds[j] = float(np.mean(perm_ds))
                    else:
                        idx = rng.choice(n_avail, size=n_sub, replace=False)
                        ds[j] = float(np.mean([perm_ds[i] for i in idx]))

                rho = float(stats.spearmanr(ref_ds, ds).statistic)
                rhos.append(rho)

            rho_distributions[temp][n_sub] = rhos
            mean_rhos.append(float(np.mean(rhos)))
            std_rhos.append(float(np.std(rhos)))

        ax.errorbar(
            subsample_sizes,
            mean_rhos,
            yerr=std_rhos,
            fmt="o-",
            label=f"T={temp}",
            capsize=3,
        )

    ax.set_xlabel("Number of permutations (subsampled)")
    ax.set_ylabel("Spearman rho vs ground truth (T=1.0, all perms)")
    ax.set_title(f"H3: Permutation Efficiency ({n_prompts} prompt-level D values)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    fig.savefig(output_dir / "permutation_efficiency.png", dpi=150)
    plt.close(fig)
    print("  Saved permutation_efficiency.png")

    # --- Paired difference plot: delta_rho = rho(T>1) - rho(T=1) ---
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
            ax2.fill_between(
                subsample_sizes, ci_lo, ci_hi, alpha=0.2,
            )
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, label="no difference")
        ax2.set_xlabel("Number of permutations (subsampled)")
        ax2.set_ylabel("Δ Spearman rho  (T>1 minus T=1.0)")
        ax2.set_title(f"H3: Ranking Advantage of Higher Temperature ({n_prompts} prompts, 95% CI)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(output_dir / "permutation_efficiency_delta.png", dpi=150)
        plt.close(fig2)
        print("  Saved permutation_efficiency_delta.png")

    # --- Formal hypothesis test: Mann-Whitney U at each subsample size ---
    for t_alt in [1.5, 2.0]:
        if 1.0 not in rho_distributions or t_alt not in rho_distributions:
            continue
        print(f"\n  H3 formal tests: T={t_alt} vs T=1.0 (Mann-Whitney U, one-sided: rho(T={t_alt}) > rho(T=1))")
        print(f"  {'n_perm':>6} | {'mean rho T=1':>12} | {f'mean rho T={t_alt}':>12} | {'U stat':>8} | {'p-value':>8} | reject H0?")
        print(f"  {'---':>6}-+-{'---':>12}-+-{'---':>12}-+-{'---':>8}-+-{'---':>8}-+----------")

        h3_results: list[dict[str, Any]] = []
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
            h3_results.append({
                "n_permutations": n_sub,
                "mean_rho_T1": float(np.mean(rhos_t1)),
                f"mean_rho_T{t_alt}": float(np.mean(rhos_alt)),
                "mann_whitney_U": u_stat,
                "p_value": p_val,
                "reject_H0_alpha_0.05": reject,
            })

        with open(output_dir / f"h3_permutation_efficiency_T{t_alt}.json", "w") as f:
            json.dump(h3_results, f, indent=2)


def bonus_ak_curves(
    all_results: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Overlay a_k curves for multi_mode scenario at all temperatures."""
    temperatures = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in temperatures:
        entries = all_results[temp]["scenarios"].get("multi_mode", [])
        if not entries:
            continue
        # Use first prompt
        entry = entries[0]
        curve = entry["a_k_curve"]
        ax.plot(range(1, len(curve) + 1), curve, "o-", label=f"T={temp}", markersize=4)

    ax.set_xlabel("Response index k")
    ax.set_ylabel("a_k (total bits)")
    ax.set_title("multi_mode: a_k Curves at Different Temperatures")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "multi_mode_ak_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved multi_mode_ak_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze temperature experiments")
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_results = load_all_results(args.input_dir)
    if not all_results:
        print(f"No T_*.json files found in {args.input_dir}")
        return
    print(f"Loaded temperatures: {sorted(all_results.keys())}")

    print("\nExperiment 1 / H1: Variance reduction")
    experiment1_variance_reduction(all_results, args.output_dir)

    print("\nExperiment 2 / H2: Ranking preservation")
    experiment2_ranking_preservation(all_results, args.output_dir)

    print("\nExperiment 3 / H3: Permutation efficiency")
    experiment3_permutation_efficiency(all_results, args.output_dir)

    print("\nBonus: a_k curve overlay")
    bonus_ak_curves(all_results, args.output_dir)

    print("\nAll analysis complete!")


if __name__ == "__main__":
    main()
