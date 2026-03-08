"""Run all hypothesis tests for the Qwen2.5-32B vs GPT-2 comparison.

Reads pre-computed scenario metrics from JSON files and runs:
- Q13: Original H1-H13 hypotheses on Qwen data
- Q1-Q12: Cross-model paired comparisons (Wilcoxon signed-rank)

Usage:
    uv run python scripts/test_hypotheses.py
    uv run python scripts/test_hypotheses.py --gpt2 results/scenario_metrics.json --qwen results/scenario_metrics_qwen2.5-32b.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def get(data: dict[str, Any], scenario: str, metric: str) -> np.ndarray:
    """Extract metric values for all prompts in a scenario."""
    return np.array([m[metric] for m in data["scenarios"][scenario]])


def mean(vals: np.ndarray) -> float:
    return float(np.mean(vals))


def mannwhitney_greater(
    x: np.ndarray, y: np.ndarray, name: str, alpha: float = 0.05
) -> dict[str, Any]:
    """One-sided Mann-Whitney U: x stochastically > y."""
    stat, p = scipy_stats.mannwhitneyu(x, y, alternative="greater")
    passed = p < alpha
    return {
        "name": name,
        "x_mean": mean(x),
        "y_mean": mean(y),
        "U": float(stat),
        "p": float(p),
        "passed": passed,
    }


def wilcoxon_greater(
    x: np.ndarray, y: np.ndarray, name: str
) -> dict[str, Any]:
    """One-sided Wilcoxon signed-rank: x > y (paired)."""
    diffs = x - y
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    try:
        stat, p = scipy_stats.wilcoxon(x, y, alternative="greater")
    except ValueError:
        # All differences are zero
        stat, p = None, 1.0
    return {
        "name": name,
        "x_mean": mean(x),
        "y_mean": mean(y),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "W": float(stat) if stat is not None else None,
        "p": float(p),
        "passed": p < 0.05 if p is not None else False,
    }


def direction_check(
    x: np.ndarray, y: np.ndarray, name: str
) -> dict[str, Any]:
    """Check that mean(x) > mean(y)."""
    return {
        "name": name,
        "x_mean": mean(x),
        "y_mean": mean(y),
        "passed": mean(x) > mean(y),
    }


def threshold_check(
    vals: np.ndarray, threshold: float, direction: str, name: str
) -> dict[str, Any]:
    """Check mean(vals) < or > threshold."""
    m = mean(vals)
    if direction == "lt":
        passed = m < threshold
    else:
        passed = m > threshold
    return {
        "name": name,
        "mean": m,
        "threshold": threshold,
        "direction": direction,
        "passed": passed,
    }


def kendall_tau_check(data: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Check that a_k curves have negative Kendall tau (majority rule)."""
    taus = []
    for m in data["scenarios"][scenario]:
        curve = np.array(m["a_k_curve"])
        k = np.arange(len(curve))
        tau, p = scipy_stats.kendalltau(k, curve)
        taus.append({"tau": float(tau), "p": float(p), "label": m.get("prompt_label", "?")})
    n_negative = sum(1 for t in taus if t["tau"] < 0)
    return {
        "name": f"a_k decreasing for {scenario}",
        "taus": taus,
        "n_negative": n_negative,
        "n_total": len(taus),
        "passed": n_negative > len(taus) // 2,
    }


def monotonicity_count(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Count monotone a_k curves per scenario."""
    result = {}
    for scenario, metrics_list in data["scenarios"].items():
        n_mono = sum(1 for m in metrics_list if m["is_monotone"])
        result[scenario] = {"monotone": n_mono, "total": len(metrics_list)}
    return result


def run_q13_tests(qwen: dict[str, Any]) -> list[dict[str, Any]]:
    """Run original H1-H13 hypotheses on Qwen data."""
    results = []

    # H1: C(multi_mode) > C(noise)
    results.append(mannwhitney_greater(
        get(qwen, "multi_mode", "coherence_C"),
        get(qwen, "pure_noise", "coherence_C"),
        "H1: C(multi_mode) > C(noise)",
    ))

    # H2: C(one_mode) > C(noise)
    results.append(mannwhitney_greater(
        get(qwen, "one_mode", "coherence_C"),
        get(qwen, "pure_noise", "coherence_C"),
        "H2: C(one_mode) > C(noise)",
    ))

    # H3: C(multi_mode) > C(multi_incoherent)
    results.append(mannwhitney_greater(
        get(qwen, "multi_mode", "coherence_C"),
        get(qwen, "multi_incoherent", "coherence_C"),
        "H3: C(multi_mode) > C(multi_incoherent)",
    ))

    # H4: E(multi_mode) > E(one_mode)
    results.append(direction_check(
        get(qwen, "multi_mode", "excess_entropy_E"),
        get(qwen, "one_mode", "excess_entropy_E"),
        "H4: E(multi_mode) > E(one_mode)",
    ))

    # H5: D(multi_mode) > D(one_mode)
    results.append(direction_check(
        get(qwen, "multi_mode", "diversity_score_D"),
        get(qwen, "one_mode", "diversity_score_D"),
        "H5: D(multi_mode) > D(one_mode)",
    ))

    # H6: D(multi_mode) > D(noise)
    results.append(mannwhitney_greater(
        get(qwen, "multi_mode", "diversity_score_D"),
        get(qwen, "pure_noise", "diversity_score_D"),
        "H6: D(multi_mode) > D(noise)",
    ))

    # H7: D(multi_mode) > D(multi_incoherent)
    results.append(mannwhitney_greater(
        get(qwen, "multi_mode", "diversity_score_D"),
        get(qwen, "multi_incoherent", "diversity_score_D"),
        "H7: D(multi_mode) > D(multi_incoherent)",
    ))

    # H8: sigma(mixed) > sigma(multi_mode)
    results.append(mannwhitney_greater(
        get(qwen, "mixed", "coherence_spread_sigma"),
        get(qwen, "multi_mode", "coherence_spread_sigma"),
        "H8: sigma(mixed) > sigma(multi_mode)",
    ))

    # H9: sigma(mixed) > sigma(one_mode)
    results.append(mannwhitney_greater(
        get(qwen, "mixed", "coherence_spread_sigma"),
        get(qwen, "one_mode", "coherence_spread_sigma"),
        "H9: sigma(mixed) > sigma(one_mode)",
    ))

    # H10: a_k has negative Kendall tau for multi_mode
    results.append(kendall_tau_check(qwen, "multi_mode"))

    # H11: C(noise) < 0.05
    results.append(threshold_check(
        get(qwen, "pure_noise", "coherence_C"), 0.05, "lt",
        "H11: C(noise) < 0.05",
    ))

    # H12: C(one_mode) > 0.1
    results.append(threshold_check(
        get(qwen, "one_mode", "coherence_C"), 0.1, "gt",
        "H12: C(one_mode) > 0.1",
    ))

    # H13: E(one_mode) > 0
    results.append(threshold_check(
        get(qwen, "one_mode", "excess_entropy_E"), 0.0, "gt",
        "H13: E(one_mode) > 0",
    ))

    return results


def run_cross_model_tests(
    gpt2: dict[str, Any], qwen: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run cross-model hypotheses Q1-Q12."""
    results = []

    # Q1: C_q(multi_mode) > C_g(multi_mode)
    results.append(wilcoxon_greater(
        get(qwen, "multi_mode", "coherence_C"),
        get(gpt2, "multi_mode", "coherence_C"),
        "Q1: C_q(multi_mode) > C_g(multi_mode)",
    ))

    # Q2: C_q(one_mode) > C_g(one_mode)
    results.append(wilcoxon_greater(
        get(qwen, "one_mode", "coherence_C"),
        get(gpt2, "one_mode", "coherence_C"),
        "Q2: C_q(one_mode) > C_g(one_mode)",
    ))

    # Q3: C_q(noise) < 0.02
    results.append(threshold_check(
        get(qwen, "pure_noise", "coherence_C"), 0.02, "lt",
        "Q3: C_q(noise) < 0.02",
    ))

    # Q4: C_q(multi_incoherent) ≈ C_g(multi_incoherent)
    cg = get(gpt2, "multi_incoherent", "coherence_C")
    cq = get(qwen, "multi_incoherent", "coherence_C")
    results.append({
        "name": "Q4: C_q(multi_incoherent) ≈ C_g(multi_incoherent)",
        "q_mean": mean(cq),
        "g_mean": mean(cg),
        "ratio": mean(cq) / mean(cg) if mean(cg) != 0 else float("inf"),
        "note": "Low confidence; direction unclear",
        "passed": None,  # No clear pass/fail for ≈
    })

    # Q5: E_q(multi_incoherent) > E_g(multi_incoherent)
    results.append(wilcoxon_greater(
        get(qwen, "multi_incoherent", "excess_entropy_E"),
        get(gpt2, "multi_incoherent", "excess_entropy_E"),
        "Q5: E_q(multi_incoherent) > E_g(multi_incoherent)",
    ))

    # Q6: E_q(mixed) > E_g(mixed)
    results.append(wilcoxon_greater(
        get(qwen, "mixed", "excess_entropy_E"),
        get(gpt2, "mixed", "excess_entropy_E"),
        "Q6: E_q(mixed) > E_g(mixed)",
    ))

    # Q7: E_q(multi_mode) >= E_g(multi_mode)
    eq = get(qwen, "multi_mode", "excess_entropy_E")
    eg = get(gpt2, "multi_mode", "excess_entropy_E")
    results.append(direction_check(
        eq, eg,
        "Q7: E_q(multi_mode) >= E_g(multi_mode)",
    ))

    # Q8: E_q(noise) ≈ 0
    eq_noise = get(qwen, "pure_noise", "excess_entropy_E")
    eg_noise = get(gpt2, "pure_noise", "excess_entropy_E")
    results.append({
        "name": "Q8: E_q(noise) ≈ 0",
        "q_mean": mean(eq_noise),
        "g_mean": mean(eg_noise),
        "closer_to_zero": abs(mean(eq_noise)) < abs(mean(eg_noise)),
        "passed": abs(mean(eq_noise)) < abs(mean(eg_noise)),
    })

    # Q9: D_q(multi_mode) > D_g(multi_mode)
    results.append(direction_check(
        get(qwen, "multi_mode", "diversity_score_D"),
        get(gpt2, "multi_mode", "diversity_score_D"),
        "Q9: D_q(multi_mode) > D_g(multi_mode)",
    ))

    # Q10: sigma_q(mixed) > sigma_g(mixed)
    results.append(wilcoxon_greater(
        get(qwen, "mixed", "coherence_spread_sigma"),
        get(gpt2, "mixed", "coherence_spread_sigma"),
        "Q10: sigma_q(mixed) > sigma_g(mixed)",
    ))

    # Q11: sigma_q(one_mode) ≈ sigma_g(one_mode)
    sg = get(gpt2, "one_mode", "coherence_spread_sigma")
    sq = get(qwen, "one_mode", "coherence_spread_sigma")
    results.append({
        "name": "Q11: sigma_q(one_mode) ≈ sigma_g(one_mode)",
        "q_mean": mean(sq),
        "g_mean": mean(sg),
        "ratio": mean(sq) / mean(sg) if mean(sg) != 0 else float("inf"),
        "passed": None,  # ≈ comparison
    })

    # Q12: More monotone curves for Qwen
    g_mono_total = sum(
        1 for s in gpt2["scenarios"].values() for m in s if m["is_monotone"]
    )
    q_mono_total = sum(
        1 for s in qwen["scenarios"].values() for m in s if m["is_monotone"]
    )
    mono_detail = {}
    for scenario in ["pure_noise", "multi_incoherent", "multi_mode", "one_mode", "mixed"]:
        g_n = sum(1 for m in gpt2["scenarios"][scenario] if m["is_monotone"])
        q_n = sum(1 for m in qwen["scenarios"][scenario] if m["is_monotone"])
        mono_detail[scenario] = {"gpt2": g_n, "qwen": q_n, "total": len(qwen["scenarios"][scenario])}
    results.append({
        "name": "Q12: More monotone a_k curves for Qwen",
        "gpt2_total": g_mono_total,
        "qwen_total": q_mono_total,
        "detail": mono_detail,
        "passed": q_mono_total > g_mono_total,
    })

    return results


def print_results(
    q13_results: list[dict[str, Any]],
    cross_results: list[dict[str, Any]],
) -> None:
    """Print formatted results."""
    print("=" * 80)
    print("Q13: Original H1-H13 hypotheses tested on Qwen2.5-32B")
    print("=" * 80)

    n_passed = 0
    n_total = 0
    for r in q13_results:
        name = r["name"]
        passed = r.get("passed", None)
        status = "PASS" if passed else "FAIL" if passed is not None else "N/A"

        if "U" in r:
            print(f"  {status:4s}  {name}")
            print(f"        x_mean={r['x_mean']:.4f}, y_mean={r['y_mean']:.4f}, U={r['U']:.1f}, p={r['p']:.4f}")
        elif "taus" in r:
            print(f"  {status:4s}  {name}")
            for t in r["taus"]:
                print(f"        {t['label']}: tau={t['tau']:.3f}, p={t['p']:.4f}")
            print(f"        {r['n_negative']}/{r['n_total']} negative")
        elif "threshold" in r:
            print(f"  {status:4s}  {name}")
            print(f"        mean={r['mean']:.4f}, threshold={r['threshold']}")
        elif "x_mean" in r:
            print(f"  {status:4s}  {name}")
            print(f"        x_mean={r['x_mean']:.4f}, y_mean={r['y_mean']:.4f}")

        if passed is not None:
            n_total += 1
            if passed:
                n_passed += 1

    print(f"\n  Summary: {n_passed}/{n_total} hypotheses supported")

    print("\n" + "=" * 80)
    print("Q1-Q12: Cross-model hypotheses (Qwen2.5-32B vs GPT-2)")
    print("=" * 80)

    n_passed = 0
    n_total = 0
    for r in cross_results:
        name = r["name"]
        passed = r.get("passed", None)
        status = "PASS" if passed else "FAIL" if passed is not None else "N/A"

        if "W" in r:
            w_str = f"W={r['W']:.1f}" if r["W"] is not None else "W=N/A"
            print(f"  {status:4s}  {name}")
            print(f"        q_mean={r['x_mean']:.4f}, g_mean={r['y_mean']:.4f}, +:{r['n_pos']} -:{r['n_neg']}, {w_str}, p={r['p']:.4f}")
        elif "threshold" in r:
            print(f"  {status:4s}  {name}")
            print(f"        mean={r['mean']:.4f}, threshold={r['threshold']}")
        elif "ratio" in r:
            print(f"  {status if status != 'N/A' else 'INFO':4s}  {name}")
            print(f"        q_mean={r.get('q_mean', r.get('x_mean', 0)):.4f}, g_mean={r.get('g_mean', r.get('y_mean', 0)):.4f}, ratio={r['ratio']:.2f}x")
        elif "detail" in r:
            print(f"  {status:4s}  {name}")
            print(f"        GPT-2 total: {r['gpt2_total']}/25, Qwen total: {r['qwen_total']}/25")
            for s, d in r["detail"].items():
                print(f"          {s}: GPT-2={d['gpt2']}/{d['total']}, Qwen={d['qwen']}/{d['total']}")
        elif "closer_to_zero" in r:
            print(f"  {status:4s}  {name}")
            print(f"        E_q={r['q_mean']:.4f}, E_g={r['g_mean']:.4f}, Qwen closer to 0: {r['closer_to_zero']}")
        elif "x_mean" in r:
            print(f"  {status:4s}  {name}")
            print(f"        q_mean={r['x_mean']:.4f}, g_mean={r['y_mean']:.4f}")

        if passed is not None:
            n_total += 1
            if passed:
                n_passed += 1

    print(f"\n  Summary: {n_passed}/{n_total} hypotheses supported")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hypothesis tests for Qwen2.5-32B vs GPT-2 comparison"
    )
    parser.add_argument(
        "--gpt2",
        type=Path,
        default=RESULTS_DIR / "scenario_metrics.json",
        help="GPT-2 scenario metrics JSON",
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=RESULTS_DIR / "scenario_metrics_qwen2.5-32b.json",
        help="Qwen scenario metrics JSON",
    )
    args = parser.parse_args()

    with open(args.gpt2) as f:
        gpt2 = json.load(f)
    with open(args.qwen) as f:
        qwen = json.load(f)

    print(f"GPT-2 data: {args.gpt2} (model: {gpt2.get('base_model', '?')})")
    print(f"Qwen data:  {args.qwen} (model: {qwen.get('base_model', '?')})")

    q13_results = run_q13_tests(qwen)
    cross_results = run_cross_model_tests(gpt2, qwen)
    print_results(q13_results, cross_results)


if __name__ == "__main__":
    main()
