"""Per-byte analysis of Tevet evaluation results.

Loads full sidecar JSONs (which have per-permutation a_k curves and byte counts),
computes correct per-byte a_k curves (mean of ratios, not ratio of means),
fits exponential curves, and reports Spearman correlations.

Also generates per-byte versions of key plots.

Usage:
    uv run python scripts/analyze_per_byte.py --run-tag qwen25_completion
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ak_curves import exponential_ak, fit_exponential

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"


def load_sidecar(sidecar_path: Path) -> dict[str, dict]:
    """Load full sidecar JSON."""
    with open(sidecar_path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def compute_per_byte_curve(entry: dict) -> list[float] | None:
    """Compute the correct mean per-byte a_k curve from per-permutation data.

    Returns mean of (bits_k / bytes_k) across permutations for each position k.
    This is the correct computation: mean of ratios, not ratio of means.
    """
    perm_curves = entry.get("per_permutation_a_k_curves")
    perm_bytes = entry.get("per_permutation_byte_counts")

    if perm_curves is None or perm_bytes is None:
        # Fall back to the pre-computed per-byte curve if available
        return entry.get("a_k_curve_per_byte")

    n_perms = len(perm_curves)
    n_positions = len(perm_curves[0])

    per_byte_curve = []
    for k in range(n_positions):
        ratios = []
        for p in range(n_perms):
            bits = perm_curves[p][k]
            byte_count = perm_bytes[p][k]
            if byte_count > 0:
                ratios.append(bits / byte_count)
        per_byte_curve.append(sum(ratios) / len(ratios) if ratios else 0.0)

    return per_byte_curve


def load_csv_rows(csv_path: Path) -> dict[str, dict]:
    """Load CSV rows keyed by sample_id."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def analyze_dataset(
    csv_path: Path,
    sidecar_path: Path,
    tag: str,
) -> dict[str, dict]:
    """Analyze a single dataset, returning per-sample metrics for both bits and bits/byte."""
    rows = load_csv_rows(csv_path)
    sidecar = load_sidecar(sidecar_path)

    e_col = f"metric_icl_E_{tag}"
    c_col = f"metric_icl_C_{tag}"

    results: dict[str, dict] = {}
    for sid, entry in sidecar.items():
        row = rows.get(sid)
        if row is None:
            continue

        a_k_bits = entry.get("a_k_curve")
        a_k_per_byte = compute_per_byte_curve(entry)
        if a_k_bits is None or a_k_per_byte is None:
            continue
        if len(a_k_bits) < 2:
            continue

        e_disc_bits = float(row.get(e_col, "nan"))
        c_val = float(row.get(c_col, "nan"))
        label = float(row.get("label_value", "nan"))
        if np.isnan(label):
            continue

        k = np.arange(1, len(a_k_bits) + 1, dtype=float)

        # Fit on total bits
        fit_bits, ok_bits = fit_exponential(k, np.array(a_k_bits))
        e_fit_bits = fit_bits["E_fit"] if ok_bits else float("nan")

        # Fit on per-byte
        fit_pb, ok_pb = fit_exponential(k, np.array(a_k_per_byte))
        e_fit_per_byte = fit_pb["E_fit"] if ok_pb else float("nan")

        # E_discrete for per-byte: sum(a_k_pb - a_n_pb)
        a_n_pb = a_k_per_byte[-1]
        e_disc_per_byte = sum(v - a_n_pb for v in a_k_per_byte)

        results[sid] = {
            "label": label,
            "C": c_val,
            "a_k_bits": a_k_bits,
            "a_k_per_byte": a_k_per_byte,
            "E_disc_bits": e_disc_bits,
            "E_fit_bits": e_fit_bits,
            "E_disc_per_byte": e_disc_per_byte,
            "E_fit_per_byte": e_fit_per_byte,
            "fit_ok_bits": ok_bits,
            "fit_ok_per_byte": ok_pb,
        }

    return results


def print_correlation_table(
    all_results: dict[str, dict[str, dict]],
    dataset_filter: str,
    label_name: str,
) -> None:
    """Print Spearman correlations for datasets matching filter."""
    print(f"\n{'=' * 80}")
    print(f"  {label_name}")
    print(f"{'=' * 80}")
    print(f"  {'Task':<35s} {'E_disc_bits':>12s} {'E_fit_bits':>12s} "
          f"{'E_disc_pb':>12s} {'E_fit_pb':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for dataset_name, samples in sorted(all_results.items()):
        if dataset_filter not in dataset_name:
            continue

        labels = [s["label"] for s in samples.values()]
        e_disc_bits = [s["E_disc_bits"] for s in samples.values()]
        e_fit_bits = [s["E_fit_bits"] for s in samples.values()]
        e_disc_pb = [s["E_disc_per_byte"] for s in samples.values()]
        e_fit_pb = [s["E_fit_per_byte"] for s in samples.values()]

        # Filter NaN for fit metrics
        def safe_rho(x: list[float], y: list[float]) -> str:
            pairs = [(a, b) for a, b in zip(x, y) if not np.isnan(a) and not np.isnan(b)]
            if len(pairs) < 5:
                return "nan"
            xs, ys = zip(*pairs)
            rho, _ = spearmanr(xs, ys)
            return f"{rho:+.3f}"

        task = dataset_name.split("/")[-1]
        print(f"  {task:<35s} {safe_rho(labels, e_disc_bits):>12s} "
              f"{safe_rho(labels, e_fit_bits):>12s} "
              f"{safe_rho(labels, e_disc_pb):>12s} "
              f"{safe_rho(labels, e_fit_pb):>12s}")


def plot_mean_ak_per_byte(
    all_results: dict[str, dict[str, dict]],
    output_dir: Path,
    tag: str,
) -> None:
    """Plot mean per-byte a_k curves with exponential fit for McDiv_nuggets."""
    nuggets = {k: v for k, v in all_results.items() if "mcdiv_nuggets" in k and "with_hds" in k}
    if not nuggets:
        return

    fig, axes = plt.subplots(1, len(nuggets), figsize=(6 * len(nuggets), 6))
    if len(nuggets) == 1:
        axes = [axes]
    fig.suptitle(f"Mean a_k (bits/byte) + Exponential Fit: McDiv_nuggets [{tag}]", fontsize=14)

    for ax_idx, (dataset_name, samples) in enumerate(sorted(nuggets.items())):
        ax = axes[ax_idx]
        task = dataset_name.split("_")[-2] + "_" + dataset_name.split("_")[-1]

        for target_label, group_label, color in [
            (1.0, "High diversity", "tab:red"),
            (0.0, "Low diversity", "tab:blue"),
        ]:
            group = [s for s in samples.values() if s["label"] == target_label]
            if not group:
                continue

            curves = np.array([s["a_k_per_byte"] for s in group])
            mean_curve = np.mean(curves, axis=0)
            sem_curve = np.std(curves, axis=0) / np.sqrt(len(group))
            n = len(mean_curve)
            k = np.arange(1, n + 1, dtype=float)

            ax.plot(k, mean_curve, "o-", color=color, markersize=4, linewidth=2,
                    label=group_label)
            ax.fill_between(k, mean_curve - sem_curve, mean_curve + sem_curve,
                            alpha=0.2, color=color)

            # E_discrete per-byte
            a_n = mean_curve[-1]
            E_disc = sum(mean_curve[i] - a_n for i in range(n))

            # Exponential fit
            fit_params, fit_ok = fit_exponential(k, mean_curve)
            if fit_ok:
                k_fine = np.linspace(0.5, n + 3, 100)
                fitted = exponential_ak(k_fine, fit_params["a_inf"],
                                        fit_params["alpha"], fit_params["beta"])
                ax.plot(k_fine, fitted, "--", color=color, linewidth=1.5,
                        label=f"  fit: E_disc={E_disc:.3f}, E_fit={fit_params['E_fit']:.3f}")

        ax.set_xticks(range(1, n + 1))
        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("k (response index)")
        ax.set_ylabel("a_k (bits/byte)")
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    path = output_dir / "mean_ak_per_byte_with_fit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_efit_per_byte_distributions(
    all_results: dict[str, dict[str, dict]],
    output_dir: Path,
    tag: str,
) -> None:
    """Plot E_fit distributions for bits vs bits/byte side by side."""
    nuggets = {k: v for k, v in all_results.items() if "mcdiv_nuggets" in k and "with_hds" in k}
    if not nuggets:
        return

    fig, axes = plt.subplots(2, len(nuggets), figsize=(6 * len(nuggets), 10))
    fig.suptitle(f"E_fit Distributions: bits vs bits/byte [{tag}]", fontsize=14)

    for col_idx, (dataset_name, samples) in enumerate(sorted(nuggets.items())):
        task = dataset_name.split("_")[-2] + "_" + dataset_name.split("_")[-1]

        for row_idx, (metric_key, metric_label) in enumerate([
            ("E_fit_bits", "E_fit (total bits)"),
            ("E_fit_per_byte", "E_fit (bits/byte)"),
        ]):
            ax = axes[row_idx][col_idx]
            high_vals = [s[metric_key] for s in samples.values()
                         if s["label"] == 1.0 and not np.isnan(s[metric_key])]
            low_vals = [s[metric_key] for s in samples.values()
                        if s["label"] == 0.0 and not np.isnan(s[metric_key])]

            if high_vals and low_vals:
                # Clip outliers for better visualization
                all_vals = high_vals + low_vals
                p99 = np.percentile(all_vals, 99)
                hist_range = (min(all_vals), min(max(all_vals), p99 * 1.5))

                ax.hist(high_vals, bins=25, alpha=0.5, color="tab:red",
                        label=f"Diverse (n={len(high_vals)})", range=hist_range)
                ax.hist(low_vals, bins=25, alpha=0.5, color="tab:blue",
                        label=f"Constant (n={len(low_vals)})", range=hist_range)

                h_mean = np.mean(high_vals)
                l_mean = np.mean(low_vals)
                ax.axvline(h_mean, color="tab:red", linestyle="--", linewidth=2)
                ax.axvline(l_mean, color="tab:blue", linestyle="--", linewidth=2)
                ax.legend()

            ax.set_title(f"{metric_label} — {task}")
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Count")

    plt.tight_layout()
    path = output_dir / "efit_per_byte_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-byte analysis of Tevet results")
    parser.add_argument("--run-tag", type=str, default="qwen25_completion")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = RESULTS_BASE / args.run_tag
    if not data_dir.exists():
        print(f"No data found at {data_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "figures" / "tevet_validation" / "inspection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    print("Loading sidecars and computing per-byte curves...")
    all_results: dict[str, dict[str, dict]] = {}

    for sidecar_path in sorted(data_dir.rglob(f"*.icl_curves.{args.run_tag}.json")):
        csv_stem = sidecar_path.name.replace(f".icl_curves.{args.run_tag}.json", "")
        csv_path = sidecar_path.with_name(csv_stem + ".csv")
        if not csv_path.exists():
            continue

        dataset_key = f"{sidecar_path.parent.name}/{csv_stem}"
        results = analyze_dataset(csv_path, sidecar_path, args.run_tag)
        all_results[dataset_key] = results
        print(f"  {dataset_key}: {len(results)} samples")

    # Print correlation tables
    print_correlation_table(all_results, "mcdiv_nuggets", "McDiv_nuggets (content diversity — PRIMARY)")
    print_correlation_table(all_results, "con_test", "ConTest (binary content diversity)")
    print_correlation_table(all_results, "dec_test", "DecTest (temperature correlation)")

    # Generate plots
    print("\nGenerating per-byte plots...")
    plot_mean_ak_per_byte(all_results, output_dir, args.run_tag)
    plot_efit_per_byte_distributions(all_results, output_dir, args.run_tag)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
