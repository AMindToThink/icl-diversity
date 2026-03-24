"""Visual inspection of Tevet evaluation results.

Generates diagnostic plots for manually verifying each step of the
ICL diversity metric pipeline. No model loading required — reads
from CSVs and mean curves JSONs.

Outputs go to figures/tevet_validation/inspection/.

Usage:
    uv run python scripts/inspect_tevet_results.py --run-tag qwen25_completion
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

# Import fitting functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ak_curves import exponential_ak, fit_exponential, fit_sigmoid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"


def load_csv_with_ids(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mean_curves(curves_path: Path) -> dict[str, dict]:
    with open(curves_path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def plot_individual_ak_curves(
    data_dir: Path, tag: str, output_dir: Path, n_examples: int = 8,
) -> None:
    """Plot 2: Individual a_k curves for high vs low diversity samples."""
    nuggets_dir = data_dir / "McDiv_nuggets"
    if not nuggets_dir.exists():
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Individual a_k Curves: High vs Low Diversity (McDiv_nuggets)", fontsize=14)

    for ax_idx, csv_path in enumerate(sorted(nuggets_dir.glob("*_with_hds_*.csv"))[:3]):
        curves_path = csv_path.with_suffix(f".icl_mean_curves.{tag}.json")
        if not curves_path.exists():
            continue

        rows = load_csv_with_ids(csv_path)
        curves = load_mean_curves(curves_path)
        task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]
        ax = axes[ax_idx]

        high_curves: list[list[float]] = []
        low_curves: list[list[float]] = []
        for row in rows:
            sid = row["sample_id"]
            if sid not in curves:
                continue
            a_k = curves[sid].get("a_k_curve")
            if a_k is None:
                continue
            label = float(row["label_value"])
            if label == 1.0:
                high_curves.append(a_k)
            else:
                low_curves.append(a_k)

        # Plot a subset
        n_k = 0
        for i, c in enumerate(high_curves[:n_examples]):
            k = np.arange(1, len(c) + 1)
            n_k = max(n_k, len(c))
            ax.plot(k, c, "o-", color="tab:red", alpha=0.3, markersize=3,
                    label="High diversity" if i == 0 else None)
        for i, c in enumerate(low_curves[:n_examples]):
            k = np.arange(1, len(c) + 1)
            n_k = max(n_k, len(c))
            ax.plot(k, c, "o-", color="tab:blue", alpha=0.3, markersize=3,
                    label="Low diversity" if i == 0 else None)

        ax.set_xticks(range(1, n_k + 1))
        ax.set_title(f"{task} (n={len(high_curves)}h/{len(low_curves)}l)")
        ax.set_xlabel("k (response index)")
        ax.set_ylabel("a_k (total bits)")
        ax.legend()

    plt.tight_layout()
    path = output_dir / "individual_ak_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mean_ak_with_fit(
    data_dir: Path, tag: str, output_dir: Path,
) -> None:
    """Plot 3: Mean a_k curves with exponential fit overlay (THE key plot)."""
    nuggets_dir = data_dir / "McDiv_nuggets"
    if not nuggets_dir.exists():
        return

    # Plot both with_hds and no_hds variants
    for variant_glob, variant_label in [
        ("*_with_hds_*.csv", "McDiv_nuggets (with HDS)"),
        ("*_no_hds_*.csv", "McDiv_nuggets (no HDS)"),
    ]:
        csv_files = sorted(nuggets_dir.glob(variant_glob))
        if not csv_files:
            continue

        fig, axes = plt.subplots(1, len(csv_files), figsize=(6 * len(csv_files), 6))
        if len(csv_files) == 1:
            axes = [axes]
        fig.suptitle(f"Mean a_k Curves + Exponential Fit: {variant_label} [{tag}]", fontsize=14)

        for ax_idx, csv_path in enumerate(csv_files):
            curves_path = csv_path.with_suffix(f".icl_mean_curves.{tag}.json")
            if not curves_path.exists():
                continue

            rows = load_csv_with_ids(csv_path)
            curves = load_mean_curves(curves_path)
            task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]
            ax = axes[ax_idx]

            high_curves_arr: list[list[float]] = []
            low_curves_arr: list[list[float]] = []
            for row in rows:
                sid = row["sample_id"]
                if sid not in curves:
                    continue
                a_k = curves[sid].get("a_k_curve")
                if a_k is None:
                    continue
                label = float(row["label_value"])
                if label == 1.0:
                    high_curves_arr.append(a_k)
                else:
                    low_curves_arr.append(a_k)

            for group_curves, group_label, color in [
                (high_curves_arr, "High diversity", "tab:red"),
                (low_curves_arr, "Low diversity", "tab:blue"),
            ]:
                if not group_curves:
                    continue

                arr = np.array(group_curves)
                mean_curve = np.mean(arr, axis=0)
                sem_curve = np.std(arr, axis=0) / np.sqrt(len(group_curves))
                n = len(mean_curve)
                k = np.arange(1, n + 1, dtype=float)

                # Plot mean ± SEM
                ax.plot(k, mean_curve, "o-", color=color, markersize=4, linewidth=2,
                        label=group_label)
                ax.fill_between(k, mean_curve - sem_curve, mean_curve + sem_curve,
                                alpha=0.2, color=color)

                # E_discrete
                a_n = mean_curve[-1]
                E_disc = sum(mean_curve[i] - a_n for i in range(n))

                # Exponential fit
                fit_params, fit_ok = fit_exponential(k, mean_curve)
                if fit_ok:
                    k_fine = np.linspace(0.5, n + 3, 100)
                    fitted = exponential_ak(k_fine, fit_params["a_inf"],
                                            fit_params["alpha"], fit_params["beta"])
                    ax.plot(k_fine, fitted, "--", color=color, linewidth=1.5,
                            label=f"  fit: E_disc={E_disc:.1f}, E_fit={fit_params['E_fit']:.1f}")
                else:
                    ax.plot([], [], " ",
                            label=f"  E_disc={E_disc:.1f}, fit FAILED")

            ax.set_xticks(range(1, n + 1))
            ax.set_title(task, fontweight="bold")
            ax.set_xlabel("k (response index)")
            ax.set_ylabel("a_k (total bits)")
            ax.legend(fontsize=8, loc="best")
            ax.axvline(n, color="gray", linestyle=":", alpha=0.3)

        plt.tight_layout()
        suffix = "with_hds" if "with_hds" in variant_glob else "no_hds"
        path = output_dir / f"mean_ak_with_fit_{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_efit_distributions(
    data_dir: Path, tag: str, output_dir: Path,
) -> None:
    """Plot 5: E_fit distributions for high vs low diversity."""
    nuggets_dir = data_dir / "McDiv_nuggets"
    if not nuggets_dir.exists():
        return

    csv_files = sorted(nuggets_dir.glob("*_with_hds_*.csv"))
    if not csv_files:
        return

    fig, axes = plt.subplots(2, len(csv_files), figsize=(6 * len(csv_files), 10))
    fig.suptitle(f"E_discrete vs E_fit Distributions: High vs Low (McDiv_nuggets) [{tag}]",
                 fontsize=14)

    e_col = f"metric_icl_E_{tag}"

    for col_idx, csv_path in enumerate(csv_files):
        curves_path = csv_path.with_suffix(f".icl_mean_curves.{tag}.json")
        if not curves_path.exists():
            continue

        rows = load_csv_with_ids(csv_path)
        curves = load_mean_curves(curves_path)
        task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]

        high_e_disc: list[float] = []
        low_e_disc: list[float] = []
        high_e_fit: list[float] = []
        low_e_fit: list[float] = []

        for row in rows:
            sid = row["sample_id"]
            if sid not in curves:
                continue
            a_k = curves[sid].get("a_k_curve")
            if a_k is None:
                continue

            e_disc = float(row.get(e_col, "nan"))
            if np.isnan(e_disc):
                continue

            k = np.arange(1, len(a_k) + 1, dtype=float)
            fit_params, fit_ok = fit_exponential(k, np.array(a_k))
            e_fit = fit_params["E_fit"] if fit_ok else e_disc

            label = float(row["label_value"])
            if label == 1.0:
                high_e_disc.append(e_disc)
                high_e_fit.append(e_fit)
            else:
                low_e_disc.append(e_disc)
                low_e_fit.append(e_fit)

        for row_idx, (high_vals, low_vals, metric_label) in enumerate([
            (high_e_disc, low_e_disc, "E_discrete"),
            (high_e_fit, low_e_fit, "E_fit_exp"),
        ]):
            ax = axes[row_idx][col_idx]
            if high_vals and low_vals:
                hist_range = (min(high_vals + low_vals), max(high_vals + low_vals))
                ax.hist(high_vals, bins=20, alpha=0.5, color="tab:red",
                        label=f"Diverse (n={len(high_vals)})", range=hist_range)
                ax.hist(low_vals, bins=20, alpha=0.5, color="tab:blue",
                        label=f"Constant (n={len(low_vals)})", range=hist_range)

                # Annotate means
                h_mean = np.mean(high_vals)
                l_mean = np.mean(low_vals)
                ax.axvline(h_mean, color="tab:red", linestyle="--", linewidth=2)
                ax.axvline(l_mean, color="tab:blue", linestyle="--", linewidth=2)
                ax.legend()

            ax.set_title(f"{metric_label} — {task}")
            ax.set_xlabel(f"{metric_label} (bits)")
            ax.set_ylabel("Count")

    plt.tight_layout()
    path = output_dir / "efit_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_sample_text(
    data_dir: Path, tag: str, output_dir: Path,
) -> None:
    """Plot 6: Print actual response text for a few samples."""
    nuggets_dir = data_dir / "McDiv_nuggets"
    if not nuggets_dir.exists():
        return

    csv_path = sorted(nuggets_dir.glob("*_with_hds_story_gen*.csv"))[0]
    rows = load_csv_with_ids(csv_path)

    resp_cols = sorted(
        [c for c in rows[0].keys() if c.startswith("resp_") and c.replace("resp_", "").isdigit()],
        key=lambda x: int(x.replace("resp_", "")),
    )

    output_lines: list[str] = []
    output_lines.append("=" * 80)
    output_lines.append("SAMPLE TEXT INSPECTION — McDiv_nuggets story_gen (with HDS)")
    output_lines.append("=" * 80)

    # Find one high-div and one low-div sample
    for target_label, label_name in [(1.0, "HIGH DIVERSITY"), (0.0, "LOW DIVERSITY")]:
        for row in rows:
            if float(row["label_value"]) == target_label:
                output_lines.append(f"\n{'─' * 60}")
                output_lines.append(f"  {label_name} — sample_id: {row['sample_id']}")
                output_lines.append(f"{'─' * 60}")
                output_lines.append(f"  Context: {row['context']}")
                for i, col in enumerate(resp_cols):
                    output_lines.append(f"  resp_{i}: {row[col]}")

                # Show what the completion format looks like
                context = row["context"]
                responses = [row[col] for col in resp_cols]
                output_lines.append(f"\n  --- Completion format (what the model sees) ---")
                for i, resp in enumerate(responses):
                    output_lines.append(f"  {i+1}. {context}{resp}")
                    if i < len(responses) - 1:
                        output_lines.append("")

                break

    text = "\n".join(output_lines)
    print(text)

    # Also save to file
    path = output_dir / "sample_text_inspection.txt"
    with open(path, "w") as f:
        f.write(text)
    print(f"\nSaved: {path}")


def plot_dectest_efit_vs_temperature(
    data_dir: Path, tag: str, output_dir: Path,
) -> None:
    """Plot 7: E and D (discrete + fit) vs temperature for DecTest."""
    dectest_dir = data_dir / "decTest"
    if not dectest_dir.exists():
        return

    e_col = f"metric_icl_E_{tag}"
    c_col = f"metric_icl_C_{tag}"

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"DecTest: E & D vs Temperature [{tag}]", fontsize=14)

    for ax_idx, csv_path in enumerate(sorted(dectest_dir.glob("*1000*.csv"))[:3]):
        curves_path = csv_path.with_suffix(f".icl_mean_curves.{tag}.json")
        if not curves_path.exists():
            continue

        rows = load_csv_with_ids(csv_path)
        curves = load_mean_curves(curves_path)
        task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]

        temps: list[float] = []
        e_discs: list[float] = []
        e_fits: list[float] = []
        d_discs: list[float] = []
        d_fits: list[float] = []

        for row in rows:
            sid = row["sample_id"]
            if sid not in curves:
                continue
            a_k = curves[sid].get("a_k_curve")
            if a_k is None:
                continue

            e_disc = float(row.get(e_col, "nan"))
            c_val = float(row.get(c_col, "nan"))
            if np.isnan(e_disc) or np.isnan(c_val):
                continue

            k = np.arange(1, len(a_k) + 1, dtype=float)
            fit_params, fit_ok = fit_exponential(k, np.array(a_k))
            e_fit = fit_params["E_fit"] if fit_ok else e_disc

            temps.append(float(row["label_value"]))
            e_discs.append(e_disc)
            e_fits.append(e_fit)
            d_discs.append(c_val * e_disc)
            d_fits.append(c_val * e_fit)

        if not temps:
            continue

        row_configs = [
            (0, e_discs, "E_discrete", "E_discrete (bits)", "tab:blue"),
            (1, e_fits, "E_fit_exp", "E_fit_exp (bits)", "tab:red"),
            (2, d_discs, "D_discrete", "D_discrete = C × E_disc", "tab:blue"),
            (3, d_fits, "D_fit_exp", "D_fit = C × E_fit", "tab:red"),
        ]

        for row_idx, values, metric_name, ylabel, color in row_configs:
            ax = axes[row_idx][ax_idx]
            ax.scatter(temps, values, alpha=0.3, s=10, color=color)
            rho, _ = spearmanr(temps, values)
            ax.set_title(f"{metric_name} — {task}\nρ = {rho:.3f}")
            ax.set_xlabel("Temperature")
            ax.set_ylabel(ylabel)

    plt.tight_layout()
    path = output_dir / "dectest_metrics_vs_temperature.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual inspection of Tevet results")
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

    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}\n")

    print("=" * 60)
    print("2. Individual a_k curves")
    print("=" * 60)
    plot_individual_ak_curves(data_dir, args.run_tag, output_dir)

    print("\n" + "=" * 60)
    print("3. Mean a_k curves with exponential fit")
    print("=" * 60)
    plot_mean_ak_with_fit(data_dir, args.run_tag, output_dir)

    print("\n" + "=" * 60)
    print("5. E_fit distributions")
    print("=" * 60)
    plot_efit_distributions(data_dir, args.run_tag, output_dir)

    print("\n" + "=" * 60)
    print("6. Sample text inspection")
    print("=" * 60)
    print_sample_text(data_dir, args.run_tag, output_dir)

    print("\n" + "=" * 60)
    print("7. DecTest E_fit vs temperature")
    print("=" * 60)
    plot_dectest_efit_vs_temperature(data_dir, args.run_tag, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
