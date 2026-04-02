"""Cross-model comparison of pairwise symmetry metrics.

Loads pairwise_matrix.json from each model's output directory and produces:
1. Summary table of all metrics
2. Grid of symmetry scatters (one per model)
3. Metrics vs model size plots
4. Hypothesis test results

Usage:
    uv run python investigations/cross_mode_surprise_drop/12_scaling_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR = FIGURES_DIR / "scaling_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models in size order, with their data directories
MODELS = [
    ("GPT-2 (124M)", FIGURES_DIR / "gpt2", 0.124),
    ("Llama-3.2-1B", FIGURES_DIR / "llama_3_2_1b", 1.0),
    ("Llama-3.2-3B", FIGURES_DIR / "llama_3_2_3b", 3.0),
    ("Qwen2.5-3B", FIGURES_DIR, 3.0),  # original Qwen data is in figures/ root
    ("Llama-3.1-8B", FIGURES_DIR / "llama_3_1_8b", 8.0),
    ("Llama-3.1-70B", FIGURES_DIR / "llama_3_1_70b", 70.0),
]

# Llama-only subset for hypothesis testing
LLAMA_MODELS = [m for m in MODELS if "Llama" in m[0]]


def load_model_data(label: str, data_dir: Path) -> dict | None:
    json_path = data_dir / "pairwise_matrix.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found, skipping {label}")
        return None
    with open(json_path) as f:
        data = json.load(f)

    red = np.array(data["reduction_mean"])
    n = red.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)

    # Symmetry scatter data
    xs, ys = [], []
    for i in range(n):
        for j in range(i + 1, n):
            xs.append(red[i, j])
            ys.append(red[j, i])
    xs, ys = np.array(xs), np.array(ys)

    coeffs = np.polyfit(xs, ys, 1)
    r2 = np.corrcoef(xs, ys)[0, 1] ** 2

    off_diag_vals = red[off_diag_mask]

    # Bootstrap R² and slope CIs
    n_boot = 1000
    rng = np.random.RandomState(42)
    boot_r2s = []
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        bx, by = xs[idx], ys[idx]
        bc = np.polyfit(bx, by, 1)
        br2 = np.corrcoef(bx, by)[0, 1] ** 2
        boot_r2s.append(br2)
        boot_slopes.append(bc[0])

    return {
        "label": label,
        "diagonal_mean": float(np.diag(red).mean()),
        "off_diagonal_mean": float(off_diag_vals.mean()),
        "off_diagonal_frac_pos": float((off_diag_vals > 0).mean()),
        "r2": float(r2),
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "r2_ci": (float(np.percentile(boot_r2s, 2.5)), float(np.percentile(boot_r2s, 97.5))),
        "slope_ci": (float(np.percentile(boot_slopes, 2.5)), float(np.percentile(boot_slopes, 97.5))),
        "xs": xs,
        "ys": ys,
        "off_diag_vals": off_diag_vals,
        "n_off_diag": len(off_diag_vals),
        # t-test: is off-diagonal mean significantly different from 0?
        "off_diag_tstat": float(scipy_stats.ttest_1samp(off_diag_vals, 0).statistic),
        "off_diag_pval": float(scipy_stats.ttest_1samp(off_diag_vals, 0).pvalue),
    }


def main() -> None:
    print("Loading model data...")
    all_data: list[tuple[str, float, dict]] = []
    for label, data_dir, size in MODELS:
        d = load_model_data(label, data_dir)
        if d is not None:
            all_data.append((label, size, d))

    # --- Summary table ---
    print("\n" + "=" * 110)
    print(f"{'Model':<20s} {'Size':>6s} {'Diag':>8s} {'Off-diag':>9s} "
          f"{'Frac>0':>7s} {'R²':>8s} {'R² 95% CI':>16s} "
          f"{'Slope':>7s} {'Slope 95% CI':>16s} {'t-stat':>7s} {'p':>8s}")
    print("-" * 110)
    for label, size, d in all_data:
        print(
            f"{label:<20s} {size:>5.1f}B {d['diagonal_mean']:>7.1f} "
            f"{d['off_diagonal_mean']:>+8.1f} {d['off_diagonal_frac_pos']:>6.1%} "
            f"{d['r2']:>7.3f} [{d['r2_ci'][0]:.3f}, {d['r2_ci'][1]:.3f}] "
            f"{d['slope']:>6.3f} [{d['slope_ci'][0]:.3f}, {d['slope_ci'][1]:.3f}] "
            f"{d['off_diag_tstat']:>6.2f} {d['off_diag_pval']:>7.4f}"
        )
    print("=" * 110)

    # --- Hypothesis tests (Llama only) ---
    llama_data = [(l, s, d) for l, s, d in all_data if "Llama" in l]
    if len(llama_data) == 4:
        print("\n=== Hypothesis Tests (Llama family) ===")
        # Order by size
        llama_data.sort(key=lambda x: x[1])

        hypotheses = [
            ("H1: 70B > 8B (same gen 3.1)", 3, 2),
            ("H2: 3B > 1B (same gen 3.2)", 1, 0),
            ("H3: 8B > 3B (cross-gen)", 2, 1),
        ]

        for h_name, idx_a, idx_b in hypotheses:
            a_label, a_size, a = llama_data[idx_a]
            b_label, b_size, b = llama_data[idx_b]
            print(f"\n{h_name}  ({a_label} vs {b_label})")
            for metric in ["off_diagonal_mean", "off_diagonal_frac_pos", "r2", "slope"]:
                va, vb = a[metric], b[metric]
                direction = ">" if va > vb else "<"
                print(f"  {metric:25s}: {va:.3f} {direction} {vb:.3f}  ({'supports' if va > vb else 'CONTRADICTS'})")

        # Monotonicity check
        print("\n=== Monotonicity Check (ordered by size) ===")
        for metric in ["off_diagonal_mean", "off_diagonal_frac_pos", "r2", "slope"]:
            vals = [d[metric] for _, _, d in llama_data]
            is_mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
            print(f"  {metric:25s}: {[f'{v:.3f}' for v in vals]}  {'MONOTONE' if is_mono else 'not monotone'}")

    # --- Grid of symmetry scatters ---
    n_models = len(all_data)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (label, size, d) in enumerate(all_data):
        ax = axes[idx]
        ax.scatter(d["xs"], d["ys"], s=20, alpha=0.6, color="steelblue")

        # Fit line
        fit_x = np.linspace(
            min(d["xs"].min(), d["ys"].min()) - 2,
            max(d["xs"].max(), d["ys"].max()) + 2,
            100,
        )
        fit_y = d["slope"] * fit_x + d["intercept"]
        ax.plot(fit_x, fit_y, "r--", linewidth=1.5, alpha=0.7)

        # y=x reference
        lims = [min(d["xs"].min(), d["ys"].min()) - 3, max(d["xs"].max(), d["ys"].max()) + 3]
        ax.plot(lims, lims, "k:", linewidth=0.8, alpha=0.4)

        ax.set_title(
            f"{label}\n"
            f"R\u00b2={d['r2']:.2f}, slope={d['slope']:.2f}, "
            f"off-diag={d['off_diagonal_mean']:+.1f}",
            fontsize=10,
        )
        ax.set_xlabel("reduction[i,j] (bits)")
        ax.set_ylabel("reduction[j,i] (bits)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "symmetry_grid.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved symmetry grid to {fig_path}")
    plt.close()

    # --- Metrics vs size ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = [
        ("off_diagonal_mean", "Off-diagonal mean (bits)", axes[0, 0]),
        ("off_diagonal_frac_pos", "Fraction off-diagonal > 0", axes[0, 1]),
        ("r2", "Symmetry R\u00b2", axes[1, 0]),
        ("slope", "Symmetry slope", axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics_to_plot:
        sizes = [s for _, s, _ in all_data]
        vals = [d[metric] for _, _, d in all_data]
        labels = [l for l, _, _ in all_data]

        ax.scatter(sizes, vals, s=60, zorder=5, color="steelblue")
        for i, label in enumerate(labels):
            ax.annotate(
                label, (sizes[i], vals[i]),
                fontsize=7, textcoords="offset points", xytext=(5, 5),
            )
        ax.set_xlabel("Model size (B params)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        if metric == "slope":
            ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="ideal (slope=1)")
            ax.legend(fontsize=8)
        if metric == "off_diagonal_mean":
            ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5, label="zero (independent modes)")
            ax.legend(fontsize=8)

    plt.suptitle("Cross-mode symmetry metrics vs model size", fontsize=13)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "metrics_vs_size.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved metrics vs size to {fig_path}")
    plt.close()

    # Save summary as JSON
    summary = {
        "models": [
            {
                "label": label,
                "size_B": size,
                "diagonal_mean": d["diagonal_mean"],
                "off_diagonal_mean": d["off_diagonal_mean"],
                "off_diagonal_frac_pos": d["off_diagonal_frac_pos"],
                "r2": d["r2"],
                "r2_ci": d["r2_ci"],
                "slope": d["slope"],
                "slope_ci": d["slope_ci"],
                "off_diag_tstat": d["off_diag_tstat"],
                "off_diag_pval": d["off_diag_pval"],
            }
            for label, size, d in all_data
        ]
    }
    json_path = OUTPUT_DIR / "scaling_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
