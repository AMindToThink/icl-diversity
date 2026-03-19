"""Plot individual permutation a_k curves at T=1.0 vs T=2.0 to visualize variance reduction.

Side-by-side panels showing all ~100 permutation curves (thin, transparent)
with the mean overlaid (thick). Same y-axis scale so the visual spread is
directly comparable.

Usage:
    uv run python scripts/plot_permutation_curves.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "temperature_experiments"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "temperature"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    temps = [1.0, 2.0]
    data = {}
    for t in temps:
        path = RESULTS_DIR / f"T_{t}.json"
        with open(path) as f:
            data[t] = json.load(f)

    # Extract per-permutation curves for multi_mode, first prompt
    curves_by_t: dict[float, np.ndarray] = {}
    for t in temps:
        entry = data[t]["scenarios"]["multi_mode"][0]
        per_perm = entry["per_permutation_a_k_curves"]
        if per_perm is None:
            raise ValueError(f"No per_permutation_a_k_curves at T={t}")
        curves_by_t[t] = np.array(per_perm)

    # Shared y-axis limits
    all_vals = np.concatenate([c.ravel() for c in curves_by_t.values()])
    y_min = float(np.min(all_vals)) * 0.9
    y_max = float(np.max(all_vals)) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, t in zip(axes, temps):
        curves = curves_by_t[t]
        n_perms, n_resp = curves.shape
        k = np.arange(1, n_resp + 1)

        # Individual permutation curves
        for i in range(n_perms):
            ax.plot(k, curves[i], color="steelblue", alpha=0.12, linewidth=0.8)

        # Mean curve
        mean_curve = np.mean(curves, axis=0)
        ax.plot(k, mean_curve, color="darkred", linewidth=2.5, label="mean")

        ax.set_xlabel("Response index k")
        ax.set_title(f"T = {t}  ({n_perms} permutations)", fontsize=13, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[0].set_ylabel("$a_k$ (total bits)")
    fig.suptitle("multi_mode: Per-Permutation a_k Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = FIGURES_DIR / "permutation_curves_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
