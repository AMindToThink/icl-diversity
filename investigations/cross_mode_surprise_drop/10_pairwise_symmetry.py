"""Plot pairwise symmetry: reduction[i,j] vs reduction[j,i] for each mode pair.

Instead of averaging into row/column means, plot every (i,j) pair as a single
point. If the matrix were symmetric, all points would lie on y=x.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent / "figures"

DATASETS = [
    ("Qwen2.5-3B", FIGURES_DIR / "pairwise_matrix.json", FIGURES_DIR),
    ("GPT-2", FIGURES_DIR / "gpt2" / "pairwise_matrix.json", FIGURES_DIR / "gpt2"),
]


def plot_symmetry(model_label: str, json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    red = np.array(data["reduction_mean"])
    names = data["mode_names"]
    n = red.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Each off-diagonal pair (i,j) with i<j becomes one point
    xs, ys, labels = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            xs.append(red[i, j])
            ys.append(red[j, i])
            labels.append(f"{names[i]}\u2194{names[j]}")

    xs, ys = np.array(xs), np.array(ys)
    ax.scatter(xs, ys, s=30, alpha=0.7, zorder=5, color="steelblue")

    # Label outliers: top 10 by distance from y=x
    dist = np.abs(xs - ys)
    top_outliers = set(np.argsort(dist)[-10:].tolist())

    # Also label points with largest magnitude
    mag = np.maximum(np.abs(xs), np.abs(ys))
    top_mag = set(np.argsort(mag)[-5:].tolist())

    for idx in top_outliers | top_mag:
        ax.annotate(
            labels[idx],
            (xs[idx], ys[idx]),
            fontsize=5.5,
            textcoords="offset points",
            xytext=(4, 4),
        )

    # Fit line
    coeffs = np.polyfit(xs, ys, 1)
    r2 = np.corrcoef(xs, ys)[0, 1] ** 2
    fit_x = np.linspace(
        min(xs.min(), ys.min()) - 2, max(xs.max(), ys.max()) + 2, 100
    )
    fit_y = np.polyval(coeffs, fit_x)
    ax.plot(
        fit_x,
        fit_y,
        "r--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Fit: y = {coeffs[0]:.2f}x {coeffs[1]:+.1f}  (R\u00b2={r2:.2f})",
    )

    # y=x reference line
    lims = [min(xs.min(), ys.min()) - 3, max(xs.max(), ys.max()) + 3]
    ax.plot(lims, lims, "k:", linewidth=0.8, alpha=0.5, label="y = x (perfect symmetry)")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(
        "reduction[i, j]: seeing mode i reduces surprise for mode j (bits)",
        fontsize=11,
    )
    ax.set_ylabel(
        "reduction[j, i]: seeing mode j reduces surprise for mode i (bits)",
        fontsize=11,
    )
    ax.set_title(
        f"{model_label}: Pairwise Symmetry \u2014 Each Point is One (i,j) Pair",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    out_path = out_dir / "pairwise_symmetry.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    print(f"  {model_label}: slope={coeffs[0]:.2f}, intercept={coeffs[1]:.1f}, R\u00b2={r2:.2f}")
    print(f"  n_pairs={len(xs)}, mean |asymmetry|={np.abs(xs - ys).mean():.1f} bits")


if __name__ == "__main__":
    for model_label, json_path, out_dir in DATASETS:
        if json_path.exists():
            plot_symmetry(model_label, json_path, out_dir)
        else:
            print(f"Skipping {model_label}: {json_path} not found")
