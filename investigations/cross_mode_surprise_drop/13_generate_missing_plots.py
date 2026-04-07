"""Generate missing per-model figures from cached pairwise_matrix.json files.

Reads each model's pairwise_matrix.json and generates pairwise_row_vs_col.png
(row mean vs column mean scatter, testing mode-level symmetry).

No API calls or GPU needed — reads only cached data.

Usage:
    uv run python investigations/cross_mode_surprise_drop/13_generate_missing_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent / "figures"

MODELS = [
    ("llama_3_2_1b", "meta-llama/Llama-3.2-1B"),
    ("llama_3_2_3b", "meta-llama/Llama-3.2-3B"),
    ("llama_3_1_8b", "meta-llama/Llama-3.1-8B"),
    ("llama_3_1_70b", "meta-llama/Llama-3.1-70B"),
]


def generate_row_vs_col(data_dir: Path, model_label: str) -> None:
    json_path = data_dir / "pairwise_matrix.json"
    if not json_path.exists():
        print(f"  SKIP: {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    red = np.array(data["reduction_mean"])
    std = np.array(data["reduction_std"])
    names = data["mode_names"]
    n = red.shape[0]
    m = data["m_samples"]

    sem = std / np.sqrt(m)
    row_means = red.mean(axis=1)
    col_means = red.mean(axis=0)
    row_sems = np.sqrt((sem**2).sum(axis=1)) / n
    col_sems = np.sqrt((sem**2).sum(axis=0)) / n

    rc_coeffs = np.polyfit(row_means, col_means, 1)
    rc_r2 = np.corrcoef(row_means, col_means)[0, 1] ** 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(
        row_means,
        col_means,
        xerr=row_sems,
        yerr=col_sems,
        fmt="o",
        markersize=6,
        alpha=0.7,
        color="steelblue",
        capsize=3,
    )
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (row_means[i], col_means[i]),
            fontsize=6.5,
            textcoords="offset points",
            xytext=(4, 4),
        )

    fit_x = np.linspace(row_means.min() - 1, row_means.max() + 1, 100)
    fit_y = np.polyval(rc_coeffs, fit_x)
    ax.plot(
        fit_x,
        fit_y,
        "r--",
        linewidth=1.5,
        alpha=0.7,
        label=f"y = {rc_coeffs[0]:.2f}x {rc_coeffs[1]:+.1f}  (R\u00b2={rc_r2:.2f})",
    )

    ax.set_xlabel("Row mean: how informative as context (bits)")
    ax.set_ylabel("Column mean: how much benefit from context (bits)")
    ax.set_title(f"{model_label}\nMode informativeness vs. context benefit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = data_dir / "pairwise_row_vs_col.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main() -> None:
    for tag, label in MODELS:
        data_dir = FIGURES_DIR / tag
        print(f"{label}:")
        generate_row_vs_col(data_dir, label)


if __name__ == "__main__":
    main()
