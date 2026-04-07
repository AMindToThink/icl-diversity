"""Generate paper-quality figure for cross-mode scaling with model size.

Two panels:
  Left: off-diagonal mean vs model size (with bootstrap 95% CI)
  Right: fraction of off-diagonal > 0 vs model size

Includes all 6 models (GPT-2, 4 Llama, Qwen2.5-3B).

Usage:
    uv run python investigations/cross_mode_surprise_drop/14_paper_scaling_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR = FIGURES_DIR / "scaling_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("GPT-2\n(124M)", FIGURES_DIR / "gpt2", 0.124, "dense"),
    ("Llama-3.2\n1B", FIGURES_DIR / "llama_3_2_1b", 1.0, "dense"),
    ("Llama-3.2\n3B", FIGURES_DIR / "llama_3_2_3b", 3.0, "dense"),
    ("Qwen2.5\n3B", FIGURES_DIR, 3.0, "dense"),
    ("Llama-3.1\n8B", FIGURES_DIR / "llama_3_1_8b", 8.0, "dense"),
    ("Llama-3.1\n70B", FIGURES_DIR / "llama_3_1_70b", 70.0, "dense"),
]


def load_off_diag(data_dir: Path) -> np.ndarray | None:
    json_path = data_dir / "pairwise_matrix.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    red = np.array(data["reduction_mean"])
    n = red.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return red[mask]


def bootstrap_ci(vals: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    means = [vals[rng.choice(len(vals), size=len(vals), replace=True)].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    sizes = []
    means = []
    ci_lo = []
    ci_hi = []
    fracs = []
    labels = []
    colors = []

    llama_color = "#2196F3"
    other_color = "#9E9E9E"

    for label, data_dir, size, arch in MODELS:
        vals = load_off_diag(data_dir)
        if vals is None:
            continue
        lo, hi = bootstrap_ci(vals)
        sizes.append(size)
        means.append(vals.mean())
        ci_lo.append(lo)
        ci_hi.append(hi)
        fracs.append((vals > 0).mean())
        labels.append(label)
        colors.append(llama_color if "Llama" in label else other_color)

    sizes = np.array(sizes)
    means = np.array(means)
    ci_lo = np.array(ci_lo)
    ci_hi = np.array(ci_hi)
    fracs = np.array(fracs)

    # Left panel: off-diagonal mean with CI
    for i in range(len(sizes)):
        ax1.errorbar(
            sizes[i], means[i],
            yerr=[[means[i] - ci_lo[i]], [ci_hi[i] - means[i]]],
            fmt="o", markersize=8, color=colors[i], capsize=4, capthick=1.5,
            zorder=5,
        )
        ax1.annotate(
            labels[i], (sizes[i], means[i]),
            fontsize=7, textcoords="offset points", xytext=(8, 0),
            va="center",
        )

    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Model size (B parameters)")
    ax1.set_ylabel("Off-diagonal mean (bits)")
    ax1.set_title("Cross-mode information transfer")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.2)

    # Annotate regions
    ax1.text(0.03, 0.97, "context helps \u2191", transform=ax1.transAxes,
             fontsize=8, va="top", color="green", alpha=0.6)
    ax1.text(0.03, 0.03, "context hurts \u2193", transform=ax1.transAxes,
             fontsize=8, va="bottom", color="red", alpha=0.6)

    # Right panel: fraction > 0
    for i in range(len(sizes)):
        ax2.scatter(sizes[i], fracs[i], s=60, color=colors[i], zorder=5)
        ax2.annotate(
            labels[i], (sizes[i], fracs[i]),
            fontsize=7, textcoords="offset points", xytext=(8, 0),
            va="center",
        )

    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Model size (B parameters)")
    ax2.set_ylabel("Fraction of mode pairs with\npositive cross-mode transfer")
    ax2.set_title("Prevalence of cross-mode learning")
    ax2.set_xscale("log")
    ax2.set_ylim(0.15, 0.75)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "paper_scaling_figure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Also save as PDF for LaTeX
    pdf_path = OUTPUT_DIR / "paper_scaling_figure.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    plt.close()


if __name__ == "__main__":
    main()
