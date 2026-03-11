"""Violin + strip summary of per-token surprise reduction distribution.

For each mode pair, shows the full distribution of per-token mean deltas
as a violin plot, with individual tokens overlaid as dots colored by their
relative position (dark=early, light=late). This replaces the arbitrary
25%-cutoff bar chart.

Reads from token_attribution.json (output of 08_token_attribution.py).
Generates plots for both Qwen2.5-3B and GPT-2 if available.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent / "figures"

DATASETS = [
    ("Qwen2.5-3B", FIGURES_DIR / "token_attribution.json", FIGURES_DIR),
    ("GPT-2", FIGURES_DIR / "gpt2" / "token_attribution.json", FIGURES_DIR / "gpt2"),
]


def plot_summary(model_label: str, json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        results = json.load(f)

    n_pairs = len(results)
    fig, ax = plt.subplots(figsize=(max(12, n_pairs * 1.4), 7))

    pair_labels = []
    positions = []
    all_deltas = []  # list of arrays, one per pair
    total_deltas = []

    for idx, res in enumerate(results):
        label = f"{res['context_mode']}\u2192{res['target_mode']}"
        if res.get("is_same_mode"):
            label += " (same)"
        pair_labels.append(label)
        positions.append(idx)

        deltas = np.array(res["per_token_delta_mean"])
        all_deltas.append(deltas)
        total_deltas.append(res["total_delta_bits"])

    # Violin plots
    parts = ax.violinplot(
        all_deltas,
        positions=positions,
        showextrema=False,
        showmedians=False,
        widths=0.7,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.3)

    # Overlay individual tokens as dots, colored by relative position
    cmap = plt.cm.plasma
    for idx, deltas in enumerate(all_deltas):
        n_tok = len(deltas)
        if n_tok == 0:
            continue
        # Relative position: 0 = first token, 1 = last token
        rel_pos = np.arange(n_tok) / max(n_tok - 1, 1)
        colors = cmap(rel_pos)

        # Jitter x positions for visibility
        rng = np.random.default_rng(42 + idx)
        jitter = rng.uniform(-0.15, 0.15, size=n_tok)

        ax.scatter(
            np.full(n_tok, idx) + jitter,
            deltas,
            c=colors,
            s=12,
            alpha=0.7,
            edgecolors="none",
            zorder=3,
        )

    # Median markers
    for idx, deltas in enumerate(all_deltas):
        if len(deltas) > 0:
            med = np.median(deltas)
            ax.plot(idx, med, "k_", markersize=12, markeredgewidth=2, zorder=4)

    ax.axhline(0, color="black", linewidth=1)

    # Annotate with total delta
    for idx, td in enumerate(total_deltas):
        y_top = max(all_deltas[idx]) if len(all_deltas[idx]) > 0 else 0
        ax.text(
            idx,
            y_top + 0.5,
            f"\u0394={td:+.0f}",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-token surprise reduction (bits)", fontsize=11)
    ax.set_title(
        f"{model_label}: Distribution of Per-Token Surprise Reduction by Mode Pair",
        fontsize=13,
        fontweight="bold",
    )

    # Colorbar for token position
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Relative token position (0=first, 1=last)", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "token_attribution_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for model_label, json_path, out_dir in DATASETS:
        if json_path.exists():
            plot_summary(model_label, json_path, out_dir)
        else:
            print(f"Skipping {model_label}: {json_path} not found")
