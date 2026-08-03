"""Analyze and plot template-vs-SentBERT experiment results.

Reads the JSON produced by scripts/run_template_vs_sentbert.py and writes:

- fig1_d_vs_sentbert.png: two panels sharing x = number of syntactic frames m.
  Left: D = C * a_n (diversity_score_D_C_an), mean +/- SD across draws.
  Right: SentBERT diversity (negated mean pairwise cosine), mean +/- SD.
  Both panels show the canonical-template condition as a separate marker at
  m=1 and the paraphrase anchor as a horizontal dashed line.
- fig2_ak_curves.png: mean per-byte a_k curves by condition.
- summary.txt: per-condition table (mean +/- SD) and Spearman correlations
  between m and each metric across the frame-sweep runs.

Usage:
    uv run python scripts/plot_template_vs_sentbert.py \
        --input results/template_vs_sentbert/gpt2.json \
        --output-dir figures/template_vs_sentbert/gpt2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Colors validated with the dataviz palette checker (blue/orange pair passes
# CVD separation; orange's low surface contrast is relieved by direct labels).
SWEEP_COLOR = "#1f77b4"
CANONICAL_COLOR = "#ff7f0e"
ANCHOR_COLOR = "#666666"

METRIC_KEYS = [
    "coherence_C",
    "a_n_per_byte",
    "diversity_score_D_C_an",
    "excess_entropy_E",
    "sentbert_mean_pairwise_cosine",
    "sentbert_diversity",
    "averaged_distinct_ngrams",
]


def group_runs(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["condition"], []).append(run)
    return grouped


def cond_stats(runs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = np.array([r[key] for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1))


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def sweep_conditions(grouped: dict[str, list[dict[str, Any]]]) -> list[tuple[int, str]]:
    """Sorted (m, condition_name) for the frames_m sweep."""
    pairs = [
        (int(name.split("_")[1]), name)
        for name in grouped
        if name.startswith("frames_")
    ]
    return sorted(pairs)


def plot_metric_panel(
    ax: plt.Axes,
    grouped: dict[str, list[dict[str, Any]]],
    key: str,
    ylabel: str,
) -> None:
    sweep = sweep_conditions(grouped)
    ms = [m for m, _ in sweep]
    means, sds = zip(*[cond_stats(grouped[name], key) for _, name in sweep])
    ax.errorbar(
        ms,
        means,
        yerr=sds,
        color=SWEEP_COLOR,
        marker="o",
        markersize=5,
        linewidth=2,
        capsize=3,
        label="m random frames",
    )

    if "canonical" in grouped:
        c_mean, c_sd = cond_stats(grouped["canonical"], key)
        ax.errorbar(
            [1],
            [c_mean],
            yerr=[c_sd],
            color=CANONICAL_COLOR,
            marker="D",
            markersize=7,
            linewidth=0,
            elinewidth=2,
            capsize=3,
            label="canonical template (m=1)",
            zorder=5,
        )

    if "paraphrase" in grouped:
        p_mean, _ = cond_stats(grouped["paraphrase"], key)
        ax.axhline(
            p_mean,
            color=ANCHOR_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="paraphrase anchor",
        )

    ax.set_xlabel("number of syntactic frames m")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    style_axes(ax)


def plot_fig1(
    grouped: dict[str, list[dict[str, Any]]], base_model: str, out: Path
) -> None:
    fig, (ax_d, ax_sb) = plt.subplots(1, 2, figsize=(10, 4))
    plot_metric_panel(
        ax_d,
        grouped,
        "diversity_score_D_C_an",
        "ICL diversity  D = C x a_n  (bits/byte)",
    )
    plot_metric_panel(
        ax_sb,
        grouped,
        "sentbert_diversity",
        "SentBERT diversity  (-mean pairwise cosine)",
    )
    ax_d.set_title("ICL diversity D = C x a_n increases with m")
    ax_sb.set_title("SentBERT rates every template set near its max")
    ax_d.legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"Same semantic scatter, varying syntactic frames ({base_model})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_fig2(
    grouped: dict[str, list[dict[str, Any]]], base_model: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sweep = sweep_conditions(grouped)
    cmap = plt.get_cmap("Blues")
    denom = max(len(sweep) - 1, 1)
    for i, (m, name) in enumerate(sweep):
        curves = np.array([r["a_k_curve_per_byte"] for r in grouped[name]])
        mean_curve = curves.mean(axis=0)
        ax.plot(
            np.arange(1, len(mean_curve) + 1),
            mean_curve,
            color=cmap(0.35 + 0.55 * i / denom),
            linewidth=2,
            label=f"m={m} frames",
        )
    for name, color, style, label in [
        ("canonical", CANONICAL_COLOR, "-", "canonical template"),
        ("paraphrase", ANCHOR_COLOR, "--", "paraphrase anchor"),
    ]:
        if name in grouped:
            curves = np.array([r["a_k_curve_per_byte"] for r in grouped[name]])
            mean_curve = curves.mean(axis=0)
            ax.plot(
                np.arange(1, len(mean_curve) + 1),
                mean_curve,
                color=color,
                linewidth=2,
                linestyle=style,
                label=label,
            )
    ax.set_xlabel("response index k")
    ax.set_ylabel("a_k (bits/byte), mean over draws")
    ax.set_title(
        f"Progressive conditional surprise under {base_model}:\n"
        "the base model learns repeated frames in-context"
    )
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_summary(
    data: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    input_path: Path,
    out: Path,
) -> None:
    lines: list[str] = []
    lines.append("Template-vs-SentBERT experiment summary")
    lines.append(f"Source: {input_path}")
    lines.append("Generated by: scripts/plot_template_vs_sentbert.py")
    lines.append(
        f"base_model={data['base_model']}  sentbert_model={data['sentbert_model']}"
    )
    lines.append(
        f"n_responses={data['n_responses']}  n_draws={data['n_draws']}  "
        f"seed={data['seed']}"
    )
    lines.append("")

    order = ["canonical"]
    order += [name for _, name in sweep_conditions(grouped)]
    order += ["paraphrase"]

    header = f"{'condition':<14s}" + "".join(f"{k:>31s}" for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for name in order:
        if name not in grouped:
            continue
        row = f"{name:<14s}"
        for key in METRIC_KEYS:
            mean, sd = cond_stats(grouped[name], key)
            row += f"{mean:>19.4g} ± {sd:<9.3g}"
        lines.append(row)
    lines.append("")

    lines.append(
        "In-context drop ratio a_n / a_1 (final conditional surprise as a "
        "fraction of the first response's, per-byte curve; lower = more "
        "structure learned in-context):"
    )
    for name in order:
        if name not in grouped:
            continue
        ratios = [
            r["a_k_curve_per_byte"][-1] / r["a_k_curve_per_byte"][0]
            for r in grouped[name]
        ]
        arr = np.array(ratios)
        lines.append(f"  {name:<14s} {arr.mean():.3f} ± {arr.std(ddof=1):.3f}")
    lines.append("")

    lines.append(
        "Normalized position on the paraphrase -> frames_20 scale, "
        "(mean_cond - mean_paraphrase) / (mean_frames_20 - mean_paraphrase):"
    )
    top = f"frames_{max(m for m, _ in sweep_conditions(grouped))}"
    for key in [
        "diversity_score_D_C_an",
        "sentbert_diversity",
        "averaged_distinct_ngrams",
    ]:
        p_mean, _ = cond_stats(grouped["paraphrase"], key)
        t_mean, _ = cond_stats(grouped[top], key)
        span = t_mean - p_mean
        if span == 0:
            raise ValueError(f"degenerate span for {key}: paraphrase == {top}")
        parts = []
        for name in ["canonical", "frames_1"]:
            if name in grouped:
                c_mean, _ = cond_stats(grouped[name], key)
                parts.append(f"{name} = {(c_mean - p_mean) / span:+.2f}")
        lines.append(f"  {key:<28s} {'  '.join(parts)}")
    lines.append("")

    lines.append("Spearman rho between m and metric (frames_m sweep runs):")
    sweep_runs = [
        (m, run) for m, name in sweep_conditions(grouped) for run in grouped[name]
    ]
    ms = [m for m, _ in sweep_runs]
    for key in [
        "diversity_score_D_C_an",
        "a_n_per_byte",
        "sentbert_diversity",
        "averaged_distinct_ngrams",
    ]:
        vals = [run[key] for _, run in sweep_runs]
        rho, p = spearmanr(ms, vals)
        lines.append(f"  {key:<28s} rho = {rho:+.3f}  (p = {p:.2g}, n = {len(ms)})")

    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot template-vs-SentBERT results")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    grouped = group_runs(data["runs"])
    if not any(name.startswith("frames_") for name in grouped):
        raise ValueError(f"No frames_m sweep conditions found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1(grouped, data["base_model"], args.output_dir / "fig1_d_vs_sentbert.png")
    plot_fig2(grouped, data["base_model"], args.output_dir / "fig2_ak_curves.png")
    write_summary(data, grouped, args.input, args.output_dir / "summary.txt")
    print(f"\nFigures and summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
