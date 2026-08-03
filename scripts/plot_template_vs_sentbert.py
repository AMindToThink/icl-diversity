"""Analyze and plot template/POS-pattern vs baseline-metric experiments.

Works for both experiment JSONs (auto-detects the sweep condition prefix):

- scripts/run_template_vs_sentbert.py    -> conditions frames_m (+ canonical,
  paraphrase)
- scripts/run_pos_pattern_vs_baselines.py -> conditions patterns_m
  (+ canonical; no paraphrase anchor), or the sweep-free
  canonical-vs-scrambled control run (--pattern-counts with no values
  --include-scrambled), where fig1 and the Spearman section are skipped.

Writes to --output-dir:

- fig1_d_vs_sentbert.png: three panels sharing x = sweep size m:
  D = C * a_n (diversity_score_D_C_an), SentBERT diversity (negated mean
  pairwise cosine), and averaged distinct-n; mean +/- SD across draws.
  The canonical condition appears as a separate marker at m=1 and, when a
  paraphrase condition exists, its mean as a horizontal dashed line.
- fig2_ak_curves.png: mean per-byte a_k curves by condition.
- summary.txt: per-condition table (mean +/- SD), in-context drop ratios,
  normalized positions (only when a paraphrase anchor exists), and Spearman
  correlations between m and each metric across the sweep runs.

Usage:
    uv run python scripts/plot_template_vs_sentbert.py \
        --input results/template_vs_sentbert/gpt2.json \
        --output-dir figures/template_vs_sentbert/gpt2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, ttest_ind

# Colors validated with the dataviz palette checker (blue/orange pair passes
# CVD separation; orange's low surface contrast is relieved by direct labels).
SWEEP_COLOR = "#1f77b4"
CANONICAL_COLOR = "#ff7f0e"
ANCHOR_COLOR = "#666666"
SCRAMBLED_COLOR = "#444444"

METRIC_KEYS = [
    "coherence_C",
    "a_n_per_byte",
    "diversity_score_D_C_an",
    "excess_entropy_E",
    "sentbert_mean_pairwise_cosine",
    "sentbert_diversity",
    "averaged_distinct_ngrams",
]

SWEEP_XLABELS = {
    "frames": "number of syntactic frames m",
    "patterns": "number of POS patterns m",
}

_SWEEP_NAME_RE = re.compile(r"^([a-z]+)_(\d+)$")


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


def sweep_conditions(
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, list[tuple[int, str]]]:
    """Detect the sweep prefix and return it with sorted (m, condition_name).

    Returns (None, []) for sweep-free runs (e.g. canonical vs scrambled);
    more than one sweep prefix is still an error.
    """
    by_prefix: dict[str, list[tuple[int, str]]] = {}
    for name in grouped:
        match = _SWEEP_NAME_RE.match(name)
        if match:
            by_prefix.setdefault(match.group(1), []).append((int(match.group(2)), name))
    if len(by_prefix) > 1:
        raise ValueError(
            f"Expected at most one sweep condition prefix, found: "
            f"{sorted(by_prefix)} among {sorted(grouped)}"
        )
    if not by_prefix:
        return None, []
    prefix, pairs = next(iter(by_prefix.items()))
    return prefix, sorted(pairs)


def plot_metric_panel(
    ax: plt.Axes,
    grouped: dict[str, list[dict[str, Any]]],
    key: str,
    ylabel: str,
) -> None:
    prefix, sweep = sweep_conditions(grouped)
    if not sweep:
        raise ValueError("plot_metric_panel requires a sweep; none found")
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
        label=f"m random {prefix}",
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
            label="canonical (m=1)",
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

    ax.set_xlabel(SWEEP_XLABELS.get(prefix, f"number of {prefix} m"))
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    style_axes(ax)


def plot_fig1(
    grouped: dict[str, list[dict[str, Any]]],
    data: dict[str, Any],
    out: Path,
) -> None:
    fig, (ax_d, ax_sb, ax_dn) = plt.subplots(1, 3, figsize=(14, 4))
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
    plot_metric_panel(
        ax_dn,
        grouped,
        "averaged_distinct_ngrams",
        "averaged distinct-n  (n = 1..5)",
    )
    ax_d.set_title("ICL diversity D = C x a_n")
    ax_sb.set_title("SentBERT diversity")
    ax_dn.set_title("Averaged distinct-n")
    ax_d.legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"{data['experiment']}  ({data['base_model']}, "
        f"n={data['n_responses']} responses, {data['n_draws']} draws)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_fig2(
    grouped: dict[str, list[dict[str, Any]]],
    data: dict[str, Any],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _prefix, sweep = sweep_conditions(grouped)
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
            label=f"m={m}",
        )
    for name, color, style, label in [
        ("canonical", CANONICAL_COLOR, "-", "canonical"),
        ("scrambled", SCRAMBLED_COLOR, ":", "scrambled control"),
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
        f"Progressive conditional surprise under {data['base_model']}\n"
        f"({data['experiment']})"
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
    prefix, sweep = sweep_conditions(grouped)
    lines: list[str] = []
    lines.append(f"{data['experiment']} experiment summary")
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
    order += [name for _, name in sweep]
    order += ["scrambled", "paraphrase"]

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
        "Per-byte a_k curve endpoints, in-context drop ratio a_n / a_1 "
        "(final conditional surprise as a fraction of the first response's; "
        "lower = more structure learned in-context), and mean response "
        "length in utf-8 bytes:"
    )
    for name in order:
        if name not in grouped:
            continue
        a1 = np.array([r["a_k_curve_per_byte"][0] for r in grouped[name]])
        an = np.array([r["a_k_curve_per_byte"][-1] for r in grouped[name]])
        ratio = an / a1
        nbytes = np.array(
            [
                np.mean([len(resp.encode("utf-8")) for resp in r["responses"]])
                for r in grouped[name]
            ]
        )
        lines.append(
            f"  {name:<14s} a_1 = {a1.mean():.3f} ± {a1.std(ddof=1):.3f}   "
            f"a_n = {an.mean():.3f} ± {an.std(ddof=1):.3f}   "
            f"a_n/a_1 = {ratio.mean():.3f} ± {ratio.std(ddof=1):.3f}   "
            f"bytes/resp = {nbytes.mean():.1f} ± {nbytes.std(ddof=1):.1f}"
        )
    lines.append("")

    if "canonical" in grouped and "scrambled" in grouped:
        lines.append(
            "Canonical vs scrambled (Welch two-sided t-test on per-draw "
            "values; diff = canonical - scrambled, mean ± SE):"
        )
        per_draw: list[tuple[str, Any]] = [
            ("a_1_per_byte", lambda r: r["a_k_curve_per_byte"][0]),
            ("a_n_per_byte", lambda r: r["a_k_curve_per_byte"][-1]),
            (
                "drop_ratio_a_n_over_a_1",
                lambda r: r["a_k_curve_per_byte"][-1] / r["a_k_curve_per_byte"][0],
            ),
            ("coherence_C", lambda r: r["coherence_C"]),
            ("diversity_score_D_C_an", lambda r: r["diversity_score_D_C_an"]),
            (
                "sentbert_mean_pairwise_cosine",
                lambda r: r["sentbert_mean_pairwise_cosine"],
            ),
            ("averaged_distinct_ngrams", lambda r: r["averaged_distinct_ngrams"]),
        ]
        for label, fn in per_draw:
            a_arr = np.array([fn(r) for r in grouped["canonical"]])
            b_arr = np.array([fn(r) for r in grouped["scrambled"]])
            diff = float(a_arr.mean() - b_arr.mean())
            se = float(
                np.sqrt(a_arr.var(ddof=1) / len(a_arr) + b_arr.var(ddof=1) / len(b_arr))
            )
            t, p = ttest_ind(a_arr, b_arr, equal_var=False)
            lines.append(
                f"  {label:<30s} diff = {diff:+.4g} ± {se:.2g}   "
                f"t = {t:+.2f}  (p = {p:.2g})"
            )
        lines.append("")

    if "paraphrase" in grouped:
        top = f"{prefix}_{max(m for m, _ in sweep)}"
        lines.append(
            f"Normalized position on the paraphrase -> {top} scale, "
            f"(mean_cond - mean_paraphrase) / (mean_{top} - mean_paraphrase):"
        )
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
            for name in ["canonical", f"{prefix}_1"]:
                if name in grouped:
                    c_mean, _ = cond_stats(grouped[name], key)
                    parts.append(f"{name} = {(c_mean - p_mean) / span:+.2f}")
            lines.append(f"  {key:<28s} {'  '.join(parts)}")
        lines.append("")

    if sweep:
        lines.append(f"Spearman rho between m and metric ({prefix}_m sweep runs):")
        sweep_runs = [(m, run) for m, name in sweep for run in grouped[name]]
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
    parser = argparse.ArgumentParser(
        description="Plot template/POS-pattern vs baseline-metric results"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    grouped = group_runs(data["runs"])
    _prefix, sweep = sweep_conditions(grouped)  # fail fast on ambiguous sweep

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sweep:
        plot_fig1(grouped, data, args.output_dir / "fig1_d_vs_sentbert.png")
    plot_fig2(grouped, data, args.output_dir / "fig2_ak_curves.png")
    write_summary(data, grouped, args.input, args.output_dir / "summary.txt")
    print(f"\nFigures and summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
