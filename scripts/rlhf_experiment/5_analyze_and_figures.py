"""Analyze RLHF-diversity results: tests, figures, paper macros, TeX section.

Inputs:
    results/rlhf_experiment/icl_metrics.jsonl         # C × a_n etc. per (stage, set, prompt)
    results/rlhf_experiment/baseline_metrics.jsonl    # EAD, distinct-n, SentBERT

Outputs:
    results/rlhf_experiment/analysis.json             # machine-readable summary
    results/rlhf_experiment/tables/rlhf_diversity.tex # \input'd by paper
    results/rlhf_experiment/paper_macros_rlhf.tex     # script-generated \newcommands
    figures/rlhf_experiment/ak_curves_overlay.pdf
    figures/rlhf_experiment/per_prompt_D_violin.pdf
    figures/rlhf_experiment/metric_correlation_scatter.pdf

Statistical protocol (per Dror et al. 2018, "Hitchhiker's Guide to NLP sig"):
    H1 (confirmatory, one-sided paired Wilcoxon; Bonferroni α = 0.05/3):
        H1a: D_base > D_SFT
        H1b: D_SFT  > D_DPO
        H1c: D_base > D_RLVR
    H1' (exploratory, two-sided, uncorrected):
        D_DPO vs D_RLVR

Effect sizes: Cohen's d_z per Cohen 1988.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
FIG_DIR = REPO_ROOT / "figures" / "rlhf_experiment"
TABLE_DIR = RESULTS_DIR / "tables"

STAGES = ["base", "sft", "dpo", "instruct"]
STAGE_LABELS = {"base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct (RLVR)"}

# Pre-registered H1 contrasts: (a, b, direction) meaning H: D_a > D_b (one-sided).
# Names use only letters (no digits) so they're valid LaTeX \newcommand identifiers.
# HoneA = H1a (base > SFT), HoneB = H1b (SFT > DPO), HoneC = H1c (base > RLVR).
H1_CONTRASTS = [
    ("HoneA", "base", "sft",    "greater"),
    ("HoneB", "sft",  "dpo",    "greater"),
    ("HoneC", "base", "instruct", "greater"),
]
H1_PRIME = ("Hpa", "dpo", "instruct", "two-sided")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cohen_dz(diffs: list[float]) -> float:
    """Paired Cohen's d_z = mean(diff) / std(diff, ddof=1)."""
    import numpy as np

    d = np.asarray(diffs, dtype=float)
    if len(d) < 2:
        return float("nan")
    s = d.std(ddof=1)
    if s == 0:
        return float("nan")
    return float(d.mean() / s)


def wilcoxon_paired(
    x: list[float], y: list[float], alternative: str
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank via scipy."""
    from scipy.stats import wilcoxon

    assert len(x) == len(y)
    # Align: drop any prompt where both are NaN
    xs = []
    ys = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            continue
        xs.append(a)
        ys.append(b)
    if len(xs) < 5:
        return {"stat": float("nan"), "p": float("nan"), "n": len(xs)}
    # scipy's wilcoxon takes 'x' and 'y' and tests x - y; alternative maps directly
    res = wilcoxon(xs, ys, alternative=alternative, zero_method="wilcox")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "n": len(xs)}


def _derive_metric(rec: dict, key: str) -> float | None:
    """Return a scalar metric from an ICL / baseline row.

    Supports synthetic keys the underlying row doesn't store directly:
      - D_Can   = coherence_C * a_n_per_byte      (the paper's primary
                  C × a_n diversity score, in bits/byte)
      - a_n_per_byte = mean-of-ratios across permutations of the last
                      slot's per-byte surprise (the formula §6.3 gives).
                      Computed from ``per_permutation_*`` fields when
                      present; falls back to the precomputed
                      ``a_k_curve_per_byte[-1]`` (which is also MoR
                      after the core.py fix landed in this branch).
      - a_n_total    = a_k_curve[-1]
    Falls back to `rec.get(key)` for anything else.
    """
    if key == "D_Can":
        c = rec.get("coherence_C")
        an_pb = _derive_metric(rec, "a_n_per_byte")
        if c is None or an_pb is None:
            return None
        return float(c) * an_pb
    if key == "a_n_per_byte":
        # Prefer per-permutation MoR when available — that is what the
        # paper specifies (see src/icl_diversity/per_byte.py and the
        # implement-math skill).
        pp_curves = rec.get("per_permutation_a_k_curves")
        pp_bytes = rec.get("per_permutation_byte_counts")
        if pp_curves and pp_bytes:
            from icl_diversity.per_byte import compute_a_n_per_byte_mor
            try:
                return compute_a_n_per_byte_mor(pp_curves, pp_bytes)
            except ValueError:
                return None
        # Fallback: precomputed per-byte curve.  Records written by the
        # post-fix core.py contain MoR here; pre-fix records contain RoM.
        # Old logs flow through this branch.
        curve = rec.get("a_k_curve_per_byte")
        if not curve:
            return None
        return float(curve[-1])
    if key == "a_n_total":
        curve = rec.get("a_k_curve")
        if not curve:
            return None
        return float(curve[-1])
    v = rec.get(key)
    return float(v) if v is not None else None


def collect_metric(
    rows: list[dict], metric_key: str, prompt_set: str
) -> dict[str, dict[str, float]]:
    """Return {stage: {prompt_id: value}} for rows in the given prompt_set."""
    out: dict[str, dict[str, float]] = {s: {} for s in STAGES}
    for r in rows:
        if r.get("prompt_set") != prompt_set:
            continue
        s = r.get("stage")
        pid = r.get("prompt_id")
        v = _derive_metric(r, metric_key)
        if s in out and pid is not None and v is not None:
            out[s][pid] = float(v)
    return out


def paired_vectors(
    per_stage: dict[str, dict[str, float]], stage_a: str, stage_b: str
) -> tuple[list[float], list[float], list[str]]:
    ids = sorted(set(per_stage[stage_a].keys()) & set(per_stage[stage_b].keys()))
    xa = [per_stage[stage_a][i] for i in ids]
    xb = [per_stage[stage_b][i] for i in ids]
    return xa, xb, ids


def run_contrasts(
    per_stage: dict[str, dict[str, float]]
) -> dict[str, Any]:
    tests = {}
    for name, a, b, direction in H1_CONTRASTS:
        xa, xb, ids = paired_vectors(per_stage, a, b)
        res = wilcoxon_paired(xa, xb, alternative=direction)
        diffs = [x - y for x, y in zip(xa, xb)]
        tests[name] = {
            "a": a,
            "b": b,
            "alternative": direction,
            "stat": res["stat"],
            "p_raw": res["p"],
            "p_bonferroni": min(1.0, res["p"] * len(H1_CONTRASTS)) if res["p"] == res["p"] else float("nan"),
            "n": res["n"],
            "mean_a": sum(xa) / len(xa) if xa else float("nan"),
            "mean_b": sum(xb) / len(xb) if xb else float("nan"),
            "mean_diff": sum(diffs) / len(diffs) if diffs else float("nan"),
            "cohen_dz": cohen_dz(diffs),
        }
    # H1'
    name, a, b, direction = H1_PRIME
    xa, xb, _ = paired_vectors(per_stage, a, b)
    res = wilcoxon_paired(xa, xb, alternative=direction)
    diffs = [x - y for x, y in zip(xa, xb)]
    tests[name] = {
        "a": a,
        "b": b,
        "alternative": direction,
        "stat": res["stat"],
        "p_raw": res["p"],
        "p_bonferroni": None,  # uncorrected
        "n": res["n"],
        "mean_a": sum(xa) / len(xa) if xa else float("nan"),
        "mean_b": sum(xb) / len(xb) if xb else float("nan"),
        "mean_diff": sum(diffs) / len(diffs) if diffs else float("nan"),
        "cohen_dz": cohen_dz(diffs),
    }
    return tests


def stage_means(per_stage: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for s in STAGES:
        vals = list(per_stage[s].values())
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            out[s] = {"mean": mean, "std": math.sqrt(var), "n": len(vals)}
        else:
            out[s] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return out


def plot_ak_curves(icl_rows: list[dict], prompt_set: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # a_k_curve in icl metrics dict — average across prompts per stage
    per_stage_curves: dict[str, list[list[float]]] = defaultdict(list)
    for r in icl_rows:
        if r.get("prompt_set") != prompt_set:
            continue
        curve = r.get("a_k_curve")
        if not curve:
            continue
        per_stage_curves[r["stage"]].append(curve)

    if not per_stage_curves:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in STAGES:
        if s not in per_stage_curves:
            continue
        curves = np.array(per_stage_curves[s])  # (prompts, K)
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / math.sqrt(len(curves))
        ks = list(range(1, len(mean) + 1))
        ax.plot(ks, mean, label=STAGE_LABELS[s], marker="o")
        ax.fill_between(ks, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlabel("k (response index)")
    ax.set_ylabel(r"$\bar a_k$ (bits/byte)")
    ax.set_title(f"Progressive conditional surprise on {prompt_set}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_violins(per_stage: dict[str, dict[str, float]], prompt_set: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    data = [list(per_stage[s].values()) for s in STAGES if per_stage[s]]
    labels = [STAGE_LABELS[s] for s in STAGES if per_stage[s]]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot(data, showmeans=True, showextrema=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel(r"$C \times a_n$ (bits/byte)")
    ax.set_title(f"Per-prompt diversity on {prompt_set}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _format_corr_pvalue(p: float) -> str:
    """Format a two-sided p-value for an in-figure correlation annotation."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = ---"
    if p <= 0.0:
        return "p < 1e-300"
    if p < 1e-3:
        exp = int(math.floor(math.log10(p)))
        return f"p < 1e{exp + 1}"
    return f"p = {p:.3f}"


def _corr_annotation(x, y) -> str:
    """Pearson r and Spearman ρ (two-sided) annotation for a scatter panel.

    Requires at least 3 finite paired points; returns "n < 3" otherwise.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return f"n = {x.size} (need >= 3)"
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    return (
        f"Pearson r = {r:.2f} ({_format_corr_pvalue(float(p_r))})\n"
        f"Spearman ρ = {rho:.2f} ({_format_corr_pvalue(float(p_rho))})"
    )


def plot_metric_correlation(
    icl_rows: list[dict], baseline_rows: list[dict], prompt_set: str, out: Path
) -> None:
    import matplotlib.pyplot as plt

    # Join on (stage, prompt_id) for the given prompt_set
    icl_map = {
        (r["stage"], r["prompt_id"]): _derive_metric(r, "D_Can")
        for r in icl_rows
        if r.get("prompt_set") == prompt_set
    }
    base_map_ead = {
        (r["stage"], r["prompt_id"]): r.get("ead")
        for r in baseline_rows
        if r.get("prompt_set") == prompt_set
    }
    base_map_sb = {
        (r["stage"], r["prompt_id"]): r.get("sentbert_diversity")
        for r in baseline_rows
        if r.get("prompt_set") == prompt_set
    }
    stages_present = [s for s in STAGES if any(k[0] == s for k in icl_map)]
    if not stages_present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colours = {"base": "tab:blue", "sft": "tab:orange", "dpo": "tab:green", "instruct": "tab:red"}
    for ax, (base_map, title) in zip(
        axes,
        [(base_map_ead, "EAD"), (base_map_sb, "SentBERT diversity")],
    ):
        pooled_x: list[float] = []
        pooled_y: list[float] = []
        for s in stages_present:
            xs = []
            ys = []
            for k, d in icl_map.items():
                if k[0] != s:
                    continue
                b = base_map.get(k)
                if d is None or b is None:
                    continue
                xs.append(b)
                ys.append(d)
            if xs:
                ax.scatter(xs, ys, s=8, alpha=0.6, c=colours.get(s, "gray"), label=STAGE_LABELS[s])
                pooled_x.extend(xs)
                pooled_y.extend(ys)
        ax.set_xlabel(title)
        ax.set_ylabel(r"$C \times a_n$")
        ax.grid(True, alpha=0.3)
        annot = _corr_annotation(pooled_x, pooled_y)
        ax.text(
            0.02,
            0.98,
            annot,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"C × a_n vs baselines on {prompt_set}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def safe_num(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    return f"{x:.3f}"


def format_p(p: float | None) -> str:
    """Format a p-value for use *inside* an existing math-mode wrapper
    like `$p={...}$`. Must therefore not introduce its own `$..$`."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "\\text{---}"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def write_results_table(analysis: dict, out_path: Path) -> None:
    """One LaTeX table per prompt_set, with per-stage means and H1 tests."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pset in ["alpacaeval", "nbcurated"]:
        a = analysis[pset]
        ms = a["stage_means"]["D_Can"]
        tests = a["tests"]["D_Can"]
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"Stage & $D = C \cdot a_n$ mean & std & $n$ \\")
        lines.append(r"\midrule")
        for s in STAGES:
            v = ms.get(s, {})
            lines.append(
                f"{STAGE_LABELS[s]} & {safe_num(v.get('mean'))} & "
                f"{safe_num(v.get('std'))} & {v.get('n', 0)} \\\\"
            )
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{4}{l}{\textbf{H1 tests (paired Wilcoxon, Bonferroni)}} \\")
        for name, a_s, b_s, _ in H1_CONTRASTS:
            t = tests[name]
            lines.append(
                f"{STAGE_LABELS[a_s]}$>$ {STAGE_LABELS[b_s]} & "
                f"$\\Delta={safe_num(t['mean_diff'])}$ & "
                f"$d_z={safe_num(t['cohen_dz'])}$ & "
                f"$p={format_p(t['p_bonferroni'])}$ \\\\"
            )
        t = tests["Hpa"]
        lines.append(
            r"\multicolumn{4}{l}{\textbf{H1' (exploratory two-sided, uncorrected)}} \\"
        )
        lines.append(
            f"DPO $\\neq$ Instruct & "
            f"$\\Delta={safe_num(t['mean_diff'])}$ & "
            f"$d_z={safe_num(t['cohen_dz'])}$ & "
            f"$p={format_p(t['p_raw'])}$ \\\\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")
        lines.append(r"\vspace{0.5em}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def write_paper_macros(analysis: dict, out_path: Path, name_infix: str = "") -> None:
    """Emit `\\newcommand` macros for every scalar the TeX section will cite.

    `name_infix` is inserted between the `\\olmo` prefix and the rest of every
    macro name. Use `"Lm"` for the length-matched variant so its macros do not
    collide with the raw-generation `\\olmoAlpacaBaseDMean` etc.
    """
    lines = ["% Auto-generated by scripts/rlhf_experiment/5_analyze_and_figures.py", ""]
    for pset in ["alpacaeval", "nbcurated"]:
        a = analysis[pset]
        ms = a["stage_means"]["D_Can"]
        tests = a["tests"]["D_Can"]
        tag = "Alpaca" if pset == "alpacaeval" else "NbCurated"
        prefix = f"\\olmo{name_infix}{tag}"
        for stage in STAGES:
            cmd = f"{prefix}{stage.capitalize()}DMean"
            lines.append(f"\\newcommand{{{cmd}}}{{{safe_num(ms[stage]['mean'])}}}")
        for name, a_s, b_s, _ in H1_CONTRASTS:
            t = tests[name]
            stem = f"{prefix}{name}"
            lines.append(f"\\newcommand{{{stem}Pbonf}}{{{safe_num(t['p_bonferroni'])}}}")
            lines.append(f"\\newcommand{{{stem}Dz}}{{{safe_num(t['cohen_dz'])}}}")
            lines.append(f"\\newcommand{{{stem}Diff}}{{{safe_num(t['mean_diff'])}}}")
        t = tests["Hpa"]
        stem = f"{prefix}Hpa"
        lines.append(f"\\newcommand{{{stem}P}}{{{safe_num(t['p_raw'])}}}")
        lines.append(f"\\newcommand{{{stem}Dz}}{{{safe_num(t['cohen_dz'])}}}")
        lines.append(f"\\newcommand{{{stem}Diff}}{{{safe_num(t['mean_diff'])}}}")
        lines.append(f"\\newcommand{{{prefix}N}}{{{ms['base']['n']}}}")
    out_path.write_text("\n".join(lines) + "\n")


def analyze(icl_rows: list[dict], baseline_rows: list[dict]) -> dict:
    """Per-prompt-set analysis.

    Primary diversity metric: `D_Can = C × a_n` (per-byte), the paper's
    recommended scalar (§6.3). NOTE: core.py's `diversity_score_D` key
    stores `C × E`, an alternative from an earlier draft; we derive D_Can
    ourselves from `coherence_C * a_k_curve_per_byte[-1]`.
    """
    out: dict[str, Any] = {}
    for pset in ["alpacaeval", "nbcurated"]:
        per_stage = {
            "D_Can": collect_metric(icl_rows, "D_Can", pset),                   # primary
            "a_n_per_byte": collect_metric(icl_rows, "a_n_per_byte", pset),
            "coherence_C": collect_metric(icl_rows, "coherence_C", pset),
            "diversity_score_D_rate": collect_metric(              # C × E_rate
                icl_rows, "diversity_score_D_rate", pset
            ),
            "ead": collect_metric(baseline_rows, "ead", pset),
            "distinct_n": collect_metric(baseline_rows, "distinct_n", pset),
            "sentbert_diversity": collect_metric(baseline_rows, "sentbert_diversity", pset),
        }
        stage_m = {k: stage_means(v) for k, v in per_stage.items()}
        tests = {k: run_contrasts(v) for k, v in per_stage.items()}
        out[pset] = {
            "stage_means": stage_m,
            "tests": tests,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icl", type=Path, default=RESULTS_DIR / "icl_metrics.jsonl")
    ap.add_argument("--baselines", type=Path, default=RESULTS_DIR / "baseline_metrics.jsonl")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument(
        "--analysis-name",
        type=str,
        default="analysis.json",
        help="Filename for the JSON summary written under --out-dir.",
    )
    ap.add_argument(
        "--table-subdir",
        type=str,
        default="tables",
        help="Subdir of --out-dir to receive the rlhf_diversity.tex table.",
    )
    ap.add_argument(
        "--macros-name",
        type=str,
        default="paper_macros_rlhf.tex",
        help="Filename for the \\newcommand macros file under --out-dir.",
    )
    ap.add_argument(
        "--macro-infix",
        type=str,
        default="",
        help=(
            "Infix inserted between '\\olmo' and the rest of each macro name. "
            "Use 'Lm' for length-matched runs to avoid colliding with the raw "
            "macros."
        ),
    )
    ap.add_argument(
        "--fig-suffix",
        type=str,
        default="",
        help="String appended before .pdf for each figure (e.g. '_lm').",
    )
    args = ap.parse_args()

    icl = read_jsonl(args.icl) if args.icl.exists() else []
    baselines = read_jsonl(args.baselines) if args.baselines.exists() else []
    print(f"[analyze] {len(icl)} ICL rows, {len(baselines)} baseline rows")

    analysis = analyze(icl, baselines)
    (args.out_dir / args.analysis_name).write_text(json.dumps(analysis, indent=2))

    for pset in ["alpacaeval", "nbcurated"]:
        plot_ak_curves(icl, pset, args.fig_dir / f"ak_curves_overlay_{pset}{args.fig_suffix}.pdf")
        per_stage_D = collect_metric(icl, "D_Can", pset)
        plot_violins(per_stage_D, pset, args.fig_dir / f"per_prompt_D_violin_{pset}{args.fig_suffix}.pdf")
        plot_metric_correlation(
            icl, baselines, pset,
            args.fig_dir / f"metric_correlation_scatter_{pset}{args.fig_suffix}.pdf",
        )

    (args.out_dir / args.table_subdir).mkdir(parents=True, exist_ok=True)
    write_results_table(analysis, args.out_dir / args.table_subdir / "rlhf_diversity.tex")
    write_paper_macros(analysis, args.out_dir / args.macros_name, name_infix=args.macro_infix)
    print(f"[analyze] wrote {args.analysis_name}, {args.table_subdir}/, figures/, "
          f"{args.macros_name}")


if __name__ == "__main__":
    main()
