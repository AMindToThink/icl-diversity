"""Paper-ready outputs for the OLMo self-scoring matrix experiment.

Consumes the outputs of 6_matrix_analysis.py (matrix_summary.json is the
source of truth for cell means, contrasts, and fold-changes; the per-grader
JSONLs provide per-prompt D for the variance decomposition) and emits every
number the paper section and FINDINGS.md cite, so nothing is hand-typed:

  results/rlhf_experiment/matrix/variance_decomposition.json
  results/rlhf_experiment/matrix/generated_tables.md
  results/rlhf_experiment/tables_matrix/olmo_matrix_D.tex
  results/rlhf_experiment/tables_matrix/olmo_matrix_R.tex
  results/rlhf_experiment/tables_matrix/olmo_matrix_variance.tex
  results/rlhf_experiment/paper_macros_matrix.tex

Variance decomposition (grader identity vs generating stage):
  The design is fully crossed and balanced (5 grader thetas x 4 generator
  stages x n prompts, one D = C*a_n observation per cell), so sums of squares
  are unambiguous. Two views are computed per prompt set:

  1. Cell-mean level (the matrix the reader sees): two-way decomposition of
     the 5x4 matrix of prompt-averaged D into grader main effect, stage main
     effect, and their interaction (residual). Reported as fractions of total
     sum of squares (eta squared). Also computed for the in-family 4x4
     submatrix (OLMo grader rows only).
  2. Per-prompt level: full three-way crossed decomposition of the
     (grader, stage, prompt) tensor. Prompt identity is a nuisance factor, so
     the reported shares are of the *within-prompt* sum of squares
     (total minus the prompt main effect), attributing grader = grader main +
     grader x prompt, stage = stage main + stage x prompt, interaction =
     grader x stage + three-way residual.

Fail-fast checks: every cell must exist with an identical prompt-id set, and
recomputed cell means must match matrix_summary.json (guards against a stale
summary from an old run of 6_matrix_analysis.py).

Usage:
  uv run python scripts/rlhf_experiment/7_matrix_paper_outputs.py
  uv run python scripts/rlhf_experiment/7_matrix_paper_outputs.py --update-findings
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_matrix_analysis():
    """Import 6_matrix_analysis.py (digit-leading name) for its loaders/constants."""
    path = Path(__file__).with_name("6_matrix_analysis.py")
    spec = importlib.util.spec_from_file_location("matrix_analysis_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MA = _load_matrix_analysis()

THETA_LABELS = [label for label, _, _ in MA.THETA_ROWS]
OLMO_THETA_LABELS = [t for t in THETA_LABELS if t != "Qwen-3B"]

THETA_MACRO = {
    "Qwen-3B": "ThetaQwen",
    "OLMo-base": "ThetaBase",
    "OLMo-sft": "ThetaSft",
    "OLMo-dpo": "ThetaDpo",
    "OLMo-instruct": "ThetaInstruct",
}
GEN_MACRO = {"base": "GenBase", "sft": "GenSft", "dpo": "GenDpo", "instruct": "GenInstruct"}
PSET_MACRO = {"alpacaeval": "Alpaca", "nbcurated": "Nb"}
STEP_MACRO = {"base>sft": "BaseSft", "sft>dpo": "SftDpo"}

THETA_TEX = {
    "Qwen-3B": "Qwen2.5-3B (cross-family)",
    "OLMo-base": "OLMo base",
    "OLMo-sft": "OLMo SFT",
    "OLMo-dpo": "OLMo DPO",
    "OLMo-instruct": "OLMo Instruct (RLVR)",
}
GEN_TEX = {"base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct (RLVR)"}
PSET_TEX = {"alpacaeval": "AlpacaEval", "nbcurated": "NoveltyBench curated"}


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def pct1(frac: float) -> str:
    return f"{100 * frac:.1f}"


def fmt_p(p: float) -> str:
    """LaTeX-math p-value, matching the 5_analyze_and_figures.py convention."""
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{float(mant):.1f} \\times 10^{{{int(exp)}}}"


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def two_way_decomposition(M: np.ndarray) -> dict:
    """Balanced two-way decomposition of an (R, C) matrix, one obs per cell.

    Returns sums of squares for row and column main effects and the residual
    (interaction), plus fractions of total SS.
    """
    if M.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {M.shape}")
    n_rows, n_cols = M.shape
    grand = M.mean()
    row_eff = M.mean(axis=1) - grand
    col_eff = M.mean(axis=0) - grand
    resid = M - grand - row_eff[:, None] - col_eff[None, :]
    ss = {
        "row": float(n_cols * (row_eff**2).sum()),
        "col": float(n_rows * (col_eff**2).sum()),
        "interaction": float((resid**2).sum()),
    }
    ss_total = float(((M - grand) ** 2).sum())
    if not np.isclose(sum(ss.values()), ss_total, rtol=1e-9, atol=1e-15):
        raise AssertionError("two-way SS components do not sum to total SS")
    frac = {k: (v / ss_total if ss_total > 0 else float("nan")) for k, v in ss.items()}
    return {"ss": ss, "ss_total": ss_total, "frac": frac}


def three_way_decomposition(X: np.ndarray) -> dict:
    """Balanced three-way crossed decomposition of a (G, S, P) tensor.

    Factors: grader (axis 0), stage (axis 1), prompt (axis 2); one observation
    per cell, so the three-way term is the residual.
    """
    if X.ndim != 3:
        raise ValueError(f"expected 3-D tensor, got shape {X.shape}")
    n_g, n_s, n_p = X.shape
    grand = X.mean()
    a = X.mean(axis=(1, 2)) - grand
    b = X.mean(axis=(0, 2)) - grand
    c = X.mean(axis=(0, 1)) - grand
    ab = X.mean(axis=2) - grand - a[:, None] - b[None, :]
    ac = X.mean(axis=1) - grand - a[:, None] - c[None, :]
    bc = X.mean(axis=0) - grand - b[:, None] - c[None, :]
    abc = (
        X
        - grand
        - a[:, None, None]
        - b[None, :, None]
        - c[None, None, :]
        - ab[:, :, None]
        - ac[:, None, :]
        - bc[None, :, :]
    )
    ss = {
        "grader": float(n_s * n_p * (a**2).sum()),
        "stage": float(n_g * n_p * (b**2).sum()),
        "prompt": float(n_g * n_s * (c**2).sum()),
        "grader_x_stage": float(n_p * (ab**2).sum()),
        "grader_x_prompt": float(n_s * (ac**2).sum()),
        "stage_x_prompt": float(n_g * (bc**2).sum()),
        "residual": float((abc**2).sum()),
    }
    ss_total = float(((X - grand) ** 2).sum())
    if not np.isclose(sum(ss.values()), ss_total, rtol=1e-9, atol=1e-15):
        raise AssertionError("three-way SS components do not sum to total SS")
    ss_within_prompt = ss_total - ss["prompt"]
    within_frac = {
        "grader": (ss["grader"] + ss["grader_x_prompt"]) / ss_within_prompt,
        "stage": (ss["stage"] + ss["stage_x_prompt"]) / ss_within_prompt,
        "interaction": (ss["grader_x_stage"] + ss["residual"]) / ss_within_prompt,
    }
    return {
        "ss": ss,
        "ss_total": ss_total,
        "prompt_frac_of_total": ss["prompt"] / ss_total,
        "within_prompt_frac": within_frac,
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_tensor(theta_cells: dict, pset: str) -> tuple[np.ndarray, list[str]]:
    """(n_theta, n_gen, n_prompt) tensor of D = C*a_n; fails on any gap/mismatch."""
    prompt_ids: set[str] | None = None
    for label in THETA_LABELS:
        for gen in MA.GEN_ORDER:
            cell = theta_cells[label].get((gen, pset), {})
            if not cell:
                raise ValueError(f"empty cell theta={label} gen={gen} pset={pset}")
            ids = set(cell)
            if prompt_ids is None:
                prompt_ids = ids
            elif ids != prompt_ids:
                raise ValueError(
                    f"prompt-id mismatch at theta={label} gen={gen} pset={pset}: "
                    f"{len(ids ^ prompt_ids)} ids differ from the first cell"
                )
    assert prompt_ids is not None
    ordered = sorted(prompt_ids)
    X = np.array(
        [
            [
                [theta_cells[label][(gen, pset)][p]["diversity_score_D_C_an"] for p in ordered]
                for gen in MA.GEN_ORDER
            ]
            for label in THETA_LABELS
        ],
        dtype=float,
    )
    return X, ordered


def check_against_summary(X: np.ndarray, summary: dict, pset: str) -> None:
    """Recomputed cell means must match matrix_summary.json — else it is stale."""
    matrix = summary["prompt_sets"][pset]["matrix"]
    for i, label in enumerate(THETA_LABELS):
        for j, gen in enumerate(MA.GEN_ORDER):
            expected = matrix[label][gen]["D_mean"]
            got = float(X[i, j].mean())
            if not np.isclose(got, expected, rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"cell mean mismatch vs matrix_summary.json at theta={label} "
                    f"gen={gen} pset={pset}: recomputed {got!r} vs summary {expected!r}. "
                    "Re-run 6_matrix_analysis.py first."
                )


def decompose(X: np.ndarray) -> dict:
    """All decomposition views for one prompt set's (5, 4, P) tensor."""
    olmo_idx = [THETA_LABELS.index(t) for t in OLMO_THETA_LABELS]
    cell_means = X.mean(axis=2)
    out = {
        "n_prompts": int(X.shape[2]),
        "cell_means_all_graders": two_way_decomposition(cell_means),
        "cell_means_olmo_only": two_way_decomposition(cell_means[olmo_idx]),
        "per_prompt_all_graders": three_way_decomposition(X),
        "per_prompt_olmo_only": three_way_decomposition(X[olmo_idx]),
    }
    # Relabel the two-way row/col keys to the factors they mean here.
    for key in ("cell_means_all_graders", "cell_means_olmo_only"):
        for sub in ("ss", "frac"):
            d = out[key][sub]
            out[key][sub] = {
                "grader": d["row"],
                "stage": d["col"],
                "interaction": d["interaction"],
            }
    return out


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

GENERATED_BY = "Auto-generated by scripts/rlhf_experiment/7_matrix_paper_outputs.py -- do not hand-edit."


def emit_tex_D(summary: dict) -> str:
    lines = ["% " + GENERATED_BY, "\\begin{tabular}{lrrrr}", "\\toprule"]
    for pset in MA.PROMPT_SETS:
        matrix = summary["prompt_sets"][pset]["matrix"]
        n = matrix[THETA_LABELS[0]]["base"]["n"]
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{\\textbf{{{PSET_TEX[pset]}}} ($n={n}$ prompts)}} \\\\"
        )
        lines.append("\\midrule")
        header = " & ".join(GEN_TEX[g] for g in MA.GEN_ORDER)
        lines.append(f"$\\theta$ (grader) & {header} \\\\")
        lines.append("\\midrule")
        for label in THETA_LABELS:
            cells = " & ".join(fmt3(matrix[label][g]["D_mean"]) for g in MA.GEN_ORDER)
            lines.append(f"{THETA_TEX[label]} & {cells} \\\\")
        if pset != MA.PROMPT_SETS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_tex_R(summary: dict) -> str:
    lines = [
        "% " + GENERATED_BY,
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{AlpacaEval} & \\multicolumn{2}{c}{NB curated} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "$\\theta$ (grader) & $R_\\theta$ & 95\\% CI & $R_\\theta$ & 95\\% CI \\\\",
        "\\midrule",
    ]
    for label in THETA_LABELS:
        cells = []
        for pset in MA.PROMPT_SETS:
            r = summary["prompt_sets"][pset]["R_theta"][label]
            cells.append(fmt2(r["R"]))
            cells.append(f"[{fmt2(r['ci95'][0])}, {fmt2(r['ci95'][1])}]")
        lines.append(f"{THETA_TEX[label]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_tex_variance(var: dict) -> str:
    lines = [
        "% " + GENERATED_BY,
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{AlpacaEval} & \\multicolumn{2}{c}{NB curated} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Variance source & cell means & within-prompt & cell means & within-prompt \\\\",
        "\\midrule",
    ]
    rows = [
        ("Generator stage", "stage"),
        ("Grader $\\theta$", "grader"),
        ("Grader $\\times$ stage interaction", "interaction"),
    ]
    for display, key in rows:
        cells = []
        for pset in MA.PROMPT_SETS:
            v = var[pset]
            cells.append(pct1(v["cell_means_all_graders"]["frac"][key]) + "\\%")
            cells.append(pct1(v["per_prompt_all_graders"]["within_prompt_frac"][key]) + "\\%")
        lines.append(f"{display} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_macros(summary: dict, var: dict) -> str:
    lines = [
        "% " + GENERATED_BY,
        "% Inline-number macros for the OLMo self-scoring matrix experiment",
        "% (paper/sections/appG_olmo_self_scoring.tex and FINDINGS.md prose).",
    ]

    def add(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\olmoMx{name}}}{{{value}}}")

    all_p_holm: list[float] = []
    all_R: list[float] = []
    all_R_lo: list[float] = []
    n_contrasts = 0
    n_sig = 0

    for pset in MA.PROMPT_SETS:
        ps = PSET_MACRO[pset]
        out = summary["prompt_sets"][pset]
        add(f"{ps}N", str(out["matrix"][THETA_LABELS[0]]["base"]["n"]))
        # Full D / C / a_n matrices.
        for label in THETA_LABELS:
            th = THETA_MACRO[label]
            for gen in MA.GEN_ORDER:
                cell = out["matrix"][label][gen]
                add(f"{ps}D{th}{GEN_MACRO[gen]}", fmt3(cell["D_mean"]))
                add(f"{ps}C{th}{GEN_MACRO[gen]}", fmt3(cell["C_mean"]))
                add(f"{ps}An{th}{GEN_MACRO[gen]}", fmt3(cell["a_n_mean"]))
            fold = out["matrix"][label]["base"]["a_n_mean"] / out["matrix"][label]["instruct"]["a_n_mean"]
            add(f"{ps}AnFold{th}", f"{fold:.1f}")
        # Confirmatory contrasts (Holm-corrected one-sided Wilcoxon on D).
        for c in out["contrasts"]:
            n_contrasts += 1
            all_p_holm.append(c["p_holm"])
            if c["p_holm"] < 0.05:
                n_sig += 1
            add(
                f"{ps}Pholm{THETA_MACRO[c['theta']]}{STEP_MACRO[c['step']]}",
                fmt_p(c["p_holm"]),
            )
        # Fold-changes.
        for label in THETA_LABELS:
            r = out["R_theta"][label]
            th = THETA_MACRO[label]
            all_R.append(r["R"])
            all_R_lo.append(r["ci95"][0])
            add(f"{ps}R{th}", fmt2(r["R"]))
            add(f"{ps}RLo{th}", fmt2(r["ci95"][0]))
            add(f"{ps}RHi{th}", fmt2(r["ci95"][1]))
        # Variance decomposition.
        v = var[pset]
        for key in ("stage", "grader", "interaction"):
            cap = key.capitalize()
            add(f"{ps}Var{cap}Pct", pct1(v["cell_means_all_graders"]["frac"][key]))
            add(f"{ps}Var{cap}InFamPct", pct1(v["cell_means_olmo_only"]["frac"][key]))
            add(f"{ps}VarWithin{cap}Pct", pct1(v["per_prompt_all_graders"]["within_prompt_frac"][key]))
        add(f"{ps}VarPromptPct", pct1(v["per_prompt_all_graders"]["prompt_frac_of_total"]))

    add("NContrasts", str(n_contrasts))
    add("NContrastsSig", str(n_sig))
    add("MaxPholm", fmt_p(max(all_p_holm)))
    add("MinR", fmt2(min(all_R)))
    add("MinRLo", fmt2(min(all_R_lo)))
    return "\n".join(lines) + "\n"


def emit_markdown(summary: dict, var: dict) -> dict[str, str]:
    """Named markdown blocks spliced into FINDINGS.md between markers."""
    blocks: dict[str, str] = {}

    # D matrix (AlpacaEval), with the monotone-chain verdict column.
    lines = ["| θ (grader) \\ gen | base | sft | dpo | instruct | base>sft>dpo |", "|---|---|---|---|---|---|"]
    matrix = summary["prompt_sets"]["alpacaeval"]["matrix"]
    for label in THETA_LABELS:
        row = matrix[label]
        vals = [row[g]["D_mean"] for g in MA.GEN_ORDER]
        mono = "✓" if vals[0] > vals[1] > vals[2] else "✗"
        cells = " | ".join(fmt3(v) for v in vals)
        lines.append(f"| {label} | {cells} | {mono} |")
    blocks["d-matrix-alpacaeval"] = "\n".join(lines)

    # R_theta fold-change table, both prompt sets.
    lines = ["| θ | R (AlpacaEval) | 95% CI | R (NB) | 95% CI |", "|---|---|---|---|---|"]
    for label in THETA_LABELS:
        cells = []
        for pset in MA.PROMPT_SETS:
            r = summary["prompt_sets"][pset]["R_theta"][label]
            cells.append(fmt2(r["R"]))
            cells.append(f"[{fmt2(r['ci95'][0])}, {fmt2(r['ci95'][1])}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    blocks["r-table"] = "\n".join(lines)

    # Variance decomposition, both levels, both prompt sets.
    lines = [
        "| Variance source | AlpacaEval cell means | AlpacaEval within-prompt | NB cell means | NB within-prompt |",
        "|---|---|---|---|---|",
    ]
    for display, key in [
        ("Generator stage", "stage"),
        ("Grader θ", "grader"),
        ("Grader × stage interaction", "interaction"),
    ]:
        cells = []
        for pset in MA.PROMPT_SETS:
            v = var[pset]
            cells.append(pct1(v["cell_means_all_graders"]["frac"][key]) + "%")
            cells.append(pct1(v["per_prompt_all_graders"]["within_prompt_frac"][key]) + "%")
        lines.append(f"| {display} | " + " | ".join(cells) + " |")
    blocks["variance"] = "\n".join(lines)

    return blocks


def splice_findings(findings_path: Path, blocks: dict[str, str]) -> None:
    """Replace marker-delimited blocks in FINDINGS.md with generated content."""
    text = findings_path.read_text()
    for name, content in blocks.items():
        begin = f"<!-- BEGIN GENERATED: {name} -->"
        end = f"<!-- END GENERATED: {name} -->"
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(text):
            raise ValueError(f"FINDINGS.md is missing markers for block {name!r}")
        text = pattern.sub(begin + "\n" + content + "\n" + end, text)
    findings_path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_root: Path, update_findings: bool) -> dict:
    summary_path = REPO_ROOT / "results" / "rlhf_experiment" / "matrix" / "matrix_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found; run 6_matrix_analysis.py first")
    summary = json.loads(summary_path.read_text())
    if summary.get("missing_theta"):
        raise ValueError(f"matrix_summary.json reports missing theta rows: {summary['missing_theta']}")

    theta_cells = {}
    for label, _model_id, path in MA.THETA_ROWS:
        rows = MA.load_rows(path)
        if not rows:
            raise FileNotFoundError(f"no rows loaded for theta={label} from {path}")
        theta_cells[label] = MA.index_by_cell(rows)

    var: dict = {}
    for pset in MA.PROMPT_SETS:
        X, _prompt_ids = build_tensor(theta_cells, pset)
        check_against_summary(X, summary, pset)
        var[pset] = decompose(X)

    matrix_dir = output_root / "results" / "rlhf_experiment" / "matrix"
    tables_dir = output_root / "results" / "rlhf_experiment" / "tables_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    variance_out = {
        "generated_by": "scripts/rlhf_experiment/7_matrix_paper_outputs.py",
        "inputs": {
            "matrix_summary": str(summary_path.relative_to(REPO_ROOT)),
            "metric_key": "diversity_score_D_C_an",
        },
        "theta_order": THETA_LABELS,
        "gen_order": list(MA.GEN_ORDER),
        "prompt_sets": var,
    }
    (matrix_dir / "variance_decomposition.json").write_text(json.dumps(variance_out, indent=2))
    (tables_dir / "olmo_matrix_D.tex").write_text(emit_tex_D(summary))
    (tables_dir / "olmo_matrix_R.tex").write_text(emit_tex_R(summary))
    (tables_dir / "olmo_matrix_variance.tex").write_text(emit_tex_variance(var))
    macros = emit_macros(summary, var)
    (output_root / "results" / "rlhf_experiment" / "paper_macros_matrix.tex").write_text(macros)

    blocks = emit_markdown(summary, var)
    md_lines = [f"<!-- {GENERATED_BY} -->", ""]
    for name, content in blocks.items():
        md_lines += [f"## {name}", "", content, ""]
    (matrix_dir / "generated_tables.md").write_text("\n".join(md_lines))

    if update_findings:
        splice_findings(REPO_ROOT / "results" / "rlhf_experiment" / "matrix" / "FINDINGS.md", blocks)

    return variance_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT,
                    help="root under which results/... outputs are written (for tests)")
    ap.add_argument("--update-findings", action="store_true",
                    help="splice generated tables into FINDINGS.md between markers")
    args = ap.parse_args()

    variance_out = run(args.output_root, args.update_findings)
    for pset, v in variance_out["prompt_sets"].items():
        cm = v["cell_means_all_graders"]["frac"]
        wp = v["per_prompt_all_graders"]["within_prompt_frac"]
        print(
            f"[{pset}] cell-mean SS: stage {100*cm['stage']:.1f}% / grader {100*cm['grader']:.1f}%"
            f" / interaction {100*cm['interaction']:.1f}%  |  within-prompt SS:"
            f" stage {100*wp['stage']:.1f}% / grader {100*wp['grader']:.1f}%"
            f" / interaction {100*wp['interaction']:.1f}%"
        )


if __name__ == "__main__":
    main()
