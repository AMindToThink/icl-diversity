"""Paper-ready outputs for the OLMo self-scoring matrix experiment.

Consumes the outputs of 6_matrix_analysis.py (matrix_summary.json is the
source of truth for the pre-registered 5-grader cell means, contrasts, and
fold-changes; the per-grader JSONLs provide per-prompt D for the variance
decomposition) and emits every number the paper section and FINDINGS.md cite,
so nothing is hand-typed:

  results/rlhf_experiment/matrix/variance_decomposition.json
  results/rlhf_experiment/matrix/generated_tables.md
  results/rlhf_experiment/tables_matrix/olmo_matrix_D.tex
  results/rlhf_experiment/tables_matrix/olmo_matrix_R.tex
  results/rlhf_experiment/tables_matrix/olmo_matrix_variance.tex
  results/rlhf_experiment/paper_macros_matrix.tex

Grader rows come in two tiers:

  * The frozen, pre-registered 5-row family analyzed by 6_matrix_analysis.py
    (Qwen2.5-3B + the four OLMo stages). Its statistics are read from
    matrix_summary.json and are never recomputed here.
  * The extended cross-family rows (Llama-3.1-8B base, GPT-2), scored later as
    a robustness check. Their ordering tests are exploratory (their own Holm
    family, labeled as such), and they join an extended variance decomposition.

GPT-2's 1024-token context cannot fit every AlpacaEval group; those groups were
skipped explicitly at scoring time and recorded in a
.skipped_context_overflow.json sidecar. This script verifies that every prompt
missing from an extended row is exactly accounted for by that sidecar (any
other gap means an incomplete scoring run and raises), then computes the
extended decomposition on the common prompt subset, reporting what was dropped.

Variance decomposition (grader identity vs generating stage):
  The design is fully crossed and balanced (graders x 4 generator stages x n
  prompts, one D = C*a_n observation per cell), so sums of squares are
  unambiguous. Two views are computed per prompt set and per grader tier:

  1. Cell-mean level (the matrix the reader sees): two-way decomposition of
     the grader x stage matrix of prompt-averaged D into grader main effect,
     stage main effect, and their interaction (residual). Reported as
     fractions of total sum of squares (eta squared).
  2. Per-prompt level: full three-way crossed decomposition of the
     (grader, stage, prompt) tensor. Prompt identity is a nuisance factor, so
     the reported shares are of the *within-prompt* sum of squares
     (total minus the prompt main effect), attributing grader = grader main +
     grader x prompt, stage = stage main + stage x prompt, interaction =
     grader x stage + three-way residual.

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
from scipy import stats

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

# Extended cross-family graders (robustness tier; not part of the frozen
# pre-registered family in 6_matrix_analysis.py).
EXTENDED_THETA_ROWS = [
    ("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B",
     MA.MATRIX_DIR / "icl_metrics_lm_theta-llama8b.jsonl"),
    ("GPT-2", "gpt2",
     MA.MATRIX_DIR / "icl_metrics_lm_theta-gpt2.jsonl"),
]
EXTENDED_LABELS = [label for label, _, _ in EXTENDED_THETA_ROWS]
ALL_LABELS = THETA_LABELS + EXTENDED_LABELS

THETA_MACRO = {
    "Qwen-3B": "ThetaQwen",
    "OLMo-base": "ThetaBase",
    "OLMo-sft": "ThetaSft",
    "OLMo-dpo": "ThetaDpo",
    "OLMo-instruct": "ThetaInstruct",
    "Llama-3.1-8B": "ThetaLlama",
    "GPT-2": "ThetaGptTwo",
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
    "Llama-3.1-8B": "Llama-3.1-8B (cross-family)",
    "GPT-2": "GPT-2 124M (cross-family)",
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


def _relabel_two_way(d: dict) -> dict:
    """Rename generic row/col keys to the factors they mean here."""
    for sub in ("ss", "frac"):
        d[sub] = {
            "grader": d[sub]["row"],
            "stage": d[sub]["col"],
            "interaction": d[sub]["interaction"],
        }
    return d


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_theta_cells(rows_spec: list[tuple[str, str, Path]]) -> dict:
    theta_cells = {}
    for label, _model_id, path in rows_spec:
        rows = MA.load_rows(path)
        if not rows:
            raise FileNotFoundError(f"no rows loaded for theta={label} from {path}")
        theta_cells[label] = MA.index_by_cell(rows)
    return theta_cells


def canonical_prompt_ids(theta_cells: dict, pset: str) -> set[str]:
    """The prompt-id set shared by every cell of every pre-registered row."""
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
    return prompt_ids


def load_skip_sidecar(path: Path) -> set[tuple[str, str, str]]:
    """(stage, prompt_set, prompt_id) keys recorded as skipped at scoring time."""
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {(s["stage"], s["prompt_set"], s["prompt_id"]) for s in data["skipped"]}


def extended_row_missing(
    theta_cells: dict, label: str, jsonl_path: Path, canonical: set[str], pset: str
) -> set[str]:
    """Prompt ids missing anywhere in an extended row; must match the sidecar.

    A missing (stage, prompt) pair is legitimate only if the scorer recorded it
    as a context-overflow skip; anything else means the scoring run is
    incomplete and we refuse to proceed.
    """
    sidecar = load_skip_sidecar(
        jsonl_path.with_name(jsonl_path.stem + ".skipped_context_overflow.json")
    )
    missing_pairs = set()
    for gen in MA.GEN_ORDER:
        cell = theta_cells[label].get((gen, pset), {})
        for pid in canonical - set(cell):
            missing_pairs.add((gen, pset, pid))
    unaccounted = missing_pairs - sidecar
    if unaccounted:
        raise ValueError(
            f"theta={label} pset={pset}: {len(unaccounted)} missing (stage, prompt) "
            f"pairs are NOT in the skip sidecar (incomplete scoring run?): "
            f"{sorted(unaccounted)[:5]}..."
        )
    return {pid for _gen, _ps, pid in missing_pairs}


def build_tensor(
    theta_cells: dict, pset: str, labels: list[str], prompt_ids: list[str]
) -> np.ndarray:
    """(n_theta, n_gen, n_prompt) tensor of D = C*a_n over an explicit id list."""
    return np.array(
        [
            [
                [theta_cells[label][(gen, pset)][p]["diversity_score_D_C_an"]
                 for p in prompt_ids]
                for gen in MA.GEN_ORDER
            ]
            for label in labels
        ],
        dtype=float,
    )


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


def decompose_views(X: np.ndarray, labels: list[str]) -> dict:
    """Cell-mean and per-prompt decomposition views for one tensor."""
    return {
        "theta_order": labels,
        "n_prompts": int(X.shape[2]),
        "cell_means": _relabel_two_way(two_way_decomposition(X.mean(axis=2))),
        "per_prompt": three_way_decomposition(X),
    }


def analyze_extended(
    theta_cells: dict, pset: str, canonical: set[str], n_boot: int, seed: int
) -> dict:
    """Extended-tier analysis: subset bookkeeping, decompositions, exploratory stats."""
    dropped: dict[str, dict] = {}
    for label, _mid, path in EXTENDED_THETA_ROWS:
        missing = extended_row_missing(theta_cells, label, path, canonical, pset)
        dropped[label] = {"n_missing": len(missing), "prompt_ids": sorted(missing)}

    common = sorted(canonical - {p for d in dropped.values() for p in d["prompt_ids"]})

    out: dict = {
        "n_prompts_canonical": len(canonical),
        "n_prompts_common": len(common),
        "dropped_per_grader": dropped,
        "all_graders_common": decompose_views(
            build_tensor(theta_cells, pset, ALL_LABELS, common), ALL_LABELS
        ),
    }

    # Llama-only extension is complete on the canonical set: decompose without
    # GPT-2 on the full prompt set as a no-subset robustness view.
    no_gpt_labels = THETA_LABELS + ["Llama-3.1-8B"]
    if dropped["Llama-3.1-8B"]["n_missing"] == 0:
        out["no_gpt2_full"] = decompose_views(
            build_tensor(theta_cells, pset, no_gpt_labels, sorted(canonical)),
            no_gpt_labels,
        )
    else:
        raise ValueError(
            f"Llama-3.1-8B row is missing {dropped['Llama-3.1-8B']['n_missing']} "
            f"prompts on {pset}; it has no context limit, so the run is incomplete."
        )

    # Per-row cell means on each extended row's own complete prompt set, plus
    # exploratory ordering stats (own Holm family: 2 graders x 2 steps).
    cell_means: dict[str, dict] = {}
    raw_contrasts: list[dict] = []
    r_theta: dict[str, dict] = {}
    for label in EXTENDED_LABELS:
        row_ids = sorted(canonical - set(dropped[label]["prompt_ids"]))
        cells = theta_cells[label]
        cell_means[label] = {
            gen: {
                "D_mean": float(np.mean(
                    [cells[(gen, pset)][p]["diversity_score_D_C_an"] for p in row_ids]
                )),
                "C_mean": float(np.mean(
                    [cells[(gen, pset)][p]["coherence_C"] for p in row_ids]
                )),
                "a_n_mean": float(np.mean(
                    [cells[(gen, pset)][p]["a_n_per_byte"] for p in row_ids]
                )),
                "n": len(row_ids),
            }
            for gen in MA.GEN_ORDER
        }
        for hi, lo in MA.CONFIRMATORY_STEPS:
            a = np.array([cells[(hi, pset)][p]["diversity_score_D_C_an"] for p in row_ids])
            b = np.array([cells[(lo, pset)][p]["diversity_score_D_C_an"] for p in row_ids])
            stat, p = stats.wilcoxon(a, b, alternative="greater")
            raw_contrasts.append({
                "theta": label, "step": f"{hi}>{lo}",
                "p_raw": float(p), "dz": MA.cohen_dz(a - b),
                "delta": float((a - b).mean()), "n": len(row_ids),
            })
        base = np.array([cells[("base", pset)][p]["diversity_score_D_C_an"] for p in row_ids])
        dpo = np.array([cells[("dpo", pset)][p]["diversity_score_D_C_an"] for p in row_ids])
        lo_ci, hi_ci = MA.bootstrap_ratio_ci(base, dpo, n_boot, seed)
        r_theta[label] = {
            "R": float(base.mean() / dpo.mean()), "ci95": [lo_ci, hi_ci], "n": len(row_ids),
        }
    for c, pa in zip(raw_contrasts, MA.holm([c["p_raw"] for c in raw_contrasts])):
        c["p_holm"] = pa
    out["new_row_cell_means"] = cell_means
    out["new_row_contrasts"] = raw_contrasts
    out["new_row_R"] = r_theta
    out["monotone"] = {
        label: bool(
            cell_means[label]["base"]["D_mean"]
            > cell_means[label]["sft"]["D_mean"]
            > cell_means[label]["dpo"]["D_mean"]
        )
        for label in EXTENDED_LABELS
    }
    return out


def decompose_preregistered(X: np.ndarray) -> dict:
    """All decomposition views for one prompt set's pre-registered (5, 4, P) tensor."""
    olmo_idx = [THETA_LABELS.index(t) for t in OLMO_THETA_LABELS]
    cell_means = X.mean(axis=2)
    return {
        "n_prompts": int(X.shape[2]),
        "cell_means_all_graders": _relabel_two_way(two_way_decomposition(cell_means)),
        "cell_means_olmo_only": _relabel_two_way(two_way_decomposition(cell_means[olmo_idx])),
        "per_prompt_all_graders": three_way_decomposition(X),
        "per_prompt_olmo_only": three_way_decomposition(X[olmo_idx]),
    }


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

GENERATED_BY = "Auto-generated by scripts/rlhf_experiment/7_matrix_paper_outputs.py -- do not hand-edit."


def _row_label(label: str, ext: dict, pset: str) -> str:
    """Display label; annotates n when a subset row differs from the panel n."""
    base = THETA_TEX[label]
    if label in EXTENDED_LABELS:
        n_row = ext[pset]["new_row_cell_means"][label]["base"]["n"]
        if n_row != ext[pset]["n_prompts_canonical"]:
            return f"{base} ($n{{=}}{n_row}$)"
    return base


def emit_tex_D(summary: dict, ext: dict) -> str:
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
        lines.append("\\addlinespace")
        for label in EXTENDED_LABELS:
            row = ext[pset]["new_row_cell_means"][label]
            cells = " & ".join(fmt3(row[g]["D_mean"]) for g in MA.GEN_ORDER)
            lines.append(f"{_row_label(label, ext, pset)} & {cells} \\\\")
        if pset != MA.PROMPT_SETS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_tex_R(summary: dict, ext: dict) -> str:
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
    lines.append("\\addlinespace")
    for label in EXTENDED_LABELS:
        cells = []
        for pset in MA.PROMPT_SETS:
            r = ext[pset]["new_row_R"][label]
            cells.append(fmt2(r["R"]))
            cells.append(f"[{fmt2(r['ci95'][0])}, {fmt2(r['ci95'][1])}]")
        lines.append(f"{_row_label(label, ext, pset)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _variance_panel(title: str, get_cm, get_wp) -> list[str]:
    rows = [
        ("Generator stage", "stage"),
        ("Grader $\\theta$", "grader"),
        ("Grader $\\times$ stage interaction", "interaction"),
    ]
    lines = [f"\\multicolumn{{5}}{{l}}{{\\textbf{{{title}}}}} \\\\", "\\midrule"]
    for display, key in rows:
        cells = []
        for pset in MA.PROMPT_SETS:
            cells.append(pct1(get_cm(pset)[key]) + "\\%")
            cells.append(pct1(get_wp(pset)[key]) + "\\%")
        lines.append(f"{display} & " + " & ".join(cells) + " \\\\")
    return lines


def emit_tex_variance(var: dict, ext: dict) -> str:
    lines = [
        "% " + GENERATED_BY,
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{AlpacaEval} & \\multicolumn{2}{c}{NB curated} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Variance source & cell means & within-prompt & cell means & within-prompt \\\\",
        "\\midrule",
    ]
    lines += _variance_panel(
        "Pre-registered graders (5)",
        lambda ps: var[ps]["cell_means_all_graders"]["frac"],
        lambda ps: var[ps]["per_prompt_all_graders"]["within_prompt_frac"],
    )
    lines.append("\\midrule")
    lines += _variance_panel(
        "Extended graders (7, common prompts)",
        lambda ps: ext[ps]["all_graders_common"]["cell_means"]["frac"],
        lambda ps: ext[ps]["all_graders_common"]["per_prompt"]["within_prompt_frac"],
    )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_macros(summary: dict, var: dict, ext: dict) -> str:
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
    ext_p_holm: list[float] = []
    ext_R: list[float] = []
    ext_R_lo: list[float] = []
    ext_n_contrasts = 0
    ext_n_sig = 0

    for pset in MA.PROMPT_SETS:
        ps = PSET_MACRO[pset]
        out = summary["prompt_sets"][pset]
        add(f"{ps}N", str(out["matrix"][THETA_LABELS[0]]["base"]["n"]))
        # Pre-registered D / C / a_n matrices.
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
        # Pre-registered variance decomposition.
        v = var[pset]
        for key in ("stage", "grader", "interaction"):
            cap = key.capitalize()
            add(f"{ps}Var{cap}Pct", pct1(v["cell_means_all_graders"]["frac"][key]))
            add(f"{ps}Var{cap}InFamPct", pct1(v["cell_means_olmo_only"]["frac"][key]))
            add(f"{ps}VarWithin{cap}Pct", pct1(v["per_prompt_all_graders"]["within_prompt_frac"][key]))
        add(f"{ps}VarPromptPct", pct1(v["per_prompt_all_graders"]["prompt_frac_of_total"]))

        # Extended tier.
        e = ext[pset]
        add(f"{ps}NCommonExt", str(e["n_prompts_common"]))
        for label in EXTENDED_LABELS:
            th = THETA_MACRO[label]
            row = e["new_row_cell_means"][label]
            for gen in MA.GEN_ORDER:
                add(f"{ps}D{th}{GEN_MACRO[gen]}", fmt3(row[gen]["D_mean"]))
                add(f"{ps}C{th}{GEN_MACRO[gen]}", fmt3(row[gen]["C_mean"]))
                add(f"{ps}An{th}{GEN_MACRO[gen]}", fmt3(row[gen]["a_n_mean"]))
            add(f"{ps}N{th}", str(row["base"]["n"]))
            r = e["new_row_R"][label]
            ext_R.append(r["R"])
            ext_R_lo.append(r["ci95"][0])
            add(f"{ps}R{th}", fmt2(r["R"]))
            add(f"{ps}RLo{th}", fmt2(r["ci95"][0]))
            add(f"{ps}RHi{th}", fmt2(r["ci95"][1]))
        for c in e["new_row_contrasts"]:
            ext_n_contrasts += 1
            ext_p_holm.append(c["p_holm"])
            if c["p_holm"] < 0.05:
                ext_n_sig += 1
            add(
                f"{ps}Pexp{THETA_MACRO[c['theta']]}{STEP_MACRO[c['step']]}",
                fmt_p(c["p_holm"]),
            )
        for key in ("stage", "grader", "interaction"):
            cap = key.capitalize()
            add(f"{ps}VarExt{cap}Pct",
                pct1(e["all_graders_common"]["cell_means"]["frac"][key]))
            add(f"{ps}VarExtWithin{cap}Pct",
                pct1(e["all_graders_common"]["per_prompt"]["within_prompt_frac"][key]))
            add(f"{ps}VarExtNoGptTwo{cap}Pct",
                pct1(e["no_gpt2_full"]["cell_means"]["frac"][key]))

    add("NContrasts", str(n_contrasts))
    add("NContrastsSig", str(n_sig))
    add("MaxPholm", fmt_p(max(all_p_holm)))
    add("MinR", fmt2(min(all_R)))
    add("MinRLo", fmt2(min(all_R_lo)))
    add("NGradersExt", str(len(ALL_LABELS)))
    add("NContrastsExt", str(ext_n_contrasts))
    add("NContrastsExtSig", str(ext_n_sig))
    add("MaxPexp", fmt_p(max(ext_p_holm)))
    add("MinRExt", fmt2(min(ext_R)))
    add("MinRLoExt", fmt2(min(ext_R_lo)))
    return "\n".join(lines) + "\n"


def emit_markdown(summary: dict, var: dict, ext: dict) -> dict[str, str]:
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

    # Extended rows (own complete prompt subsets), AlpacaEval.
    lines = ["| θ (grader) | n | base | sft | dpo | instruct | base>sft>dpo |", "|---|---|---|---|---|---|---|"]
    for label in EXTENDED_LABELS:
        row = ext["alpacaeval"]["new_row_cell_means"][label]
        vals = [row[g]["D_mean"] for g in MA.GEN_ORDER]
        mono = "✓" if ext["alpacaeval"]["monotone"][label] else "✗"
        cells = " | ".join(fmt3(v) for v in vals)
        lines.append(f"| {label} | {row['base']['n']} | {cells} | {mono} |")
    blocks["extended-d-matrix-alpacaeval"] = "\n".join(lines)

    # R_theta fold-change table, both prompt sets, all rows.
    lines = ["| θ | R (AlpacaEval) | 95% CI | R (NB) | 95% CI |", "|---|---|---|---|---|"]
    for label in THETA_LABELS:
        cells = []
        for pset in MA.PROMPT_SETS:
            r = summary["prompt_sets"][pset]["R_theta"][label]
            cells.append(fmt2(r["R"]))
            cells.append(f"[{fmt2(r['ci95'][0])}, {fmt2(r['ci95'][1])}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for label in EXTENDED_LABELS:
        cells = []
        for pset in MA.PROMPT_SETS:
            r = ext[pset]["new_row_R"][label]
            cells.append(fmt2(r["R"]))
            cells.append(f"[{fmt2(r['ci95'][0])}, {fmt2(r['ci95'][1])}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    blocks["r-table"] = "\n".join(lines)

    # Variance decomposition, both tiers, both prompt sets.
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
        lines.append(f"| {display} (5 graders) | " + " | ".join(cells) + " |")
    for display, key in [
        ("Generator stage", "stage"),
        ("Grader θ", "grader"),
        ("Grader × stage interaction", "interaction"),
    ]:
        cells = []
        for pset in MA.PROMPT_SETS:
            e = ext[pset]["all_graders_common"]
            cells.append(pct1(e["cell_means"]["frac"][key]) + "%")
            cells.append(pct1(e["per_prompt"]["within_prompt_frac"][key]) + "%")
        lines.append(f"| {display} (7 graders) | " + " | ".join(cells) + " |")
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

def run(output_root: Path, update_findings: bool, n_boot: int = 10000, seed: int = 42) -> dict:
    summary_path = REPO_ROOT / "results" / "rlhf_experiment" / "matrix" / "matrix_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found; run 6_matrix_analysis.py first")
    summary = json.loads(summary_path.read_text())
    if summary.get("missing_theta"):
        raise ValueError(f"matrix_summary.json reports missing theta rows: {summary['missing_theta']}")

    theta_cells = load_theta_cells(list(MA.THETA_ROWS) + EXTENDED_THETA_ROWS)

    var: dict = {}
    ext: dict = {}
    for pset in MA.PROMPT_SETS:
        canonical = canonical_prompt_ids(theta_cells, pset)
        X = build_tensor(theta_cells, pset, THETA_LABELS, sorted(canonical))
        check_against_summary(X, summary, pset)
        var[pset] = decompose_preregistered(X)
        ext[pset] = analyze_extended(theta_cells, pset, canonical, n_boot, seed)

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
        "extended_theta_order": EXTENDED_LABELS,
        "gen_order": list(MA.GEN_ORDER),
        "prompt_sets": var,
        "extended": ext,
    }
    (matrix_dir / "variance_decomposition.json").write_text(json.dumps(variance_out, indent=2))
    (tables_dir / "olmo_matrix_D.tex").write_text(emit_tex_D(summary, ext))
    (tables_dir / "olmo_matrix_R.tex").write_text(emit_tex_R(summary, ext))
    (tables_dir / "olmo_matrix_variance.tex").write_text(emit_tex_variance(var, ext))
    macros = emit_macros(summary, var, ext)
    (output_root / "results" / "rlhf_experiment" / "paper_macros_matrix.tex").write_text(macros)

    blocks = emit_markdown(summary, var, ext)
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
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    variance_out = run(args.output_root, args.update_findings, args.n_boot, args.seed)
    for pset in MA.PROMPT_SETS:
        cm = variance_out["prompt_sets"][pset]["cell_means_all_graders"]["frac"]
        ecm = variance_out["extended"][pset]["all_graders_common"]["cell_means"]["frac"]
        print(
            f"[{pset}] pre-registered (5g) cell-mean SS: stage {100*cm['stage']:.1f}%"
            f" / grader {100*cm['grader']:.1f}% / interaction {100*cm['interaction']:.1f}%"
        )
        print(
            f"[{pset}] extended (7g, n_common="
            f"{variance_out['extended'][pset]['n_prompts_common']}) cell-mean SS:"
            f" stage {100*ecm['stage']:.1f}% / grader {100*ecm['grader']:.1f}%"
            f" / interaction {100*ecm['interaction']:.1f}%"
        )


if __name__ == "__main__":
    main()
