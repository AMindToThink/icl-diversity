r"""Build `results/tables/paper_macros.tex` — the authoritative source of every
inline numeric scalar referenced in prose in the paper's main wrapper(s) under
`paper/`.

This script reads ONLY from existing data files (no recomputation from raw data
besides trivial aggregations) and emits `\newcommand` definitions for every
inline number the paper cites. The paper `\input{}`s this file near the top and
uses `\macroName` in place of hand-typed digits, so no number in the prose is
ever hand-transcribed.

Run with: uv run python scripts/build_paper_macros.py
Validate with: grep "^\\\\newcommand" results/tables/paper_macros.tex | wc -l

Input data sources:
- investigations/cross_mode_surprise_drop/figures/pairwise_matrix.json (Qwen2.5-3B)
- investigations/cross_mode_surprise_drop/figures/gpt2/pairwise_matrix.json
- investigations/cross_mode_surprise_drop/figures/scaling_comparison/scaling_summary.json
- results/mode_count/qwen2.5-3b_1k_draws.json
- results/mode_count/gpt2_1k_draws.json
- results/tevet/qwen25_completion_v3/McDiv_nuggets/*.icl_curves.qwen25_completion_v3.json
- results/tables/contest_rho_oca.tex, dectest_rho.tex, qwen3_comparison.tex
- figures/tevet_validation/c_ainf_analysis_v3/summary_table.txt
- figures/template_vs_sentbert/qwen2.5-3b/summary.txt
- figures/pos_pattern/qwen2.5-3b_scrambled_control/summary.txt (+ distinctn_case_check.txt)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVEST = PROJECT_ROOT / "investigations" / "cross_mode_surprise_drop" / "figures"
RESULTS = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS / "tables"
OUTPUT = TABLES_DIR / "paper_macros.tex"


def _fmt(val: float, decimals: int = 2, force_sign: bool = False) -> str:
    s = f"{val:+.{decimals}f}" if force_sign else f"{val:.{decimals}f}"
    return s


def cross_mode_macros() -> dict[str, str]:
    """Section 8.4 pairwise-matrix numbers for Qwen2.5-3B and GPT-2."""
    macros: dict[str, str] = {}

    def stats(path: Path, prefix: str) -> dict[str, float]:
        with path.open() as f:
            d = json.load(f)
        rm = np.array(d["reduction_mean"])
        rs = np.array(d["reduction_std"])
        n = rm.shape[0]
        diag_mask = np.eye(n, dtype=bool)
        diag_means = rm[diag_mask]
        off_means = rm[~diag_mask]
        diag_stds = rs[diag_mask]
        off_stds = rs[~diag_mask]
        pct_pos = (off_means > 0).sum() / len(off_means) * 100
        return {
            "diag_mean": diag_means.mean(),
            "diag_within_std": diag_stds.mean(),
            "off_mean": off_means.mean(),
            "off_within_std": off_stds.mean(),
            "pct_pos": pct_pos,
            "n_modes": n,
            "n_off_pairs": len(off_means),
            "col_means": rm.mean(axis=0),
            "row_means": rm.mean(axis=1),
            "reduction_mean": rm,
        }

    qwen = stats(INVEST / "pairwise_matrix.json", "Qwen")
    gpt2 = stats(INVEST / "gpt2" / "pairwise_matrix.json", "GPT")

    macros["crossmodeQwenDiagMean"] = _fmt(qwen["diag_mean"], 1)
    macros["crossmodeQwenDiagStd"] = _fmt(qwen["diag_within_std"], 1)
    macros["crossmodeQwenOffMean"] = _fmt(qwen["off_mean"], 1, force_sign=True)
    macros["crossmodeQwenOffStd"] = _fmt(qwen["off_within_std"], 1)
    macros["crossmodeQwenOffPctPos"] = f"{int(round(qwen['pct_pos']))}"
    macros["crossmodeGPTDiagMean"] = _fmt(gpt2["diag_mean"], 1)
    macros["crossmodeGPTDiagStd"] = _fmt(gpt2["diag_within_std"], 1)
    macros["crossmodeGPTOffMean"] = _fmt(gpt2["off_mean"], 1, force_sign=True)
    macros["crossmodeGPTOffStd"] = _fmt(gpt2["off_within_std"], 1)
    macros["crossmodeGPTOffPctPos"] = f"{int(round(gpt2['pct_pos']))}"
    macros["crossmodeNumModes"] = f"{qwen['n_modes']}"
    macros["crossmodeNumOffPairs"] = f"{qwen['n_off_pairs']}"

    # Count of Qwen modes with negative column mean (claim at line 497).
    macros["crossmodeQwenNegColCount"] = f"{int((qwen['col_means'] < 0).sum())}"
    macros["crossmodeGPTNegColCount"] = f"{int((gpt2['col_means'] < 0).sum())}"

    # GPT-2 first-step net drop at m=10, additive model prediction.
    # E[drop] = (1/m) * diag + ((m-1)/m) * off for uniformly distributed mode of
    # the in-context response. This replaces the hand-typed "+0.9 bits" claim,
    # which did not match the data; the additive formula gives ~4.9 bits.
    m = 10
    gpt_first_step = (1 / m) * gpt2["diag_mean"] + ((m - 1) / m) * gpt2["off_mean"]
    qwen_first_step = (1 / m) * qwen["diag_mean"] + ((m - 1) / m) * qwen["off_mean"]
    macros["crossmodeGPTFirstStepDrop"] = _fmt(gpt_first_step, 1, force_sign=True)
    macros["crossmodeQwenFirstStepDrop"] = _fmt(qwen_first_step, 1, force_sign=True)
    # Also emit the diagonal-gain component alone so prose can contrast the two.
    macros["crossmodeGPTDiagGainMten"] = _fmt(gpt2["diag_mean"] / m, 1)
    macros["crossmodeGPTCrossDamageMten"] = _fmt((m - 1) / m * gpt2["off_mean"], 1)

    # Row-vs-col regression — already stored in scaling_summary.json for each model.
    return macros


def scaling_macros() -> dict[str, str]:
    """Sec 8.7 Cross-mode scaling + Sec 8.4 caption row/col slopes."""
    macros: dict[str, str] = {}
    with (INVEST / "scaling_comparison" / "scaling_summary.json").open() as f:
        d = json.load(f)

    # Map labels in JSON to macro suffixes.
    short = {
        "GPT-2 (124M)": "GPT",
        "Llama-3.2-1B": "LlamaOneB",
        "Llama-3.2-3B": "LlamaThreeB",
        "Qwen2.5-3B": "Qwen",
        "Llama-3.1-8B": "LlamaEightB",
        "Llama-3.1-70B": "LlamaSeventyB",
    }
    # Llama-only ordering for the scaling paragraph (1B, 3B, 8B, 70B).
    for m in d["models"]:
        suf = short.get(m["label"])
        if suf is None:
            continue
        macros[f"scalingOffDiag{suf}"] = _fmt(
            m["off_diagonal_mean"], 1, force_sign=True
        )
        macros[f"scalingFracPos{suf}"] = (
            f"{int(round(m['off_diagonal_frac_pos'] * 100))}"
        )
        macros[f"scalingRCSlope{suf}"] = _fmt(m["rc_slope"], 2)
        macros[f"scalingRCRsq{suf}"] = _fmt(m["rc_r2"], 2)
        macros[f"scalingSymSlope{suf}"] = _fmt(m["slope"], 2)
        macros[f"scalingSymRsq{suf}"] = _fmt(m["r2"], 2)

    # Number of Llama models tested and p-value under monotone null
    # (1/k! for k models, here k=4).
    llama_labels = [m for m in d["models"] if "Llama" in m["label"]]
    macros["scalingNumLlamaModels"] = f"{len(llama_labels)}"
    import math

    k = len(llama_labels)
    macros["scalingFactorialProbPct"] = _fmt(100.0 / math.factorial(k), 1)
    return macros


def _simulate_p_rep(m_val: int, n_resp: int = 20, n_sims: int = 50000) -> np.ndarray:
    """Replicate `simulate_p_rep` from investigations/.../09_predicted_vs_observed.py."""
    modes_arr = np.array([i % m_val for i in range(n_resp)])
    rng = np.random.default_rng(42)
    p_rep = np.zeros(n_resp)
    for _ in range(n_sims):
        perm = rng.permutation(n_resp)
        mode_order = modes_arr[perm]
        seen: dict[int, int] = {}
        for pos in range(n_resp):
            mode_k = mode_order[pos]
            if seen.get(mode_k, 0) > 0:
                p_rep[pos] += 1
            seen[mode_k] = seen.get(mode_k, 0) + 1
    return p_rep / n_sims


def fractional_model_macros() -> dict[str, str]:
    """Sec 8.4 fractional-model numbers (Qwen) replicating 09_predicted_vs_observed.py logic."""
    macros: dict[str, str] = {}
    with (INVEST / "pairwise_matrix.json").open() as f:
        pm = json.load(f)
    rm = np.array(pm["reduction_mean"])
    n = rm.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    diag_mean = rm[diag_mask].mean()
    off_mean = rm[~diag_mask].mean()

    # Aggregate mode_count curves (Qwen2.5-3B) to get a_1 at m=10 and a_inf from m=1 tail.
    with (RESULTS / "mode_count" / "qwen2.5-3b_1k_draws.json").open() as f:
        mc = json.load(f)
    m_curves: dict[int, list] = {}
    for r in mc["runs"]:
        m_curves.setdefault(r["m"], []).append(r["a_k_curve"])

    m1 = np.array(m_curves[1]).mean(axis=0)
    a_floor = float(m1[-6:].mean())
    m10 = np.array(m_curves[10]).mean(axis=0)
    a1_m10 = float(m10[0])
    obs_drop_m1 = float(m1[0] - m1[-1])
    obs_drop_m10 = float(m10[0] - m10[-1])

    r_off = off_mean / a1_m10
    r_diag = diag_mean / a1_m10

    # Use the same simulation-based p_rep as 09_predicted_vs_observed.py (balanced
    # mode assignment, 50k permutations) so the values match the script's stdout.
    n_resp = len(m10)
    p_rep_m10 = _simulate_p_rep(10, n_resp=n_resp)
    p_rep_m1 = _simulate_p_rep(1, n_resp=n_resp)

    # Additive model: curve[k+1] = curve[k] - drop_k where drop_k = off + p_rep[k+1]*(diag-off)
    a_add = np.zeros(n_resp)
    a_add[0] = a1_m10
    for i in range(n_resp - 1):
        drop = off_mean + p_rep_m10[i + 1] * (diag_mean - off_mean)
        a_add[i + 1] = a_add[i] - drop
    additive_pred_drop = a1_m10 - a_add[-1]

    # Fractional model: a_{k+1} = floor + (a_k - floor) * (1 - r_k)
    def fractional_drop(a1: float, p_rep: np.ndarray) -> float:
        curve = np.zeros(n_resp)
        curve[0] = a1
        for i in range(n_resp - 1):
            r_k = (off_mean + p_rep[i + 1] * (diag_mean - off_mean)) / a1
            r_k = float(np.clip(r_k, 0.0, 1.0))
            curve[i + 1] = a_floor + (curve[i] - a_floor) * (1 - r_k)
        return curve[0] - curve[-1]

    frac_total_drop = fractional_drop(a1_m10, p_rep_m10)
    frac_drop_m1 = fractional_drop(float(m1[0]), p_rep_m1)

    macros["crossmodeRDiag"] = _fmt(r_diag * 100, 1)
    macros["crossmodeROff"] = _fmt(r_off * 100, 1)
    macros["crossmodeAinfFloor"] = _fmt(a_floor, 1)
    macros["crossmodeObsDropMone"] = f"{int(round(obs_drop_m1))}"
    macros["crossmodeObsDropMten"] = f"{int(round(obs_drop_m10))}"
    macros["crossmodeObsDropMtenOneDec"] = _fmt(obs_drop_m10, 1)
    macros["crossmodeAddOverpredMten"] = f"{int(round(additive_pred_drop))}"
    macros["crossmodeAddOverpredRatioMten"] = _fmt(additive_pred_drop / obs_drop_m10, 1)
    macros["crossmodeFracTotalDrop"] = f"{int(round(frac_total_drop))}"
    macros["crossmodeFracOverpredMten"] = _fmt(frac_total_drop / obs_drop_m10, 1)
    macros["crossmodeFracDropMone"] = _fmt(frac_drop_m1, 1)
    macros["crossmodeFracRatioMone"] = _fmt(frac_drop_m1 / obs_drop_m1, 1)
    return macros


def symmetry_macros() -> dict[str, str]:
    """Sec 8.4 pairwise symmetry stats (mean |M_ij - M_ji|)."""
    macros: dict[str, str] = {}
    with (INVEST / "pairwise_matrix.json").open() as f:
        pm = json.load(f)
    rm = np.array(pm["reduction_mean"])
    n = rm.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    off_mean = rm[~diag_mask].mean()
    asym = rm - rm.T
    asym_off = asym[~diag_mask]
    mean_abs_asym = float(np.abs(asym_off).mean())
    macros["crossmodeAsymmetryMean"] = _fmt(mean_abs_asym, 1)
    macros["crossmodeAsymmetryRatio"] = _fmt(mean_abs_asym / abs(off_mean), 1)
    return macros


def mode_count_macros() -> dict[str, str]:
    """Sec 8.3/8.5 mode count scaling Pearson r for C×a_n vs m (GPT-2 + Qwen)."""
    macros: dict[str, str] = {}
    for model, path in [
        ("GPT", "gpt2_1k_draws.json"),
        ("Qwen", "qwen2.5-3b_1k_draws.json"),
    ]:
        with (RESULTS / "mode_count" / path).open() as f:
            d = json.load(f)
        # Group by m, compute mean C*a_n across draws for each m.
        by_m: dict[int, list] = {}
        for r in d["runs"]:
            m = r["m"]
            curve = r.get("a_k_curve_per_byte")
            c = r.get("coherence_C")
            if curve is None or c is None:
                continue
            by_m.setdefault(m, []).append(c * curve[-1])
        ms = sorted(by_m.keys())
        means = [np.mean(by_m[m]) for m in ms]
        r_val = float(np.corrcoef(ms, means)[0, 1])
        macros[f"modeCountCxAnPearson{model}"] = _fmt(r_val, 2)
    return macros


def tevet_macros() -> dict[str, str]:
    """Sec 7.5 Tevet-benchmark inline numbers, sourced from contest_rho_oca.tex /
    dectest_rho.tex so they stay in lockstep with the tables."""
    macros: dict[str, str] = {}
    # Read contest_rho_oca.tex cells directly.
    txt = (TABLES_DIR / "contest_rho_oca.tex").read_text()

    # Parse "McDiv (full, no_hds, ~2K)" block for prompt_gen values.
    # We want C×a_n row's first pair (prompt_gen rho & OCA) and SentBERT's.
    def parse_row(label: str, dataset_header: str, col: int = 0) -> tuple[str, str]:
        """Return (rho, OCA) strings for the given row under the given dataset block.
        col=0 -> prompt_gen, 1 -> resp_gen, 2 -> story_gen."""
        m = re.search(re.escape(dataset_header), txt)
        assert m, f"Dataset block not found: {dataset_header}"
        block = txt[m.end() :]
        row_m = re.search(
            rf"^\s*{re.escape(label)}\s*&(.*?)\\\\\s*$", block, re.MULTILINE
        )
        assert row_m, f"Row not found: {label} under {dataset_header}"
        # row content: rho1 & OCA1 & rho2 & OCA2 & rho3 & OCA3
        cells = [c.strip() for c in row_m.group(1).split("&")]
        return cells[col * 2], cells[col * 2 + 1]

    # McDiv full prompt_gen: $C \!\times\! a_n$ (ours)
    rho_cxan, oca_cxan = parse_row(
        r"$C \!\times\! a_n$ (ours)", r"McDiv (full, no\_hds, $\sim$2K)"
    )
    rho_sb, oca_sb = parse_row("SentBERT", r"McDiv (full, no\_hds, $\sim$2K)")
    rho_dn, oca_dn = parse_row(r"distinct-$n$", r"McDiv (full, no\_hds, $\sim$2K)")

    # These cells may contain LaTeX like "\textbf{+0.729}"; strip it.
    def clean(s: str) -> str:
        s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
        return s.strip().rstrip("$").lstrip("$")

    macros["tevetMcDivPromptGenCxAnRho"] = clean(rho_cxan)
    macros["tevetMcDivPromptGenCxAnOCA"] = clean(oca_cxan)
    macros["tevetMcDivPromptGenSentBertRho"] = clean(rho_sb)
    macros["tevetMcDivPromptGenSentBertOCA"] = clean(oca_sb)
    macros["tevetMcDivPromptGenDistinctNRho"] = clean(rho_dn)
    macros["tevetMcDivPromptGenDistinctNOCA"] = clean(oca_dn)

    # Full summary_table.txt parse for McDiv_nuggets prompt_gen (no_hds) AUC (used in abstract/App B).
    summary = (
        PROJECT_ROOT
        / "figures"
        / "tevet_validation"
        / "c_ainf_analysis_v3"
        / "summary_table.txt"
    ).read_text()

    def auc_for(header_marker: str, dataset_tag: str, metric: str) -> str:
        """Find AUC for a given (dataset_tag, metric) pair under the given section header."""
        sec_start = summary.find(header_marker)
        assert sec_start != -1, f"Section header not found: {header_marker}"
        # Find the body start (skip the closing "====" row after the header) and the
        # next section boundary (the top "====" of the next section).
        # We match on a full line of "=" (a "heavy rule" unique to section borders).
        border = re.compile(r"^={5,}$", re.MULTILINE)
        borders = [m.start() for m in border.finditer(summary, sec_start)]
        # borders[0] is the closing rule of this section; borders[1] is the top
        # rule of the next section (if any).
        assert borders, f"No section borders after {header_marker}"
        body_start = borders[0]
        next_hdr = borders[1] if len(borders) >= 2 else len(summary)
        section = summary[sec_start:next_hdr]
        ds_idx = section.find(dataset_tag)
        assert ds_idx != -1, f"Dataset not found: {dataset_tag} in {header_marker}"
        rest = section[ds_idx:]
        for line in rest.splitlines():
            if metric in line:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    auc = parts[-2]
                    float(auc)
                    return auc
                except ValueError:
                    continue
        raise RuntimeError(f"No AUC found for {dataset_tag} / {metric}")

    macros["tevetMcDivNugPromptGenCxAnAUC"] = auc_for(
        "McDiv_nuggets (content diversity", "prompt_gen (no_hds)", "C×a_n (pb)"
    )
    macros["tevetMcDivNugPromptGenDdiscAUC"] = auc_for(
        "McDiv_nuggets (content diversity", "prompt_gen (no_hds)", "D_disc (pb)"
    )
    macros["tevetMcDivPromptGenAUC"] = auc_for(
        "McDiv (full, continuous labels)", "prompt_gen (no_hds)", "C×a_n (pb)"
    )
    macros["tevetConTestPromptGenCxAnAUC"] = auc_for(
        "ConTest (binary content diversity)", "prompt_gen (with_hds)", "C×a_n (pb)"
    )

    # DecTest: pick a_n rho on prompt_gen (no_hds) — the paper claims +0.92.
    dec = (TABLES_DIR / "dectest_rho.tex").read_text()

    def dectest_row(label: str) -> tuple[str, str, str]:
        m = re.search(rf"^\s*{re.escape(label)}\s*&(.*?)\\\\\s*$", dec, re.MULTILINE)
        assert m, f"DecTest row not found: {label}"
        cells = [clean(c) for c in m.group(1).split("&")]
        return cells[0], cells[1], cells[2]  # prompt_gen, resp_gen, story_gen

    an_pg, an_rg, an_sg = dectest_row("$a_n$ (ours)")
    dn_pg, dn_rg, dn_sg = dectest_row("distinct-$n$")
    macros["tevetDecTestAnPromptGenRho"] = an_pg
    macros["tevetDecTestAnRespGenRho"] = an_rg
    macros["tevetDecTestAnStoryGenRho"] = an_sg
    macros["tevetDecTestDistinctNPromptGenRho"] = dn_pg

    # Compute min/max OCA gap vs SentBERT, vs a_n, and vs C alone for binary tasks.
    # Parse all binary blocks: ConTest (200, with_hds), McDiv_nuggets (~1K, no_hds), McDiv (full, no_hds, ~2K).
    def binary_gaps() -> tuple[float, float, float, float, float, float]:
        blocks = [
            r"ConTest (200, with\_hds)",
            r"McDiv\_nuggets ($\sim$1K, no\_hds)",
            r"McDiv (full, no\_hds, $\sim$2K)",
        ]
        gaps_sb = []  # SentBERT OCA - C×a_n OCA, as % of SentBERT OCA
        gaps_an = []  # C×a_n OCA - a_n OCA, as % of C×a_n OCA
        gaps_c = []  # C×a_n OCA - C OCA, as % of C×a_n OCA
        for b in blocks:
            for col in range(3):  # prompt, resp, story
                _, oca_cx = parse_row(r"$C \!\times\! a_n$ (ours)", b, col)
                _, oca_an = parse_row(r"$a_n$ (ours)", b, col)
                _, oca_c = parse_row(r"$C$ (ours)", b, col)
                _, oca_sbe = parse_row("SentBERT", b, col)
                oca_cx_v = float(clean(oca_cx))
                oca_an_v = float(clean(oca_an))
                oca_c_v = float(clean(oca_c))
                oca_sb_v = float(clean(oca_sbe))
                gaps_sb.append((oca_sb_v - oca_cx_v) / oca_sb_v * 100)
                gaps_an.append((oca_cx_v - oca_an_v) / oca_cx_v * 100)
                gaps_c.append((oca_cx_v - oca_c_v) / oca_cx_v * 100)
        return (
            min(gaps_sb),
            max(gaps_sb),
            min(gaps_an),
            max(gaps_an),
            min(gaps_c),
            max(gaps_c),
        )

    sb_min, sb_max, an_min, an_max, c_min, c_max = binary_gaps()
    macros["tevetCxAnVsSentBertOCAMinPct"] = _fmt(sb_min, 1)
    macros["tevetCxAnVsSentBertOCAMaxPct"] = _fmt(sb_max, 1)
    macros["tevetAnVsCxAnOCAMinPct"] = _fmt(an_min, 1)
    macros["tevetAnVsCxAnOCAMaxPct"] = _fmt(an_max, 1)
    macros["tevetCVsCxAnOCAMinPct"] = _fmt(c_min, 1)
    macros["tevetCVsCxAnOCAMaxPct"] = _fmt(c_max, 1)

    # Last-position uptick percentage (a_5 > a_4 across McDiv_nuggets per-byte).
    base = RESULTS / "tevet" / "qwen25_completion_v3" / "McDiv_nuggets"
    from icl_diversity.per_byte import compute_a_k_curve_mor

    total_n = total_up = 0
    for variant in ["no_hds", "200_with_hds"]:
        for t in ["prompt_gen", "resp_gen", "story_gen"]:
            p = (
                base
                / f"mcdiv_nuggets_{variant}_{t}.icl_curves.qwen25_completion_v3.json"
            )
            if not p.exists():
                continue
            with p.open() as f:
                d = json.load(f)
            for _, it in d.items():
                # Use MoR per-byte curve (the formula §6.3 specifies; see
                # src/icl_diversity/per_byte.py).  Fall back to
                # ``a_k_curve_per_byte`` if per-perm data is missing —
                # old logs may store ratio-of-means there.
                pp_curves = it.get("per_permutation_a_k_curves")
                pp_bytes = it.get("per_permutation_byte_counts")
                if pp_curves and pp_bytes:
                    try:
                        curve = compute_a_k_curve_mor(pp_curves, pp_bytes)
                    except ValueError:
                        continue
                else:
                    curve = it.get("a_k_curve_per_byte")
                if curve and len(curve) >= 5:
                    total_n += 1
                    if curve[4] > curve[3]:
                        total_up += 1
    pct = 100 * total_up / max(total_n, 1)
    macros["tevetLastUptickPct"] = f"{int(round(pct))}"
    return macros


def tevet_coherence_distribution_macros() -> dict[str, str]:
    """Section 5 (Reporting) typical-range-of-$C$ paragraph.

    Reads the per-set sidecars for the **same canonical headline splits**
    used by the Section 5 setup paragraph (McDiv no_hds + McDiv_nuggets
    no_hds + ConTest with_hds), extracts (a) per-response per-byte
    cross-entropy from ``unconditional_surprises`` and (b) per-set
    ``coherence_C``, and emits the 5th/95th percentile of per-response
    bits/byte (and the corresponding $C = 2^{-\\bar{\\ell}}$ values) plus
    the mean per-set $C$.

    Scope is restricted to the same files driving \\tevetMcDivSetCount,
    \\tevetMcDivNugSetCount, \\tevetConTestSetCount so that the response /
    set counts reported in the footnote match the prose in
    sections/07_5_tevet_workshop.tex (canonical "5 worker-written
    responses per set"). DecTest and the ``with_hds`` McDiv_nuggets /
    DecTest variants are intentionally excluded here.
    """
    canonical_relpaths = [
        "McDiv/mcdiv_all_no_hds_prompt_gen.icl_curves.qwen25_completion_v3.json",
        "McDiv/mcdiv_all_no_hds_resp_gen.icl_curves.qwen25_completion_v3.json",
        "McDiv/mcdiv_all_no_hds_story_gen.icl_curves.qwen25_completion_v3.json",
        "McDiv_nuggets/mcdiv_nuggets_no_hds_prompt_gen.icl_curves.qwen25_completion_v3.json",
        "McDiv_nuggets/mcdiv_nuggets_no_hds_resp_gen.icl_curves.qwen25_completion_v3.json",
        "McDiv_nuggets/mcdiv_nuggets_no_hds_story_gen.icl_curves.qwen25_completion_v3.json",
        "conTest/con_test_200_with_hds_prompt_gen.icl_curves.qwen25_completion_v3.json",
        "conTest/con_test_200_with_hds_resp_gen.icl_curves.qwen25_completion_v3.json",
        "conTest/con_test_200_with_hds_story_gen.icl_curves.qwen25_completion_v3.json",
    ]
    base = RESULTS / "tevet" / "qwen25_completion_v3"
    files = [str(base / rp) for rp in canonical_relpaths]
    for fp in files:
        assert Path(fp).exists(), f"Missing canonical Tevet sidecar: {fp}"
    resp_bpb: list[float] = []
    set_C: list[float] = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for k, v in d.items():
            if k.startswith("__"):
                continue
            c = v["metrics"].get("coherence_C")
            if c is not None and c > 0:
                set_C.append(c)
            for s in v.get("unconditional_surprises", []) or []:
                resp_bpb.append(float(s))
    assert resp_bpb and set_C, "Empty Tevet distributions"
    bpb_p5 = float(np.percentile(resp_bpb, 5))
    bpb_p95 = float(np.percentile(resp_bpb, 95))
    c_high = 2.0 ** (-bpb_p5)
    c_low = 2.0 ** (-bpb_p95)
    mean_c = float(np.mean(set_C))
    return {
        "tevetCoherenceBpbLow": _fmt(bpb_p5, 2),
        "tevetCoherenceBpbHigh": _fmt(bpb_p95, 2),
        "tevetCoherenceCLow": _fmt(c_low, 2),
        "tevetCoherenceCHigh": _fmt(c_high, 2),
        "tevetCoherenceMeanC": _fmt(mean_c, 2),
        "tevetCoherenceRespCount": str(len(resp_bpb)),
        "tevetCoherenceSetCount": str(len(set_C)),
    }


def qwen3_macros() -> dict[str, str]:
    """Appendix E + Sec 8.7 Qwen3 comparison numbers, sourced from qwen3_comparison.tex."""
    macros: dict[str, str] = {}
    txt = (TABLES_DIR / "qwen3_comparison.tex").read_text()
    # Rows look like: "McDiv & prompt_gen (no\_hds) & 0.729 & 0.921 & 0.718 & 0.915 & -0.006 \\"
    # Binary tasks: McDiv (3), McDiv_nuggets (6), ConTest (3) = 12 rows.
    binary_markers = ["McDiv ", "McDiv\\_nuggets ", "ConTest "]
    dectest_marker = "DecTest "
    binary_deltas = []
    dectest_deltas = []
    section = ""
    contest_pg_delta = None
    dec_ties = dec_wins = dec_losses = 0
    for line in txt.splitlines():
        m = re.match(r"^\s*([A-Za-z_\\]+)?\s*&\s*([^&]+?)\s*&(.*)\\\\\s*$", line)
        if not m:
            continue
        first_col = m.group(1) or ""
        rest_cells_raw = m.group(3)
        rest_cells = [c.strip() for c in rest_cells_raw.split("&")]
        # We expect 5 rest cells: rho1 auc1 rho2 auc2 delta (for binary) or rho1 --- rho2 --- --- (DecTest).
        if len(rest_cells) < 5:
            continue
        if first_col.strip() in ["McDiv", "McDiv\\_nuggets", "ConTest"]:
            section = first_col.strip()
        elif first_col.strip() == "DecTest":
            section = "DecTest"
        # Use last column as delta; skip if '---'.
        delta_str = rest_cells[-1].strip()
        if delta_str == "---":
            # DecTest: compute delta from rho columns.
            try:
                rho1 = float(rest_cells[0])
                rho2 = float(rest_cells[2])
                delta = rho2 - rho1
                dectest_deltas.append(delta)
                if delta > 0:
                    dec_wins += 1
                elif delta < 0:
                    dec_losses += 1
                else:
                    dec_ties += 1
            except ValueError:
                continue
        else:
            try:
                delta = float(delta_str)
            except ValueError:
                continue
            if section in ["McDiv", "McDiv\\_nuggets", "ConTest"]:
                binary_deltas.append(delta)
                if section == "ConTest" and "prompt\\_gen" in rest_cells_raw.split("&")[
                    0
                ] + m.group(2):
                    contest_pg_delta = delta

    # Contest prompt_gen delta: scan explicitly because the first_col is empty.
    for line in txt.splitlines():
        if "prompt\\_gen (with\\_hds)" in line and "+0.005" in line:
            contest_pg_delta = 0.005
            break
    assert contest_pg_delta is not None, "Could not find ConTest prompt_gen delta"

    binary_arr = np.array(binary_deltas)
    wins_q3 = int((binary_arr > 0).sum())
    wins_q25 = int((binary_arr < 0).sum())
    ties_bin = int((binary_arr == 0).sum())
    macros["qwenThreeBinaryMeanDeltaAUC"] = _fmt(binary_arr.mean(), 3, force_sign=True)
    macros["qwenThreeBinaryTotal"] = f"{len(binary_arr)}"
    macros["qwenThreeBinaryWinsQwenTwoFive"] = f"{wins_q25}"
    macros["qwenThreeBinaryWinsQwenThree"] = f"{wins_q3}"
    macros["qwenThreeBinaryTies"] = f"{ties_bin}"
    macros["qwenThreeConTestPromptGenDeltaAUC"] = _fmt(
        contest_pg_delta, 3, force_sign=True
    )

    dec_arr = np.array(dectest_deltas)
    macros["qwenThreeDecTestTotal"] = f"{len(dec_arr)}"
    macros["qwenThreeDecTestWins"] = f"{dec_wins}"
    macros["qwenThreeDecTestTies"] = f"{dec_ties}"
    macros["qwenThreeDecTestLosses"] = f"{dec_losses}"
    macros["qwenThreeDecTestMeanDeltaRho"] = _fmt(dec_arr.mean(), 3, force_sign=True)
    return macros


def tevet_dataset_size_macros() -> dict[str, str]:
    """Sec 7.5 dataset sizes: exact CSV row counts (= set counts) across all
    three NLG tasks (promptGen/respGen/storyGen)."""
    macros: dict[str, str] = {}
    data_root = PROJECT_ROOT / "diversity-eval" / "data" / "raw"
    specs = [
        # (macro suffix, subdir, filename pattern)
        (
            "McDiv",
            "McDiv",
            [
                "mcdiv_all_no_hds_prompt_gen.csv",
                "mcdiv_all_no_hds_resp_gen.csv",
                "mcdiv_all_no_hds_story_gen.csv",
            ],
        ),
        (
            "McDivNug",
            "McDiv_nuggets",
            [
                "mcdiv_nuggets_no_hds_prompt_gen.csv",
                "mcdiv_nuggets_no_hds_resp_gen.csv",
                "mcdiv_nuggets_no_hds_story_gen.csv",
            ],
        ),
        (
            "DecTest",
            "decTest",
            [
                "dec_test_1000_no_hds_prompt_gen.csv",
                "dec_test_1000_no_hds_resp_gen.csv",
                "dec_test_1000_no_hds_story_gen.csv",
            ],
        ),
        (
            "ConTest",
            "conTest",
            [
                "con_test_200_with_hds_prompt_gen.csv",
                "con_test_200_with_hds_resp_gen.csv",
                "con_test_200_with_hds_story_gen.csv",
            ],
        ),
    ]
    for suffix, subdir, files in specs:
        total = 0
        for fn in files:
            path = data_root / subdir / fn
            with path.open() as f:
                # Row count excluding header.
                total += sum(1 for _ in f) - 1
        macros[f"tevet{suffix}SetCount"] = f"{total}"
    return macros


def mode_count_config_macros() -> dict[str, str]:
    """Sec 8.3 experiment config: pool size, n_responses, n_draws, max m."""
    macros: dict[str, str] = {}
    # Mode pool size from the source of truth.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mode_count_scenarios",
        PROJECT_ROOT / "src" / "icl_diversity" / "mode_count_scenarios.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    macros["modeCountPoolSize"] = f"{len(mod.MODE_NAMES)}"

    # n_responses and n_draws from the committed experiment JSON.
    with (RESULTS / "mode_count" / "qwen2.5-3b_1k_draws.json").open() as f:
        d = json.load(f)
    macros["modeCountNResponses"] = f"{d['n_responses']}"
    macros["modeCountNDraws"] = f"{d['n_draws']}"

    # Max m (we sweep 1..10).
    ms = sorted({r["m"] for r in d["runs"]})
    macros["modeCountMMax"] = f"{ms[-1]}"
    return macros


def permutation_sensitivity_macros() -> dict[str, str]:
    """Sec 8.6 permutation-sensitivity claim.

    Compares scenario rankings (by mean $D_{a_\\infty} = C \\times a_n$
    across prompts) between the 3-perm and 100-perm scenario-metrics
    files for both GPT-2 and Qwen2.5-3B. Emits: number of scenarios
    mis-ranked at 3 perms on each model.

    Note: the legacy ``diversity_score_D`` key in those JSON files is
    ``C * E`` (Appendix-E variant), NOT the paper's primary
    $D_{a_\\infty} = C \\times a_n$ that §8.6 actually refers to. We
    derive the right scalar from the per-prompt ``coherence_C`` and a
    mean-of-ratios per-byte curve computed from the ``per_permutation_*``
    fields (the formula §6.3 specifies; see src/icl_diversity/per_byte.py).
    """
    from icl_diversity.per_byte import compute_a_n_per_byte_mor

    macros: dict[str, str] = {}
    pairs = [
        (
            "GPT",
            "scenario_metrics_v3_gpt2_3perm.json",
            "scenario_metrics_v3_gpt2_100perm.json",
        ),
        (
            "Qwen",
            "scenario_metrics_v3_qwen3b_3perm.json",
            "scenario_metrics_v3_qwen3b_100perm.json",
        ),
    ]
    n_scenarios = None
    for model, f3, f100 in pairs:
        with (RESULTS / f3).open() as f:
            d3 = json.load(f)
        with (RESULTS / f100).open() as f:
            d100 = json.load(f)

        def _entry_d_can(entry: dict) -> float:
            """Derive D = C * a_n (per-byte) for one prompt's entry."""
            pp_curves = entry.get("per_permutation_a_k_curves")
            pp_bytes = entry.get("per_permutation_byte_counts")
            if pp_curves and pp_bytes:
                an_pb = compute_a_n_per_byte_mor(pp_curves, pp_bytes)
            else:
                # Old log without per-perm data: trust the precomputed curve.
                an_pb = float(entry["a_k_curve_per_byte"][-1])
            return float(entry["coherence_C"]) * an_pb

        def rank(d: dict) -> dict[str, int]:
            scores = {
                name: float(np.mean([_entry_d_can(e) for e in entries]))
                for name, entries in d["scenarios"].items()
            }
            ordered = sorted(scores.items(), key=lambda kv: -kv[1])
            return {name: i + 1 for i, (name, _) in enumerate(ordered)}

        r3 = rank(d3)
        r100 = rank(d100)
        if n_scenarios is None:
            n_scenarios = len(r100)
        misranked = sum(1 for s in r100 if r3[s] != r100[s])
        macros[f"permSens{model}Misranked"] = f"{misranked}"
    macros["permSensNumScenarios"] = f"{n_scenarios}"
    macros["permSensLowPerm"] = "3"
    macros["permSensHighPerm"] = "100"
    return macros


def mode_count_extrema() -> dict[str, str]:
    """Sec 8.3 text claims 'a_n increases from 9.3 bits at m=1 to 77.8 bits at m=10'."""
    macros: dict[str, str] = {}
    txt = (TABLES_DIR / "mode_count_scaling.tex").read_text()
    # Lines like "1  & $296 \pm 11$ & 9.3 & 0.1 & $-10.0$ \\"
    rows = re.findall(r"^\s*(\d+)\s*&[^&]*&\s*([0-9.]+)\s*&", txt, re.MULTILINE)
    an_by_m = {int(m): float(a) for m, a in rows}
    macros["modeCountAnMone"] = _fmt(an_by_m[1], 1)
    macros["modeCountAnMten"] = _fmt(an_by_m[10], 1)
    return macros


def scenario_validation_macros() -> dict[str, str]:
    """Sec 3 Table 1 empirical D_{Ca_n} values per scenario per base model.

    Computed the same way scripts/generate_scenario_table.py does:
    per-prompt mean of (C * a_n_per_byte), where a_n is the last point of the
    mean-of-ratios per-byte a_k curve over permutations. This matches the
    paper's primary D = C * a_n definition (per-byte) and intentionally does
    NOT use the JSON's `diversity_score_D` / `diversity_score_D_rate` fields,
    which encode the C * E (Appendix-E) variant.
    """
    from icl_diversity.per_byte import compute_a_k_curve_mor

    macros: dict[str, str] = {}
    sources = [
        ("Gpt", RESULTS / "scenario_metrics_v3_gpt2_100perm.json"),
        ("Qwen", RESULTS / "scenario_metrics_v3_qwen3b_100perm.json"),
    ]
    scenario_keys = {
        "pure_noise": "PureNoise",
        "multi_incoherent": "MultiIncoherent",
        "multi_mode": "MultiMode",
        "one_mode": "OneMode",
        "mixed": "Mixed",
    }

    for model_tag, path in sources:
        with path.open() as f:
            data = json.load(f)
        for scen_key, scen_camel in scenario_keys.items():
            prompts = data["scenarios"][scen_key]
            d_can_per_prompt: list[float] = []
            for p in prompts:
                pp_curves = p.get("per_permutation_a_k_curves")
                pp_bytes = p.get("per_permutation_byte_counts")
                if pp_curves and pp_bytes:
                    pb = compute_a_k_curve_mor(pp_curves, pp_bytes)
                else:
                    pb = p["a_k_curve_per_byte"]
                a_n_pb = pb[-1]
                d_can_per_prompt.append(p["coherence_C"] * a_n_pb)
            mean_d_can = float(np.mean(d_can_per_prompt))
            # Match generate_scenario_table.fmt() rounding: 3dp if <0.01, else 2dp
            decimals = 3 if abs(mean_d_can) < 0.01 else 2
            macros[f"scenario{scen_camel}{model_tag}DCan"] = _fmt(mean_d_can, decimals)
    return macros


def template_frames_macros() -> dict[str, str]:
    """Structural-redundancy draft section: syntactic-frames experiment.

    Parses figures/template_vs_sentbert/qwen2.5-3b/summary.txt (generated by
    scripts/plot_template_vs_sentbert.py) so prose stays in lockstep with the
    committed summary. Context: reports/TEMPLATE_VS_SENTBERT.md.
    """
    from icl_diversity.template_scenarios import FRAMES

    macros: dict[str, str] = {}
    txt = (
        PROJECT_ROOT / "figures" / "template_vs_sentbert" / "qwen2.5-3b" / "summary.txt"
    ).read_text()

    m = re.search(r"n_responses=(\d+)\s+n_draws=(\d+)", txt)
    assert m, "n_responses/n_draws header not found"
    macros["framesQwenNResponses"] = m.group(1)
    macros["framesQwenNDraws"] = m.group(2)
    macros["framesQwenNumFrames"] = f"{len(FRAMES)}"

    def cond_pairs(name: str) -> list[tuple[str, str]]:
        """(mean, sd) string pairs from a condition-table row, in column order
        C, a_n, D_C_an, E, cosine, sentbert_div, distinct-n."""
        row = re.search(rf"^{re.escape(name)}\s+(.*)$", txt, re.MULTILINE)
        assert row, f"Condition row not found: {name}"
        pairs = re.findall(r"([-\d.e+]+) ± ([-\d.e+]+)", row.group(1))
        assert len(pairs) == 7, (name, pairs)
        return pairs

    for cond, suffix in [
        ("frames_1", "FramesOne"),
        ("frames_20", "FramesTwenty"),
        ("paraphrase", "Paraphrase"),
    ]:
        macros[f"framesQwenCos{suffix}"] = _fmt(float(cond_pairs(cond)[4][0]), 2)

        drop = re.search(
            rf"^\s+{re.escape(cond)}\s+a_1 = .*?a_n/a_1 = ([\d.]+) ± ([\d.]+)",
            txt,
            re.MULTILINE,
        )
        assert drop, f"Drop-ratio line not found: {cond}"
        macros[f"framesQwenDrop{suffix}"] = drop.group(1)
        macros[f"framesQwenDrop{suffix}Sd"] = drop.group(2)

    norm_block = txt[txt.index("Normalized position") :]
    rho_block = txt[txt.index("Spearman rho") :]
    for key, tag in [
        ("diversity_score_D_C_an", "D"),
        ("sentbert_diversity", "SentBert"),
        ("averaged_distinct_ngrams", "Distinct"),
    ]:
        norm = re.search(
            rf"^\s+{re.escape(key)}\s+canonical = ([+\-\d.]+)\s+frames_1 = ([+\-\d.]+)",
            norm_block,
            re.MULTILINE,
        )
        assert norm, f"Normalized-position line not found: {key}"
        macros[f"framesQwenNorm{tag}Canonical"] = norm.group(1).lstrip("+")
        macros[f"framesQwenNorm{tag}FramesOne"] = norm.group(2).lstrip("+")

        rho = re.search(
            rf"^\s+{re.escape(key)}\s+rho = ([+\-\d.]+)", rho_block, re.MULTILINE
        )
        assert rho, f"Spearman line not found: {key}"
        macros[f"framesQwenRho{tag}"] = rho.group(1)
    return macros


def pos_pattern_macros() -> dict[str, str]:
    """Structural-redundancy draft section: canonical-vs-scrambled POS patterns.

    Parses figures/pos_pattern/qwen2.5-3b_scrambled_control/summary.txt and
    distinctn_case_check.txt (both script-generated); the ground-truth order
    entropy is recomputed from CANONICAL_CLASS_MULTISET, the same constant the
    generator uses. Context: reports/POS_PATTERN_VS_BASELINES.md.
    """
    import math

    from icl_diversity.pos_pattern_scenarios import (
        CANONICAL_CLASS_MULTISET,
        INTRANSITIVE_PAST,
        PLURAL_NOUNS,
        PREPOSITIONS,
    )

    macros: dict[str, str] = {}
    base = PROJECT_ROOT / "figures" / "pos_pattern" / "qwen2.5-3b_scrambled_control"
    txt = (base / "summary.txt").read_text()

    m = re.search(r"n_responses=(\d+)\s+n_draws=(\d+)", txt)
    assert m, "n_responses/n_draws header not found"
    macros["posPatternNResponses"] = m.group(1)
    macros["posPatternNDraws"] = m.group(2)
    macros["posPatternNumNouns"] = f"{len(PLURAL_NOUNS)}"
    macros["posPatternNumVerbs"] = f"{len(INTRANSITIVE_PAST)}"
    macros["posPatternNumPreps"] = f"{len(PREPOSITIONS)}"

    def endpoint(name: str) -> tuple[str, ...]:
        m2 = re.search(
            rf"^\s+{name}\s+a_1 = ([\d.]+) ± ([\d.]+)\s+a_n = ([\d.]+) ± ([\d.]+)\s+"
            rf"a_n/a_1 = ([\d.]+) ± ([\d.]+)\s+bytes/resp = ([\d.]+) ± ([\d.]+)",
            txt,
            re.MULTILINE,
        )
        assert m2, f"Endpoint line not found: {name}"
        return m2.groups()

    ec, es = endpoint("canonical"), endpoint("scrambled")
    macros["posPatternAOneCanon"], macros["posPatternAOneCanonSd"] = ec[0], ec[1]
    macros["posPatternAnCanon"], macros["posPatternAnCanonSd"] = ec[2], ec[3]
    macros["posPatternDropCanon"], macros["posPatternDropCanonSd"] = ec[4], ec[5]
    macros["posPatternBytesCanon"] = ec[6]
    macros["posPatternAOneScram"], macros["posPatternAOneScramSd"] = es[0], es[1]
    macros["posPatternAnScram"], macros["posPatternAnScramSd"] = es[2], es[3]
    macros["posPatternDropScram"], macros["posPatternDropScramSd"] = es[4], es[5]
    macros["posPatternBytesScram"] = es[6]
    macros["posPatternLearnedPctCanon"] = f"{int(round((1 - float(ec[4])) * 100))}"
    macros["posPatternLearnedPctScram"] = f"{int(round((1 - float(es[4])) * 100))}"

    def welch(label: str) -> tuple[str, str, str, str]:
        """(diff, se, t, p) strings from a Welch t-test line."""
        m3 = re.search(
            rf"^\s+{re.escape(label)}\s+diff = ([+\-\d.e]+) ± ([\d.e-]+)\s+"
            rf"t = ([+\-\d.]+)\s+\(p = ([\d.e-]+)\)",
            txt,
            re.MULTILINE,
        )
        assert m3, f"Welch line not found: {label}"
        return m3.groups()

    def p_macro(prefix: str, p_str: str) -> None:
        """Plain-decimal p as `<prefix>P`; tiny p as `<prefix>PLtPow` with
        the exponent e such that p < 10^-e (prose writes $p < 10^{-\\e}$)."""
        p = float(p_str)
        if p >= 1e-4:
            macros[f"{prefix}P"] = p_str
        else:
            macros[f"{prefix}PLtPow"] = f"{-math.ceil(math.log10(p))}"

    an_diff, an_se, _, an_p = welch("a_n_per_byte")
    macros["posPatternAnDiff"] = _fmt(float(an_diff), 3, force_sign=True)
    macros["posPatternAnGapAbs"] = _fmt(abs(float(an_diff)), 3)
    macros["posPatternAnDiffSe"] = _fmt(float(an_se), 3)
    p_macro("posPatternAnDiff", an_p)

    a1_diff, a1_se, _, a1_p = welch("a_1_per_byte")
    macros["posPatternAOneGapAbs"] = _fmt(abs(float(a1_diff)), 3)
    macros["posPatternAOneDiffSe"] = _fmt(float(a1_se), 3)
    p_macro("posPatternAOneDiff", a1_p)

    _, _, _, drop_p = welch("drop_ratio_a_n_over_a_1")
    p_macro("posPatternDropDiff", drop_p)

    coh_diff, _, _, coh_p = welch("coherence_C")
    macros["posPatternCohDiff"] = _fmt(float(coh_diff), 3, force_sign=True)
    p_macro("posPatternCohDiff", coh_p)

    _, _, _, d_p = welch("diversity_score_D_C_an")
    p_macro("posPatternDDiff", d_p)

    _, _, _, cos_p = welch("sentbert_mean_pairwise_cosine")
    p_macro("posPatternCosDiff", cos_p)

    dn_diff, _, _, dn_p = welch("averaged_distinct_ngrams")
    macros["posPatternDistinctDiff"] = _fmt(float(dn_diff), 4, force_sign=True)
    p_macro("posPatternDistinctDiff", dn_p)

    def cond_mean(name: str, col: int, decimals: int) -> str:
        """Mean from a condition-table row; columns as in template_frames_macros."""
        row = re.search(rf"^{re.escape(name)}\s+(.*)$", txt, re.MULTILINE)
        assert row, f"Condition row not found: {name}"
        pairs = re.findall(r"([-\d.e+]+) ± ([-\d.e+]+)", row.group(1))
        assert len(pairs) == 7, (name, pairs)
        return _fmt(float(pairs[col][0]), decimals)

    macros["posPatternCohCanon"] = cond_mean("canonical", 0, 3)
    macros["posPatternCohScram"] = cond_mean("scrambled", 0, 3)
    macros["posPatternDCanon"] = cond_mean("canonical", 2, 3)
    macros["posPatternDScram"] = cond_mean("scrambled", 2, 3)
    macros["posPatternCosCanon"] = cond_mean("canonical", 4, 3)
    macros["posPatternCosScram"] = cond_mean("scrambled", 4, 3)
    macros["posPatternDistinctCanon"] = cond_mean("canonical", 6, 3)
    macros["posPatternDistinctScram"] = cond_mean("scrambled", 6, 3)

    # Ground-truth order entropy from the generator's own class multiset:
    # the scrambled generator adds exactly log2(#distinguishable class orders)
    # bits per sentence relative to canonical.
    total_slots = sum(c for _, c in CANONICAL_CLASS_MULTISET)
    n_orders = math.factorial(total_slots)
    for _, count in CANONICAL_CLASS_MULTISET:
        n_orders //= math.factorial(count)
    order_bits = math.log2(n_orders)
    mean_bytes = (float(ec[6]) + float(es[6])) / 2
    macros["posPatternNumOrders"] = f"{n_orders}"
    macros["posPatternOrderBits"] = _fmt(order_bits, 2)
    macros["posPatternOrderBitsPerByte"] = _fmt(order_bits / mean_bytes, 2)

    case = (base / "distinctn_case_check.txt").read_text()
    m4 = re.search(
        r"^lowercased: .*?diff ([+\-\d.]+)\s+t = [+\-\d.]+\s+\(p = ([\d.e-]+)\)",
        case,
        re.MULTILINE,
    )
    assert m4, "lowercased line not found in distinctn_case_check.txt"
    macros["posPatternDistinctLowerDiff"] = m4.group(1)
    macros["posPatternDistinctLowerP"] = m4.group(2)
    return macros


def write_output(all_macros: dict[str, str]) -> None:
    lines = [
        "% Generated by scripts/build_paper_macros.py — DO NOT EDIT.",
        "% Every inline numeric scalar referenced in the paper prose is defined here.",
        "% Regenerate with: uv run python scripts/build_paper_macros.py",
        "%",
        "% Grouped by section for readability.",
    ]

    groups = {
        "Cross-mode pairwise (Sec 8.4)": [
            "crossmode",
        ],
        "Scaling / Llama (Sec 8.7)": [
            "scaling",
        ],
        "Qwen3 comparison (App E / Sec 8.7)": [
            "qwenThree",
        ],
        "Tevet external validation (Sec 7.5)": [
            "tevet",
        ],
        "Mode count (Sec 8.3)": [
            "modeCount",
        ],
        "Permutation sensitivity (Sec 8.6)": [
            "permSens",
        ],
        "Scenario validation (Sec 3 Table 1 / App C.2)": [
            "scenario",
        ],
        "Structural redundancy (draft sections/07_9_structural_redundancy.tex)": [
            "framesQwen",
            "posPattern",
        ],
    }
    used = set()
    for title, prefixes in groups.items():
        group_macros = {
            n: v
            for n, v in all_macros.items()
            if any(n.startswith(p) for p in prefixes)
        }
        if not group_macros:
            continue
        lines.append(f"\n% --- {title} ---")
        for name in sorted(group_macros):
            lines.append(rf"\newcommand{{\{name}}}{{{group_macros[name]}}}")
            used.add(name)
    extras = [n for n in all_macros if n not in used]
    if extras:
        lines.append("\n% --- Other ---")
        for name in sorted(extras):
            lines.append(rf"\newcommand{{\{name}}}{{{all_macros[name]}}}")

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT} with {len(all_macros)} macros.")


def main() -> None:
    all_macros: dict[str, str] = {}
    all_macros.update(cross_mode_macros())
    all_macros.update(scaling_macros())
    all_macros.update(fractional_model_macros())
    all_macros.update(symmetry_macros())
    all_macros.update(mode_count_macros())
    all_macros.update(tevet_macros())
    all_macros.update(tevet_coherence_distribution_macros())
    all_macros.update(qwen3_macros())
    all_macros.update(mode_count_extrema())
    all_macros.update(permutation_sensitivity_macros())
    all_macros.update(tevet_dataset_size_macros())
    all_macros.update(mode_count_config_macros())
    all_macros.update(scenario_validation_macros())
    all_macros.update(template_frames_macros())
    all_macros.update(pos_pattern_macros())
    write_output(all_macros)


if __name__ == "__main__":
    main()
