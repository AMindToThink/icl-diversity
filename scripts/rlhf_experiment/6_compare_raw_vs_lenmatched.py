"""Side-by-side comparison: raw vs length-matched RLHF-diversity analysis.

Reads:
    results/rlhf_experiment/analysis.json                # raw
    results/rlhf_experiment/analysis_length_matched.json # length-matched

Writes:
    investigations/length_matched_rlhf/04_comparison.txt
    investigations/length_matched_rlhf/04_comparison.json

Compares per-stage D = C * a_n means and the three pre-registered H1 contrasts
(H1a base>SFT, H1b SFT>DPO, H1c base>RLVR) for both prompt sets.

NOTE: the raw analysis is over 200 alpacaeval / 100 nbcurated prompts; the
length-matched analysis is over the 150 / 39 prompts that survived the 50-byte
truncation floor (see length_match_report.json). The contrast is therefore on
overlapping but non-identical prompt sets — the comparison still answers the
question "does the monotone D drop survive when responses are length-matched
on the prompts where length-matching is meaningful?", but readers should know
the n changes.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
INV_DIR = REPO_ROOT / "investigations" / "length_matched_rlhf"

STAGES = ["base", "sft", "dpo", "instruct"]
LABELS = {"base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
CONTRAST_NAMES = [
    ("HoneA", "H1a (Base > SFT)"),
    ("HoneB", "H1b (SFT > DPO)"),
    ("HoneC", "H1c (Base > Instruct)"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def safe(x):
    if x is None:
        return float("nan")
    if isinstance(x, float):
        return x
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt_num(x: float) -> str:
    if x != x:  # NaN
        return "  ---"
    return f"{x:7.3f}"


def fmt_p(x: float) -> str:
    if x != x:
        return "  ---"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def main() -> None:
    raw = load(RESULTS_DIR / "analysis.json")
    lm = load(RESULTS_DIR / "analysis_length_matched.json")

    out_lines: list[str] = []
    out_lines.append("Raw vs length-matched RLHF-diversity comparison")
    out_lines.append("=" * 80)
    out_lines.append("")
    out_lines.append("D = C * a_n (per-byte; paper's primary diversity scalar, sec 6.3)")
    out_lines.append("")

    summary: dict = {}

    for pset in ["alpacaeval", "nbcurated"]:
        out_lines.append(f"--- prompt_set = {pset} ---")
        ms_raw = raw[pset]["stage_means"]["D_Can"]
        ms_lm = lm[pset]["stage_means"]["D_Can"]

        n_raw = ms_raw["base"]["n"]
        n_lm = ms_lm["base"]["n"]
        out_lines.append(f"  n: raw={n_raw}, length-matched={n_lm}")
        out_lines.append("")
        out_lines.append(f"  {'stage':<10} {'raw mean':>10} {'raw std':>10} "
                         f"{'lm mean':>10} {'lm std':>10}  {'mean_diff':>10}")
        for s in STAGES:
            mr = safe(ms_raw[s]["mean"])
            sr = safe(ms_raw[s]["std"])
            ml = safe(ms_lm[s]["mean"])
            sl = safe(ms_lm[s]["std"])
            out_lines.append(
                f"  {LABELS[s]:<10}{fmt_num(mr):>11}{fmt_num(sr):>11}"
                f"{fmt_num(ml):>11}{fmt_num(sl):>11} {fmt_num(ml - mr):>11}"
            )

        # Monotone-decrease check across stages
        means_raw = [safe(ms_raw[s]["mean"]) for s in STAGES]
        means_lm = [safe(ms_lm[s]["mean"]) for s in STAGES]
        diffs_raw = [means_raw[i + 1] - means_raw[i] for i in range(3)]
        diffs_lm = [means_lm[i + 1] - means_lm[i] for i in range(3)]
        mono_raw = all(d < 0 for d in diffs_raw)
        mono_lm = all(d < 0 for d in diffs_lm)
        out_lines.append(
            f"  monotone decrease across stages: raw={mono_raw}, length-matched={mono_lm}"
        )

        out_lines.append("")
        out_lines.append("  H1 contrasts (paired Wilcoxon, Bonferroni-corrected):")
        out_lines.append(
            f"    {'name':<22} {'raw d_z':>10} {'raw p_bonf':>12}  "
            f"{'lm d_z':>10} {'lm p_bonf':>12}  {'lm Delta':>10}"
        )
        contrasts_summary = []
        for key, label in CONTRAST_NAMES:
            tr = raw[pset]["tests"]["D_Can"][key]
            tl = lm[pset]["tests"]["D_Can"][key]
            out_lines.append(
                f"    {label:<22} {fmt_num(safe(tr['cohen_dz'])):>10}"
                f" {fmt_p(safe(tr['p_bonferroni'])):>12}  "
                f"{fmt_num(safe(tl['cohen_dz'])):>10} "
                f"{fmt_p(safe(tl['p_bonferroni'])):>12}"
                f" {fmt_num(safe(tl['mean_diff'])):>10}"
            )
            contrasts_summary.append({
                "name": key,
                "label": label,
                "raw": {
                    "cohen_dz": safe(tr["cohen_dz"]),
                    "p_bonferroni": safe(tr["p_bonferroni"]),
                    "mean_diff": safe(tr["mean_diff"]),
                    "n": int(tr.get("n", 0)),
                },
                "length_matched": {
                    "cohen_dz": safe(tl["cohen_dz"]),
                    "p_bonferroni": safe(tl["p_bonferroni"]),
                    "mean_diff": safe(tl["mean_diff"]),
                    "n": int(tl.get("n", 0)),
                },
            })

        # H1' two-sided
        tr = raw[pset]["tests"]["D_Can"]["Hpa"]
        tl = lm[pset]["tests"]["D_Can"]["Hpa"]
        out_lines.append("")
        out_lines.append("  H1' (DPO vs Instruct, two-sided uncorrected):")
        out_lines.append(
            f"    raw: d_z={fmt_num(safe(tr['cohen_dz']))} p={fmt_p(safe(tr['p_raw']))} "
            f"Delta={fmt_num(safe(tr['mean_diff']))}"
        )
        out_lines.append(
            f"    lm : d_z={fmt_num(safe(tl['cohen_dz']))} p={fmt_p(safe(tl['p_raw']))} "
            f"Delta={fmt_num(safe(tl['mean_diff']))}"
        )

        out_lines.append("")
        out_lines.append("")

        # Magnitude of per-stage shift base->instruct, then ratio
        raw_drop = means_raw[0] - means_raw[3]
        lm_drop = means_lm[0] - means_lm[3]
        attenuation = (1 - lm_drop / raw_drop) if raw_drop != 0 else float("nan")
        out_lines.append(f"  Base->Instruct mean(D) drop: raw={raw_drop:+.3f}, "
                         f"length-matched={lm_drop:+.3f} "
                         f"(attenuation under length-match: {attenuation:.1%})")
        out_lines.append("")
        summary[pset] = {
            "n_raw": n_raw,
            "n_length_matched": n_lm,
            "stage_means_raw": {s: safe(ms_raw[s]["mean"]) for s in STAGES},
            "stage_means_length_matched": {s: safe(ms_lm[s]["mean"]) for s in STAGES},
            "monotone_decrease_raw": mono_raw,
            "monotone_decrease_length_matched": mono_lm,
            "base_minus_instruct_raw": raw_drop,
            "base_minus_instruct_length_matched": lm_drop,
            "attenuation_pct": attenuation,
            "contrasts": contrasts_summary,
        }

    INV_DIR.mkdir(parents=True, exist_ok=True)
    (INV_DIR / "04_comparison.txt").write_text("\n".join(out_lines) + "\n")
    (INV_DIR / "04_comparison.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(out_lines))
    print(f"\n[compare] wrote {INV_DIR / '04_comparison.txt'} and "
          f"{INV_DIR / '04_comparison.json'}")


if __name__ == "__main__":
    main()
