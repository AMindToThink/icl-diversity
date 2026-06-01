"""Analyze the OLMo in-family self-scoring matrix.

Re-scores were produced by 3_score_icl_diversity.py with --scorer-model set to
each OLMo pipeline stage; this script assembles the 5 (grader theta) x 4 (gen
stage) matrices of D = C * a_n, C, and a_n per prompt set, and runs the frozen
pre-registered analyses:

  H1 (headline): in EVERY theta row, D decreases base > sft > dpo. Tested via the
      adjacent steps base>sft and sft>dpo (one-sided paired Wilcoxon over prompts,
      Holm-corrected within each prompt set). dpo->instruct carries NO prediction
      and is reported descriptively only.
  H2 (support): a_n decreases base > sft > dpo in every row.
  H8 (control): C rises (or is flat) across gen stages -> C alone is not the
      diversity signal.
  H6 (weak, descriptive): fold-change R_theta = mean_D(base)/mean_D(dpo) per row,
      with a bootstrap-over-prompts CI; expected to shrink for in-family graders
      vs Qwen but stay > 1.

Pairing is by prompt_id within a (theta, prompt_set); D is aggregated as the mean
of the per-prompt product C_p * a_n,p (never mean(C)*mean(a_n)).

Inputs:
  results/rlhf_experiment/matrix/icl_metrics_lm_theta-{base,sft,dpo,instruct}.jsonl
  results/rlhf_experiment/icl_metrics_length_matched.jsonl   (committed Qwen row)

Outputs:
  results/rlhf_experiment/matrix/matrix_summary.json
  results/rlhf_experiment/matrix/matrix_summary.txt
  figures/rlhf_experiment/matrix/ak_overlay_theta-{...}_{set}.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
MATRIX_DIR = RESULTS_DIR / "matrix"
FIG_DIR = REPO_ROOT / "figures" / "rlhf_experiment" / "matrix"

# Display order along the pipeline (= predicted diversity order, descending D).
GEN_ORDER = ["base", "sft", "dpo", "instruct"]
# The confirmatory adjacent steps (dpo->instruct carries no prediction).
CONFIRMATORY_STEPS = [("base", "sft"), ("sft", "dpo")]

# theta rows: label -> (scorer_model id, path). Qwen row reuses the committed file.
THETA_ROWS = [
    ("Qwen-3B", "Qwen/Qwen2.5-3B", RESULTS_DIR / "icl_metrics_length_matched.jsonl"),
    ("OLMo-base", "allenai/OLMo-2-1124-7B", MATRIX_DIR / "icl_metrics_lm_theta-base.jsonl"),
    ("OLMo-sft", "allenai/OLMo-2-1124-7B-SFT", MATRIX_DIR / "icl_metrics_lm_theta-sft.jsonl"),
    ("OLMo-dpo", "allenai/OLMo-2-1124-7B-DPO", MATRIX_DIR / "icl_metrics_lm_theta-dpo.jsonl"),
    ("OLMo-instruct", "allenai/OLMo-2-1124-7B-Instruct", MATRIX_DIR / "icl_metrics_lm_theta-instruct.jsonl"),
]

PROMPT_SETS = ["alpacaeval", "nbcurated"]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.open() if ln.strip()]


def index_by_cell(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """(stage, prompt_set) -> {prompt_id: row}. Filters to D_C_an present."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("diversity_score_D_C_an") is None:
            continue
        out[(r["stage"], r["prompt_set"])][r["prompt_id"]] = r
    return out


def paired(metric_a: dict[str, dict], metric_b: dict[str, dict], key: str):
    """Return (vec_a, vec_b) aligned over the shared prompt_ids."""
    shared = sorted(set(metric_a) & set(metric_b))
    a = np.array([metric_a[p][key] for p in shared], dtype=float)
    b = np.array([metric_b[p][key] for p in shared], dtype=float)
    return a, b, shared


def cohen_dz(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int):
    """CI for mean(num)/mean(den), paired bootstrap over prompts (rows aligned)."""
    rng = np.random.default_rng(seed)
    n = len(num)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = den[idx].mean()
        if d != 0:
            ratios.append(num[idx].mean() / d)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(n_boot: int, seed: int) -> dict:
    # Load every theta row's per-cell index.
    theta_cells = {}
    missing = []
    for label, model_id, path in THETA_ROWS:
        rows = load_rows(path)
        if not rows:
            missing.append((label, str(path)))
            continue
        theta_cells[label] = index_by_cell(rows)

    result: dict = {"missing_theta": missing, "prompt_sets": {}}

    for pset in PROMPT_SETS:
        pset_out: dict = {"matrix": {}, "contrasts": {}, "R_theta": {}}
        # Build D/C/a_n matrices: row=theta label, col=gen stage -> mean over prompts.
        for label in theta_cells:
            cells = theta_cells[label]
            row = {}
            for gen in GEN_ORDER:
                c = cells.get((gen, pset), {})
                if not c:
                    row[gen] = None
                    continue
                D = np.array([v["diversity_score_D_C_an"] for v in c.values()], float)
                C = np.array([v["coherence_C"] for v in c.values()], float)
                an = np.array([v["a_n_per_byte"] for v in c.values()], float)
                row[gen] = {
                    "n": int(len(D)),
                    "D_mean": float(D.mean()), "D_std": float(D.std(ddof=1)),
                    "C_mean": float(C.mean()),
                    "a_n_mean": float(an.mean()),
                }
            pset_out["matrix"][label] = row

        # Per-row confirmatory contrasts on D: base>sft, sft>dpo (one-sided Wilcoxon).
        # Holm-correct within the prompt set across (rows x 2 steps).
        raw_contrasts = []  # (label, step, stat, p, dz, mean_a, mean_b, n)
        for label in theta_cells:
            cells = theta_cells[label]
            for hi_stage, lo_stage in CONFIRMATORY_STEPS:
                a, b, shared = paired(
                    cells.get((hi_stage, pset), {}),
                    cells.get((lo_stage, pset), {}),
                    "diversity_score_D_C_an",
                )
                if len(a) < 5:
                    continue
                diff = a - b
                # one-sided: H1 that hi_stage D > lo_stage D
                try:
                    stat, p = stats.wilcoxon(a, b, alternative="greater")
                except ValueError:
                    stat, p = float("nan"), float("nan")
                raw_contrasts.append({
                    "theta": label, "step": f"{hi_stage}>{lo_stage}",
                    "stat": float(stat), "p_raw": float(p),
                    "dz": cohen_dz(diff),
                    "mean_hi": float(a.mean()), "mean_lo": float(b.mean()),
                    "delta": float(diff.mean()), "n": int(len(a)),
                })
        adj = holm([c["p_raw"] for c in raw_contrasts]) if raw_contrasts else []
        for c, pa in zip(raw_contrasts, adj):
            c["p_holm"] = pa
        pset_out["contrasts"] = raw_contrasts

        # dpo->instruct: descriptive (no test in the confirmatory family).
        desc = []
        for label in theta_cells:
            cells = theta_cells[label]
            a, b, shared = paired(
                cells.get(("dpo", pset), {}), cells.get(("instruct", pset), {}),
                "diversity_score_D_C_an",
            )
            if len(a) >= 5:
                desc.append({
                    "theta": label, "step": "dpo vs instruct (no prediction)",
                    "mean_dpo": float(a.mean()), "mean_instruct": float(b.mean()),
                    "delta": float((a - b).mean()),
                })
        pset_out["dpo_instruct_descriptive"] = desc

        # R_theta = mean_D(base)/mean_D(dpo) per row, bootstrap CI.
        for label in theta_cells:
            cells = theta_cells[label]
            base = cells.get(("base", pset), {})
            dpo = cells.get(("dpo", pset), {})
            shared = sorted(set(base) & set(dpo))
            if len(shared) < 5:
                continue
            num = np.array([base[p]["diversity_score_D_C_an"] for p in shared], float)
            den = np.array([dpo[p]["diversity_score_D_C_an"] for p in shared], float)
            r_point = num.mean() / den.mean() if den.mean() != 0 else float("nan")
            lo, hi = bootstrap_ratio_ci(num, den, n_boot, seed)
            pset_out["R_theta"][label] = {
                "R": float(r_point), "ci95": [lo, hi], "n": len(shared),
            }

        result["prompt_sets"][pset] = pset_out

    return result


def fmt_table(result: dict) -> str:
    lines = []
    for pset, out in result["prompt_sets"].items():
        lines.append(f"\n{'=' * 78}\nPROMPT SET: {pset}\n{'=' * 78}")
        # D matrix
        lines.append("\nD = C * a_n  (mean over prompts; rows=grader theta, cols=gen stage)")
        header = f"{'theta \\ gen':16}" + "".join(f"{g:>11}" for g in GEN_ORDER)
        lines.append(header)
        for label, row in out["matrix"].items():
            cells = "".join(
                (f"{row[g]['D_mean']:>11.4f}" if row.get(g) else f"{'--':>11}")
                for g in GEN_ORDER
            )
            # monotone verdict on the predicted chain base>=sft>=dpo
            vals = [row[g]["D_mean"] for g in ["base", "sft", "dpo"] if row.get(g)]
            mono = "OK" if all(x > y for x, y in zip(vals, vals[1:])) else "FAIL"
            lines.append(f"{label:16}{cells}   base>sft>dpo: {mono}")
        # C matrix
        lines.append("\nC (coherence; same layout)")
        lines.append(header)
        for label, row in out["matrix"].items():
            cells = "".join(
                (f"{row[g]['C_mean']:>11.4f}" if row.get(g) else f"{'--':>11}")
                for g in GEN_ORDER
            )
            lines.append(f"{label:16}{cells}")
        # a_n matrix
        lines.append("\na_n per byte (same layout)")
        lines.append(header)
        for label, row in out["matrix"].items():
            cells = "".join(
                (f"{row[g]['a_n_mean']:>11.4f}" if row.get(g) else f"{'--':>11}")
                for g in GEN_ORDER
            )
            lines.append(f"{label:16}{cells}")
        # Contrasts
        lines.append("\nH1 confirmatory contrasts on D (one-sided paired Wilcoxon, Holm-corrected):")
        lines.append(f"{'theta':16}{'step':12}{'delta':>9}{'d_z':>8}{'p_raw':>11}{'p_holm':>11}{'n':>5}")
        for c in out["contrasts"]:
            sig = "*" if c.get("p_holm", 1) < 0.05 else " "
            lines.append(
                f"{c['theta']:16}{c['step']:12}{c['delta']:>9.4f}{c['dz']:>8.2f}"
                f"{c['p_raw']:>11.2e}{c.get('p_holm', float('nan')):>11.2e}{c['n']:>5} {sig}"
            )
        # dpo vs instruct descriptive
        lines.append("\ndpo vs instruct (NO prediction; descriptive):")
        for d in out["dpo_instruct_descriptive"]:
            lines.append(f"  {d['theta']:16} dpo={d['mean_dpo']:.4f} instruct={d['mean_instruct']:.4f} delta={d['delta']:+.4f}")
        # R_theta
        lines.append("\nR_theta = mean_D(base)/mean_D(dpo)  (fold-change; bootstrap 95% CI):")
        for label, r in out["R_theta"].items():
            lines.append(f"  {label:16} R={r['R']:.3f}  CI95=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  n={r['n']}")
    if result["missing_theta"]:
        lines.append("\n[WARNING] missing theta rows (not yet scored):")
        for label, path in result["missing_theta"]:
            lines.append(f"  {label}: {path}")
    return "\n".join(lines)


def plot_overlays(result: dict) -> list[str]:
    """Per (theta, prompt_set) a_k curve overlay across gen stages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {"base": "C0", "sft": "C1", "dpo": "C2", "instruct": "C3"}
    for label, model_id, path in THETA_ROWS:
        rows = load_rows(path)
        if not rows:
            continue
        cells = index_by_cell(rows)
        for pset in PROMPT_SETS:
            fig, ax = plt.subplots(figsize=(6, 4))
            any_curve = False
            for gen in GEN_ORDER:
                c = cells.get((gen, pset), {})
                if not c:
                    continue
                curves = [v["a_k_curve_per_byte"] for v in c.values()
                          if v.get("a_k_curve_per_byte")]
                if not curves:
                    continue
                arr = np.array(curves)  # (n_prompts, k)
                mean_curve = arr.mean(axis=0)
                ks = np.arange(1, len(mean_curve) + 1)
                ax.plot(ks, mean_curve, marker="o", ms=3, color=colors[gen], label=gen)
                any_curve = True
            if not any_curve:
                plt.close(fig)
                continue
            ax.set_xlabel("k (responses conditioned on)")
            ax.set_ylabel("a_k per byte (bits/byte)")
            ax.set_title(f"theta={label}  {pset}")
            ax.legend(title="gen stage")
            fig.tight_layout()
            out = FIG_DIR / f"ak_overlay_theta-{label}_{pset}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(str(out))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    result = analyze(args.n_boot, args.seed)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    (MATRIX_DIR / "matrix_summary.json").write_text(json.dumps(result, indent=2))
    table = fmt_table(result)
    (MATRIX_DIR / "matrix_summary.txt").write_text(table)
    print(table)

    if not args.no_plots:
        figs = plot_overlays(result)
        print(f"\n[plots] wrote {len(figs)} overlays to {FIG_DIR}")


if __name__ == "__main__":
    main()
