"""Compare D = C × a_n variants across all five paper experiments.

Three D variants per sample, plus the underlying a_n variants:

* ``D_hybrid``  = C × a_n_total       — C in bits/byte (unitless), a_n in total bits.
* ``D_pb_RoM``  = C × a_n_pb_RoM       — current code path; a_n_pb is computed
  as ratio-of-means: ``mean_perms(a_k_total[-1]) / mean_perms(byte_count[-1])``.
* ``D_pb_MoR``  = C × a_n_pb_MoR       — correct mean-of-ratios per-byte (when
  per-permutation data is logged): ``mean_perms(a_k_total[-1] / byte_count[-1])``.

This script reads existing logs only (no model calls, no GPU), so it is
reproducible offline. Outputs land under ``figures/d_variant_comparison/``.

Usage
-----
::

    uv run python scripts/compare_d_variants.py

What gets compared per experiment
---------------------------------
* **Tevet** (primary, human-grounded): Spearman ρ between each variant and
  ``label_value`` per (dataset, task). MoR is not reproducible from the
  current ``*.icl_mean_curves.*.json`` sidecars (no per-perm preserved).
* **OLMo RLHF** (primary, AI-side): per-stage means (base / sft / dpo /
  instruct) for each variant on AlpacaEval and NoveltyBench-curated, plus
  the length-matched re-run.  All three variants computable.
* **Synthetic scenarios**: per-scenario means under each variant for
  GPT-2 and Qwen2.5-3B (v3, 100 permutations).  All three variants.
* **Synthetic mode count**: Spearman ρ between each variant and m
  (number of modes), per draw.  Per-draw n_permutations=1, so RoM≡MoR.
* **Model scaling** (cross-mode pairwise reduction): diagonal / off-
  diagonal mean of (uncond − cond) in total bits and in bits/byte across
  six models. Different shape from C × a_n — this isn't a direct D
  variant comparison; it's the same total-vs-per-byte question on the
  pairwise reduction matrix.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
INVESTIGATIONS = PROJECT_ROOT / "investigations"
OUT_DIR = PROJECT_ROOT / "figures" / "d_variant_comparison"


# ---------------------------------------------------------------------------
# Core: per-record variant computation
# ---------------------------------------------------------------------------


@dataclass
class Variants:
    """Four scalars per record.  ``MoR`` fields are None when per-permutation
    data is unavailable in the source log."""

    C: float
    a_n_total: float
    a_n_pb_RoM: float
    a_n_pb_MoR: float | None
    D_hybrid: float
    D_pb_RoM: float
    D_pb_MoR: float | None


def _coherence_from_record(rec: dict[str, Any]) -> float:
    """C = 2^(-mean(per-byte unconditional surprise))."""
    if rec.get("coherence_C") is not None:
        return float(rec["coherence_C"])
    if rec.get("unconditional_surprises"):
        return 2.0 ** (-mean(rec["unconditional_surprises"]))
    if rec.get("unconditional_total_bits") and rec.get("a_k_byte_counts"):
        per_byte = [
            tb / bc
            for tb, bc in zip(rec["unconditional_total_bits"], rec["a_k_byte_counts"])
            if bc > 0
        ]
        if not per_byte:
            raise ValueError("All byte counts are zero; cannot derive C")
        return 2.0 ** (-mean(per_byte))
    raise KeyError(
        "record needs one of: coherence_C, unconditional_surprises, "
        "or (unconditional_total_bits + a_k_byte_counts)"
    )


def compute_variants(rec: dict[str, Any]) -> Variants:
    """Compute the three D variants from a record.

    Args:
        rec: dict-like with at least ``a_k_curve`` (total bits, length n)
            and ``a_k_byte_counts`` (length n), and a way to derive C
            (see :func:`_coherence_from_record`).  If ``per_permutation_a_k_curves``
            and ``per_permutation_byte_counts`` are present and non-None,
            the MoR variants are also computed.

    Returns:
        :class:`Variants`.
    """
    a_k = rec["a_k_curve"]
    bc = rec["a_k_byte_counts"]
    if not a_k or not bc:
        raise ValueError("empty a_k_curve or a_k_byte_counts")
    if bc[-1] <= 0:
        raise ValueError(f"final byte count is non-positive: {bc[-1]}")

    C = _coherence_from_record(rec)
    a_n_total = float(a_k[-1])
    a_n_pb_RoM = a_n_total / bc[-1]

    pp_curves = rec.get("per_permutation_a_k_curves")
    pp_bytes = rec.get("per_permutation_byte_counts")
    a_n_pb_MoR: float | None = None
    if pp_curves and pp_bytes and len(pp_curves) > 0:
        ratios: list[float] = []
        for cur, b in zip(pp_curves, pp_bytes):
            if b[-1] > 0:
                ratios.append(cur[-1] / b[-1])
        if ratios:
            a_n_pb_MoR = sum(ratios) / len(ratios)

    return Variants(
        C=C,
        a_n_total=a_n_total,
        a_n_pb_RoM=a_n_pb_RoM,
        a_n_pb_MoR=a_n_pb_MoR,
        D_hybrid=C * a_n_total,
        D_pb_RoM=C * a_n_pb_RoM,
        D_pb_MoR=(C * a_n_pb_MoR) if a_n_pb_MoR is not None else None,
    )


# ---------------------------------------------------------------------------
# Tevet: ρ vs human labels
# ---------------------------------------------------------------------------

TEVET_TASKS = ("prompt_gen", "resp_gen", "story_gen")


def _safe_rho(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
    if len(pairs) < 5:
        return float("nan")
    a, b = zip(*pairs)
    rho, _ = spearmanr(a, b)
    return float(rho)


def analyze_tevet(run_tag: str = "qwen25_completion_v3") -> list[dict[str, Any]]:
    """For each (dataset, task) pair, compute Spearman ρ for D_hybrid and D_pb_RoM
    against ``label_value``.  MoR is not derivable from mean_curves sidecars
    (per-perm data not preserved); column is reported as NaN."""
    rows: list[dict[str, Any]] = []
    base = RESULTS / "tevet" / run_tag
    if not base.exists():
        print(f"[tevet] missing: {base}")
        return rows

    sidecar_paths = sorted(base.rglob(f"*.icl_mean_curves.{run_tag}.json"))
    if not sidecar_paths:
        print(f"[tevet] no mean_curves sidecars under {base}")
        return rows

    for sc_path in sidecar_paths:
        csv_path = sc_path.with_name(
            sc_path.name.replace(f".icl_mean_curves.{run_tag}.json", ".csv")
        )
        if not csv_path.exists():
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        key_col = (
            "sidecar_key" if csv_rows and "sidecar_key" in csv_rows[0] else "sample_id"
        )
        rows_by_key = {r[key_col]: r for r in csv_rows}
        sidecar = json.load(open(sc_path))

        d_hybrid_scores: list[float] = []
        d_pb_RoM_scores: list[float] = []
        a_n_bits_scores: list[float] = []
        a_n_pb_scores: list[float] = []
        labels: list[float] = []

        for sid, entry in sidecar.items():
            if sid.startswith("__"):
                continue
            row = rows_by_key.get(sid)
            if row is None:
                continue
            try:
                lab = float(row.get("label_value", "nan"))
            except ValueError:
                continue
            if np.isnan(lab):
                continue
            try:
                v = compute_variants(entry)
            except (KeyError, ValueError):
                continue
            d_hybrid_scores.append(v.D_hybrid)
            d_pb_RoM_scores.append(v.D_pb_RoM)
            a_n_bits_scores.append(v.a_n_total)
            a_n_pb_scores.append(v.a_n_pb_RoM)
            labels.append(lab)

        if len(labels) < 5:
            continue

        dataset = sc_path.parent.name
        stem = sc_path.name.replace(f".icl_mean_curves.{run_tag}.json", "")
        # extract task suffix
        for t in TEVET_TASKS:
            if stem.endswith("_" + t):
                task = t
                subset = stem[: -len("_" + t)]
                break
        else:
            task = ""
            subset = stem

        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "task": task,
                "n": len(labels),
                "rho_D_hybrid": _safe_rho(d_hybrid_scores, labels),
                "rho_D_pb_RoM": _safe_rho(d_pb_RoM_scores, labels),
                "rho_D_pb_MoR": float("nan"),  # not reproducible from mean_curves
                "rho_a_n_bits": _safe_rho(a_n_bits_scores, labels),
                "rho_a_n_pb_RoM": _safe_rho(a_n_pb_scores, labels),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# OLMo RLHF: per-stage means + ordering
# ---------------------------------------------------------------------------

RLHF_STAGES = ("base", "sft", "dpo", "instruct")
RLHF_PROMPT_SETS = ("alpacaeval", "nbcurated")


def _stage_means(
    records: list[dict[str, Any]], stage_field: str = "stage"
) -> dict[str, dict[str, dict[str, float]]]:
    """Group by ``prompt_set`` then ``stage_field``, compute mean of each variant."""
    by_set: dict[str, dict[str, list[Variants]]] = {}
    for r in records:
        pset = r.get("prompt_set", "")
        st = r.get(stage_field, "")
        try:
            v = compute_variants(r)
        except (KeyError, ValueError):
            continue
        by_set.setdefault(pset, {}).setdefault(st, []).append(v)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for pset, by_stage in by_set.items():
        out[pset] = {}
        for st, vlist in by_stage.items():
            d_hybrid = [v.D_hybrid for v in vlist]
            d_pb_RoM = [v.D_pb_RoM for v in vlist]
            d_pb_MoR = [v.D_pb_MoR for v in vlist if v.D_pb_MoR is not None]
            a_n_total = [v.a_n_total for v in vlist]
            a_n_pb_RoM = [v.a_n_pb_RoM for v in vlist]
            a_n_pb_MoR = [v.a_n_pb_MoR for v in vlist if v.a_n_pb_MoR is not None]
            out[pset][st] = {
                "n": float(len(vlist)),
                "D_hybrid_mean": float(np.mean(d_hybrid)) if d_hybrid else float("nan"),
                "D_hybrid_std": float(np.std(d_hybrid, ddof=1))
                if len(d_hybrid) > 1
                else float("nan"),
                "D_pb_RoM_mean": float(np.mean(d_pb_RoM)) if d_pb_RoM else float("nan"),
                "D_pb_RoM_std": float(np.std(d_pb_RoM, ddof=1))
                if len(d_pb_RoM) > 1
                else float("nan"),
                "D_pb_MoR_mean": float(np.mean(d_pb_MoR)) if d_pb_MoR else float("nan"),
                "D_pb_MoR_std": float(np.std(d_pb_MoR, ddof=1))
                if len(d_pb_MoR) > 1
                else float("nan"),
                "a_n_total_mean": float(np.mean(a_n_total))
                if a_n_total
                else float("nan"),
                "a_n_pb_RoM_mean": float(np.mean(a_n_pb_RoM))
                if a_n_pb_RoM
                else float("nan"),
                "a_n_pb_MoR_mean": float(np.mean(a_n_pb_MoR))
                if a_n_pb_MoR
                else float("nan"),
            }
    return out


def _stage_ordering_pass(
    stage_means: dict[str, dict[str, float]], variant: str
) -> dict[str, Any]:
    """Check whether base > sft > dpo > instruct holds for ``variant``."""
    means = {
        st: stage_means[st][f"{variant}_mean"]
        for st in RLHF_STAGES
        if st in stage_means
    }
    pairs: list[tuple[str, str, bool]] = []
    for i, a in enumerate(RLHF_STAGES):
        for b in RLHF_STAGES[i + 1 :]:
            if a in means and b in means:
                pairs.append((a, b, means[a] > means[b]))
    n_correct = sum(1 for _, _, ok in pairs if ok)
    return {
        "n_pairs": len(pairs),
        "n_correct": n_correct,
        "details": pairs,
    }


def analyze_rlhf() -> dict[str, Any]:
    """Read RLHF metric jsonls and compute per-stage means + ordering check."""
    out: dict[str, Any] = {}
    for tag, fname in (
        ("raw", "icl_metrics.jsonl"),
        ("length_matched", "icl_metrics_length_matched.jsonl"),
    ):
        path = RESULTS / "rlhf_experiment" / fname
        if not path.exists():
            print(f"[rlhf] missing: {path}")
            continue
        records = [json.loads(line) for line in open(path)]
        means = _stage_means(records)
        out[tag] = {
            "stage_means": means,
            "orderings": {
                pset: {
                    var: _stage_ordering_pass(means[pset], var)
                    for var in ("D_hybrid", "D_pb_RoM", "D_pb_MoR")
                }
                for pset in means
            },
        }
    return out


# ---------------------------------------------------------------------------
# Synthetic scenarios: per-scenario means
# ---------------------------------------------------------------------------

SCENARIO_FILES = {
    "qwen2.5-3b": "scenario_metrics_v3_qwen3b_100perm.json",
    "gpt2": "scenario_metrics_v3_gpt2_100perm.json",
}


def analyze_scenarios() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_label, fname in SCENARIO_FILES.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"[scenarios] missing: {path}")
            continue
        data = json.load(open(path))
        scenarios = data.get("scenarios", {})
        per_scenario: dict[str, dict[str, float]] = {}
        for sc_name, prompt_records in scenarios.items():
            vlist: list[Variants] = []
            for rec in prompt_records:
                try:
                    vlist.append(compute_variants(rec))
                except (KeyError, ValueError):
                    continue
            if not vlist:
                continue
            d_hybrid = [v.D_hybrid for v in vlist]
            d_pb_RoM = [v.D_pb_RoM for v in vlist]
            d_pb_MoR = [v.D_pb_MoR for v in vlist if v.D_pb_MoR is not None]
            per_scenario[sc_name] = {
                "n_prompts": float(len(vlist)),
                "D_hybrid_mean": float(np.mean(d_hybrid)),
                "D_pb_RoM_mean": float(np.mean(d_pb_RoM)),
                "D_pb_MoR_mean": float(np.mean(d_pb_MoR)) if d_pb_MoR else float("nan"),
                "a_n_total_mean": float(np.mean([v.a_n_total for v in vlist])),
                "a_n_pb_RoM_mean": float(np.mean([v.a_n_pb_RoM for v in vlist])),
                "C_mean": float(np.mean([v.C for v in vlist])),
            }
        out[model_label] = per_scenario
    return out


# ---------------------------------------------------------------------------
# Mode count: ρ vs m
# ---------------------------------------------------------------------------

MODE_COUNT_FILES = {
    "qwen2.5-3b": "qwen2.5-3b_1k_draws.json",
    "gpt2": "gpt2_1k_draws.json",
}


def analyze_mode_count() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_label, fname in MODE_COUNT_FILES.items():
        path = RESULTS / "mode_count" / fname
        if not path.exists():
            print(f"[mode_count] missing: {path}")
            continue
        data = json.load(open(path))
        runs = data.get("runs", [])
        ms: list[float] = []
        d_hybrid: list[float] = []
        d_pb_RoM: list[float] = []
        d_pb_MoR: list[float] = []  # equals RoM here (n_perm=1) but kept for symmetry
        for r in runs:
            try:
                v = compute_variants(r)
            except (KeyError, ValueError):
                continue
            ms.append(float(r["m"]))
            d_hybrid.append(v.D_hybrid)
            d_pb_RoM.append(v.D_pb_RoM)
            d_pb_MoR.append(v.D_pb_MoR if v.D_pb_MoR is not None else v.D_pb_RoM)
        out[model_label] = {
            "n_draws": len(ms),
            "rho_D_hybrid_vs_m": _safe_rho(d_hybrid, ms),
            "rho_D_pb_RoM_vs_m": _safe_rho(d_pb_RoM, ms),
            "rho_D_pb_MoR_vs_m": _safe_rho(d_pb_MoR, ms),  # ≡ RoM at n_perm=1
        }
    return out


# ---------------------------------------------------------------------------
# Model scaling: cross-mode pairwise reduction in bits vs bits/byte
# ---------------------------------------------------------------------------


def analyze_scaling() -> dict[str, Any]:
    """For each model's pairwise_matrix.json, recompute diag/off-diag mean of
    reduction (uncond − cond) in total bits and in bits/byte.

    Note: this is the cross-mode building block (used by §scaling), NOT
    the C × a_n score directly.  We're answering the same total-vs-per-byte
    question on a different geometric object.
    """
    base = INVESTIGATIONS / "cross_mode_surprise_drop" / "figures"
    if not base.exists():
        print(f"[scaling] missing: {base}")
        return {}
    out: dict[str, Any] = {}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        pm = model_dir / "pairwise_matrix.json"
        if not pm.exists():
            continue
        d = json.load(open(pm))
        if "unconditional_all" not in d:
            continue
        uncond = np.array(d["unconditional_all"], dtype=float)  # (N, M) bits
        ubytes = np.array(d["unconditional_bytes_all"], dtype=float)  # (N, M)
        cond = np.array(d["conditional_all"], dtype=float)  # (N, N, M) bits
        # reduction[i, j, k] = uncond[j, k] − cond[i, j, k]    (i conditions on, j is target)
        red_bits = uncond[np.newaxis, :, :] - cond  # (N, N, M)
        # per-byte reduction: divide by bytes of the target response (j, k)
        red_pb = red_bits / ubytes[np.newaxis, :, :]
        red_bits_mean = red_bits.mean(axis=2)  # (N, N)
        red_pb_mean = red_pb.mean(axis=2)  # (N, N)
        N = red_bits_mean.shape[0]
        diag_mask = np.eye(N, dtype=bool)
        off_mask = ~diag_mask
        out[model_dir.name] = {
            "n_modes": int(N),
            "diag_mean_bits": float(red_bits_mean[diag_mask].mean()),
            "off_diag_mean_bits": float(red_bits_mean[off_mask].mean()),
            "off_diag_frac_pos_bits": float((red_bits_mean[off_mask] > 0).mean()),
            "diag_mean_pb": float(red_pb_mean[diag_mask].mean()),
            "off_diag_mean_pb": float(red_pb_mean[off_mask].mean()),
            "off_diag_frac_pos_pb": float((red_pb_mean[off_mask] > 0).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _format_tevet_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["## Tevet — Spearman ρ vs human label (higher = better)"]
    lines.append(
        "  Note: MoR not reproducible from mean_curves sidecars (per-perm not preserved)."
    )
    header = (
        f"  {'dataset':<14} {'subset':<32} {'task':<10} {'n':>5} "
        f"{'D_hybrid':>10} {'D_pb_RoM':>10}  {'a_n_bits':>10} {'a_n_pb':>10}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rows:
        lines.append(
            f"  {r['dataset']:<14} {r['subset']:<32} {r['task']:<10} {r['n']:>5} "
            f"{r['rho_D_hybrid']:>+10.3f} {r['rho_D_pb_RoM']:>+10.3f}  "
            f"{r['rho_a_n_bits']:>+10.3f} {r['rho_a_n_pb_RoM']:>+10.3f}"
        )
    return "\n".join(lines)


def _format_rlhf_summary(payload: dict[str, Any]) -> str:
    lines = ["## OLMo RLHF — per-stage means (higher D = more diverse)"]
    for tag, block in payload.items():
        lines.append(f"\n### {tag}")
        for pset, stages in block["stage_means"].items():
            lines.append(f"  prompt_set = {pset}")
            header = (
                f"    {'stage':<10} {'n':>5} "
                f"{'D_hybrid':>14} {'D_pb_RoM':>14} {'D_pb_MoR':>14} "
                f"{'a_n_bits':>10} {'a_n_pb_RoM':>10} {'a_n_pb_MoR':>10}"
            )
            lines.append(header)
            lines.append("    " + "-" * (len(header) - 4))
            for st in RLHF_STAGES:
                if st not in stages:
                    continue
                d = stages[st]
                lines.append(
                    f"    {st:<10} {int(d['n']):>5} "
                    f"{d['D_hybrid_mean']:>14.4f} {d['D_pb_RoM_mean']:>14.4f} {d['D_pb_MoR_mean']:>14.4f} "
                    f"{d['a_n_total_mean']:>10.2f} {d['a_n_pb_RoM_mean']:>10.4f} {d['a_n_pb_MoR_mean']:>10.4f}"
                )
            lines.append("")
            lines.append(
                f"    Pairwise ordering passes (out of {len(RLHF_STAGES) * (len(RLHF_STAGES) - 1) // 2}):"
            )
            ord_block = block["orderings"][pset]
            for var in ("D_hybrid", "D_pb_RoM", "D_pb_MoR"):
                o = ord_block[var]
                lines.append(f"      {var:<10}: {o['n_correct']}/{o['n_pairs']}")
    return "\n".join(lines)


def _format_scenarios_summary(payload: dict[str, Any]) -> str:
    lines = ["## Synthetic scenarios — per-scenario means (5 prompts each)"]
    for model_label, by_sc in payload.items():
        lines.append(f"\n### model = {model_label}")
        header = (
            f"  {'scenario':<22} {'n':>3} "
            f"{'D_hybrid':>14} {'D_pb_RoM':>14} {'D_pb_MoR':>14}  "
            f"{'a_n_bits':>10} {'a_n_pb_RoM':>10}  {'C':>8}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for sc_name, d in by_sc.items():
            lines.append(
                f"  {sc_name:<22} {int(d['n_prompts']):>3} "
                f"{d['D_hybrid_mean']:>14.4f} {d['D_pb_RoM_mean']:>14.4f} {d['D_pb_MoR_mean']:>14.4f}  "
                f"{d['a_n_total_mean']:>10.2f} {d['a_n_pb_RoM_mean']:>10.4f}  {d['C_mean']:>8.4f}"
            )
    return "\n".join(lines)


def _format_mode_count_summary(payload: dict[str, Any]) -> str:
    lines = ["## Mode count — Spearman ρ between each variant and m (higher = better)"]
    lines.append("  Note: per-draw n_permutations=1, so RoM ≡ MoR by construction.")
    header = f"  {'model':<14} {'n_draws':>8} {'ρ(D_hybrid, m)':>16} {'ρ(D_pb_RoM, m)':>18} {'ρ(D_pb_MoR, m)':>18}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for model_label, d in payload.items():
        lines.append(
            f"  {model_label:<14} {d['n_draws']:>8} "
            f"{d['rho_D_hybrid_vs_m']:>+16.3f} {d['rho_D_pb_RoM_vs_m']:>+18.3f} {d['rho_D_pb_MoR_vs_m']:>+18.3f}"
        )
    return "\n".join(lines)


def _format_scaling_summary(payload: dict[str, Any]) -> str:
    lines = [
        "## Model scaling — cross-mode pairwise reduction (NOT C × a_n; same total-vs-per-byte question)"
    ]
    header = (
        f"  {'model':<24} {'N':>4} "
        f"{'diag (bits)':>14} {'off-diag (bits)':>16} {'off+ frac':>10}  "
        f"{'diag (pb)':>10} {'off-diag (pb)':>14} {'off+ frac':>10}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for model_dir, d in payload.items():
        lines.append(
            f"  {model_dir:<24} {d['n_modes']:>4} "
            f"{d['diag_mean_bits']:>+14.2f} {d['off_diag_mean_bits']:>+16.2f} {d['off_diag_frac_pos_bits']:>10.2%}  "
            f"{d['diag_mean_pb']:>+10.4f} {d['off_diag_mean_pb']:>+14.4f} {d['off_diag_frac_pos_pb']:>10.2%}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUT_DIR}")

    print("\n--- Tevet ---")
    tevet = analyze_tevet()
    _write_csv(OUT_DIR / "tevet.csv", tevet)

    print("\n--- OLMo RLHF ---")
    rlhf = analyze_rlhf()
    (OUT_DIR / "rlhf.json").write_text(json.dumps(rlhf, indent=2))

    print("\n--- Synthetic scenarios ---")
    scenarios = analyze_scenarios()
    (OUT_DIR / "scenarios.json").write_text(json.dumps(scenarios, indent=2))

    print("\n--- Mode count ---")
    mc = analyze_mode_count()
    (OUT_DIR / "mode_count.json").write_text(json.dumps(mc, indent=2))

    print("\n--- Model scaling ---")
    scaling = analyze_scaling()
    (OUT_DIR / "scaling.json").write_text(json.dumps(scaling, indent=2))

    summary = "\n\n".join(
        [
            "# D-variant comparison (generated by scripts/compare_d_variants.py)\n",
            _format_tevet_summary(tevet),
            _format_rlhf_summary(rlhf),
            _format_scenarios_summary(scenarios),
            _format_mode_count_summary(mc),
            _format_scaling_summary(scaling),
        ]
    )
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print(f"\nwrote: {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
