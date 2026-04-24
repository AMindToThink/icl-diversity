"""Cross-model comparison table.

Combines OLMo-2-1124-7B four-stage scores with the 7 alon-albalak external
frontier-model scores on the common 100 NB-curated prompts. Emits:

  results/rlhf_experiment/tables/cross_model.tex
  results/rlhf_experiment/tables/cross_model_macros.tex   (scalar macros for
                                                           the accompanying prose)

Primary metric: D = C × a_n (per-byte), computed as
  `coherence_C * a_k_curve_per_byte[-1]`
for each per-prompt record, then averaged across the 100 NB-curated prompts
per model.

Note: external responses were truncated to 100 words before scoring so the
Qwen2.5-3B forward pass fits within 48 GB VRAM AND so the compared-context
lengths roughly match OLMo-2-7B's max_new_tokens=100 sampling ceiling. This
truncation is a real caveat — we report it in the table caption / surrounding
prose rather than hide it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"

ICL_PATH = RESULTS_DIR / "icl_metrics.jsonl"
EXT_PATH = RESULTS_DIR / "external_nb_metrics.jsonl"
OUT_TABLE = RESULTS_DIR / "tables" / "cross_model.tex"
OUT_MACROS = RESULTS_DIR / "tables" / "cross_model_macros.tex"

# Display order and labels for the OLMo-2-7B four stages + the seven external
# models, grouped by family with dividers between groups.
OLMO_STAGES = [
    ("base",     r"OLMo-2-1124-7B (base)"),
    ("sft",      r"OLMo-2-1124-7B-SFT"),
    ("dpo",      r"OLMo-2-1124-7B-DPO"),
    ("instruct", r"OLMo-2-1124-7B-Instruct (RLVR)"),
]

EXTERNAL_MODELS = [
    ("qwen-4b",           r"Qwen-4B-Instruct",     "crossModelQwenFourB"),
    ("qwen-8b",           r"Qwen-8B-Instruct",     "crossModelQwenEightB"),
    ("qwen-235b-a22b",    r"Qwen-235B-A22B",       "crossModelQwenTwoThreeFiveB"),
    ("llama-33-70b",      r"Llama-3.3-70B-Instruct", "crossModelLlamaThreeThreeSeventyB"),
    ("gpt-5-nano",        r"GPT-5 nano",           "crossModelGptFiveNano"),
    ("gpt-5",             r"GPT-5",                "crossModelGptFive"),
    ("claude-sonnet-4-5", r"Claude Sonnet 4.5",    "crossModelClaudeSonnetFourFive"),
]


def _derive_d_can(rec: dict) -> float | None:
    c = rec.get("coherence_C")
    curve = rec.get("a_k_curve_per_byte")
    if c is None or not curve:
        return None
    return float(c) * float(curve[-1])


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def olmo_stats(icl_rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per-stage mean/std of D on the nbcurated prompt set only."""
    per_stage: dict[str, list[float]] = defaultdict(list)
    for r in icl_rows:
        if r.get("prompt_set") != "nbcurated":
            continue
        d = _derive_d_can(r)
        if d is not None:
            per_stage[r["stage"]].append(d)
    out = {}
    for stage, vs in per_stage.items():
        n = len(vs)
        mean = sum(vs) / n if n else float("nan")
        var = sum((v - mean) ** 2 for v in vs) / (n - 1) if n > 1 else float("nan")
        out[stage] = {"mean": mean, "std": math.sqrt(var) if not math.isnan(var) else float("nan"), "n": n}
    return out


def external_stats(ext_rows: list[dict]) -> dict[str, dict[str, float]]:
    per_model: dict[str, list[float]] = defaultdict(list)
    for r in ext_rows:
        d = _derive_d_can(r)
        if d is not None:
            per_model[r["external_model"]].append(d)
    out = {}
    for tag, vs in per_model.items():
        n = len(vs)
        mean = sum(vs) / n if n else float("nan")
        var = sum((v - mean) ** 2 for v in vs) / (n - 1) if n > 1 else float("nan")
        out[tag] = {"mean": mean, "std": math.sqrt(var) if not math.isnan(var) else float("nan"), "n": n}
    return out


def fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    return f"{x:.3f}"


def write_table(olmo: dict, ext: dict, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & $D = C \times a_n$ mean & std & $n$ \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{OLMo-2-1124-7B post-training stages}} \\",
    ]
    for stage, label in OLMO_STAGES:
        s = olmo.get(stage, {})
        lines.append(
            f"\\quad {label} & {fmt(s.get('mean'))} & {fmt(s.get('std'))} & {s.get('n', 0)} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{External instruct / frontier models (released generations, truncated to 100 words)}} \\")
    for tag, label, _macro in EXTERNAL_MODELS:
        s = ext.get(tag, {})
        lines.append(
            f"\\quad {label} & {fmt(s.get('mean'))} & {fmt(s.get('std'))} & {s.get('n', 0)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_macros(olmo: dict, ext: dict, out_path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/rlhf_experiment/5c_cross_model_table.py",
        "",
    ]
    for tag, _label, macro in EXTERNAL_MODELS:
        m = ext.get(tag, {})
        lines.append(f"\\newcommand{{\\{macro}}}{{{fmt(m.get('mean'))}}}")
    # Also emit OLMo stages under a crossModel* prefix so the comparison
    # prose can reference them uniformly.
    for stage, _label in OLMO_STAGES:
        s = olmo.get(stage, {})
        cmd = f"crossModelOlmo{stage.capitalize()}"
        lines.append(f"\\newcommand{{\\{cmd}}}{{{fmt(s.get('mean'))}}}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    icl = read_jsonl(ICL_PATH)
    ext = read_jsonl(EXT_PATH)
    olmo = olmo_stats(icl)
    external = external_stats(ext)
    write_table(olmo, external, OUT_TABLE)
    write_macros(olmo, external, OUT_MACROS)
    print(f"[cross-model] wrote {OUT_TABLE}")
    print(f"[cross-model] wrote {OUT_MACROS}")
    print()
    print("OLMo stages (NB-curated):")
    for stage, _ in OLMO_STAGES:
        s = olmo.get(stage, {})
        print(f"  {stage:10s}  D={fmt(s.get('mean'))}  σ={fmt(s.get('std'))}  n={s.get('n', 0)}")
    print("External models (NB-curated, 100-word truncated):")
    for tag, _, _ in EXTERNAL_MODELS:
        s = external.get(tag, {})
        print(f"  {tag:22s}  D={fmt(s.get('mean'))}  σ={fmt(s.get('std'))}  n={s.get('n', 0)}")


if __name__ == "__main__":
    main()
