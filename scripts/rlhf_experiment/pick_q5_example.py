"""Pick one real example prompt for NeurIPS reviewer 2bo8's Q5.

Q5 asked: "Do you have example continuation texts corresponding to [Section 5,
the OLMo-2-7B post-training experiment], demonstrating the decrease in
diversity indicated by the DCaN metric?"

This script selects ONE prompt from the OLMo-2-7B RLHF-diversity experiment
and extracts real base-stage and instruct(RLVR)-stage response excerpts for
it, so the reviewer can see visible mode collapse rather than just a table
of scalars.

Selection rule (mirrors the "representative, not outlier" convention used
for the McDiv_nuggets confound examples in scripts/dataset_confound_analysis.py
and documented in paper/sections/appC_mcdiv_confounds.tex):

    1. Restrict to prompt_ids with a metrics record at all four stages
       (base, sft, dpo, instruct) in both prompt_set groups combined.
    2. Rank those prompts by D_drop = D(base) - D(instruct), where D is the
       paper's primary diversity_score_D_C_an (C * a_n, per-byte, sec 6.3).
    3. Take the top quartile by D_drop (floor(0.75 * n) .. n-1 in ascending
       order) -- these are the prompts with the most dramatic collapse.
    4. Within the top quartile, prefer the nbcurated (NoveltyBench) subset
       if it is non-empty (creative-writing prompts make diversity visually
       obvious); otherwise fall back to alpacaeval.
    5. Pick the MEDIAN prompt of that preferred subset by D_drop (not the
       maximum) -- a representative case, not a cherry-picked extreme.

Manual eyeball check (documented per the task's honesty requirement): the
first candidate produced by this rule, nbcurated prompt "curated-32" (a
GPU-recommendation prompt), was inspected directly against the generation
files. Base-stage completions are visibly divergent (different questions,
different GPUs, different tone -- the base model doesn't even reliably
answer the question). Instruct-stage completions are visibly near-identical
in structure ("For a productivity/budget of $1000 ... I would recommend
the ASUS ROG Strix / NVIDIA GeForce RTX 3060 ..."). This satisfied the
"visually obvious collapse" criterion on the first try, so no fallback to
the next-ranked prompt was needed. PICK_RANK_OFFSET below is left at 0 for
that reason, but is a documented knob: bump it to 1, 2, ... to move to the
next prompt up the ranking if a future re-run of this script (e.g. after
regenerating results/rlhf_experiment/icl_metrics_length_matched.jsonl)
produces a pick that no longer shows visible collapse by eye.

Reads:
    results/rlhf_experiment/icl_metrics_length_matched.jsonl
    results/rlhf_experiment/generations_length_matched/base_{prompt_set}.jsonl
    results/rlhf_experiment/generations_length_matched/instruct_{prompt_set}.jsonl

Writes:
    paper/neurips_reviews/q5_example.md

Usage:
    uv run python scripts/rlhf_experiment/pick_q5_example.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
METRICS_PATH = RESULTS_DIR / "icl_metrics_length_matched.jsonl"
GEN_DIR = RESULTS_DIR / "generations_length_matched"
OUTPUT_PATH = REPO_ROOT / "paper" / "neurips_reviews" / "q5_example.md"

STAGES = ["base", "sft", "dpo", "instruct"]
STAGE_LABELS = {"base": "base", "sft": "SFT", "dpo": "DPO", "instruct": "RLVR"}
N_EXCERPTS = 4
EXCERPT_MAX_CHARS = 120
PROMPT_MAX_CHARS = 200
QUOTABLE_BLOCK_MAX_CHARS = 2500

# See the module docstring: the median-of-top-quartile pick already showed
# visible collapse by eye, so no fallback was exercised. Left here as a
# documented, reproducible knob rather than a silently hardcoded index.
PICK_RANK_OFFSET = 0


@dataclass(frozen=True)
class PromptStageMetrics:
    prompt_set: str
    prompt_id: str
    d_by_stage: dict[str, float]

    @property
    def d_drop(self) -> float:
        return self.d_by_stage["base"] - self.d_by_stage["instruct"]


def load_complete_prompts(metrics_path: Path) -> list[PromptStageMetrics]:
    """Load per-prompt D values, keeping only prompts present at all 4 stages."""
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            key = (row["prompt_set"], row["prompt_id"])
            by_key.setdefault(key, {})[row["stage"]] = row["diversity_score_D_C_an"]

    complete: list[PromptStageMetrics] = []
    for (prompt_set, prompt_id), stage_d in by_key.items():
        if all(stage in stage_d for stage in STAGES):
            complete.append(PromptStageMetrics(prompt_set, prompt_id, dict(stage_d)))

    if not complete:
        raise ValueError(f"No prompts found with all 4 stages in {metrics_path}")
    return complete


def select_prompt(
    complete: list[PromptStageMetrics], rank_offset: int = 0
) -> tuple[PromptStageMetrics, dict]:
    """Apply the selection rule from the module docstring.

    Returns the selected prompt plus a dict of selection-audit info (for the
    md footer / for printing).
    """
    ranked = sorted(complete, key=lambda p: p.d_drop)
    n = len(ranked)
    q75_start = int(n * 0.75)
    top_quartile = ranked[q75_start:]

    nb_subset = [p for p in top_quartile if p.prompt_set == "nbcurated"]
    aa_subset = [p for p in top_quartile if p.prompt_set == "alpacaeval"]
    preferred_set = "nbcurated" if nb_subset else "alpacaeval"
    subset = nb_subset if nb_subset else aa_subset
    subset_sorted = sorted(subset, key=lambda p: p.d_drop)

    median_idx = len(subset_sorted) // 2
    pick_idx = median_idx + rank_offset
    if not (0 <= pick_idx < len(subset_sorted)):
        raise ValueError(
            f"rank_offset={rank_offset} moves pick_idx to {pick_idx}, "
            f"outside subset of size {len(subset_sorted)}"
        )
    picked = subset_sorted[pick_idx]

    audit = {
        "n_complete_prompts": n,
        "top_quartile_start_idx": q75_start,
        "top_quartile_size": len(top_quartile),
        "preferred_set": preferred_set,
        "subset_size": len(subset_sorted),
        "median_idx": median_idx,
        "pick_idx": pick_idx,
        "rank_offset": rank_offset,
    }
    return picked, audit


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (incl. newlines) to single spaces.

    Raw responses can contain literal blank lines (e.g. the base model
    continuing with paragraph breaks). Left as-is, those newlines break
    Markdown numbered-list rendering when the block is pasted into an
    OpenReview comment (list items spill across lines / numbering breaks).
    This only affects display formatting, not content: no words are added,
    removed, or reordered.
    """
    return re.sub(r"\s+", " ", text).strip()


def truncate_at_word_boundary(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Truncate text to ~max_chars, breaking at the last word boundary, with '...'."""
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "..."


def load_first_n_responses(
    prompt_set: str, prompt_id: str, stage: str, n: int = N_EXCERPTS
) -> tuple[str, list[str]]:
    """Return (prompt_text, first n response texts) in sample_idx order."""
    path = GEN_DIR / f"{stage}_{prompt_set}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["prompt_id"] == prompt_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"prompt_id={prompt_id!r} not found in {path}")
    rows.sort(key=lambda r: r["sample_idx"])
    prompt_text = rows[0]["prompt"]
    responses = [r["response"] for r in rows[:n]]
    if len(responses) < n:
        raise ValueError(
            f"Only found {len(responses)} responses for {prompt_id!r} in {path}, need {n}"
        )
    return prompt_text, responses


def build_markdown(
    picked: PromptStageMetrics,
    audit: dict,
    prompt_text: str,
    base_excerpts: list[str],
    instruct_excerpts: list[str],
) -> str:
    prompt_display = truncate_at_word_boundary(prompt_text.strip(), PROMPT_MAX_CHARS)
    d = picked.d_by_stage

    lines: list[str] = []
    lines.append(
        "## Response to Q5: example continuations, OLMo-2-7B base vs RLVR "
        "(NoveltyBench)"
    )
    lines.append("")
    lines.append(f'**Prompt** (`{picked.prompt_id}`): "{prompt_display}"')
    lines.append("")
    lines.append(
        f"**D = C x a_n by stage** (this prompt): "
        f"base {d['base']:.3f}, SFT {d['sft']:.3f}, DPO {d['dpo']:.3f}, "
        f"RLVR {d['instruct']:.3f}"
    )
    lines.append("")
    lines.append(f"**Base** ({len(base_excerpts)} of 10 samples, OLMo-2-7B base):")
    for i, resp in enumerate(base_excerpts, 1):
        excerpt = truncate_at_word_boundary(resp.strip())
        lines.append(f'{i}. "{excerpt}"')
    lines.append("")
    lines.append(
        f"**RLVR** ({len(instruct_excerpts)} of 10 samples, OLMo-2-7B Instruct/RLVR):"
    )
    for i, resp in enumerate(instruct_excerpts, 1):
        excerpt = truncate_at_word_boundary(resp.strip())
        lines.append(f'{i}. "{excerpt}"')
    lines.append("")
    lines.append(
        f"*Selection rule: among the {audit['n_complete_prompts']} prompts with "
        "metrics at all four stages (base/SFT/DPO/RLVR), we took the top "
        f"quartile by D(base) - D(RLVR) ({audit['top_quartile_size']} prompts), "
        f"restricted to the NoveltyBench subset within it "
        f"({audit['subset_size']} prompts, preferred over AlpacaEval so the "
        "example comes from a benchmark built to elicit distinct answers), "
        "and picked the median by drop "
        f"(rank {audit['pick_idx'] + 1} of {audit['subset_size']}) -- a "
        "representative case, not the single largest drop.*"
    )

    quotable_block = "\n".join(lines)
    if len(quotable_block) > QUOTABLE_BLOCK_MAX_CHARS:
        raise ValueError(
            f"Quotable block is {len(quotable_block)} chars, exceeds "
            f"{QUOTABLE_BLOCK_MAX_CHARS}-char budget for an OpenReview comment"
        )

    footer = (
        "\n\n<!-- Generated by scripts/rlhf_experiment/pick_q5_example.py.\n"
        "     Source: results/rlhf_experiment/icl_metrics_length_matched.jsonl,\n"
        f"     results/rlhf_experiment/generations_length_matched/"
        f"{{base,instruct}}_{picked.prompt_set}.jsonl.\n"
        "     Regenerate: uv run python scripts/rlhf_experiment/pick_q5_example.py -->\n"
    )
    return quotable_block + footer, len(quotable_block)


def main() -> None:
    complete = load_complete_prompts(METRICS_PATH)
    picked, audit = select_prompt(complete, rank_offset=PICK_RANK_OFFSET)

    print(f"Loaded {len(complete)} prompts with all 4 stages present.")
    print(
        f"Top quartile: {audit['top_quartile_size']} prompts "
        f"(indices {audit['top_quartile_start_idx']}..{len(complete) - 1} of "
        f"{len(complete)}, ascending D_drop)"
    )
    print(
        f"Preferred subset: {audit['preferred_set']} "
        f"({audit['subset_size']} prompts in top quartile)"
    )
    print(
        f"Picked rank {audit['pick_idx'] + 1} of {audit['subset_size']} "
        f"(median_idx={audit['median_idx']}, rank_offset={audit['rank_offset']})"
    )
    print(
        f"Selected: prompt_set={picked.prompt_set!r} prompt_id={picked.prompt_id!r} "
        f"D_drop={picked.d_drop:.4f}"
    )
    print(
        "D by stage: "
        + ", ".join(f"{STAGE_LABELS[s]}={picked.d_by_stage[s]:.4f}" for s in STAGES)
    )

    prompt_text, base_responses = load_first_n_responses(
        picked.prompt_set, picked.prompt_id, "base"
    )
    _, instruct_responses = load_first_n_responses(
        picked.prompt_set, picked.prompt_id, "instruct"
    )

    markdown, quotable_chars = build_markdown(
        picked, audit, prompt_text, base_responses, instruct_responses
    )
    print(f"Quotable block: {quotable_chars} chars (budget {QUOTABLE_BLOCK_MAX_CHARS})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
