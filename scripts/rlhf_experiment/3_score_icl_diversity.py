"""Score every (stage, prompt_id) group with the ICL diversity metric.

Reads generations from results/rlhf_experiment/generations/{stage}_{set}.jsonl
and writes per-set metrics to results/rlhf_experiment/icl_metrics.jsonl
(one row per (stage, prompt_set, prompt_id) group).

The scorer (theta) defaults to Qwen2.5-3B-base (the paper's default). Override
with --scorer-model to re-score the same generations under a different theta;
the OLMo self-scoring matrix (in-family graders) uses e.g.
--scorer-model allenai/OLMo-2-1124-7B. fp16 by default.

GPU selection: set CUDA_VISIBLE_DEVICES in the environment BEFORE launching
(e.g. `CUDA_VISIBLE_DEVICES=0 uv run python ...`). The script defaults to GPU 1
only if the variable is unset, preserving the project-wide "GPU 1" default while
letting the matrix run two graders on two GPUs in parallel.

Permutations: 25 by default. Idempotent: skips (stage, prompt_set, prompt_id,
n_permutations) keys already scored in the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- GPU default (must happen before any torch import) ----
# Respect a caller-set CUDA_VISIBLE_DEVICES; default to GPU 1 if unset so the
# matrix can place each grader on its own GPU via the environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
DEFAULT_GEN_DIR = RESULTS_DIR / "generations"

DEFAULT_SCORER_MODEL = "Qwen/Qwen2.5-3B"
N_PERMUTATIONS = 25
BATCH_SIZE = 8
SEED = 42
# Safety margin for the upfront context-length check: permutation reordering can
# shift BPE merges at response boundaries by a few tokens vs the canonical order.
CONTEXT_MARGIN = 8


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_score_keys(path: Path) -> set[tuple[str, str, str, int]]:
    """(stage, prompt_set, prompt_id, n_permutations) keys already scored."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add(
                (rec["stage"], rec["prompt_set"], rec["prompt_id"], int(rec["n_permutations"]))
            )
    return keys


def group_by_prompt(rows: list[dict]) -> dict[str, list[dict]]:
    """Group generation rows by prompt_id, sort by sample_idx."""
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    for pid in by_prompt:
        by_prompt[pid].sort(key=lambda r: r["sample_idx"])
    return by_prompt


def score_all(
    stages: list[str],
    prompt_sets: list[str],
    n_permutations: int,
    batch_size: int,
    out_path: Path,
    limit: int | None,
    gen_dir: Path,
    scorer_model: str,
    max_context_tokens: int | None = None,
    skip_over_context: bool = False,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    from icl_diversity.core import compute_icl_diversity_metrics

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[scorer] loading {scorer_model} on cuda:0 (CUDA_VISIBLE_DEVICES={visible})", flush=True)
    tok = AutoTokenizer.from_pretrained(scorer_model)
    model = AutoModelForCausalLM.from_pretrained(
        scorer_model, torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()

    seen = existing_score_keys(out_path)
    print(f"[scorer] {len(seen)} (stage,set,prompt,perms) keys already scored")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: build work lists, then validate context lengths upfront so a
    # short-context scorer (e.g. GPT-2, 1024) fails loudly before any GPU work
    # instead of crashing mid-run on the first oversized group.
    work: list[tuple[str, str, dict, list[str]]] = []
    for stage in stages:
        for pset in prompt_sets:
            gen_path = gen_dir / f"{stage}_{pset}.jsonl"
            if not gen_path.exists():
                print(f"[scorer] skipping missing {gen_path}", file=sys.stderr)
                continue
            rows = read_jsonl(gen_path)
            by_prompt = group_by_prompt(rows)
            todo = [
                pid for pid in by_prompt
                if (stage, pset, pid, n_permutations) not in seen
            ]
            if limit is not None:
                todo = todo[:limit]
            work.append((stage, pset, by_prompt, todo))

    ctx_limit = max_context_tokens
    if ctx_limit is None:
        ctx_limit = getattr(model.config, "max_position_embeddings", None)
    if ctx_limit is None:
        ctx_limit = getattr(model.config, "n_positions", None)
    if not isinstance(ctx_limit, int):
        # No usable limit in the config (e.g. mocked models in tests): skip the check.
        ctx_limit = None
    overflow: list[dict] = []
    if ctx_limit is not None:
        from icl_diversity.core import format_conditioning_context

        for stage, pset, by_prompt, todo in work:
            for pid in todo:
                group = by_prompt[pid]
                prefix, last = format_conditioning_context(
                    group[0]["prompt"],
                    [g["response"] for g in group[:-1]],
                    group[-1]["response"],
                    format_mode="instruct",
                )
                n_tok = len(tok(prefix + last)["input_ids"])
                if n_tok + CONTEXT_MARGIN > ctx_limit:
                    overflow.append({
                        "stage": stage, "prompt_set": pset, "prompt_id": pid,
                        "n_tokens": n_tok, "context_limit": int(ctx_limit),
                    })
        if overflow and not skip_over_context:
            listing = "\n".join(
                f"  {o['stage']}/{o['prompt_set']}/{o['prompt_id']}: {o['n_tokens']} tokens"
                for o in overflow
            )
            raise ValueError(
                f"{len(overflow)} groups exceed the scorer's context limit "
                f"({ctx_limit} tokens, margin {CONTEXT_MARGIN}):\n{listing}\n"
                "Re-run with --skip-over-context to skip them explicitly; skips are "
                "recorded in a .skipped_context_overflow.json sidecar for downstream "
                "analysis to account for."
            )
        if overflow:
            skipped_path = out_path.with_name(
                out_path.stem + ".skipped_context_overflow.json"
            )
            skipped_path.parent.mkdir(parents=True, exist_ok=True)
            skipped_path.write_text(json.dumps({
                "scorer_model": scorer_model,
                "context_limit": int(ctx_limit),
                "margin": CONTEXT_MARGIN,
                "skipped": overflow,
            }, indent=2))
            for o in overflow:
                print(
                    f"[scorer] SKIP (context {o['n_tokens']} + margin > {ctx_limit}): "
                    f"{o['stage']}/{o['prompt_set']}/{o['prompt_id']}",
                    flush=True,
                )
            print(f"[scorer] {len(overflow)} skipped groups recorded in {skipped_path}",
                  flush=True)
            skip_keys = {(o["stage"], o["prompt_set"], o["prompt_id"]) for o in overflow}
            work = [
                (s, p, bp, [pid for pid in td if (s, p, pid) not in skip_keys])
                for s, p, bp, td in work
            ]

    # Pass 2: score.
    n_todo_total = 0
    for stage, pset, by_prompt, todo in work:
        n_todo_total += len(todo)
        print(f"[scorer] stage={stage} set={pset}: {len(todo)} prompts to score"
              f" ({len(by_prompt) - len(todo)} already done or skipped)")

        with out_path.open("a", encoding="utf-8") as f:
            for i, pid in enumerate(todo):
                group = by_prompt[pid]
                prompt = group[0]["prompt"]
                responses = [g["response"] for g in group]
                t0 = time.perf_counter()
                metrics = compute_icl_diversity_metrics(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    responses=responses,
                    n_permutations=n_permutations,
                    seed=SEED,
                    batch_size=batch_size,
                    format_mode="instruct",
                )
                dt = time.perf_counter() - t0
                record = {
                    "stage": stage,
                    "prompt_set": pset,
                    "prompt_id": pid,
                    "n_permutations": n_permutations,
                    "n_responses": len(responses),
                    "scorer_model": scorer_model,
                    "elapsed_s": dt,
                    **metrics,
                }
                f.write(json.dumps(record, default=_jsonable) + "\n")
                f.flush()
                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"  [{i + 1}/{len(todo)}] {stage}/{pset}/{pid} "
                        f"D={metrics.get('diversity_score_D_C_an', float('nan')):.4f} "
                        f"({dt:.1f}s)",
                        flush=True,
                    )
    print(f"[scorer] done, scored {n_todo_total} new prompts")


def _jsonable(obj):
    # Handle torch Tensors and numpy arrays in metric outputs
    try:
        import numpy as np  # noqa: WPS433
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    try:
        import torch  # noqa: WPS433
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["base", "sft", "dpo", "instruct"],
    )
    ap.add_argument(
        "--prompt-sets",
        nargs="+",
        default=["alpacaeval", "nbcurated"],
    )
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument(
        "--scorer-model",
        type=str,
        default=DEFAULT_SCORER_MODEL,
        help=(
            "HF model id for theta (the scorer). Default Qwen/Qwen2.5-3B. "
            "Set e.g. allenai/OLMo-2-1124-7B for in-family self-scoring."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "icl_metrics.jsonl",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: score only first N prompts per (stage,set). Useful for smoke tests.",
    )
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help=(
            "Context-length limit for the upfront feasibility check. Default: the "
            "scorer's max_position_embeddings / n_positions from its config."
        ),
    )
    ap.add_argument(
        "--skip-over-context",
        action="store_true",
        help=(
            "Skip groups whose conditioning context exceeds the scorer's context "
            "limit instead of raising. Skips are printed and recorded in a "
            ".skipped_context_overflow.json sidecar next to --out."
        ),
    )
    ap.add_argument(
        "--gen-dir",
        type=Path,
        default=DEFAULT_GEN_DIR,
        help=(
            "Directory containing {stage}_{set}.jsonl generation files. "
            "Override to score the length-matched copy at "
            "results/rlhf_experiment/generations_length_matched/."
        ),
    )
    args = ap.parse_args()

    score_all(
        stages=args.stages,
        prompt_sets=args.prompt_sets,
        n_permutations=args.n_permutations,
        batch_size=args.batch_size,
        out_path=args.out,
        limit=args.limit,
        gen_dir=args.gen_dir,
        scorer_model=args.scorer_model,
        max_context_tokens=args.max_context_tokens,
        skip_over_context=args.skip_over_context,
    )


if __name__ == "__main__":
    main()
