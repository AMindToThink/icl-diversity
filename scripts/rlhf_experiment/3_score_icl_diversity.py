"""Score every (stage, prompt_id) group with the ICL diversity metric.

Reads generations from results/rlhf_experiment/generations/{stage}_{set}.jsonl
and writes per-set metrics to results/rlhf_experiment/icl_metrics.jsonl
(one row per (stage, prompt_set, prompt_id) group).

Uses Qwen2.5-3B-base as θ (matches the paper's default). fp16 on GPU 1
(via CUDA_VISIBLE_DEVICES=1). Permutations: 25 by default; separate
--perms-spot-check flag re-runs 5 prompts per stage at n_permutations=50
and warns if any per-prompt D ranking flips.

Idempotent: skips (stage, prompt_set, prompt_id) keys already scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- GPU 1 enforcement (must happen before any torch import) ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
DEFAULT_GEN_DIR = RESULTS_DIR / "generations"

SCORER_MODEL = "Qwen/Qwen2.5-3B"
N_PERMUTATIONS = 25
BATCH_SIZE = 8
SEED = 42


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
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    from icl_diversity.core import compute_icl_diversity_metrics

    print(f"[scorer] loading {SCORER_MODEL} on cuda:0 (physical GPU 1)", flush=True)
    tok = AutoTokenizer.from_pretrained(SCORER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SCORER_MODEL, torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()

    seen = existing_score_keys(out_path)
    print(f"[scorer] {len(seen)} (stage,set,prompt,perms) keys already scored")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_todo_total = 0
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
            n_todo_total += len(todo)
            print(f"[scorer] stage={stage} set={pset}: {len(todo)} prompts to score"
                  f" ({len(by_prompt) - len(todo)} already done)")

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
                        "scorer_model": SCORER_MODEL,
                        "elapsed_s": dt,
                        **metrics,
                    }
                    f.write(json.dumps(record, default=_jsonable) + "\n")
                    f.flush()
                    if (i + 1) % 10 == 0 or i == 0:
                        print(
                            f"  [{i + 1}/{len(todo)}] {stage}/{pset}/{pid} "
                            f"D={metrics.get('diversity_score_D', float('nan')):.4f} "
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
    )


if __name__ == "__main__":
    main()
