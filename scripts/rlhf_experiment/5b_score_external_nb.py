"""Score the alon-albalak/*-noveltybench-responses datasets with C × a_n.

These are seven external models' K=10 responses on the NoveltyBench
curated 100-prompt set. Scoring them gives an independent data point for
our metric (complementary to the four OLMo-2-7B stages).

Models scored (all cached after first run):
  qwen-4b, qwen-8b, qwen-235b-a22b, llama-33-70b,
  gpt-5, gpt-5-nano, claude-sonnet-4-5

Output: results/rlhf_experiment/external_nb_metrics.jsonl
        one row per (model, prompt_id), schema similar to icl_metrics.jsonl.

GPU 1 only: CUDA_VISIBLE_DEVICES=1 before any torch import.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"

SCORER_MODEL = "Qwen/Qwen2.5-3B"
N_PERMUTATIONS = 25
BATCH_SIZE = 8
SEED = 42

MODELS = [
    "qwen-4b",
    "qwen-8b",
    "qwen-235b-a22b",
    "llama-33-70b",
    "gpt-5",
    "gpt-5-nano",
    "claude-sonnet-4-5",
]


def load_external_responses(model_tag: str) -> list[dict]:
    """Download the parquet and return list of {prompt_id, prompt, completions}."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    repo = f"alon-albalak/{model_tag}-noveltybench-responses"
    path = hf_hub_download(
        repo_id=repo, repo_type="dataset", filename="data/train-00000-of-00001.parquet"
    )
    table = pq.read_table(path)
    rows = []
    for rec in table.to_pylist():
        rows.append(
            {
                "prompt_id": rec["id"],
                "prompt": rec["prompt"],
                "completions": list(rec["completions"]),
            }
        )
    return rows


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add((rec["external_model"], rec["prompt_id"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "external_nb_metrics.jsonl"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="First N prompts per model (smoke-test mode).",
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    from icl_diversity.core import compute_icl_diversity_metrics

    print(f"[ext] loading {SCORER_MODEL} fp16 on cuda:0 (physical GPU 1)")
    tok = AutoTokenizer.from_pretrained(SCORER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SCORER_MODEL, torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()

    seen = existing_keys(args.out)
    print(f"[ext] {len(seen)} external (model, prompt) keys already scored")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for tag in args.models:
        print(f"[ext] === {tag} ===")
        rows = load_external_responses(tag)
        if args.limit:
            rows = rows[: args.limit]
        with args.out.open("a") as f:
            for i, row in enumerate(rows):
                key = (tag, row["prompt_id"])
                if key in seen:
                    continue
                responses = row["completions"]
                if len(responses) < 2:
                    print(f"[ext] skipping {key} (only {len(responses)} completions)")
                    continue
                t0 = time.perf_counter()
                metrics = compute_icl_diversity_metrics(
                    model=model,
                    tokenizer=tok,
                    prompt=row["prompt"],
                    responses=responses,
                    n_permutations=args.n_permutations,
                    seed=SEED,
                    batch_size=args.batch_size,
                    format_mode="instruct",
                )
                dt = time.perf_counter() - t0
                record = {
                    "external_model": tag,
                    "prompt_id": row["prompt_id"],
                    "n_responses": len(responses),
                    "n_permutations": args.n_permutations,
                    "scorer_model": SCORER_MODEL,
                    "elapsed_s": dt,
                    **metrics,
                }
                f.write(json.dumps(record, default=_jsonable) + "\n")
                f.flush()
                seen.add(key)
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{i + 1}/{len(rows)}] {tag}/{row['prompt_id']} "
                        f"D={metrics.get('diversity_score_D', float('nan')):.4f} "
                        f"({dt:.1f}s)"
                    )


def _jsonable(obj):
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


if __name__ == "__main__":
    main()
