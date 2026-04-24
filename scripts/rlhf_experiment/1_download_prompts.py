"""Download AlpacaEval 200 and NoveltyBench curated prompt sets.

Reads `tatsu-lab/alpaca_eval` eval split (805 prompts) → random seed=42 subsample of 200.
Reads `yimingzhang/novelty-bench` curated split (100 prompts) in full.

Writes two JSONLs under `results/rlhf_experiment/`:
  prompts_alpacaeval.jsonl    — 200 rows, {prompt_id, prompt, source}
  prompts_nbcurated.jsonl     — 100 rows, {prompt_id, prompt, source}

Idempotent: re-running skips prompts already present by prompt_id.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "rlhf_experiment"

ALPACAEVAL_REPO = "tatsu-lab/alpaca_eval"
ALPACAEVAL_FILE = "alpaca_eval.json"  # 805 prompts as JSON list
NBCURATED_REPO = "yimingzhang/novelty-bench"

SEED = 42
ALPACAEVAL_N = 200


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_alpacaeval(n: int, seed: int, out_path: Path) -> None:
    """AlpacaEval eval has 805 instructions. We subsample `n` with `seed`."""
    local_path = hf_hub_download(
        repo_id=ALPACAEVAL_REPO, repo_type="dataset", filename=ALPACAEVAL_FILE
    )
    with open(local_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) < n:
        raise RuntimeError(
            f"AlpacaEval has only {len(records)} rows; cannot subsample {n}."
        )
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), n)
    indices.sort()
    rows = []
    for i, idx in enumerate(indices):
        rec = records[idx]
        rows.append(
            {
                "prompt_id": f"alpacaeval-{i:03d}",
                "orig_index": int(idx),
                "prompt": rec["instruction"],
                "source": ALPACAEVAL_REPO,
                "source_split": "eval",
                "dataset_name": rec.get("dataset"),
            }
        )
    _write_jsonl(out_path, rows)
    print(f"wrote {out_path}: {len(rows)} prompts (from {len(records)} total, seed={seed})")


def download_nbcurated(out_path: Path) -> None:
    """NoveltyBench repo has parquet files at nb-curated-*.parquet; we pull via hf_hub_download."""
    # The dataset has two configs, both parquet: curated (100) and wildchat (1000).
    # We want curated.
    local_path = hf_hub_download(
        repo_id=NBCURATED_REPO,
        repo_type="dataset",
        filename="data/curated-00000-of-00001.parquet",
    )
    import pyarrow.parquet as pq

    table = pq.read_table(local_path)
    rows_pa = table.to_pylist()
    rows = []
    for rec in rows_pa:
        rows.append(
            {
                "prompt_id": rec["id"],
                "prompt": rec["prompt"],
                "source": NBCURATED_REPO,
                "source_split": "curated",
            }
        )
    _write_jsonl(out_path, rows)
    print(f"wrote {out_path}: {len(rows)} prompts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--alpacaeval-n", type=int, default=ALPACAEVAL_N)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    download_alpacaeval(
        n=args.alpacaeval_n,
        seed=args.seed,
        out_path=args.out_dir / "prompts_alpacaeval.jsonl",
    )
    download_nbcurated(out_path=args.out_dir / "prompts_nbcurated.jsonl")


if __name__ == "__main__":
    main()
