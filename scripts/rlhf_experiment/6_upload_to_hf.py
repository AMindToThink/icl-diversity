"""Upload the OLMo-2-7B four-stage × 2-prompt-set generations to a PRIVATE HF dataset.

Created PRIVATE first (per user's global convention); user flips to public
when ready.

Usage:
    export HF_TOKEN=...     # must be set; fails loudly otherwise
    uv run python scripts/rlhf_experiment/6_upload_to_hf.py

Default repo: AMindToThink/olmo-2-1124-7b-four-stage-samples
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
GEN_DIR = RESULTS_DIR / "generations"

DEFAULT_REPO = "AMindToThink/olmo-2-1124-7b-four-stage-samples"
STAGES = ["base", "sft", "dpo", "instruct"]
PROMPT_SETS = ["alpacaeval", "nbcurated"]

STAGE_HF_PATH = {
    "base":     "allenai/OLMo-2-1124-7B",
    "sft":      "allenai/OLMo-2-1124-7B-SFT",
    "dpo":      "allenai/OLMo-2-1124-7B-DPO",
    "instruct": "allenai/OLMo-2-1124-7B-Instruct",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_rows(path: Path) -> int:
    n = 0
    with path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_staging_dir(root: Path, staging: Path) -> dict:
    """Copy generation JSONLs into a per-prompt-set layout and emit metadata.json."""
    staging.mkdir(parents=True, exist_ok=True)
    meta = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "description": (
            "OLMo-2-1124-7B four-stage (base / SFT / DPO / Instruct) samples "
            "on AlpacaEval (200 prompts) and NoveltyBench curated (100 prompts), "
            "K=10 per prompt, T=1.0, top_p=1.0, max_new_tokens=100. Produced for "
            "the ICL diversity metric paper; released because no prior work "
            "publishes K>=10 generations across a paired SFT/DPO/RL pipeline."
        ),
        "config": {
            "K": 10,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 100,
            "seed": 42,
        },
        "stages": {s: STAGE_HF_PATH[s] for s in STAGES},
        "prompt_sets": {
            "alpacaeval": {
                "source": "tatsu-lab/alpaca_eval",
                "subsample": "seed=42, n=200 out of 805",
            },
            "nbcurated": {
                "source": "yimingzhang/novelty-bench",
                "subsample": "curated split, all 100 prompts",
            },
        },
        "files": {},
    }

    for pset in PROMPT_SETS:
        pset_dir = staging / pset
        pset_dir.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            src = GEN_DIR / f"{stage}_{pset}.jsonl"
            if not src.exists():
                print(f"[upload] warning: {src} missing; skipping")
                continue
            dst = pset_dir / f"{stage}.jsonl"
            dst.write_bytes(src.read_bytes())
            rel = f"{pset}/{stage}.jsonl"
            meta["files"][rel] = {
                "n_rows": _count_rows(dst),
                "sha256": _sha256(dst),
                "size_bytes": dst.stat().st_size,
            }

    # Copy the prompt JSONLs too so consumers have the original inputs.
    prompts_dir = staging / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for pset in PROMPT_SETS:
        src = RESULTS_DIR / f"prompts_{pset}.jsonl"
        if src.exists():
            dst = prompts_dir / f"{pset}.jsonl"
            dst.write_bytes(src.read_bytes())
            rel = f"prompts/{pset}.jsonl"
            meta["files"][rel] = {
                "n_rows": _count_rows(dst),
                "sha256": _sha256(dst),
                "size_bytes": dst.stat().st_size,
            }

    (staging / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def write_readme(staging: Path, meta: dict, repo: str) -> None:
    total_rows = sum(v["n_rows"] for v in meta["files"].values())
    content = f"""---
license: cc-by-nc-4.0
task_categories:
  - text-generation
tags:
  - diversity
  - rlhf
  - dpo
  - olmo
pretty_name: OLMo-2-1124-7B four-stage diversity samples
---

# OLMo-2-1124-7B four-stage samples (for diversity research)

This dataset contains **K=10 sampled responses per prompt** from four stages
of the OLMo-2-1124-7B post-training pipeline on two prompt sets, produced
for the ICL diversity metric paper (Khoriaty, Williams-King, Feng, in preparation).

**Why this exists.** As of April 2026, no prior work releases ≥10 samples
per prompt from a paired SFT/DPO/RL pipeline. Kirk et al.'s original
RLHF-gen-div paper logged their K=16 BoN outputs to a private Weights &
Biases project. NoveltyBench publishes prompts and scoring rubrics but not
the 20-model generations its leaderboard was built on. Tulu-3 and OLMo-2
release weights + preference pairs (K=2) but not diversity-scale
K-samples-per-prompt. This dataset closes that gap for one well-known
paired pipeline.

## Stages

| Split | Model |
|-------|-------|
| `base`     | [`allenai/OLMo-2-1124-7B`](https://huggingface.co/allenai/OLMo-2-1124-7B) |
| `sft`      | [`allenai/OLMo-2-1124-7B-SFT`](https://huggingface.co/allenai/OLMo-2-1124-7B-SFT) |
| `dpo`      | [`allenai/OLMo-2-1124-7B-DPO`](https://huggingface.co/allenai/OLMo-2-1124-7B-DPO) |
| `instruct` | [`allenai/OLMo-2-1124-7B-Instruct`](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) (RLVR final) |

## Prompt sets

| Split | Source | N prompts |
|-------|--------|-----------|
| `alpacaeval` | `tatsu-lab/alpaca_eval` (subsampled seed=42, n=200/805) | 200 |
| `nbcurated`  | `yimingzhang/novelty-bench` curated split (all 100 prompts) | 100 |

## Sampling configuration

- K = 10 samples per prompt per stage
- Temperature = 1.0, top-p = 1.0
- `max_new_tokens = 100`
- Seed = 42
- Backend: vLLM
- Base stage → raw prompt (no chat template)
- SFT / DPO / Instruct → `tokenizer.apply_chat_template(..., add_generation_prompt=True)`

Per-row schema:
```
{{ "prompt_id", "prompt", "stage", "model", "prompt_set",
   "sample_idx", "response", "temperature", "top_p",
   "max_new_tokens", "seed", "finish_reason" }}
```

Total rows: **{total_rows}** across **{len(meta['files'])}** files.

## Layout

```
alpacaeval/{{base,sft,dpo,instruct}}.jsonl
nbcurated/{{base,sft,dpo,instruct}}.jsonl
prompts/{{alpacaeval,nbcurated}}.jsonl      # original inputs
metadata.json                               # checksums + row counts
```

## Loading

```python
from datasets import load_dataset

ds = load_dataset("{repo}", data_files="alpacaeval/sft.jsonl")
```

## Citation

If you use these samples, please cite the ICL diversity paper (in preparation)
and credit the AllenAI OLMo-2 team for the underlying checkpoints.

## License

CC-BY-NC 4.0 (matching Kirk et al.'s original rlhf-gen-div license — our
experimental protocol derives from theirs).
"""
    (staging / "README.md").write_text(content)


def upload(repo: str, staging: Path, private: bool) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required. Set it to an HF access "
            "token with 'write' scope before running this script."
        )
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"[upload] pushed to https://huggingface.co/datasets/{repo} (private={private})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument(
        "--staging",
        type=Path,
        default=REPO_ROOT / "results" / "rlhf_experiment" / "_hf_staging",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public. Default is PRIVATE (per user convention).",
    )
    args = ap.parse_args()

    meta = build_staging_dir(REPO_ROOT, args.staging)
    write_readme(args.staging, meta, args.repo)
    print(f"[upload] staged {len(meta['files'])} files in {args.staging}")
    upload(repo=args.repo, staging=args.staging, private=not args.public)


if __name__ == "__main__":
    main()
