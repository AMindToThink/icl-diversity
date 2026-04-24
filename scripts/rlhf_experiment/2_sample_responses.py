"""Sample K responses per prompt per policy for the RLHF diversity experiment.

Policies: OLMo-2-1124-7B base / SFT / DPO / Instruct.
Prompts:  AlpacaEval 200 + NoveltyBench curated 100.
Samples:  K=10 per prompt, T=1, top_p=1, max_new_tokens=100.

Behavior:
  - base → raw prompt (no chat template)
  - SFT / DPO / Instruct → tokenizer.apply_chat_template with the
    model's own template and `add_generation_prompt=True`.

Enforces GPU 1 only by setting `CUDA_VISIBLE_DEVICES=1` at import time,
so *inside* the process "cuda:0" refers to physical GPU 1. This matches
the project-wide "GPU 1 only" constraint.

Incremental JSONL: each (stage, prompt_id, sample_idx) is written as
soon as it's produced. Re-running the script skips any (stage,
prompt_id) whose K rows are already present.

Usage:
    uv run python scripts/rlhf_experiment/2_sample_responses.py \
        --stage base
    # repeat for sft, dpo, instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---- GPU 1 enforcement (must happen before any torch/vllm import) ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# -----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
GEN_DIR = RESULTS_DIR / "generations"

STAGE_TO_HF = {
    "base":     "allenai/OLMo-2-1124-7B",
    "sft":      "allenai/OLMo-2-1124-7B-SFT",
    "dpo":      "allenai/OLMo-2-1124-7B-DPO",
    "instruct": "allenai/OLMo-2-1124-7B-Instruct",
}

K_SAMPLES = 10
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 100
SEED = 42

PROMPT_SETS = [
    ("alpacaeval", RESULTS_DIR / "prompts_alpacaeval.jsonl"),
    ("nbcurated", RESULTS_DIR / "prompts_nbcurated.jsonl"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_keys(path: Path) -> set[tuple[str, int]]:
    """Return (prompt_id, sample_idx) keys already present in the JSONL."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add((rec["prompt_id"], int(rec["sample_idx"])))
    return keys


def build_inputs(
    prompts: list[dict], stage: str, tokenizer
) -> list[str]:
    """Build the raw text that will be tokenised by the policy for generation."""
    out = []
    for p in prompts:
        user_prompt = p["prompt"]
        if stage == "base":
            out.append(user_prompt)
            continue
        # Chat template for SFT/DPO/Instruct
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out.append(text)
    return out


def sample_stage(stage: str, max_new_tokens: int, seed: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    hf_path = STAGE_TO_HF[stage]
    print(f"=== stage={stage}  model={hf_path} ===", flush=True)

    # Tokeniser is needed for chat templating on the non-base stages.
    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    # vLLM engine.
    # Quadro RTX 8000 is compute-capability 7.5 → no bf16 support; use fp16.
    llm = LLM(
        model=hf_path,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        seed=seed,
    )

    for prompt_set_name, prompt_path in PROMPT_SETS:
        out_path = GEN_DIR / f"{stage}_{prompt_set_name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = read_jsonl(prompt_path)
        seen = existing_keys(out_path)
        needed = [
            (i, p) for i, p in enumerate(prompts)
            if not all((p["prompt_id"], k) in seen for k in range(K_SAMPLES))
        ]
        if not needed:
            print(f"  {prompt_set_name}: all {len(prompts)} prompts already sampled, skipping.")
            continue

        print(
            f"  {prompt_set_name}: {len(needed)} prompts to sample "
            f"({len(prompts) - len(needed)} already present)"
        )

        prompts_to_sample = [p for _, p in needed]
        inputs = build_inputs(prompts_to_sample, stage, tokenizer)

        sampling = SamplingParams(
            n=K_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        # vLLM returns outputs in the same order as the inputs
        outputs = llm.generate(inputs, sampling)

        # Append-mode; ensure we don't re-emit already-seen keys
        with out_path.open("a", encoding="utf-8") as f:
            for p, req_out in zip(prompts_to_sample, outputs):
                for k, comp in enumerate(req_out.outputs):
                    key = (p["prompt_id"], k)
                    if key in seen:
                        continue
                    row = {
                        "prompt_id": p["prompt_id"],
                        "prompt": p["prompt"],
                        "stage": stage,
                        "model": hf_path,
                        "prompt_set": prompt_set_name,
                        "sample_idx": k,
                        "response": comp.text,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_new_tokens": max_new_tokens,
                        "seed": seed,
                        "finish_reason": comp.finish_reason,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen.add(key)
        print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=list(STAGE_TO_HF.keys()),
        required=True,
        help="Which OLMo-2-7B stage to sample from.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # Sanity: verify GPU 1 is actually what we're using.
    import torch  # noqa: WPS433
    if not torch.cuda.is_available():
        print("CUDA not available; aborting.", file=sys.stderr)
        sys.exit(1)
    n = torch.cuda.device_count()
    print(f"[gpu] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"device_count={n}  name0={torch.cuda.get_device_name(0)}")

    sample_stage(
        stage=args.stage,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
