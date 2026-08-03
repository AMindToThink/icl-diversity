"""Template-sentence experiment: ICL diversity (D = C * a_n) vs SentBERT.

Hypothesis: response sets built from a fixed syntactic template with random
word slots share no semantic content, so embedding-based diversity
(Tevet & Berant's SentBERT baseline) rates them maximally diverse, while the
ICL diversity metric detects the shared structure in-context: the a_k curve
drops as the base model theta learns the template, giving a low
D = C * a_n (``diversity_score_D_C_an``).

Conditions (all template conditions draw from the same word lists, holding
semantic scatter constant while varying only syntactic variety):

- ``canonical``:   all responses use frame 0,
                   "The {adj} {noun} {verb} the {adj2} {noun2}."
- ``frames_m``:    responses drawn evenly from m randomly chosen frames
                   (m in --frame-counts; m=20 is the genuinely
                   structure-diverse control).
- ``paraphrase``:  20 hand-written paraphrases of one meaning; the agreement
                   anchor where both metrics should be low.

Each condition runs --n-draws fully independent draws (fresh words, fresh
frame subset, fresh order), mirroring scripts/run_mode_count_experiment.py:
n_permutations=1 per draw, statistical power from the independent draws.

Usage:
    uv run python scripts/run_template_vs_sentbert.py \
        --base-model gpt2 --device cuda:0 --batch-size 16 \
        --output results/template_vs_sentbert/gpt2.json

    uv run python scripts/run_template_vs_sentbert.py \
        --base-model Qwen/Qwen2.5-3B --device cuda:0 --torch-dtype bfloat16 \
        --batch-size 8 --output results/template_vs_sentbert/qwen2.5-3b.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity import compute_icl_diversity_metrics
from icl_diversity.baseline_metrics import (
    averaged_distinct_ngrams,
    mean_pairwise_cosine_similarity,
    sentbert_diversity,
)
from icl_diversity.template_scenarios import (
    FRAME_LABELS,
    PARAPHRASE_RESPONSES,
    TEMPLATE_PROMPT,
    generate_template_responses,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "template_vs_sentbert"
    / "gpt2.json"
)
# Tevet & Berant's SentBERT model (diversity-eval/diversity_metrics.py,
# class SentBert): 'bert-large-nli-stsb-mean-tokens'.
DEFAULT_SENTBERT = "sentence-transformers/bert-large-nli-stsb-mean-tokens"


def build_conditions(
    frame_counts: list[int],
) -> list[tuple[str, int | None, list[int] | None]]:
    """(condition_name, n_frames, frame_pool) triples."""
    conditions: list[tuple[str, int | None, list[int] | None]] = [("canonical", 1, [0])]
    conditions += [(f"frames_{m}", m, None) for m in frame_counts]
    conditions.append(("paraphrase", None, None))
    return conditions


def draw_responses(
    condition: str,
    n_frames: int | None,
    frame_pool: list[int] | None,
    n_responses: int,
    draw_seed: int,
) -> tuple[list[str], list[str] | None]:
    """Responses (shuffled) and, for template conditions, their frame labels."""
    if condition == "paraphrase":
        if n_responses > len(PARAPHRASE_RESPONSES):
            raise ValueError(
                f"paraphrase condition has only {len(PARAPHRASE_RESPONSES)} "
                f"responses, requested {n_responses}"
            )
        rng = random.Random(draw_seed ^ 0x5A5A5A5A)
        responses = list(PARAPHRASE_RESPONSES)
        rng.shuffle(responses)
        return responses[:n_responses], None

    assert n_frames is not None
    responses, frame_ids = generate_template_responses(
        n_frames, n=n_responses, seed=draw_seed, frame_pool=frame_pool
    )
    # Extra order shuffle, decoupled from the generation RNG
    # (same convention as run_mode_count_experiment.py).
    order = list(range(len(responses)))
    random.Random(draw_seed ^ 0x5A5A5A5A).shuffle(order)
    responses = [responses[i] for i in order]
    labels = [FRAME_LABELS[frame_ids[i]] for i in order]
    return responses, labels


def estimate_context_tokens(responses: list[str], tokenizer: Any) -> int:
    """Worst-case token estimate for the concatenated instruct-format context."""
    labels = [chr(ord("A") + i) if i < 26 else f"R{i}" for i in range(len(responses))]
    parts = [TEMPLATE_PROMPT + "\n\n"]
    parts += [f"Response {lab}: {resp}\n\n" for lab, resp in zip(labels, responses)]
    return len(tokenizer.encode("".join(parts)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Template sentences: ICL diversity vs SentBERT baseline"
    )
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sentbert-model", default=DEFAULT_SENTBERT)
    parser.add_argument(
        "--sentbert-device",
        default=None,
        help="Device for the SentBERT encoder (default: same as --device, "
        "or cuda:0 when --device auto)",
    )
    parser.add_argument(
        "--frame-counts", type=int, nargs="+", default=[1, 2, 5, 10, 20]
    )
    parser.add_argument("--n-responses", type=int, default=20)
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Resolve dtype
    torch_dtype = None
    if args.torch_dtype is not None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if args.torch_dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype: {args.torch_dtype}. Use float16, bfloat16, or float32."
            )
        torch_dtype = dtype_map[args.torch_dtype]

    # Load base model theta
    use_device_map = args.device == "auto"
    print(f"Loading {args.base_model} (dtype={args.torch_dtype}, device={args.device})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    load_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if use_device_map:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if not use_device_map and args.device != "cpu":
        model = model.to(args.device)
    model.eval()

    # Load SentBERT encoder
    sentbert_device = args.sentbert_device
    if sentbert_device is None:
        sentbert_device = "cuda:0" if use_device_map else args.device
    print(f"Loading SentBERT encoder {args.sentbert_model} on {sentbert_device}")
    st_model = SentenceTransformer(args.sentbert_model, device=sentbert_device)

    conditions = build_conditions(args.frame_counts)

    # Upfront context-length validation (fail fast, never skip)
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx:
        worst = 0
        for condition, n_frames, frame_pool in conditions:
            for check_seed in [42, 137, 256, 0, 999]:
                responses, _ = draw_responses(
                    condition, n_frames, frame_pool, args.n_responses, check_seed
                )
                worst = max(worst, estimate_context_tokens(responses, tokenizer))
        print(f"Worst-case context estimate: {worst} tokens (model max {max_ctx})")
        if worst >= max_ctx:
            raise ValueError(
                f"Context estimate {worst} exceeds model max {max_ctx}; "
                "reduce --n-responses."
            )

    seed_rng = random.Random(args.seed)
    draw_seeds = [seed_rng.randint(0, 2**31) for _ in range(args.n_draws)]

    results: dict[str, Any] = {
        "experiment": "template_vs_sentbert",
        "base_model": args.base_model,
        "sentbert_model": args.sentbert_model,
        "format_mode": "instruct",
        "prompt": TEMPLATE_PROMPT,
        "frame_counts": args.frame_counts,
        "n_responses": args.n_responses,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "draw_seeds": draw_seeds,
        "runs": [],
    }

    pbar = tqdm(total=len(conditions) * args.n_draws, desc="condition/draws")
    for condition, n_frames, frame_pool in conditions:
        for draw_idx, draw_seed in enumerate(draw_seeds):
            t0 = time.time()
            responses, frame_labels = draw_responses(
                condition, n_frames, frame_pool, args.n_responses, draw_seed
            )

            metrics = compute_icl_diversity_metrics(
                model,
                tokenizer,
                TEMPLATE_PROMPT,
                responses,
                n_permutations=1,
                seed=draw_seed,
                batch_size=args.batch_size,
            )

            embeddings = st_model.encode(
                responses, convert_to_numpy=True, show_progress_bar=False
            )
            mean_sim = mean_pairwise_cosine_similarity(embeddings)

            results["runs"].append(
                {
                    "condition": condition,
                    "n_frames": n_frames,
                    "draw_idx": draw_idx,
                    "draw_seed": draw_seed,
                    "frame_labels": frame_labels,
                    "responses": responses,
                    "sentbert_mean_pairwise_cosine": mean_sim,
                    "sentbert_diversity": sentbert_diversity(embeddings),
                    "averaged_distinct_ngrams": averaged_distinct_ngrams(responses),
                    "elapsed_seconds": round(time.time() - t0, 2),
                    **metrics,
                }
            )
            pbar.set_postfix(
                cond=condition,
                draw=f"{draw_idx + 1}/{args.n_draws}",
                D=f"{metrics['diversity_score_D_C_an']:.3g}",
                sb=f"{mean_sim:.2f}",
            )
            pbar.update(1)
    pbar.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
