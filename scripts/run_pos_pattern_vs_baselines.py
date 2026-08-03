"""POS-pattern experiment: structural redundancy with zero lexical overlap.

Follow-up to scripts/run_template_vs_sentbert.py. There, repeated frames
shared literal function words, so averaged distinct-n also detected the
redundancy. Here every word of every sentence is freshly sampled; sentences
share only their part-of-speech pattern (canonical:
"{Noun} {verb} {preposition} {noun} {preposition} {noun}."). With no shared
lexical material, distinct-n and SentBERT should both be flat in the number
of patterns m, while D = C * a_n (``diversity_score_D_C_an``) should rise
with m if the base model learns POS patterns in-context.

Conditions (no paraphrase anchor; the claim is within-sweep, see
reports/TEMPLATE_VS_SENTBERT.md for the cross-metric anchoring experiment):

- ``canonical``:   every response uses pattern 0 (N Vi P N P N).
- ``patterns_m``:  responses drawn evenly from m randomly chosen patterns
                   (m in --pattern-counts).
- ``scrambled``:   (--include-scrambled) composition-matched control: each
                   sentence samples the canonical class multiset (3 N, 1 Vi,
                   2 P) then shuffles its own word order. Same vocabulary
                   statistics, no consistent structure.

Each condition runs --n-draws fully independent draws (fresh words, fresh
pattern subset, fresh order); n_permutations=1 per draw, statistical power
from the independent draws.

Usage:
    uv run python scripts/run_pos_pattern_vs_baselines.py \
        --base-model Qwen/Qwen2.5-3B --device cuda:1 --sentbert-device cuda:1 \
        --torch-dtype bfloat16 --batch-size 8 \
        --output results/pos_pattern/qwen2.5-3b.json
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
from icl_diversity.pos_pattern_scenarios import (
    POS_PATTERN_LABELS,
    POS_PATTERN_PROMPT,
    generate_pos_pattern_responses,
    generate_scrambled_canonical_responses,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "pos_pattern"
    / "qwen2.5-3b.json"
)
# Tevet & Berant's SentBERT model, as in run_template_vs_sentbert.py.
DEFAULT_SENTBERT = "sentence-transformers/bert-large-nli-stsb-mean-tokens"


def build_conditions(
    pattern_counts: list[int],
    include_scrambled: bool,
) -> list[tuple[str, int | None, list[int] | None]]:
    """(condition_name, n_patterns, pattern_pool) triples."""
    conditions: list[tuple[str, int | None, list[int] | None]] = [("canonical", 1, [0])]
    conditions += [(f"patterns_{m}", m, None) for m in pattern_counts]
    if include_scrambled:
        conditions.append(("scrambled", None, None))
    return conditions


def draw_responses(
    condition: str,
    n_patterns: int | None,
    pattern_pool: list[int] | None,
    n_responses: int,
    draw_seed: int,
) -> tuple[list[str], list[str] | None]:
    """Responses (shuffled) and, for pattern conditions, their pattern labels."""
    if condition == "scrambled":
        # Already order-free; still shuffle for symmetry with other conditions.
        responses = generate_scrambled_canonical_responses(
            n=n_responses, seed=draw_seed
        )
        random.Random(draw_seed ^ 0x5A5A5A5A).shuffle(responses)
        return responses, None

    assert n_patterns is not None
    responses, pattern_ids = generate_pos_pattern_responses(
        n_patterns, n=n_responses, seed=draw_seed, pattern_pool=pattern_pool
    )
    # Extra order shuffle, decoupled from the generation RNG
    # (same convention as run_template_vs_sentbert.py).
    order = list(range(len(responses)))
    random.Random(draw_seed ^ 0x5A5A5A5A).shuffle(order)
    responses = [responses[i] for i in order]
    labels = [POS_PATTERN_LABELS[pattern_ids[i]] for i in order]
    return responses, labels


def estimate_context_tokens(responses: list[str], tokenizer: Any) -> int:
    """Worst-case token estimate for the concatenated instruct-format context."""
    labels = [chr(ord("A") + i) if i < 26 else f"R{i}" for i in range(len(responses))]
    parts = [POS_PATTERN_PROMPT + "\n\n"]
    parts += [f"Response {lab}: {resp}\n\n" for lab, resp in zip(labels, responses)]
    return len(tokenizer.encode("".join(parts)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POS patterns: ICL diversity vs SentBERT and distinct-n"
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sentbert-model", default=DEFAULT_SENTBERT)
    parser.add_argument(
        "--sentbert-device",
        default=None,
        help="Device for the SentBERT encoder (default: same as --device, "
        "or cuda:0 when --device auto)",
    )
    parser.add_argument(
        "--pattern-counts",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 12],
        help="Sweep sizes m; pass with no values for a sweep-free run",
    )
    parser.add_argument(
        "--include-scrambled",
        action="store_true",
        help="Add the order-scrambled no-structure control condition "
        "(canonical class multiset, per-sentence random word order)",
    )
    parser.add_argument("--n-responses", type=int, default=40)
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

    conditions = build_conditions(args.pattern_counts, args.include_scrambled)

    # Upfront context-length validation (fail fast, never skip)
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx:
        worst = 0
        for condition, n_patterns, pattern_pool in conditions:
            for check_seed in [42, 137, 256, 0, 999]:
                responses, _ = draw_responses(
                    condition, n_patterns, pattern_pool, args.n_responses, check_seed
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
        "experiment": "pos_pattern_vs_baselines",
        "base_model": args.base_model,
        "sentbert_model": args.sentbert_model,
        "format_mode": "instruct",
        "prompt": POS_PATTERN_PROMPT,
        "pattern_counts": args.pattern_counts,
        "n_responses": args.n_responses,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "draw_seeds": draw_seeds,
        "runs": [],
    }

    pbar = tqdm(total=len(conditions) * args.n_draws, desc="condition/draws")
    for condition, n_patterns, pattern_pool in conditions:
        for draw_idx, draw_seed in enumerate(draw_seeds):
            t0 = time.time()
            responses, pattern_labels = draw_responses(
                condition, n_patterns, pattern_pool, args.n_responses, draw_seed
            )

            metrics = compute_icl_diversity_metrics(
                model,
                tokenizer,
                POS_PATTERN_PROMPT,
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
                    "n_patterns": n_patterns,
                    "draw_idx": draw_idx,
                    "draw_seed": draw_seed,
                    "pattern_labels": pattern_labels,
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
                dn=f"{results['runs'][-1]['averaged_distinct_ngrams']:.3f}",
            )
            pbar.update(1)
    pbar.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
