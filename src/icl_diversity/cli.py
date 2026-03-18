"""
CLI script for computing the ICL diversity metric on responses.

Reads a responses.jsonl file (with fields: prompt, prompt_idx, response_idx,
scale, response), groups by (scale, prompt_idx), and computes the ICL diversity
metric for each group using a base model theta.

Usage:
    calculate-icl-diversity \
        --input outputs/experiment/responses.jsonl \
        --base-model meta-llama/Llama-3.1-8B \
        --output outputs/experiment/icl_diversity.json

    python -m icl_diversity \
        --input outputs/experiment/responses.jsonl \
        --base-model gpt2 \
        --n-permutations 3 \
        --device cuda
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity import APIModel, compute_icl_diversity_metrics


def load_responses_with_prompts(
    path: Path,
) -> dict[float, dict[int, tuple[str, list[str]]]]:
    """Load responses.jsonl and group by scale -> prompt_idx -> (prompt_text, [responses]).

    Args:
        path: Path to the responses.jsonl file.

    Returns:
        Nested dict: scale -> prompt_idx -> (prompt_text, list of response texts).
    """
    data: dict[float, dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"prompt": None, "responses": {}})
    )

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            scale = float(record["scale"])
            prompt_idx = int(record["prompt_idx"])
            response_idx = int(record["response_idx"])
            prompt_text = record["prompt"]
            response_text = record["response"]

            entry = data[scale][prompt_idx]
            entry["prompt"] = prompt_text
            entry["responses"][response_idx] = response_text

    # Convert to final format with sorted responses
    result: dict[float, dict[int, tuple[str, list[str]]]] = {}
    for scale, prompts in sorted(data.items()):
        result[scale] = {}
        for prompt_idx, entry in sorted(prompts.items()):
            responses_sorted = [
                entry["responses"][i] for i in sorted(entry["responses"].keys())
            ]
            result[scale][prompt_idx] = (entry["prompt"], responses_sorted)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ICL diversity metric using a base model's "
        "progressive conditional surprise."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to responses.jsonl (fields: prompt, prompt_idx, response_idx, scale, response)",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HuggingFace model ID for theta (base model, e.g. meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: icl_diversity.json in input dir)",
    )
    parser.add_argument(
        "--n-permutations",
        default=1,
        type=int,
        help="Number of random orderings to average over (paper suggests 3-5)",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--provider",
        choices=["local", "together", "fireworks"],
        default="local",
        help="Model provider: local (HuggingFace), together, or fireworks (default: local)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for provider (default: uses TOGETHER_API_KEY or FIREWORKS_API_KEY env var)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5, only for API providers)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: cuda, cpu, or auto (default: auto, only for local provider)",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Model dtype: float16, bfloat16, float32 (default: float16, only for local provider)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for scaling base model logits before softmax (default: 1.0)",
    )
    args = parser.parse_args()

    # Resolve output path
    output_path = args.output or args.input.parent / "icl_diversity.json"

    if args.provider != "local":
        # API-based model
        from dotenv import load_dotenv

        load_dotenv()
        print(f"Using API model: {args.base_model} via {args.provider}")
        model = APIModel(
            model_name=args.base_model,
            provider=args.provider,
            api_key=args.api_key,
            max_concurrent_requests=args.max_concurrent,
        )
        tokenizer = model.tokenizer
    else:
        # Local HuggingFace model
        # Resolve device
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        # Resolve dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(args.torch_dtype)
        if torch_dtype is None:
            print(f"Unknown dtype: {args.torch_dtype}. Use float16, bfloat16, or float32.")
            sys.exit(1)

        # Load model
        print(
            f"Loading model: {args.base_model} (dtype={args.torch_dtype}, device={device})"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch_dtype,
            device_map=device if device == "auto" else None,
        )
        if device != "auto":
            model = model.to(device)
        model.eval()

    # Load responses
    print(f"Loading responses from: {args.input}")
    grouped = load_responses_with_prompts(args.input)

    # Compute metrics
    all_results: dict[str, Any] = {
        "base_model": args.base_model,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "scales": {},
    }

    for scale, prompts in grouped.items():
        print(f"\n=== Scale: {scale} ===")
        scale_results: list[dict[str, Any]] = []
        ds = []
        d_rates = []
        es = []
        e_rates = []
        cs = []
        sigmas = []

        for prompt_idx, (prompt_text, responses) in sorted(prompts.items()):
            print(
                f"  Prompt {prompt_idx}: {len(responses)} responses, "
                f"computing ICL diversity..."
            )

            metrics = compute_icl_diversity_metrics(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                responses=responses,
                n_permutations=args.n_permutations,
                seed=args.seed,
                temperature=args.temperature,
            )

            per_prompt_result = {
                "prompt_idx": prompt_idx,
                "prompt": prompt_text,
                "n_responses": len(responses),
                **metrics,
            }
            scale_results.append(per_prompt_result)

            ds.append(metrics["diversity_score_D"])
            d_rates.append(metrics["diversity_score_D_rate"])
            es.append(metrics["excess_entropy_E"])
            e_rates.append(metrics["excess_entropy_E_rate"])
            cs.append(metrics["coherence_C"])
            sigmas.append(metrics["coherence_spread_sigma"])

            print(
                f"    D={metrics['diversity_score_D']:.4f}, "
                f"D_rate={metrics['diversity_score_D_rate']:.4f}, "
                f"E={metrics['excess_entropy_E']:.2f}, "
                f"E_rate={metrics['excess_entropy_E_rate']:.4f}, "
                f"C={metrics['coherence_C']:.4f}, "
                f"monotone={metrics['is_monotone']}"
            )

        aggregate = {
            "mean_D": float(np.mean(ds)) if ds else 0.0,
            "mean_D_rate": float(np.mean(d_rates)) if d_rates else 0.0,
            "mean_E": float(np.mean(es)) if es else 0.0,
            "mean_E_rate": float(np.mean(e_rates)) if e_rates else 0.0,
            "mean_C": float(np.mean(cs)) if cs else 0.0,
            "mean_sigma": float(np.mean(sigmas)) if sigmas else 0.0,
        }

        all_results["scales"][str(scale)] = {
            "per_prompt": scale_results,
            "aggregate": aggregate,
        }

        print(
            f"  Aggregate: mean_D={aggregate['mean_D']:.4f}, "
            f"mean_E={aggregate['mean_E']:.2f}, "
            f"mean_E_rate={aggregate['mean_E_rate']:.4f}, "
            f"mean_C={aggregate['mean_C']:.4f}"
        )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
