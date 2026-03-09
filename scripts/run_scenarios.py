"""Compute ICL diversity metrics for all eight validation scenarios and save to JSON.

Uses the same scenario data, seed, and parameters as test_icl_diversity_scenarios.py.
Output: results/scenario_metrics.json

Usage:
    uv run scripts/run_scenarios.py
    uv run scripts/run_scenarios.py --base-model Qwen/Qwen2.5-32B --device auto --torch-dtype float16
    uv run scripts/run_scenarios.py --output results/my_run.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity import compute_icl_diversity_metrics
from icl_diversity.scenarios import (
    NOISE_PROMPTS,
    INCOHERENT_PROMPTS,
    MULTI_MODE_PROMPTS_AND_RESPONSES,
    MULTI_MODE_PROMPT_LABELS,
    ONE_MODE_PROMPTS_AND_RESPONSES,
    ONE_MODE_PROMPT_LABELS,
    MIXED_PROMPTS_AND_RESPONSES,
    MIXED_PROMPT_LABELS,
    HIGH_DIVERSITY_PROMPTS_AND_RESPONSES,
    HIGH_DIVERSITY_PROMPT_LABELS,
    HIGH_DIVERSITY_N_RESPONSES,
    OPEN_CREATIVE_PROMPTS_AND_RESPONSES,
    OPEN_CREATIVE_PROMPT_LABELS,
    OPEN_CREATIVE_N_RESPONSES,
    PROBLEM_SOLVING_PROMPTS_AND_RESPONSES,
    PROBLEM_SOLVING_PROMPT_LABELS,
    PROBLEM_SOLVING_N_RESPONSES,
    N_RESPONSES,
    N_PERMUTATIONS,
    generate_noise_responses,
    generate_multi_incoherent_responses,
)

SEED = 42
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "results" / "scenario_metrics.json"
)


def compute_all_scenarios(
    model: torch.nn.Module,
    tokenizer: Any,
    base_model: str = "gpt2",
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, Any]:
    """Compute metrics for all eight scenarios."""
    result: dict[str, Any] = {
        "base_model": base_model,
        "n_permutations": n_permutations,
        "n_responses": N_RESPONSES,
        "seed": SEED,
        "scenarios": {},
    }

    # Build list of (scenario_name, prompt_label, prompt_text, responses)
    all_items: list[tuple[str, str, str, list[str]]] = []

    for i, prompt in enumerate(NOISE_PROMPTS):
        responses = generate_noise_responses(n=N_RESPONSES, seed=i * 100)
        all_items.append(("pure_noise", f"Prompt {i}", prompt, responses))

    for i, prompt in enumerate(INCOHERENT_PROMPTS):
        responses = generate_multi_incoherent_responses(n=N_RESPONSES, seed=i * 100)
        all_items.append(("multi_incoherent", f"Prompt {i}", prompt, responses))

    for i, (prompt, responses) in enumerate(MULTI_MODE_PROMPTS_AND_RESPONSES):
        all_items.append(
            ("multi_mode", MULTI_MODE_PROMPT_LABELS[i], prompt, responses[:N_RESPONSES])
        )

    for i, (prompt, responses) in enumerate(ONE_MODE_PROMPTS_AND_RESPONSES):
        all_items.append(
            ("one_mode", ONE_MODE_PROMPT_LABELS[i], prompt, responses[:N_RESPONSES])
        )

    for i, (prompt, responses) in enumerate(MIXED_PROMPTS_AND_RESPONSES):
        all_items.append(
            ("mixed", MIXED_PROMPT_LABELS[i], prompt, responses[:N_RESPONSES])
        )

    for i, (prompt, responses) in enumerate(HIGH_DIVERSITY_PROMPTS_AND_RESPONSES):
        all_items.append(
            (
                "high_diversity",
                HIGH_DIVERSITY_PROMPT_LABELS[i],
                prompt,
                responses[:HIGH_DIVERSITY_N_RESPONSES],
            )
        )

    for i, (prompt, responses) in enumerate(OPEN_CREATIVE_PROMPTS_AND_RESPONSES):
        all_items.append(
            (
                "open_creative",
                OPEN_CREATIVE_PROMPT_LABELS[i],
                prompt,
                responses[:OPEN_CREATIVE_N_RESPONSES],
            )
        )

    for i, (prompt, responses) in enumerate(PROBLEM_SOLVING_PROMPTS_AND_RESPONSES):
        all_items.append(
            (
                "problem_solving",
                PROBLEM_SOLVING_PROMPT_LABELS[i],
                prompt,
                responses[:PROBLEM_SOLVING_N_RESPONSES],
            )
        )

    # Compute metrics with progress bar
    for scenario_name, prompt_label, prompt_text, responses in tqdm(
        all_items, desc="scenarios/prompts"
    ):
        m = compute_icl_diversity_metrics(
            model,
            tokenizer,
            prompt_text,
            responses,
            n_permutations=n_permutations,
            seed=SEED,
        )
        m["prompt_label"] = prompt_label
        m["prompt_text"] = prompt_text
        result["scenarios"].setdefault(scenario_name, []).append(m)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ICL diversity validation scenarios"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path"
    )
    parser.add_argument(
        "--base-model", default="gpt2", help="HuggingFace model ID (default: gpt2)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Device: cpu, cuda, cuda:0, or auto (multi-GPU via device_map="auto")',
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Model dtype: float16, bfloat16, float32 (default: model's native dtype)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=f"Number of permutations to average over (default: {N_PERMUTATIONS})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 to prevent downloads",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Resolve dtype
    torch_dtype = None
    if args.torch_dtype is not None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(args.torch_dtype)
        if torch_dtype is None:
            print(
                f"Unknown dtype: {args.torch_dtype}. Use float16, bfloat16, or float32."
            )
            sys.exit(1)

    # Load model
    use_device_map = args.device == "auto"
    print(
        f"Loading {args.base_model} (dtype={args.torch_dtype}, device={args.device})..."
    )

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

    print("Computing metrics for all scenarios...")
    print(f"n_permutations={args.n_permutations}")
    results = compute_all_scenarios(
        model,
        tokenizer,
        base_model=args.base_model,
        n_permutations=args.n_permutations,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
