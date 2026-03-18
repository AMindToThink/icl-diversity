"""Run ICL diversity metrics across a temperature grid for all core scenarios.

Produces one JSON file per temperature in results/temperature_experiments/.

Usage:
    uv run python scripts/run_temperature_experiments.py
    uv run python scripts/run_temperature_experiments.py --device cuda --temperatures 0.5,1.0,2.0
    uv run python scripts/run_temperature_experiments.py --base-model Qwen/Qwen2.5-32B --device auto --torch-dtype float16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity import APIModel
from icl_diversity.scenarios import (
    NOISE_PROMPTS,
    INCOHERENT_PROMPTS,
    MULTI_MODE_PROMPTS_AND_RESPONSES,
    MULTI_MODE_PROMPT_LABELS,
    ONE_MODE_PROMPTS_AND_RESPONSES,
    ONE_MODE_PROMPT_LABELS,
    MIXED_PROMPTS_AND_RESPONSES,
    MIXED_PROMPT_LABELS,
    N_RESPONSES,
    generate_noise_responses,
    generate_multi_incoherent_responses,
)
from icl_diversity import compute_icl_diversity_metrics

SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "temperature_experiments"
DEFAULT_TEMPERATURES = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
CORE_SCENARIOS = ["pure_noise", "multi_incoherent", "multi_mode", "one_mode", "mixed"]


def build_scenario_items() -> list[tuple[str, str, str, list[str]]]:
    """Build (scenario_name, prompt_label, prompt_text, responses) for core scenarios."""
    items: list[tuple[str, str, str, list[str]]] = []

    for i, prompt in enumerate(NOISE_PROMPTS):
        responses = generate_noise_responses(n=N_RESPONSES, seed=i * 100)
        items.append(("pure_noise", f"Prompt {i}", prompt, responses))

    for i, prompt in enumerate(INCOHERENT_PROMPTS):
        responses = generate_multi_incoherent_responses(n=N_RESPONSES, seed=i * 100)
        items.append(("multi_incoherent", f"Prompt {i}", prompt, responses))

    for i, (prompt, responses) in enumerate(MULTI_MODE_PROMPTS_AND_RESPONSES):
        items.append(("multi_mode", MULTI_MODE_PROMPT_LABELS[i], prompt, responses[:N_RESPONSES]))

    for i, (prompt, responses) in enumerate(ONE_MODE_PROMPTS_AND_RESPONSES):
        items.append(("one_mode", ONE_MODE_PROMPT_LABELS[i], prompt, responses[:N_RESPONSES]))

    for i, (prompt, responses) in enumerate(MIXED_PROMPTS_AND_RESPONSES):
        items.append(("mixed", MIXED_PROMPT_LABELS[i], prompt, responses[:N_RESPONSES]))

    return items


def run_temperature(
    model: Any,
    tokenizer: Any,
    items: list[tuple[str, str, str, list[str]]],
    temperature: float,
    n_permutations: int,
    base_model: str,
) -> dict[str, Any]:
    """Run all scenario items at a single temperature."""
    from tqdm import tqdm

    result: dict[str, Any] = {
        "base_model": base_model,
        "temperature": temperature,
        "n_permutations": n_permutations,
        "n_responses": N_RESPONSES,
        "seed": SEED,
        "scenarios": {},
    }

    for scenario_name, prompt_label, prompt_text, responses in tqdm(
        items, desc=f"  T={temperature}", leave=False
    ):
        m = compute_icl_diversity_metrics(
            model,
            tokenizer,
            prompt_text,
            responses,
            n_permutations=n_permutations,
            seed=SEED,
            temperature=temperature,
        )
        m["prompt_label"] = prompt_label
        m["prompt_text"] = prompt_text
        result["scenarios"].setdefault(scenario_name, []).append(m)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ICL diversity metrics across a temperature grid"
    )
    parser.add_argument(
        "--base-model", default="gpt2", help="HuggingFace model ID (default: gpt2)"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device: cpu, cuda, or auto"
    )
    parser.add_argument(
        "--torch-dtype", default=None, help="Model dtype: float16, bfloat16, float32"
    )
    parser.add_argument(
        "--temperatures",
        default=",".join(str(t) for t in DEFAULT_TEMPERATURES),
        help=f"Comma-separated temperatures (default: {','.join(str(t) for t in DEFAULT_TEMPERATURES)})",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=100, help="Number of permutations (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--provider",
        choices=["local", "together", "fireworks"],
        default="local",
        help="Model provider (default: local)",
    )
    parser.add_argument("--api-key", default=None, help="API key for provider")
    parser.add_argument(
        "--offline", action="store_true", help="Set HF_HUB_OFFLINE=1"
    )
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Load model
    if args.provider != "local":
        from dotenv import load_dotenv
        load_dotenv()
        print(f"Using API model: {args.base_model} via {args.provider}")
        model = APIModel(
            model_name=args.base_model,
            provider=args.provider,
            api_key=args.api_key,
        )
        tokenizer = model.tokenizer
    else:
        torch_dtype = None
        if args.torch_dtype is not None:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(args.torch_dtype)
            if torch_dtype is None:
                print(f"Unknown dtype: {args.torch_dtype}")
                sys.exit(1)

        use_device_map = args.device == "auto"
        print(f"Loading {args.base_model} (dtype={args.torch_dtype}, device={args.device})...")
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

    # Build scenario items once
    items = build_scenario_items()
    print(f"Built {len(items)} scenario items across {len(set(s for s, *_ in items))} scenarios")

    # Run for each temperature
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for temp in temperatures:
        print(f"\n=== Temperature: {temp} ===")
        result = run_temperature(
            model, tokenizer, items, temp, args.n_permutations, args.base_model
        )
        out_path = args.output_dir / f"T_{temp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
