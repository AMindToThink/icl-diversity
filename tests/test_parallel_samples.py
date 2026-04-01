"""Tests verifying that parallel sample processing produces identical results.

The key invariant: processing N samples with parallel_samples=K must produce
the exact same metrics as processing them sequentially (parallel_samples=1).

Includes:
- Live Tinker test (skipped without TINKER_API_KEY): verifies concurrent API
  calls don't interfere with each other.
- Auto-detection test (mocked): TinkerModel → 8, local → 1.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip live tests if no API key
needs_tinker = pytest.mark.skipif(
    not os.environ.get("TINKER_API_KEY"),
    reason="TINKER_API_KEY not set",
)

MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Small test data — 4 samples, cheap to run
TEST_SAMPLES = [
    {
        "prompt": "What color is the sky?",
        "responses": [
            "The sky is blue.",
            "It is blue during the day.",
            "Blue, typically.",
            "The sky appears blue due to Rayleigh scattering.",
            "It depends on the time of day.",
        ],
    },
    {
        "prompt": "Name a fruit.",
        "responses": [
            "Apple.",
            "Banana.",
            "Orange.",
            "Mango.",
            "Strawberry.",
        ],
    },
    {
        "prompt": "What is 2+2?",
        "responses": [
            "4.",
            "The answer is 4.",
            "Two plus two equals four.",
            "It's 4.",
            "2+2=4.",
        ],
    },
    {
        "prompt": "Tell me about rain.",
        "responses": [
            "Rain is water falling from clouds.",
            "Rain forms when water vapor condenses.",
            "It rains when clouds become saturated.",
            "Rain is precipitation in liquid form.",
            "Rainfall is part of the water cycle.",
        ],
    },
]


@needs_tinker
class TestParallelMatchesSequential:
    def test_parallel_matches_sequential(self) -> None:
        """Concurrent sample processing must produce identical results to sequential."""
        from icl_diversity import compute_icl_diversity_metrics
        from icl_diversity.tinker_model import TinkerModel

        model = TinkerModel(model_name=MODEL_NAME)

        def compute_one(sample: dict) -> dict:
            return compute_icl_diversity_metrics(
                model=model,
                tokenizer=None,
                prompt=sample["prompt"],
                responses=sample["responses"],
                n_permutations=3,
                seed=42,
            )

        # Sequential
        sequential_results = [compute_one(s) for s in TEST_SAMPLES]

        # Parallel (4 workers)
        parallel_results: list[dict | None] = [None] * len(TEST_SAMPLES)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(compute_one, s): i for i, s in enumerate(TEST_SAMPLES)
            }
            for future in as_completed(futures):
                idx = futures[future]
                parallel_results[idx] = future.result()

        # Compare every metric. Use 1e-4 tolerance in case concurrent
        # requests hit different Tinker server instances with slightly
        # different numerical behavior.
        keys_to_compare = [
            "excess_entropy_E",
            "excess_entropy_E_rate",
            "coherence_C",
            "diversity_score_D",
            "diversity_score_D_rate",
            "coherence_spread_sigma",
        ]

        for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
            assert par is not None, f"Sample {i} missing from parallel results"
            for key in keys_to_compare:
                assert seq[key] == pytest.approx(par[key], abs=1e-4), (
                    f"Sample {i}, {key}: sequential={seq[key]}, parallel={par[key]}"
                )
            assert seq["is_monotone"] == par["is_monotone"], (
                f"Sample {i}, is_monotone: sequential={seq['is_monotone']}, "
                f"parallel={par['is_monotone']}"
            )

            # Compare a_k curves element by element
            for j, (s_ak, p_ak) in enumerate(
                zip(seq["a_k_curve"], par["a_k_curve"])
            ):
                assert s_ak == pytest.approx(p_ak, abs=1e-4), (
                    f"Sample {i}, a_k[{j}]: sequential={s_ak}, parallel={p_ak}"
                )


class TestParallelSamplesAutoDetection:
    def _import_resolve(self):
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        mod = importlib.import_module("compute_icl_metrics_for_tevet")
        sys.path.pop(0)
        return mod.resolve_parallel_samples

    def test_tinker_model_gets_parallel(self) -> None:
        """Auto-detection should give parallel_samples > 1 for TinkerModel."""
        resolve = self._import_resolve()

        class TinkerModel:
            pass

        assert resolve("auto", TinkerModel()) == 8

    def test_local_model_gets_sequential(self) -> None:
        """Auto-detection should give parallel_samples=1 for local models."""
        resolve = self._import_resolve()

        class LlamaForCausalLM:
            pass

        assert resolve("auto", LlamaForCausalLM()) == 1

    def test_explicit_override(self) -> None:
        """Explicit int should override auto-detection."""
        from unittest.mock import MagicMock

        resolve = self._import_resolve()
        mock = MagicMock()
        assert resolve("4", mock) == 4
        assert resolve("1", mock) == 1
