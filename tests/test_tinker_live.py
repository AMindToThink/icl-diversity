"""Live integration tests for TinkerModel against Tinker API.

These tests call the real Tinker API and cross-validate against a locally-loaded
Llama-3.2-1B model. They only run when TINKER_API_KEY is set (via .env or env var).

Run with:
    uv run pytest tests/test_tinker_live.py -v -s

Note on tolerances: Tinker's inference engine uses custom batch-invariant
matmul kernels (see https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
that produce slightly different numerical results from HuggingFace transformers'
standard matmul (cuBLAS on GPU, MKL/OpenBLAS on CPU). Empirically this gives
~0.02 bits mean per-token diff with occasional outliers up to ~0.35 bits at
high-surprise positions. This is NOT a bug in our code — the shift logic,
nats-to-bits conversion, and padding are all verified exact.
"""

from __future__ import annotations

import os

import pytest
import torch
from dotenv import load_dotenv

load_dotenv()

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("TINKER_API_KEY"),
    reason="TINKER_API_KEY not set — skipping live Tinker tests",
)

MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Test strings of varying complexity
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1492, Columbus sailed the ocean blue.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
]

# Tolerances calibrated from empirical comparison (see scripts/compare_tinker_local.py).
# Root cause: different matmul kernels in Tinker's inference engine vs HuggingFace.
PER_TOKEN_ATOL = 0.5  # bits — outliers up to ~0.35 observed at high-surprise positions
METRIC_REL_TOL = 0.05  # 5% relative tolerance on aggregate metrics


@pytest.fixture(scope="module")
def tinker_sampling_client():
    """Create a Tinker SamplingClient for direct API calls."""
    import tinker

    service_client = tinker.ServiceClient()
    return service_client.create_sampling_client(base_model=MODEL_NAME)


@pytest.fixture(scope="module")
def tinker_model():
    """Create a TinkerModel instance."""
    from icl_diversity.tinker_model import TinkerModel

    return TinkerModel(model_name=MODEL_NAME)


@pytest.fixture(scope="module")
def local_model_and_tokenizer():
    """Load Llama-3.2-1B locally for cross-validation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# API format verification
# ---------------------------------------------------------------------------


class TestComputeLogprobsFormat:
    def test_returns_list_of_float_or_none(
        self, tinker_sampling_client
    ) -> None:
        """compute_logprobs returns list[float | None] with correct length."""
        import tinker

        tokenizer = tinker_sampling_client.get_tokenizer()
        text = "Hello, world!"
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        prompt = tinker.types.ModelInput.from_ints(tokens=token_ids)
        raw = tinker_sampling_client.compute_logprobs(prompt).result()

        assert len(raw) == len(token_ids), (
            f"Expected {len(token_ids)} logprobs, got {len(raw)}"
        )

        # First element should be None (no conditioning for first token)
        assert raw[0] is None, f"Expected None for first token, got {raw[0]}"

        # Remaining elements should be negative floats (log-probabilities)
        for i, lp in enumerate(raw[1:], start=1):
            assert isinstance(lp, float), (
                f"Position {i}: expected float, got {type(lp)}"
            )
            assert lp <= 0.0, (
                f"Position {i}: logprob should be <= 0, got {lp}"
            )

    def test_compute_logprobs_vs_sample_prompt_logprobs(
        self, tinker_sampling_client
    ) -> None:
        """compute_logprobs and sample(include_prompt_logprobs=True) agree."""
        import tinker

        tokenizer = tinker_sampling_client.get_tokenizer()
        text = "The cat sat on the mat."
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt = tinker.types.ModelInput.from_ints(tokens=token_ids)

        # Method 1: compute_logprobs
        logprobs_direct = tinker_sampling_client.compute_logprobs(prompt).result()

        # Method 2: sample with prompt_logprobs
        sample_response = tinker_sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
        ).result()

        logprobs_via_sample = sample_response.prompt_logprobs
        assert logprobs_via_sample is not None, (
            "sample() did not return prompt_logprobs"
        )

        assert len(logprobs_direct) == len(logprobs_via_sample), (
            f"Length mismatch: compute_logprobs={len(logprobs_direct)}, "
            f"sample={len(logprobs_via_sample)}"
        )

        for i in range(1, len(logprobs_direct)):
            assert logprobs_direct[i] == pytest.approx(
                logprobs_via_sample[i], abs=1e-6
            ), f"Position {i}: {logprobs_direct[i]} != {logprobs_via_sample[i]}"


# ---------------------------------------------------------------------------
# Cross-validation against local model
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_tinker_logprobs_match_local_model(
        self,
        tinker_model,
        local_model_and_tokenizer: tuple,
    ) -> None:
        """TinkerModel.score_sequences matches local _forward_log_probs.

        Tinker uses custom batch-invariant matmul kernels that produce slightly
        different results from HuggingFace's standard kernels. Tolerances
        account for this; see module docstring.
        """
        from icl_diversity.core import _forward_log_probs

        local_model, local_tokenizer = local_model_and_tokenizer

        for text in TEST_TEXTS:
            token_ids = local_tokenizer.encode(text, add_special_tokens=False)
            input_ids = torch.tensor([token_ids])

            tinker_logprobs = tinker_model.score_sequences(input_ids)

            local_logprobs = _forward_log_probs(
                [local_model], input_ids, temperature=1.0
            )

            torch.testing.assert_close(
                tinker_logprobs,
                local_logprobs,
                atol=PER_TOKEN_ATOL,
                rtol=0.1,
                msg=f"Logprob mismatch for text: {text!r}",
            )

    def test_metrics_match_local_model(
        self,
        tinker_model,
        local_model_and_tokenizer: tuple,
    ) -> None:
        """Full pipeline metrics match between Tinker and local model.

        Tolerances account for per-token diffs accumulating across the
        progressive surprise curve.
        """
        from icl_diversity import compute_icl_diversity_metrics

        local_model, local_tokenizer = local_model_and_tokenizer

        prompt = "What is the capital of France?"
        responses = [
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is the city of Paris, located in the north.",
        ]

        tinker_metrics = compute_icl_diversity_metrics(
            model=tinker_model,
            tokenizer=None,
            prompt=prompt,
            responses=responses,
            n_permutations=1,
            seed=42,
        )

        local_metrics = compute_icl_diversity_metrics(
            model=local_model,
            tokenizer=local_tokenizer,
            prompt=prompt,
            responses=responses,
            n_permutations=1,
            seed=42,
        )

        # Compare key metrics with relative tolerance
        for key in [
            "excess_entropy_E",
            "excess_entropy_E_rate",
            "coherence_C",
            "diversity_score_D",
            "diversity_score_D_rate",
        ]:
            assert tinker_metrics[key] == pytest.approx(
                local_metrics[key], rel=METRIC_REL_TOL
            ), f"{key}: tinker={tinker_metrics[key]}, local={local_metrics[key]}"

        # Compare a_k curves
        for i, (t_ak, l_ak) in enumerate(
            zip(tinker_metrics["a_k_curve"], local_metrics["a_k_curve"])
        ):
            assert t_ak == pytest.approx(l_ak, rel=METRIC_REL_TOL), (
                f"a_k[{i}]: tinker={t_ak}, local={l_ak}"
            )
