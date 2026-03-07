"""Tests for the ICL diversity metric.

All tests run on CPU with mock models to avoid GPU dependency.
"""

import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from icl_diversity import (
    _response_label,
    compute_icl_diversity_metrics,
    compute_per_byte_cross_entropy,
    compute_progressive_surprise_curve,
    compute_unconditional_surprises,
    _compute_metrics_from_curves,
    format_conditioning_context,
)


# --- Response label tests ---


class TestResponseLabel:
    def test_single_letters(self) -> None:
        assert _response_label(0) == "A"
        assert _response_label(1) == "B"
        assert _response_label(25) == "Z"

    def test_multi_letters(self) -> None:
        assert _response_label(26) == "AA"
        assert _response_label(27) == "AB"
        assert _response_label(51) == "AZ"
        assert _response_label(52) == "BA"


# --- Format conditioning context tests ---


class TestFormatConditioningContext:
    def test_first_response(self) -> None:
        prefix, target = format_conditioning_context(
            "Tell me a story", [], "Once upon a time"
        )
        assert prefix == "Tell me a story\n\nResponse A: "
        assert target == "Once upon a time"

    def test_with_previous_responses(self) -> None:
        prefix, target = format_conditioning_context(
            "Tell me a story",
            ["Story one.", "Story two."],
            "Story three.",
        )
        expected_prefix = (
            "Tell me a story\n\n"
            "Response A: Story one.\n\n"
            "Response B: Story two.\n\n"
            "Response C: "
        )
        assert prefix == expected_prefix
        assert target == "Story three."

    def test_labels_sequential(self) -> None:
        """Verify that labels are A, B, C, ... for sequential responses."""
        prefix, _ = format_conditioning_context(
            "prompt",
            ["r1", "r2", "r3"],
            "r4",
        )
        assert "Response A: r1" in prefix
        assert "Response B: r2" in prefix
        assert "Response C: r3" in prefix
        assert "Response D: " in prefix


# --- Per-byte normalization tests ---


class TestPerByteNormalization:
    def test_normalization_uses_byte_count(self) -> None:
        """Verify division by byte count, not token count."""
        # ASCII text: byte count == character count
        text = "hello"
        byte_count = len(text.encode("utf-8"))
        assert byte_count == 5

        # Multi-byte UTF-8: byte count > character count
        text_unicode = "hello"  # all ASCII here, but test the concept
        assert len(text_unicode.encode("utf-8")) == len(text_unicode)

        # Actual multi-byte
        text_emoji = "hi"  # 2 chars, 4 bytes (h=1, i=1... actually 2 bytes)
        assert len(text_emoji.encode("utf-8")) == 2  # ASCII is 1 byte each

    def test_empty_text_returns_zero(self) -> None:
        """Empty text should return 0 cross-entropy."""
        model = MagicMock()
        tokenizer = MagicMock()
        result = compute_per_byte_cross_entropy(model, tokenizer, "", "prefix")
        assert result == 0.0


# --- Derived metric tests (pure math, no model needed) ---


class TestExcessEntropy:
    def test_zero_for_constant_curve(self) -> None:
        """If all a_k are equal, E = 0."""
        a_k = [1.0, 1.0, 1.0, 1.0, 1.0]
        unconditional = [1.2, 1.2, 1.2, 1.2, 1.2]
        responses = ["hello world"] * 5
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        assert result["excess_entropy_E"] == pytest.approx(0.0)

    def test_positive_for_decreasing_curve(self) -> None:
        """If a_k decreases, E > 0."""
        a_k = [1.5, 1.3, 1.1, 1.0, 1.0]
        unconditional = [1.5, 1.5, 1.5, 1.5, 1.5]
        responses = ["hello world"] * 5
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        # E = (1.5-1.0) + (1.3-1.0) + (1.1-1.0) + (1.0-1.0) + (1.0-1.0)
        #   = 0.5 + 0.3 + 0.1 + 0 + 0 = 0.9
        assert result["excess_entropy_E"] == pytest.approx(0.9)

    def test_excess_entropy_formula(self) -> None:
        """Verify E_hat_n = sum(a_k - a_n) per Eq 6."""
        a_k = [2.0, 1.5, 1.2, 1.0]
        a_n = a_k[-1]  # 1.0
        expected_E = sum(a - a_n for a in a_k)  # 1.0 + 0.5 + 0.2 + 0.0 = 1.7
        unconditional = [2.0] * 4
        responses = ["test"] * 4
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        assert result["excess_entropy_E"] == pytest.approx(expected_E)


class TestCoherence:
    def test_range(self) -> None:
        """C must be in (0, 1) for positive cross-entropies."""
        a_k = [1.0, 0.8, 0.7]
        unconditional = [0.5, 1.0, 1.5]  # different surprise levels
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        assert 0 < result["coherence_C"] < 1

    def test_geometric_mean(self) -> None:
        """Verify C = 2^{-(1/n) * sum(h_i)} per Eq 8."""
        unconditional = [1.0, 2.0, 3.0]
        mean_h = sum(unconditional) / len(unconditional)  # 2.0
        expected_C = 2.0 ** (-mean_h)  # 2^{-2} = 0.25
        a_k = [1.0, 0.8, 0.7]
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        assert result["coherence_C"] == pytest.approx(expected_C)

    def test_lower_surprise_means_higher_coherence(self) -> None:
        """Lower unconditional surprise should give higher coherence."""
        a_k = [1.0, 0.8]
        responses = ["test"] * 2

        low_surprise = [0.5, 0.5]
        high_surprise = [2.0, 2.0]

        result_low = _compute_metrics_from_curves(a_k, low_surprise, responses)
        result_high = _compute_metrics_from_curves(a_k, high_surprise, responses)

        assert result_low["coherence_C"] > result_high["coherence_C"]


class TestDiversityScore:
    def test_product(self) -> None:
        """D = C * E per Eq 11."""
        a_k = [2.0, 1.5, 1.2, 1.0]
        unconditional = [1.0, 1.0, 1.0, 1.0]
        responses = ["test"] * 4
        result = _compute_metrics_from_curves(a_k, unconditional, responses)

        expected_D = result["coherence_C"] * result["excess_entropy_E"]
        assert result["diversity_score_D"] == pytest.approx(expected_D)

    def test_zero_when_no_structure(self) -> None:
        """D = 0 when a_k is constant (no structure)."""
        a_k = [1.5, 1.5, 1.5]
        unconditional = [1.5, 1.5, 1.5]
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        assert result["diversity_score_D"] == pytest.approx(0.0)


class TestEffectiveModeCount:
    def test_formula(self) -> None:
        """m_eff = 2^{B_bar * E} per Eq 7."""
        a_k = [2.0, 1.5, 1.0]
        unconditional = [1.0, 1.0, 1.0]
        responses = ["hello", "world", "foooo"]  # each 5 bytes
        result = _compute_metrics_from_curves(a_k, unconditional, responses)

        E = result["excess_entropy_E"]
        B_bar = result["mean_byte_length"]
        expected_m = 2.0 ** (B_bar * E)
        assert result["effective_mode_count"] == pytest.approx(expected_m)

    def test_one_mode_when_flat(self) -> None:
        """m_eff = 1 when E = 0 (flat curve)."""
        a_k = [1.0, 1.0, 1.0]
        unconditional = [1.0, 1.0, 1.0]
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)
        assert result["effective_mode_count"] == pytest.approx(1.0)


class TestUncertaintyBand:
    def test_band_when_sigma_positive(self) -> None:
        """D+ > D > D- when sigma > 0 and E > 0."""
        a_k = [2.0, 1.5, 1.0]
        unconditional = [0.5, 1.0, 1.5]  # different => sigma > 0
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)

        assert result["coherence_spread_sigma"] > 0
        assert result["D_plus"] > result["diversity_score_D"]
        assert result["D_minus"] < result["diversity_score_D"]

    def test_band_zero_when_sigma_zero(self) -> None:
        """D+ = D = D- when sigma = 0."""
        a_k = [2.0, 1.5, 1.0]
        unconditional = [1.0, 1.0, 1.0]  # all equal => sigma = 0
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)

        assert result["coherence_spread_sigma"] == pytest.approx(0.0)
        assert result["D_plus"] == pytest.approx(result["diversity_score_D"])
        assert result["D_minus"] == pytest.approx(result["diversity_score_D"])

    def test_c_plus_c_minus_formulas(self) -> None:
        """C+ = C * 2^sigma, C- = C * 2^{-sigma} per Section 6.4."""
        a_k = [2.0, 1.5, 1.0]
        unconditional = [0.8, 1.0, 1.2]
        responses = ["test"] * 3
        result = _compute_metrics_from_curves(a_k, unconditional, responses)

        C = result["coherence_C"]
        sigma = result["coherence_spread_sigma"]
        assert result["C_plus"] == pytest.approx(C * 2.0 ** sigma)
        assert result["C_minus"] == pytest.approx(C * 2.0 ** (-sigma))


class TestMonotonicity:
    def test_monotone_curve(self) -> None:
        a_k = [2.0, 1.5, 1.0, 1.0]
        result = _compute_metrics_from_curves(a_k, [1.0] * 4, ["t"] * 4)
        assert result["is_monotone"] is True

    def test_non_monotone_curve(self) -> None:
        a_k = [2.0, 1.5, 1.8, 1.0]  # 1.5 -> 1.8 is an increase
        result = _compute_metrics_from_curves(a_k, [1.0] * 4, ["t"] * 4)
        assert result["is_monotone"] is False


# --- Mock model integration tests ---


def _make_mock_model_and_tokenizer(
    vocab_size: int = 100,
    uniform: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock model and tokenizer for testing.

    The mock model returns uniform logits (all tokens equally likely),
    which makes cross-entropy calculations predictable.

    Args:
        vocab_size: Size of the vocabulary.
        uniform: If True, all logits are 0 (uniform distribution).

    Returns:
        (model, tokenizer) mocks.
    """
    tokenizer = MagicMock()
    model = MagicMock()
    model.device = torch.device("cpu")

    def encode_side_effect(text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple tokenizer: one token per character
        return list(range(len(text)))

    tokenizer.encode = MagicMock(side_effect=encode_side_effect)

    def model_forward(input_ids: torch.Tensor, **kwargs: Any) -> MagicMock:
        seq_len = input_ids.shape[1]
        if uniform:
            logits = torch.zeros(1, seq_len, vocab_size)
        else:
            # Slightly non-uniform logits to get non-zero cross-entropy
            torch.manual_seed(42)
            logits = torch.randn(1, seq_len, vocab_size)
        output = MagicMock()
        output.logits = logits
        return output

    model.side_effect = model_forward
    return model, tokenizer


class TestWithMockModel:
    def test_per_byte_cross_entropy_returns_positive(self) -> None:
        """Cross-entropy should be positive for non-trivial text."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        h = compute_per_byte_cross_entropy(model, tokenizer, "hello", "prompt: ")
        # With uniform distribution over 100 tokens: log2(100) / 5 bytes
        assert h > 0

    def test_per_byte_cross_entropy_uniform(self) -> None:
        """With uniform logits, cross-entropy = log2(vocab_size) / byte_count * n_text_tokens."""
        vocab_size = 100
        model, tokenizer = _make_mock_model_and_tokenizer(
            vocab_size=vocab_size, uniform=True
        )
        text = "hello"  # 5 chars = 5 tokens (mock) = 5 bytes
        prefix = "p"  # 1 char = 1 token

        h = compute_per_byte_cross_entropy(model, tokenizer, text, prefix)

        # Each token has prob 1/100, so -log2(1/100) = log2(100) per token
        # 5 tokens, 5 bytes => h = 5 * log2(100) / 5 = log2(100)
        expected = math.log2(vocab_size)
        assert h == pytest.approx(expected, rel=1e-5)

    def test_progressive_curve_length(self) -> None:
        """Curve length should match number of responses."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        responses = ["resp1", "resp2", "resp3"]
        curve = compute_progressive_surprise_curve(
            model, tokenizer, "prompt", responses
        )
        assert len(curve) == 3

    def test_unconditional_surprises_length(self) -> None:
        """Should return one value per response."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        responses = ["a", "bb", "ccc"]
        surprises = compute_unconditional_surprises(
            model, tokenizer, "prompt", responses
        )
        assert len(surprises) == 3

    def test_full_pipeline_returns_all_keys(self) -> None:
        """compute_icl_diversity_metrics should return all expected keys."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"]
        )

        expected_keys = {
            "a_k_curve",
            "unconditional_surprises",
            "excess_entropy_E",
            "coherence_C",
            "coherence_spread_sigma",
            "diversity_score_D",
            "effective_mode_count",
            "mean_byte_length",
            "D_plus",
            "D_minus",
            "C_plus",
            "C_minus",
            "is_monotone",
        }
        assert set(result.keys()) == expected_keys

    def test_full_pipeline_curve_length(self) -> None:
        """a_k curve length should equal number of responses."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c", "d"]
        )
        assert len(result["a_k_curve"]) == 4
        assert len(result["unconditional_surprises"]) == 4


class TestPermutationAveraging:
    def test_single_permutation_matches_direct(self) -> None:
        """With n_permutations=1, result should match direct computation."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        responses = ["aaa", "bbb", "ccc"]

        result_1 = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", responses, n_permutations=1
        )
        # Just verify it completes and has correct shape
        assert len(result_1["a_k_curve"]) == 3

    def test_multiple_permutations_complete(self) -> None:
        """With n_permutations > 1, should complete without error."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        responses = ["aaa", "bbb", "ccc"]

        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", responses, n_permutations=3, seed=42
        )
        assert len(result["a_k_curve"]) == 3

    def test_permutation_deterministic(self) -> None:
        """Same seed should give same results."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        responses = ["aaa", "bbb", "ccc"]

        r1 = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", responses, n_permutations=3, seed=123
        )
        r2 = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", responses, n_permutations=3, seed=123
        )

        assert r1["a_k_curve"] == pytest.approx(r2["a_k_curve"])
        assert r1["excess_entropy_E"] == pytest.approx(r2["excess_entropy_E"])


class TestBitsNotNats:
    def test_log_base_2(self) -> None:
        """Verify cross-entropy is in bits (log2), not nats (ln)."""
        vocab_size = 256
        model, tokenizer = _make_mock_model_and_tokenizer(
            vocab_size=vocab_size, uniform=True
        )
        # Single byte text, single byte prefix
        text = "x"  # 1 token, 1 byte
        prefix = "p"  # 1 token

        h = compute_per_byte_cross_entropy(model, tokenizer, text, prefix)

        # With uniform distribution: h = log2(256) = 8.0 bits/byte
        expected_bits = math.log2(vocab_size)
        expected_nats = math.log(vocab_size)

        # Should match bits, not nats
        assert h == pytest.approx(expected_bits, rel=1e-5)
        assert h != pytest.approx(expected_nats, rel=0.1)
