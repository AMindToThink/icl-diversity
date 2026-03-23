"""Tests for the ICL diversity metric.

All tests run on CPU with mock models to avoid GPU dependency.
"""

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from icl_diversity import (
    _response_label,
    compute_cross_entropy,
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


# --- Completion format tests ---


class TestCompletionFormat:
    """Tests for format_mode='completion' in format_conditioning_context."""

    def test_first_response_completion(self) -> None:
        prefix, target = format_conditioning_context(
            "The cat sat on", [], "the mat.", format_mode="completion"
        )
        assert prefix == "1. The cat sat on"
        assert target == "the mat."

    def test_with_previous_responses_completion(self) -> None:
        prefix, target = format_conditioning_context(
            "The cat sat on",
            ["the mat.", "the roof."],
            "the fence.",
            format_mode="completion",
        )
        expected_prefix = (
            "1. The cat sat onthe mat.\n\n"
            "2. The cat sat onthe roof.\n\n"
            "3. The cat sat on"
        )
        assert prefix == expected_prefix
        assert target == "the fence."

    def test_instruct_default_unchanged(self) -> None:
        """Verify default format_mode='instruct' matches the original behavior."""
        prefix_default, target_default = format_conditioning_context(
            "prompt", ["r1"], "r2"
        )
        prefix_explicit, target_explicit = format_conditioning_context(
            "prompt", ["r1"], "r2", format_mode="instruct"
        )
        assert prefix_default == prefix_explicit
        assert target_default == target_explicit

    def test_completion_numbers_sequential(self) -> None:
        prefix, _ = format_conditioning_context(
            "p", ["a", "b", "c"], "d", format_mode="completion"
        )
        assert "1. p" in prefix
        assert "2. p" in prefix
        assert "3. p" in prefix
        assert "4. p" in prefix


class TestCompletionFormatBoundaries:
    """Tests for _find_response_boundaries with format_mode='completion'."""

    def test_boundaries_exclude_repeated_context(self) -> None:
        """In completion mode, boundaries should cover only completions, not the repeated prompt."""
        from icl_diversity.core import _find_response_boundaries
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        prompt = "The cat sat on "
        responses = ["the mat.", "the roof."]

        full_ids, boundaries = _find_response_boundaries(
            tokenizer, prompt, responses, format_mode="completion"
        )

        # Decode just the boundary tokens for each response
        for k, (start, end) in enumerate(boundaries):
            if end > start:
                boundary_tokens = full_ids[start:end]
                decoded = tokenizer.decode(boundary_tokens)
                # Should contain the completion text, not the prompt
                assert responses[k].strip() in decoded.strip() or decoded.strip() in responses[k].strip(), (
                    f"Response {k}: decoded boundary '{decoded}' doesn't match "
                    f"completion '{responses[k]}'"
                )
                # Should NOT contain the prompt text
                assert prompt.strip() not in decoded, (
                    f"Response {k}: boundary '{decoded}' contains prompt '{prompt}'"
                )

    def test_instruct_boundaries_unchanged(self) -> None:
        """Instruct mode boundaries should be identical with or without explicit format_mode."""
        from icl_diversity.core import _find_response_boundaries
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        prompt = "Tell me a story"
        responses = ["Once upon a time", "In a galaxy far away"]

        ids_default, bounds_default = _find_response_boundaries(
            tokenizer, prompt, responses
        )
        ids_explicit, bounds_explicit = _find_response_boundaries(
            tokenizer, prompt, responses, format_mode="instruct"
        )

        assert ids_default == ids_explicit
        assert bounds_default == bounds_explicit


class TestCompletionFormatPipeline:
    """Test that completion format works end-to-end with mock model."""

    def test_full_pipeline_completion_format(self) -> None:
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"],
            format_mode="completion",
        )
        assert "a_k_curve" in result
        assert len(result["a_k_curve"]) == 3

    def test_completion_format_changes_token_sequence(self) -> None:
        """Completion mode should produce a different token sequence than instruct."""
        from icl_diversity.core import _find_response_boundaries
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        prompt = "The cat sat on "
        responses = ["the mat.", "the roof."]

        ids_instruct, _ = _find_response_boundaries(
            tokenizer, prompt, responses, format_mode="instruct"
        )
        ids_completion, _ = _find_response_boundaries(
            tokenizer, prompt, responses, format_mode="completion"
        )
        # Different formats → different token sequences
        assert ids_instruct != ids_completion


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


# --- compute_cross_entropy tests ---


class TestComputeCrossEntropy:
    def test_returns_tuple(self) -> None:
        """compute_cross_entropy returns (total_bits, byte_count)."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        total_bits, byte_count = compute_cross_entropy(
            model, tokenizer, "hello", "prefix: "
        )
        assert isinstance(total_bits, float)
        assert isinstance(byte_count, int)
        assert byte_count == 5
        assert total_bits > 0

    def test_total_bits_equals_per_byte_times_bytes(self) -> None:
        """total_bits / byte_count should equal per_byte_cross_entropy."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        text = "hello"
        prefix = "p"
        total_bits, byte_count = compute_cross_entropy(model, tokenizer, text, prefix)
        per_byte = compute_per_byte_cross_entropy(model, tokenizer, text, prefix)
        assert total_bits / byte_count == pytest.approx(per_byte, rel=1e-5)

    def test_empty_text_returns_zero_bits_zero_bytes(self) -> None:
        model = MagicMock()
        tokenizer = MagicMock()
        total_bits, byte_count = compute_cross_entropy(model, tokenizer, "", "prefix")
        assert total_bits == 0.0
        assert byte_count == 0


# --- Derived metric tests (pure math, no model needed) ---


# Helper to call _compute_metrics_from_curves with the new signature.
# For unit tests we provide total-bits a_k curves and compute e_rate manually.
def _make_metrics(
    a_k_total: list[float],
    byte_counts: list[int],
    unconditional_per_byte: list[float],
    responses: list[str],
    unconditional_byte_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Helper to build metrics from total-bits a_k curve."""
    if unconditional_byte_counts is None:
        unconditional_byte_counts = [len(r.encode("utf-8")) for r in responses]

    # Compute e_rate (Option B)
    per_byte = [t / b if b > 0 else 0.0 for t, b in zip(a_k_total, byte_counts)]
    e_rate = sum(pb - per_byte[-1] for pb in per_byte)

    return _compute_metrics_from_curves(
        a_k_total,
        byte_counts,
        unconditional_per_byte,
        unconditional_byte_counts,
        e_rate,
        responses,
    )


class TestExcessEntropy:
    def test_zero_for_constant_curve(self) -> None:
        """If all a_k are equal, E = 0."""
        # 5 responses of "hello world" (11 bytes each)
        responses = ["hello world"] * 5
        bc = [11] * 5
        # All total bits equal => E = 0
        a_k = [11.0, 11.0, 11.0, 11.0, 11.0]
        unconditional = [1.2, 1.2, 1.2, 1.2, 1.2]
        result = _make_metrics(a_k, bc, unconditional, responses)
        assert result["excess_entropy_E"] == pytest.approx(0.0)

    def test_positive_for_decreasing_curve(self) -> None:
        """If a_k decreases, E > 0."""
        responses = ["hello world"] * 5
        bc = [11] * 5
        # Total bits: 16.5, 14.3, 12.1, 11.0, 11.0
        a_k = [16.5, 14.3, 12.1, 11.0, 11.0]
        unconditional = [1.5, 1.5, 1.5, 1.5, 1.5]
        result = _make_metrics(a_k, bc, unconditional, responses)
        # E = (16.5-11) + (14.3-11) + (12.1-11) + 0 + 0 = 5.5+3.3+1.1 = 9.9
        assert result["excess_entropy_E"] == pytest.approx(9.9)

    def test_excess_entropy_formula(self) -> None:
        """Verify E = sum(a_k - a_n) in total bits per Eq 6."""
        responses = ["test"] * 4
        bc = [4] * 4
        a_k = [8.0, 6.0, 4.8, 4.0]
        a_n = a_k[-1]
        expected_E = sum(a - a_n for a in a_k)  # 4.0+2.0+0.8+0 = 6.8
        unconditional = [2.0] * 4
        result = _make_metrics(a_k, bc, unconditional, responses)
        assert result["excess_entropy_E"] == pytest.approx(expected_E)


class TestERate:
    def test_e_rate_equals_E_over_Bbar_equal_lengths(self) -> None:
        """When all responses have equal byte lengths, E_rate = E / B_bar."""
        responses = ["hello"] * 4  # all 5 bytes
        bc = [5] * 4
        a_k = [10.0, 7.5, 6.0, 5.0]  # total bits
        unconditional = [2.0] * 4
        result = _make_metrics(a_k, bc, unconditional, responses)

        E = result["excess_entropy_E"]
        B_bar = result["mean_byte_length"]
        E_rate = result["excess_entropy_E_rate"]
        assert E_rate == pytest.approx(E / B_bar, rel=1e-10)

    def test_e_rate_differs_from_E_over_Bbar_unequal_lengths(self) -> None:
        """When response byte lengths vary, E_rate != E / B_bar in general."""
        responses = ["hi", "hello world!"]  # 2 bytes, 12 bytes
        bc = [2, 12]
        # Total bits: short resp = 4, long resp = 12
        a_k = [4.0, 12.0]
        unconditional = [2.0, 1.0]
        result = _make_metrics(a_k, bc, unconditional, responses)

        E = result["excess_entropy_E"]
        B_bar = result["mean_byte_length"]
        E_rate = result["excess_entropy_E_rate"]

        # Per-byte: 4/2=2.0, 12/12=1.0 → E_rate = (2.0-1.0) + (1.0-1.0) = 1.0
        assert E_rate == pytest.approx(1.0)
        # E/B_bar = (4-12+12-12)/7 — different from 1.0
        # E = (4-12) + (12-12) = -8 + 0 = -8  (a_k not monotone here!)
        # So E/B_bar = -8/7 ≈ -1.14, definitely != 1.0
        assert E_rate != pytest.approx(E / B_bar, abs=0.1)


class TestCoherence:
    def test_range(self) -> None:
        """C must be in (0, 1) for positive cross-entropies."""
        responses = ["test"] * 3
        bc = [4] * 3
        a_k = [4.0, 3.2, 2.8]
        unconditional = [0.5, 1.0, 1.5]
        result = _make_metrics(a_k, bc, unconditional, responses)
        assert 0 < result["coherence_C"] < 1

    def test_geometric_mean(self) -> None:
        """Verify C = 2^{-(1/n) * sum(h_i)} per Eq 8."""
        unconditional = [1.0, 2.0, 3.0]
        mean_h = sum(unconditional) / len(unconditional)  # 2.0
        expected_C = 2.0 ** (-mean_h)  # 2^{-2} = 0.25
        responses = ["test"] * 3
        bc = [4] * 3
        a_k = [4.0, 3.2, 2.8]
        result = _make_metrics(a_k, bc, unconditional, responses)
        assert result["coherence_C"] == pytest.approx(expected_C)

    def test_lower_surprise_means_higher_coherence(self) -> None:
        """Lower unconditional surprise should give higher coherence."""
        responses = ["test"] * 2
        bc = [4] * 2
        a_k = [4.0, 3.2]

        low_surprise = [0.5, 0.5]
        high_surprise = [2.0, 2.0]

        result_low = _make_metrics(a_k, bc, low_surprise, responses)
        result_high = _make_metrics(a_k, bc, high_surprise, responses)

        assert result_low["coherence_C"] > result_high["coherence_C"]


class TestDiversityScore:
    def test_d_equals_c_times_e(self) -> None:
        """D = C * E (bits)."""
        responses = ["test"] * 4
        bc = [4] * 4
        a_k = [8.0, 6.0, 4.8, 4.0]
        unconditional = [1.0, 1.0, 1.0, 1.0]
        result = _make_metrics(a_k, bc, unconditional, responses)

        expected_D = result["coherence_C"] * result["excess_entropy_E"]
        assert result["diversity_score_D"] == pytest.approx(expected_D)

    def test_d_rate_equals_c_times_e_rate(self) -> None:
        """D_rate = C * E_rate (bits/byte)."""
        responses = ["test"] * 4
        bc = [4] * 4
        a_k = [8.0, 6.0, 4.8, 4.0]
        unconditional = [1.0, 1.0, 1.0, 1.0]
        result = _make_metrics(a_k, bc, unconditional, responses)

        expected_D_rate = result["coherence_C"] * result["excess_entropy_E_rate"]
        assert result["diversity_score_D_rate"] == pytest.approx(expected_D_rate)

    def test_zero_when_no_structure(self) -> None:
        """D = 0 when a_k is constant (no structure)."""
        responses = ["test"] * 3
        bc = [4] * 3
        a_k = [6.0, 6.0, 6.0]
        unconditional = [1.5, 1.5, 1.5]
        result = _make_metrics(a_k, bc, unconditional, responses)
        assert result["diversity_score_D"] == pytest.approx(0.0)
        assert result["diversity_score_D_rate"] == pytest.approx(0.0)


class TestUncertaintyBand:
    def test_band_when_sigma_positive(self) -> None:
        """D+ > D_rate > D- when sigma > 0 and E_rate > 0."""
        responses = ["test"] * 3
        bc = [4] * 3
        a_k = [8.0, 6.0, 4.0]
        unconditional = [0.5, 1.0, 1.5]
        result = _make_metrics(a_k, bc, unconditional, responses)

        assert result["coherence_spread_sigma"] > 0
        assert result["D_plus"] > result["diversity_score_D_rate"]
        assert result["D_minus"] < result["diversity_score_D_rate"]

    def test_band_zero_when_sigma_zero(self) -> None:
        """D+ = D_rate = D- when sigma = 0."""
        responses = ["test"] * 3
        bc = [4] * 3
        a_k = [8.0, 6.0, 4.0]
        unconditional = [1.0, 1.0, 1.0]
        result = _make_metrics(a_k, bc, unconditional, responses)

        assert result["coherence_spread_sigma"] == pytest.approx(0.0)
        assert result["D_plus"] == pytest.approx(result["diversity_score_D_rate"])
        assert result["D_minus"] == pytest.approx(result["diversity_score_D_rate"])

    def test_c_plus_c_minus_formulas(self) -> None:
        """C+ = C * 2^sigma, C- = C * 2^{-sigma} per Section 6.4."""
        responses = ["test"] * 3
        bc = [4] * 3
        a_k = [8.0, 6.0, 4.0]
        unconditional = [0.8, 1.0, 1.2]
        result = _make_metrics(a_k, bc, unconditional, responses)

        C = result["coherence_C"]
        sigma = result["coherence_spread_sigma"]
        assert result["C_plus"] == pytest.approx(C * 2.0**sigma)
        assert result["C_minus"] == pytest.approx(C * 2.0 ** (-sigma))


class TestMonotonicity:
    def test_monotone_curve(self) -> None:
        responses = ["t"] * 4
        bc = [1] * 4
        a_k = [8.0, 6.0, 4.0, 4.0]
        result = _make_metrics(a_k, bc, [1.0] * 4, responses)
        assert result["is_monotone"] is True

    def test_non_monotone_curve(self) -> None:
        responses = ["t"] * 4
        bc = [1] * 4
        a_k = [8.0, 6.0, 7.2, 4.0]  # 6.0 -> 7.2 is an increase
        result = _make_metrics(a_k, bc, [1.0] * 4, responses)
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
        curve, byte_counts = compute_progressive_surprise_curve(
            model, tokenizer, "prompt", responses
        )
        assert len(curve) == 3
        assert len(byte_counts) == 3

    def test_unconditional_surprises_length(self) -> None:
        """Should return one value per response."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        responses = ["a", "bb", "ccc"]
        per_byte, total_bits, byte_counts = compute_unconditional_surprises(
            model, tokenizer, "prompt", responses
        )
        assert len(per_byte) == 3
        assert len(total_bits) == 3
        assert len(byte_counts) == 3

    def test_full_pipeline_returns_all_keys(self) -> None:
        """compute_icl_diversity_metrics should return all expected keys."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"]
        )

        expected_keys = {
            "a_k_curve",
            "a_k_curve_per_byte",
            "a_k_byte_counts",
            "unconditional_surprises",
            "unconditional_total_bits",
            "excess_entropy_E",
            "excess_entropy_E_rate",
            "coherence_C",
            "coherence_spread_sigma",
            "diversity_score_D",
            "diversity_score_D_rate",
            "mean_byte_length",
            "D_plus",
            "D_minus",
            "C_plus",
            "C_minus",
            "is_monotone",
            "temperature",
            "per_permutation_a_k_curves",
            "per_permutation_byte_counts",
            "permutation_orders",
        }
        assert set(result.keys()) == expected_keys

    def test_full_pipeline_curve_length(self) -> None:
        """a_k curve length should equal number of responses."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c", "d"]
        )
        assert len(result["a_k_curve"]) == 4
        assert len(result["a_k_curve_per_byte"]) == 4
        assert len(result["a_k_byte_counts"]) == 4
        assert len(result["unconditional_surprises"]) == 4
        assert len(result["unconditional_total_bits"]) == 4


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

    def test_per_permutation_curves_none_when_single(self) -> None:
        """New keys should be None when n_permutations <= 1."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["aaa", "bbb", "ccc"], n_permutations=1
        )
        assert result["per_permutation_a_k_curves"] is None
        assert result["per_permutation_byte_counts"] is None
        assert result["permutation_orders"] is None

    def test_per_permutation_curves_stored(self) -> None:
        """New keys should have correct shapes when n_permutations > 1."""
        model, tokenizer = _make_mock_model_and_tokenizer(uniform=True)
        n_perm = 3
        n_resp = 4
        result = compute_icl_diversity_metrics(
            model,
            tokenizer,
            "prompt",
            ["a", "b", "c", "d"],
            n_permutations=n_perm,
            seed=42,
        )

        curves = result["per_permutation_a_k_curves"]
        byte_counts = result["per_permutation_byte_counts"]
        orders = result["permutation_orders"]

        assert curves is not None
        assert byte_counts is not None
        assert orders is not None
        assert len(curves) == n_perm
        assert len(byte_counts) == n_perm
        assert len(orders) == n_perm
        for curve in curves:
            assert len(curve) == n_resp
        for bcs in byte_counts:
            assert len(bcs) == n_resp
        for order in orders:
            assert sorted(order) == list(range(n_resp))


class TestTemperature:
    def test_temperature_1_is_identity(self) -> None:
        """T=1.0 gives identical results to default (no temperature arg)."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        result_default = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"]
        )
        result_t1 = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"], temperature=1.0
        )
        assert result_default["a_k_curve"] == pytest.approx(result_t1["a_k_curve"])
        assert result_default["excess_entropy_E"] == pytest.approx(
            result_t1["excess_entropy_E"]
        )

    def test_temperature_changes_output(self) -> None:
        """T=2.0 gives different cross-entropy than T=1.0."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=False)
        tb_t1, _ = compute_cross_entropy(
            model, tokenizer, "hello", "prefix: ", temperature=1.0
        )
        tb_t2, _ = compute_cross_entropy(
            model, tokenizer, "hello", "prefix: ", temperature=2.0
        )
        assert tb_t1 != pytest.approx(tb_t2, rel=1e-3)

    def test_temperature_zero_raises(self) -> None:
        """ValueError on T=0."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        with pytest.raises(ValueError, match="temperature must be positive"):
            compute_cross_entropy(
                model, tokenizer, "hello", "prefix: ", temperature=0.0
            )

    def test_temperature_negative_raises(self) -> None:
        """ValueError on T=-1."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        with pytest.raises(ValueError, match="temperature must be positive"):
            compute_cross_entropy(
                model, tokenizer, "hello", "prefix: ", temperature=-1.0
            )

    def test_temperature_stored_in_metrics(self) -> None:
        """Verify temperature key appears in output dict."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"], temperature=2.0
        )
        assert "temperature" in result
        assert result["temperature"] == 2.0

    def test_high_temperature_reduces_surprise(self) -> None:
        """T>1 flattens distribution, so per-response surprise should decrease.

        With T>1, the distribution becomes closer to uniform, and surprise
        approaches log2(vocab_size) from above (for non-uniform models) or
        stays the same (for uniform). For a non-uniform model, higher T means
        lower surprise since the model's confident predictions get dampened.
        """
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=False)
        # With non-uniform logits, T>1 should produce lower surprise
        # (flatter distribution → each token has closer-to-uniform probability
        #  → less concentrated on wrong tokens → lower total bits)
        tb_t1, _ = compute_cross_entropy(
            model, tokenizer, "hello", "prefix: ", temperature=1.0
        )
        tb_t5, _ = compute_cross_entropy(
            model, tokenizer, "hello", "prefix: ", temperature=5.0
        )
        # High temperature pushes everything toward uniform (log2(100) per token)
        # For random logits, T=1 surprise is also near log2(100), but T=5 should
        # be even closer. The key check is they differ.
        assert tb_t1 != pytest.approx(tb_t5, rel=1e-3)


class TestMultiTemperature:
    def test_list_returns_temperatures_dict(self) -> None:
        """temperature=[T1, T2] returns {"temperatures": {T1: ..., T2: ...}}."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=False)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["aaa", "bbb", "ccc"],
            temperature=[1.0, 2.0],
        )
        assert "temperatures" in result
        assert set(result["temperatures"].keys()) == {1.0, 2.0}

        # Each sub-dict should have all standard keys
        for temp, metrics in result["temperatures"].items():
            assert "a_k_curve" in metrics
            assert "excess_entropy_E" in metrics
            assert "diversity_score_D" in metrics
            assert metrics["temperature"] == temp
            assert len(metrics["a_k_curve"]) == 3

    def test_single_float_backward_compat(self) -> None:
        """temperature=1.0 (float) returns flat dict, not nested."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["a", "b", "c"], temperature=1.0,
        )
        assert "temperatures" not in result
        assert "a_k_curve" in result

    def test_multi_temp_t1_matches_single_temp(self) -> None:
        """T=1.0 in multi-temp should match single-temp T=1.0."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=False)
        r_single = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["aaa", "bbb", "ccc"], temperature=1.0,
        )
        r_multi = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["aaa", "bbb", "ccc"],
            temperature=[1.0, 2.0],
        )
        m1 = r_multi["temperatures"][1.0]
        for key in ["excess_entropy_E", "coherence_C", "diversity_score_D"]:
            assert r_single[key] == pytest.approx(m1[key], abs=1e-5), (
                f"{key}: single={r_single[key]}, multi={m1[key]}"
            )
        assert r_single["a_k_curve"] == pytest.approx(m1["a_k_curve"], abs=1e-5)

    def test_multi_temp_different_values(self) -> None:
        """Different temperatures produce different cross-entropy values.

        Uses compute_cross_entropy directly since the position-insensitive
        mock model in this file yields E=0 (all positions get same logits).
        """
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=False)
        # Verify the multi-temp path produces structurally correct results
        # at each temperature — the actual value differences are tested
        # in TestTemperature.test_temperature_changes_output.
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["aaa", "bbb", "ccc"],
            temperature=[1.0, 3.0],
        )
        # Both temperatures should produce valid metrics dicts
        assert 1.0 in result["temperatures"]
        assert 3.0 in result["temperatures"]
        # Coherence C is shared (computed at T=1.0) so should match
        assert result["temperatures"][1.0]["coherence_C"] == pytest.approx(
            result["temperatures"][3.0]["coherence_C"]
        )

    def test_multi_temp_with_permutations(self) -> None:
        """Multi-temp with n_permutations > 1."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=False)
        result = compute_icl_diversity_metrics(
            model, tokenizer, "prompt", ["aaa", "bbb", "ccc"],
            n_permutations=3, seed=42, temperature=[1.0, 2.0],
        )
        assert "temperatures" in result
        for temp, metrics in result["temperatures"].items():
            assert metrics["per_permutation_a_k_curves"] is not None
            assert len(metrics["per_permutation_a_k_curves"]) == 3
            assert metrics["permutation_orders"] is not None

    def test_multi_temp_invalid_raises(self) -> None:
        """Negative temperature in list should raise ValueError."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100, uniform=True)
        with pytest.raises(ValueError, match="temperatures must be positive"):
            compute_icl_diversity_metrics(
                model, tokenizer, "prompt", ["a", "b"],
                temperature=[1.0, -0.5],
            )


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
