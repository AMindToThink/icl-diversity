"""
Equivalence tests: single-pass vs multi-pass a_k computation.

These are integration tests that require GPT-2 (~500 MB, runs on CPU).
They verify that the single-pass optimization produces identical results
to the original n-forward-pass implementation.
"""

import os

import numpy as np
import pytest

from icl_diversity import (
    compute_progressive_surprise_curve,
    compute_progressive_surprise_curve_single_pass,
    compute_unconditional_surprises,
)
from icl_diversity.core import _response_label

# ---------------------------------------------------------------------------
# Model loading (skip all tests if GPT-2 not available)
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _MODEL_ID = "gpt2"
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
    _model.eval()
    _HAS_MODEL = True
except Exception:
    _HAS_MODEL = False

pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="GPT-2 model not available")


# ============================================================================
# Test data
# ============================================================================

PROMPT = "Write a short story about a cat."
RESPONSES = [
    "The cat sat on the mat and purred softly.",
    "Once upon a time, a brave kitten ventured into the woods.",
    "Mr. Whiskers had a secret: he could fly.",
    "It was raining, and the tabby watched from the windowsill.",
    "The alley cat dodged between trash cans, hunting for dinner.",
]

PROMPT_SHORT = "Tell me a joke."
RESPONSES_SHORT = [
    "Why did the chicken cross the road?",
    "To get to the other side!",
]


# ============================================================================
# Equivalence tests
# ============================================================================


class TestSingleVsMultiPass:
    """The single-pass optimization must produce the same a_k values as the
    original n-forward-pass implementation."""

    def test_five_responses(self) -> None:
        a_k_multi = compute_progressive_surprise_curve(
            _model, _tokenizer, PROMPT, RESPONSES
        )
        a_k_single = compute_progressive_surprise_curve_single_pass(
            _model, _tokenizer, PROMPT, RESPONSES
        )

        a_multi = np.array(a_k_multi)
        a_single = np.array(a_k_single)

        print(f"\n  multi-pass:  {np.round(a_multi, 6)}")
        print(f"  single-pass: {np.round(a_single, 6)}")
        print(f"  diff:        {np.round(a_single - a_multi, 8)}")

        np.testing.assert_allclose(
            a_single,
            a_multi,
            atol=1e-4,
            err_msg="Single-pass and multi-pass a_k values diverged",
        )

    def test_two_responses(self) -> None:
        a_k_multi = compute_progressive_surprise_curve(
            _model, _tokenizer, PROMPT_SHORT, RESPONSES_SHORT
        )
        a_k_single = compute_progressive_surprise_curve_single_pass(
            _model, _tokenizer, PROMPT_SHORT, RESPONSES_SHORT
        )

        np.testing.assert_allclose(
            np.array(a_k_single),
            np.array(a_k_multi),
            atol=1e-4,
            err_msg="Single-pass and multi-pass a_k values diverged (2 responses)",
        )

    def test_single_response(self) -> None:
        """Edge case: only one response."""
        responses = [RESPONSES[0]]
        a_k_multi = compute_progressive_surprise_curve(
            _model, _tokenizer, PROMPT, responses
        )
        a_k_single = compute_progressive_surprise_curve_single_pass(
            _model, _tokenizer, PROMPT, responses
        )

        np.testing.assert_allclose(
            np.array(a_k_single),
            np.array(a_k_multi),
            atol=1e-4,
        )

    def test_empty_responses(self) -> None:
        """Edge case: no responses."""
        assert compute_progressive_surprise_curve_single_pass(
            _model, _tokenizer, PROMPT, []
        ) == []


# ============================================================================
# Boundary roundtrip test
# ============================================================================


class TestBoundaryRoundtrip:
    """Verify that token boundaries correctly identify response regions.

    BPE tokenization may merge characters at the boundary between the
    delimiter (e.g. ": ") and the response start, so decoded slices may
    differ slightly from the original text (e.g. missing a leading capital
    that was merged into the delimiter token).  We check that:

    1. The decoded slice is a suffix of the original response (possibly
       missing a few leading characters absorbed by the delimiter token).
    2. The number of tokens assigned to each response is reasonable.
    3. Boundaries cover the full sequence without gaps or overlaps.
    """

    @staticmethod
    def _compute_boundaries(
        prompt: str, responses: list[str]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        parts = [prompt]
        for i, resp in enumerate(responses):
            label = _response_label(i)
            parts.append(f"\n\nResponse {label}: {resp}")
        full_text = "".join(parts)
        full_ids = _tokenizer.encode(full_text, add_special_tokens=False)

        boundaries: list[tuple[int, int]] = []
        running_text = prompt + f"\n\nResponse {_response_label(0)}: "
        for k in range(len(responses)):
            n_prefix = len(
                _tokenizer.encode(running_text, add_special_tokens=False)
            )
            running_text += responses[k]
            n_with_resp = len(
                _tokenizer.encode(running_text, add_special_tokens=False)
            )
            boundaries.append((n_prefix, n_with_resp))
            if k < len(responses) - 1:
                running_text += f"\n\nResponse {_response_label(k + 1)}: "

        return full_ids, boundaries

    def test_boundaries_cover_sequence(self) -> None:
        """Boundaries should cover from first response start to end of sequence."""
        full_ids, boundaries = self._compute_boundaries(PROMPT, RESPONSES)
        # Last boundary end should equal sequence length
        assert boundaries[-1][1] == len(full_ids)
        # No gaps between consecutive responses (there may be delimiter tokens
        # between end of response k and start of response k+1, which is expected)
        for k in range(len(boundaries) - 1):
            assert boundaries[k][1] <= boundaries[k + 1][0], (
                f"Overlap between response {k} and {k + 1}: "
                f"{boundaries[k]} vs {boundaries[k + 1]}"
            )

    def test_decoded_slices_are_response_suffixes(self) -> None:
        """Each decoded slice should be a suffix of the original response,
        possibly with minor leading character differences due to BPE merging."""
        full_ids, boundaries = self._compute_boundaries(PROMPT, RESPONSES)
        for i, (start, end) in enumerate(boundaries):
            decoded = _tokenizer.decode(full_ids[start:end])
            # The decoded text should end with the response (possibly with
            # leading whitespace/characters absorbed into delimiter token)
            assert RESPONSES[i].endswith(decoded.lstrip()), (
                f"Response {i}: decoded {decoded!r} is not a suffix of "
                f"{RESPONSES[i]!r}"
            )
            # At most a few characters should be missing
            assert len(decoded.strip()) >= len(RESPONSES[i]) - 5, (
                f"Response {i}: too many characters lost. "
                f"Original={RESPONSES[i]!r}, decoded={decoded!r}"
            )

    def test_each_response_has_tokens(self) -> None:
        """Each response should have at least one token."""
        _, boundaries = self._compute_boundaries(PROMPT_SHORT, RESPONSES_SHORT)
        for i, (start, end) in enumerate(boundaries):
            assert end > start, f"Response {i} has no tokens: ({start}, {end})"


# ============================================================================
# a_1 consistency test
# ============================================================================


class TestA1Consistency:
    """a_1 from single-pass should equal unconditional cross-entropy of r_1,
    since r_1 is conditioned only on the prompt in both cases (same formatting)."""

    def test_a1_equals_unconditional(self) -> None:
        a_k_single = compute_progressive_surprise_curve_single_pass(
            _model, _tokenizer, PROMPT, RESPONSES
        )
        unconditional = compute_unconditional_surprises(
            _model, _tokenizer, PROMPT, RESPONSES
        )

        a1 = a_k_single[0]
        h1 = unconditional[0]

        print(f"\n  a_1 (single-pass): {a1:.6f}")
        print(f"  h(r_1|p) (unconditional): {h1:.6f}")
        print(f"  diff: {abs(a1 - h1):.8f}")

        # These should match because the formatting for the first response
        # is identical: prompt + "\n\nResponse A: " + response
        np.testing.assert_allclose(
            a1,
            h1,
            atol=1e-4,
            err_msg="a_1 from single-pass should match unconditional h(r_1|p)",
        )
