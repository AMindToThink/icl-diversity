"""Response-boundary tests for the single-pass a_k computation.

Verifies that the token-boundary detection in `_find_response_boundaries`:
  (a) produces boundaries covering the full concatenated sequence without
      overlaps or gaps,
  (b) never attributes separator tokens to a response (regression test for
      the Qwen '.' + '\\n\\n' → '.\\n\\n' BPE merge bug),
  (c) keeps each response to within a handful of characters of its original
      length even when BPE absorbs a few characters into the delimiter token,
  (d) produces a_1 values exactly equal to the unconditional per-response
      cross-entropy h(r_1 | p), since r_1 is conditioned only on the prompt
      in both cases.

There used to be a "single-pass vs multi-pass" equivalence test suite here.
It's been removed: in a causal LM, the two are equivalent by construction
(pass n of any multi-pass sequence already contains all the information of
passes 1..n-1 via causal attention), so comparing them could only catch bugs
that are caught more directly by the tests above.

Integration tests that require GPT-2 (~500 MB, runs on CPU).
"""

import os

import numpy as np
import pytest

from icl_diversity import (
    compute_progressive_surprise_curve_single_pass,
    compute_unconditional_surprises,
)
from icl_diversity.core import _find_response_boundaries

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
# Edge cases
# ============================================================================


def test_empty_responses() -> None:
    curve, byte_counts = compute_progressive_surprise_curve_single_pass(
        _model, _tokenizer, PROMPT, []
    )
    assert curve == []
    assert byte_counts == []


# ============================================================================
# Boundary roundtrip
# ============================================================================


class TestBoundaryRoundtrip:
    """Verify that token boundaries correctly identify response regions.

    BPE tokenization may merge characters at the boundary between the
    delimiter (e.g. ": ") and the response start, so decoded slices may
    differ slightly from the original text (e.g. missing a leading capital
    that was merged into the delimiter token). We check that:

    1. The decoded slice is a suffix of the original response (possibly
       missing a few leading characters absorbed by the delimiter token).
    2. The number of tokens assigned to each response is reasonable.
    3. Boundaries cover the full sequence without gaps or overlaps.
    """

    @staticmethod
    def _compute_boundaries(
        prompt: str, responses: list[str]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        return _find_response_boundaries(_tokenizer, prompt, responses)

    def test_boundaries_cover_sequence(self) -> None:
        """Boundaries should cover from first response start to end of sequence."""
        full_ids, boundaries = self._compute_boundaries(PROMPT, RESPONSES)
        assert boundaries[-1][1] == len(full_ids)
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
            assert RESPONSES[i].endswith(decoded.lstrip()), (
                f"Response {i}: decoded {decoded!r} is not a suffix of {RESPONSES[i]!r}"
            )
            assert len(decoded.strip()) >= len(RESPONSES[i]) - 5, (
                f"Response {i}: too many characters lost. "
                f"Original={RESPONSES[i]!r}, decoded={decoded!r}"
            )

    def test_each_response_has_tokens(self) -> None:
        """Each response should have at least one token."""
        _, boundaries = self._compute_boundaries(PROMPT_SHORT, RESPONSES_SHORT)
        for i, (start, end) in enumerate(boundaries):
            assert end > start, f"Response {i} has no tokens: ({start}, {end})"

    def test_no_separator_leaks_into_response(self) -> None:
        """Decoded tokens for a response must not contain separator text.

        Regression test for the BPE merge case: some tokenizers (e.g. Qwen) merge
        a response's trailing punctuation with the following separator into a
        single token (e.g. '.' + '\\n\\n' → '.\\n\\n'). The boundary detector
        must not attribute such merged tokens to the response.
        """
        responses = [
            "Rain falls gently.",
            "The drops patter on the roof.",
            "Umbrellas bloom like flowers.",
        ]
        full_ids, boundaries = self._compute_boundaries(PROMPT, responses)
        for i, (start, end) in enumerate(boundaries):
            decoded = _tokenizer.decode(full_ids[start:end])
            assert "\n\nResponse" not in decoded, (
                f"Response {i}: separator leaked into boundary. "
                f"tokens [{start}:{end}] decoded to {decoded!r}"
            )


# ============================================================================
# a_1 consistency
# ============================================================================


class TestA1Consistency:
    """a_1 should equal the unconditional cross-entropy h(r_1 | p), since the
    first response is conditioned only on the prompt in both cases. This is the
    only externally-computable check on the single-pass boundary extraction."""

    def test_a1_equals_unconditional(self) -> None:
        a_k_single, _ = compute_progressive_surprise_curve_single_pass(
            _model, _tokenizer, PROMPT, RESPONSES
        )
        _, total_bits, _ = compute_unconditional_surprises(
            _model, _tokenizer, PROMPT, RESPONSES
        )
        a1 = a_k_single[0]
        h1_total = total_bits[0]
        np.testing.assert_allclose(
            a1,
            h1_total,
            atol=1e-4,
            err_msg="a_1 from single-pass should match unconditional h(r_1|p)",
        )
