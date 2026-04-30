"""Response-boundary tests for the single-pass a_k computation.

Boundary tests use the Qwen2.5-3B tokenizer (the paper's primary base model),
which exhibits the `.` + `\\n\\n` -> `.\\n\\n` BPE merge that GPT-2 does not.
The a_1 forward-pass consistency check uses GPT-2 (cheap; tokenizer-agnostic).

Verifies that the token-boundary detection in `_find_response_boundaries`:
  (a) produces boundaries covering the full concatenated sequence without
      overlaps or gaps,
  (b) never lets the separator's `Response` label leak into a response's
      decoded slice,
  (c) keeps each response to within a handful of characters of its original
      length even when BPE absorbs a few characters into the delimiter token,
  (d) attributes the Qwen `.\\n\\n` merged token to the response (the actual
      behavior of the character-span overlap rule, documented in the paper's
      "Boundary handling" paragraph),
  (e) produces a_1 values exactly equal to the unconditional per-response
      cross-entropy h(r_1 | p), since r_1 is conditioned only on the prompt
      in both cases.

There used to be a "single-pass vs multi-pass" equivalence test suite here.
It's been removed: in a causal LM, the two are equivalent by construction
(pass n of any multi-pass sequence already contains all the information of
passes 1..n-1 via causal attention), so comparing them could only catch bugs
that are caught more directly by the tests above.
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
# Tokenizer / model loading
# ---------------------------------------------------------------------------
# Qwen tokenizer for boundary tests (no weights needed); GPT-2 model+tokenizer
# for the a_1 forward-pass consistency check. Each fixture is independent: if
# Qwen's tokenizer isn't available, only the boundary tests skip; if GPT-2
# isn't available, only the a_1 test skips.
try:
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _QWEN_TOKENIZER_ID = "Qwen/Qwen2.5-3B"
    _qwen_tokenizer = AutoTokenizer.from_pretrained(_QWEN_TOKENIZER_ID)
    _HAS_QWEN_TOKENIZER = True
except Exception:
    _HAS_QWEN_TOKENIZER = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F811

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _GPT2_MODEL_ID = "gpt2"
    _gpt2_tokenizer = AutoTokenizer.from_pretrained(_GPT2_MODEL_ID)
    _gpt2_model = AutoModelForCausalLM.from_pretrained(_GPT2_MODEL_ID)
    _gpt2_model.eval()
    _HAS_GPT2 = True
except Exception:
    _HAS_GPT2 = False


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


@pytest.mark.skipif(not _HAS_GPT2, reason="GPT-2 not available")
def test_empty_responses() -> None:
    curve, byte_counts = compute_progressive_surprise_curve_single_pass(
        _gpt2_model, _gpt2_tokenizer, PROMPT, []
    )
    assert curve == []
    assert byte_counts == []


# ============================================================================
# Boundary roundtrip
# ============================================================================


@pytest.mark.skipif(not _HAS_QWEN_TOKENIZER, reason="Qwen2.5-3B tokenizer not available")
class TestBoundaryRoundtrip:
    """Verify that token boundaries correctly identify response regions.

    Uses the Qwen2.5-3B tokenizer (the paper's primary base model). BPE
    tokenization may merge characters at the boundary between the delimiter
    (e.g. ": ") and the response start, and may merge a response's trailing
    character with the following separator (e.g. "." + "\\n\\n" -> ".\\n\\n").
    The character-span overlap rule in `_find_response_boundaries` attributes
    any token whose char span overlaps the response's span to that response,
    so trailing merged tokens land inside the response's boundary. We check:

    1. The decoded slice is a suffix of the original response (possibly
       missing a few leading characters absorbed by the delimiter token).
    2. The number of tokens assigned to each response is reasonable.
    3. Boundaries cover the full sequence without gaps or overlaps.
    4. The Qwen ".\\n\\n" merge lands inside the response's boundary
       (documenting the actual overlap-rule behavior).
    """

    @staticmethod
    def _compute_boundaries(
        prompt: str, responses: list[str]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        return _find_response_boundaries(_qwen_tokenizer, prompt, responses)

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
        """Each decoded slice should contain the original response as a substring,
        possibly with minor leading/trailing character differences due to BPE merging."""
        full_ids, boundaries = self._compute_boundaries(PROMPT, RESPONSES)
        for i, (start, end) in enumerate(boundaries):
            decoded = _qwen_tokenizer.decode(full_ids[start:end])
            # Strip leading whitespace (delimiter space absorbed into first token)
            # and trailing newline characters (Qwen's ".\n\n" merge at the response/
            # separator boundary, attributed to the response by the overlap rule).
            stripped = decoded.lstrip().rstrip("\n")
            assert stripped.endswith(RESPONSES[i][-min(len(RESPONSES[i]), 10):]) or RESPONSES[i].endswith(stripped), (
                f"Response {i}: decoded {decoded!r} does not align with {RESPONSES[i]!r}"
            )
            assert len(stripped) >= len(RESPONSES[i]) - 5, (
                f"Response {i}: too many characters lost. "
                f"Original={RESPONSES[i]!r}, decoded={decoded!r}"
            )

    def test_each_response_has_tokens(self) -> None:
        """Each response should have at least one token."""
        _, boundaries = self._compute_boundaries(PROMPT_SHORT, RESPONSES_SHORT)
        for i, (start, end) in enumerate(boundaries):
            assert end > start, f"Response {i} has no tokens: ({start}, {end})"

    def test_no_separator_leaks_into_response(self) -> None:
        """Decoded tokens for a response must not contain the separator's
        ``Response`` label text.

        This is a weaker invariant than full separator exclusion: with Qwen,
        the trailing ``.\\n\\n`` merged token IS attributed to the response
        (see ``test_qwen_trailing_merge_attributed_to_response``), so the
        decoded slice can contain ``\\n\\n``. What it must never contain is
        the next response's ``Response`` prefix label.
        """
        responses = [
            "Rain falls gently.",
            "The drops patter on the roof.",
            "Umbrellas bloom like flowers.",
        ]
        full_ids, boundaries = self._compute_boundaries(PROMPT, responses)
        for i, (start, end) in enumerate(boundaries):
            decoded = _qwen_tokenizer.decode(full_ids[start:end])
            assert "Response" not in decoded, (
                f"Response {i}: separator label leaked into boundary. "
                f"tokens [{start}:{end}] decoded to {decoded!r}"
            )

    def test_qwen_trailing_merge_attributed_to_response(self) -> None:
        """When Qwen merges a response's trailing ``.`` with the following
        ``\\n\\n`` separator into a single ``.\\n\\n`` token, the boundary
        detector's character-span overlap rule attributes that token to the
        response (not the separator).

        This pins down the actual behavior described in the paper's
        ``Boundary handling`` paragraph (Practical Findings). Earlier versions
        of the docs claimed the opposite, which the code never did.
        """
        responses = [
            "Rain falls gently.",
            "The drops patter on the roof.",
            "Umbrellas bloom like flowers.",
        ]
        full_ids, boundaries = self._compute_boundaries(PROMPT, responses)

        # Confirm the Qwen merge actually happens for non-final responses:
        # each one's last token should decode to ".\n\n" (period + separator).
        for i in range(len(responses) - 1):
            start, end = boundaries[i]
            last_token = _qwen_tokenizer.decode([full_ids[end - 1]])
            assert last_token == ".\n\n", (
                f"Response {i}: expected merged token '.\\n\\n' as last token in "
                f"the response's boundary, got {last_token!r}. If Qwen's tokenizer "
                f"merge behavior changed, update the paper's Boundary handling "
                f"paragraph and this test together."
            )

        # The final response is not followed by a separator, so no merge.
        start, end = boundaries[-1]
        last_token = _qwen_tokenizer.decode([full_ids[end - 1]])
        assert "\n" not in last_token, (
            f"Final response unexpectedly has newline in last token: {last_token!r}"
        )


# ============================================================================
# a_1 consistency
# ============================================================================


@pytest.mark.skipif(not _HAS_GPT2, reason="GPT-2 not available")
class TestA1Consistency:
    """a_1 should equal the unconditional cross-entropy h(r_1 | p), since the
    first response is conditioned only on the prompt in both cases. This is the
    only externally-computable check on the single-pass boundary extraction."""

    def test_a1_equals_unconditional(self) -> None:
        a_k_single, _ = compute_progressive_surprise_curve_single_pass(
            _gpt2_model, _gpt2_tokenizer, PROMPT, RESPONSES
        )
        _, total_bits, _ = compute_unconditional_surprises(
            _gpt2_model, _gpt2_tokenizer, PROMPT, RESPONSES
        )
        a1 = a_k_single[0]
        h1_total = total_bits[0]
        np.testing.assert_allclose(
            a1,
            h1_total,
            atol=1e-4,
            err_msg="a_1 from single-pass should match unconditional h(r_1|p)",
        )
