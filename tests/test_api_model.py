"""Tests for APIModel (mocked, no real API calls).

Tests cover:
- score_sequences parsing of prompt_logprobs
- Token count validation (mismatch raises)
- Retry on API errors
- Full pipeline: mock APIModel → compute_icl_diversity_metrics
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from icl_diversity.api_model import APIModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(
    encode_result: list[int] | None = None,
    decode_result: str | None = None,
) -> MagicMock:
    """Create a mock tokenizer."""
    tok = MagicMock()
    tok.name_or_path = "mock-model"
    if encode_result is not None:
        tok.encode = MagicMock(return_value=encode_result)
    if decode_result is not None:
        tok.decode = MagicMock(return_value=decode_result)
    tok.pad_token_id = 0
    tok.eos_token_id = 0
    return tok


def _make_mock_completion_response(
    prompt_logprobs: list[dict[str, float] | None],
) -> MagicMock:
    """Create a mock OpenAI completion response with prompt_logprobs."""
    resp = MagicMock()
    resp.prompt_logprobs = prompt_logprobs
    return resp


# ---------------------------------------------------------------------------
# APIModel construction
# ---------------------------------------------------------------------------


class TestAPIModelInit:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            APIModel("some-model", provider="badprovider", api_key="key")

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                APIModel("some-model", provider="together")

    def test_explicit_api_key(self) -> None:
        tok = _make_mock_tokenizer()
        with patch("icl_diversity.api_model.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = tok
            model = APIModel(
                "test-model",
                provider="together",
                api_key="test-key-123",
            )
            assert model.model_name == "test-model"
            assert model.provider == "together"
            assert model.tokenizer is tok

    def test_custom_tokenizer(self) -> None:
        tok = _make_mock_tokenizer()
        model = APIModel(
            "test-model",
            provider="together",
            api_key="test-key",
            tokenizer=tok,
        )
        assert model.tokenizer is tok


# ---------------------------------------------------------------------------
# score_single_text
# ---------------------------------------------------------------------------


class TestScoreSingleText:
    def _make_model_with_mock_client(
        self, tokenizer: MagicMock | None = None
    ) -> APIModel:
        tok = tokenizer or _make_mock_tokenizer()
        model = APIModel(
            "test-model",
            provider="together",
            api_key="test-key",
            tokenizer=tok,
        )
        return model

    def test_parses_prompt_logprobs(self) -> None:
        """Verify correct parsing of Together AI prompt_logprobs format."""
        tok = _make_mock_tokenizer(encode_result=[10, 20, 30])
        model = self._make_model_with_mock_client(tok)

        # 3 tokens: first has None logprob, others have logprobs
        prompt_logprobs = [
            None,  # first token
            {"token_20": -1.5},  # log P(token 20 | token 10)
            {"token_30": -2.0},  # log P(token 30 | tokens 10, 20)
        ]
        mock_resp = _make_mock_completion_response(prompt_logprobs)
        model._client.completions.create = MagicMock(return_value=mock_resp)

        result = model._score_single_text("hello world foo")

        assert len(result) == 3
        # result[0] = log2 P(token[1] | token[0]) = prompt_logprobs[1] / ln(2)
        assert result[0] == pytest.approx(-1.5 / math.log(2))
        # result[1] = log2 P(token[2] | token[0..1]) = prompt_logprobs[2] / ln(2)
        assert result[1] == pytest.approx(-2.0 / math.log(2))
        # result[2] = 0.0 (last position padding)
        assert result[2] == 0.0

    def test_empty_text(self) -> None:
        tok = _make_mock_tokenizer(encode_result=[])
        model = self._make_model_with_mock_client(tok)
        result = model._score_single_text("")
        assert result == []

    def test_single_token(self) -> None:
        tok = _make_mock_tokenizer(encode_result=[42])
        model = self._make_model_with_mock_client(tok)

        prompt_logprobs = [None]
        mock_resp = _make_mock_completion_response(prompt_logprobs)
        model._client.completions.create = MagicMock(return_value=mock_resp)

        result = model._score_single_text("x")
        assert len(result) == 1
        assert result[0] == 0.0  # single token, no next token

    def test_token_count_mismatch_raises(self) -> None:
        tok = _make_mock_tokenizer(encode_result=[10, 20, 30])
        model = self._make_model_with_mock_client(tok)

        # API returns 2 tokens but local has 3
        prompt_logprobs = [None, {"t": -1.0}]
        mock_resp = _make_mock_completion_response(prompt_logprobs)
        model._client.completions.create = MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="Token count mismatch"):
            model._score_single_text("hello world foo")

    def test_no_prompt_logprobs_raises(self) -> None:
        tok = _make_mock_tokenizer(encode_result=[10])
        model = self._make_model_with_mock_client(tok)

        mock_resp = _make_mock_completion_response(None)
        mock_resp.prompt_logprobs = None
        model._client.completions.create = MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="did not return prompt_logprobs"):
            model._score_single_text("x")

    def test_retry_on_error(self) -> None:
        tok = _make_mock_tokenizer(encode_result=[10, 20])
        model = self._make_model_with_mock_client(tok)

        prompt_logprobs = [None, {"t": -1.0}]
        mock_resp = _make_mock_completion_response(prompt_logprobs)

        # First call fails, second succeeds
        model._client.completions.create = MagicMock(
            side_effect=[RuntimeError("rate limit"), mock_resp]
        )

        result = model._score_single_text("hello")
        assert len(result) == 2
        assert model._client.completions.create.call_count == 2

    def test_all_retries_exhausted(self) -> None:
        tok = _make_mock_tokenizer(encode_result=[10])
        model = self._make_model_with_mock_client(tok)
        model._retry_attempts = 2

        model._client.completions.create = MagicMock(
            side_effect=RuntimeError("always fails")
        )

        with pytest.raises(RuntimeError, match="API call failed after 2 attempts"):
            model._score_single_text("x")


# ---------------------------------------------------------------------------
# score_sequences (batch)
# ---------------------------------------------------------------------------


class TestScoreSequences:
    def test_batch_scoring(self) -> None:
        """score_sequences should handle a batch of sequences."""
        tok = _make_mock_tokenizer()
        # Override encode/decode per call
        tok.encode = MagicMock(return_value=[10, 20, 30])
        tok.decode = MagicMock(return_value="hello world foo")

        model = APIModel(
            "test-model",
            provider="together",
            api_key="test-key",
            tokenizer=tok,
        )

        prompt_logprobs = [None, {"t": -1.0}, {"t": -2.0}]
        mock_resp = _make_mock_completion_response(prompt_logprobs)
        model._client.completions.create = MagicMock(return_value=mock_resp)

        input_ids = torch.tensor([[10, 20, 30], [10, 20, 30]])
        result = model.score_sequences(input_ids)

        assert result.shape == (2, 3)
        # Both sequences should have same values
        torch.testing.assert_close(result[0], result[1])

    def test_with_attention_mask(self) -> None:
        """Padding positions should produce 0.0."""
        tok = _make_mock_tokenizer()
        # Real tokens: [10, 20], padded with 0
        tok.encode = MagicMock(return_value=[10, 20])
        tok.decode = MagicMock(return_value="hello")

        model = APIModel(
            "test-model",
            provider="together",
            api_key="test-key",
            tokenizer=tok,
        )

        prompt_logprobs = [None, {"t": -1.0}]
        mock_resp = _make_mock_completion_response(prompt_logprobs)
        model._client.completions.create = MagicMock(return_value=mock_resp)

        input_ids = torch.tensor([[10, 20, 0]])
        attention_mask = torch.tensor([[1, 1, 0]])
        result = model.score_sequences(input_ids, attention_mask)

        assert result.shape == (1, 3)
        # Last position (padding) should be 0.0
        assert result[0, 2].item() == 0.0
        # First two should have real values
        assert result[0, 0].item() == pytest.approx(-1.0 / math.log(2))
        assert result[0, 1].item() == 0.0  # last real token → padding


# ---------------------------------------------------------------------------
# Full pipeline with mocked APIModel
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_api_model_through_compute_icl_diversity_metrics(self) -> None:
        """A mocked APIModel should produce valid metrics through the full pipeline."""
        from icl_diversity import compute_icl_diversity_metrics

        tok = MagicMock()
        tok.name_or_path = "mock-model"
        tok.pad_token_id = 0
        tok.eos_token_id = 0

        # We need encode to return consistent token IDs for boundary detection.
        # Use a simple scheme: each character → one token.
        def mock_encode(text: str, add_special_tokens: bool = True) -> list[int]:
            return [ord(c) for c in text]

        def mock_decode(ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

        tok.encode = mock_encode
        tok.decode = mock_decode

        model = APIModel(
            "test-model",
            provider="together",
            api_key="test-key",
            tokenizer=tok,
        )

        # Mock score_sequences to return uniform negative log-probs
        def mock_score_sequences(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, seq_len = input_ids.shape
            # Return -1.0 bit for every position (except last = 0.0)
            result = torch.full((batch, seq_len), -1.0)
            result[:, -1] = 0.0
            return result

        model.score_sequences = mock_score_sequences  # type: ignore[assignment]

        prompt = "Q"
        responses = ["AB", "CD", "EF"]

        metrics = compute_icl_diversity_metrics(
            model=model,
            tokenizer=None,  # Should auto-resolve from APIModel
            prompt=prompt,
            responses=responses,
            n_permutations=1,
        )

        # Basic sanity: metrics dict should have expected keys
        assert "excess_entropy_E" in metrics
        assert "coherence_C" in metrics
        assert "diversity_score_D" in metrics
        assert "a_k_curve" in metrics
        assert len(metrics["a_k_curve"]) == 3

        # With uniform log-probs, all responses should have similar surprise
        # so excess entropy should be near 0
        assert metrics["excess_entropy_E"] >= 0.0
