"""Tests for TinkerModel (mocked, no real API calls).

Tests cover:
- TinkerModel construction and validation
- score_sequences parsing of compute_logprobs output
- Retry on API errors
- Full pipeline: mock TinkerModel → compute_icl_diversity_metrics
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from icl_diversity.tinker_model import TinkerModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tinker_model_with_mock_client(
    logprobs_results: list[list[float | None]] | None = None,
    api_key: str = "test-key",
) -> TinkerModel:
    """Create a TinkerModel with mocked internals (no real API calls).

    Patches __init__ to avoid connecting to Tinker, then sets up
    the mock sampling client on the instance.
    """
    mock_tinker = MagicMock()

    mock_client = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_client.get_tokenizer.return_value = mock_tokenizer

    if logprobs_results is not None:
        futures = []
        for result in logprobs_results:
            future = MagicMock()
            future.result.return_value = result
            futures.append(future)
        mock_client.compute_logprobs.side_effect = futures

    # Bypass __init__ entirely, construct manually
    model = object.__new__(TinkerModel)
    model._model_name = "test-model"
    model._max_concurrent = 5
    model._retry_attempts = 3
    model._tinker = mock_tinker
    model._sampling_client = mock_client
    model._tokenizer = mock_tokenizer

    return model


# ---------------------------------------------------------------------------
# TinkerModel construction
# ---------------------------------------------------------------------------


class TestTinkerModelInit:
    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("dotenv.load_dotenv"):
                with pytest.raises(ValueError, match="No API key"):
                    TinkerModel("some-model")

    def test_properties(self) -> None:
        model = _make_tinker_model_with_mock_client([])
        assert model.model_name == "test-model"
        assert model.provider == "tinker"
        assert model.tokenizer is not None


# ---------------------------------------------------------------------------
# _score_single_sequence
# ---------------------------------------------------------------------------


class TestScoreSingleSequence:
    def test_parses_logprobs_correctly(self) -> None:
        """Verify correct shift-by-1 and nats→bits conversion."""
        raw = [None, -1.5, -2.0]
        model = _make_tinker_model_with_mock_client([raw])

        result = model._score_single_sequence([10, 20, 30])

        assert len(result) == 3
        assert result[0] == pytest.approx(-1.5 / math.log(2))
        assert result[1] == pytest.approx(-2.0 / math.log(2))
        assert result[2] == 0.0

    def test_single_token(self) -> None:
        raw = [None]
        model = _make_tinker_model_with_mock_client([raw])

        result = model._score_single_sequence([42])
        assert len(result) == 1
        assert result[0] == 0.0

    def test_empty_sequence(self) -> None:
        model = _make_tinker_model_with_mock_client([])
        result = model._score_single_sequence([])
        assert result == []

    def test_unexpected_none_raises(self) -> None:
        """None at position > 0 should raise."""
        raw = [None, None, -2.0]
        model = _make_tinker_model_with_mock_client([raw])

        with pytest.raises(RuntimeError, match="Unexpected None logprob"):
            model._score_single_sequence([10, 20, 30])

    def test_token_count_mismatch_raises(self) -> None:
        raw = [None, -1.0]
        model = _make_tinker_model_with_mock_client([raw])

        with pytest.raises(RuntimeError, match="Token count mismatch"):
            model._score_single_sequence([10, 20, 30])

    def test_retry_on_error(self) -> None:
        model = _make_tinker_model_with_mock_client([])

        success_future = MagicMock()
        success_future.result.return_value = [None, -1.0]
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("rate limit")
        model._sampling_client.compute_logprobs.side_effect = [
            fail_future,
            success_future,
        ]

        result = model._score_single_sequence([10, 20])
        assert len(result) == 2
        assert model._sampling_client.compute_logprobs.call_count == 2


# ---------------------------------------------------------------------------
# score_sequences (batch)
# ---------------------------------------------------------------------------


class TestScoreSequences:
    def test_batch_scoring(self) -> None:
        """score_sequences handles a batch of sequences."""
        raw = [None, -1.0, -2.0]
        model = _make_tinker_model_with_mock_client([raw, raw])

        input_ids = torch.tensor([[10, 20, 30], [10, 20, 30]])
        result = model.score_sequences(input_ids)

        assert result.shape == (2, 3)
        torch.testing.assert_close(result[0], result[1])

    def test_with_attention_mask(self) -> None:
        """Padding positions produce 0.0."""
        raw = [None, -1.0]
        model = _make_tinker_model_with_mock_client([raw])

        input_ids = torch.tensor([[10, 20, 0]])
        attention_mask = torch.tensor([[1, 1, 0]])
        result = model.score_sequences(input_ids, attention_mask)

        assert result.shape == (1, 3)
        assert result[0, 2].item() == 0.0
        assert result[0, 0].item() == pytest.approx(-1.0 / math.log(2))
        assert result[0, 1].item() == 0.0  # last real token → 0.0


# ---------------------------------------------------------------------------
# Full pipeline with mocked TinkerModel
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_tinker_model_through_compute_icl_diversity_metrics(self) -> None:
        """A mocked TinkerModel should produce valid metrics through the full pipeline."""
        from icl_diversity import compute_icl_diversity_metrics

        mock_tok = MagicMock()
        mock_tok.name_or_path = "mock-model"
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 0

        def mock_encode(text: str, add_special_tokens: bool = True) -> list[int]:
            return [ord(c) for c in text]

        def mock_decode(ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

        mock_tok.encode = mock_encode
        mock_tok.decode = mock_decode

        model = _make_tinker_model_with_mock_client([])
        model._tokenizer = mock_tok

        def mock_score_sequences(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, seq_len = input_ids.shape
            if seq_len == 0:
                return torch.zeros(batch, 0)
            result = torch.full((batch, seq_len), -1.0)
            result[:, -1] = 0.0
            return result

        model.score_sequences = mock_score_sequences  # type: ignore[assignment]

        prompt = "Q"
        responses = ["AB", "CD", "EF"]

        metrics = compute_icl_diversity_metrics(
            model=model,
            tokenizer=None,
            prompt=prompt,
            responses=responses,
            n_permutations=1,
        )

        assert "excess_entropy_E" in metrics
        assert "coherence_C" in metrics
        assert "diversity_score_D" in metrics
        assert "a_k_curve" in metrics
        assert len(metrics["a_k_curve"]) == 3
        assert metrics["excess_entropy_E"] >= 0.0
