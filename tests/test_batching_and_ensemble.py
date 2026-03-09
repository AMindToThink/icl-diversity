"""
Tests for batching and model ensembling features.

- Batching: verify that batch_size > 1 produces identical results to batch_size=1.
- Ensembling: verify that a single-model "ensemble" matches single-model results,
  and that a two-model ensemble averages softmax probabilities correctly.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from icl_diversity import (
    compute_cross_entropy,
    compute_icl_diversity_metrics,
    compute_per_byte_cross_entropy,
    compute_progressive_surprise_curve,
    compute_progressive_surprise_curve_single_pass,
    compute_unconditional_surprises,
)
from icl_diversity.core import (
    _ensure_models,
    _forward_log_probs,
    _get_pad_token_id,
    _right_pad_and_batch,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mock_model_and_tokenizer(
    vocab_size: int = 100,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock model/tokenizer pair for unit tests.

    The mock model simulates a causal LM: logits at each position are a
    deterministic function of the *actual* (non-padding) tokens that precede
    it.  Specifically, a cumulative hash of input token IDs is used as a
    per-position seed so that:

    - logits vary across positions (position-sensitive),
    - padding tokens are excluded via attention_mask, and
    - the same prefix always produces the same logits (deterministic).

    This means extracting log-probs from wrong positions (e.g. due to an
    incorrect padding offset) will produce detectably different results.
    """
    model = MagicMock()
    model.device = torch.device("cpu")

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def encode_fn(text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(2, 2 + len(text)))

    tokenizer.encode = MagicMock(side_effect=encode_fn)

    def model_forward(input_ids: torch.Tensor, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        for b in range(batch_size):
            # Build a cumulative hash over real tokens to simulate causal
            # dependence on context.  Padding tokens (mask=0) are excluded
            # so that right-padded batches produce the same logits for real
            # positions as unbatched sequences.
            h = 0
            for t in range(seq_len):
                is_real = (
                    attention_mask[b, t].item() if attention_mask is not None else 1
                )
                if is_real:
                    h = hash((h, input_ids[b, t].item())) & 0xFFFFFFFF
                # Deterministic per-position logits seeded by context hash
                gen = torch.Generator()
                gen.manual_seed(h)
                logits[b, t] = torch.randn(vocab_size, generator=gen)

        result = MagicMock()
        result.logits = logits
        return result

    model.side_effect = model_forward
    return model, tokenizer


# ---------------------------------------------------------------------------
# _ensure_models
# ---------------------------------------------------------------------------


class TestEnsureModels:
    def test_single_model(self) -> None:
        model = MagicMock()
        result = _ensure_models(model)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is model

    def test_list_of_models(self) -> None:
        models = [MagicMock(), MagicMock()]
        result = _ensure_models(models)
        assert result is models


# ---------------------------------------------------------------------------
# _get_pad_token_id
# ---------------------------------------------------------------------------


class TestGetPadTokenId:
    def test_uses_pad_token(self) -> None:
        tok = MagicMock()
        tok.pad_token_id = 42
        assert _get_pad_token_id(tok) == 42

    def test_falls_back_to_eos(self) -> None:
        tok = MagicMock()
        tok.pad_token_id = None
        tok.eos_token_id = 99
        assert _get_pad_token_id(tok) == 99


# ---------------------------------------------------------------------------
# _right_pad_and_batch
# ---------------------------------------------------------------------------


class TestRightPadAndBatch:
    def test_uniform_length(self) -> None:
        seqs = [[1, 2, 3], [4, 5, 6]]
        ids, mask = _right_pad_and_batch(seqs, pad_token_id=0)
        assert ids.shape == (2, 3)
        assert mask is None  # no padding needed

    def test_variable_length(self) -> None:
        seqs = [[1, 2], [3, 4, 5, 6]]
        ids, mask = _right_pad_and_batch(seqs, pad_token_id=0)
        assert ids.shape == (2, 4)
        assert mask is not None
        assert ids[0].tolist() == [1, 2, 0, 0]
        assert mask[0].tolist() == [1, 1, 0, 0]
        assert ids[1].tolist() == [3, 4, 5, 6]
        assert mask[1].tolist() == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# _forward_log_probs
# ---------------------------------------------------------------------------


class TestForwardLogProbs:
    def test_single_model(self) -> None:
        model, _ = _make_mock_model_and_tokenizer(vocab_size=50)
        input_ids = torch.tensor([[2, 3, 4]])
        log_probs = _forward_log_probs([model], input_ids)
        assert log_probs.shape == (1, 3, 50)
        # log_softmax values should sum to 0 in exp space (softmax sums to 1)
        assert torch.allclose(log_probs[0, 0].exp().sum(), torch.tensor(1.0), atol=1e-5)

    def test_ensemble_averages_probs(self) -> None:
        """Two identical models ensembled should give same result as one."""
        model, _ = _make_mock_model_and_tokenizer(vocab_size=50)
        input_ids = torch.tensor([[2, 3, 4]])

        single = _forward_log_probs([model], input_ids)
        ensemble = _forward_log_probs([model, model], input_ids)

        # Ensemble of identical models = same as single model
        np.testing.assert_allclose(ensemble.numpy(), single.cpu().numpy(), atol=1e-5)

    def test_ensemble_with_attention_mask(self) -> None:
        model, _ = _make_mock_model_and_tokenizer(vocab_size=50)
        input_ids = torch.tensor([[0, 2, 3]])
        mask = torch.tensor([[0, 1, 1]])
        log_probs = _forward_log_probs([model, model], input_ids, mask)
        assert log_probs.shape == (1, 3, 50)


# ---------------------------------------------------------------------------
# Batched unconditional surprises
# ---------------------------------------------------------------------------


class TestBatchedUnconditionalSurprises:
    def test_batch_equals_sequential(self) -> None:
        """batch_size=N should give identical results to batch_size=1."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello", "world", "foo", "bar"]

        seq_pb, seq_tb, seq_bc = compute_unconditional_surprises(
            model, tokenizer, prompt, responses, batch_size=1
        )
        batch_pb, batch_tb, batch_bc = compute_unconditional_surprises(
            model, tokenizer, prompt, responses, batch_size=2
        )
        full_batch_pb, full_batch_tb, full_batch_bc = compute_unconditional_surprises(
            model, tokenizer, prompt, responses, batch_size=10
        )

        assert seq_bc == batch_bc == full_batch_bc
        np.testing.assert_allclose(batch_tb, seq_tb, atol=1e-5)
        np.testing.assert_allclose(batch_pb, seq_pb, atol=1e-5)
        np.testing.assert_allclose(full_batch_tb, seq_tb, atol=1e-5)
        np.testing.assert_allclose(full_batch_pb, seq_pb, atol=1e-5)

    def test_empty_response_in_batch(self) -> None:
        """Empty responses should produce 0 bits, 0 bytes."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        responses = ["hello", "", "world"]
        pb, tb, bc = compute_unconditional_surprises(
            model, tokenizer, "p", responses, batch_size=3
        )
        assert bc[1] == 0
        assert tb[1] == 0.0
        assert pb[1] == 0.0
        # Others should be non-zero
        assert tb[0] > 0
        assert tb[2] > 0


# ---------------------------------------------------------------------------
# Batched full pipeline
# ---------------------------------------------------------------------------


class TestBatchedPipeline:
    def test_batch_size_does_not_change_results(self) -> None:
        """Full pipeline with batch_size > 1 should match batch_size=1."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello world", "foo bar", "baz qux"]

        r1 = compute_icl_diversity_metrics(
            model,
            tokenizer,
            prompt,
            responses,
            n_permutations=3,
            seed=42,
            batch_size=1,
        )
        r3 = compute_icl_diversity_metrics(
            model,
            tokenizer,
            prompt,
            responses,
            n_permutations=3,
            seed=42,
            batch_size=3,
        )

        # All scalar metrics should match
        for key in [
            "excess_entropy_E",
            "excess_entropy_E_rate",
            "coherence_C",
            "diversity_score_D",
            "diversity_score_D_rate",
            "coherence_spread_sigma",
        ]:
            assert r1[key] == pytest.approx(r3[key], abs=1e-5), (
                f"{key}: {r1[key]} != {r3[key]}"
            )

        # Curves should match
        np.testing.assert_allclose(r1["a_k_curve"], r3["a_k_curve"], atol=1e-5)
        assert r1["a_k_byte_counts"] == r3["a_k_byte_counts"]
        assert r1["permutation_orders"] == r3["permutation_orders"]

    def test_single_permutation_batched(self) -> None:
        """Single permutation with batch_size>1 should work."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello", "world"]

        r = compute_icl_diversity_metrics(
            model,
            tokenizer,
            prompt,
            responses,
            n_permutations=1,
            batch_size=2,
        )
        assert r["per_permutation_a_k_curves"] is None
        assert len(r["a_k_curve"]) == 2


# ---------------------------------------------------------------------------
# Ensemble support
# ---------------------------------------------------------------------------


class TestEnsembleSupport:
    def test_single_model_list_matches_plain(self) -> None:
        """Passing [model] should give same result as passing model."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello world", "foo bar"]

        r_plain = compute_icl_diversity_metrics(
            model, tokenizer, prompt, responses, seed=42
        )
        r_list = compute_icl_diversity_metrics(
            [model], tokenizer, prompt, responses, seed=42
        )

        for key in ["excess_entropy_E", "coherence_C", "diversity_score_D"]:
            assert r_plain[key] == pytest.approx(r_list[key], abs=1e-6)

    def test_identical_ensemble_matches_single(self) -> None:
        """Ensemble of two identical models should match single model."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello world", "foo bar"]

        r_single = compute_icl_diversity_metrics(
            model, tokenizer, prompt, responses, seed=42
        )
        r_ensemble = compute_icl_diversity_metrics(
            [model, model], tokenizer, prompt, responses, seed=42
        )

        for key in [
            "excess_entropy_E",
            "excess_entropy_E_rate",
            "coherence_C",
            "diversity_score_D",
            "diversity_score_D_rate",
        ]:
            assert r_single[key] == pytest.approx(r_ensemble[key], abs=1e-4), (
                f"{key}: single={r_single[key]}, ensemble={r_ensemble[key]}"
            )

    def test_ensemble_with_permutations(self) -> None:
        """Ensemble should work with multiple permutations."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello", "world", "foo"]

        r = compute_icl_diversity_metrics(
            [model, model],
            tokenizer,
            prompt,
            responses,
            n_permutations=3,
            seed=42,
        )
        assert r["per_permutation_a_k_curves"] is not None
        assert len(r["per_permutation_a_k_curves"]) == 3
        assert len(r["a_k_curve"]) == 3

    def test_ensemble_cross_entropy(self) -> None:
        """compute_cross_entropy should accept model list."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        tb_single, bc_single = compute_cross_entropy(
            model, tokenizer, "hello", "prefix "
        )
        tb_ensemble, bc_ensemble = compute_cross_entropy(
            [model, model], tokenizer, "hello", "prefix "
        )
        assert bc_single == bc_ensemble
        assert tb_single == pytest.approx(tb_ensemble, abs=1e-4)

    def test_ensemble_per_byte_cross_entropy(self) -> None:
        """compute_per_byte_cross_entropy should accept model list."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        pb_single = compute_per_byte_cross_entropy(model, tokenizer, "hello", "prefix ")
        pb_ensemble = compute_per_byte_cross_entropy(
            [model, model], tokenizer, "hello", "prefix "
        )
        assert pb_single == pytest.approx(pb_ensemble, abs=1e-4)

    def test_ensemble_progressive_curve(self) -> None:
        """Both progressive curve functions should accept model list."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        responses = ["hello", "world"]

        c1, bc1 = compute_progressive_surprise_curve(
            [model], tokenizer, "test", responses
        )
        c2, bc2 = compute_progressive_surprise_curve_single_pass(
            [model], tokenizer, "test", responses
        )
        assert len(c1) == 2
        assert len(c2) == 2
        assert bc1 == bc2


# ---------------------------------------------------------------------------
# Ensemble + batching combined
# ---------------------------------------------------------------------------


class TestEnsembleBatching:
    def test_ensemble_with_batching(self) -> None:
        """Ensemble + batching should work together."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        prompt = "test"
        responses = ["hello world", "foo bar", "baz qux"]

        r_seq = compute_icl_diversity_metrics(
            [model, model],
            tokenizer,
            prompt,
            responses,
            n_permutations=3,
            seed=42,
            batch_size=1,
        )
        r_batch = compute_icl_diversity_metrics(
            [model, model],
            tokenizer,
            prompt,
            responses,
            n_permutations=3,
            seed=42,
            batch_size=3,
        )

        for key in ["excess_entropy_E", "coherence_C", "diversity_score_D"]:
            assert r_seq[key] == pytest.approx(r_batch[key], abs=1e-5)

    def test_batched_unconditional_ensemble(self) -> None:
        """Batched unconditional surprises with ensemble."""
        model, tokenizer = _make_mock_model_and_tokenizer(vocab_size=100)
        responses = ["hello", "world", "foo", "bar"]

        seq_pb, seq_tb, seq_bc = compute_unconditional_surprises(
            [model, model], tokenizer, "prompt", responses, batch_size=1
        )
        batch_pb, batch_tb, batch_bc = compute_unconditional_surprises(
            [model, model], tokenizer, "prompt", responses, batch_size=4
        )

        assert seq_bc == batch_bc
        np.testing.assert_allclose(batch_tb, seq_tb, atol=1e-5)


# ---------------------------------------------------------------------------
# Ensemble probability averaging (mathematical correctness)
# ---------------------------------------------------------------------------


class TestEnsembleProbabilityAveraging:
    """Verify that ensemble averages softmax probs, not logits."""

    def test_mixture_probabilities(self) -> None:
        """The ensemble log-prob should be log of averaged softmax probs."""
        vocab_size = 10

        # Model A: peaked at token 0
        model_a = MagicMock()
        model_a.device = torch.device("cpu")

        def forward_a(input_ids, attention_mask=None, **kwargs):
            b, s = input_ids.shape
            logits = torch.zeros(b, s, vocab_size)
            logits[:, :, 0] = 5.0  # peaked at token 0
            result = MagicMock()
            result.logits = logits
            return result

        model_a.side_effect = forward_a

        # Model B: peaked at token 1
        model_b = MagicMock()
        model_b.device = torch.device("cpu")

        def forward_b(input_ids, attention_mask=None, **kwargs):
            b, s = input_ids.shape
            logits = torch.zeros(b, s, vocab_size)
            logits[:, :, 1] = 5.0  # peaked at token 1
            result = MagicMock()
            result.logits = logits
            return result

        model_b.side_effect = forward_b

        input_ids = torch.tensor([[2, 3]])

        # Single model log-probs
        lp_a = _forward_log_probs([model_a], input_ids)
        lp_b = _forward_log_probs([model_b], input_ids)

        # Ensemble log-probs
        lp_ens = _forward_log_probs([model_a, model_b], input_ids)

        # The mixture should have: p_ens(token) = 0.5 * p_a(token) + 0.5 * p_b(token)
        expected_probs = 0.5 * lp_a.exp().cpu() + 0.5 * lp_b.exp().cpu()
        expected_log_probs = torch.log(expected_probs)

        np.testing.assert_allclose(
            lp_ens.numpy(),
            expected_log_probs.numpy(),
            atol=1e-5,
        )

        # The mixture should assign reasonable prob to both token 0 and token 1
        ens_probs = lp_ens[0, 0].exp()
        assert ens_probs[0].item() > 0.4  # model A's contribution
        assert ens_probs[1].item() > 0.4  # model B's contribution
