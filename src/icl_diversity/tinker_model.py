"""Tinker API model support for ICL diversity metric computation.

Uses Tinker's ``compute_logprobs`` endpoint to score sequences against base
models hosted by Thinking Machines. This returns per-token log-probabilities
which is exactly what the ICL diversity metric needs.

**Numerical note:** Tinker produces slightly different logprobs from
HuggingFace transformers running the same model weights locally (~0.02 bits
mean per-token difference, outliers up to ~0.35 bits at high-surprise
positions). The cause is unknown. It is consistent across all local dtypes
(float32/16/bf16), devices (CPU/GPU), and attention implementations
(eager/SDPA). Our shift logic, nats-to-bits conversion, and padding are
verified correct. See ``scripts/compare_tinker_local.py`` for visual
validation and ``tests/test_tinker_live.py`` for cross-validation tests.

*Speculation:* Thinking Machines uses custom batch-invariant matmul kernels
(see `their blog post
<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>`_)
which use fixed tile sizes instead of cuBLAS. Different matmul algorithms
accumulate floating-point products in different orders, producing slightly
different results even with identical weights and dtype. We have not confirmed
this is the cause.

Usage::

    from icl_diversity.tinker_model import TinkerModel

    model = TinkerModel("meta-llama/Llama-3.1-8B")
    # Use like a local model in compute_icl_diversity_metrics
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import PreTrainedTokenizerBase


class TinkerModel:
    """Wraps a Tinker-hosted base model for log-prob scoring.

    Compatible with the ``ModelInput`` type in ``core.py``. The
    ``score_sequences`` method returns per-position next-token log-probs
    in the same format as ``_forward_log_probs``.

    Args:
        model_name: Base model identifier on Tinker (e.g.
            ``"meta-llama/Llama-3.1-8B"``).
        api_key: Tinker API key. Defaults to ``TINKER_API_KEY`` env var
            (loaded from ``.env`` via ``python-dotenv``).
        max_concurrent_requests: Max parallel ``compute_logprobs`` calls
            within a batch.
        retry_attempts: Number of retries on transient errors.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_concurrent_requests: int = 50,
        retry_attempts: int = 3,
    ) -> None:
        from dotenv import load_dotenv

        load_dotenv()

        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package is required for the Tinker provider. "
                "Install with: uv add tinker"
            ) from None

        resolved_key = api_key or os.environ.get("TINKER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set TINKER_API_KEY in your .env file "
                "or pass api_key= to TinkerModel."
            )

        # Set the env var so the tinker SDK picks it up
        os.environ["TINKER_API_KEY"] = resolved_key

        self._model_name = model_name
        self._max_concurrent = max_concurrent_requests
        self._retry_attempts = retry_attempts
        self._tinker = tinker

        service_client = tinker.ServiceClient()
        self._sampling_client = service_client.create_sampling_client(
            base_model=model_name
        )
        self._tokenizer: PreTrainedTokenizerBase = (
            self._sampling_client.get_tokenizer()
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "tinker"

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _score_single_sequence(self, token_ids: list[int]) -> list[float]:
        """Score a single token sequence via Tinker, returning log-probs in bits.

        Returns a list of length ``len(token_ids)`` where entry ``[t]`` =
        ``log2 P(token[t+1] | token[0..t])``. The last entry is 0.0.

        Raises:
            RuntimeError: If the API call fails after all retry attempts.
        """
        n = len(token_ids)
        if n == 0:
            return []

        prompt = self._tinker.types.ModelInput.from_ints(tokens=token_ids)

        last_error = None
        for attempt in range(self._retry_attempts):
            try:
                raw_logprobs = self._sampling_client.compute_logprobs(
                    prompt
                ).result()
                break
            except Exception as e:
                last_error = e
                if attempt < self._retry_attempts - 1:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(
                f"Tinker API call failed after {self._retry_attempts} "
                f"attempts: {last_error}"
            ) from last_error

        if len(raw_logprobs) != n:
            raise RuntimeError(
                f"Token count mismatch: sent {n} tokens but Tinker returned "
                f"{len(raw_logprobs)} logprobs."
            )

        # raw_logprobs[t] = log P(token[t] | token[0..t-1]) in nats.
        # raw_logprobs[0] is None (no conditioning for first token).
        # We want result[t] = log2 P(token[t+1] | token[0..t]).
        # So result[t] = raw_logprobs[t+1] / ln(2) for t=0..n-2,
        # and result[n-1] = 0.0 (no next token).
        result: list[float] = []
        for t in range(n - 1):
            entry = raw_logprobs[t + 1]
            if entry is None:
                raise RuntimeError(
                    f"Unexpected None logprob at position {t + 1} "
                    f"(only position 0 should be None)."
                )
            result.append(entry / math.log(2))
        result.append(0.0)  # last position padding

        return result

    def score_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score sequences via Tinker, returning per-position next-token log-probs.

        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            attention_mask: ``(batch, seq_len)`` mask (1=real, 0=pad).

        Returns:
            ``(batch, seq_len)`` tensor where entry ``[b, t]`` =
            ``log2 P(input_ids[b, t+1] | input_ids[b, 0..t])`` in **bits**.
            Last position is 0.0. Padding positions are 0.0.
        """
        batch_size, seq_len = input_ids.shape
        result = torch.zeros(batch_size, seq_len)

        # Extract real token IDs for each sequence (skip padding)
        token_id_lists: list[list[int]] = []
        for b in range(batch_size):
            if attention_mask is not None:
                real_len = int(attention_mask[b].sum().item())
                real_ids = input_ids[b, :real_len].tolist()
            else:
                real_ids = input_ids[b].tolist()
            token_id_lists.append(real_ids)

        # Parallel API calls
        def _score_idx(idx: int) -> tuple[int, list[float]]:
            return idx, self._score_single_sequence(token_id_lists[idx])

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = [pool.submit(_score_idx, i) for i in range(batch_size)]
            for future in as_completed(futures):
                idx, log_probs = future.result()
                n = len(log_probs)
                if n > 0:
                    result[idx, :n] = torch.tensor(log_probs)

        return result
