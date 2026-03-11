"""API-based model support for ICL diversity metric computation.

Supports providers that expose ``prompt_logprobs`` on their OpenAI-compatible
``/v1/completions`` endpoint (Together AI, Fireworks AI). This returns
``log P(token_t | context)`` for every prompt token, which is exactly what the
ICL diversity metric needs.

Usage::

    from icl_diversity.api_model import APIModel

    model = APIModel("meta-llama/Llama-3.1-8B", provider="together")
    # Use like a local model in compute_icl_diversity_metrics
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_var": "TOGETHER_API_KEY",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_var": "FIREWORKS_API_KEY",
    },
}


class APIModel:
    """Wraps an API-hosted model for log-prob scoring.

    Compatible with the ``ModelInput`` type in ``core.py``. The
    ``score_sequences`` method returns per-position next-token log-probs
    in the same format as ``_forward_log_probs``.

    Args:
        model_name: Model identifier on the provider (e.g.
            ``"meta-llama/Llama-3.1-8B"``).
        provider: API provider — ``"together"`` or ``"fireworks"``.
        api_key: API key. Defaults to the provider's environment variable.
        tokenizer: HuggingFace tokenizer. Auto-loaded from ``model_name``
            if not provided.
        max_concurrent_requests: Max parallel API calls within a batch.
        retry_attempts: Number of retries on transient errors.
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "together",
        api_key: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_concurrent_requests: int = 5,
        retry_attempts: int = 3,
    ) -> None:
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Supported: {list(PROVIDER_CONFIG.keys())}"
            )

        config = PROVIDER_CONFIG[provider]
        resolved_key = api_key or os.environ.get(config["env_var"])
        if not resolved_key:
            raise ValueError(
                f"No API key provided. Set {config['env_var']} environment "
                f"variable or pass api_key= to APIModel."
            )

        self._model_name = model_name
        self._provider = provider
        self._max_concurrent = max_concurrent_requests
        self._retry_attempts = retry_attempts

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=config["base_url"],
        )

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _score_single_text(self, text: str) -> list[float]:
        """Score a single text via API, returning per-token log-probs in bits.

        Returns a list of length ``n_tokens`` where entry ``[t]`` =
        ``log2 P(token[t+1] | token[0..t])``. The last entry is 0.0.

        Raises:
            RuntimeError: If the API returns a different token count than
                the local tokenizer.
        """
        # Local tokenization for validation
        local_ids = self._tokenizer.encode(text, add_special_tokens=False)
        n_local = len(local_ids)

        if n_local == 0:
            return []

        last_error = None
        for attempt in range(self._retry_attempts):
            try:
                response = self._client.completions.create(
                    model=self._model_name,
                    prompt=text,
                    max_tokens=0,
                    extra_body={"prompt_logprobs": 1},
                )
                break
            except Exception as e:
                last_error = e
                if attempt < self._retry_attempts - 1:
                    import time

                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                f"API call failed after {self._retry_attempts} attempts: {last_error}"
            ) from last_error

        # Parse prompt_logprobs from response
        # Together AI returns prompt_logprobs as a list of dicts, one per token.
        # The first token has no logprob (it's the start), subsequent tokens
        # have their conditional logprob.
        prompt_logprobs = response.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError(
                "API did not return prompt_logprobs. Ensure the model "
                f"({self._model_name}) supports prompt_logprobs on {self._provider}."
            )

        # prompt_logprobs is a list of length n_tokens. Each entry is a dict
        # or None. The first entry is None (no conditioning for first token).
        # Subsequent entries have the logprob for that token.
        n_api = len(prompt_logprobs)
        if n_api != n_local:
            raise RuntimeError(
                f"Token count mismatch: local tokenizer produced {n_local} "
                f"tokens but API returned {n_api} prompt_logprobs. This "
                f"indicates a tokenization disagreement between the local "
                f"tokenizer ({self._tokenizer.name_or_path}) and the API "
                f"model ({self._model_name})."
            )

        # Build per-position next-token log-probs in bits.
        # prompt_logprobs[t] gives log P(token[t] | token[0..t-1]) in nats.
        # We want log_probs[t] = log2 P(token[t+1] | token[0..t]).
        # So log_probs[t] = prompt_logprobs[t+1] / ln(2) for t=0..n-2,
        # and log_probs[n-1] = 0.0 (no next token).
        result: list[float] = []
        for t in range(n_local - 1):
            entry = prompt_logprobs[t + 1]
            if entry is None:
                # Shouldn't happen for t+1 > 0, but be safe
                result.append(0.0)
            else:
                # Together AI format: each entry is a dict with token → logprob
                # The entry for the actual token is the one we want
                logprob_nats = list(entry.values())[0]
                result.append(logprob_nats / math.log(2))
        result.append(0.0)  # last position padding

        return result

    def score_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score sequences via API, returning per-position next-token log-probs.

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

        # Decode each sequence (skipping padding)
        texts: list[str] = []
        real_lengths: list[int] = []
        for b in range(batch_size):
            if attention_mask is not None:
                mask = attention_mask[b]
                real_len = mask.sum().item()
                real_ids = input_ids[b, :real_len].tolist()
            else:
                real_ids = input_ids[b].tolist()
                real_len = len(real_ids)
            real_lengths.append(real_len)
            texts.append(self._tokenizer.decode(real_ids))

        # Parallel API calls
        def _score_idx(idx: int) -> tuple[int, list[float]]:
            return idx, self._score_single_text(texts[idx])

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = [pool.submit(_score_idx, i) for i in range(batch_size)]
            for future in as_completed(futures):
                idx, log_probs = future.result()
                n = len(log_probs)
                if n > 0:
                    result[idx, :n] = torch.tensor(log_probs)

        return result
