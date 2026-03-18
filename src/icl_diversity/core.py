"""
ICL (In-Context Learning) Diversity Metric.

Measures diversity of LLM outputs using a trusted base model's token-level
log-probabilities via progressive conditional surprise. The metric is described
in detail in `in_context_diversity_metric.tex`.

Key idea: if a base model theta can predict response r_k more easily after
seeing responses r_1..r_{k-1}, then those responses share structure (low
diversity). The progressive conditional surprise curve a_k captures this.

The primary a_k curve is in total bits. Per-byte normalized quantities
(E_rate, C, D) provide tokenizer-agnostic comparisons. Uses log base 2
throughout.

Supports:
- **Batching**: Forward passes for unconditional surprises and permutations
  are batched for GPU parallelism (controlled via ``batch_size``).
- **Model ensembling** (Section 7.5): Multiple base models can be ensembled
  at the token level by averaging softmax probabilities. All models must
  share the same tokenizer/vocabulary.

Assumptions:
- theta must have strong in-context learning capability
- theta should be a BASE model (not instruction-tuned) to avoid confounding
  coherence-as-fluency with coherence-as-alignment
- The same theta must be used across all policies being compared

Reference equations from the paper:
- Eq 1: per-byte cross-entropy h_theta(r | p)
- Eq 4: progressive conditional surprise a_k (total bits)
- Eq 6: excess entropy E = sum(a_k - a_n) in total bits
- Eq: excess entropy rate E_rate = sum of per-byte normalized excess
- Eq 8: coherence C = 2^{-(1/n) * sum(h_theta(r_i | p))} (per-byte)
- Eq 12: diversity score D = C * E (bits)
- Eq 13: diversity score rate D_rate = C * E_rate (bits/byte)
- Eq 27: ensemble theta_bar = (1/M) * sum_j theta_j (token-level mixture)
"""

from __future__ import annotations

import math
import random
import string
from typing import Any, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Type alias: single model, list of models (ensemble), or API model.
# APIModel is referenced by string to avoid circular import.
ModelInput = Union[PreTrainedModel, list[PreTrainedModel], "APIModel"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_api_model(model: ModelInput) -> bool:
    """Check if model is an APIModel (without importing at module level)."""
    return type(model).__name__ == "APIModel"


def _ensure_models(model: ModelInput) -> list[PreTrainedModel] | "APIModel":
    """Normalize model input to a list of local models, or return APIModel as-is."""
    if _is_api_model(model):
        return model  # type: ignore[return-value]
    if isinstance(model, list):
        return model
    return [model]


def _get_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return pad token id, falling back to eos_token_id."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def _gather_diagonal_log_probs(
    log_probs_full: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract next-token log-probs along the diagonal from full log-probs.

    Args:
        log_probs_full: ``(batch, seq_len, vocab_size)`` full log-probs (nats).
        input_ids: ``(batch, seq_len)`` token IDs.

    Returns:
        ``(batch, seq_len)`` tensor where entry ``[b, t]`` =
        ``log2 P(input_ids[b, t+1] | input_ids[b, 0..t])``.
        The last position is 0.0 (no next token to predict).
        Values are in **bits** (log base 2), not nats.
    """
    batch, seq_len = input_ids.shape
    # Shift: log_probs_full[:, t, :] predicts token at position t+1
    # So we gather input_ids[:, 1:] from log_probs_full[:, :-1, :]
    if seq_len <= 1:
        return torch.zeros(batch, seq_len, device=input_ids.device)

    next_token_ids = input_ids[:, 1:]  # (batch, seq_len-1)
    # Gather the log-prob of each next token
    gathered = log_probs_full[:, :-1, :].gather(
        2, next_token_ids.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)

    # Convert nats → bits: divide by ln(2)
    gathered = gathered / math.log(2)

    # Pad with 0.0 at the end (last position has no next token)
    pad = torch.zeros(batch, 1, device=gathered.device, dtype=gathered.dtype)
    return torch.cat([gathered, pad], dim=1)


def _forward_log_probs(
    models: list[PreTrainedModel] | "APIModel",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Run forward pass and return per-position next-token log-probs in bits.

    For a single model, computes log_softmax(logits) then extracts the
    diagonal. For multiple models, averages softmax probabilities at each
    token position (Section 7.5, Eq 27), then extracts the diagonal.
    For an APIModel, delegates to ``APIModel.score_sequences``.

    Args:
        models: List of models to ensemble, a single-element list, or an
            APIModel instance.
        input_ids: ``(batch, seq_len)`` token IDs.
        attention_mask: ``(batch, seq_len)`` mask (1=real, 0=pad). Optional.
        temperature: Temperature for scaling logits before softmax. Must be
            positive. T>1 flattens predictions (reduces variance), T<1
            sharpens predictions (amplifies signal). Default 1.0 (no scaling).

    Returns:
        ``(batch, seq_len)`` tensor where entry ``[b, t]`` =
        ``log2 P(input_ids[b, t+1] | input_ids[b, 0..t])`` in **bits**.
        Last position is 0.0. On the model's device for single model, on
        CPU for ensemble/API.

    Raises:
        ValueError: If temperature <= 0 or temperature != 1.0 for API models.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # API model dispatch
    if _is_api_model(models):
        if temperature != 1.0:
            raise ValueError(
                f"temperature scaling is not supported for API models "
                f"(no logits available), got temperature={temperature}"
            )
        return models.score_sequences(input_ids, attention_mask)  # type: ignore[union-attr]

    if len(models) == 1:
        model = models[0]
        ids = input_ids.to(model.device)
        mask = attention_mask.to(model.device) if attention_mask is not None else None
        with torch.no_grad():
            logits = model(ids, attention_mask=mask, use_cache=False).logits
            logits = logits / temperature
        log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
        return _gather_diagonal_log_probs(log_probs_full, ids)

    # Ensemble: average softmax probabilities across models (Eq 27)
    # Temperature is applied per-model before softmax, then probabilities averaged.
    accumulated_probs: torch.Tensor | None = None
    for model in models:
        ids = input_ids.to(model.device)
        mask = attention_mask.to(model.device) if attention_mask is not None else None
        with torch.no_grad():
            logits = model(ids, attention_mask=mask, use_cache=False).logits
            logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        if accumulated_probs is None:
            accumulated_probs = probs
        else:
            accumulated_probs = accumulated_probs + probs
        del logits, probs
    assert accumulated_probs is not None
    accumulated_probs /= len(models)
    log_probs_full = torch.log(accumulated_probs.clamp(min=1e-45))
    return _gather_diagonal_log_probs(log_probs_full, input_ids)


def _find_response_boundaries(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Find token boundaries for each response in a concatenated context.

    Uses the tokenizer's character offset mapping to find boundaries, which
    correctly handles BPE merges at response/separator boundaries (e.g. Qwen
    merging ``"."`` + ``"\\n\\n"`` into a single token ``".\\n\\n"``).

    Returns:
        ``(full_ids, boundaries)`` where ``full_ids`` is the tokenized full
        concatenation and ``boundaries[k] = (start, end)`` gives the token
        range for ``responses[k]``.
    """
    n = len(responses)
    parts = [prompt]
    for i, resp in enumerate(responses):
        parts.append(f"\n\nResponse {_response_label(i)}: {resp}")
    full_text = "".join(parts)

    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    full_ids = encoding["input_ids"]
    offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]

    # Compute character spans for each response in full_text
    char_spans: list[tuple[int, int]] = []
    cursor = len(prompt)
    for k in range(n):
        label_prefix = f"\n\nResponse {_response_label(k)}: "
        cursor += len(label_prefix)
        char_start = cursor
        cursor += len(responses[k])
        char_spans.append((char_start, cursor))

    assert cursor == len(full_text), (
        f"Character span computation mismatch: cursor={cursor}, "
        f"full_text length={len(full_text)}"
    )

    # Map character spans to token index ranges. A token belongs to response k
    # if its start character falls within the response's character span.
    boundaries: list[tuple[int, int]] = []
    for char_start, char_end in char_spans:
        tok_start = None
        tok_end = None
        for t, (c_start, c_end) in enumerate(offset_mapping):
            if c_start >= char_start and c_start < char_end:
                if tok_start is None:
                    tok_start = t
                tok_end = t + 1
        if tok_start is None:
            prev_end = boundaries[-1][1] if boundaries else 0
            boundaries.append((prev_end, prev_end))
        else:
            boundaries.append((tok_start, tok_end))

    return full_ids, boundaries


def _extract_response_log_probs(
    log_probs: torch.Tensor,
    boundaries: list[tuple[int, int]],
    responses: list[str],
) -> tuple[list[float], list[int]]:
    """Extract per-response total bits from a diagonal log-probs tensor.

    Args:
        log_probs: ``(seq_len,)`` per-position next-token log-probs in bits.
            Entry ``[t]`` = ``log2 P(token[t+1] | token[0..t])``.
        boundaries: ``(start, end)`` token ranges for each response.
        responses: Response texts (for byte count computation).

    Returns:
        ``(curve, byte_counts)`` where ``curve[k]`` is total bits for
        response k and ``byte_counts[k]`` is its byte length.
    """
    curve: list[float] = []
    byte_counts: list[int] = []
    for k, (start, end) in enumerate(boundaries):
        bc = len(responses[k].encode("utf-8"))
        byte_counts.append(bc)
        if bc == 0 or end <= start:
            curve.append(0.0)
            continue

        # log_probs[t-1] gives log2 P(token[t] | token[0..t-1])
        # Already in bits from _forward_log_probs
        total_bits = -log_probs[start - 1 : end - 1].sum().item()
        curve.append(total_bits)

    return curve, byte_counts


def _right_pad_and_batch(
    sequences: list[list[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Right-pad token sequences into a batch.

    Right-padding is preferred over left-padding for scoring (non-generation)
    because real tokens keep their natural position IDs (starting at 0),
    exactly matching the pretraining regime.  Padding tokens appended at the
    end cannot influence earlier positions due to the causal attention mask.

    Returns:
        ``(input_ids, attention_mask)`` where ``attention_mask`` is ``None``
        when no padding is needed (all sequences have the same length).
    """
    max_len = max(len(ids) for ids in sequences)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    needs_padding = False
    for ids in sequences:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            needs_padding = True
        padded.append(ids + [pad_token_id] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(masks) if needs_padding else None
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _response_label(index: int) -> str:
    """Generate response labels: A, B, C, ..., Z, AA, AB, ...

    Args:
        index: 0-based index of the response.

    Returns:
        Label string (e.g., "A" for index 0, "Z" for 25, "AA" for 26).
    """
    letters = string.ascii_uppercase
    if index < 26:
        return letters[index]
    # For index >= 26, use multi-character labels
    result = []
    n = index
    while True:
        if n < 26:
            result.append(letters[n])
            break
        result.append(letters[n % 26])
        n = n // 26 - 1
    return "".join(reversed(result))


def format_conditioning_context(
    prompt: str,
    previous_responses: list[str],
    current_response: str,
) -> tuple[str, str]:
    """Format conditioning context per Section 7 of the paper.

    The format is::

        [prompt p]

        Response A: [r_1]

        Response B: [r_2]
        ...
        Response X: [current_response]

    Args:
        prompt: The original prompt p.
        previous_responses: List of responses r_1..r_{k-1} seen so far.
        current_response: The response r_k whose surprise we measure.

    Returns:
        Tuple of (prefix, target) where prefix is the formatted context
        before the current response text, and target is the current response
        text. The full context is prefix + target.
    """
    parts = [prompt]
    for i, resp in enumerate(previous_responses):
        label = _response_label(i)
        parts.append(f"\n\nResponse {label}: {resp}")

    current_label = _response_label(len(previous_responses))
    parts.append(f"\n\nResponse {current_label}: ")

    prefix = "".join(parts)
    return prefix, current_response


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_cross_entropy(
    model: ModelInput,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    prefix: str,
    temperature: float = 1.0,
) -> tuple[float, int]:
    """Compute total cross-entropy (in bits) and byte count of text conditioned on prefix.

    Tokenizes prefix+text together, computes log-probs only for tokens
    corresponding to ``text``, returns total bits and byte count separately.

    Supports model ensembling: pass a list of models to average their
    softmax probabilities at each token position (Section 7.5, Eq 27).

    Args:
        model: The base model theta, or a list of models to ensemble.
        tokenizer: Tokenizer for theta.
        text: The response r whose cross-entropy we compute.
        prefix: Everything before the response (prompt + previous responses).
        temperature: Temperature for scaling logits before softmax. Default 1.0.

    Returns:
        Tuple of ``(total_bits, byte_count)`` where total_bits = -sum(log2(p))
        for all tokens in ``text``, and byte_count = len(text.encode("utf-8")).
    """
    models = _ensure_models(model)
    byte_count = len(text.encode("utf-8"))
    if byte_count == 0:
        return 0.0, 0

    # Tokenize prefix and full sequence separately to find where text tokens start
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + text, add_special_tokens=False)

    n_prefix_tokens = len(prefix_ids)
    n_full_tokens = len(full_ids)

    if n_full_tokens <= n_prefix_tokens:
        return 0.0, byte_count

    input_ids = torch.tensor([full_ids])
    log_probs = _forward_log_probs(models, input_ids, temperature=temperature)[0]  # (seq_len,) in bits

    # log_probs[t-1] = log2 P(token[t] | token[0..t-1])
    # Sum over text token positions (already in bits)
    total_bits = -log_probs[n_prefix_tokens - 1 : n_full_tokens - 1].sum().item()

    return total_bits, byte_count


def compute_per_byte_cross_entropy(
    model: ModelInput,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    prefix: str,
    temperature: float = 1.0,
) -> float:
    """Compute per-byte cross-entropy of text conditioned on prefix.

    Eq 1: h_theta(r | p) = (1/||r||) * sum_t -log2 theta(r^t | r^{<t}, p)

    Thin wrapper around :func:`compute_cross_entropy` that divides by byte count.

    Args:
        model: The base model theta, or a list of models to ensemble.
        tokenizer: Tokenizer for theta.
        text: The response r whose cross-entropy we compute.
        prefix: Everything before the response (prompt + previous responses).
        temperature: Temperature for scaling logits before softmax. Default 1.0.

    Returns:
        Per-byte cross-entropy in bits/byte.
    """
    total_bits, byte_count = compute_cross_entropy(
        model, tokenizer, text, prefix, temperature=temperature
    )
    return total_bits / byte_count if byte_count > 0 else 0.0


def compute_progressive_surprise_curve(
    model: ModelInput,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
    temperature: float = 1.0,
) -> tuple[list[float], list[int]]:
    """Compute the progressive conditional surprise curve.

    Eq 4: a_k = -log2 P(r_k | r_{<k}, p) for k=1..n (total bits)

    Performs n forward passes with growing context. Prefer
    :func:`compute_progressive_surprise_curve_single_pass` for efficiency.

    Args:
        model: The base model theta, or a list of models to ensemble.
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of n responses [r_1, ..., r_n].
        temperature: Temperature for scaling logits before softmax. Default 1.0.

    Returns:
        Tuple of ``(curve, byte_counts)`` where curve is a list of a_k values
        in total bits, and byte_counts is the byte count of each response.
    """
    curve: list[float] = []
    byte_counts: list[int] = []
    for k in range(len(responses)):
        previous = responses[:k]
        current = responses[k]
        prefix, target = format_conditioning_context(prompt, previous, current)
        total_bits, bc = compute_cross_entropy(
            model, tokenizer, target, prefix, temperature=temperature
        )
        curve.append(total_bits)
        byte_counts.append(bc)
    return curve, byte_counts


def compute_progressive_surprise_curve_single_pass(
    model: ModelInput,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
    temperature: float = 1.0,
) -> tuple[list[float], list[int]]:
    """Compute the progressive conditional surprise curve using a single forward pass.

    Equivalent to :func:`compute_progressive_surprise_curve` but concatenates all
    responses into one sequence and runs a single forward pass, then extracts
    per-response log-probs by locating token boundaries.

    Supports model ensembling: pass a list of models to average their softmax
    probabilities at each token position (Section 7.5, Eq 27).

    Args:
        model: The base model theta, or a list of models to ensemble.
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of n responses [r_1, ..., r_n].
        temperature: Temperature for scaling logits before softmax. Default 1.0.

    Returns:
        Tuple of ``(curve, byte_counts)`` where curve is a list of a_k values
        in total bits, and byte_counts is the byte count of each response.
    """
    models = _ensure_models(model)
    n = len(responses)
    if n == 0:
        return [], []

    full_ids, boundaries = _find_response_boundaries(tokenizer, prompt, responses)

    # Single forward pass (ensemble-aware)
    input_ids = torch.tensor([full_ids])
    log_probs = _forward_log_probs(models, input_ids, temperature=temperature)[0]  # (seq_len,) in bits

    return _extract_response_log_probs(log_probs, boundaries, responses)


def compute_unconditional_surprises(
    model: ModelInput,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
    batch_size: int = 1,
    temperature: float = 1.0,
) -> tuple[list[float], list[float], list[int]]:
    """Compute unconditional cross-entropy for each response.

    h_theta(r_i | p) for each response scored independently against just
    the prompt (no conditioning on other responses).

    The n forward passes are embarrassingly parallel (Section 7.2) and are
    batched according to ``batch_size`` for GPU efficiency.

    Args:
        model: The base model theta, or a list of models to ensemble.
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of responses.
        batch_size: Number of responses to process in parallel. Default 1
            (sequential). Increase for GPU acceleration.
        temperature: Temperature for scaling logits before softmax. Default 1.0.

    Returns:
        Tuple of ``(per_byte_surprises, total_bits_list, byte_counts)``:
        - per_byte_surprises: h_theta(r_i | p) in bits/byte
        - total_bits_list: -log2 P(r_i | p) in total bits
        - byte_counts: byte count of each response
    """
    models = _ensure_models(model)

    # Prepare all sequences
    all_full_ids: list[list[int]] = []
    all_prefix_lens: list[int] = []
    all_byte_counts: list[int] = []
    for resp in responses:
        prefix, target = format_conditioning_context(prompt, [], resp)
        full_ids = tokenizer.encode(prefix + target, add_special_tokens=False)
        prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
        all_full_ids.append(full_ids)
        all_prefix_lens.append(prefix_len)
        all_byte_counts.append(len(target.encode("utf-8")))

    per_byte_surprises = [0.0] * len(responses)
    total_bits_list = [0.0] * len(responses)
    pad_token_id = _get_pad_token_id(tokenizer)

    for batch_start in range(0, len(responses), batch_size):
        batch_end = min(batch_start + batch_size, len(responses))
        batch_ids = all_full_ids[batch_start:batch_end]
        batch_prefix_lens = all_prefix_lens[batch_start:batch_end]

        # Right-pad and batch
        input_ids, attention_mask = _right_pad_and_batch(batch_ids, pad_token_id)
        log_probs = _forward_log_probs(
            models, input_ids, attention_mask, temperature=temperature
        )

        for i in range(batch_end - batch_start):
            seq_idx = batch_start + i
            ids = batch_ids[i]
            prefix_len = batch_prefix_lens[i]
            bc = all_byte_counts[seq_idx]

            if bc == 0 or len(ids) <= prefix_len:
                continue

            start = prefix_len
            end = len(ids)

            # log_probs[i, t-1] = log2 P(token[t] | token[0..t-1]), already in bits
            total_bits = -log_probs[i, start - 1 : end - 1].sum().item()
            total_bits_list[seq_idx] = total_bits
            per_byte_surprises[seq_idx] = total_bits / bc if bc > 0 else 0.0

        del log_probs

    return per_byte_surprises, total_bits_list, all_byte_counts


def _compute_permutation_curves_batched(
    models: list[PreTrainedModel] | "APIModel",
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
    permutations: list[list[int]],
    batch_size: int = 1,
    temperature: float = 1.0,
) -> list[tuple[list[float], list[int]]]:
    """Compute single-pass a_k curves for multiple permutations, batched.

    Each permutation reorders the responses and produces an independent
    concatenated context.  Permutations are processed in batches of
    ``batch_size`` forward passes.

    Args:
        models: List of models (ensemble) or single-element list.
        tokenizer: Tokenizer shared by all models.
        prompt: The original prompt.
        responses: Original (unpermuted) response list.
        permutations: List of permutation orderings (each a list of indices).
        batch_size: Number of permutations per batch.
        temperature: Temperature for scaling logits before softmax. Default 1.0.

    Returns:
        List of ``(total_bits_curve, byte_counts)`` per permutation.
    """
    pad_token_id = _get_pad_token_id(tokenizer)

    # Pre-compute tokenized sequences and boundaries for every permutation
    all_full_ids: list[list[int]] = []
    all_boundaries: list[list[tuple[int, int]]] = []
    all_permuted_responses: list[list[str]] = []

    for perm in permutations:
        permuted = [responses[i] for i in perm]
        all_permuted_responses.append(permuted)
        full_ids, boundaries = _find_response_boundaries(tokenizer, prompt, permuted)
        all_full_ids.append(full_ids)
        all_boundaries.append(boundaries)

    results: list[tuple[list[float], list[int]]] = []

    # Progress bar
    progress = None
    if len(permutations) > 5:
        progress = tqdm(total=len(permutations), desc="  permutations", leave=False)

    for batch_start in range(0, len(permutations), batch_size):
        batch_end = min(batch_start + batch_size, len(permutations))
        batch_ids = all_full_ids[batch_start:batch_end]

        # Right-pad and batch
        input_ids, attention_mask = _right_pad_and_batch(batch_ids, pad_token_id)
        log_probs = _forward_log_probs(
            models, input_ids, attention_mask, temperature=temperature
        )

        for i in range(batch_end - batch_start):
            perm_idx = batch_start + i
            curve, byte_counts = _extract_response_log_probs(
                log_probs[i],
                all_boundaries[perm_idx],
                all_permuted_responses[perm_idx],
            )
            results.append((curve, byte_counts))

        del log_probs

        if progress is not None:
            progress.update(batch_end - batch_start)

    if progress is not None:
        progress.close()

    return results


# ---------------------------------------------------------------------------
# Derived metrics (pure math, no model calls)
# ---------------------------------------------------------------------------


def compute_excess_entropy(a_k_curve: list[float]) -> float:
    """Compute excess entropy E = sum(a_k - a_n) from an a_k curve in total bits.

    This is Eq 6 from the paper. a_n (the last point) is used as the
    estimate of a_infinity.

    Args:
        a_k_curve: Progressive conditional surprise curve in total bits.

    Returns:
        Excess entropy E in bits.
    """
    a_n = a_k_curve[-1]
    return sum(a_k - a_n for a_k in a_k_curve)


def _compute_metrics_from_curves(
    a_k_curve_total_bits: list[float],
    a_k_byte_counts: list[int],
    unconditional_per_byte: list[float],
    unconditional_byte_counts: list[int],
    e_rate: float,
    responses: list[str],
) -> dict[str, Any]:
    """Compute all derived metrics from curves and pre-computed E_rate.

    Args:
        a_k_curve_total_bits: Progressive conditional surprise curve in total bits.
        a_k_byte_counts: Byte count per response in the a_k curve.
        unconditional_per_byte: h_theta(r_i | p) per-byte for each response.
        unconditional_byte_counts: Byte count per response for unconditional.
        e_rate: Pre-computed E_rate (per-byte excess entropy, Option B).
        responses: The response texts (needed for byte length).

    Returns:
        Dict with all metric values.
    """
    n = len(a_k_curve_total_bits)

    # Eq 6: excess entropy E = sum(a_k - a_n) in total bits
    excess_entropy_E = compute_excess_entropy(a_k_curve_total_bits)

    # Per-byte a_k curve (for plotting and backward compat)
    a_k_curve_per_byte = [
        t / b if b > 0 else 0.0 for t, b in zip(a_k_curve_total_bits, a_k_byte_counts)
    ]

    # Mean unconditional surprise (per-byte)
    mean_h = sum(unconditional_per_byte) / n

    # Eq 8: coherence C = 2^{-(1/n) * sum(h_theta(r_i | p))} (per-byte)
    coherence_C = 2.0 ** (-mean_h)

    # Coherence spread (Section 6.4): std of h_theta(r_i | p) (per-byte)
    coherence_spread_sigma = float(np.std(unconditional_per_byte, ddof=0))

    # E_rate stored from caller
    excess_entropy_E_rate = e_rate

    # Eq 12: diversity score D = C * E (bits)
    diversity_score_D = coherence_C * excess_entropy_E

    # Eq 13: diversity score rate D_rate = C * E_rate (bits/byte)
    diversity_score_D_rate = coherence_C * excess_entropy_E_rate

    # Mean byte length
    byte_lengths = [len(r.encode("utf-8")) for r in responses]
    mean_byte_length = sum(byte_lengths) / n

    # Uncertainty band (Section 6.4) — uses E_rate
    C_plus = coherence_C * (2.0**coherence_spread_sigma)
    C_minus = coherence_C * (2.0 ** (-coherence_spread_sigma))
    D_plus = C_plus * excess_entropy_E_rate
    D_minus = C_minus * excess_entropy_E_rate

    # Diagnostic: is the curve monotonically non-increasing?
    is_monotone = all(
        a_k_curve_total_bits[i] >= a_k_curve_total_bits[i + 1] for i in range(n - 1)
    )

    return {
        "a_k_curve": a_k_curve_total_bits,
        "a_k_curve_per_byte": a_k_curve_per_byte,
        "a_k_byte_counts": a_k_byte_counts,
        "unconditional_surprises": unconditional_per_byte,
        "excess_entropy_E": excess_entropy_E,
        "excess_entropy_E_rate": excess_entropy_E_rate,
        "coherence_C": coherence_C,
        "coherence_spread_sigma": coherence_spread_sigma,
        "diversity_score_D": diversity_score_D,
        "diversity_score_D_rate": diversity_score_D_rate,
        "mean_byte_length": mean_byte_length,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "C_plus": C_plus,
        "C_minus": C_minus,
        "is_monotone": is_monotone,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_icl_diversity_metrics(
    model: ModelInput,
    tokenizer: PreTrainedTokenizerBase | None,
    prompt: str,
    responses: list[str],
    n_permutations: int = 1,
    seed: int = 42,
    batch_size: int = 1,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Full ICL diversity metric computation for one prompt.

    Computes the progressive conditional surprise curve and all derived
    metrics. If ``n_permutations > 1``, averages over random orderings of the
    responses per Section 7.3 of the paper.

    Supports:

    - **Batching** (``batch_size > 1``): unconditional surprises and
      permutation forward passes are batched for GPU parallelism.
    - **Model ensembling** (pass a list of models): softmax probabilities
      are averaged at each token position per Section 7.5 (Eq 27).  All
      models must share the same tokenizer/vocabulary.
    - **Temperature** (``temperature != 1.0``): scales logits before softmax.
      T>1 flattens predictions (reduces per-permutation variance), T<1
      sharpens predictions (amplifies signal).

    Args:
        model: The base model theta (should be a base model, not
            instruction-tuned).  Pass a list of models for ensembling.
        tokenizer: Tokenizer for theta (shared by all models in an ensemble).
        prompt: The original prompt p.
        responses: List of n responses sampled from the policy under evaluation.
        n_permutations: Number of random orderings to average over (paper
            suggests 3-5). Default 1 uses the given order.
        seed: Random seed for permutation generation.
        batch_size: Number of forward passes to batch together. Default 1
            (sequential). Increase for GPU acceleration.  Applies to both
            unconditional surprises and permutation curves.
        temperature: Temperature for scaling logits before softmax. Must be
            positive. T>1 reduces variance (fewer permutations needed), T<1
            amplifies signal. Default 1.0 (no scaling).

    Returns:
        Dict with:
        - a_k_curve: list[float]           # total bits (progressive conditional surprise)
        - a_k_curve_per_byte: list[float]  # per-byte normalized
        - a_k_byte_counts: list[int]       # byte count per response
        - unconditional_surprises: list[float]  # h_theta(r_i | p) per-byte
        - unconditional_total_bits: list[float] # -log2 P(r_i | p) total bits
        - excess_entropy_E: float          # total bits excess entropy
        - excess_entropy_E_rate: float     # per-byte excess entropy rate (Option B)
        - coherence_C: float               # per-byte coherence
        - coherence_spread_sigma: float    # std of h_theta(r_i|p)
        - diversity_score_D: float         # C * E (bits)
        - diversity_score_D_rate: float    # C * E_rate (bits/byte)
        - mean_byte_length: float          # B_bar
        - D_plus: float                    # C+ * E_rate
        - D_minus: float                   # C- * E_rate
        - C_plus: float                    # C * 2^sigma
        - C_minus: float                   # C * 2^{-sigma}
        - is_monotone: bool                # diagnostic
        - temperature: float               # the temperature used
        - per_permutation_a_k_curves: list[list[float]] | None
            Raw a_k curve (total bits) from each permutation.
        - per_permutation_byte_counts: list[list[int]] | None
            Byte counts from each permutation.
        - permutation_orders: list[list[int]] | None
            The ordering used for each permutation.
    """
    models = _ensure_models(model)

    # For API models, auto-resolve tokenizer if not provided
    if tokenizer is None:
        if _is_api_model(models):
            tokenizer = models.tokenizer  # type: ignore[union-attr]
        else:
            raise ValueError("tokenizer is required for local models")

    # Unconditional surprises are order-independent, compute once (batched)
    unconditional_per_byte, unconditional_total_bits, unconditional_byte_counts = (
        compute_unconditional_surprises(
            models, tokenizer, prompt, responses, batch_size=batch_size,
            temperature=temperature,
        )
    )

    if n_permutations <= 1:
        # Single ordering — single forward pass
        a_k_total, byte_counts = compute_progressive_surprise_curve_single_pass(
            models, tokenizer, prompt, responses, temperature=temperature
        )
        # Compute E_rate: normalize per response, then sum excess
        per_byte = [t / b if b > 0 else 0.0 for t, b in zip(a_k_total, byte_counts)]
        e_rate = sum(pb - per_byte[-1] for pb in per_byte)

        metrics = _compute_metrics_from_curves(
            a_k_total,
            byte_counts,
            unconditional_per_byte,
            unconditional_byte_counts,
            e_rate,
            responses,
        )
        metrics["unconditional_total_bits"] = unconditional_total_bits
        metrics["temperature"] = temperature
        metrics["per_permutation_a_k_curves"] = None
        metrics["per_permutation_byte_counts"] = None
        metrics["permutation_orders"] = None
        return metrics

    # Multiple permutations: generate all, then batch compute
    rng = random.Random(seed)
    n = len(responses)
    all_perms: list[list[int]] = []
    for _ in range(n_permutations):
        perm = list(range(n))
        rng.shuffle(perm)
        all_perms.append(perm)

    perm_results = _compute_permutation_curves_batched(
        models, tokenizer, prompt, responses, all_perms, batch_size,
        temperature=temperature,
    )

    # Unpack results
    all_total_bits_curves = [r[0] for r in perm_results]
    all_byte_counts = [r[1] for r in perm_results]
    all_per_byte_curves = [
        [t / b if b > 0 else 0.0 for t, b in zip(tb, bc)] for tb, bc in perm_results
    ]

    # Average across permutations
    avg_total_bits = [
        sum(curves[k] for curves in all_total_bits_curves) / n_permutations
        for k in range(n)
    ]
    avg_per_byte = [
        sum(curves[k] for curves in all_per_byte_curves) / n_permutations
        for k in range(n)
    ]
    # Use average byte counts for reference
    avg_byte_counts = [
        round(sum(bcs[k] for bcs in all_byte_counts) / n_permutations) for k in range(n)
    ]

    # E_rate from averaged per-byte curve (Option B)
    e_rate = sum(avg_per_byte[k] - avg_per_byte[-1] for k in range(n))

    metrics = _compute_metrics_from_curves(
        avg_total_bits,
        avg_byte_counts,
        unconditional_per_byte,
        unconditional_byte_counts,
        e_rate,
        responses,
    )
    metrics["unconditional_total_bits"] = unconditional_total_bits
    metrics["temperature"] = temperature
    metrics["per_permutation_a_k_curves"] = all_total_bits_curves
    metrics["per_permutation_byte_counts"] = all_byte_counts
    metrics["permutation_orders"] = all_perms
    return metrics
