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
- Eq: total coherence C_total = 2^{mean(log2(P(r_i | p)))} (total bits)
- Eq 11: diversity score D = C * E_rate (per-byte)
- Eq: total diversity score D_total = C_total * E (total bits)
"""

import math
import random
import string
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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

    The format is:
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


def compute_cross_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    prefix: str,
) -> tuple[float, int]:
    """Compute total cross-entropy (in bits) and byte count of text conditioned on prefix.

    Tokenizes prefix+text together, computes log-probs only for tokens
    corresponding to `text`, returns total bits and byte count separately.

    Args:
        model: The base model theta.
        tokenizer: Tokenizer for theta.
        text: The response r whose cross-entropy we compute.
        prefix: Everything before the response (prompt + previous responses).

    Returns:
        Tuple of (total_bits, byte_count) where total_bits = -sum(log2(p))
        for all tokens in `text`, and byte_count = len(text.encode("utf-8")).
    """
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

    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute log-probs for each position
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Sum log-probs for text tokens only
    # For token at position t, the logit at position t-1 predicts it
    total_log_prob = 0.0
    for t in range(n_prefix_tokens, n_full_tokens):
        token_id = full_ids[t]
        # logits at position t-1 predict token at position t
        total_log_prob += log_probs[t - 1, token_id].item()

    # Convert from nats (ln) to bits (log2): log2(x) = ln(x) / ln(2)
    total_log2_prob = total_log_prob / math.log(2)

    # Total cross-entropy in bits (Eq 1 without the 1/||r|| normalization)
    total_bits = -total_log2_prob
    return total_bits, byte_count


def compute_per_byte_cross_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    prefix: str,
) -> float:
    """Compute per-byte cross-entropy of text conditioned on prefix.

    Eq 1: h_theta(r | p) = (1/||r||) * sum_t -log2 theta(r^t | r^{<t}, p)

    Thin wrapper around compute_cross_entropy that divides by byte count.

    Args:
        model: The base model theta.
        tokenizer: Tokenizer for theta.
        text: The response r whose cross-entropy we compute.
        prefix: Everything before the response (prompt + previous responses).

    Returns:
        Per-byte cross-entropy in bits/byte.
    """
    total_bits, byte_count = compute_cross_entropy(model, tokenizer, text, prefix)
    return total_bits / byte_count if byte_count > 0 else 0.0


def compute_progressive_surprise_curve(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
) -> tuple[list[float], list[int]]:
    """Compute the progressive conditional surprise curve.

    Eq 4: a_k = -log2 P(r_k | r_{<k}, p) for k=1..n (total bits)

    Performs n forward passes with growing context.

    Args:
        model: The base model theta.
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of n responses [r_1, ..., r_n].

    Returns:
        Tuple of (curve, byte_counts) where curve is a list of a_k values
        in total bits, and byte_counts is the byte count of each response.
    """
    curve: list[float] = []
    byte_counts: list[int] = []
    for k in range(len(responses)):
        previous = responses[:k]
        current = responses[k]
        prefix, target = format_conditioning_context(prompt, previous, current)
        total_bits, bc = compute_cross_entropy(model, tokenizer, target, prefix)
        curve.append(total_bits)
        byte_counts.append(bc)
    return curve, byte_counts


def compute_progressive_surprise_curve_single_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
) -> tuple[list[float], list[int]]:
    """Compute the progressive conditional surprise curve using a single forward pass.

    Equivalent to :func:`compute_progressive_surprise_curve` but concatenates all
    responses into one sequence and runs a single forward pass, then extracts
    per-response log-probs by locating token boundaries.

    The boundary-finding logic mirrors :func:`compute_cross_entropy`:
    we tokenize progressive prefixes to determine where each response's tokens
    start and end in the full token sequence.

    Args:
        model: The base model theta.
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of n responses [r_1, ..., r_n].

    Returns:
        Tuple of (curve, byte_counts) where curve is a list of a_k values
        in total bits, and byte_counts is the byte count of each response.
    """
    n = len(responses)
    if n == 0:
        return [], []

    # Build the full concatenated context
    parts = [prompt]
    for i, resp in enumerate(responses):
        label = _response_label(i)
        parts.append(f"\n\nResponse {label}: {resp}")
    full_text = "".join(parts)

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Find response token boundaries using progressive prefix tokenization.
    # This mirrors the boundary logic in compute_cross_entropy:
    # tokenize prefix alone to get start index, then prefix+response to get end.
    boundaries: list[tuple[int, int]] = []
    running_text = prompt + f"\n\nResponse {_response_label(0)}: "
    for k in range(n):
        n_prefix = len(tokenizer.encode(running_text, add_special_tokens=False))
        running_text += responses[k]
        n_with_resp = len(tokenizer.encode(running_text, add_special_tokens=False))
        boundaries.append((n_prefix, n_with_resp))
        if k < n - 1:
            running_text += f"\n\nResponse {_response_label(k + 1)}: "

    # Sanity check: final running_text should equal full_text
    assert len(tokenizer.encode(running_text, add_special_tokens=False)) == len(
        full_ids
    ), (
        f"Tokenization mismatch: progressive="
        f"{len(tokenizer.encode(running_text, add_special_tokens=False))}, "
        f"full={len(full_ids)}"
    )

    # Single forward pass
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute log-probs once for the full sequence
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Extract per-response a_k values
    curve: list[float] = []
    byte_counts_list: list[int] = []
    for k, (start, end) in enumerate(boundaries):
        byte_count = len(responses[k].encode("utf-8"))
        byte_counts_list.append(byte_count)
        if byte_count == 0 or end <= start:
            curve.append(0.0)
            continue

        # Causal shift: logits[t-1] predicts token at position t
        # Gather log-probs for token IDs at positions start..end-1
        token_positions = torch.arange(start, end, device=logits.device)
        token_ids = torch.tensor(
            full_ids[start:end], device=logits.device, dtype=torch.long
        )
        # log_probs[t-1] gives the prediction for position t
        total_log_prob = log_probs[token_positions - 1, token_ids].sum().item()

        # Convert nats → bits (total, not per-byte)
        total_log2_prob = total_log_prob / math.log(2)
        a_k = -total_log2_prob
        curve.append(a_k)

    return curve, byte_counts_list


def compute_unconditional_surprises(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
) -> tuple[list[float], list[float], list[int]]:
    """Compute unconditional cross-entropy for each response.

    h_theta(r_i | p) for each response scored independently against just
    the prompt (no conditioning on other responses).

    Args:
        model: The base model theta.
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of responses.

    Returns:
        Tuple of (per_byte_surprises, total_bits_list, byte_counts):
        - per_byte_surprises: h_theta(r_i | p) in bits/byte
        - total_bits_list: -log2 P(r_i | p) in total bits
        - byte_counts: byte count of each response
    """
    per_byte_surprises: list[float] = []
    total_bits_list: list[float] = []
    byte_counts: list[int] = []
    for resp in responses:
        prefix, target = format_conditioning_context(prompt, [], resp)
        total_bits, bc = compute_cross_entropy(model, tokenizer, target, prefix)
        byte_counts.append(bc)
        total_bits_list.append(total_bits)
        per_byte_surprises.append(total_bits / bc if bc > 0 else 0.0)
    return per_byte_surprises, total_bits_list, byte_counts


def _compute_metrics_from_curves(
    a_k_curve_total_bits: list[float],
    a_k_byte_counts: list[int],
    unconditional_per_byte: list[float],
    unconditional_total_bits: list[float],
    unconditional_byte_counts: list[int],
    e_rate: float,
    responses: list[str],
) -> dict[str, Any]:
    """Compute all derived metrics from curves and pre-computed E_rate.

    Args:
        a_k_curve_total_bits: Progressive conditional surprise curve in total bits.
        a_k_byte_counts: Byte count per response in the a_k curve.
        unconditional_per_byte: h_theta(r_i | p) per-byte for each response.
        unconditional_total_bits: -log2 P(r_i | p) total bits for each response.
        unconditional_byte_counts: Byte count per response for unconditional.
        e_rate: Pre-computed E_rate (per-byte excess entropy, Option B).
        responses: The response texts (needed for byte length).

    Returns:
        Dict with all metric values.
    """
    n = len(a_k_curve_total_bits)
    a_n = a_k_curve_total_bits[-1]  # estimate of a_infinity

    # Eq 6: excess entropy E = sum(a_k - a_n) in total bits
    excess_entropy_E = sum(a_k - a_n for a_k in a_k_curve_total_bits)

    # Per-byte a_k curve (for plotting and backward compat)
    a_k_curve_per_byte = [
        t / b if b > 0 else 0.0 for t, b in zip(a_k_curve_total_bits, a_k_byte_counts)
    ]

    # Mean unconditional surprise (per-byte)
    mean_h = sum(unconditional_per_byte) / n

    # Eq 8: coherence C = 2^{-(1/n) * sum(h_theta(r_i | p))} (per-byte)
    coherence_C = 2.0 ** (-mean_h)

    # Total coherence C_total = 2^{mean(log2(P(r_i | p)))}
    # log2(P(r_i | p)) = -total_bits_i
    mean_log2_prob = sum(-tb for tb in unconditional_total_bits) / n
    coherence_C_total = 2.0**mean_log2_prob

    # Coherence spread (Section 6.4): std of h_theta(r_i | p) (per-byte)
    coherence_spread_sigma = float(np.std(unconditional_per_byte, ddof=0))

    # E_rate stored from caller
    excess_entropy_E_rate = e_rate

    # Eq 11: diversity score D = C * E_rate (per-byte)
    diversity_score_D = coherence_C * excess_entropy_E_rate

    # Total diversity score D_total = C_total * E (total bits)
    diversity_score_D_total = coherence_C_total * excess_entropy_E

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
        "unconditional_total_bits": unconditional_total_bits,
        "excess_entropy_E": excess_entropy_E,
        "excess_entropy_E_rate": excess_entropy_E_rate,
        "coherence_C": coherence_C,
        "coherence_C_total": coherence_C_total,
        "coherence_spread_sigma": coherence_spread_sigma,
        "diversity_score_D": diversity_score_D,
        "diversity_score_D_total": diversity_score_D_total,
        "mean_byte_length": mean_byte_length,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "C_plus": C_plus,
        "C_minus": C_minus,
        "is_monotone": is_monotone,
    }


def compute_icl_diversity_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    responses: list[str],
    n_permutations: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Full ICL diversity metric computation for one prompt.

    Computes the progressive conditional surprise curve and all derived
    metrics. If n_permutations > 1, averages over random orderings of the
    responses per Section 7.3 of the paper.

    Args:
        model: The base model theta (should be a base model, not instruction-tuned).
        tokenizer: Tokenizer for theta.
        prompt: The original prompt p.
        responses: List of n responses sampled from the policy under evaluation.
        n_permutations: Number of random orderings to average over (paper
            suggests 3-5). Default 1 uses the given order.
        seed: Random seed for permutation generation.

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
        - coherence_C_total: float         # total coherence
        - coherence_spread_sigma: float    # std of h_theta(r_i|p)
        - diversity_score_D: float         # C * E_rate
        - diversity_score_D_total: float   # C_total * E
        - mean_byte_length: float          # B_bar
        - D_plus: float                    # C+ * E_rate
        - D_minus: float                   # C- * E_rate
        - C_plus: float                    # C * 2^sigma
        - C_minus: float                   # C * 2^{-sigma}
        - is_monotone: bool                # diagnostic
        - per_permutation_a_k_curves: list[list[float]] | None
            Raw a_k curve (total bits) from each permutation.
        - per_permutation_byte_counts: list[list[int]] | None
            Byte counts from each permutation.
        - permutation_orders: list[list[int]] | None
            The ordering used for each permutation.
    """
    # Unconditional surprises are order-independent, compute once
    unconditional_per_byte, unconditional_total_bits, unconditional_byte_counts = (
        compute_unconditional_surprises(model, tokenizer, prompt, responses)
    )

    if n_permutations <= 1:
        # Single ordering — single forward pass
        a_k_total, byte_counts = compute_progressive_surprise_curve_single_pass(
            model, tokenizer, prompt, responses
        )
        # Compute E_rate: normalize per response, then sum excess
        per_byte = [t / b if b > 0 else 0.0 for t, b in zip(a_k_total, byte_counts)]
        e_rate = sum(pb - per_byte[-1] for pb in per_byte)

        metrics = _compute_metrics_from_curves(
            a_k_total,
            byte_counts,
            unconditional_per_byte,
            unconditional_total_bits,
            unconditional_byte_counts,
            e_rate,
            responses,
        )
        metrics["per_permutation_a_k_curves"] = None
        metrics["per_permutation_byte_counts"] = None
        metrics["permutation_orders"] = None
        return metrics

    # Multiple permutations: average a_k curves
    rng = random.Random(seed)
    n = len(responses)
    all_total_bits_curves: list[list[float]] = []
    all_byte_counts: list[list[int]] = []
    all_per_byte_curves: list[list[float]] = []
    all_perms: list[list[int]] = []

    perm_iter = range(n_permutations)
    if n_permutations > 5:
        perm_iter = tqdm(perm_iter, desc="  permutations", leave=False)

    for _ in perm_iter:
        perm = list(range(n))
        rng.shuffle(perm)
        all_perms.append(list(perm))
        permuted_responses = [responses[i] for i in perm]
        total_bits, byte_counts = compute_progressive_surprise_curve_single_pass(
            model, tokenizer, prompt, permuted_responses
        )
        all_total_bits_curves.append(total_bits)
        all_byte_counts.append(byte_counts)
        all_per_byte_curves.append(
            [t / b if b > 0 else 0.0 for t, b in zip(total_bits, byte_counts)]
        )

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
        unconditional_total_bits,
        unconditional_byte_counts,
        e_rate,
        responses,
    )
    metrics["per_permutation_a_k_curves"] = all_total_bits_curves
    metrics["per_permutation_byte_counts"] = all_byte_counts
    metrics["permutation_orders"] = all_perms
    return metrics
