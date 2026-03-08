# Migration Plan: Updating Existing Code to Match Current Paper

The existing code implements the metric as described in the earlier version of the paper (per-byte a_k, per-byte E, m_eff). This plan describes how to update it to match the current paper (`in_context_diversity_metric.tex`). All section references are to the current paper.

## Summary of Changes

| Quantity | Old code | New code |
|----------|----------|----------|
| a_k | bits/byte (divided by ‖r_k‖) | total bits (−log₂ θ(r_k \| r_{<k}, p)) |
| E | sum of per-byte excess (bits/byte) | sum of total-bits excess (bits) (§4.1) |
| E_rate | not separate from E | per-byte rate via Option B: normalize per-response before averaging permutations (§4.2) |
| m_eff | computed | removed (§6.5 explains why) |
| C | per-byte coherence | unchanged, used for D (§5.2) |
| 𝒞 | not implemented | geometric mean of total probabilities, used for 𝒟 and the triple (§5.2) |
| σ_ℓ | std of per-byte cross-entropies | unchanged — remains per-byte for diagnostic utility (§6.2, §6.4) |
| 𝒟 | not implemented | **primary scalar**: 𝒞 × E (bits) (§6.3) |
| D | C × E (old E was bits/byte) | **secondary scalar**: C × E_rate (bits/byte) (§6.3) |
| Triple | (E, C, σ_ℓ) | (E, 𝒞, σ_ℓ) — note 𝒞 replaces C (§6.2) |

## Step-by-Step Changes

### 1. Change a_k to total bits

In the function that computes a_k, stop dividing by byte count. The raw sum of token log-probs (converted to log base 2) IS a_k.

```python
# OLD
a_k = -total_logprob / byte_count  # bits/byte

# NEW
a_k = -total_logprob  # bits (where total_logprob is already in log base 2)
```

Keep the byte count around — you'll need it for E_rate, C, and σ_ℓ.

### 2. Change E to total bits

E is now simply the sum of total-bits excess values. No change to the formula, just different inputs:

```python
# E = sum(a_k - a_n) for k=1..n
# This was already the formula; it just operates on total-bits a_k now
E = sum(a_k_values - a_k_values[-1])
```

Units are now bits, not bits/byte.

### 3. Implement E_rate via Option B (§4.2)

This is new. E_rate is the per-byte excess entropy rate, computed by normalizing each response's surprise by its byte count BEFORE averaging across permutations. This is NOT the same as E / mean_bytes (see tests in step 9).

```python
def compute_E_rate(a_k_per_permutation, byte_counts_per_permutation):
    """
    Args:
        a_k_per_permutation: list of arrays, each of shape (n,).
            a_k_per_permutation[perm][k] is the total-bits surprise of the 
            response at position k in permutation perm.
        byte_counts_per_permutation: list of arrays, each of shape (n,).
            byte_counts_per_permutation[perm][k] is ||r_{sigma(k)}|| for that permutation.
    """
    # For each permutation, compute per-byte surprise at each position
    per_byte_curves = []
    for perm in range(len(a_k_per_permutation)):
        per_byte = a_k_per_permutation[perm] / byte_counts_per_permutation[perm]
        per_byte_curves.append(per_byte)
    
    # Average per-byte curves across permutations
    avg_per_byte = np.mean(per_byte_curves, axis=0)  # shape (n,)
    
    # Floor from last position
    floor = avg_per_byte[-1]
    
    # Sum excess
    E_rate = np.sum(avg_per_byte - floor)
    
    return E_rate  # bits/byte
```

This means the permutation loop must now track byte counts alongside a_k values, since which response lands at each position changes per permutation.

### 4. Add total coherence 𝒞 (§5.2)

This is the PRIMARY coherence measure for the triple and for 𝒟.

```python
# Total coherence: geometric mean of total probabilities
# unconditional_logprobs[i] = log2 θ(r_i | p)  (negative number)
# Geometric mean = (prod θ(r_i | p))^(1/n) = 2^(mean(log2_probs))

C_total = 2 ** np.mean(unconditional_logprobs)  # where logprobs are negative
```

Note: `unconditional_logprobs` here are NOT per-byte. They are the total log₂ probability of each response. If your existing code stores per-byte values, multiply by byte count first:

```python
total_logprob_i = per_byte_logprob_i * byte_count_i  # convert back to total
```

The per-byte coherence C is still needed for σ_ℓ, the D_± band, and the per-byte diversity score D. Keep computing it as before.

### 5. Add total diversity score 𝒟 and update D (§6.3)

𝒟 is the PRIMARY scalar. D is the secondary length-normalized variant.

```python
D_total = C_total * E   # bits (primary)
D = C * E_rate           # bits/byte (secondary, length-normalized)
```

### 6. Remove m_eff

Delete any computation of `m_eff = 2 ** (mean_bytes * E)` or similar. It is no longer part of the metric. The paper explains why in a footnote in §4.1 and in §6.5.

### 7. Update reporting

The `full_report` function should return:

```python
{
    # Primary: the curve (total bits, permutation-averaged)
    "a_k_curve": avg_a_k,              # shape (n,), bits

    # The (E, 𝒞, σ_ℓ) triple
    "E": E,                             # bits (total learnable structure)
    "C_total": C_total,                 # dimensionless (geometric mean of total probs)
    "sigma_ell": sigma_ell,             # bits/byte (per-byte, isolates quality from length)

    # Primary scalar
    "D_total": D_total,                 # bits (𝒞 × E)

    # Secondary (length-normalized) quantities
    "E_rate": E_rate,                   # bits/byte
    "C": C,                             # dimensionless (per-byte coherence)
    "D": D,                             # bits/byte (C × E_rate)
    
    # Band (when sigma_ell is large)
    "D_plus": D_plus,                   # bits/byte
    "D_minus": D_minus,                 # bits/byte
    
    # Per-response diagnostics
    "per_response_coherence": c_i,      # array of per-byte coherences
    "byte_counts": byte_counts,         # array of byte counts
}
```

### 8. Update permutation loop

The permutation loop needs to be restructured to support both total-bits E and Option B E_rate. For each permutation:

1. Shuffle the response ordering
2. Run single-pass forward to get total-bits a_k for each position
3. Record both a_k AND the byte count of the response at each position
4. After all permutations: average a_k → total-bits curve and E. Average a_k/byte_count → per-byte curve and E_rate.

```python
all_a_k = []           # total bits curves, one per permutation
all_per_byte = []      # per-byte curves, one per permutation

for perm in range(n_permutations):
    order = np.random.permutation(n)
    responses_shuffled = [responses[i] for i in order]
    byte_counts_shuffled = np.array([byte_counts[i] for i in order])
    
    a_k = compute_a_k_single_pass(model, tokenizer, prompt, responses_shuffled)
    # a_k is in total bits (NOT per-byte)
    
    all_a_k.append(a_k)
    all_per_byte.append(a_k / byte_counts_shuffled)

# Total-bits curve and E
avg_a_k = np.mean(all_a_k, axis=0)
E = np.sum(avg_a_k - avg_a_k[-1])

# Per-byte curve and E_rate (Option B: normalize before averaging)
avg_per_byte = np.mean(all_per_byte, axis=0)
E_rate = np.sum(avg_per_byte - avg_per_byte[-1])
```

### 9. Update tests

Update existing tests to check that a_k values are in total bits (they should be much larger than before — roughly 50–200 for typical responses, vs 0.5–2.0 when per-byte).

Add a test confirming E_rate ≠ E / mean_bytes when response lengths vary:

```python
def test_E_rate_differs_from_E_over_Bbar():
    """E_rate (Option B) should differ from E/Bbar when response lengths vary."""
    # Use responses of deliberately different lengths
    responses = [
        "Short.",
        "This is a much longer response with many more bytes in it for testing.",
    ] * 5  # repeat to get n=10
    
    # ... compute E and E_rate ...
    
    E_over_Bbar = E / np.mean(byte_counts)
    assert not np.isclose(E_rate, E_over_Bbar, rtol=0.01), \
        "E_rate should differ from E/Bbar when lengths vary"
```

Add a test confirming E_rate ≈ E / Bbar when all responses have equal length:

```python
def test_E_rate_equals_E_over_Bbar_equal_lengths():
    """When all responses are the same length, E_rate should equal E/Bbar."""
    responses = ["This response is exactly this long."] * 10
    
    # ... compute E and E_rate ...
    
    E_over_Bbar = E / np.mean(byte_counts)
    assert np.isclose(E_rate, E_over_Bbar, rtol=0.01)
```

Add a test confirming 𝒞 is tokenizer-agnostic (total probability doesn't depend on tokenization):

```python
def test_C_total_is_total_probability():
    """𝒞 should be the geometric mean of total probabilities."""
    # ... compute unconditional total logprobs for each response ...
    
    total_probs = [2 ** lp for lp in unconditional_logprobs]  # each is θ(r_i | p)
    geometric_mean = np.prod(total_probs) ** (1/n)
    
    assert np.isclose(C_total, geometric_mean, rtol=1e-6)
```
