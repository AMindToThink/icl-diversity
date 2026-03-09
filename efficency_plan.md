# Optimization Plan: Single-Pass a_k Computation

## Context

We have a working implementation that computes the a_k curve using n separate forward passes (the "slow" version). This plan covers optimizing it to a single forward pass, which produces identical results.

## What to Implement

### 1. Single-pass a_k computation

Replace the n-forward-pass loop with:

1. Format the full concatenated context: `[prompt p]\n\nResponse A: [r_1]\n\nResponse B: [r_2]\n\n...`
2. Tokenize the concatenation. Run one forward pass to get all token logits.
3. Extract per-token log-probs via the standard causal shift (logprob of `input_ids[t]` is at `logits[t-1]`).
4. Find response boundaries in the token sequence (see below).
5. For each response k, sum its log-probs and divide by `len(response_k.encode('utf-8'))` to get `a_k` in bits/byte.

### 2. Boundary finding

The formatting places `\n\nResponse X: ` between each response. Modern tokenizers (GPT-2, Llama, Mistral, Qwen) split on whitespace/newlines before BPE, so these delimiters will always be cleanly tokenized as separate tokens from the response content. Approach:

1. Tokenize each delimiter string (e.g., `"\n\nResponse B: "`).
2. Search for these token sequences in the concatenated `input_ids`.
3. Response k starts immediately after delimiter k's tokens; response k ends at the start of delimiter k+1 (or end of sequence).

### 3. Equivalence test

Write a test that runs both the existing slow version and the new single-pass version on the same inputs, and asserts the a_k values match.

```python
def test_single_vs_multi_pass():
    """The single-pass optimization must produce the same a_k values as the 
    original n-forward-pass implementation."""
    model, tokenizer = load_small_model()  # e.g. gpt2 or whatever you're using
    prompt = "Write a short story about a cat."
    responses = [
        "The cat sat on the mat and purred softly.",
        "Once upon a time, a brave kitten ventured into the woods.",
        "Mr. Whiskers had a secret: he could fly.",
        "It was raining, and the tabby watched from the windowsill.",
        "The alley cat dodged between trash cans, hunting for dinner.",
    ]
    
    a_k_slow = compute_a_k_multi_pass(model, tokenizer, prompt, responses)
    a_k_fast = compute_a_k_single_pass(model, tokenizer, prompt, responses)
    
    assert np.allclose(a_k_slow, a_k_fast, atol=1e-4), \
        f"Mismatch:\nslow: {a_k_slow}\nfast: {a_k_fast}\ndiff: {a_k_slow - a_k_fast}"
```

Additionally, add a boundary-finding roundtrip check:

```python
def test_boundary_roundtrip():
    """Decoding each token slice should recover the original response."""
    tokenizer = load_tokenizer()
    prompt = "Tell me a joke."
    responses = ["Why did the chicken cross the road?", "To get to the other side!"]
    
    formatted = format_concatenated_context(prompt, responses)
    input_ids = tokenizer.encode(formatted)
    boundaries = find_response_token_boundaries(input_ids, tokenizer, len(responses))
    
    for i, (start, end) in enumerate(boundaries):
        decoded = tokenizer.decode(input_ids[start:end]).strip()
        assert decoded == responses[i].strip(), \
            f"Response {i}: expected '{responses[i]}', got '{decoded}'"
```

And the free consistency check:

```python
def test_a1_equals_unconditional():
    """a_1 from the concatenated pass should equal the unconditional 
    cross-entropy of r_1, since r_1 is conditioned only on the prompt 
    in both cases."""
    # ... compute both, assert np.isclose(a_k_fast[0], unconditional_h[0], atol=1e-4)
    # NOTE: only works if the formatting is identical in both code paths
    # (same "Response A:" prefix, same whitespace).
```

## Reminders

- **Log base**: Everything is bits/byte. Torch gives natural log, so divide by `ln(2)` or use `torch.log2`.
- **Byte counts**: `len(response.encode('utf-8'))`, not `len(response)`.
- **Causal shift**: logprob of `input_ids[t]` is at `logits[t-1]`. Off-by-one here silently produces wrong results. The equivalence test is the primary guard.
- **Formatting consistency**: The concatenated context must format r_1 identically to how the slow version formats it (same "Response A:" prefix, same whitespace). Otherwise a_1 won't match.
