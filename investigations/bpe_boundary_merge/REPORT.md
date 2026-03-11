# Investigation: BPE Boundary Merge in Single-Pass Computation

## Question

The single-pass a_k computation scores all responses in one forward pass over the concatenated context. When BPE tokenizers merge characters across response/separator boundaries (e.g., Qwen merges `"."` + `"\n\n"` into a single token `".\n\n"`), does this introduce tokenizer-dependent artifacts into the otherwise tokenizer-independent bits/byte metric?

## Background

The metric is designed to be tokenizer-independent: all quantities are in bits or bits/byte, not bits/token. However, the *set of tokens scored* for each response depends on how the tokenizer splits the concatenated context, which varies across tokenizers.

**Format**: `"Response A: [r_1]\n\nResponse B: [r_2]\n\n..."`

**The merge**: Qwen2.5-3B's BPE vocabulary includes token 382 = `'.\n\n'`, which merges the response-terminal period with the separator newlines. GPT-2 keeps `'.'` (token 13) and `'\n\n'` (token 198) as separate tokens.

## Findings

### 1. The merge is real and systematic (`01_demonstrate_merge.py`)

When the full context `"...gently.\n\nResponse B:..."` is tokenized:
- **Qwen**: token at boundary = `'.\n\n'` (merged)
- **GPT-2**: tokens at boundary = `'.'`, `'\n\n'` (separate)

The truncated (prefix) tokenization of `"...gently."` produces `'.'` as a standalone token for both tokenizers. So the progressive boundary detection was computing positions from a tokenization that disagrees with full_ids at every non-final response boundary.

### 2. Boundary positions happen to be identical (`02_boundary_comparison.py`)

Despite the token-level disagreement, the old progressive approach and the new offset-mapping approach produce **identical boundary positions** for all tested cases across both tokenizers. This is because the merge replaces two tokens with one while the progressive approach doesn't include the continuation — the token counts happen to align.

Tested with 7 response sets covering periods, exclamation marks, parentheses, ellipses, percent signs, and quotes. All cases: `0/7 differ` for both Qwen and GPT-2.

### 3. The distortion is small but real (`03_quantify_distortion.py`)

Single-pass vs multi-pass comparison (Qwen2.5-3B, m=10, n=20, 20 draws):

```
Total absolute distortion: ~3.5 bits across 20 positions
SP curve decline (a_1 - a_n): ~59 bits
MP curve decline (a_1 - a_n): ~62 bits
Distortion as % of MP decline: ~6%
Mean distortion per position: ~0.17 bits = ~0.001 bits/byte
```

The distortion grows slightly with position (from +1.2 bits at k=1 to -2.3 bits at k=18) as the model learns the `'.\n\n'` pattern.

### 4. Mechanism: separator log-prob overhead (`04_separator_logprob.py`)

By the chain rule: `-log P('.\n\n') = -log P('.') + -log P('\n\n' | '.')`

The second term is the separator overhead — extra bits for predicting `'\n\n'` as part of the response. Since the model strongly expects `'\n\n'` after a period in this format, this overhead is very small (~0.1-0.3 bits per boundary).

With 19 internal boundaries in a 20-response curve, total overhead is ~2-5 bits, consistent with the measured 3.5-bit aggregate distortion.

## Root Cause

This is **not a boundary detection bug**. Both old and new boundary detection give the same positions. The issue is that in single-pass, the model predicts a different token (`'.\n\n'`) than in multi-pass (`'.'`) at each response boundary. The merged token's log-prob includes a small separator component that is not part of the response content.

This cannot be fixed without abandoning single-pass computation, because the token the model predicts is determined by the full context's tokenization.

## Impact on Results

- **Magnitude**: ~0.001 bits/byte per position, ~6% of total curve decline
- **Direction**: Systematic — slightly increases apparent surprise at boundaries
- **Shape**: Does not affect curve shape (exponential vs sigmoidal). The multi-pass curve shows the same exponential decay for Qwen at high m.
- **Cross-model comparison**: GPT-2 is not affected (no merges at these boundaries), so direct numerical comparison of Qwen vs GPT-2 single-pass curves has a ~0.001 bits/byte systematic offset.

## Resolution

**Adopted approach**: Keep single-pass with offset-mapping boundary detection (more principled code) and document the artifact as a caveat.

**Code change**: `_find_response_boundaries` now uses `return_offsets_mapping=True` from the tokenizer to derive boundaries directly from the full tokenization's character offsets, rather than progressive re-tokenization. This is more robust even though outputs match for tested cases.

**Caveat for paper**: Single-pass computation scores the token predicted by the model at each position. When BPE merges response-terminal characters with separator characters (e.g., `'.\n\n'` in Qwen's vocabulary), the scored token includes a small separator component. Empirically, this contributes < 0.02 bits/byte per position (< 6% of total curve decline), verified by comparison against the multi-pass baseline.

**Validation**: The multi-pass implementation (`compute_progressive_surprise_curve`) is retained as a ground-truth reference. The existing `test_single_pass.py::TestSingleVsMultiPass` validates agreement to within 1e-4 bits for GPT-2 (no merges).

## Date

2025-03-11
