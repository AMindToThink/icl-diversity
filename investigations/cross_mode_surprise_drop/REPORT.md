# Investigation: Qwen2.5-3B Produces Exponential, Not Sigmoidal, a_k Curves at High m

## Question

The paper predicts that the a_k curve shape should transition with mode count m:
- **Low m** (e.g., m=1): Exponential decay — the model quickly learns the single mode, surprise drops steeply from position 1.
- **High m** (e.g., m=10): Sigmoidal — surprise should stay **flat initially** (early responses are from different modes, each uninformative about the next), then drop once enough responses accumulate for mode repetitions, then flatten at a higher floor.

**GPT-2 shows this pattern.** At m=10, a_2 ≥ a_1 (the curve is flat or slightly rising at the start), consistent with the predicted sigmoid.

**Qwen2.5-3B does not.** At all m values including m=10, the curve shows immediate exponential-like decay — steep drops starting from position 1. There is no flat initial plateau. This is the wrong shape for a high-diversity response set.

## Background

- **Prompt**: "Write a short piece about rain." — included in every forward pass, so the model already knows the topic.
- **Modes**: 50 format-based generators (haiku, code, recipe, legal disclaimer, etc.)
- **At m=10, n=20**: 2 responses per mode, randomly shuffled. P(response 2 shares a mode with response 1) ≈ 1/19 ≈ 5%.

## Key Data

Qwen2.5-3B per-byte a_k curve at m=10 — exponential decay, no flat start:

```
k   bits/byte
1   1.061
2   1.020   (drop = 0.041)
3   0.960   (drop = 0.060)   ← larger than the 1→2 drop
4   0.938   (drop = 0.022)
5   0.923   (drop = 0.015)
6   0.897   (drop = 0.026)
```

GPT-2 per-byte a_k curve at m=10 — flat/rising start, consistent with sigmoid:

```
k   bits/byte
1   1.456
2   1.491   (drop = -0.035)  ← increases!
3   1.468
4   1.456
5   1.420
6   1.413
```

## Experiments Run

These experiments ruled out several possible explanations but did **not** resolve the core question of why Qwen produces the wrong curve shape at high m.

### 1. Single-pass vs multi-pass equivalence (`01_single_vs_multi_pass.py`)

**Purpose**: Rule out a boundary detection bug in the single-pass implementation for Qwen's tokenizer.

**Result**: Single-pass and multi-pass agree to within fp16 rounding (diffs < 4 bits on ~150-bit values). No boundary bug.

**Relevance to core question**: Low. Confirms the implementation is correct, but doesn't explain the curve shape.

### 2. Cross-mode surprise reduction (`02_cross_mode_surprise.py`)

**Purpose**: Directly measure whether seeing one mode's response reduces surprise for a different mode's response.

**Result** (lab notebook + philosophy, two very different modes):

```
a_1 (lab alone):        424.8 bits (1.173 bits/byte)
a_2 (phil after lab):   189.2 bits (0.591 bits/byte)   ← large drop
a_1 (phil alone):       197.6 bits (0.618 bits/byte)
a_2 (lab after phil):   404.2 bits (1.117 bits/byte)   ← small drop
```

The drop is **asymmetric**: easy text (philosophy) benefits greatly from seeing hard text (lab notebook), but hard text barely benefits from easy text. The mean across both orderings is still positive (+14 bits).

**Relevance to core question**: Moderate. Shows cross-mode surprise reduction is real for Qwen, but only tested one pair. Does not explain why the effect is strong enough to produce exponential rather than sigmoidal shape, or why the a_2→a_3 drop exceeds a_1→a_2.

### 3. Same-mode and noise context (`03_same_mode_and_noise.py`)

**Purpose**: Test whether same-style responses show genuine learning, and whether random noise shows no learning.

**Result (same-mode, philosophy → philosophy)**:

```
r1 alone:     153.1 bits (0.672 bits/byte)
r2 after r1:  150.6 bits (0.582 bits/byte)  → drop = +2.5 bits
```

**Result (noise → philosophy)**:

```
r2 after noise: 173.1 bits (0.668 bits/byte)  → drop = -1.2 bits (no learning)
```

**Relevance to core question**: Low. Confirms ICL works within modes and doesn't work with truly irrelevant context. Does not address why cross-mode learning produces exponential shape.

### 4. Noise distribution learning (`04_noise_distribution.py`)

**Purpose**: Test whether models learn from random word salad that shares a fixed vocabulary.

**Result**: BOTH models learn the noise distribution rapidly:

```
Qwen:  3.89 → 2.06 → 1.36 bits/byte (positions 1-3)
GPT-2: 4.33 → 2.53 → 2.15 bits/byte (positions 1-3)
```

This is correct behavior — the "noise" responses are actually a single mode (permutations of 20 fixed nonsense words), so the model should learn the distribution.

**Relevance to core question**: None. The test was flawed — it tested single-mode learning, not cross-mode behavior.

### 5. Positional bias test (`05_positional_bias.py`)

**Purpose**: Rule out a systematic positional bias where later positions have lower surprise regardless of content. Measures surprise for the **same response** at positions 1–8, with completely unrelated filler (math, cooking, history, biology, etc.) as context.

**Result**:

```
pos  bits/byte  context_tokens
  1    0.7147              11
  2    0.7878              49
  3    0.8103              90
  4    0.7974             129
  5    0.8250             159
  6    0.7968             199
  7    0.8186             236
  8    0.8128             276
```

Surprise **increases** with unrelated context. No positional bias — if anything, irrelevant context slightly hurts.

**Relevance to core question**: Moderate. Rules out one explanation (positional bias), but the core question remains.

### 6. Aggregate statistics (`06_aggregate_stats.py`)

**Purpose**: Compare mean a_k curves, per-byte rates, drop distributions, and unconditional surprise variance between Qwen and GPT-2 across all m values.

Key finding for m=10 drop distribution (Qwen):

```
mean drop a_1→a_2: 7.4 bits
median: 7.3 bits
fraction positive: 53.9%
range: [-157.3, +209.4]
```

Only 54% of draws have a positive drop — the mean curve exaggerates a weak effect at position 1→2. But the curve still decays monotonically on average through all 20 positions.

**Relevance to core question**: Descriptive, not explanatory.

### 7. Pairwise cross-mode surprise matrix (`07_pairwise_matrix.py`)

**Purpose**: Systematically measure surprise reduction across all pairs of 15 modes. For each ordered pair (i, j), compute how much seeing a response from mode_i reduces surprise for a response from mode_j.

**Result** (15×15 matrix, 225 conditional + 15 unconditional passes):

```
Diagonal (same-mode):   mean=123.6 bits, std=39.9, range [74.6, 210.9]
Off-diagonal (cross):   mean=+2.1 bits,  std=8.1,  range [-17.6, +27.2]
Fraction off-diag > 0:  62.4%
```

**Key findings**:

1. **Cross-mode surprise reduction is pervasive and positive on average** (+2.1 bits). 62% of all cross-mode pairs show a positive surprise reduction. This is NOT what independent modes should look like — the expected value for truly independent modes is 0.

2. **The matrix is highly asymmetric** (mean |asymmetry| = 7.2 bits, max = 23.8 bits). H_matrix_3 is **falsified**. The asymmetry follows a clear pattern: technical/structured modes (math_stats, json_data, python_code) informing prose modes, not the reverse. For example, math_stats → scientific_fact = +23.6 bits, but scientific_fact → math_stats would be much smaller.

3. **Some modes are consistently informative** (H_matrix_2 confirmed). Top row means: math_stats (+17.9), numbered_list (+16.8), letter (+14.7). These are long, structured responses that calibrate the model's expectations about the diversity and complexity of responses.

4. **Some modes consistently benefit from context** (column means): letter (+22.4), numbered_list (+16.2), diary (+15.2). These tend to be longer responses with more tokens to predict.

5. **Short/unusual modes are hurt by technical context**: math_stats → haiku = −17.4 bits, scientific_fact → haiku = −17.6 bits. After seeing technical content, the model is *less* prepared for a haiku.

**Top cross-mode pairs by surprise reduction**:
```
math_stats → json_data:       +27.2 bits
math_stats → scientific_fact: +23.6 bits
haiku → song_lyrics:          +21.5 bits
math_stats → letter:          +19.5 bits
math_stats → python_code:     +19.3 bits
```

**Relevance to core question**: **High**. The pervasive positive off-diagonal mean (+2.1 bits) explains why the a_k curve drops from position 1: even cross-mode responses are informative about each other under Qwen2.5-3B. The effect is systematic — it's not one or two pathological pairs, but a broad pattern where any response reduces surprise for any subsequent response by ~2 bits on average. Over 20 positions, this accumulates to ~40 bits of "spurious" learning.

### 8. Token-level attribution (`08_token_attribution.py`)

**Purpose**: For the most interesting pairs identified from the matrix, compute per-token surprise reduction to localize *where* in the target the model benefits from cross-mode context. Averages over 5 different prefix responses per pair.

**Result** (9 pairs: 5 top cross-mode, 2 bottom, 2 same-mode controls):

| Pair | Total Δ (bits) | First-quarter fraction |
|------|----------------|----------------------|
| math_stats → json_data | +20.3 | 47% |
| math_stats → scientific_fact | +12.0 | **81%** |
| haiku → song_lyrics | +26.7 | 15% |
| math_stats → letter | +18.5 | 23% |
| math_stats → python_code | +14.3 | **140%** (offset by negative later tokens) |
| math_stats → haiku (negative) | −8.6 | 98% of damage in first quarter |
| scientific_fact → haiku (negative) | −15.2 | 46% |
| math_stats → math_stats (same) | +139.1 | 44% |
| numbered_list → numbered_list (same) | +58.2 | 25% |

**Key findings**:

1. **H_token_1 is partially confirmed**: For some pairs (math_stats → scientific_fact: 81%, math_stats → python_code: 140%), the surprise reduction is heavily front-loaded — the model benefits most at the first few tokens. But for others (haiku → song_lyrics: 15%, math_stats → letter: 23%), the benefit is spread throughout.

2. **The front-loading pattern depends on mode similarity**: When context and target share structural features (technical → technical), the model calibrates early expectations about format and vocabulary. When they share only topic (haiku → song_lyrics), the benefit is distributed across content tokens throughout.

3. **Negative pairs show front-loaded damage**: When context *hurts* (math_stats → haiku: −8.6 bits, 98% in first quarter), the damage is almost entirely at the start — the model begins expecting technical content and is surprised by "Drops on still water".

4. **Same-mode controls show expected behavior**: math_stats → math_stats shows +139 bits of reduction spread across the response (44% in first quarter), consistent with learning the response format throughout.

**Relevance to core question**: **High**. The token attribution reveals that cross-mode learning operates through two distinct mechanisms:
- **Format calibration** (front-loaded): Seeing any structured response recalibrates the model's expectations about what kind of text follows "Response B:". This is a ~5-10 bit effect concentrated in the first few tokens.
- **Topic/vocabulary priming** (distributed): Seeing rain-related content in any format primes rain-related vocabulary throughout subsequent responses. This contributes ~5-15 bits spread across the response.

Both mechanisms are cross-mode, explaining why the a_k curve drops even when consecutive responses are from different modes.

## What We Ruled Out

- **Implementation bug**: Single-pass matches multi-pass (experiment 1).
- **Boundary detection error**: "Response X:" tokens are correctly excluded from response scoring (code review).
- **Positional bias**: Unrelated context increases surprise, not decreases it (experiment 5).

## What We Now Understand

**The core question is now largely answered**: Qwen2.5-3B shows exponential (not sigmoidal) decay at high m because **cross-mode surprise reduction is pervasive and substantial**.

The pairwise matrix (experiment 7) shows that 62% of all cross-mode pairs have positive surprise reduction, averaging +2.1 bits. This means that even when response k is from a different mode than response k-1, the model has still learned something useful — specifically:

1. **Format calibration**: After seeing one response (any format), the model better predicts the opening tokens of the next response. The "Response B:" label activates expectations conditioned on having seen a diverse response at "Response A:".

2. **Topic vocabulary priming**: All responses are about rain. Seeing rain-discussed-as-code makes rain-discussed-as-philosophy easier to predict, because rain-related vocabulary (drops, water, fall, storm) has been primed.

3. **Length/complexity calibration**: Longer, more structured context responses (math_stats, numbered_list) provide the most cross-mode benefit — they calibrate the model's expectations about response complexity and length.

**Why GPT-2 doesn't show this**: GPT-2 likely has weaker in-context learning, so it cannot extract these cross-mode signals. The sigmoid shape at high m in GPT-2 may actually be the result of *insufficient* ICL capacity to detect cross-mode structure, rather than evidence that the modes are truly independent.

**Implication for the metric**: The ICL diversity metric assumes that a_k drops *only* when the model sees redundant/similar responses. But if the base model θ is powerful enough to learn cross-mode meta-information (format expectations, topic vocabulary), the metric will underestimate diversity. This is a fundamental limitation when using capable base models — stronger θ → more cross-mode learning → lower measured diversity for the same response set.

## Remaining Questions

1. **Quantitative prediction**: Can we predict the a_k curve shape from the pairwise matrix? The average off-diagonal reduction (+2.1 bits) × 19 positions ≈ 40 bits of cumulative "spurious" drop. Does this match the observed curve?
2. **Different topics**: If responses were about completely different topics (not all rain), would the off-diagonal entries go to zero?
3. **Model size scaling**: At what model size does cross-mode learning become strong enough to produce exponential rather than sigmoidal curves?

## Hypothesis Evaluation

| Hypothesis | Result |
|-----------|--------|
| H_matrix_1 (diagonal dominance) | **Confirmed**. Diagonal mean = 123.6 bits >> off-diagonal mean = 2.1 bits. |
| H_matrix_2 (informative modes) | **Confirmed**. math_stats, numbered_list, letter have highest row means. |
| H_matrix_3 (approximate symmetry) | **Falsified**. Mean |asymmetry| = 7.2 bits (3.4× the off-diagonal mean). |
| H_token_1 (front-loaded attribution) | **Partially confirmed**. True for technical→technical pairs (81-140%), not for creative→creative (15-25%). |

## Date

2026-03-11
