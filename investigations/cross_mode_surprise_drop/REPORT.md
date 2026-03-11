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

## What We Ruled Out

- **Implementation bug**: Single-pass matches multi-pass (experiment 1).
- **Boundary detection error**: "Response X:" tokens are correctly excluded from response scoring (code review).
- **Positional bias**: Unrelated context increases surprise, not decreases it (experiment 5).

## What Remains Unexplained

The core question is unanswered: **Why does Qwen2.5-3B show exponential decay rather than sigmoidal shape at high m?**

Specific puzzles:
1. The a_2→a_3 per-byte drop (0.060) exceeds the a_1→a_2 drop (0.041) at m=10. No proposed mechanism explains increasing drops.
2. The decay continues monotonically through all 20 positions, even though by position ~10, the model has seen one response from each of the 10 modes and subsequent responses are repetitions — which should be where the sigmoid *starts* dropping, not where it's already well into its tail.
3. GPT-2 produces the correct shape despite being a weaker model. Whatever Qwen is doing differently, it's not just "better ICL."

### Possible directions for future investigation
- Test whether the cross-mode learning effect (experiment 2) is systematic across many mode pairs, not just lab+philosophy.
- Compare Qwen's curve shape with responses that are truly independent (different topics, not just different formats on "rain").
- Check whether Qwen2.5-3B-Instruct shows the same pattern (to distinguish pretraining vs instruction-tuning effects).
- Test with intermediate model sizes (e.g., Qwen2.5-0.5B, Qwen2.5-1.5B) to find where the shape transition occurs.

## Date

2026-03-11
