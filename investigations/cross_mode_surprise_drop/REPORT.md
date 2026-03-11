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

**Result** (15×15 matrix with M=5 samples per mode, 1125 conditional + 75 unconditional passes):

```
Diagonal (same-mode):   mean=122.4 ± 11.8 bits, range [86.2, 209.7]
Off-diagonal (cross):   mean=+1.9 ± 2.4 bits,   range [-19.6, +22.7]
Fraction off-diag > 0:  62.4%
```

**Key findings**:

1. **Cross-mode surprise reduction is pervasive and positive on average** (+1.9 bits). 62% of all cross-mode pairs show a positive surprise reduction. This is NOT what independent modes should look like — the expected value for truly independent modes is 0.

2. **The matrix is highly asymmetric** (mean |asymmetry| = 5.6 bits, max = 18.3 bits). H_matrix_3 is **falsified**. The asymmetry follows a clear pattern: technical/structured modes (math_stats, json_data, python_code) informing prose modes, not the reverse. For example, math_stats → scientific_fact = +13.1 ± 1.1 bits, but scientific_fact → math_stats = +14.3 ± 1.0 bits (this particular pair is roughly symmetric, but most are not).

3. **Some modes are consistently informative** (H_matrix_2 confirmed). Top row means: math_stats (+16.8 ± 0.4), letter (+14.3 ± 0.4), numbered_list (+13.7 ± 0.7). These are long, structured responses that calibrate the model's expectations about the diversity and complexity of responses.

4. **Some modes consistently benefit from context** (column means): letter (+21.3 ± 0.4), math_stats (+14.8 ± 0.4), dialogue (+13.1 ± 0.7). These tend to be longer responses with more tokens to predict.

5. **Short/unusual modes are hurt by technical context**: scientific_fact → haiku = −19.6 ± 3.1 bits, math_stats → haiku = −16.3 ± 2.4 bits. After seeing technical content, the model is *less* prepared for a haiku.

**Top cross-mode pairs by surprise reduction**:
```
haiku → song_lyrics:          +22.7 ± 3.0 bits
math_stats → json_data:       +22.6 ± 2.2 bits
math_stats → python_code:     +22.4 ± 6.7 bits
python_code → math_stats:     +18.8 ± 6.9 bits
json_data → python_code:      +16.8 ± 3.8 bits
```

**Relevance to core question**: **High, but requires quantitative nuance** (see "Quantitative decomposition" below). The pervasive positive off-diagonal mean (+1.9 bits) contributes to the a_k curve drop from position 1, but the diagonal (same-mode) contribution is ~3× larger despite the low probability of mode repetition.

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

## Quantitative Decomposition: Does +1.9 bits/pair Explain the Curve?

The off-diagonal mean of +1.9 bits sounds small next to the diagonal mean of 122.4 bits. But the a_k curve drop depends on the *expected* reduction at each position, which mixes diagonal and off-diagonal contributions weighted by mode-repetition probability.

### Simple model

At m=10 with n=20 (2 responses per mode), each response has exactly one mode-mate among the other 19 responses. Response k's mode-mate has probability **(k-1)/(n-1)** of being somewhere among positions 1..k-1. This grows with k: 5% at position 2, 21% at position 5, 47% at position 10.

The expected **cumulative** surprise reduction at position k (compared to unconditional) comes from two sources:

1. **Off-diagonal**: Each of the k-1 prior cross-mode responses contributes ~1.9 bits. Cumulative: **(k-1) × 1.9** bits.
2. **Diagonal**: If the mode-mate is among positions 1..k-1 (probability (k-1)/19), the extra reduction beyond off-diagonal is (122.4 - 1.9) = 120.5 bits. Expected cumulative diagonal contribution: **(k-1)/19 × 120.5** bits.
3. **Total expected cumulative drop at position k**: (k-1) × 1.9 + (k-1)/19 × 120.5 = **(k-1) × 8.2** bits.

This predicts the cumulative drop is **linear in k** (constant 8.2 bits/step), with the diagonal contributing 77% and the off-diagonal 23%.

### Comparison with observations

The observed Qwen2.5-3B mean a_k drops at m=10 (in total bits, ~130 bytes/response):

```
step        predicted    observed
a_1→a_2     8.2 bits     5.3 bits
a_2→a_3     8.2 bits     7.8 bits
a_3→a_4     8.2 bits     2.9 bits
a_4→a_5     8.2 bits     1.9 bits
a_5→a_6     8.2 bits     3.4 bits
```

The model predicts constant 8.2 bits/step; reality averages ~4 bits/step with high variance. The discrepancy is expected — the simple model assumes each prior response contributes independently, but in reality information overlaps (diminishing marginal returns).

Note the simple model actually predicts a **linear** cumulative drop (constant marginal), not an exponential one. The observed *decelerating* drops (large early, smaller later) suggest that the first few responses provide most of the format/topic calibration, and additional responses have diminishing marginal value. This saturation effect is what gives the curve its exponential-looking shape.

### The independent model fails catastrophically (`09_predicted_vs_observed.py`)

Comparing the pairwise prediction with observed 1k-draw mean curves reveals that the independent model is fundamentally wrong:

**At m=10**: The "pairwise w/ growing P(mate)" model predicts 1,241 bits of cumulative drop (a_1 to a_20), but the observed drop is only **58.8 bits** — a **21× overprediction**. Even the constant-rate independent model predicts 156 bits (2.7× overprediction).

More critically, the models predict the **wrong shape**:
- **Predicted**: Marginal drops should **increase** with k (more repetitions become likely at later positions). At k=1 the model predicts 8.2 bits; at k=19 it predicts 122 bits.
- **Observed**: Marginal drops **decrease** with k. The first step drops ~7.4 bits; subsequent steps average ~3 bits; some steps are negative.

The "efficiency" (observed/predicted) drops from ~90% at k=1 to ~5% by k=5 and near 0% thereafter. The pairwise matrix is only predictive of the very first step.

**Why the independent model fails**: Responses do not contribute independent information. The model rapidly **saturates** — seeing the first 1-3 responses provides most of the format calibration and topic priming available, and additional responses (whether same-mode or cross-mode) add almost nothing. The 122-bit same-mode pairwise reduction measures how much a *single* same-mode response helps in isolation, but when there are already 5+ responses in context, the marginal value of a same-mode response is much smaller — most of that information is already captured.

### Revised understanding

The pairwise matrix measures **potential** information transfer in isolation but not **marginal** contribution in a long context. The exponential shape of the a_k curve is NOT well-explained by summing pairwise contributions. Instead, the curve shape reflects:

1. **Rapid early saturation**: The first few responses (regardless of mode) teach the model most of what it can learn about the response format, topic, and length distribution. This produces the large early drops.
2. **Diminishing returns**: Each additional response provides less new information because the context already contains most of the learnable structure.
3. **Mode repetitions provide extra but diminishing boosts**: When a same-mode response finally appears, it does help, but much less than the 122 bits predicted by the isolated pairwise measure.

The difference between Qwen and GPT-2 likely comes from ICL capacity — Qwen saturates quickly (strong early drops) while GPT-2 barely detects cross-mode structure (flat early curve). But this is about the **first few positions**, not the accumulation of pairwise effects across 20 positions.

## What We Now Understand

The core question has two layers:

**Layer 1 — Why does the curve drop at all?** The pairwise matrix shows cross-mode surprise reduction is pervasive (+1.9 bits, 62% of pairs positive). Token attribution reveals two mechanisms:

1. **Format calibration** (front-loaded): Seeing any response recalibrates expectations about what follows "Response B:".
2. **Topic vocabulary priming** (distributed): All responses are about rain, so rain-related vocabulary gets primed cross-mode.

**Layer 2 — Why is the drop so steep (exponential, not sigmoidal)?** The off-diagonal alone (+1.9 bits/step) would produce a gentle linear slope, not the observed convex decay. The steep *early* decay is primarily driven by the **diagonal**: same-mode reduction averaging 122.4 bits, which even at P=(k-1)/19 contributes ~6.3 bits/step in expectation. Combined with the off-diagonal, the simple model predicts ~8.2 bits/step of linear cumulative drop. The observed *decelerating* shape (large early drops, smaller later) comes from diminishing marginal returns — the first few responses provide most of the format/topic calibration, so subsequent responses add less. This saturation is what produces the exponential-looking curve rather than the predicted sigmoid.

**Why GPT-2 doesn't show this**: GPT-2 almost certainly has a much smaller diagonal (weaker same-mode ICL) and near-zero off-diagonal. If its diagonal were ~30 bits, the expected drop per step would be only ~1.6 bits — small enough to be noise, producing the observed flat/rising early curve. The sigmoid shape at high m in GPT-2 is not evidence that modes are independent — it's evidence that GPT-2's ICL is too weak to detect either cross-mode or (at low P) same-mode structure until enough repetitions accumulate.

**Implication for the metric**: The ICL diversity metric assumes a_k drops *only* from mode redundancy. But with capable base models, a_k drops from three sources: (1) same-mode repetition (the intended signal), (2) cross-mode format calibration, and (3) cross-mode topic priming. Sources 2-3 cause the metric to underestimate diversity. Stronger θ → more cross-mode learning → lower measured diversity for the same response set.

### 10. GPT-2 pairwise matrix and token attribution (`07/08_*_gpt2.py`)

**Purpose**: Test the hypothesis that GPT-2's flat early curves at high m are due to a smaller diagonal (weaker same-mode ICL).

**Result**: The hypothesis is **completely wrong**. GPT-2's matrix is qualitatively different from Qwen's:

| Metric | Qwen2.5-3B | GPT-2 |
|--------|-----------|-------|
| Diagonal (same-mode) | 122.4 ± 11.8 bits | **172.4 ± 13.2 bits** |
| Off-diagonal (cross-mode) | +1.9 bits | **−3.8 bits** |
| Off-diagonal > 0 | 62% | **30%** |

GPT-2's diagonal is **larger** than Qwen's (172 vs 122 bits), meaning it's *better* at recognizing same-mode responses in isolation. But its off-diagonal is **negative** (−3.8 bits) — cross-mode context actively *hurts* GPT-2, increasing surprise for subsequent responses.

**Expected first-step drop at m=10**:

```
              Cross-mode    Same-mode    Total
              (18/19 ×)     (1/19 ×)
Qwen2.5-3B:  +1.8 bits     +6.4 bits    = +8.2 bits  → drops
GPT-2:       -3.6 bits     +9.1 bits    = +5.5 bits  → barely drops
```

At m=10 with 95% probability of seeing a cross-mode response first, GPT-2 *loses* 3.6 bits from cross-mode confusion, partially canceling the 9.1-bit expected diagonal gain. Qwen *gains* 1.8 bits from cross-mode priming on top of its 6.4-bit diagonal gain. This difference in off-diagonal sign — not diagonal magnitude — explains the curve shape difference.

**GPT-2 token attribution**: Cross-mode pairs show mixed effects. Some pairs with positive total deltas (python_code→numbered_list: +37 bits, song_lyrics→haiku: +25 bits) show the benefit concentrated in specific tokens. The negative pairs (scientific_fact→python_code: −23 bits, recipe→math_stats: −41 bits) show damage spread throughout, confirming that GPT-2 is confused by cross-mode context broadly.

**Relevance to core question**: **Critical**. This overturns the original hypothesis and provides the real explanation:
- The curve shape difference is driven by the **sign of the off-diagonal**, not the magnitude of the diagonal.
- GPT-2's negative off-diagonal means cross-mode responses CANCEL diagonal benefits → flat early curve.
- Qwen's positive off-diagonal means cross-mode responses AMPLIFY learning → monotonic decay.

## Remaining Questions

1. **Why does GPT-2 have negative off-diagonal?** GPT-2 apparently treats cross-mode context as noise that interferes with prediction. Qwen learned to extract cross-mode information (format calibration, topic priming) during pretraining.
2. **Different topics**: If responses were about completely different topics (not all rain), would Qwen's off-diagonal go negative too?
3. **Model size scaling**: At what model size does the off-diagonal transition from negative to positive?
4. **Diminishing returns**: A model accounting for overlapping information would better predict the observed sublinear accumulation (the independent model overpredicts by 21×).

## Hypothesis Evaluation

| Hypothesis | Result |
|-----------|--------|
| H_matrix_1 (diagonal dominance) | **Confirmed**. Diagonal mean = 122.4 ± 11.8 bits >> off-diagonal mean = 1.9 ± 2.4 bits. |
| H_matrix_2 (informative modes) | **Confirmed**. math_stats (+16.8), letter (+14.3), numbered_list (+13.7) have highest row means. |
| H_matrix_3 (approximate symmetry) | **Falsified**. Mean |asymmetry| = 5.6 bits (2.9× the off-diagonal mean). |
| H_token_1 (front-loaded attribution) | **Partially confirmed**. True for technical→technical pairs (81-140%), not for creative→creative (15-25%). |

## Date

2026-03-11
