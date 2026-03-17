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

*Note: An earlier version of this experiment used the same sample index for both context and target, inflating diagonal values ~2× via self-prediction. The results below use different samples for context and target (context sample = (k+1) % M).*

```
Diagonal (same-mode):   mean=60.5 ± 14.4 bits, range [26.9, 154.3]
Off-diagonal (cross):   mean=+1.9 ± 2.5 bits,   range [-19.3, +23.6]
Fraction off-diag > 0:  63.8%
```

The diagonal mean (60.5 bits) is consistent with the observed m=1 first marginal drop (~62 bits from the 1k-draw experiment), validating the measurement.

**Key findings**:

1. **Cross-mode surprise reduction is pervasive and positive on average** (+1.9 bits). 64% of all cross-mode pairs show a positive surprise reduction. This is NOT what independent modes should look like — the expected value for truly independent modes is 0.

2. **The matrix is highly asymmetric** (mean |asymmetry| = 5.6 bits). H_matrix_3 is **falsified**. The asymmetry follows a clear pattern: technical/structured modes (math_stats, json_data, python_code) informing prose modes, not the reverse.

3. **Some modes are consistently informative** (H_matrix_2 confirmed). Top row means: math_stats (+13.1), letter (+9.5), json_data (+9.0). These are long, structured responses that calibrate the model's expectations about the diversity and complexity of responses.

4. **Some modes consistently benefit from context** (column means): letter (+16.3), math_stats (+11.3), json_data (+10.3). These tend to be longer responses with more tokens to predict.

5. **Short/unusual modes are hurt by technical context**: scientific_fact → haiku = −19.3 bits, math_stats → haiku = −16.1 bits. After seeing technical content, the model is *less* prepared for a haiku.

**Top cross-mode pairs by surprise reduction**:
```
math_stats → json_data:       +23.6 ± 3.6 bits
haiku → song_lyrics:          +23.4 ± 1.7 bits
math_stats → python_code:     +22.6 ± 6.7 bits
python_code → math_stats:     +19.2 ± 6.5 bits
json_data → python_code:      +16.8 ± 4.2 bits
```

**Relevance to core question**: **High, but requires quantitative nuance** (see "Quantitative decomposition" below). The pervasive positive off-diagonal mean (+1.9 bits) contributes to the a_k curve drop from position 1, but the diagonal (same-mode) contribution is larger despite the low probability of mode repetition.

### 8. Token-level attribution (`08_token_attribution.py`)

**Purpose**: For the most interesting pairs identified from the matrix, compute per-token surprise reduction to localize *where* in the target the model benefits from cross-mode context. Averages over 5 different prefix responses per pair.

**Result** (9 pairs: 5 top cross-mode, 2 bottom, 2 same-mode controls — pairs auto-selected from corrected pairwise matrix):

| Pair | Total Δ (bits) | First-quarter fraction |
|------|----------------|----------------------|
| math_stats → json_data | +20.3 | 47% |
| haiku → song_lyrics | +26.7 | 15% |
| math_stats → python_code | +14.3 | **140%** (offset by negative later tokens) |
| python_code → math_stats | +29.2 | **68%** |
| json_data → python_code | +15.1 | **108%** (offset by negative later tokens) |
| song_lyrics → historical_fact (negative) | −20.0 | 32% |
| scientific_fact → haiku (negative) | −15.2 | 46% |
| math_stats → math_stats (same) | +139.1 | 44% |
| letter → letter (same) | +50.6 | 21% |

**Key findings**:

1. **H_token_1 is partially confirmed**: For technical→technical pairs (math_stats → python_code: 140%, json_data → python_code: 108%, python_code → math_stats: 68%), the surprise reduction is heavily front-loaded — the model benefits most at the first few tokens. But for creative pairs (haiku → song_lyrics: 15%), the benefit is spread throughout.

2. **The front-loading pattern depends on mode similarity**: When context and target share structural features (technical → technical), the model calibrates early expectations about format and vocabulary. When they share only topic (haiku → song_lyrics), the benefit is distributed across content tokens throughout.

3. **Negative pairs show distributed damage**: When context *hurts* (song_lyrics → historical_fact: −20 bits, scientific_fact → haiku: −15 bits), the damage is spread across the response — the model's expectations are miscalibrated throughout, not just at the start.

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

The off-diagonal mean of +1.9 bits sounds small next to the diagonal mean of 60.5 bits. But the a_k curve drop depends on the *expected* reduction at each position, which mixes diagonal and off-diagonal contributions weighted by mode-repetition probability.

### Simple model

At m=10 with n=20 (2 responses per mode), each response has exactly one mode-mate among the other 19 responses. Response k's mode-mate has probability **(k-1)/(n-1)** of being somewhere among positions 1..k-1. This grows with k: 5% at position 2, 21% at position 5, 47% at position 10.

The expected **cumulative** surprise reduction at position k (compared to unconditional) comes from two sources:

1. **Off-diagonal**: Each of the k-1 prior cross-mode responses contributes ~1.9 bits. Cumulative: **(k-1) × 1.9** bits.
2. **Diagonal**: If the mode-mate is among positions 1..k-1 (probability (k-1)/19), the extra reduction beyond off-diagonal is (60.5 - 1.9) = 58.6 bits. Expected cumulative diagonal contribution: **(k-1)/19 × 58.6** bits.
3. **Total expected cumulative drop at position k**: (k-1) × 1.9 + (k-1)/19 × 58.6 = **(k-1) × 5.0** bits.

This predicts the cumulative drop is **linear in k** (constant 5.0 bits/step), with the diagonal contributing 62% and the off-diagonal 38%.

### Comparison with observations

The observed Qwen2.5-3B mean a_k drops at m=10 (in total bits, ~130 bytes/response):

```
step        predicted    observed
a_1→a_2     5.0 bits     7.4 bits
a_2→a_3     5.0 bits     3.5 bits
a_3→a_4     5.0 bits     4.5 bits
a_4→a_5     5.0 bits     2.8 bits
a_5→a_6     5.0 bits     3.4 bits
```

The model predicts constant 5.0 bits/step; the observed first step (~7.4 bits) overshoots slightly, while later steps (~3 bits) fall below. The discrepancy grows with k because the simple model assumes each prior response contributes independently, but in reality information overlaps (diminishing marginal returns).

Note the simple model actually predicts a **linear** cumulative drop (constant marginal), not an exponential one. The observed *decelerating* drops (large early, smaller later) suggest that the first few responses provide most of the format/topic calibration, and additional responses have diminishing marginal value. This saturation effect is what gives the curve its exponential-looking shape.

### The additive model fails catastrophically (`09_predicted_vs_observed.py`)

The additive model (each response removes a fixed number of bits) fails in two ways:

**At m=10**: The additive pairwise model predicts 622 bits of cumulative drop (a_1 to a_20), but the observed drop is only **58.8 bits** — a **10.6× overprediction**.

More critically, it predicts the **wrong shape**: marginal drops should **increase** with k (more repetitions become likely at later positions), but the observed marginal drops **decrease** with k.

### Fractional (multiplicative) model

The additive model's failure suggests a different framing: each response should resolve a **fraction** of remaining uncertainty rather than a fixed number of bits. The pairwise matrix gives us these fractions:

- `r_offdiag = offdiag / a_1 = 1.9 / 136.6 = 1.4%` — cross-mode fraction resolved per step
- `r_diag = diag / a_1 = 60.5 / 136.6 = 44.3%` — same-mode fraction resolved per step

The model: `a_{k+1} = a_∞ + (a_k - a_∞) × (1 - r_k)`, where `r_k = r_offdiag + P(mate at k) × (r_diag - r_offdiag)` and `a_∞` is the irreducible floor.

The floor `a_∞ = 7.1 bits` is estimated from the m=1 asymptote (mean of positions 15-20 where same-mode saturation is complete). This is independently measured, not fitted to the curves being predicted.

**Results**:

| m | Observed drop | Fractional + floor | Ratio | Additive | Ratio |
|---|--------------|-------------------|-------|----------|-------|
| 1 | 127.5 | 129.8 | **1.0x** | 1149.7 | 9.0x |
| 2 | 121.2 | 131.0 | 1.1x | 1091.1 | 9.0x |
| 5 | 88.1 | 129.0 | 1.5x | 915.2 | 10.4x |
| 10 | 58.8 | 129.0 | 2.2x | 622.2 | 10.6x |

**What the fractional model gets right**:
1. **Shape**: Produces exponentially decaying marginals (matching observation), not increasing marginals (additive)
2. **m=1**: Nearly perfect prediction (1.0x) — validates the floor estimate and fractional rate
3. **First few steps**: First 2-3 positions match well at all m values

**What the fractional model gets wrong**:
1. **Predicts the same total drop (~130 bits) for all m values**. By k=20, the cumulative fractional decay drives the curve to the floor regardless of mode composition. The real data shows total drops ranging from 128 bits (m=1) to 59 bits (m=10).
2. **Decays too fast at high m**. At m=10, the predicted a_20 = 7.6 bits but observed = 77.8 bits.

**Why the fractional model fails at high m**: The fractional rate `r_diag = 44%` is measured from a *single* same-mode encounter in isolation. The model applies this same 44% to every same-mode encounter, but the second same-mode response should resolve *less* than 44% of the remaining uncertainty because much of the "easy" information (format, vocabulary, topic specifics) was already extracted by the first encounter. The pairwise matrix only measures first-encounter effects and cannot predict multi-step saturation dynamics.

### Revised understanding

Neither model adequately predicts the a_k curve from pairwise measurements alone:

- The **additive** model gets the first step right but predicts the wrong shape (increasing marginals, 10x overprediction)
- The **fractional** model gets the shape right but can't distinguish m values (constant ~130-bit total drop)

The core limitation is that the pairwise matrix measures **single-encounter** information transfer. Real a_k curves involve **repeated encounters** with overlapping information, where each subsequent encounter has diminishing marginal returns. A complete model would need to account for how much of the "resolvable uncertainty" is shared between different responses — information that the pairwise matrix does not capture.

What the pairwise matrix *does* successfully explain:
1. **The first step at each m** — well-calibrated, validates the measurement
2. **Why Qwen drops and GPT-2 doesn't** — the sign of the off-diagonal (Qwen: +1.9, GPT-2: −3.7) determines whether the curve drops or rises initially
3. **The qualitative shape** — fractional decay produces the right exponential-looking curve

## What We Now Understand

The core question has two layers:

**Layer 1 — Why does the curve drop at all?** The pairwise matrix shows cross-mode surprise reduction is pervasive (+1.9 bits, 62% of pairs positive). Token attribution reveals two mechanisms:

1. **Format calibration** (front-loaded): Seeing any response recalibrates expectations about what follows "Response B:".
2. **Topic vocabulary priming** (distributed): All responses are about rain, so rain-related vocabulary gets primed cross-mode.

**Layer 2 — Why is the drop so steep (exponential, not sigmoidal)?** The fractional model explains the shape: each response resolves ~1.4% (cross-mode) or ~44% (same-mode) of remaining resolvable uncertainty. This produces exponentially decaying marginals — large early drops that shrink as uncertainty is depleted. The additive model incorrectly predicts increasing marginals. The fractional model correctly predicts the first few steps and the qualitative shape, but overpredicts the total drop at high m because the pairwise-measured fractional rates don't account for diminishing returns from repeated encounters.

**Why GPT-2 doesn't show this**: GPT-2 actually has a *larger* diagonal (83 vs 60.5 bits — better same-mode ICL) but a **negative** off-diagonal (−3.7 bits). At m=10, the expected first-step net drop is only +0.9 bits (cross-mode damage almost cancels diagonal gain). The sigmoid shape at high m in GPT-2 is not evidence that modes are independent — it's evidence that cross-mode context actively interferes with GPT-2's predictions, producing a flat or slightly rising early curve until enough same-mode repetitions accumulate to overcome the cross-mode penalty.

**Implication for the metric**: The ICL diversity metric assumes a_k drops *only* from mode redundancy. But with capable base models, a_k drops from three sources: (1) same-mode repetition (the intended signal), (2) cross-mode format calibration, and (3) cross-mode topic priming. Sources 2-3 cause the metric to underestimate diversity. Stronger θ → more cross-mode learning → lower measured diversity for the same response set.

### 10. GPT-2 pairwise matrix and token attribution (`07/08_*_gpt2.py`)

**Purpose**: Test the hypothesis that GPT-2's flat early curves at high m are due to a smaller diagonal (weaker same-mode ICL).

**Result**: The hypothesis is **completely wrong**. GPT-2's matrix is qualitatively different from Qwen's:

| Metric | Qwen2.5-3B | GPT-2 |
|--------|-----------|-------|
| Diagonal (same-mode) | 60.5 ± 14.4 bits | **83.0 ± 20.0 bits** |
| Off-diagonal (cross-mode) | +1.9 bits | **−3.7 bits** |
| Off-diagonal > 0 | 64% | **31%** |

GPT-2's diagonal is **larger** than Qwen's (83 vs 60.5 bits), meaning it's *better* at recognizing same-mode responses in isolation. But its off-diagonal is **negative** (−3.7 bits) — cross-mode context actively *hurts* GPT-2, increasing surprise for subsequent responses.

**Expected first-step drop at m=10**:

```
              Cross-mode    Same-mode    Total
              (18/19 ×)     (1/19 ×)
Qwen2.5-3B:  +1.8 bits     +3.1 bits    = +4.9 bits  → drops
GPT-2:       -3.5 bits     +4.4 bits    = +0.9 bits  → barely drops
```

At m=10 with 95% probability of seeing a cross-mode response first, GPT-2 *loses* 3.5 bits from cross-mode confusion, almost completely canceling the 4.4-bit expected diagonal gain. Qwen *gains* 1.8 bits from cross-mode priming on top of its 3.1-bit diagonal gain. This difference in off-diagonal sign — not diagonal magnitude — explains the curve shape difference.

**GPT-2 token attribution**: Cross-mode pairs show mixed effects. Some pairs with positive total deltas (python_code→numbered_list: +37 bits, song_lyrics→haiku: +25 bits) show the benefit concentrated in specific tokens. The negative pairs (scientific_fact→python_code: −23 bits, recipe→math_stats: −41 bits) show damage spread throughout, confirming that GPT-2 is confused by cross-mode context broadly. Same-mode controls are very large (math_stats→math_stats: +257 bits, python_code→python_code: +121 bits), consistent with GPT-2's larger diagonal.

**Relevance to core question**: **Critical**. This overturns the original hypothesis and provides the real explanation:
- The curve shape difference is driven by the **sign of the off-diagonal**, not the magnitude of the diagonal.
- GPT-2's negative off-diagonal means cross-mode responses CANCEL diagonal benefits → flat early curve.
- Qwen's positive off-diagonal means cross-mode responses AMPLIFY learning → monotonic decay.

## Remaining Questions

1. **Why does GPT-2 have negative off-diagonal?** GPT-2 apparently treats cross-mode context as noise that interferes with prediction. Qwen learned to extract cross-mode information (format calibration, topic priming) during pretraining.
2. **Different topics**: If responses were about completely different topics (not all rain), would Qwen's off-diagonal go negative too?
3. **Model size scaling**: At what model size does the off-diagonal transition from negative to positive?
4. **Multi-encounter saturation**: The fractional model applies the same rate to every encounter, but repeated same-mode responses should have diminishing fractional rates. A model with encounter-dependent rates could bridge the gap between the fractional model's shape accuracy and its failure to distinguish m values.

## Hypothesis Evaluation

| Hypothesis | Result |
|-----------|--------|
| H_matrix_1 (diagonal dominance) | **Confirmed**. Diagonal mean = 60.5 ± 14.4 bits >> off-diagonal mean = 1.9 ± 2.5 bits. |
| H_matrix_2 (informative modes) | **Confirmed**. math_stats (+13.1), letter (+9.5), json_data (+9.0) have highest row means. |
| H_matrix_3 (approximate symmetry) | **Falsified**. Mean |asymmetry| = 5.6 bits (2.9× the off-diagonal mean). |
| H_token_1 (front-loaded attribution) | **Partially confirmed**. True for technical→technical pairs (68-140%), not for creative→creative (15-21%). |

## Date

2026-03-11
