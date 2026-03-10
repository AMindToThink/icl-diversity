# Mode Count Experiment Report

## Overview

We tested how the number of distinct response modes (m) affects the ICL diversity metric's a_k curve shape and derived quantities (E, C, D, σ_ℓ). This is the core controlled experiment for the metric: synthetic responses with known mode structure, varying m while holding total response count n fixed.

## Design

- **Modes**: 50 format-based generators (haiku, code, recipe, legal disclaimer, thesis statement, etc.) on the topic "rain"
- **Mode counts**: m ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
- **Total responses**: n = 12 (fixed across all m; max that fits GPT-2's 1024-token context for all mode combinations)
- **Outer draws**: 50 (each independently selects m modes and generates n responses)
- **Inner permutations**: 20 per draw (response ordering variance, handled by `compute_icl_diversity_metrics`)
- **Base model**: GPT-2

### Architecture: Single Source of Truth

All metrics (E, E_rate, C, σ_ℓ, D, D_rate, C_±, D_±) come directly from `compute_icl_diversity_metrics` in core.py — no manual re-computation. The experiment script's role is limited to:
- **Outer randomness**: selecting m modes and generating responses (50 draws)
- **Inner randomness**: delegated to core via `n_permutations=20`

This design fixes three bugs in the previous version:
1. E was computed as `a_1 - a_n` instead of `Σ(a_k - a_n)` (Eq 6)
2. C was computed as `1 - a_n/a_1` instead of `2^(-mean_h)` (Eq 14)
3. σ_ℓ was not computed at all (should be std of per-byte unconditional surprises, Eq 22)

## Figures

### Per-Panel a_k Curves (`figures/mode_count/ak_curves_by_m.png`)

![Per-panel a_k curves](../figures/mode_count/ak_curves_by_m.png)

Per-panel a_k curves for each mode count m ∈ {1, ..., 10}, with n = 12 responses fixed across all m. Each panel shows 5 sample draws (thin lines) and the mean ± 1 SD band across all 50 draws. All curves share the same x-axis (k = 1..12), enabling direct shape comparison. At low m, curves drop steeply from the first response. At high m, curves are flatter — the base model gains less from each additional response because mode coverage per response is lower. Base model: GPT-2 (1024-token context).

### Overlay a_k Curves (`figures/mode_count/ak_curves_overlay.png`)

![Overlay a_k curves](../figures/mode_count/ak_curves_overlay.png)

Overlay of permutation-averaged a_k curves for all m values on shared axes, with ±1 SD bands across 50 draws. All curves span k = 1..12 (fixed n = 12). Higher m values start at similar a_1 but decay less, resulting in higher asymptotic surprise. The m = 1 curve shows the steepest descent — one mode is fully learnable from few examples. The m = 10 curve remains relatively flat, consistent with the base model seeing too few examples per mode (12/10 ≈ 1.2) to learn any pattern well. Base model: GPT-2.

### Aggregate Metrics (`figures/mode_count/metrics_vs_m.png`)

![Metrics vs m](../figures/mode_count/metrics_vs_m.png)

Aggregate ICL diversity metrics as a function of mode count m. Top row: E and E_sig (excess entropy using a_n vs sigmoid-fitted a_∞), D and D_sig (diversity scores), C (coherence), and σ_ℓ (coherence spread). Bottom row: a_n (last curve point) and summary table. C is flat across m, as expected: C = 2^(−mean_h) depends only on unconditional per-response surprise, which is independent of mode count (all modes are drawn from the same pool). The C panel serves as a sanity check, not a finding. The shrinking error bars at higher m reflect the law of large numbers — more distinct modes in each draw stabilize the mean surprise. σ_ℓ increases monotonically with m, confirming that more modes produce higher per-response surprise variance. Base model: GPT-2, n = 12.

### Sigmoid Fits (`figures/mode_count/sigmoid_fits.png`)

![Sigmoid fits](../figures/mode_count/sigmoid_fits.png)

Four-parameter sigmoid fits (a_∞, A, k₀, β) to a_k curves for each m value. Shows 5 sample runs per m with data points and dashed fit lines. Fits are strong for low m (near-exponential decay) but weaker for high m where curves are noisy and nearly flat. Base model: GPT-2.

### Fit Parameters vs m (`figures/mode_count/fit_params_vs_m.png`)

![Fit parameters vs m](../figures/mode_count/fit_params_vs_m.png)

Sigmoid fit parameters as a function of mode count m. k₀ (inflection point) increases with m, consistent with H4 — more modes delay the onset of diminishing returns. Fitted a_∞ increases with m (H3 supported). Error bars show ±1 SD across 50 draws. Base model: GPT-2.

## Results

All metrics computed by `compute_icl_diversity_metrics` (core.py). E = Σ(a_k - a_n) (Eq 6), C = 2^(-mean_h) (Eq 14), σ_ℓ = std of per-byte unconditional surprises (Eq 22), D = C × E. 50 outer draws per m, 20 inner permutations each.

| m | E (bits) | E_rate (bits/byte) | C | σ_ℓ | D (bits) | a_n (bits) |
|---|----------|-------------------|---|-----|----------|------------|
| 1 | 520 ± 223 | 3.558 ± 1.227 | 0.381 ± 0.088 | 0.114 ± 0.057 | 193 ± 82 | 29 ± 24 |
| 2 | 598 ± 246 | 4.215 ± 1.510 | 0.368 ± 0.074 | 0.291 ± 0.152 | 220 ± 98 | 53 ± 25 |
| 3 | 570 ± 200 | 4.378 ± 1.319 | 0.371 ± 0.061 | 0.338 ± 0.143 | 211 ± 79 | 74 ± 31 |
| 4 | 469 ± 229 | 3.587 ± 1.651 | 0.367 ± 0.051 | 0.388 ± 0.146 | 171 ± 82 | 99 ± 30 |
| 5 | 383 ± 184 | 3.110 ± 1.480 | 0.365 ± 0.046 | 0.390 ± 0.127 | 137 ± 66 | 116 ± 34 |
| 6 | 423 ± 260 | 3.278 ± 1.977 | 0.366 ± 0.042 | 0.398 ± 0.129 | 154 ± 98 | 124 ± 39 |
| 7 | 331 ± 227 | 2.610 ± 1.728 | 0.363 ± 0.042 | 0.402 ± 0.130 | 117 ± 80 | 136 ± 32 |
| 8 | 288 ± 204 | 2.077 ± 1.651 | 0.361 ± 0.037 | 0.428 ± 0.115 | 102 ± 72 | 146 ± 30 |
| 9 | 177 ± 189 | 1.618 ± 1.545 | 0.360 ± 0.035 | 0.437 ± 0.107 | 63 ± 66 | 163 ± 27 |
| 10 | 141 ± 205 | 1.231 ± 1.609 | 0.359 ± 0.030 | 0.442 ± 0.109 | 49 ± 72 | 172 ± 29 |

## Hypothesis Evaluation

**H1 (Curve Shape Transition): Partially supported.** Low-m curves show steep exponential-like decay. High-m curves are flatter. The full sigmoidal shape with a flat initial plateau is not clearly visible, likely because n = 12 is too short relative to m = 10 (only 1.2 responses per mode) to observe the inflection.

**H2 (E Monotonicity): Non-monotonic, then decreasing.** E peaks around m = 2–3, then decreases. With corrected E (Σ(a_k - a_n) rather than a_1 - a_n), the relationship is more nuanced: at very low m the total area under the curve grows as modes add learnable structure, but eventually more modes with fewer responses per mode reduce total learnable redundancy.

**H3 (Asymptote Rises with m): Supported.** a_n increases monotonically from ~29 bits (m=1) to ~172 bits (m=10). More modes → higher residual surprise even after full conditioning.

**H4 (Sigmoid Inflection Point): Partially supported.** Fitted k₀ increases with m, as predicted. At high m, k₀ approaches the edge of the data (n = 12), suggesting the sigmoid is not fully resolved.

**H5 (Extrapolation Quality): Inconclusive.** Would require higher-n runs for ground truth a_∞ comparison.

**H6 (Sign Consistency): Partially supported.** E and D generally decrease with m for m ≥ 3, consistent with the inverse relationship between diversity and learnable structure.

**H7 (σ_ℓ Increases with m): Supported.** σ_ℓ increases monotonically from 0.114 (m=1) to 0.442 (m=10). More diverse response sets produce higher variance in per-response unconditional surprise, as expected — responses from different modes have different baseline predictability under the base model.

## Interpretation

The corrected metrics reveal a richer picture than the previous buggy implementation:

1. **E peaks at m ≈ 2–3**: With Σ(a_k - a_n) instead of a_1 - a_n, E captures the *total area* of learnable structure in the curve. A moderate number of modes maximizes this: enough diversity to generate a wide curve, but enough repetition for the base model to learn patterns.

2. **C is flat (sanity check, not a finding)**: C = 2^(−mean_h) depends only on unconditional per-response surprise, which is independent of mode count — each response is scored against the prompt alone, with no inter-response conditioning. C should be flat, and it is. The shrinking error bars at higher m are just the law of large numbers: more distinct modes per draw stabilize the mean unconditional surprise.

3. **σ_ℓ is the strongest signal**: The monotonic increase in coherence spread (0.114 → 0.442) is the clearest indicator of diversity. More modes → more variance in how predictable individual responses are.

4. **a_n is also monotonic**: The asymptotic surprise increases steadily with m, confirming that more modes leave more residual unpredictability.

## Limitations

- **n = 12 is small**: GPT-2's 1024-token context limits n. Larger models (Qwen 2.5-32B with 32K context) would allow n = 20+ and better curve resolution, especially for high m.
- **Mode quality varies**: Some modes (thesis statements) are much longer than others (fortune cookies), creating per-mode token count variance.
- **High variance at extreme m**: At m = 9–10, SD exceeds the mean for E and D, reflecting the fundamental difficulty of learning 10 modes from 12 examples.

## Next Steps

1. Run with Qwen 2.5-32B (n = 20, m up to 25) for better curve resolution
2. Test whether E's relationship to m changes with larger n/m ratios (e.g., n = 100, m = 1..20)
3. Compare sigmoid-extrapolated a_∞ against empirical values from high-n runs
4. Investigate whether σ_ℓ or a_n provides a more robust diversity signal than D
