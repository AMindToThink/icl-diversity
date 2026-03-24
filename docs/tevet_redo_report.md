# Tevet Redo Report — March 2026

## Background

The original Tevet evaluation (GPT-2, instruct format, discrete E) failed all hypotheses — E was negatively correlated with diversity labels. We hypothesized two causes:

1. **Non-convergence bug**: E was computed as `sum(a_k - a_n)` using the last observed point as a_inf. With only 5 responses, diverse sets have curves still dropping at k=5, so a_5 overestimates a_inf and compresses E toward zero. Constant sets converge by k=3, getting accurate E. This asymmetric truncation creates a spurious negative correlation.

2. **Wrong format for base models**: The instruct-style `"Response A: ..."` framing is meaningless noise to GPT-2/Qwen (base models). Tevet's data is story completions — a completion-style format is more appropriate.

The mode_count experiments (200 draws × 10 mode counts, GPT-2) empirically confirmed that E correctly tracks mode count when computed via exponential curve fitting (`D_fit = C × E_fit`, commit 6cd111c). The Tevet evaluation was run before this fix.

## What We Did

### Code Changes (commit 86e2b70)

1. **Added `format_mode` parameter** (`"instruct"` | `"completion"`) threaded through all of `core.py`. Completion format: `"1. {prompt}{response}\n\n2. {prompt}{response}..."`. Logprobs measured only on completion portions via boundary masking.

2. **Mean curves storage**: Compute script now saves lightweight `*.icl_mean_curves.{tag}.json` per CSV (~6 MB total, committed to git). Contains mean a_k curve, byte counts, and unconditional surprises per sample. Enables offline sigmoid/exponential refitting without re-running the model.

3. **Analysis script with E_fit**: Fits both exponential (3 params) and sigmoid (4 params) to each sample's mean a_k curve. Reports E_fit, D_fit alongside E_discrete, D_discrete. Side-by-side Spearman ρ and OCA.

4. **58 unit tests pass**, including new completion format regression tests and boundary verification.

### Compute Run (commit 8b3cf54)

- **Model**: Qwen/Qwen2.5-3B (base), float16
- **Format**: completion mode
- **Permutations**: 50
- **Batch size**: 8, cuda:0
- **Total**: 13,929 samples across 18 CSVs, 0 skips, 0 errors
- **Run time**: ~11 hours (CPU contention from other processes)
- **Run tag**: `qwen25_completion`

## Results

### Fit Quality

- Exponential: **100% success rate** (9,885/9,885), median R² = 0.959
- Sigmoid: 99.99% success rate (9,884/9,885), but median R² includes -inf values (unstable on some 5-point curves)
- Exponential is clearly the better choice for 5-point curves

### McDiv_nuggets (content diversity ground truth — PRIMARY)

| Task | E_discrete ρ | E_fit_exp ρ | E_fit_sig ρ |
|------|-------------|-------------|-------------|
| prompt_gen (hds) | **-0.490** | **-0.137** | -0.501 |
| resp_gen (hds) | -0.395 | **-0.269** | -0.321 |
| story_gen (hds) | **-0.467** | **-0.142** | -0.392 |
| prompt_gen (no hds) | -0.363 | **-0.017** | -0.333 |
| resp_gen (no hds) | -0.328 | **-0.174** | -0.317 |
| story_gen (no hds) | -0.417 | **-0.119** | -0.335 |

**Exponential E_fit reduces magnitude by 60-95%.** On prompt_gen (no hds), ρ is nearly zero (-0.017). Sigmoid E_fit barely helps. But correlations are still weakly negative — the sign has not fully flipped.

### ConTest (binary content diversity)

| Task | E_discrete ρ | E_fit_exp ρ | E_fit_sig ρ |
|------|-------------|-------------|-------------|
| prompt_gen | nan | nan | nan |
| resp_gen | -0.281 | **-0.237** | -0.301 |
| story_gen | -0.531 | **-0.278** | -0.457 |

### DecTest (temperature correlation)

| Task | E_discrete ρ | E_fit_exp ρ | E_fit_sig ρ |
|------|-------------|-------------|-------------|
| prompt_gen | **-0.279** | **+0.074** | -0.328 |
| resp_gen | -0.026 | -0.031 | -0.046 |
| story_gen | -0.163 | -0.055 | -0.168 |

**On DecTest prompt_gen, E_fit_exp flipped the sign positive (+0.074).** Others moved toward zero.

### Summary

The non-convergence bug was a major contributor to the wrong-sign results. Exponential E_fit dramatically improves things, but doesn't fully fix it. The residual negative correlation may stem from additional issues (see bugs and open questions below).

## Confirmed Bugs Found During Review

### Bug 1: Missing space in completion format

In `format_conditioning_context`, completion mode produces `"{prompt}{response}"` with no space between them. E.g.: `"...The man used primer the next time.The paint still peeled off."` — the period and capital letter run together.

**Fix**: Add a `completion_separator` parameter (default `" "`) inserted between prompt and response. Matthew wants this as an optional flag defaulting to True.

**Location**: `src/icl_diversity/core.py`, lines 522-531

**Impact**: Affects all 13,929 computed samples. Requires re-running the compute.

### Bug 2: Off-by-one in boundary detection (both modes)

In `_find_response_boundaries`, the character-span-to-token mapping uses `c_start >= char_start` which excludes the first token of each response when BPE merges a leading space from the separator into the response's first token. E.g., response `"Once upon a time"` → tokens `[" Once", " upon", " a", " time"]`, but boundary starts at `" upon"`, missing `" Once"`.

**Observed in**: masking verification plot for instruct mode. Completion mode is similarly affected.

**Fix**: Change boundary condition to include tokens whose character span overlaps with the response span, not just tokens whose start falls within it.

**Location**: `src/icl_diversity/core.py`, `_find_response_boundaries`, around line 365

**Impact**: All metrics are systematically off by one token per response. Affects all computed results. Requires re-running.

### Bug 3: No space between context and response in completion format (duplicate of Bug 1 from different angle)

Visible in `figures/tevet_validation/inspection/sample_text_inspection.txt`.

## Unexplained Anomaly: k=5 Uptick

The mean a_k curve shows a systematic uptick at k=5 (the last response position) across all McDiv_nuggets datasets. ~63% of individual samples show `a_5 > a_4`. This also appears in per-permutation curves (31/50 permutations), ruling out it being an artifact of averaging.

This is **unexplained and concerning**. The a_k curve should be monotonically non-increasing in expectation — seeing more responses in context can only help predict the next one. The uptick is not an edge effect: the metric treats every prefix identically (positions 0-4 and 0-5 should behave the same way).

Possible investigations:
- Check if the uptick exists in DecTest (10 responses) at k=10
- Check if it exists with instruct format (to isolate whether completion format causes it)
- Check if it exists with GPT-2 (to isolate whether Qwen causes it)
- Run with 6 or 7 responses on synthetic data where we control the ground truth

## Cosmetic Issues in Plots

1. **Half-integer k lines**: Individual a_k curve plot draws connecting lines at k=1.5, 2.5 etc. Should use markers only.
2. **Missing legend colors**: `mean_ak_with_fit_no_hds.png` legend doesn't say which color is high vs low diversity.
3. **E_fit vs E_discrete plot**: Current plot uses identity line but points are too compressed against the y-axis. Should be a regular scatter with line of best fit and R values.

## Key Next Steps

### Priority 1: Disagreement Analysis
Identify response sets where Tevet labels and our metric **disagree** — samples where Tevet says "high diversity" but E_fit is low (or vice versa). This is critical for understanding whether the remaining negative correlation is a metric problem or a data/labeling issue. Focus on McDiv_nuggets.

### Priority 2: Fix Bugs and Re-run
1. Fix missing space in completion format (add `completion_separator` flag)
2. Fix off-by-one boundary detection
3. Re-run compute with Qwen2.5-3B (another ~11 hours)
4. Re-run analysis

### Priority 3: Investigate k=5 Uptick
This may or may not be related to the bugs above. The off-by-one bug affects all positions equally and shouldn't cause a position-specific anomaly. The missing space is also uniform. Need targeted investigation.

### Priority 4: Fix Plot Issues
- Individual a_k: markers only, no half-integer lines
- Mean a_k: label colors in legend
- E_fit vs E_discrete: scatter with line of best fit + R values
- Add D vs temperature plots (D = C × E adjusts for coherence degradation at high temperature)

## File Locations

### Code
- `src/icl_diversity/core.py` — core metric with format_mode parameter
- `scripts/compute_icl_metrics_for_tevet.py` — compute pipeline (saves mean curves + format_mode flag)
- `scripts/analyze_tevet_validation.py` — analysis with E_fit (exponential + sigmoid)
- `scripts/inspect_tevet_results.py` — visual inspection plots
- `scripts/verify_completion_masking.py` — masking verification bar charts
- `scripts/fit_ak_curves.py` — exponential/sigmoid fitting (reused by analysis)

### Results
- `results/tevet/qwen25_completion/` — CSVs + mean curves JSONs (18 files each)
- `results/tevet/gpt2/` — old GPT-2 instruct results (CSVs only, no mean curves)

### Figures
- `figures/tevet_validation/old_gpt2_instruct/` — old broken results (for reference)
- `figures/tevet_validation/qwen25_completion/` — new results + masking verification
- `figures/tevet_validation/inspection/` — diagnostic plots for manual review

### Documentation
- `docs/tevet_redo_report.md` — this report
- `hypotheses/tevet_results_gpt2.md` — old GPT-2 results (wrong-sign analysis, partially incorrect interpretation about E measuring "redundancy")
- `docs/images_results.txt` — Matthew's visual inspection feedback
