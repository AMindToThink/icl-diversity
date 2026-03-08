# Pre-Registered Hypotheses: Qwen2.5-32B vs GPT-2

These predictions are made *before* seeing any Qwen2.5-32B results. The same
5 scenarios with the same prompts and responses are used, so cross-model
comparisons are paired (same prompt under both models, n=5 per scenario).

## Setup

- **Base model under test**: Qwen/Qwen2.5-32B (base, not instruct)
- **Reference model**: GPT-2 (124M), results already in `results/scenario_metrics.json`
- **Hardware**: 2x Quadro RTX 8000 (46GB each), float16, `device_map="auto"`
- **Same scenarios, prompts, responses, seed, and n_permutations as the GPT-2 run**

## Notation

- Subscript `q` = Qwen2.5-32B, `g` = GPT-2
- C = 2^{-mean_h} (coherence, higher = more plausible per byte)
- E = sum(a_k - a_n) (excess entropy, learnable structure)
- D = C x E (diversity score)
- sigma = std(unconditional_surprises) (coherence spread)

## GPT-2 Reference Values (means across 5 prompts)

| Scenario | E | C | D | sigma | Monotone a_k |
|----------|----:|-----:|-----:|------:|:----:|
| Pure noise | 0.787 | 0.006 | 0.005 | 0.195 | 0/5 |
| Multi incoherent | -1.289 | 0.121 | -0.160 | 0.508 | 0/5 |
| Multi mode (3 modes) | 1.701 | 0.495 | 0.818 | 0.082 | 0/5 |
| One mode (paraphrase) | 1.046 | 0.496 | 0.518 | 0.057 | 0/5 |
| Mixed coh+incoh | -2.424 | 0.290 | -0.574 | 1.123 | 0/5 |

## Hypotheses

### Coherence (C) predictions

| ID | Hypothesis | Rationale | Confidence |
|----|-----------|-----------|------------|
| Q1 | C_q(multi_mode) > C_g(multi_mode) | Stronger model assigns lower per-byte CE to coherent English (paper Section 5: "a better model assigns lower cross-entropy") | High |
| Q2 | C_q(one_mode) > C_g(one_mode) | Same reasoning as Q1 | High |
| Q3 | C_q(noise) < 0.02 | Random ASCII is maximally surprising for any model; C stays near floor | High |
| Q4 | C_q(multi_incoherent) ~ C_g(multi_incoherent) | Both models find incoherent text implausible; direction unclear | Low |

### Excess entropy (E) predictions

| ID | Hypothesis | Rationale | Confidence |
|----|-----------|-----------|------------|
| Q5 | E_q(multi_incoherent) > E_g(multi_incoherent) | GPT-2 has E = -1.29 (ICL failure: a_k increases as incoherent context degrades predictions). Paper Section 3.3 identifies this as theta's predictions degrading with complex contexts. Qwen's stronger ICL should be more robust, giving E less negative or near 0 | High |
| Q6 | E_q(mixed) > E_g(mixed) | Same reasoning as Q5. GPT-2 has E = -2.42. Qwen should be more robust to mixed-quality context | High |
| Q7 | E_q(multi_mode) >= E_g(multi_mode) | Theory: E ~ log2(m)/B_bar is model-independent for perfect ICL. Better ICL moves closer to theoretical limit. But lower absolute a_k values could partially offset | Moderate |
| Q8 | E_q(noise) ~ 0 | No learnable structure in noise; GPT-2's E=0.787 for noise is likely an artifact that Qwen's more stable ICL may reduce | Moderate |

### Diversity score (D) predictions

| ID | Hypothesis | Rationale | Confidence |
|----|-----------|-----------|------------|
| Q9 | D_q(multi_mode) > D_g(multi_mode) | Higher C (Q1) x similar or higher E (Q7) -> higher D | Moderate-High |

### Coherence spread (sigma) predictions

| ID | Hypothesis | Rationale | Confidence |
|----|-----------|-----------|------------|
| Q10 | sigma_q(mixed) > sigma_g(mixed) | Stronger model discriminates better: assigns much lower CE to coherent responses and similar/higher CE to incoherent -> wider spread | Moderate |
| Q11 | sigma_q(one_mode) ~ sigma_g(one_mode) | Uniform-quality responses; both models should have low spread | High |

### Monotonicity predictions

| ID | Hypothesis | Rationale | Confidence |
|----|-----------|-----------|------------|
| Q12 | More monotone a_k curves for Qwen than GPT-2 (especially multi_mode, one_mode) | Paper Section 3.3: monotonicity is an empirical property of theta's ICL quality. Better ICL -> smoother, more monotone curves. GPT-2: 0/25 monotone | Moderate |

### Within-model hypotheses

| ID | Hypothesis | Rationale | Confidence |
|----|-----------|-----------|------------|
| Q13 | All 13 original hypotheses (H1-H13) also hold for Qwen | The metric's theoretical properties (C separates coherent/incoherent, E captures structure, D combines them correctly) are model-independent. The scenarios were designed to have clear ground-truth orderings | High |

## Statistical Approach

For cross-model comparisons (Q1-Q12): paired direction checks (same prompts
under both models, n=5 per scenario). Wilcoxon signed-rank test where feasible
(minimum p for n=5 with perfect separation is 1/32 = 0.03125). For Q13: same
tests as original H1-H13 but on Qwen data.
