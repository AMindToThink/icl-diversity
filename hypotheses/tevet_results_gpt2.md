# Experiment Results: ICL Diversity Metric on Tevet's diversity-eval (GPT-2)

## Setup

- **Base model**: GPT-2 (124M parameters, 1024 token context window)
- **n_permutations**: 50
- **batch_size**: 8 (CUDA)
- **Datasets**: All 18 CSVs from Tevet's diversity-eval (13,679 response sets total, zero skips)
- **Response set sizes**: 5 responses (ConTest, McDiv, McDiv_nuggets), 10 responses (DecTest)

## Summary

Every hypothesis fails except H5 (orthogonality). With GPT-2 as the base model, E and D show near-zero or **negative** correlations with the ground-truth diversity labels — meaning the metric assigns *higher* diversity scores to constant-content response sets than to diverse ones.

## Results by Hypothesis

### H1: E on ConTest — FAIL

| Task | E ρ | E OCA | distinct-n ρ | distinct-n OCA |
|------|-----|-------|-------------|---------------|
| story_gen | -0.182 | 0.596 | 0.573 | 0.772 |
| resp_gen | -0.085 | 0.541 | 0.346 | 0.677 |
| prompt_gen | 0.000 | 0.500 | 0.333 | 0.675 |

E is at or below chance (OCA ≈ 0.5) and its correlation goes in the wrong direction on story_gen. ConTest has paired samples — the same sample_id appears twice (once labeled diverse, once constant) with the same responses receiving identical E values, which explains the exact zeros on prompt_gen.

### H2: D vs E on ConTest — FAIL (moot)

D ≈ E throughout (diff < 0.004 OCA). Both are near chance. Since C is constant for paired samples with identical responses, D = C × E can't help.

### H3: E on DecTest — FAIL (wrong sign)

| Dataset | Task | E ρ | distinct-n ρ |
|---------|------|-----|-------------|
| dec_test_1000 | prompt_gen | -0.493 | 0.912 |
| dec_test_1000 | story_gen | -0.272 | 0.758 |
| dec_test_1000 | resp_gen | -0.057 | 0.894 |
| dec_test_200 | prompt_gen | -0.458 | 0.918 |
| dec_test_200 | story_gen | -0.365 | 0.787 |
| dec_test_200 | resp_gen | -0.145 | 0.893 |

E is negatively correlated with temperature: higher temperature → lower E. The magnitude varies by task (prompt_gen is worst, resp_gen is near zero).

### H4: E on McDiv_nuggets — wrong sign, but beats distinct-n in magnitude

| Dataset | Task | E ρ | distinct-n ρ | E OCA | distinct-n OCA |
|---------|------|-----|-------------|-------|---------------|
| nuggets_with_hds | story_gen | -0.396 | 0.039 | 0.675 | 0.570 |
| nuggets_with_hds | resp_gen | -0.242 | -0.014 | 0.630 | 0.535 |
| nuggets_with_hds | prompt_gen | -0.361 | 0.120 | 0.665 | 0.565 |
| nuggets (no hds) | story_gen | -0.186 | -0.002 | 0.578 | 0.510 |
| nuggets (no hds) | resp_gen | -0.125 | -0.002 | 0.561 | 0.507 |
| nuggets (no hds) | prompt_gen | -0.224 | -0.003 | 0.606 | 0.514 |

E detects *something* that distinct-n can't (higher |ρ| and OCA), but the sign is flipped. With a sign flip E would outperform distinct-n on this form-neutralized benchmark, which is promising in a backwards way.

### H5: E is orthogonal to existing metrics — PASS

Overall Pearson r(E, distinct-n) = -0.375 (n=13,929). |r| < 0.5 as targeted. r(E, sent-BERT) = -0.045. E is measuring something genuinely different from surface overlap metrics.

## The Wrong-Sign Problem

The most striking finding is that E is consistently higher for constant-content (low diversity) response sets. This holds for both total-bits E and per-byte E_rate, ruling out a simple length artifact:

| Dataset | Task | E_rate (diverse) | E_rate (constant) | Diff |
|---------|------|-----------------|------------------|------|
| conTest | story_gen | 0.141 | 0.226 | -0.085 |
| conTest | resp_gen | 0.129 | 0.171 | -0.043 |
| McDiv_nuggets (hds) | story_gen | 0.180 | 0.462 | -0.282 |
| McDiv_nuggets (hds) | prompt_gen | 0.376 | 0.574 | -0.198 |

### Why constant content → higher E with GPT-2

E = Σ(a_k - a_∞) measures the total area above the asymptote in the progressive surprise curve. Higher E means the curve starts high and drops more — i.e., there's more "learnable" structure across responses.

With **constant content** (5 paraphrases of the same idea), GPT-2 should in theory learn the pattern quickly: a_1 is high (novel), a_2 drops (seen this before), a_3..5 are low (fully predicted). This produces a steep curve and high E.

With **diverse content** (5 different approaches), each response is novel regardless of conditioning. The curve stays relatively flat — GPT-2 can't predict response k from responses 1..k-1 because they're all different topics. This produces low E.

**The metric is working as designed — it's measuring the progressive mutual information between responses.** The problem is interpretive: high E means "responses are predictable from each other" (= high redundancy = low content diversity), not "responses explore many modes."

This is a fundamental insight: **E measures inter-response redundancy, not diversity.** The paper's framing equates "many modes" with "high surprise that persists" (flat curve, high asymptote), but what actually produces high E is a curve that starts high and drops — meaning θ learns to predict later responses from earlier ones, which happens most strongly when responses repeat the same content.

### Non-convergence compounds the problem

With only 5 responses per set, the a_k curves haven't converged to their asymptote. The metric estimates a_∞ from a_5, which is unreliable. For diverse sets where the curve is still dropping at k=5, a_5 overestimates a_∞, compressing E toward zero. For constant sets where the curve flattens by k=3, a_5 is a better estimate, preserving E.

### GPT-2's limited capacity

GPT-2 has 124M parameters and a 1024-token context. It may lack the capacity to represent the conditional distribution p(r_k | p, r_1...r_{k-1}) well enough to detect semantic overlap between diverse responses. A stronger model might show different behavior — but the sign-flip pattern is so consistent that it likely reflects a real property of the metric, not just model weakness.

## Implications

1. **E measures redundancy, not diversity** — at least with GPT-2 and these dataset sizes. The metric detects when responses repeat content (progressive surprise drops), which is the opposite of what we want for a diversity score.

2. **1 - E or -E might work** — since the sign is consistently wrong, a simple sign flip could yield a competitive diversity metric, especially on McDiv_nuggets where E already outperforms distinct-n in magnitude.

3. **More responses per set are needed** — 5 responses is too few for the a_k curve to converge. The metric was designed for scenarios with enough responses (10-20+) to see modes repeated.

4. **Qwen 2.5 may help** — a 32B model with longer context could better model the conditional structure, potentially changing the picture. But the sign issue likely persists since it reflects what E fundamentally measures.

5. **Consider redefining the score** — instead of E = Σ(a_k - a_∞), a diversity score based on the *asymptote level* a_∞ (how surprising responses remain even after full conditioning) might better capture "how many distinct modes exist." High a_∞ = responses remain surprising even after seeing many examples = many modes.
