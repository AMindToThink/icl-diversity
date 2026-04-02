# Pre-Registered Hypotheses: Model Quality and Information-Theoretic Reasoning

## Context

The pairwise cross-mode surprise reduction matrix reveals how well a base model
treats information from different modes as (approximately) mutual information.
True mutual information is symmetric: I(X;Y) = I(Y;X). In our setting,
reduction[i,j] measures how much seeing mode_i reduces surprise for mode_j.

Prior results (2 models):
- **GPT-2**: off-diagonal mean = −3.7 bits, R² = 0.14, slope = 0.33
- **Qwen2.5-3B**: off-diagonal mean = +1.9 bits, R² = 0.24, slope = 0.46

Hypothesis formulated before seeing any Llama results: better language models
(lower pretraining loss) should be better information-theoretic reasoners.
Model size is a proxy for quality since we don't have the actual pretraining
losses. We test on the Llama family (all dense) to control for MoE confounds.

## Models (ordered by size)

1. meta-llama/Llama-3.2-1B (1B params)
2. meta-llama/Llama-3.2-3B (3B params)
3. meta-llama/Llama-3.1-8B (8B params)
4. meta-llama/Llama-3.1-70B (70B params)

## Hypotheses

### H1: Llama-3.1-70B > Llama-3.1-8B

Same generation (3.1), 8.75× size difference. Cleanest test of the size
hypothesis.

### H2: Llama-3.2-3B > Llama-3.2-1B

Same generation (3.2), 3× size difference. Tests size hypothesis in the
smaller regime.

### H3: Llama-3.1-8B > Llama-3.2-3B

Cross-generation, 2.67× size difference. Confounded by generation (3.1 vs
3.2 differ in training data/procedure). If this fails but H1/H2 hold, it
tells us generation matters more than size alone.

## Operationalization of ">"

For model A > model B, we expect A to show more of:

1. **R² (symmetry)**: Higher R² between reduction[i,j] and reduction[j,i]
2. **Slope**: Closer to 1.0 on the symmetry scatter
3. **Off-diagonal mean**: Higher (more positive cross-mode information transfer)
4. **Fraction off-diagonal > 0**: Higher (more pairs show positive transfer)

## Statistical tests

1. **Per-hypothesis paired comparison** on each of the 4 metrics
2. **Monotonicity check**: probability of accidental monotonicity across 4
   size-ordered models is 1/24 ≈ 4% under the null
3. **Bootstrap 95% CIs on R² and slope** from the 105 symmetry-scatter points
4. **Per-model t-test** on off-diagonal entries (210 pairs) against zero

## Date

2026-04-02
