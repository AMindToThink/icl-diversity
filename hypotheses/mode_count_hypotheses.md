# Mode Count Experiment: Pre-registered Hypotheses

**Date**: 2026-03-10
**Experiment**: Curve shape vs mode count (m)
**Setup**: Synthetic format-based modes (haiku, code, recipe, etc.), m ∈ {1,2,3,5,10,15}
**Base models**: GPT-2 (m ≤ 10), Qwen 2.5-32B (full range)
**Responses per mode**: 4 (total n = 4m)
**Permutations**: 20

---

## H1: Curve Shape Transition

For m ≤ 3, the a_k curve is concave-up (exponential decay). For m ≥ 10, the a_k curve is sigmoidal with a flat initial region where k ≪ m.

**Rationale**: When there are few modes, the first few responses already cover most modes, so surprise drops immediately. With many modes, early responses only cover a fraction of modes, so surprise stays high until enough modes are seen.

## H2: E Monotonicity

E (excess entropy) increases monotonically with m (more modes = more excess entropy = more "learnable" inter-response structure).

**Rationale**: E measures how much the a_k curve drops from its initial value. More modes means more redundancy structure to learn — each new response is somewhat predictable given prior responses from the same mode.

## H3: Asymptote Rises with m

a_∞ (the asymptote of the a_k curve) increases with m. More modes = higher residual surprise even after full conditioning.

**Rationale**: With many distinct modes, the base model never fully learns to predict responses because the mode variety provides a floor of irreducible surprise.

## H4: Sigmoid Inflection Point

k₀ (sigmoid inflection point) increases approximately linearly with m. When m ≤ 3, k₀ < 1 (sigmoid degenerates to exponential).

**Rationale**: The inflection point should occur roughly when k ≈ m — that's when the model has seen approximately one example of each mode and further examples start to provide diminishing returns.

## H5: Extrapolation Quality

Sigmoid-fitted a_∞ is closer to the true asymptote (from high-n runs) than raw a_n.

**Rationale**: When n is small relative to m, a_n hasn't converged to a_∞. The sigmoid fit can extrapolate the tail behavior.

## H6: Sign Consistency

E increases with m (not decreases), confirming E measures redundancy — more modes = more learnable structure across responses.

**Rationale**: This is a sanity check. The Tevet validation showed E is inverted relative to diversity. Here, increasing m should increase the learnable structure (each mode is internally coherent), increasing E.
