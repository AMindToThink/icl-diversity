# Proposed Metric Revision: C × a_∞ as Primary Diversity Score

## Problem Statement

The current primary diversity score D = C × E relies on the excess entropy E = Σ(a_k − a_∞), which measures how many bits of learnable structure θ discovers across the progressive conditioning process. This works well when the response set contains **repeated modes** (Regime 1), because θ sees the same mode multiple times and learns to predict it, driving the a_k curve down.

However, E fails in **Regime 2**: when the response set contains few samples with no repeated modes (e.g., the Tevet benchmark uses 5 diverse responses). In this regime:
- Diverse responses: the a_k curve stays flat (θ finds no shared structure), so E ≈ 0.
- Non-diverse responses (paraphrases): the a_k curve drops somewhat (θ recognizes shared meaning), so E > 0.
- But E is small and noisy in both cases because n is too small for θ to converge.

The distinguishing signal in Regime 2 is not the *area above the floor* (E) but the *level of the floor itself* (a_∞). Diverse responses have high a_∞ (θ can't predict them even after seeing the others); non-diverse responses have low a_∞ (θ learns their shared structure).

## Proposed Score: C × a_∞

### Definition

The diversity score is:

$$D = C \times a_\infty$$

where:
- C is the per-byte coherence (geometric mean per-byte probability under θ), unchanged from the current paper
- a_∞ is the asymptotic conditional surprise floor, estimated either by:
  - **Sigmoid/exponential fit** to the a_k curve, extrapolating the floor (preferred, based on results from mode_count experiments)
  - **a_n** (the last observed value), as a simpler fallback

Units: bits (if a_∞ is in total bits) or bits/byte (if a_∞ is from the per-byte-normalized curve).

### Interpretation

C × a_∞ measures: **"How many bits of surprise remain per byte after θ has learned everything it can from the other responses, weighted by how plausible the outputs are?"**

High C × a_∞ means the responses are both individually coherent (high C) and mutually unpredictable even after full conditioning (high a_∞). This is the signature of genuine diversity.

### Information-Theoretic Decomposition

The unconditional surprise of a response decomposes as:

$$a_1 = a_\infty + (a_1 - a_\infty)$$

Recalling that a_k = −log₂ θ(r | r_{<k}, p), this is:

$$-\log_2 \theta(r \mid p) = -\log_2 \theta(r \mid r_{<\infty}, p) + \log_2 \frac{\theta(r \mid r_{<\infty}, p)}{\theta(r \mid p)}$$

The second term is the **pointwise mutual information** between the response r and the infinite conditioning context r_{<∞}:

$$a_1 - a_\infty = \mathrm{pmi}_\theta(r;\, r_{<\infty} \mid p)$$

So the decomposition is:

> **total surprise = irreducible surprise + learnable redundancy (PMI)**

Multiplying through by C:

$$C \times a_1 = C \times a_\infty + C \times \mathrm{pmi}_\theta(r;\, r_{<\infty} \mid p)$$

This gives a coherence-weighted decomposition of total surprise into:
- **C × a_∞**: diversity signal (what θ can't learn from the other responses)
- **C × (a_1 − a_∞)**: redundancy signal (what θ can learn, i.e., mutual information between a response and the population)

In expectation over draws from π, the PMI term becomes the mutual information between a response and the infinite context, which is the asymptotic per-response MI rate already in the paper.

## Edge Case Analysis

| Scenario | C | a_∞ | C × a_∞ | Correct? |
|----------|---|-----|---------|----------|
| **Pure noise** | low | high | low | ✓ (noise is not diverse) |
| **Diverse-coherent** | high | high | high | ✓ |
| **Non-diverse paraphrases** | high | low (θ learns shared meaning) | low | ✓ |
| **Single coherent mode** | high | low (θ learns the mode) | low | ✓ |
| **Multiple incoherent modes** | low | high | low | ✓ |
| **One verbose high-entropy mode** | high | low (θ learns the mode's structure, a_1 is high but a_∞ drops) | low | ✓ |

All six edge cases behave correctly. The key distinction from raw a_∞ (which would incorrectly reward noise) is that C suppresses incoherent outputs.

## Relationship to E

E and a_∞ are not redundant — they capture different aspects of the a_k curve:

- **E = Σ(a_k − a_∞)**: total area above the floor. Measures *how much* learnable structure exists and *how it is distributed* across the conditioning process. Informative when n is large enough for θ to learn (Regime 1).
- **a_∞**: the floor itself. Measures *what remains unpredictable* after θ has learned everything it can. Informative in both regimes.

In Regime 1 (many samples, repeated modes), E provides structural information that a_∞ does not: two policies with the same a_∞ could have very different E (e.g., 3 modes vs. 20 modes with the same within-mode variation). In Regime 2 (few samples, no repeats), E ≈ 0 for both diverse and non-diverse sets, and a_∞ carries the signal.

**Recommendation**: C × a_∞ as the primary scalar diversity score (works in both regimes). E as supplementary structural information when n is large. The (E, C, σ_ℓ) triple remains useful for detailed diagnostics.

## Estimation of a_∞

Two approaches, both to be tested:

### 1. Parametric Fit (preferred)
Fit a sigmoid to the a_k curve:

$$a_k = a_\infty + \frac{\alpha}{1 + e^{\beta(k - k_0)}}$$

Extract a_∞ from the fit. This works even when the curve hasn't fully converged at k = n, and avoids pushing θ into long-context OOD regimes. When m is small, the sigmoid degenerates to exponential decay (k_0 < 1), which is fine — the fit still extracts a_∞.

### 2. Direct Estimate: a_n
Use the last observed value a_n as a proxy for a_∞. Simpler but biased upward (a_n ≥ a_∞), especially when the curve hasn't converged.

## Experiment Plan

Run on the Tevet benchmark data (5 diverse vs. 5 non-diverse responses per prompt):
1. Compute the a_k curves for both conditions.
2. Estimate a_∞ via both sigmoid fit and a_n.
3. Compute C × a_∞ for both conditions using both estimation methods.
4. Compute all of the above in both total bits (a_∞ from the raw curve) and bits/byte (a_∞ from the per-byte-normalized curve). The bits version has natural appeal as a diversity measure but is length-sensitive; the bits/byte version normalizes away length differences.
5. Check: does C × a_∞ separate the diverse from non-diverse conditions?
6. Compare with C × E to confirm that E fails in this regime.
7. Also compute C × (a_1 − a_∞) as the redundancy measure and verify it gives the inverse ranking.
