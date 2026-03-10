# Experiments Plan: Progressive Conditional Surprise Metric

## 1. Curve Shape vs. Mode Count

**Question:** Does the $a_k$ curve transition from concave-up (exponential decay) to sigmoidal as the number of modes $m$ increases?

**Setup:** Construct synthetic response datasets with known mode counts $m \in \{3, 5, 10, 15, 25, 50\}$. Each mode should be clearly distinguishable (e.g., different topics, styles, or formats). Run the metric and plot $a_k$ curves for each $m$.

**Prediction:** For small $m$ (≲5), concave-up everywhere. For large $m$, sigmoidal with flat initial region where $k \ll m$.

**Why it matters:** Determines whether parametric extrapolation of $a_\infty$ is viable, and which functional form to fit. Also determines whether the elbow-based mode count estimate (Section 5.5) is reliable.

**Status:** Not started

---

## 2. Parametric Fit for $a_\infty$ Extrapolation

**Question:** Can we fit a parametric model to the $a_k$ curve and extrapolate $a_\infty$ more accurately than using $a_n$ directly?

**Preferred model:** Sigmoid: $a_k = a_\infty + \frac{\alpha}{1 + e^{\beta(k - k_0)}}$

Theoretical reasons predict sigmoidal curves (flat → drop → floor). Exponential decay is the degenerate case where the inflection point $k_0$ is off the left edge of the plot (i.e., $k_0 < 1$), which occurs when $m \ll n$. Always fitting a sigmoid and reading $k_0$ from the fit is cleaner than switching between functional forms. When $k_0 \ll 1$, the sigmoid reduces to exponential decay; when $k_0 > 1$, it provides the inflection point as a bonus diagnostic (approximate mode count).

**Depends on:** Experiment 1 (to validate across different $m$ values).

**Status:** Not started

---
