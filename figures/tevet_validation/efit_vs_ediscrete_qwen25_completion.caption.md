## E_fit vs E_discrete outlier analysis

There are extreme outliers with E_fit up to ~50,000 bits. These all share the same pattern: **flat/noisy a_k curves with near-zero beta (decay rate)**.

For example, the worst outlier (E_fit = 49,521):
- `a_k = [78.7, 86.9, 77.6, 96.8, 95.0, 86.2, 76.9, 79.9, 71.5, 94.4]`
- The curve bounces around ~85 bits with no consistent downward trend
- The exponential fit finds `a_inf=0.0, alpha=85.0, beta=0.0017`
- Since `E_fit = alpha/beta`, a tiny beta produces a massive E_fit

**Root cause**: When the a_k curve is flat (no learning from context), the exponential `a_inf + alpha * exp(-beta*k)` degenerates — it fits `a_inf ≈ 0` and `alpha ≈ mean(a_k)` with `beta → 0`, making `E_fit = alpha/beta → ∞`. This is a fitting degeneracy, not meaningful diversity.

Key observations:
- Almost all outliers are from DecTest (10 responses, temperature-varied) — not McDiv_nuggets
- The one McDiv_nuggets outlier (E_fit = 10,679, story_gen sample 00307) has `a_k = [82.1, 85.3, 75.3, 86.9, 78.2]` — again flat/noisy
- These are cases where the model gains essentially **no information** from seeing previous responses in context, so the curve never decays
- This suggests the exponential fit needs a **minimum beta constraint** or a **sanity check** that rejects fits with `beta < threshold` (falling back to E_discrete or capping E_fit)
