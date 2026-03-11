"""Compare predicted a_k curve from pairwise matrix with observed curves.

Three models:
1. Independent pairwise model: each prior response contributes independently
2. Paper's prediction (offdiag=0): only same-mode repetitions reduce surprise
3. Observed mean curve from 1k draws

Shows that the independent model predicts INCREASING marginal drops (more
repetitions at later positions), while the observed curve shows DECREASING
marginal drops (diminishing returns / saturation).
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# Pairwise matrix results
with open(FIGURES_DIR / "pairwise_matrix.json") as f:
    matrix_data = json.load(f)

diag_mean = np.mean(np.diag(np.array(matrix_data["reduction_mean"])))
offdiag_mask = ~np.eye(matrix_data["n_modes"], dtype=bool)
offdiag_mean = np.array(matrix_data["reduction_mean"])[offdiag_mask].mean()

print(f"Diagonal mean: {diag_mean:.1f} bits")
print(f"Off-diagonal mean: {offdiag_mean:.1f} bits")

# Observed a_k curves at m=10
with open(
    Path(__file__).parent.parent.parent / "results" / "mode_count" / "qwen2.5-3b_1k_draws.json"
) as f:
    experiment_data = json.load(f)

# Load all m values for comparison
all_m_curves: dict[int, np.ndarray] = {}
for m_val in range(1, 11):
    runs = [r for r in experiment_data["runs"] if r["m"] == m_val]
    all_m_curves[m_val] = np.array([r["a_k_curve"] for r in runs])

runs_m10 = [r for r in experiment_data["runs"] if r["m"] == 10]
observed_curves = np.array([r["a_k_curve"] for r in runs_m10])
observed_mean = np.mean(observed_curves, axis=0)
observed_sem = np.std(observed_curves, axis=0) / np.sqrt(len(runs_m10))
n = len(observed_mean)
k = np.arange(1, n + 1)

# ---------------------------------------------------------------------------
# Predicted curves from independent pairwise model
# ---------------------------------------------------------------------------
m = 10
n_resp = 20

# P(mode-mate in context at position k) = (k-1)/(n_resp-1)
p_mate = np.array([(ki - 1) / (n_resp - 1) for ki in k])

# Model 1: Independent pairwise (both off-diagonal and diagonal)
# a_k = a_1 - (k-1) * offdiag - P(mate in 1..k-1) * (diag - offdiag)
# Marginal: a_{k-1} - a_k = offdiag + P(mate newly at k-1) * (diag - offdiag)
#         = offdiag + 1/(n_resp-1) * (diag - offdiag)  [constant!]
a1 = observed_mean[0]
independent_pred = a1 - (k - 1) * (offdiag_mean + (diag_mean - offdiag_mean) / (n_resp - 1))

# Model 2: Paper's prediction (offdiag=0, only diagonal matters)
# a_k = a_1 - P(mate in 1..k-1) * diag
# This is NOT constant marginal — P(mate) grows with k
paper_pred_no_offdiag = a1 - p_mate * diag_mean

# Model 3: Full pairwise with P(mate) applied to cumulative
# a_k = a_1 - (k-1)*offdiag - p_mate[k] * (diag - offdiag)
# This accounts for growing P(mate) but still assumes independence
full_pairwise_pred = a1 - (k - 1) * offdiag_mean - p_mate * (diag_mean - offdiag_mean)

# Compute marginal drops for each
observed_marginals = -np.diff(observed_mean)  # a_{k} - a_{k+1}, positive = drop
independent_marginals = np.full(n - 1, offdiag_mean + (diag_mean - offdiag_mean) / (n_resp - 1))
paper_marginals = -np.diff(paper_pred_no_offdiag)
full_pairwise_marginals = -np.diff(full_pairwise_pred)

# ---------------------------------------------------------------------------
# Figure 1: Predicted vs observed a_k curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Cumulative curves
ax = axes[0]
ax.plot(k, observed_mean, "ko-", linewidth=2, markersize=4, label="Observed (1k draws)", zorder=5)
ax.fill_between(k, observed_mean - observed_sem, observed_mean + observed_sem, alpha=0.2, color="black")
ax.plot(k, independent_pred, "b--", linewidth=2, label=f"Independent (const {offdiag_mean + (diag_mean - offdiag_mean)/(n_resp-1):.1f} bits/step)")
ax.plot(k, full_pairwise_pred, "r-.", linewidth=2, label="Pairwise w/ growing P(mate)")
ax.plot(k, paper_pred_no_offdiag, "g:", linewidth=2, label="Paper model (offdiag=0)")
ax.set_xlabel("k (response index)", fontsize=12)
ax.set_ylabel("$a_k$ (bits)", fontsize=12)
ax.set_title("Predicted vs Observed $a_k$ Curve (m=10)", fontweight="bold")
ax.legend(fontsize=8, loc="upper right")

# Panel 2: Marginal drops
ax = axes[1]
k_mid = np.arange(1, n)  # positions 1..19 (drop from k to k+1)
ax.plot(k_mid, observed_marginals, "ko-", linewidth=2, markersize=4, label="Observed", zorder=5)
ax.plot(k_mid, independent_marginals, "b--", linewidth=2, label="Independent (constant)")
ax.plot(k_mid, full_pairwise_marginals, "r-.", linewidth=2, label="Pairwise w/ growing P(mate)")
ax.plot(k_mid, paper_marginals, "g:", linewidth=2, label="Paper (offdiag=0)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("k (step from $a_k$ to $a_{k+1}$)", fontsize=12)
ax.set_ylabel("Marginal drop $a_k - a_{k+1}$ (bits)", fontsize=12)
ax.set_title("Marginal Drops: Predicted vs Observed", fontweight="bold")
ax.legend(fontsize=8)

# Annotate the key insight
ax.annotate(
    "Models predict\nINCREASING drops\n(more repetitions)",
    xy=(15, full_pairwise_marginals[14]),
    xytext=(10, max(observed_marginals) * 0.8),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red",
)
ax.annotate(
    "Observed:\nDECREASING drops\n(saturation)",
    xy=(15, observed_marginals[14]),
    xytext=(10, -5),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="black"),
    color="black",
)

# Panel 3: Cumulative drops as fraction of independent prediction
ax = axes[2]
observed_cum_drop = observed_mean[0] - observed_mean
independent_cum_drop = observed_mean[0] - independent_pred
full_cum_drop = observed_mean[0] - full_pairwise_pred

ax.plot(k, observed_cum_drop, "ko-", linewidth=2, markersize=4, label="Observed cumulative drop")
ax.plot(k, independent_cum_drop, "b--", linewidth=2, label="Independent prediction")
ax.plot(k, full_cum_drop, "r-.", linewidth=2, label="Pairwise w/ P(mate)")
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel("Cumulative drop from $a_1$ (bits)", fontsize=12)
ax.set_title("Cumulative Drop: How Much Has the Model Learned?", fontweight="bold")
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = FIGURES_DIR / "predicted_vs_observed_m10.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Marginal drops across all m values
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, m_val in enumerate(range(1, 11)):
    row, col = divmod(idx, 5)
    ax = axes[row][col]

    curves = all_m_curves[m_val]
    mean_curve = np.mean(curves, axis=0)
    marginals_obs = -np.diff(mean_curve)
    n_k = len(mean_curve)
    k_plot = np.arange(1, n_k)

    # Predicted marginals for this m
    # Build mode assignments: distribute n_resp responses across m_val modes
    # (cycling, same as generate_mode_count_responses)
    modes_arr = np.array([i % m_val for i in range(n_resp)])
    rng = np.random.default_rng(42)
    n_sims = 50000
    p_rep = np.zeros(n_resp)
    for _ in range(n_sims):
        perm = rng.permutation(n_resp)
        mode_order = modes_arr[perm]
        seen = {}
        for pos in range(n_resp):
            mode_k = mode_order[pos]
            if seen.get(mode_k, 0) > 0:
                p_rep[pos] += 1
            seen[mode_k] = seen.get(mode_k, 0) + 1
    p_rep /= n_sims

    pred_marginals = offdiag_mean + p_rep[1:] * (diag_mean - offdiag_mean)

    ax.bar(k_plot - 0.15, marginals_obs, width=0.3, color="black", alpha=0.7, label="Observed")
    ax.bar(k_plot + 0.15, pred_marginals, width=0.3, color="red", alpha=0.5, label="Pairwise pred")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"m = {m_val}", fontweight="bold", fontsize=10)
    if row == 1:
        ax.set_xlabel("k")
    if col == 0:
        ax.set_ylabel("Marginal drop (bits)")
    if idx == 0:
        ax.legend(fontsize=6)

fig.suptitle(
    "Marginal Drops: Observed vs Independent Pairwise Prediction (all m values)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path2 = FIGURES_DIR / "marginal_drops_all_m.png"
plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path2}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: The "efficiency" — observed drop / predicted drop at each k
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for m_val in [1, 3, 5, 10]:
    curves = all_m_curves[m_val]
    mean_curve = np.mean(curves, axis=0)
    marginals_obs = -np.diff(mean_curve)

    modes_arr = np.array([i % m_val for i in range(n_resp)])
    rng = np.random.default_rng(42)
    n_sims = 50000
    p_rep = np.zeros(n_resp)
    for _ in range(n_sims):
        perm = rng.permutation(n_resp)
        mode_order = modes_arr[perm]
        seen = {}
        for pos in range(n_resp):
            mode_k = mode_order[pos]
            if seen.get(mode_k, 0) > 0:
                p_rep[pos] += 1
            seen[mode_k] = seen.get(mode_k, 0) + 1
    p_rep /= n_sims

    pred_marginals = offdiag_mean + p_rep[1:] * (diag_mean - offdiag_mean)

    # Efficiency = observed / predicted (capped for display)
    efficiency = marginals_obs / np.clip(pred_marginals, 0.1, None)
    k_plot = np.arange(1, n_resp)
    ax.plot(k_plot, efficiency, "o-", markersize=4, linewidth=2, label=f"m = {m_val}")

ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", label="Perfect prediction")
ax.set_xlabel("k (step from $a_k$ to $a_{k+1}$)", fontsize=12)
ax.set_ylabel("Observed / Predicted marginal drop", fontsize=12)
ax.set_title("Efficiency: How Well Does the Independent Model Predict Each Step?", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(-0.5, 3.0)

plt.tight_layout()
fig_path3 = FIGURES_DIR / "prediction_efficiency.png"
plt.savefig(fig_path3, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path3}")
plt.close()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n=== Key comparison at m=10 ===")
print(f"{'k':>3s}  {'obs_marginal':>13s}  {'pred_indep':>11s}  {'pred_pairwise':>14s}  {'ratio':>6s}")
for i in range(n - 1):
    obs = observed_marginals[i]
    indep = independent_marginals[i]
    pw = offdiag_mean + p_mate[i + 1] * (diag_mean - offdiag_mean)
    ratio = obs / pw if pw > 0.1 else float("nan")
    print(f"{i+1:3d}  {obs:13.1f}  {indep:11.1f}  {pw:14.1f}  {ratio:6.2f}")

print(f"\nObserved total drop a_1→a_20: {observed_mean[0] - observed_mean[-1]:.1f} bits")
print(f"Independent prediction:       {(n-1) * (offdiag_mean + (diag_mean - offdiag_mean)/(n_resp-1)):.1f} bits")
print(f"Pairwise w/ P(mate):          {sum(offdiag_mean + p_mate[i+1] * (diag_mean - offdiag_mean) for i in range(n-1)):.1f} bits")
print(f"\nThe pairwise model overpredicts by {sum(offdiag_mean + p_mate[i+1] * (diag_mean - offdiag_mean) for i in range(n-1)) / (observed_mean[0] - observed_mean[-1]):.1f}x")
