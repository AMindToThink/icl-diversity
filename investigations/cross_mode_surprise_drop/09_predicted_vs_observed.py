"""Compare predicted a_k curve from pairwise matrix with observed curves.

Four models:
1. Additive independent: each response removes a fixed number of bits
2. Additive w/ growing P(mate): accounts for mode repetitions, still additive
3. Multiplicative (fractional): each response resolves a fraction of remaining
   uncertainty, where the fraction is derived from the pairwise matrix
4. Paper's prediction (offdiag=0): only same-mode repetitions reduce surprise

The multiplicative model is motivated by the observation that the additive model
predicts the wrong shape (increasing marginals vs observed decreasing marginals).
If each response resolves a fraction r of remaining uncertainty rather than a
fixed number of bits, we get exponential decay — matching the observation.
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
with open(FIGURES_DIR / "pairwise_matrix.json") as f:
    matrix_data = json.load(f)

reduction_matrix = np.array(matrix_data["reduction_mean"])
diag_mean = np.mean(np.diag(reduction_matrix))
offdiag_mask = ~np.eye(matrix_data["n_modes"], dtype=bool)
offdiag_mean = reduction_matrix[offdiag_mask].mean()

print(f"Diagonal mean: {diag_mean:.1f} bits")
print(f"Off-diagonal mean: {offdiag_mean:.1f} bits")

# Observed a_k curves
with open(
    Path(__file__).parent.parent.parent / "results" / "mode_count" / "qwen2.5-3b_1k_draws.json"
) as f:
    experiment_data = json.load(f)

all_m_curves: dict[int, np.ndarray] = {}
for m_val in range(1, 11):
    runs = [r for r in experiment_data["runs"] if r["m"] == m_val]
    all_m_curves[m_val] = np.array([r["a_k_curve"] for r in runs])

n_resp = 20


# ---------------------------------------------------------------------------
# Helper: simulate P(mode-mate already seen) at each position for given m
# ---------------------------------------------------------------------------
def simulate_p_rep(m_val: int, n_resp: int = 20, n_sims: int = 50000) -> np.ndarray:
    """P(at position k, the response's mode has already appeared in 0..k-1)."""
    modes_arr = np.array([i % m_val for i in range(n_resp)])
    rng = np.random.default_rng(42)
    p_rep = np.zeros(n_resp)
    for _ in range(n_sims):
        perm = rng.permutation(n_resp)
        mode_order = modes_arr[perm]
        seen: dict[int, int] = {}
        for pos in range(n_resp):
            mode_k = mode_order[pos]
            if seen.get(mode_k, 0) > 0:
                p_rep[pos] += 1
            seen[mode_k] = seen.get(mode_k, 0) + 1
    return p_rep / n_sims


def predict_multiplicative(
    a1: float, p_rep: np.ndarray, diag: float, offdiag: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Multiplicative model: each response resolves a fraction of remaining
    *resolvable* uncertainty (above the floor).

    The fraction r_k = (offdiag + p_rep[k] * (diag - offdiag)) / a_1
    is derived from the pairwise matrix: the absolute reduction measured in
    isolation, divided by the starting surprise, gives the fraction resolved.

    a_{k+1} = floor + (a_k - floor) * (1 - r_k)

    The floor represents irreducible surprise that context cannot eliminate.
    When floor=0, all surprise is treated as resolvable (decays to zero).
    """
    curve = np.zeros(len(p_rep))
    curve[0] = a1
    for i in range(len(p_rep) - 1):
        r_k = (offdiag + p_rep[i + 1] * (diag - offdiag)) / a1
        r_k = np.clip(r_k, 0.0, 1.0)
        curve[i + 1] = floor + (curve[i] - floor) * (1 - r_k)
    return curve


def predict_additive(
    a1: float, p_rep: np.ndarray, diag: float, offdiag: float
) -> np.ndarray:
    """Additive model: each response removes a fixed number of bits."""
    curve = np.zeros(len(p_rep))
    curve[0] = a1
    for i in range(len(p_rep) - 1):
        drop = offdiag + p_rep[i + 1] * (diag - offdiag)
        curve[i + 1] = curve[i] - drop
    return curve


# ---------------------------------------------------------------------------
# Estimate irreducible floor from m=1 asymptote
# ---------------------------------------------------------------------------
# At m=1 (all same-mode), the curve saturates by k~5. The asymptote is
# irreducible surprise that context cannot eliminate. We use the mean of
# a_15..a_20 from the m=1 curves (well past saturation) as the floor.
# This is independently measured — NOT fitted to the curves we predict.
m1_curves = all_m_curves[1]
m1_mean = np.mean(m1_curves, axis=0)
# Average the last 6 positions (k=15..20) where m=1 has fully saturated
a_floor = np.mean(m1_mean[-6:])
print(f"Estimated floor (m=1 asymptote): {a_floor:.1f} bits")

# ---------------------------------------------------------------------------
# Figure 1: m=10, models vs observed
# ---------------------------------------------------------------------------
runs_m10 = [r for r in experiment_data["runs"] if r["m"] == 10]
observed_curves_m10 = np.array([r["a_k_curve"] for r in runs_m10])
observed_mean = np.mean(observed_curves_m10, axis=0)
observed_sem = np.std(observed_curves_m10, axis=0) / np.sqrt(len(runs_m10))
n = len(observed_mean)
k = np.arange(1, n + 1)
a1 = observed_mean[0]

p_rep_m10 = simulate_p_rep(10, n_resp)

mult_nofloor = predict_multiplicative(a1, p_rep_m10, diag_mean, offdiag_mean, floor=0.0)
mult_floor = predict_multiplicative(a1, p_rep_m10, diag_mean, offdiag_mean, floor=a_floor)
add_pred = predict_additive(a1, p_rep_m10, diag_mean, offdiag_mean)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: a_k curves
ax = axes[0]
ax.plot(k, observed_mean, "ko-", linewidth=2, markersize=4, label="Observed (1k draws)", zorder=5)
ax.fill_between(k, observed_mean - observed_sem, observed_mean + observed_sem, alpha=0.2, color="black")
ax.plot(k, mult_floor, "m-", linewidth=2.5, label=f"Fractional + floor ({a_floor:.0f})", zorder=4)
ax.plot(k, mult_nofloor, "m--", linewidth=1.5, alpha=0.5, label="Fractional (no floor)")
ax.plot(k, add_pred, "r-.", linewidth=1.5, alpha=0.5, label="Additive")
ax.axhline(a_floor, color="purple", linewidth=0.8, linestyle=":", alpha=0.5, label="Floor")
ax.set_xlabel("k (response index)", fontsize=12)
ax.set_ylabel("$a_k$ (bits)", fontsize=12)
ax.set_title("Predicted vs Observed $a_k$ Curve (m=10)", fontweight="bold")
ax.legend(fontsize=7.5)

# Panel 2: Marginal drops
ax = axes[1]
k_mid = np.arange(1, n)
obs_marg = -np.diff(observed_mean)
floor_marg = -np.diff(mult_floor)
nofloor_marg = -np.diff(mult_nofloor)
add_marg = -np.diff(add_pred)

ax.plot(k_mid, obs_marg, "ko-", linewidth=2, markersize=4, label="Observed", zorder=5)
ax.plot(k_mid, floor_marg, "m-", linewidth=2.5, label="Fractional + floor", zorder=4)
ax.plot(k_mid, nofloor_marg, "m--", linewidth=1.5, alpha=0.5, label="Fractional (no floor)")
ax.plot(k_mid, add_marg, "r-.", linewidth=1.5, alpha=0.5, label="Additive")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("k (step from $a_k$ to $a_{k+1}$)", fontsize=12)
ax.set_ylabel("Marginal drop $a_k - a_{k+1}$ (bits)", fontsize=12)
ax.set_title("Marginal Drops: Predicted vs Observed", fontweight="bold")
ax.legend(fontsize=8)

# Panel 3: Ratio observed/predicted for floor model
ax = axes[2]
floor_ratio = obs_marg / np.clip(np.abs(floor_marg), 0.01, None) * np.sign(floor_marg)
ax.plot(k_mid, floor_ratio, "mo-", linewidth=2, markersize=4, label="Fractional + floor")
add_ratio = obs_marg / np.clip(np.abs(add_marg), 0.01, None) * np.sign(add_marg)
ax.plot(k_mid, add_ratio, "r.-", linewidth=1.5, alpha=0.5, label="Additive")
ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", label="Perfect prediction")
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel("Observed / Predicted marginal drop", fontsize=12)
ax.set_title("Prediction Accuracy (ratio)", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(-1, 5)

plt.tight_layout()
fig_path = FIGURES_DIR / "predicted_vs_observed_m10.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Fractional + floor vs observed across ALL m values
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(22, 8))

for idx, m_val in enumerate(range(1, 11)):
    row, col = divmod(idx, 5)
    ax = axes[row][col]

    curves = all_m_curves[m_val]
    mean_curve = np.mean(curves, axis=0)
    n_k = len(mean_curve)
    k_arr = np.arange(1, n_k + 1)
    a1_m = mean_curve[0]

    p_rep = simulate_p_rep(m_val, n_resp)
    floor_curve = predict_multiplicative(a1_m, p_rep, diag_mean, offdiag_mean, floor=a_floor)
    nofloor_curve = predict_multiplicative(a1_m, p_rep, diag_mean, offdiag_mean, floor=0.0)

    ax.plot(k_arr, mean_curve, "ko-", linewidth=2, markersize=3, label="Observed", zorder=5)
    ax.plot(k_arr, floor_curve, "m-", linewidth=2, label="Fractional + floor", zorder=4)
    ax.plot(k_arr, nofloor_curve, "m--", linewidth=1, alpha=0.4, label="Fractional (no floor)")
    ax.axhline(a_floor, color="purple", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_title(f"m = {m_val}", fontweight="bold", fontsize=11)
    if row == 1:
        ax.set_xlabel("k")
    if col == 0:
        ax.set_ylabel("$a_k$ (bits)")
    if idx == 0:
        ax.legend(fontsize=5.5)

fig.suptitle(
    f"Fractional Model (floor = {a_floor:.0f} bits from m=1 asymptote) vs Observed",
    fontsize=14, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path2 = FIGURES_DIR / "predicted_vs_observed_all_m.png"
plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path2}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Marginal drops comparison (bar chart) across all m
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(22, 8))

for idx, m_val in enumerate(range(1, 11)):
    row, col = divmod(idx, 5)
    ax = axes[row][col]

    curves = all_m_curves[m_val]
    mean_curve = np.mean(curves, axis=0)
    obs_marginals = -np.diff(mean_curve)
    n_k = len(mean_curve)
    k_plot = np.arange(1, n_k)
    a1_m = mean_curve[0]

    p_rep = simulate_p_rep(m_val, n_resp)
    floor_curve = predict_multiplicative(a1_m, p_rep, diag_mean, offdiag_mean, floor=a_floor)
    floor_marginals = -np.diff(floor_curve)
    add_marginals = offdiag_mean + p_rep[1:] * (diag_mean - offdiag_mean)

    w = 0.25
    ax.bar(k_plot - w, obs_marginals, width=w, color="black", alpha=0.7, label="Observed")
    ax.bar(k_plot, floor_marginals, width=w, color="purple", alpha=0.5, label="Fractional + floor")
    ax.bar(k_plot + w, add_marginals, width=w, color="red", alpha=0.3, label="Additive")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"m = {m_val}", fontweight="bold", fontsize=10)
    if row == 1:
        ax.set_xlabel("k")
    if col == 0:
        ax.set_ylabel("Marginal drop (bits)")
    if idx == 0:
        ax.legend(fontsize=5)

fig.suptitle(
    "Marginal Drops: Observed vs Fractional + Floor vs Additive (all m)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path3 = FIGURES_DIR / "marginal_drops_all_m.png"
plt.savefig(fig_path3, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path3}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Predicted vs observed total drop (scatter) for all models
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

colors_m = plt.cm.viridis(np.linspace(0.1, 0.9, 10))
for idx, m_val in enumerate(range(1, 11)):
    curves = all_m_curves[m_val]
    mean_curve = np.mean(curves, axis=0)
    a1_m = mean_curve[0]

    p_rep = simulate_p_rep(m_val, n_resp)
    floor_curve = predict_multiplicative(a1_m, p_rep, diag_mean, offdiag_mean, floor=a_floor)

    obs_drop = mean_curve[0] - mean_curve[-1]
    pred_drop = floor_curve[0] - floor_curve[-1]

    ax.scatter(pred_drop, obs_drop, s=100, color=colors_m[idx], zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(
        f"m={m_val}",
        (pred_drop, obs_drop),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=9,
    )

lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="y = x (perfect)")
ax.set_xlabel("Predicted total drop $a_1 - a_{20}$ (bits)", fontsize=12)
ax.set_ylabel("Observed total drop $a_1 - a_{20}$ (bits)", fontsize=12)
ax.set_title(f"Fractional + Floor Model: Predicted vs Observed Total Drop", fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.tight_layout()
fig_path4 = FIGURES_DIR / "multiplicative_total_drop.png"
plt.savefig(fig_path4, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path4}")
plt.close()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n=== Fractional model parameters ===")
r_offdiag = offdiag_mean / a1
r_diag = diag_mean / a1
print(f"a_1 (m=10): {a1:.1f} bits")
print(f"Floor (m=1 asymptote): {a_floor:.1f} bits")
print(f"Resolvable: a_1 - floor = {a1 - a_floor:.1f} bits")
print(f"r_offdiag = offdiag/a_1 = {offdiag_mean:.1f}/{a1:.1f} = {r_offdiag:.4f} ({r_offdiag*100:.2f}%)")
print(f"r_diag = diag/a_1 = {diag_mean:.1f}/{a1:.1f} = {r_diag:.4f} ({r_diag*100:.1f}%)")

print(f"\n=== Comparison at m=10 ===")
print(f"{'k':>3s}  {'observed':>9s}  {'frac+floor':>11s}  {'frac_only':>10s}  {'additive':>9s}")
for i in range(n):
    obs = observed_mean[i]
    mf = mult_floor[i]
    mn = mult_nofloor[i]
    ap = add_pred[i]
    print(f"{i+1:3d}  {obs:9.1f}  {mf:11.1f}  {mn:10.1f}  {ap:9.1f}")

print(f"\n=== Total drop comparison (all m) ===")
print(f"{'m':>3s}  {'observed':>9s}  {'frac+floor':>11s}  {'ratio':>6s}  {'additive':>9s}  {'ratio':>6s}")
for m_val in range(1, 11):
    curves = all_m_curves[m_val]
    mean_curve = np.mean(curves, axis=0)
    a1_m = mean_curve[0]
    p_rep = simulate_p_rep(m_val, n_resp)
    fc = predict_multiplicative(a1_m, p_rep, diag_mean, offdiag_mean, floor=a_floor)
    ac = predict_additive(a1_m, p_rep, diag_mean, offdiag_mean)
    obs_drop = mean_curve[0] - mean_curve[-1]
    floor_drop = fc[0] - fc[-1]
    add_drop = ac[0] - ac[-1]
    print(
        f"m={m_val:2d}  {obs_drop:9.1f}  {floor_drop:11.1f}  {floor_drop/obs_drop:5.1f}x  "
        f"{add_drop:9.1f}  {add_drop/obs_drop:5.1f}x"
    )
