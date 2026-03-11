"""Pairwise cross-mode surprise reduction matrix for Qwen2.5-3B.

For each ordered pair (mode_i, mode_j) from N modes, with M samples per mode:
  - conditional: surprise(target_j_k | prompt + "Response A: " + context_i_k + "\n\nResponse B: ")
  - unconditional: surprise(target_j_k | prompt + "\n\nResponse A: ")
  - surprise_reduction[i, j, k] = unconditional_j_k - conditional[i, j, k]

The resulting N×N matrix (with error bars from M samples) reveals which mode pairs
share mutual information under the base model.
"""

import json
import random
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.core import compute_cross_entropy
from icl_diversity.mode_count_scenarios import (
    MODE_NAMES,
    PROMPT,
    _ALL_RAIN_MODES,
)

matplotlib.use("Agg")

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_MODES = 15  # Start with 15; scale to 50 if results are interesting
M_SAMPLES = 5  # Responses per mode for error bars
SEED = 42
DEVICE = "cuda:1"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=DTYPE, device_map=DEVICE
)
model.eval()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Generate M responses per mode (different seeds for each sample)
# ---------------------------------------------------------------------------
modes = _ALL_RAIN_MODES[:N_MODES]
names = MODE_NAMES[:N_MODES]

# responses_by_mode[mode_idx][sample_idx] = response string
responses_by_mode: list[list[str]] = []
for mode_idx, mode_fn in enumerate(modes):
    samples: list[str] = []
    for s in range(M_SAMPLES):
        rng = random.Random(SEED + mode_idx * 1000 + s)
        samples.append(mode_fn(rng))
    responses_by_mode.append(samples)

print(f"Generated {N_MODES} × {M_SAMPLES} = {N_MODES * M_SAMPLES} responses:")
for i, (name, samples) in enumerate(zip(names, responses_by_mode)):
    preview = samples[0][:60].replace("\n", " ")
    print(f"  [{i:2d}] {name:20s}: {preview}... ({M_SAMPLES} samples)")
print()

# ---------------------------------------------------------------------------
# Compute unconditional surprises (N × M forward passes)
# ---------------------------------------------------------------------------
print(f"Computing unconditional surprises ({N_MODES * M_SAMPLES} passes)...")
# unconditional_all[j][k] = total_bits for mode j, sample k
unconditional_all: list[list[float]] = []
unconditional_bytes_all: list[list[int]] = []
for j in range(N_MODES):
    bits_list: list[float] = []
    bytes_list: list[int] = []
    for k in range(M_SAMPLES):
        prefix = PROMPT + "\n\nResponse A: "
        total_bits, byte_count = compute_cross_entropy(
            model, tokenizer, responses_by_mode[j][k], prefix
        )
        bits_list.append(total_bits)
        bytes_list.append(byte_count)
    unconditional_all.append(bits_list)
    unconditional_bytes_all.append(bytes_list)
    mean_bits = np.mean(bits_list)
    std_bits = np.std(bits_list)
    print(
        f"  mode {j:2d} ({names[j]:20s}): {mean_bits:7.1f} ± {std_bits:.1f} bits"
    )
print()

# ---------------------------------------------------------------------------
# Compute conditional surprises (N × N × M forward passes)
# For each (i, j, k): context = mode_i sample k, target = mode_j sample k
# ---------------------------------------------------------------------------
total_passes = N_MODES * N_MODES * M_SAMPLES
print(f"Computing {N_MODES}×{N_MODES}×{M_SAMPLES} = {total_passes} conditional surprises...")
# conditional_all[i][j][k] = total_bits
conditional_all = np.zeros((N_MODES, N_MODES, M_SAMPLES))
t0 = time.time()

for i in range(N_MODES):
    for j in range(N_MODES):
        for k in range(M_SAMPLES):
            prefix = (
                PROMPT
                + "\n\nResponse A: "
                + responses_by_mode[i][k]
                + "\n\nResponse B: "
            )
            total_bits, _ = compute_cross_entropy(
                model, tokenizer, responses_by_mode[j][k], prefix
            )
            conditional_all[i, j, k] = total_bits

    elapsed = time.time() - t0
    passes_done = (i + 1) * N_MODES * M_SAMPLES
    rate = passes_done / elapsed
    eta = (total_passes - passes_done) / rate
    print(
        f"  row {i:2d}/{N_MODES} ({names[i]:20s}) done — "
        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
    )

print(f"\nAll conditional passes done in {time.time() - t0:.0f}s.\n")

# ---------------------------------------------------------------------------
# Compute surprise reduction matrix with error bars
# ---------------------------------------------------------------------------
# reduction_all[i, j, k] = unconditional_j_k - conditional[i, j, k]
unconditional_arr = np.array(unconditional_all)  # (N_MODES, M_SAMPLES)
reduction_all = unconditional_arr[np.newaxis, :, :] - conditional_all  # (N, N, M)

reduction_mean = reduction_all.mean(axis=2)  # (N, N)
reduction_std = reduction_all.std(axis=2)    # (N, N)
reduction_sem = reduction_std / np.sqrt(M_SAMPLES)  # standard error of mean

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------
diag_mean = np.diag(reduction_mean)
diag_std = np.diag(reduction_std)
off_diag_mask = ~np.eye(N_MODES, dtype=bool)
off_diag_mean = reduction_mean[off_diag_mask]
off_diag_std = reduction_std[off_diag_mask]

print("=== Surprise Reduction Matrix (bits) ===")
print(f"Diagonal (same-mode):   mean={diag_mean.mean():.1f} ± {diag_std.mean():.1f}, "
      f"min={diag_mean.min():.1f}, max={diag_mean.max():.1f}")
print(f"Off-diagonal (cross):   mean={off_diag_mean.mean():.1f} ± {off_diag_std.mean():.1f}, "
      f"min={off_diag_mean.min():.1f}, max={off_diag_mean.max():.1f}")
print(f"Fraction off-diag > 0:  {(off_diag_mean > 0).mean():.1%}")
print()

# Row means (how informative is mode_i on average?)
row_means = reduction_mean.mean(axis=1)
# Error on row mean: propagate SEM across columns
row_sems = np.sqrt((reduction_sem ** 2).sum(axis=1)) / N_MODES
print("Row means (informativeness of each mode as context):")
sorted_rows = np.argsort(row_means)[::-1]
for idx in sorted_rows:
    print(f"  {names[idx]:20s}: {row_means[idx]:+.1f} ± {row_sems[idx]:.1f} bits")
print()

# Column means (how much does mode_j benefit from context on average?)
col_means = reduction_mean.mean(axis=0)
col_sems = np.sqrt((reduction_sem ** 2).sum(axis=0)) / N_MODES
print("Column means (benefit from context for each mode):")
sorted_cols = np.argsort(col_means)[::-1]
for idx in sorted_cols:
    print(f"  {names[idx]:20s}: {col_means[idx]:+.1f} ± {col_sems[idx]:.1f} bits")
print()

# Symmetry check
asymmetry = reduction_mean - reduction_mean.T
print("Asymmetry (reduction[i,j] - reduction[j,i]):")
print(f"  mean abs asymmetry:  {np.abs(asymmetry[off_diag_mask]).mean():.1f} bits")
print(f"  max abs asymmetry:   {np.abs(asymmetry[off_diag_mask]).max():.1f} bits")
print()

# Top 10 cross-mode pairs
print("Top 10 cross-mode pairs (corrected):")
cross_pairs: list[tuple[int, int, float, float]] = []
for i in range(N_MODES):
    for j in range(N_MODES):
        if i != j:
            cross_pairs.append((i, j, reduction_mean[i, j], reduction_std[i, j]))
cross_pairs.sort(key=lambda x: x[2], reverse=True)
for i, j, val, std in cross_pairs[:10]:
    print(
        f"  {names[i]:20s} → {names[j]:20s}: "
        f"{val:+.1f} ± {std:.1f} bits"
    )
print("\nBottom 10 cross-mode pairs (largest surprise *increase*):")
for i, j, val, std in cross_pairs[-10:]:
    print(
        f"  {names[i]:20s} → {names[j]:20s}: "
        f"{val:+.1f} ± {std:.1f} bits"
    )

# ---------------------------------------------------------------------------
# Save raw data as JSON
# ---------------------------------------------------------------------------
results = {
    "model": MODEL_NAME,
    "n_modes": N_MODES,
    "m_samples": M_SAMPLES,
    "seed": SEED,
    "mode_names": names,
    "unconditional_all": unconditional_all,
    "unconditional_bytes_all": unconditional_bytes_all,
    "conditional_all": conditional_all.tolist(),
    "reduction_all": reduction_all.tolist(),
    "reduction_mean": reduction_mean.tolist(),
    "reduction_std": reduction_std.tolist(),
    "responses_by_mode": responses_by_mode,
}
json_path = OUTPUT_DIR / "pairwise_matrix.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw data to {json_path}")

# ---------------------------------------------------------------------------
# Plot heatmap: mean reduction + std heatmap side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(26, 8))

# Mean surprise reduction heatmap
ax = axes[0]
vmax = max(abs(reduction_mean.min()), abs(reduction_mean.max()))
im = ax.imshow(
    reduction_mean,
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    aspect="equal",
)
ax.set_xticks(range(N_MODES))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(N_MODES))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("Target mode j (response being scored)")
ax.set_ylabel("Context mode i (response seen first)")
ax.set_title(f"Mean Surprise Reduction (bits, n={M_SAMPLES})\npositive = context helps")
plt.colorbar(im, ax=ax, shrink=0.8)

# Annotate cells with mean ± std
for i in range(N_MODES):
    for j in range(N_MODES):
        m = reduction_mean[i, j]
        s = reduction_std[i, j]
        color = "white" if abs(m) > vmax * 0.6 else "black"
        ax.text(
            j, i, f"{m:.0f}±{s:.0f}",
            ha="center", va="center", fontsize=4.5, color=color,
        )

# Std heatmap (uncertainty)
ax2 = axes[1]
im2 = ax2.imshow(reduction_std, cmap="YlOrRd", aspect="equal")
ax2.set_xticks(range(N_MODES))
ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax2.set_yticks(range(N_MODES))
ax2.set_yticklabels(names, fontsize=7)
ax2.set_xlabel("Target mode j")
ax2.set_ylabel("Context mode i")
ax2.set_title(f"Std of Surprise Reduction (bits, n={M_SAMPLES})")
plt.colorbar(im2, ax=ax2, shrink=0.8)

for i in range(N_MODES):
    for j in range(N_MODES):
        s = reduction_std[i, j]
        color = "white" if s > reduction_std.max() * 0.6 else "black"
        ax2.text(j, i, f"{s:.0f}", ha="center", va="center", fontsize=5, color=color)

# Unconditional surprises bar chart with error bars
ax3 = axes[2]
uncond_means = np.array([np.mean(u) for u in unconditional_all])
uncond_stds = np.array([np.std(u) for u in unconditional_all])
sorted_idx = np.argsort(uncond_means)[::-1]
ax3.barh(
    range(N_MODES),
    [uncond_means[i] for i in sorted_idx],
    xerr=[uncond_stds[i] for i in sorted_idx],
    color="steelblue",
    capsize=3,
)
ax3.set_yticks(range(N_MODES))
ax3.set_yticklabels([names[i] for i in sorted_idx], fontsize=7)
ax3.set_xlabel("Unconditional surprise (bits)")
ax3.set_title(f"Unconditional surprise per mode (n={M_SAMPLES})")
ax3.invert_yaxis()

plt.tight_layout()
fig_path = OUTPUT_DIR / "pairwise_matrix_heatmap.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved heatmap to {fig_path}")
plt.close()

# ---------------------------------------------------------------------------
# Plot: row means vs column means scatter with error bars
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.errorbar(
    row_means, col_means,
    xerr=row_sems, yerr=col_sems,
    fmt="o", markersize=6, capsize=3, zorder=5, color="steelblue",
)
for i in range(N_MODES):
    ax.annotate(
        names[i],
        (row_means[i], col_means[i]),
        fontsize=7,
        textcoords="offset points",
        xytext=(5, 5),
    )
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

# Line of best fit
coeffs = np.polyfit(row_means, col_means, 1)
fit_x = np.linspace(row_means.min() - 1, row_means.max() + 1, 100)
fit_y = np.polyval(coeffs, fit_x)
ax.plot(fit_x, fit_y, color="red", linewidth=1, linestyle="--", alpha=0.7,
        label=f"y = {coeffs[0]:.2f}x {coeffs[1]:+.1f}  (R²={np.corrcoef(row_means, col_means)[0,1]**2:.2f})")
ax.legend(fontsize=8, loc="lower right")

ax.set_xlabel("Row mean: how informative as context (bits)")
ax.set_ylabel("Column mean: how much benefit from context (bits)")
ax.set_title("Mode informativeness vs. context benefit")
ax.set_aspect("equal")
plt.tight_layout()
fig_path2 = OUTPUT_DIR / "pairwise_row_vs_col.png"
plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
print(f"Saved scatter to {fig_path2}")
plt.close()

print("\nDone.")
