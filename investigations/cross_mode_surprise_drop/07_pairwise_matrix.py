"""Pairwise cross-mode surprise reduction matrix for Qwen2.5-3B.

For each ordered pair (mode_i, mode_j) from N modes:
  - conditional: surprise(target_j | prompt + "Response A: " + context_i + "\n\nResponse B: ")
  - unconditional: surprise(target_j | prompt + "\n\nResponse A: ")
  - surprise_reduction[i, j] = unconditional_j - conditional[i, j]

The resulting N×N matrix reveals which mode pairs share mutual information
under the base model, and whether cross-mode learning is systematic or
concentrated in specific pairs.
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
# Generate one response per mode (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
modes = _ALL_RAIN_MODES[:N_MODES]
names = MODE_NAMES[:N_MODES]

responses: list[str] = []
for mode_fn in modes:
    responses.append(mode_fn(rng))

print(f"Generated {N_MODES} responses:")
for i, (name, resp) in enumerate(zip(names, responses)):
    preview = resp[:80].replace("\n", " ")
    print(f"  [{i:2d}] {name:20s}: {preview}...")
print()

# ---------------------------------------------------------------------------
# Compute unconditional surprises (N forward passes)
# ---------------------------------------------------------------------------
print("Computing unconditional surprises...")
unconditional_bits: list[float] = []
unconditional_bytes: list[int] = []
for j, resp in enumerate(responses):
    prefix = PROMPT + "\n\nResponse A: "
    total_bits, byte_count = compute_cross_entropy(model, tokenizer, resp, prefix)
    unconditional_bits.append(total_bits)
    unconditional_bytes.append(byte_count)
    print(
        f"  mode {j:2d} ({names[j]:20s}): {total_bits:7.1f} bits "
        f"({total_bits / byte_count:.3f} bits/byte)"
    )
print()

# ---------------------------------------------------------------------------
# Compute conditional surprises (N×N forward passes)
# ---------------------------------------------------------------------------
print(f"Computing {N_MODES}×{N_MODES} = {N_MODES**2} conditional surprises...")
conditional_bits = np.zeros((N_MODES, N_MODES))
t0 = time.time()

for i in range(N_MODES):
    for j in range(N_MODES):
        # Surprise of response_j after seeing response_i
        prefix = (
            PROMPT
            + "\n\nResponse A: "
            + responses[i]
            + "\n\nResponse B: "
        )
        total_bits, _ = compute_cross_entropy(
            model, tokenizer, responses[j], prefix
        )
        conditional_bits[i, j] = total_bits

    elapsed = time.time() - t0
    rate = (i + 1) * N_MODES / elapsed
    eta = (N_MODES - i - 1) * N_MODES / rate
    print(
        f"  row {i:2d}/{N_MODES} ({names[i]:20s}) done — "
        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
    )

print(f"\nAll conditional passes done in {time.time() - t0:.0f}s.\n")

# ---------------------------------------------------------------------------
# Compute surprise reduction matrix
# ---------------------------------------------------------------------------
# reduction[i, j] = unconditional_j - conditional[i, j]
# Positive = seeing mode_i reduces surprise for mode_j
unconditional_arr = np.array(unconditional_bits)
reduction = unconditional_arr[np.newaxis, :] - conditional_bits

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------
diag = np.diag(reduction)
off_diag_mask = ~np.eye(N_MODES, dtype=bool)
off_diag = reduction[off_diag_mask]

print("=== Surprise Reduction Matrix (bits) ===")
print(f"Diagonal (same-mode):   mean={diag.mean():.1f}, std={diag.std():.1f}, "
      f"min={diag.min():.1f}, max={diag.max():.1f}")
print(f"Off-diagonal (cross):   mean={off_diag.mean():.1f}, std={off_diag.std():.1f}, "
      f"min={off_diag.min():.1f}, max={off_diag.max():.1f}")
print(f"Fraction off-diag > 0:  {(off_diag > 0).mean():.1%}")
print()

# Row means (how informative is mode_i on average?)
row_means = reduction.mean(axis=1)
print("Row means (informativeness of each mode as context):")
sorted_rows = np.argsort(row_means)[::-1]
for idx in sorted_rows:
    print(f"  {names[idx]:20s}: {row_means[idx]:+.1f} bits")
print()

# Column means (how much does mode_j benefit from context on average?)
col_means = reduction.mean(axis=0)
print("Column means (benefit from context for each mode):")
sorted_cols = np.argsort(col_means)[::-1]
for idx in sorted_cols:
    print(f"  {names[idx]:20s}: {col_means[idx]:+.1f} bits")
print()

# Symmetry check
asymmetry = reduction - reduction.T
print("Asymmetry (reduction[i,j] - reduction[j,i]):")
print(f"  mean abs asymmetry:  {np.abs(asymmetry[off_diag_mask]).mean():.1f} bits")
print(f"  max abs asymmetry:   {np.abs(asymmetry[off_diag_mask]).max():.1f} bits")
print()

# Top 10 cross-mode pairs by surprise reduction
print("Top 10 cross-mode pairs by surprise reduction:")
for rank in range(10):
    # Mask diagonal
    masked = reduction.copy()
    np.fill_diagonal(masked, -np.inf)
    for prev_rank in range(rank):
        prev_i, prev_j = divmod(
            np.argmax(masked if prev_rank == 0 else masked), N_MODES
        )
    flat_idx = np.argmax(masked)
    i, j = divmod(flat_idx, N_MODES)
    print(
        f"  {names[i]:20s} → {names[j]:20s}: "
        f"{reduction[i, j]:+.1f} bits "
        f"(uncond={unconditional_bits[j]:.1f}, cond={conditional_bits[i, j]:.1f})"
    )
    masked[i, j] = -np.inf

# Redo top-10 properly (the loop above was buggy)
print("\nTop 10 cross-mode pairs (corrected):")
cross_pairs: list[tuple[int, int, float]] = []
for i in range(N_MODES):
    for j in range(N_MODES):
        if i != j:
            cross_pairs.append((i, j, reduction[i, j]))
cross_pairs.sort(key=lambda x: x[2], reverse=True)
for i, j, val in cross_pairs[:10]:
    print(
        f"  {names[i]:20s} → {names[j]:20s}: "
        f"{val:+.1f} bits "
        f"(uncond={unconditional_bits[j]:.1f}, cond={conditional_bits[i, j]:.1f})"
    )
print("\nBottom 10 cross-mode pairs (largest surprise *increase*):")
for i, j, val in cross_pairs[-10:]:
    print(
        f"  {names[i]:20s} → {names[j]:20s}: "
        f"{val:+.1f} bits "
        f"(uncond={unconditional_bits[j]:.1f}, cond={conditional_bits[i, j]:.1f})"
    )

# ---------------------------------------------------------------------------
# Save raw data as JSON
# ---------------------------------------------------------------------------
results = {
    "model": MODEL_NAME,
    "n_modes": N_MODES,
    "seed": SEED,
    "mode_names": names,
    "unconditional_bits": unconditional_bits,
    "unconditional_bytes": unconditional_bytes,
    "conditional_bits": conditional_bits.tolist(),
    "reduction_matrix": reduction.tolist(),
    "responses": responses,
}
json_path = OUTPUT_DIR / "pairwise_matrix.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw data to {json_path}")

# ---------------------------------------------------------------------------
# Plot heatmap
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Surprise reduction heatmap
ax = axes[0]
vmax = max(abs(reduction.min()), abs(reduction.max()))
im = ax.imshow(
    reduction,
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
ax.set_title("Surprise Reduction (bits)\npositive = context helps")
plt.colorbar(im, ax=ax, shrink=0.8)

# Annotate cells with values
for i in range(N_MODES):
    for j in range(N_MODES):
        val = reduction[i, j]
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5, color=color)

# Unconditional surprises bar chart
ax2 = axes[1]
sorted_idx = np.argsort(unconditional_bits)[::-1]
ax2.barh(
    range(N_MODES),
    [unconditional_bits[i] for i in sorted_idx],
    color="steelblue",
)
ax2.set_yticks(range(N_MODES))
ax2.set_yticklabels([names[i] for i in sorted_idx], fontsize=7)
ax2.set_xlabel("Unconditional surprise (bits)")
ax2.set_title("Unconditional surprise per mode")
ax2.invert_yaxis()

plt.tight_layout()
fig_path = OUTPUT_DIR / "pairwise_matrix_heatmap.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved heatmap to {fig_path}")
plt.close()

# ---------------------------------------------------------------------------
# Plot: row means vs column means scatter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(row_means, col_means, s=50, zorder=5)
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
