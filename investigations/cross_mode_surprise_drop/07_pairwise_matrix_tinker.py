"""Pairwise cross-mode surprise reduction matrix — parameterized for any model.

Adapted from 07_pairwise_matrix.py to support Tinker API and CLI arguments.
Uses the same experimental setup (15 modes, 5 samples, same seed) so results
are directly comparable across models.

Usage:
    # Via Tinker
    uv run python investigations/cross_mode_surprise_drop/07_pairwise_matrix_tinker.py \
        --base-model meta-llama/Llama-3.2-1B --provider tinker

    # Local GPU
    uv run python investigations/cross_mode_surprise_drop/07_pairwise_matrix_tinker.py \
        --base-model Qwen/Qwen2.5-3B --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_diversity.core import compute_cross_entropy  # noqa: E402
from icl_diversity.mode_count_scenarios import (  # noqa: E402
    MODE_NAMES,
    PROMPT,
    _ALL_RAIN_MODES,
)

matplotlib.use("Agg")
torch.set_grad_enabled(False)

# Experiment parameters (same as 07_pairwise_matrix.py for comparability)
N_MODES = 15
M_SAMPLES = 5
SEED = 42


def derive_model_tag(model_name: str) -> str:
    """Derive a short directory name from model identifier."""
    short = model_name.split("/")[-1].lower()
    short = short.replace(".", "_").replace("-", "_")
    return short


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pairwise cross-mode surprise reduction matrix"
    )
    parser.add_argument(
        "--base-model", required=True, help="Model identifier"
    )
    parser.add_argument(
        "--provider",
        choices=["local", "tinker"],
        default="local",
        help="Model provider (default: local)",
    )
    parser.add_argument(
        "--device", default="auto", help="Device for local models (default: auto)"
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for local models (default: float16)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: figures/<model_tag>)",
    )
    args = parser.parse_args()

    # Output directory
    figures_dir = Path(__file__).parent / "figures"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        tag = derive_model_tag(args.base_model)
        output_dir = figures_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.provider == "tinker":
        from icl_diversity.tinker_model import TinkerModel

        print(f"Connecting to Tinker: {args.base_model}")
        model = TinkerModel(model_name=args.base_model)
        tokenizer = model.tokenizer
    else:
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[args.torch_dtype]

        print(f"Loading {args.base_model} ({args.torch_dtype}, {device})...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch_dtype, device_map=device
        )
        model.eval()

    print("Model ready.\n")

    # -----------------------------------------------------------------------
    # Generate responses (deterministic, same for all models)
    # -----------------------------------------------------------------------
    modes = _ALL_RAIN_MODES[:N_MODES]
    names = MODE_NAMES[:N_MODES]

    responses_by_mode: list[list[str]] = []
    for mode_idx, mode_fn in enumerate(modes):
        samples: list[str] = []
        for s in range(M_SAMPLES):
            rng = random.Random(SEED + mode_idx * 1000 + s)
            samples.append(mode_fn(rng))
        responses_by_mode.append(samples)

    print(f"Generated {N_MODES} x {M_SAMPLES} = {N_MODES * M_SAMPLES} responses:")
    for i, (name, samples) in enumerate(zip(names, responses_by_mode)):
        preview = samples[0][:60].replace("\n", " ")
        print(f"  [{i:2d}] {name:20s}: {preview}... ({M_SAMPLES} samples)")
    print()

    # -----------------------------------------------------------------------
    # Compute unconditional surprises
    # -----------------------------------------------------------------------
    print(f"Computing unconditional surprises ({N_MODES * M_SAMPLES} passes)...")
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
            f"  mode {j:2d} ({names[j]:20s}): {mean_bits:7.1f} +/- {std_bits:.1f} bits"
        )
    print()

    # -----------------------------------------------------------------------
    # Compute conditional surprises
    # -----------------------------------------------------------------------
    total_passes = N_MODES * N_MODES * M_SAMPLES
    print(
        f"Computing {N_MODES}x{N_MODES}x{M_SAMPLES} = {total_passes} "
        f"conditional surprises..."
    )
    conditional_all = np.zeros((N_MODES, N_MODES, M_SAMPLES))
    t0 = time.time()

    for i in range(N_MODES):
        for j in range(N_MODES):
            for k in range(M_SAMPLES):
                context_k = (k + 1) % M_SAMPLES
                prefix = (
                    PROMPT
                    + "\n\nResponse A: "
                    + responses_by_mode[i][context_k]
                    + "\n\nResponse B: "
                )
                total_bits, _ = compute_cross_entropy(
                    model, tokenizer, responses_by_mode[j][k], prefix
                )
                conditional_all[i, j, k] = total_bits

        elapsed = time.time() - t0
        passes_done = (i + 1) * N_MODES * M_SAMPLES
        rate = passes_done / elapsed
        eta = (total_passes - passes_done) / rate if rate > 0 else 0
        print(
            f"  row {i:2d}/{N_MODES} ({names[i]:20s}) done — "
            f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        )

    print(f"\nAll conditional passes done in {time.time() - t0:.0f}s.\n")

    # -----------------------------------------------------------------------
    # Compute surprise reduction matrix
    # -----------------------------------------------------------------------
    unconditional_arr = np.array(unconditional_all)
    reduction_all = unconditional_arr[np.newaxis, :, :] - conditional_all
    reduction_mean = reduction_all.mean(axis=2)
    reduction_std = reduction_all.std(axis=2)

    # Summary statistics
    diag_mean = np.diag(reduction_mean)
    off_diag_mask = ~np.eye(N_MODES, dtype=bool)
    off_diag_vals = reduction_mean[off_diag_mask]

    print("=== Surprise Reduction Matrix (bits) ===")
    print(
        f"Diagonal (same-mode):   mean={diag_mean.mean():.1f} +/- "
        f"{np.diag(reduction_std).mean():.1f}"
    )
    print(
        f"Off-diagonal (cross):   mean={off_diag_vals.mean():.1f} +/- "
        f"{off_diag_vals.std():.1f}"
    )
    print(f"Fraction off-diag > 0:  {(off_diag_vals > 0).mean():.1%}")

    # Symmetry statistics
    xs, ys = [], []
    for i in range(N_MODES):
        for j in range(i + 1, N_MODES):
            xs.append(reduction_mean[i, j])
            ys.append(reduction_mean[j, i])
    xs, ys = np.array(xs), np.array(ys)
    coeffs = np.polyfit(xs, ys, 1)
    r2 = np.corrcoef(xs, ys)[0, 1] ** 2

    print(f"\nSymmetry: slope={coeffs[0]:.3f}, intercept={coeffs[1]:.2f}, R²={r2:.3f}")
    print()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "model": args.base_model,
        "provider": args.provider,
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
        "summary": {
            "diagonal_mean": float(diag_mean.mean()),
            "off_diagonal_mean": float(off_diag_vals.mean()),
            "off_diagonal_frac_positive": float((off_diag_vals > 0).mean()),
            "symmetry_r2": float(r2),
            "symmetry_slope": float(coeffs[0]),
            "symmetry_intercept": float(coeffs[1]),
        },
    }
    json_path = output_dir / "pairwise_matrix.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw data to {json_path}")

    # -----------------------------------------------------------------------
    # Plot heatmap
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    ax = axes[0]
    vmax = max(abs(reduction_mean.min()), abs(reduction_mean.max()))
    im = ax.imshow(reduction_mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(N_MODES))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(N_MODES))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Target mode j")
    ax.set_ylabel("Context mode i")
    ax.set_title(
        f"{args.base_model}\nMean Surprise Reduction (bits, n={M_SAMPLES})"
    )
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(N_MODES):
        for j in range(N_MODES):
            m = reduction_mean[i, j]
            s = reduction_std[i, j]
            color = "white" if abs(m) > vmax * 0.6 else "black"
            ax.text(j, i, f"{m:.0f}", ha="center", va="center", fontsize=5, color=color)

    ax2 = axes[1]
    im2 = ax2.imshow(reduction_std, cmap="YlOrRd", aspect="equal")
    ax2.set_xticks(range(N_MODES))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(N_MODES))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_title(f"Std of Surprise Reduction (n={M_SAMPLES})")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

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
    ax3.set_title(f"Unconditional surprise per mode")
    ax3.invert_yaxis()

    plt.tight_layout()
    fig_path = output_dir / "pairwise_matrix_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {fig_path}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot symmetry scatter
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(xs, ys, s=30, alpha=0.7, zorder=5, color="steelblue")

    fit_x = np.linspace(
        min(xs.min(), ys.min()) - 2, max(xs.max(), ys.max()) + 2, 100
    )
    fit_y = np.polyval(coeffs, fit_x)
    ax.plot(
        fit_x, fit_y, "r--", linewidth=1.5, alpha=0.7,
        label=f"Fit: y = {coeffs[0]:.2f}x {coeffs[1]:+.1f}  (R\u00b2={r2:.2f})",
    )

    lims = [min(xs.min(), ys.min()) - 3, max(xs.max(), ys.max()) + 3]
    ax.plot(lims, lims, "k:", linewidth=0.8, alpha=0.5, label="y = x (perfect symmetry)")

    ax.set_xlabel("reduction[i, j]: seeing mode i reduces surprise for mode j (bits)")
    ax.set_ylabel("reduction[j, i]: seeing mode j reduces surprise for mode i (bits)")
    ax.set_title(
        f"{args.base_model}: Pairwise Symmetry \u2014 Each Point is One (i,j) Pair"
    )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig_path = output_dir / "pairwise_symmetry.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved symmetry plot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
