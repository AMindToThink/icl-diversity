"""Token-level attribution: where in a target response does cross-mode context help?

For selected (mode_i, mode_j) pairs:
  - Generate 5 prefix responses from mode_i
  - Generate 1 target response from mode_j
  - Compute per-token log-probs of the target with and without each prefix
  - Average the per-token deltas across the 5 prefixes
  - Visualize which tokens benefit most from cross-mode context

Requires: Run 07_pairwise_matrix.py first to identify interesting pairs,
OR use the default pairs specified below.
"""

import json
import math
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.core import _ensure_models, _forward_log_probs
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
SEED = 42
N_PREFIX_SAMPLES = 5  # Number of prefix responses to average over
DEVICE = "cuda:1"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Pairs to analyze: (context_mode_idx, target_mode_idx)
# Default selection covers diverse structural contrasts.
# Override with results from 07_pairwise_matrix.py if available.
DEFAULT_PAIRS: list[tuple[int, int]] = [
    # Same-mode control
    (7, 7),    # philosophy → philosophy
    (1, 1),    # python_code → python_code
    # Cross-mode: structurally very different
    (1, 7),    # python_code → philosophy
    (7, 1),    # philosophy → python_code
    (14, 7),   # json_data → philosophy
    (2, 7),    # recipe → philosophy
    # Cross-mode: similar structure
    (3, 11),   # scientific_fact → historical_fact
    (8, 7),    # song_lyrics → philosophy
]

# Try to load pairs from pairwise matrix results
MATRIX_JSON = OUTPUT_DIR / "pairwise_matrix.json"


def load_interesting_pairs() -> list[tuple[int, int]]:
    """Load top cross-mode pairs from pairwise matrix results if available."""
    if not MATRIX_JSON.exists():
        print("No pairwise matrix results found, using default pairs.")
        return DEFAULT_PAIRS

    with open(MATRIX_JSON) as f:
        data = json.load(f)

    reduction = np.array(data["reduction_matrix"])
    n = reduction.shape[0]
    mode_names = data["mode_names"]

    # Find top 5 cross-mode pairs + bottom 2 + 2 same-mode controls
    cross_pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i != j:
                cross_pairs.append((i, j, reduction[i, j]))
    cross_pairs.sort(key=lambda x: x[2], reverse=True)

    pairs: list[tuple[int, int]] = []
    # Top 5 cross-mode
    for i, j, val in cross_pairs[:5]:
        pairs.append((i, j))
        print(f"  Top pair: {mode_names[i]} → {mode_names[j]}: {val:+.1f} bits")
    # Bottom 2 cross-mode (surprise increase)
    for i, j, val in cross_pairs[-2:]:
        pairs.append((i, j))
        print(f"  Bottom pair: {mode_names[i]} → {mode_names[j]}: {val:+.1f} bits")
    # 2 same-mode controls (highest diagonal)
    diag = np.diag(reduction)
    top_diag = np.argsort(diag)[::-1][:2]
    for idx in top_diag:
        pairs.append((idx, idx))
        print(f"  Same-mode: {mode_names[idx]} → {mode_names[idx]}: {diag[idx]:+.1f} bits")

    return pairs


def compute_per_token_log_probs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    prefix: str,
) -> tuple[list[float], list[str]]:
    """Compute per-token log-probs (bits) for text conditioned on prefix.

    Returns:
        (log_probs_per_token, token_strings) where each log_prob is in bits
        (positive = surprising).
    """
    models = _ensure_models(model)

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    n_full = len(full_ids)

    if n_full <= n_prefix:
        return [], []

    input_ids = torch.tensor([full_ids])
    log_probs = _forward_log_probs(models, input_ids)[0]  # (seq_len,) in bits

    # Extract per-token log-probs for the text portion
    # log_probs[t-1] = log2 P(token[t] | token[0..t-1])
    text_log_probs = log_probs[n_prefix - 1 : n_full - 1]
    per_token_bits = [-lp.item() for lp in text_log_probs]

    # Decode each text token individually for labels
    text_token_ids = full_ids[n_prefix:]
    token_strings = [tokenizer.decode([tid]) for tid in text_token_ids]

    return per_token_bits, token_strings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    print("Model loaded.\n")

    pairs = load_interesting_pairs()
    all_modes = _ALL_RAIN_MODES
    all_names = MODE_NAMES

    results: list[dict] = []

    for pair_idx, (ctx_mode, tgt_mode) in enumerate(pairs):
        ctx_name = all_names[ctx_mode]
        tgt_name = all_names[tgt_mode]
        is_same = ctx_mode == tgt_mode
        label = f"{ctx_name}→{tgt_name}" + (" (same)" if is_same else "")
        print(f"\n{'='*60}")
        print(f"Pair {pair_idx + 1}/{len(pairs)}: {label}")
        print(f"{'='*60}")

        # Generate target response (fixed seed for consistency across pairs)
        tgt_rng = random.Random(SEED + 1000 + tgt_mode)
        target_text = all_modes[tgt_mode](tgt_rng)
        print(f"  Target ({tgt_name}): {target_text[:80]}...")

        # Unconditional: target with no context response
        uncond_prefix = PROMPT + "\n\nResponse A: "
        uncond_bits, uncond_tokens = compute_per_token_log_probs(
            model, tokenizer, target_text, uncond_prefix
        )
        total_uncond = sum(uncond_bits)
        print(f"  Unconditional: {total_uncond:.1f} total bits")

        # Conditional: average over N_PREFIX_SAMPLES different prefix responses
        all_cond_bits: list[list[float]] = []
        for s in range(N_PREFIX_SAMPLES):
            ctx_rng = random.Random(SEED + s * 100 + ctx_mode)
            ctx_text = all_modes[ctx_mode](ctx_rng)
            cond_prefix = (
                PROMPT
                + "\n\nResponse A: "
                + ctx_text
                + "\n\nResponse B: "
            )
            cond_bits, _ = compute_per_token_log_probs(
                model, tokenizer, target_text, cond_prefix
            )
            all_cond_bits.append(cond_bits)
            total_cond = sum(cond_bits)
            print(f"  Conditional (prefix {s}): {total_cond:.1f} total bits")

        # Average conditional bits across prefixes
        n_tokens = len(uncond_bits)
        mean_cond_bits = [
            sum(all_cond_bits[s][t] for s in range(N_PREFIX_SAMPLES)) / N_PREFIX_SAMPLES
            for t in range(n_tokens)
        ]

        # Delta: positive = context reduces surprise (helps)
        delta_bits = [uncond_bits[t] - mean_cond_bits[t] for t in range(n_tokens)]
        total_delta = sum(delta_bits)
        print(f"  Mean conditional: {sum(mean_cond_bits):.1f} total bits")
        print(f"  Total surprise reduction: {total_delta:+.1f} bits")

        # Where is the reduction concentrated?
        if n_tokens > 0:
            first_quarter = sum(delta_bits[: n_tokens // 4])
            last_three_quarters = sum(delta_bits[n_tokens // 4 :])
            print(
                f"  First quarter: {first_quarter:+.1f} bits "
                f"({first_quarter / total_delta * 100:.0f}% of total)"
                if abs(total_delta) > 0.1
                else f"  First quarter: {first_quarter:+.1f} bits"
            )

        results.append({
            "context_mode": ctx_name,
            "target_mode": tgt_name,
            "is_same_mode": bool(is_same),
            "n_tokens": n_tokens,
            "total_unconditional_bits": total_uncond,
            "total_conditional_bits": sum(mean_cond_bits),
            "total_delta_bits": total_delta,
            "per_token_delta": delta_bits,
            "per_token_unconditional": uncond_bits,
            "per_token_conditional": mean_cond_bits,
            "token_strings": uncond_tokens,
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    json_path = OUTPUT_DIR / "token_attribution.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # ------------------------------------------------------------------
    # Plot: per-token delta for each pair
    # ------------------------------------------------------------------
    n_pairs = len(results)
    n_cols = 2
    n_rows = math.ceil(n_pairs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes.flatten() if n_pairs > 1 else [axes]

    for idx, (res, ax) in enumerate(zip(results, axes_flat)):
        delta = res["per_token_delta"]
        tokens = res["token_strings"]
        n = len(delta)
        colors = ["green" if d > 0 else "red" for d in delta]
        ax.bar(range(n), delta, color=colors, width=1.0, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(
            f"{res['context_mode']} → {res['target_mode']}"
            + (" (same)" if res["is_same_mode"] else "")
            + f"\ntotal Δ = {res['total_delta_bits']:+.1f} bits",
            fontsize=9,
        )
        ax.set_xlabel("Token position")
        ax.set_ylabel("Surprise reduction (bits)")

        # Add token labels for top-5 most affected positions
        abs_delta = [abs(d) for d in delta]
        top_positions = sorted(range(n), key=lambda i: abs_delta[i], reverse=True)[:5]
        for pos in top_positions:
            token_label = tokens[pos].replace("\n", "\\n").strip()
            if len(token_label) > 10:
                token_label = token_label[:10] + "…"
            ax.annotate(
                f'"{token_label}"',
                (pos, delta[pos]),
                fontsize=5,
                rotation=45,
                ha="left",
                va="bottom" if delta[pos] > 0 else "top",
            )

    # Hide unused axes
    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "token_attribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fig_path}")
    plt.close()

    # ------------------------------------------------------------------
    # Summary plot: fraction of delta in first quarter vs pair
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_labels = []
    first_q_fracs = []
    total_deltas = []
    for res in results:
        label = f"{res['context_mode']}→{res['target_mode']}"
        if res["is_same_mode"]:
            label += " (same)"
        pair_labels.append(label)
        delta = res["per_token_delta"]
        n = len(delta)
        total = sum(delta)
        first_q = sum(delta[: n // 4])
        frac = first_q / total if abs(total) > 0.1 else 0
        first_q_fracs.append(frac)
        total_deltas.append(total)

    x = range(len(pair_labels))
    bars = ax.bar(x, first_q_fracs, color="steelblue", alpha=0.8)
    ax.axhline(0.25, color="gray", linewidth=1, linestyle="--", label="uniform (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of surprise reduction in first 25% of tokens")
    ax.set_title("Where is cross-mode surprise reduction concentrated?")
    ax.legend()

    # Annotate with total delta
    for i, (bar, td) in enumerate(zip(bars, total_deltas)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"Δ={td:+.0f}",
            ha="center",
            fontsize=6,
        )

    plt.tight_layout()
    fig_path2 = OUTPUT_DIR / "token_attribution_summary.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    print(f"Saved summary to {fig_path2}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
