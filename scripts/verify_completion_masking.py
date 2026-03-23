"""Verify completion-mode masking: visualize which tokens contribute to a_k curves.

Loads one sample from Tevet's McDiv_nuggets dataset, runs a single GPT-2
forward pass on CPU using the completion format, and produces a bar chart
showing per-token surprise with grey (masked) vs red (unmasked) bars.

Grey tokens are parts of the repeated context or labels ("1. {prompt}") that
are NOT counted in the a_k curve. Red tokens are the completion portions that
ARE measured.

The coloring uses the exact same boundary logic from _find_response_boundaries
with format_mode="completion" — not a reimplementation.

Usage:
    uv run python scripts/verify_completion_masking.py
    uv run python scripts/verify_completion_masking.py --device cuda:0
    uv run python scripts/verify_completion_masking.py --model Qwen/Qwen2.5-3B --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_diversity.core import (  # noqa: E402
    FormatMode,
    _ensure_models,
    _find_response_boundaries,
    _forward_log_probs,
)

matplotlib.use("Agg")
torch.set_grad_enabled(False)

FIGURES_DIR = PROJECT_ROOT / "figures" / "tevet_validation"


def load_sample(
    csv_path: Path, row_idx: int = 0
) -> tuple[str, list[str], str, str]:
    """Load a single sample from a Tevet CSV.

    Returns (context, responses, sample_id, label).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        resp_cols = sorted(
            [c for c in fieldnames if c.startswith("resp_") and c.replace("resp_", "").isdigit()],
            key=lambda x: int(x.replace("resp_", "")),
        )
        for i, row in enumerate(reader):
            if i == row_idx:
                context = row.get("context", "")
                responses = [row[c] for c in resp_cols]
                sample_id = row.get("sample_id", f"row_{i}")
                label = row.get("label_value", "?")
                return context, responses, sample_id, label
    raise ValueError(f"Row {row_idx} not found in {csv_path}")


def plot_masking(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context: str,
    responses: list[str],
    format_mode: FormatMode,
    sample_id: str,
    label: str,
    output_path: Path,
) -> None:
    """Plot per-token surprise with masked/unmasked coloring."""
    models = _ensure_models(model)

    # Get boundaries using the exact same function as the metric
    full_ids, boundaries = _find_response_boundaries(
        tokenizer, context, responses, format_mode=format_mode,
    )

    # Forward pass
    input_ids = torch.tensor([full_ids])
    log_probs = _forward_log_probs(models, input_ids)[0]  # (seq_len,) in bits

    # Per-token surprise: -log_probs[t-1] = surprise of token t
    # log_probs[t] = log2 P(token[t+1] | token[0..t])
    n_tokens = len(full_ids)
    surprises = np.zeros(n_tokens)
    for t in range(1, n_tokens):
        surprises[t] = -log_probs[t - 1].item()

    # Build mask: which tokens are inside a response boundary?
    is_unmasked = np.zeros(n_tokens, dtype=bool)
    for start, end in boundaries:
        for t in range(start, end):
            is_unmasked[t] = True

    # Decode tokens for labels
    token_strings = [tokenizer.decode([tid]) for tid in full_ids]

    # Plot
    fig, ax = plt.subplots(figsize=(max(20, n_tokens * 0.35), 6))
    x = np.arange(n_tokens)

    colors = ["#d32f2f" if u else "#9e9e9e" for u in is_unmasked]
    ax.bar(x, surprises, color=colors, width=0.8, alpha=0.8)

    # Vertical lines at response boundaries
    for k, (start, end) in enumerate(boundaries):
        if start > 0:
            ax.axvline(start - 0.5, color="blue", linewidth=1, linestyle="--", alpha=0.5)
        if end < n_tokens:
            ax.axvline(end - 0.5, color="blue", linewidth=1, linestyle="--", alpha=0.5)
        # Label
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5, f"r_{k+1}",
                ha="center", va="bottom", fontsize=8, color="blue", fontweight="bold")

    ax.set_xlabel("Token position")
    ax.set_ylabel("Surprise (bits)")
    ax.set_title(
        f"Completion Masking Verification — {sample_id} (label={label})\n"
        f"Red = measured (completion), Grey = masked (context/labels)\n"
        f"format_mode={format_mode!r}",
        fontsize=11,
    )

    # Token labels on x-axis
    display_tokens = []
    for tok in token_strings:
        t = tok.replace("\n", "\\n").replace("\t", "\\t")
        if not t.strip():
            t = repr(tok).strip("'")
        display_tokens.append(t)

    ax.set_xticks(x)
    ax.set_xticklabels(
        display_tokens,
        rotation=90,
        fontsize=5,
        fontfamily="monospace",
        ha="center",
    )
    ax.set_xlim(-0.5, n_tokens - 0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", alpha=0.8, label="Unmasked (completion)"),
        Patch(facecolor="#9e9e9e", alpha=0.8, label="Masked (context/label)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify completion masking")
    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="Model name (default: gpt2)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="CSV to load sample from (default: McDiv_nuggets story_gen)",
    )
    parser.add_argument(
        "--row", type=int, default=0,
        help="Row index to use (default: 0)",
    )
    parser.add_argument(
        "--format-mode", type=str, default="completion",
        choices=["instruct", "completion"],
        help="Format mode to visualize (default: completion)",
    )
    args = parser.parse_args()

    # Load sample
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = (
            PROJECT_ROOT / "results" / "tevet" / "gpt2" / "McDiv_nuggets"
            / "mcdiv_nuggets_no_hds_story_gen.csv"
        )
    print(f"Loading sample from {csv_path}")
    context, responses, sample_id, label = load_sample(csv_path, args.row)
    print(f"  Sample: {sample_id}, label={label}")
    print(f"  Context: {context[:80]}...")
    print(f"  {len(responses)} responses")

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()

    # Plot both formats for comparison
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for fmt in ["completion", "instruct"]:
        output_path = FIGURES_DIR / f"masking_verification_{fmt}.png"
        print(f"\nPlotting {fmt} format...")
        plot_masking(
            model, tokenizer, context, responses,
            format_mode=fmt,
            sample_id=sample_id, label=label,
            output_path=output_path,
        )

    print("\nDone! Compare the two plots to verify masking.")


if __name__ == "__main__":
    main()
