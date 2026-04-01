"""Compare per-token logprobs from Tinker API vs local HuggingFace model.

Generates a side-by-side bar chart showing per-token surprise (in bits)
for the same input text, using both Tinker and a locally-loaded model.

Usage:
    uv run python scripts/compare_tinker_local.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.core import _forward_log_probs
from icl_diversity.tinker_model import TinkerModel

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = Path("figures/tinker_validation")

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1492, Columbus sailed the ocean blue.",
    (
        "def fibonacci(n):\n    if n <= 1:\n"
        "        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    ),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading local model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    local_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32
    )
    local_model.eval()

    print(f"Connecting to Tinker: {MODEL_NAME}")
    tinker_model = TinkerModel(model_name=MODEL_NAME)

    for text_idx, text in enumerate(TEST_TEXTS):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([token_ids])

        # Get logprobs from both
        tinker_lp = tinker_model.score_sequences(input_ids)
        local_lp = _forward_log_probs([local_model], input_ids, temperature=1.0)

        # Convert to surprise (negate log-prob → positive bits)
        tinker_surprise = -tinker_lp[0].numpy()
        local_surprise = -local_lp[0].numpy()

        # Decode tokens for x-axis labels
        token_labels = [
            tokenizer.decode([tid]).replace("\n", "\\n") for tid in token_ids
        ]

        n_tokens = len(token_ids)
        # Last position is always 0 (no next token), skip it for plotting
        positions = range(n_tokens - 1)

        fig, axes = plt.subplots(2, 1, figsize=(max(12, n_tokens * 0.6), 8))

        # Top: side-by-side bars
        width = 0.35
        x = list(positions)
        axes[0].bar(
            [xi - width / 2 for xi in x],
            tinker_surprise[:-1],
            width,
            label="Tinker API",
            color="#2196F3",
            alpha=0.8,
        )
        axes[0].bar(
            [xi + width / 2 for xi in x],
            local_surprise[:-1],
            width,
            label="Local HF (float32)",
            color="#FF9800",
            alpha=0.8,
        )
        axes[0].set_ylabel("Surprise (bits)")
        axes[0].set_title(
            f"Per-token surprise: Tinker vs Local\n"
            f"Text: {text[:80]}{'...' if len(text) > 80 else ''}"
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            [f"{token_labels[i]}→{token_labels[i+1]}" for i in positions],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axes[0].legend()

        # Bottom: difference
        diff = tinker_surprise[:-1] - local_surprise[:-1]
        colors = ["#4CAF50" if abs(d) < 0.01 else "#f44336" for d in diff]
        axes[1].bar(x, diff, color=colors, alpha=0.8)
        axes[1].axhline(y=0, color="black", linewidth=0.5)
        axes[1].set_ylabel("Difference (Tinker - Local, bits)")
        axes[1].set_title("Per-token difference")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [f"{token_labels[i]}→{token_labels[i+1]}" for i in positions],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        max_diff = max(abs(d) for d in diff)
        mean_diff = sum(abs(d) for d in diff) / len(diff)
        axes[1].text(
            0.98,
            0.95,
            f"max |diff| = {max_diff:.4f} bits\nmean |diff| = {mean_diff:.4f} bits",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"tinker_vs_local_text{text_idx}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
        print(f"    max|diff|={max_diff:.4f} bits, mean|diff|={mean_diff:.4f} bits")


if __name__ == "__main__":
    main()
