"""Analyze the McDiv_nuggets dataset construction confound.

Demonstrates that low-diversity story completions have systematically higher
per-byte surprise (a_1) than high-diversity ones, due to crowd-workers
paraphrasing specific/dramatic endings for the low-diversity condition.

Generates:
- Examples table comparing high vs low diversity response sets
- Per-token surprise bar graphs for selected examples
- Summary statistics for the a_1 gap

Usage:
    uv run python scripts/dataset_confound_analysis.py --device cuda:0
    uv run python scripts/dataset_confound_analysis.py --device cuda:0 --model Qwen/Qwen2.5-3B
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_diversity.core import (
    _ensure_models,
    _find_response_boundaries,
    _forward_log_probs,
    format_conditioning_context,
)

matplotlib.use("Agg")
torch.set_grad_enabled(False)

RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "dataset_confound"


def _format_token_label(tok: str) -> str:
    """Format a token for display. ▁ prefix marks leading space."""
    if tok.startswith(" "):
        return "▁" + tok[1:].replace("\n", "↵").replace("\t", "→")
    t = tok.replace("\n", "↵").replace("\t", "→")
    if not t.strip():
        return repr(tok).strip("'").replace("\\n", "↵").replace("\\t", "→")
    return t


def load_story_gen_data(
    data_dir: Path, tag: str,
) -> list[dict]:
    """Load story_gen with_hds samples with per-byte a_1 values."""
    csv_path = sorted(data_dir.glob("*_with_hds_story_gen*.csv"))[0]
    sidecar_path = csv_path.with_suffix(f".icl_curves.{tag}.json")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(sidecar_path) as f:
        sidecar = {k: v for k, v in json.load(f).items() if not k.startswith("__")}

    resp_cols = sorted(
        [c for c in rows[0] if c.startswith("resp_") and c.replace("resp_", "").isdigit()],
        key=lambda x: int(x.replace("resp_", "")),
    )

    samples: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in sidecar:
            continue
        entry = sidecar[sid]
        a_k_pb = entry.get("a_k_curve_per_byte")
        a_k = entry.get("a_k_curve")
        if a_k_pb is None or a_k is None:
            continue

        responses = [row[c] for c in resp_cols]
        label = float(row["label_value"])
        mean_bytes = float(np.mean([len(r.encode("utf-8")) for r in responses]))

        samples.append({
            "sample_id": sid,
            "label": label,
            "context": row["context"],
            "responses": responses,
            "a_k": a_k,
            "a_k_per_byte": a_k_pb,
            "a_1_per_byte": a_k_pb[0],
            "a_1_bits": a_k[0],
            "mean_bytes": mean_bytes,
        })

    return samples


def find_illustrative_examples(
    samples: list[dict], n: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Find examples that clearly illustrate the confound.

    Returns (high_div_examples, low_div_examples) where:
    - high_div examples have LOW per-byte a_1 (generic completions)
    - low_div examples have HIGH per-byte a_1 (specific/dramatic completions)
    """
    high_div = [s for s in samples if s["label"] == 1.0]
    low_div = [s for s in samples if s["label"] == 0.0]

    # Sort by a_1 per-byte
    high_div.sort(key=lambda s: s["a_1_per_byte"])
    low_div.sort(key=lambda s: s["a_1_per_byte"], reverse=True)

    # Pick examples near the median to avoid outliers
    mid_h = len(high_div) // 4
    mid_l = len(low_div) // 4

    return high_div[mid_h : mid_h + n], low_div[mid_l : mid_l + n]


def write_examples_report(
    samples: list[dict],
    high_examples: list[dict],
    low_examples: list[dict],
    output_path: Path,
) -> None:
    """Write a text report with examples and statistics."""
    high_div = [s for s in samples if s["label"] == 1.0]
    low_div = [s for s in samples if s["label"] == 0.0]

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("DATASET CONFOUND ANALYSIS — McDiv_nuggets story_gen")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Low-diversity response sets have systematically higher per-byte")
    lines.append("surprise (a_1) than high-diversity sets. This is because low-diversity")
    lines.append("sets are paraphrases of specific/dramatic endings, while high-diversity")
    lines.append("sets contain multiple generic/predictable completions.")
    lines.append("")

    # Statistics
    h_a1_pb = [s["a_1_per_byte"] for s in high_div]
    l_a1_pb = [s["a_1_per_byte"] for s in low_div]
    h_a1_bits = [s["a_1_bits"] for s in high_div]
    l_a1_bits = [s["a_1_bits"] for s in low_div]
    h_bytes = [s["mean_bytes"] for s in high_div]
    l_bytes = [s["mean_bytes"] for s in low_div]

    lines.append("SUMMARY STATISTICS")
    lines.append(f"  {'':30s} {'HIGH diversity':>15s} {'LOW diversity':>15s} {'Gap':>10s}")
    lines.append(f"  {'':30s} {'─'*15} {'─'*15} {'─'*10}")
    lines.append(f"  {'a_1 (bits/byte)':30s} {np.mean(h_a1_pb):>15.4f} {np.mean(l_a1_pb):>15.4f} {np.mean(l_a1_pb)-np.mean(h_a1_pb):>+10.4f}")
    lines.append(f"  {'a_1 (total bits)':30s} {np.mean(h_a1_bits):>15.1f} {np.mean(l_a1_bits):>15.1f} {np.mean(l_a1_bits)-np.mean(h_a1_bits):>+10.1f}")
    lines.append(f"  {'Mean response bytes':30s} {np.mean(h_bytes):>15.1f} {np.mean(l_bytes):>15.1f} {np.mean(l_bytes)-np.mean(h_bytes):>+10.1f}")
    lines.append(f"  {'n samples':30s} {len(high_div):>15d} {len(low_div):>15d}")
    lines.append("")

    for label_name, examples in [("HIGH DIVERSITY (low a_1)", high_examples),
                                  ("LOW DIVERSITY (high a_1)", low_examples)]:
        lines.append(f"\n{'━' * 70}")
        lines.append(f"  {label_name} — illustrative examples")
        lines.append(f"{'━' * 70}")
        for s in examples:
            lines.append(f"\n  Sample: {s['sample_id']}")
            lines.append(f"  a_1: {s['a_1_per_byte']:.3f} bits/byte ({s['a_1_bits']:.1f} total bits)")
            lines.append(f"  Mean response length: {s['mean_bytes']:.0f} bytes")
            lines.append(f"  Context: {s['context']}")
            for j, r in enumerate(s["responses"]):
                lines.append(f"    r{j}: {r}")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"Saved report: {output_path}")


def plot_a1_distributions(
    samples: list[dict],
    output_path: Path,
) -> None:
    """Plot a_1 distributions for high vs low diversity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("McDiv_nuggets story_gen: a₁ distributions by diversity label", fontsize=13)

    high = [s for s in samples if s["label"] == 1.0]
    low = [s for s in samples if s["label"] == 0.0]

    for ax, key, xlabel in [
        (axes[0], "a_1_per_byte", "a₁ (bits/byte)"),
        (axes[1], "a_1_bits", "a₁ (total bits)"),
    ]:
        h_vals = [s[key] for s in high]
        l_vals = [s[key] for s in low]
        hist_range = (min(h_vals + l_vals), max(h_vals + l_vals))

        ax.hist(h_vals, bins=20, alpha=0.5, color="tab:red",
                label=f"High diversity (n={len(high)})", range=hist_range)
        ax.hist(l_vals, bins=20, alpha=0.5, color="tab:blue",
                label=f"Low diversity (n={len(low)})", range=hist_range)
        ax.axvline(np.mean(h_vals), color="tab:red", linestyle="--", linewidth=2)
        ax.axvline(np.mean(l_vals), color="tab:blue", linestyle="--", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_token_surprise(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sample: dict,
    format_mode: str,
    output_path: Path,
) -> None:
    """Plot per-token surprise bar graph for a single sample's first response.

    Shows the unconditional surprise of response 0 (k=1, no other responses in context).
    """
    context = sample["context"]
    responses = sample["responses"]
    models = _ensure_models(model)

    # k=1: just prompt + first response
    full_ids, boundaries = _find_response_boundaries(
        tokenizer, context, responses[:1], format_mode=format_mode,
    )

    input_ids = torch.tensor([full_ids], device=next(model.parameters()).device)
    log_probs = _forward_log_probs(models, input_ids)[0]

    n_tokens = len(full_ids)
    surprises = np.zeros(n_tokens)
    for t in range(1, n_tokens):
        surprises[t] = -log_probs[t - 1].item()

    is_unmasked = np.zeros(n_tokens, dtype=bool)
    for start, end in boundaries:
        for t in range(start, end):
            is_unmasked[t] = True

    token_strings = [tokenizer.decode([tid]) for tid in full_ids]

    fig, ax = plt.subplots(figsize=(max(14, n_tokens * 0.45), 4.5))
    x = np.arange(n_tokens)
    colors = ["#d32f2f" if u else "#9e9e9e" for u in is_unmasked]
    ax.bar(x, surprises, color=colors, width=0.8, alpha=0.85)

    # Response boundary
    if boundaries:
        start, end = boundaries[0]
        if start > 0:
            ax.axvline(start - 0.5, color="blue", linewidth=1, linestyle="--", alpha=0.5)

    # Compute total bits for response
    if boundaries:
        s, e = boundaries[0]
        resp_bits = sum(surprises[t] for t in range(s, e))
        resp_bytes = len(responses[0].encode("utf-8"))
        bits_per_byte = resp_bits / resp_bytes if resp_bytes > 0 else 0
    else:
        resp_bits = 0
        bits_per_byte = 0

    label_str = "HIGH" if sample["label"] == 1.0 else "LOW"
    ax.set_title(
        f"{label_str} diversity — r₀: \"{responses[0][:60]}{'…' if len(responses[0]) > 60 else ''}\"\n"
        f"a₁ = {resp_bits:.1f} bits / {resp_bytes} bytes = {bits_per_byte:.3f} bits/byte",
        fontsize=10,
    )

    display_tokens = [_format_token_label(tok) for tok in token_strings]
    ax.set_xticks(x)
    ax.set_xticklabels(display_tokens, rotation=90, fontsize=8,
                       fontfamily="monospace", ha="center")
    ax.set_xlim(-0.5, n_tokens - 0.5)
    ax.set_ylabel("Surprise (bits)")

    legend_elements = [
        Patch(facecolor="#d32f2f", alpha=0.85, label="Response tokens (measured)"),
        Patch(facecolor="#9e9e9e", alpha=0.85, label="Context tokens (masked)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset confound analysis")
    parser.add_argument("--run-tag", type=str, default="qwen25_completion")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--format-mode", type=str, default="completion")
    parser.add_argument("--n-examples", type=int, default=2,
                        help="Number of examples per group")
    args = parser.parse_args()

    data_dir = RESULTS_BASE / args.run_tag / "McDiv_nuggets"
    if not data_dir.exists():
        print(f"No data at {data_dir}")
        sys.exit(1)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load data and find examples
    print("Loading story_gen data...")
    samples = load_story_gen_data(data_dir, args.run_tag)
    print(f"  {len(samples)} samples loaded")

    high_examples, low_examples = find_illustrative_examples(samples, args.n_examples)

    # Write report
    write_examples_report(samples, high_examples, low_examples,
                          output_dir / "confound_report.txt")

    # Plot a_1 distributions
    plot_a1_distributions(samples, output_dir / "a1_distributions.png")

    # Phase 2: Per-token surprise bar graphs
    print("\nLoading model for per-token surprise plots...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    all_examples = high_examples + low_examples
    for i, sample in enumerate(all_examples):
        label_str = "high" if sample["label"] == 1.0 else "low"
        safe_id = sample["sample_id"].replace("::", "_").replace("/", "_")
        out_path = output_dir / f"per_token_{label_str}_{i % args.n_examples}_{safe_id}.png"
        print(f"  {label_str} diversity: {sample['sample_id']}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_per_token_surprise(
                model, tokenizer, sample, args.format_mode, out_path,
            )

    print(f"\nAll outputs in {output_dir}")


if __name__ == "__main__":
    main()
