"""Disagreement analysis: find McDiv_nuggets samples where Tevet labels and E_fit disagree.

Identifies samples where Tevet says "high diversity" but E_fit is low (or vice versa),
loads Qwen2.5-3B to compute per-token logprobs for those samples, and generates
bar chart visualizations showing where the model is surprised.

Usage:
    uv run python scripts/disagreement_analysis.py --device cuda:0
    uv run python scripts/disagreement_analysis.py --device cuda:0 --top-n 5
    uv run python scripts/disagreement_analysis.py --device cuda:0 --batch-size 16
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
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fit_ak_curves import fit_exponential
from icl_diversity.core import (
    FormatMode,
    _ensure_models,
    _find_response_boundaries,
    _forward_log_probs,
    format_conditioning_context,
)

matplotlib.use("Agg")
torch.set_grad_enabled(False)

RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "tevet_validation" / "inspection"


def load_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mean_curves(curves_path: Path) -> dict[str, dict]:
    with open(curves_path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def get_resp_cols(row: dict) -> list[str]:
    return sorted(
        [c for c in row.keys() if c.startswith("resp_") and c.replace("resp_", "").isdigit()],
        key=lambda x: int(x.replace("resp_", "")),
    )


def find_disagreements(
    csv_path: Path,
    curves_path: Path,
    tag: str,
) -> list[dict]:
    """Find samples where Tevet label and E_fit disagree.

    Returns list of dicts with sample info, sorted by disagreement severity.
    """
    rows = load_csv_rows(csv_path)
    curves = load_mean_curves(curves_path)
    e_col = f"metric_icl_E_{tag}"

    samples: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in curves:
            continue
        a_k = curves[sid].get("a_k_curve")
        if a_k is None:
            continue

        e_disc = float(row.get(e_col, "nan"))
        if np.isnan(e_disc):
            continue

        k = np.arange(1, len(a_k) + 1, dtype=float)
        fit_params, fit_ok = fit_exponential(k, np.array(a_k))
        e_fit = fit_params["E_fit"] if fit_ok else e_disc

        label = float(row["label_value"])
        resp_cols = get_resp_cols(row)
        context = row.get("context", "")
        responses = [row[c] for c in resp_cols]

        samples.append({
            "sample_id": sid,
            "label": label,
            "e_discrete": e_disc,
            "e_fit": e_fit,
            "a_k": a_k,
            "context": context,
            "responses": responses,
            "fit_ok": fit_ok,
        })

    if not samples:
        return []

    # Normalize E_fit to [0, 1] for comparison with binary labels
    e_fits = [s["e_fit"] for s in samples]
    e_min, e_max = min(e_fits), max(e_fits)
    e_range = e_max - e_min if e_max > e_min else 1.0

    for s in samples:
        e_norm = (s["e_fit"] - e_min) / e_range
        # Disagreement: high label but low E_fit, or low label but high E_fit
        s["e_fit_normalized"] = e_norm
        s["disagreement"] = abs(s["label"] - e_norm)

    # Sort by disagreement (highest first)
    samples.sort(key=lambda s: s["disagreement"], reverse=True)
    return samples


def plot_per_token_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sample: dict,
    format_mode: FormatMode,
    output_path: Path,
    k_positions: list[int] | None = None,
) -> None:
    """Plot per-token logprobs for a sample across k positions.

    For each k position, shows the full context with masked/unmasked coloring.
    If k_positions is None, plots all positions.
    """
    context = sample["context"]
    responses = sample["responses"]
    n_responses = len(responses)
    models = _ensure_models(model)

    if k_positions is None:
        k_positions = list(range(n_responses))

    fig, axes = plt.subplots(
        len(k_positions), 1,
        figsize=(max(20, 60), 5 * len(k_positions)),
        squeeze=False,
    )

    for plot_idx, k_pos in enumerate(k_positions):
        # Build the context for position k_pos:
        # previous_responses = responses[:k_pos], current = responses[k_pos]
        previous = responses[:k_pos]
        current = responses[k_pos]
        prefix, target = format_conditioning_context(
            context, previous, current, format_mode=format_mode,
        )
        full_text = prefix + target

        # Get boundaries using the exact same function as the metric
        full_ids, boundaries = _find_response_boundaries(
            tokenizer, context, responses[:k_pos + 1], format_mode=format_mode,
        )

        # Forward pass
        input_ids = torch.tensor([full_ids], device=next(model.parameters()).device)
        log_probs = _forward_log_probs(models, input_ids)[0]  # (seq_len,) in bits

        n_tokens = len(full_ids)
        surprises = np.zeros(n_tokens)
        for t in range(1, n_tokens):
            surprises[t] = -log_probs[t - 1].item()

        # Build mask
        is_unmasked = np.zeros(n_tokens, dtype=bool)
        for start, end in boundaries:
            for t in range(start, end):
                is_unmasked[t] = True

        # Decode tokens
        token_strings = [tokenizer.decode([tid]) for tid in full_ids]

        ax = axes[plot_idx][0]
        x = np.arange(n_tokens)
        colors = ["#d32f2f" if u else "#9e9e9e" for u in is_unmasked]
        ax.bar(x, surprises, color=colors, width=0.8, alpha=0.8)

        # Response boundary lines
        for bk, (start, end) in enumerate(boundaries):
            if start > 0:
                ax.axvline(start - 0.5, color="blue", linewidth=1, linestyle="--", alpha=0.5)
            mid = (start + end) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 5,
                    f"r_{bk+1}", ha="center", va="top", fontsize=8,
                    color="blue", fontweight="bold")

        # Token labels
        display_tokens = []
        for tok in token_strings:
            t = tok.replace("\n", "\\n").replace("\t", "\\t")
            if not t.strip():
                t = repr(tok).strip("'")
            display_tokens.append(t)

        ax.set_xticks(x)
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=4,
                           fontfamily="monospace", ha="center")
        ax.set_xlim(-0.5, n_tokens - 0.5)
        ax.set_ylabel("Surprise (bits)")

        # Compute total bits for unmasked tokens in the current response boundary
        if k_pos < len(boundaries):
            cur_start, cur_end = boundaries[k_pos]
            cur_bits = sum(surprises[t] for t in range(cur_start, cur_end))
            ax.set_title(
                f"k={k_pos+1}: conditioning on {k_pos} previous responses, "
                f"measuring r_{k_pos+1} — total bits: {cur_bits:.1f}",
                fontsize=10,
            )
        else:
            ax.set_title(f"k={k_pos+1}", fontsize=10)

    # Main title
    label_str = "HIGH" if sample["label"] == 1.0 else "LOW"
    fig.suptitle(
        f"Disagreement Sample: {sample['sample_id']}\n"
        f"Label: {label_str} diversity | E_disc={sample['e_discrete']:.1f} | "
        f"E_fit={sample['e_fit']:.1f} | a_k={[f'{v:.1f}' for v in sample['a_k']]}",
        fontsize=12, fontweight="bold",
    )

    legend_elements = [
        Patch(facecolor="#d32f2f", alpha=0.8, label="Unmasked (completion)"),
        Patch(facecolor="#9e9e9e", alpha=0.8, label="Masked (context/label)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def write_report(
    all_disagreements: dict[str, list[dict]],
    output_path: Path,
    top_n: int,
) -> None:
    """Write a text report of disagreement samples."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("DISAGREEMENT ANALYSIS — McDiv_nuggets")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Samples where Tevet labels and E_fit disagree most strongly.")
    lines.append("Sorted by |label - normalized_E_fit| (highest disagreement first).")
    lines.append("")

    for task_name, samples in all_disagreements.items():
        lines.append(f"\n{'━' * 70}")
        lines.append(f"  TASK: {task_name}")
        lines.append(f"  Total samples: {len(samples)}")

        # Quick stats
        labels = [s["label"] for s in samples]
        e_fits = [s["e_fit"] for s in samples]
        rho, pval = spearmanr(labels, e_fits)
        lines.append(f"  Spearman ρ(label, E_fit): {rho:.3f} (p={pval:.4f})")
        lines.append(f"{'━' * 70}")

        for i, s in enumerate(samples[:top_n]):
            label_str = "HIGH" if s["label"] == 1.0 else "LOW"
            lines.append(f"\n  --- #{i+1} | {s['sample_id']} | Label: {label_str} | "
                         f"E_disc={s['e_discrete']:.2f} | E_fit={s['e_fit']:.2f} | "
                         f"disagreement={s['disagreement']:.3f} ---")
            lines.append(f"  Context: {s['context']}")
            lines.append(f"  a_k curve: {[f'{v:.1f}' for v in s['a_k']]}")
            for j, resp in enumerate(s["responses"]):
                lines.append(f"    resp_{j}: {resp}")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"\nSaved report: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disagreement analysis for Tevet evaluation")
    parser.add_argument("--run-tag", type=str, default="qwen25_completion")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top disagreement samples to visualize per task")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--format-mode", type=str, default="completion",
                        choices=["instruct", "completion"])
    parser.add_argument("--k-positions", type=str, default=None,
                        help="Comma-separated k positions to plot (default: 1 and last)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate text report, skip per-token plots (no model needed)")
    args = parser.parse_args()

    data_dir = RESULTS_BASE / args.run_tag / "McDiv_nuggets"
    if not data_dir.exists():
        print(f"No data found at {data_dir}")
        sys.exit(1)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Find disagreements from existing data
    print("=" * 60)
    print("Phase 1: Finding disagreement samples")
    print("=" * 60)

    all_disagreements: dict[str, list[dict]] = {}
    for csv_path in sorted(data_dir.glob("*_with_hds_*.csv")):
        curves_path = csv_path.with_suffix(f".icl_mean_curves.{args.run_tag}.json")
        if not curves_path.exists():
            print(f"  Skipping {csv_path.name} — no mean curves")
            continue

        task = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]
        print(f"\n  {task}:")
        samples = find_disagreements(csv_path, curves_path, args.run_tag)
        all_disagreements[task] = samples

        # Print top disagreements
        for i, s in enumerate(samples[:args.top_n]):
            label_str = "HIGH" if s["label"] == 1.0 else "LOW"
            print(f"    #{i+1}: {s['sample_id']} | {label_str} | "
                  f"E_fit={s['e_fit']:.1f} | disagree={s['disagreement']:.3f}")

    # Write text report
    write_report(all_disagreements, output_dir / "disagreement_report.txt", args.top_n)

    if args.report_only:
        print("\n--report-only: skipping per-token visualization")
        return

    # Phase 2: Load model and generate per-token visualizations
    print("\n" + "=" * 60)
    print("Phase 2: Loading model for per-token logprobs")
    print("=" * 60)

    print(f"  Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    # Parse k_positions
    if args.k_positions:
        k_positions = [int(x) - 1 for x in args.k_positions.split(",")]
    else:
        # Default: first and last positions
        k_positions = None  # will be set per-sample

    print("\n" + "=" * 60)
    print("Phase 3: Generating per-token logprob visualizations")
    print("=" * 60)

    for task_name, samples in all_disagreements.items():
        print(f"\n  Task: {task_name}")
        for i, sample in enumerate(samples[:args.top_n]):
            n_resp = len(sample["responses"])
            kp = k_positions if k_positions is not None else [0, n_resp - 1]
            safe_id = sample["sample_id"].replace("::", "_").replace("/", "_")
            out_path = output_dir / f"disagreement_{task_name}_{i}_{safe_id}.png"
            print(f"    #{i+1}: {sample['sample_id']} (k={[p+1 for p in kp]})")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_per_token_logprobs(
                    model, tokenizer, sample, args.format_mode, out_path, kp,
                )

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
