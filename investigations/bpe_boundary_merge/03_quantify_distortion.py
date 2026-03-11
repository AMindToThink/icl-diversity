"""Quantify the single-pass vs multi-pass distortion caused by BPE merges.

Runs both single-pass and multi-pass surprise curves on the same responses
and compares per-position and aggregate differences.  This measures the actual
impact of scoring merged tokens (e.g. '.\n\n') instead of standalone tokens
(e.g. '.') at response boundaries.

Requires a GPU with Qwen2.5-3B loaded.  Falls back to GPT-2 on CPU.
"""

import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity import (
    compute_progressive_surprise_curve,
    compute_progressive_surprise_curve_single_pass,
)
from icl_diversity.mode_count_scenarios import (
    PROMPT,
    generate_mode_count_responses,
)

# Configuration
N_DRAWS = 20
M = 10
N_RESPONSES = 20
SEED_START = 42


def main() -> None:
    # Try Qwen first, fall back to GPT-2
    device = "cpu"
    model_id = "gpt2"
    for try_model, try_device in [("Qwen/Qwen2.5-3B", "cuda:1"), ("gpt2", "cpu")]:
        try:
            tok = AutoTokenizer.from_pretrained(try_model)
            model = AutoModelForCausalLM.from_pretrained(
                try_model,
                torch_dtype=torch.float16 if "cuda" in try_device else None,
            )
            if try_device != "cpu":
                model = model.to(try_device)
            model.eval()
            model_id = try_model
            device = try_device
            break
        except Exception as e:
            print(f"Could not load {try_model}: {e}", file=sys.stderr)

    print(f"Model: {model_id} on {device}")
    print(f"Draws: {N_DRAWS}, m={M}, n={N_RESPONSES}\n")

    all_sp_curves = []
    all_mp_curves = []

    for draw in range(N_DRAWS):
        seed = SEED_START + draw
        responses, _ = generate_mode_count_responses(M, n=N_RESPONSES, seed=seed)

        sp_curve, sp_bc = compute_progressive_surprise_curve_single_pass(
            model, tok, PROMPT, responses
        )
        mp_curve, mp_bc = compute_progressive_surprise_curve(
            model, tok, PROMPT, responses
        )

        all_sp_curves.append(sp_curve)
        all_mp_curves.append(mp_curve)

        if draw < 3:
            diff = np.array(sp_curve) - np.array(mp_curve)
            print(f"  Draw {draw}: max|diff|={np.abs(diff).max():.2f} bits, "
                  f"mean diff={diff.mean():.2f} bits")

    sp = np.array(all_sp_curves)
    mp = np.array(all_mp_curves)
    diff = sp - mp

    print(f"\n{'='*60}")
    print(f"Aggregate over {N_DRAWS} draws, {N_RESPONSES} positions each")
    print(f"{'='*60}")

    # Per-position stats
    mean_diff = diff.mean(axis=0)
    print(f"\nMean SP-MP difference by position (bits):")
    for k in range(N_RESPONSES):
        print(f"  k={k+1:2d}: {mean_diff[k]:+.3f}")

    # Summary stats
    mean_sp = sp.mean(axis=0)
    mean_mp = mp.mean(axis=0)
    total_sp_decline = mean_sp[0] - mean_sp[-1]
    total_mp_decline = mean_mp[0] - mean_mp[-1]
    total_diff = np.abs(mean_diff).sum()

    print(f"\nTotal absolute distortion: {total_diff:.1f} bits")
    print(f"SP curve decline (a_1 - a_n): {total_sp_decline:.1f} bits")
    print(f"MP curve decline (a_1 - a_n): {total_mp_decline:.1f} bits")
    print(f"Distortion as % of MP decline: {total_diff / total_mp_decline * 100:.1f}%")

    # Per-byte impact
    mean_bytes = np.mean([len(r.encode("utf-8")) for r in responses])
    print(f"\nMean response length: {mean_bytes:.0f} bytes")
    print(f"Mean distortion per position: {np.abs(mean_diff).mean():.3f} bits "
          f"= {np.abs(mean_diff).mean() / mean_bytes:.4f} bits/byte")


if __name__ == "__main__":
    main()
