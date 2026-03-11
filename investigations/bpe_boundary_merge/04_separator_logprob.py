"""Measure the log-prob of the separator component in merged boundary tokens.

For tokenizers that merge '.' + '\n\n' into '.\n\n', the single-pass approach
scores P('.\n\n' | context) instead of P('.' | context).  By the chain rule:

    -log P('.\n\n') = -log P('.') - log P('\n\n' | '.')

This script estimates the second term — the "separator overhead" — to show
it is near zero (the model strongly expects '\n\n' after a period in this
format).

Requires Qwen2.5-3B on GPU.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.mode_count_scenarios import (
    PROMPT,
    generate_mode_count_responses,
)

N_DRAWS = 10
M = 10
N_RESPONSES = 5  # Only need a few to measure the effect


def main() -> None:
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B", torch_dtype=torch.float16
        ).to("cuda:1")
        model.eval()
    except Exception as e:
        print(f"Need Qwen2.5-3B on GPU: {e}", file=sys.stderr)
        sys.exit(1)

    print("Measuring separator overhead in merged boundary tokens\n")

    # Find the token IDs for '.' and '.\n\n'
    dot_ids = tok.encode(".", add_special_tokens=False)
    dot_nn_ids = tok.encode(".\n\n", add_special_tokens=False)
    print(f"Token '.'   : {dot_ids} = {[tok.decode([t]) for t in dot_ids]}")
    print(f"Token '.\\n\\n': {dot_nn_ids} = {[tok.decode([t]) for t in dot_nn_ids]}")
    print()

    overheads = []

    for draw in range(N_DRAWS):
        seed = 42 + draw
        responses, _ = generate_mode_count_responses(M, n=N_RESPONSES, seed=seed)

        # Build context up to first response (without separator)
        context_trunc = PROMPT + "\n\nResponse A: " + responses[0]
        # Build context with separator
        context_with_sep = context_trunc + "\n\n"

        ids_trunc = tok.encode(context_trunc, add_special_tokens=False)
        ids_with_sep = tok.encode(context_with_sep, add_special_tokens=False)

        # Get logits at the position just before the boundary
        with torch.no_grad():
            input_ids = torch.tensor([ids_trunc], device="cuda:1")
            logits_trunc = model(input_ids, use_cache=False).logits[0, -1]
            log_probs_trunc = torch.log_softmax(logits_trunc.float(), dim=-1)

        # P('.') vs P('.\n\n') at the position before
        # Actually we want: at the position OF the '.'/.'.\n\n' token,
        # what's the log-prob difference?
        # In the truncated context, last token is '.' (standalone)
        # In full context, last token before separator is '.\n\n' (merged)

        # Get log-prob of the merged token vs standalone token
        if len(dot_ids) == 1 and len(dot_nn_ids) == 1:
            lp_dot = log_probs_trunc[dot_ids[0]].item()
            lp_dot_nn = log_probs_trunc[dot_nn_ids[0]].item()
            overhead = (-lp_dot_nn - (-lp_dot)) / torch.log(torch.tensor(2.0)).item()
            overheads.append(overhead)
            if draw < 3:
                print(
                    f"  Draw {draw}: -log2 P('.')={-lp_dot / 0.6931:.2f} bits, "
                    f"-log2 P('.\\n\\n')={-lp_dot_nn / 0.6931:.2f} bits, "
                    f"overhead={overhead:.3f} bits"
                )

    if overheads:
        import numpy as np

        arr = np.array(overheads)
        print(f"\nSeparator overhead across {len(overheads)} draws:")
        print(f"  Mean: {arr.mean():.3f} bits")
        print(f"  Std:  {arr.std():.3f} bits")
        print(f"  Min:  {arr.min():.3f} bits")
        print(f"  Max:  {arr.max():.3f} bits")
        print(
            f"\nConclusion: the separator component adds ~{arr.mean():.2f} bits "
            f"per boundary token. With {N_RESPONSES} boundaries per curve, "
            f"total distortion is ~{arr.mean() * N_RESPONSES:.1f} bits."
        )


if __name__ == "__main__":
    main()
