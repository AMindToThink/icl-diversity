"""Test same-mode learning vs noise context with Qwen2.5-3B.

Checks that:
- Same-style responses (philosophy → philosophy) show genuine surprise reduction.
- Irrelevant context (random numbers → philosophy) shows no surprise reduction.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.core import compute_cross_entropy

torch.set_grad_enabled(False)

model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="cuda:1"
)
model.eval()

prompt = "Write a short piece about rain."

r1 = (
    "Rain is perhaps the most democratic force in nature — it falls on all "
    "without distinction, caring nothing for borders, class, or creed. Each "
    "drop, a tiny ambassador of the sky, carries a message older than "
    "civilization itself."
)
r2 = (
    "There is something profoundly humbling about rain. It arrives uninvited, "
    "transforms the world in minutes, and departs as quietly as it came. The "
    "earth drinks, the rivers swell, and we, mere spectators, marvel at the "
    "casual power of water falling from the sky."
)

# Same-mode test
bits_r1_alone, bc1 = compute_cross_entropy(
    model, tokenizer, r1, prefix=prompt + "\n\nResponse A: "
)
bits_r2_after_r1, bc2 = compute_cross_entropy(
    model,
    tokenizer,
    r2,
    prefix=prompt + "\n\nResponse A: " + r1 + "\n\nResponse B: ",
)

bits_r2_alone, bc3 = compute_cross_entropy(
    model, tokenizer, r2, prefix=prompt + "\n\nResponse A: "
)
bits_r1_after_r2, bc4 = compute_cross_entropy(
    model,
    tokenizer,
    r1,
    prefix=prompt + "\n\nResponse A: " + r2 + "\n\nResponse B: ",
)

print("Same-style (both philosophy):")
print(f"  r1 alone: {bits_r1_alone:.1f} bits ({bits_r1_alone/bc1:.3f} b/B)")
print(f"  r2 after r1: {bits_r2_after_r1:.1f} bits ({bits_r2_after_r1/bc2:.3f} b/B)")
print(
    f"  drop: {bits_r1_alone - bits_r2_after_r1:+.1f} bits "
    f"({(bits_r1_alone/bc1 - bits_r2_after_r1/bc2):+.3f} b/B)"
)
print()
print(f"  r2 alone: {bits_r2_alone:.1f} bits ({bits_r2_alone/bc3:.3f} b/B)")
print(f"  r1 after r2: {bits_r1_after_r2:.1f} bits ({bits_r1_after_r2/bc4:.3f} b/B)")
print(
    f"  drop: {bits_r2_alone - bits_r1_after_r2:+.1f} bits "
    f"({(bits_r2_alone/bc3 - bits_r1_after_r2/bc4):+.3f} b/B)"
)

# Noise context test
r_unrelated = (
    "7 8 12 99 3 42 0 17 65 28 11 4 83 56 71 9 33 20 47 62 15 88 6 39 74 "
    "51 2 96 30 18 67 43 5 81"
)
bits_r2_after_noise, bc5 = compute_cross_entropy(
    model,
    tokenizer,
    r2,
    prefix=prompt + "\n\nResponse A: " + r_unrelated + "\n\nResponse B: ",
)
print()
print("After random numbers:")
print(
    f"  r2 after noise: {bits_r2_after_noise:.1f} bits "
    f"({bits_r2_after_noise/bc5:.3f} b/B)"
)
print(f"  drop vs alone: {bits_r2_alone - bits_r2_after_noise:+.1f} bits")
