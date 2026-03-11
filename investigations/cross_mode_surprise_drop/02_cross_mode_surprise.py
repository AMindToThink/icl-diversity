"""Directly measure cross-mode surprise reduction with Qwen2.5-3B.

Tests whether seeing a lab-notebook response reduces surprise for a
philosophy response (and vice versa).
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

r_lab = (
    "Date: 2024-09-18. Obs: Collected 47ml of rainwater in the standard gauge "
    "over 6 hours. pH measured at 5.3, consistent with unpolluted precipitation. "
    "Cloud cover was stratus, base altitude est 800m. Temperature dropped 4C "
    "during the event. Wind was calm. Humidity peaked at 96%. Note: unusual "
    "iron-rich sediment in sample — check proximity to construction site."
)
r_phil = (
    "Rain is perhaps the most democratic force in nature — it falls on all "
    "without distinction, caring nothing for borders, class, or creed. Each "
    "drop, a tiny ambassador of the sky, carries a message older than "
    "civilization itself. To stand in the rain is to participate in an ancient "
    "dialogue between earth and atmosphere."
)

# a_1: surprise of r_lab with no context
bits_lab_alone, bc1 = compute_cross_entropy(
    model, tokenizer, r_lab, prefix=prompt + "\n\nResponse A: "
)
print(f"a_1 (lab alone): {bits_lab_alone:.1f} bits ({bits_lab_alone/bc1:.3f} bits/byte)")

# a_2: surprise of r_phil after seeing r_lab
prefix_with_lab = (
    prompt + "\n\nResponse A: " + r_lab + "\n\nResponse B: "
)
bits_phil_after_lab, bc2 = compute_cross_entropy(
    model, tokenizer, r_phil, prefix=prefix_with_lab
)
print(
    f"a_2 (phil after lab): {bits_phil_after_lab:.1f} bits "
    f"({bits_phil_after_lab/bc2:.3f} bits/byte)"
)

# Reverse: a_1 for phil, a_2 for lab after phil
bits_phil_alone, bc3 = compute_cross_entropy(
    model, tokenizer, r_phil, prefix=prompt + "\n\nResponse A: "
)
print(
    f"a_1 (phil alone): {bits_phil_alone:.1f} bits "
    f"({bits_phil_alone/bc3:.3f} bits/byte)"
)

prefix_with_phil = (
    prompt + "\n\nResponse A: " + r_phil + "\n\nResponse B: "
)
bits_lab_after_phil, bc4 = compute_cross_entropy(
    model, tokenizer, r_lab, prefix=prefix_with_phil
)
print(
    f"a_2 (lab after phil): {bits_lab_after_phil:.1f} bits "
    f"({bits_lab_after_phil/bc4:.3f} bits/byte)"
)

print()
print(f"Drop lab->phil: {bits_lab_alone - bits_phil_after_lab:.1f} bits")
print(f"Drop phil->lab: {bits_phil_alone - bits_lab_after_phil:.1f} bits")
print(
    f"Mean drop (both orderings): "
    f"{((bits_lab_alone - bits_phil_after_lab) + (bits_phil_alone - bits_lab_after_phil)) / 2:.1f} bits"
)
