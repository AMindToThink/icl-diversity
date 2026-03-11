"""Verify single-pass vs multi-pass equivalence for Qwen2.5-3B.

Rules out boundary detection bugs in the single-pass implementation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.core import (
    compute_progressive_surprise_curve,
    compute_progressive_surprise_curve_single_pass,
)
from icl_diversity.mode_count_scenarios import generate_mode_count_responses

torch.set_grad_enabled(False)

model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="cuda:1"
)
model.eval()

prompt = "Write a short piece about rain."
responses, mode_names = generate_mode_count_responses(m=10, n=10, seed=42)

print("Responses (first 50 chars each):")
for i, (r, mn) in enumerate(zip(responses, mode_names)):
    print(f"  {i}: [{mn}] {r[:50]}...")

print()
print("Single-pass vs multi-pass a_k curves:")
sp_curve, sp_bytes = compute_progressive_surprise_curve_single_pass(
    model, tokenizer, prompt, responses
)
mp_curve, mp_bytes = compute_progressive_surprise_curve(
    model, tokenizer, prompt, responses
)

print("k   single_pass   multi_pass   diff       sp_bytes  mp_bytes")
for k in range(len(sp_curve)):
    diff = sp_curve[k] - mp_curve[k]
    print(
        f"{k+1:>2}  {sp_curve[k]:>11.3f}  {mp_curve[k]:>10.3f}  "
        f"{diff:>+9.3f}  {sp_bytes[k]:>8}  {mp_bytes[k]:>8}"
    )
