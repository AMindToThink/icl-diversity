"""Test whether models learn from random word salad sharing a fixed vocabulary.

Both Qwen2.5-3B and GPT-2 are tested. The "noise" responses are random
permutations of 20 fixed nonsense words — this is actually a single mode,
so the model SHOULD learn the distribution.
"""

import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_diversity.core import compute_cross_entropy

torch.set_grad_enabled(False)

prompt = "Write a short piece about rain."
labels = "ABCDEFGHIJ"

words = [
    "xkq", "plm", "zvw", "nrt", "bjf", "ghd", "wcy", "mkp", "slv", "qxr",
    "dtn", "fhj", "bwm", "kcg", "lpz", "vrx", "nqy", "jtf", "wsd", "hgm",
]

rng = random.Random(123)
noise_responses: list[str] = []
for _ in range(8):
    shuffled = words.copy()
    rng.shuffle(shuffled)
    noise_responses.append(" ".join(shuffled[:15]))

models_to_test = [
    ("Qwen/Qwen2.5-3B", torch.float16, "cuda:1"),
    ("gpt2", torch.float32, "cpu"),
]

for model_name, dtype, device in models_to_test:
    print(f"=== {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device
    )
    model.eval()

    print(" k   total_bits  bits/byte  byte_count")
    for k in range(min(6, len(noise_responses))):
        parts = [prompt]
        for i in range(k):
            parts.append(f"\n\nResponse {labels[i]}: {noise_responses[i]}")
        parts.append(f"\n\nResponse {labels[k]}: ")
        prefix = "".join(parts)

        bits, bc = compute_cross_entropy(
            model, tokenizer, noise_responses[k], prefix=prefix
        )
        print(f"{k+1:>2}  {bits:>10.1f}  {bits/bc:>9.4f}  {bc:>10}")
    print()
