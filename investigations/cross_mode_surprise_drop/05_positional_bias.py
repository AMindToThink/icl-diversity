"""Test for positional bias in Qwen2.5-3B.

Measures surprise for the SAME response placed at positions 1-8, with
completely unrelated filler responses (math, cooking, history, etc.) as
context. If surprise decreases with position, it's a positional bias.

Result: surprise INCREASES slightly with unrelated context. No positional
bias — the effect in the real experiment is content-dependent.
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
labels = "ABCDEFGHIJKLMNOPQRST"

# Target response — always the same text
target = (
    "The old tin roof sang its familiar song as the first drops arrived, "
    "tentative and light, before the sky opened and the rain fell in earnest, "
    "drumming a rhythm that echoed through the empty house."
)

# Filler responses about COMPLETELY different topics
fillers = [
    (
        "The circumference of a circle is calculated using the formula C = 2 pi r, "
        "where r is the radius. This fundamental relationship was first established "
        "by ancient Greek mathematicians."
    ),
    (
        "To make a perfect omelette, crack three eggs into a bowl and whisk "
        "vigorously. Heat butter in a non-stick pan over medium heat until it "
        "foams but does not brown."
    ),
    (
        "The Treaty of Westphalia in 1648 ended the Thirty Years War and "
        "established the modern concept of state sovereignty. It marked a turning "
        "point in European diplomatic history."
    ),
    (
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen "
        "using sunlight. The process occurs primarily in the chloroplasts of plant "
        "cells."
    ),
    (
        "The stock market experienced significant volatility in Q3, with the S&P "
        "500 declining 4.2 percent before recovering in late September on "
        "stronger-than-expected employment data."
    ),
    (
        "A binary search tree maintains the invariant that all left descendants "
        "are less than the node and all right descendants are greater. Insertion "
        "runs in O(log n) average time."
    ),
    (
        "The migration patterns of Arctic terns span over 44,000 miles annually, "
        "from Arctic breeding grounds to Antarctic feeding areas, making it the "
        "longest migration of any animal."
    ),
    (
        "Reinforced concrete combines the compressive strength of concrete with "
        "the tensile strength of steel rebar. The coefficient of thermal expansion "
        "is similar for both materials."
    ),
]

print("Same response at different positions, with unrelated filler context:")
print("pos  total_bits  bits/byte  context_tokens")
for pos in range(min(8, len(fillers) + 1)):
    parts = [prompt]
    for i in range(pos):
        parts.append(f"\n\nResponse {labels[i]}: {fillers[i]}")
    parts.append(f"\n\nResponse {labels[pos]}: ")
    prefix = "".join(parts)

    n_prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    bits, bc = compute_cross_entropy(model, tokenizer, target, prefix=prefix)
    print(f"  {pos+1}  {bits:>10.1f}  {bits/bc:>9.4f}  {n_prefix_tokens:>14}")
