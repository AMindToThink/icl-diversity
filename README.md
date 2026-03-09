# ICL Diversity Metric

Implementation of the **in-context learning (ICL) diversity metric** from
[the paper](paper/in_context_diversity_metric.pdf). The metric measures the
diversity of LLM-generated responses by computing progressive conditional
surprise under a base model θ: as θ sees more responses in-context, its
surprise decreases proportionally to how much learnable structure (i.e.,
distinct modes) exists across the response set.

## Key Concepts

The primary a_k curve is in **total bits**. Per-byte normalized quantities
(E_rate, C, D_rate) provide tokenizer-agnostic comparisons.

| Symbol | Name | Meaning |
|--------|------|---------|
| a_k | Progressive conditional surprise | Cross-entropy of the k-th response in **total bits**, conditioned on the previous k−1 responses |
| E | Excess entropy (total bits) | Sum of (a_k − a_n) — total learnable structure in bits |
| E_rate | Excess entropy rate (bits/byte) | Per-byte normalized excess entropy, averaged across permutations |
| C | Coherence (per-byte) | 2^{−mean(h)} where h is per-byte cross-entropy — filters out incoherent text |
| D | Diversity score (bits) | C × E — high only when responses are both coherent *and* diverse |
| D_rate | Diversity score rate (bits/byte) | C × E_rate — per-byte variant |
| σ | Coherence spread | Std. dev. of per-response surprise — flags mixed-quality response sets |

## Repository Structure

```
icl_diversity/
├── src/icl_diversity/
│   ├── core.py            # Core metric computation (a_k curves, E, C, D, etc.)
│   ├── cli.py             # CLI for computing metrics on a responses.jsonl file
│   ├── scenarios.py       # Synthetic scenario data for validation experiments
│   ├── __init__.py        # Public API re-exports
│   └── __main__.py        # python -m icl_diversity entry point
├── scripts/
│   ├── run_scenarios.py   # Compute metrics for all 5 validation scenarios → JSON
│   ├── plot_ak_curves.py  # Generate a_k curve plots from saved JSON → PNGs
│   └── test_hypotheses.py # Run cross-model hypothesis tests (H1-H13, Q1-Q12)
├── tests/
│   ├── test_icl_diversity.py            # Unit tests (mock model, pure math)
│   ├── test_icl_diversity_scenarios.py  # GPT-2 integration tests (5 scenarios)
│   └── test_single_pass.py             # Single-pass vs multi-pass equivalence
├── results/
│   ├── scenario_metrics_v2_3perm.json         # GPT-2, 3 permutations
│   ├── scenario_metrics_v2_100perm.json       # GPT-2, 100 permutations
│   ├── scenario_metrics_v2_qwen_3perm.json    # Qwen2.5-32B, 3 permutations
│   └── scenario_metrics_v2_qwen_100perm.json  # Qwen2.5-32B, 100 permutations
├── figures/
│   ├── comparison_v2/     # Side-by-side GPT-2 vs Qwen plots (100 perms)
│   └── ...                # Per-model plots
├── paper/                    # Reference paper (PDF + LaTeX source)
├── EXPERIMENT_REPORT.md      # V1 validation report (historical, uses old definitions)
├── EXPERIMENT_REPORT_V2.md   # V2 validation report (current definitions)
└── pyproject.toml
```

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

## Running Experiments

### Validation scenarios

Five synthetic scenarios test the metric's behavior against the paper's
theoretical predictions (pure noise, multiple incoherent modes, multiple
coherent modes, one coherent mode, mixed coherent+incoherent).

```bash
# GPT-2 on GPU (~2 min with 100 permutations)
uv run python scripts/run_scenarios.py --n-permutations 100 \
    --output results/scenario_metrics_v2_100perm.json

# Qwen2.5-32B on multi-GPU (~80 min with 100 permutations)
uv run python scripts/run_scenarios.py \
    --base-model Qwen/Qwen2.5-32B --device auto --torch-dtype float16 \
    --n-permutations 100 --output results/scenario_metrics_v2_qwen_100perm.json

# Generate comparison plots
uv run python scripts/plot_ak_curves.py \
    --input results/scenario_metrics_v2_100perm.json \
           results/scenario_metrics_v2_qwen_100perm.json \
    --output-dir figures/comparison_v2

# Run cross-model hypothesis tests
uv run python scripts/test_hypotheses.py \
    --gpt2 results/scenario_metrics_v2_100perm.json \
    --qwen results/scenario_metrics_v2_qwen_100perm.json
```

### Run on your own responses

The CLI reads a `responses.jsonl` file (one JSON object per line with fields:
`prompt`, `prompt_idx`, `response_idx`, `scale`, `response`) and computes the
ICL diversity metric for each (scale, prompt) group.

```bash
uv run calculate-icl-diversity \
    --input path/to/responses.jsonl \
    --base-model gpt2 \
    --n-permutations 50 \
    --output results/my_metrics.json
```

Or equivalently:

```bash
uv run python -m icl_diversity \
    --input path/to/responses.jsonl \
    --base-model meta-llama/Llama-3.1-8B \
    --n-permutations 50 \
    --device cuda
```

> **Note:** We recommend `n_permutations >= 50` for reliable results.
> With only 3 permutations, E_rate can have the wrong sign for high-variance
> response sets (see EXPERIMENT_REPORT_V2.md §5).

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from icl_diversity import compute_icl_diversity_metrics

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

metrics = compute_icl_diversity_metrics(
    model, tokenizer,
    prompt="Tell me a story.",
    responses=["Story A...", "Story B...", "Story C..."],
    n_permutations=50,
    seed=42,
)

print(f"D={metrics['diversity_score_D']:.4f}")
print(f"E_rate={metrics['excess_entropy_E_rate']:.4f}")
print(f"E={metrics['excess_entropy_E']:.1f} bits")
print(f"C={metrics['coherence_C']:.4f}")
```

The returned dict includes:
- `a_k_curve` (total bits), `a_k_curve_per_byte`, `a_k_byte_counts`
- `unconditional_surprises` (per-byte), `unconditional_total_bits`
- `excess_entropy_E` (total bits), `excess_entropy_E_rate` (bits/byte)
- `coherence_C` (per-byte)
- `diversity_score_D` (C × E), `diversity_score_D_rate` (C × E_rate)
- `coherence_spread_sigma`, `D_plus`, `D_minus`, `C_plus`, `C_minus`
- `mean_byte_length`, `is_monotone`
- `per_permutation_a_k_curves`, `per_permutation_byte_counts`, `permutation_orders`

## Tests

```bash
# Unit tests (fast, mock model, no GPU needed)
uv run pytest tests/test_icl_diversity.py -v

# GPT-2 integration tests (~100s on CPU, downloads gpt2 on first run)
uv run pytest tests/test_icl_diversity_scenarios.py -v -s

# Single-pass vs multi-pass equivalence (requires GPT-2)
uv run pytest tests/test_single_pass.py -v -s
```
