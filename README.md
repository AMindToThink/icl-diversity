# ICL Diversity Metric

Implementation of the **in-context learning (ICL) diversity metric** from
[the paper](paper/in_context_diversity_metric.pdf). The metric measures the
diversity of LLM-generated responses by computing progressive conditional
surprise under a base model θ: as θ sees more responses in-context, its
surprise decreases proportionally to how much learnable structure (i.e.,
distinct modes) exists across the response set.

## Key Concepts

| Symbol | Name | Meaning |
|--------|------|---------|
| a_k | Progressive conditional surprise | Per-byte cross-entropy of the k-th response, conditioned on the previous k−1 |
| E | Excess entropy | Total area above the asymptote in the a_k curve — measures learnable inter-response structure |
| C | Coherence | Geometric-mean probability of each response under θ — filters out incoherent text |
| D | Diversity score | C × E — high only when responses are both coherent *and* diverse |
| σ | Coherence spread | Std. dev. of per-response surprise — flags mixed-quality response sets |
| m_eff | Effective mode count | 2^(B̄ × E) — interpretive estimate of the number of distinct modes |

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
│   └── plot_ak_curves.py  # Generate a_k curve plots from saved JSON → PNGs
├── tests/
│   ├── test_icl_diversity.py            # Unit tests (mock model, pure math)
│   └── test_icl_diversity_scenarios.py  # GPT-2 integration tests (5 scenarios)
├── results/
│   └── scenario_metrics.json  # Saved metrics from GPT-2 validation run
├── figures/
│   ├── ak_curves_all_scenarios.png   # Combined overview plot
│   └── ak_curve_*.png               # Per-scenario plots
├── paper/                    # Reference paper (PDF + LaTeX source)
├── EXPERIMENT_REPORT.md      # Full validation report with results and discussion
└── pyproject.toml
```

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

## Running Experiments

### Validation scenarios (GPT-2)

Five synthetic scenarios test the metric's behavior against the paper's
theoretical predictions (pure noise, multiple incoherent modes, multiple
coherent modes, one coherent mode, mixed coherent+incoherent).

```bash
# Compute metrics and save to results/scenario_metrics.json (~100s on CPU)
uv run python scripts/run_scenarios.py

# Generate plots from the saved JSON → figures/
uv run python scripts/plot_ak_curves.py
```

### Run on your own responses

The CLI reads a `responses.jsonl` file (one JSON object per line with fields:
`prompt`, `prompt_idx`, `response_idx`, `scale`, `response`) and computes the
ICL diversity metric for each (scale, prompt) group.

```bash
uv run calculate-icl-diversity \
    --input path/to/responses.jsonl \
    --base-model gpt2 \
    --n-permutations 3 \
    --output results/my_metrics.json
```

Or equivalently:

```bash
uv run python -m icl_diversity \
    --input path/to/responses.jsonl \
    --base-model meta-llama/Llama-3.1-8B \
    --n-permutations 3 \
    --device cuda
```

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
    n_permutations=3,
    seed=42,
)

print(f"D={metrics['diversity_score_D']:.4f}")
print(f"E={metrics['excess_entropy_E']:.4f}")
print(f"C={metrics['coherence_C']:.4f}")
```

The returned dict includes:
- `a_k_curve`, `unconditional_surprises`
- `excess_entropy_E`, `coherence_C`, `coherence_spread_sigma`
- `diversity_score_D`, `D_plus`, `D_minus`, `C_plus`, `C_minus`
- `effective_mode_count`, `mean_byte_length`, `is_monotone`
- `per_permutation_a_k_curves`, `permutation_orders` (when `n_permutations > 1`)

## Tests

```bash
# Unit tests (fast, mock model, no GPU needed)
uv run pytest tests/test_icl_diversity.py -v

# GPT-2 integration tests (~100s on CPU, downloads gpt2 on first run)
uv run pytest tests/test_icl_diversity_scenarios.py -v -s
```
