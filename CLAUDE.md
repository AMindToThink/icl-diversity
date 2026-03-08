# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Implementation of the ICL (in-context learning) diversity metric from `paper/in_context_diversity_metric.pdf`. The metric measures LLM output diversity by computing progressive conditional surprise under a base model θ — as θ sees more responses in-context, surprise decreases proportionally to how many distinct modes exist.

The primary a_k curve is in **total bits**. Per-byte normalized quantities (E_rate, C, D) provide tokenizer-agnostic comparisons. Total-bits quantities (E, C_total, D_total) capture absolute information content.

## Commands

```bash
# Setup
uv sync --all-extras

# Unit tests (fast, mock model)
uv run pytest tests/test_icl_diversity.py -v

# Single test
uv run pytest tests/test_icl_diversity.py::TestExcessEntropy::test_zero_for_constant_curve -v

# GPT-2 integration tests (~100s on CPU)
uv run pytest tests/test_icl_diversity_scenarios.py -v -s

# Single-pass equivalence tests (requires GPT-2)
uv run pytest tests/test_single_pass.py -v -s

# Compute validation scenario metrics → results/scenario_metrics.json
uv run python scripts/run_scenarios.py

# Compute with a different model (e.g., Qwen2.5-32B on multi-GPU)
uv run python scripts/run_scenarios.py --base-model Qwen/Qwen2.5-32B --device auto --torch-dtype float16 --output results/scenario_metrics_qwen2.5-32b.json

# Generate a_k curve plots from saved JSON → figures/
uv run python scripts/plot_ak_curves.py

# Generate comparison plots (multiple models)
uv run python scripts/plot_ak_curves.py --input results/scenario_metrics.json results/scenario_metrics_qwen2.5-32b.json --output-dir figures/comparison

# Run cross-model hypothesis tests
uv run python scripts/test_hypotheses.py

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI for custom response files
uv run calculate-icl-diversity --input responses.jsonl --base-model gpt2 --n-permutations 3
```

## Architecture

### Core computation pipeline (`src/icl_diversity/core.py`)

The metric flows through these stages, all in one file:

1. **`compute_cross_entropy(model, tokenizer, text, prefix)`** — Tokenizes prefix+text, runs a forward pass, extracts log-probs for just the text tokens. Returns `(total_bits, byte_count)`. This is the atomic building block. `compute_per_byte_cross_entropy` is a thin wrapper that divides by byte count.

2. **`compute_progressive_surprise_curve_single_pass(model, tokenizer, prompt, responses)`** — Single forward pass over the full concatenated context, extracting per-response log-probs by token boundary detection. Returns `(curve_total_bits, byte_counts)`. This is the default used by `compute_icl_diversity_metrics`. The old multi-pass version `compute_progressive_surprise_curve` is retained for testing/comparison.

3. **`compute_unconditional_surprises(model, tokenizer, prompt, responses)`** — Returns `(per_byte_surprises, total_bits, byte_counts)` for each response conditioned only on the prompt (no other responses).

4. **`_compute_metrics_from_curves(...)`** — Pure math, no model calls. Derives E, E_rate, C, C_total, D, D_total, σ, uncertainty bands from the curves. E_rate is passed in by the caller (computed in the permutation loop).

5. **`compute_icl_diversity_metrics(...)`** — Top-level entry point. Orchestrates the above. When `n_permutations > 1`, shuffles response order, computes curves for each permutation, averages them, and stores per-permutation data.

### Scenario data (`src/icl_diversity/scenarios.py`)

Shared synthetic response sets for the 5 validation scenarios (pure noise, multi incoherent, multi mode, one mode, mixed). Imported by both `tests/test_icl_diversity_scenarios.py` and `scripts/run_scenarios.py`.

### CLI (`src/icl_diversity/cli.py`)

Reads `responses.jsonl` grouped by (scale, prompt_idx), runs `compute_icl_diversity_metrics` per group, writes JSON output. The `**metrics` dict is spread directly into each result entry, so new keys added to `compute_icl_diversity_metrics` flow through automatically.

## Key Design Decisions

- **Base model requirement**: θ must be a base model (not instruction-tuned) to avoid confounding coherence-as-fluency with coherence-as-alignment.
- **Permutation averaging**: When `n_permutations > 1`, the a_k curve is averaged over random response orderings to reduce ordering sensitivity (Section 7.3 of paper). Per-permutation curves are preserved in `per_permutation_a_k_curves`.
- **Conditioning format**: Responses are formatted as `"Response A: ...\n\nResponse B: ..."` in the context window (see `format_conditioning_context`).
- The `__init__.py` re-exports from `core.py` are intentional public API — ruff warns about unused imports but they are re-exports.
