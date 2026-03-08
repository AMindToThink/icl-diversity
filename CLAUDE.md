# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Implementation of the ICL (in-context learning) diversity metric from `paper/in_context_diversity_metric.pdf`. The metric measures LLM output diversity by computing progressive conditional surprise under a base model θ — as θ sees more responses in-context, surprise decreases proportionally to how many distinct modes exist.

All information quantities are in **bits/byte** (not bits/token), making the metric tokenizer-agnostic.

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

# Compute validation scenario metrics → results/scenario_metrics.json
uv run python scripts/run_scenarios.py

# Generate a_k curve plots from saved JSON → figures/
uv run python scripts/plot_ak_curves.py

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI for custom response files
uv run calculate-icl-diversity --input responses.jsonl --base-model gpt2 --n-permutations 3
```

## Architecture

### Core computation pipeline (`src/icl_diversity/core.py`)

The metric flows through these stages, all in one file:

1. **`compute_per_byte_cross_entropy(model, tokenizer, text, prefix)`** — Tokenizes prefix+text, runs a forward pass, extracts log-probs for just the text tokens, converts to bits/byte. This is the atomic building block.

2. **`compute_progressive_surprise_curve(model, tokenizer, prompt, responses)`** — Calls `compute_per_byte_cross_entropy` for each response conditioned on all previous responses (formatted via `format_conditioning_context`). Returns the a_k curve (list of floats).

3. **`compute_unconditional_surprises(model, tokenizer, prompt, responses)`** — Per-byte cross-entropy of each response conditioned only on the prompt (no other responses).

4. **`_compute_metrics_from_curves(a_k_curve, unconditional_surprises, responses)`** — Pure math, no model calls. Derives E, C, D, σ, m_eff, uncertainty bands from the curves.

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
