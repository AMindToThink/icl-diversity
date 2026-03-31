# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Implementation of the ICL (in-context learning) diversity metric from `paper/in_context_diversity_metric.pdf`. The metric measures LLM output diversity by computing progressive conditional surprise under a base model θ — as θ sees more responses in-context, surprise decreases proportionally to how many distinct modes exist.

The primary a_k curve is in **total bits**. Per-byte normalized quantities (E_rate, C, D_rate) provide tokenizer-agnostic comparisons. D = C × E is the primary diversity score in bits; D_rate = C × E_rate is the per-byte variant.

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

# Batching and ensemble tests (fast, mock model)
uv run pytest tests/test_batching_and_ensemble.py -v

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

# Run temperature sweep experiments
uv run python scripts/run_temperature_experiments.py --device cpu --temperatures 0.5,1.0,2.0 --n-permutations 20

# Analyze temperature experiment results → figures/temperature/
uv run python scripts/analyze_temperature.py

# Interactive audit tool (click points to inspect samples)
# Requires SSH port forwarding: ssh -L 8050:localhost:8050 user@server
uv run scripts/interactive_scatter.py --run-tag qwen25_completion_v3 --device cuda:0
# Then open http://localhost:8050 in your laptop browser

# Analyze C × a_∞ metrics on Tevet evaluation data
uv run python scripts/analyze_c_ainf.py --run-tag qwen25_completion_v3 --skip-fit

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI for custom response files
uv run calculate-icl-diversity --input responses.jsonl --base-model gpt2 --n-permutations 3
```

## Architecture

### Core computation pipeline (`src/icl_diversity/core.py`)

The metric flows through these stages, all in one file. All public functions accept `model: ModelInput` (single model or list for ensembling) and the top-level function accepts `batch_size` for GPU parallelism and `temperature` for logit scaling.

**Internal helpers:**
- `_forward_log_probs(models, input_ids, attention_mask, temperature)` — Runs forward pass through one or more models. Applies `logits / temperature` before softmax. For ensembles, temperature is applied per-model before softmax, then probabilities are averaged (Section 7.5, Eq 27). Raises `ValueError` for API models when `temperature != 1.0`.
- `_forward_full_log_probs(models, input_ids, attention_mask)` — Like `_forward_log_probs` but returns full `(batch, seq_len, vocab_size)` log-probs at T=1 (nats) before diagonal extraction. Used by multi-temperature path to avoid redundant forward passes.
- `_rescale_log_probs(full_log_probs, temperature)` — Rescales T=1 full log-probs to temperature T via `log_softmax(log_probs / T)`. Identity: `log_softmax(logits/T) = log_softmax(log_probs_T1 / T)`.
- `_find_response_boundaries(tokenizer, prompt, responses)` — Tokenizes the full concatenated context and finds token index ranges for each response.
- `_extract_response_log_probs(log_probs, full_ids, boundaries, responses, pad_offset)` — Extracts per-response total bits from a log-probs tensor. Handles left-padding offset.
- `_left_pad_and_batch(sequences, pad_token_id)` — Left-pads variable-length token sequences into a batch with attention mask.

**Public functions:**

1. **`compute_cross_entropy(model, tokenizer, text, prefix)`** — Tokenizes prefix+text, runs a forward pass, extracts log-probs for just the text tokens. Returns `(total_bits, byte_count)`. This is the atomic building block. `compute_per_byte_cross_entropy` is a thin wrapper that divides by byte count.

2. **`compute_progressive_surprise_curve_single_pass(model, tokenizer, prompt, responses)`** — Single forward pass over the full concatenated context, extracting per-response log-probs by token boundary detection. Returns `(curve_total_bits, byte_counts)`. This is the default used by `compute_icl_diversity_metrics`. The old multi-pass version `compute_progressive_surprise_curve` is retained for testing/comparison.

3. **`compute_unconditional_surprises(model, tokenizer, prompt, responses, batch_size)`** — Returns `(per_byte_surprises, total_bits, byte_counts)` for each response conditioned only on the prompt (no other responses). The n forward passes are batched according to `batch_size`.

4. **`_compute_metrics_from_curves(...)`** — Pure math, no model calls. Derives E, E_rate, C, D, D_rate, σ, uncertainty bands from the curves. E_rate is passed in by the caller (computed in the permutation loop).

5. **`_compute_permutation_curves_batched(models, tokenizer, prompt, responses, permutations, batch_size)`** — Computes single-pass a_k curves for multiple permutations in batched forward passes. Each permutation is an independent sequence.

6. **`compute_icl_diversity_metrics(model, tokenizer, prompt, responses, n_permutations, seed, batch_size, temperature)`** — Top-level entry point. Orchestrates the above. When `n_permutations > 1`, generates all permutations upfront and batch-computes their curves. Supports model ensembling by passing a list of models. **Multi-temperature**: when `temperature` is a `list[float]`, performs one forward pass and derives all temperatures, returning `{"temperatures": {T: metrics_dict}}`. When `temperature` is a single float (default), backward-compatible flat dict.

### Scenario data (`src/icl_diversity/scenarios.py`)

Shared synthetic response sets for the 5 validation scenarios (pure noise, multi incoherent, multi mode, one mode, mixed). Imported by both `tests/test_icl_diversity_scenarios.py` and `scripts/run_scenarios.py`.

### CLI (`src/icl_diversity/cli.py`)

Reads `responses.jsonl` grouped by (scale, prompt_idx), runs `compute_icl_diversity_metrics` per group, writes JSON output. The `**metrics` dict is spread directly into each result entry, so new keys added to `compute_icl_diversity_metrics` flow through automatically.

## Paper Tables and Figures

Paper tables are **machine-generated** by `scripts/analyze_c_ainf.py` and `\input{}`'d by the paper — no hand-transcribed numbers.

- **Paper table bodies:** `results/tables/contest_rho_oca.tex`, `results/tables/dectest_rho.tex`
- **Full metric summary (19 variants):** `figures/tevet_validation/c_ainf_analysis_v3/summary_table.txt`
- **Regenerate all:** `uv run python scripts/analyze_c_ainf.py --run-tag qwen25_completion_v3 --output-dir figures/tevet_validation/c_ainf_analysis_v3 --skip-fit`

When reading, citing, or discussing table numbers, always read the `.tex` or `.txt` files directly. Cross-check any hand-written inline numbers in the paper prose against the generated tables.

All figures referenced by the paper are also script-generated (in `figures/`). The paper compiles from the `paper/` directory (`cd paper && latexmk -pdf`).

## Key Design Decisions

- **Base model requirement**: θ must be a base model (not instruction-tuned) to avoid confounding coherence-as-fluency with coherence-as-alignment.
- **Permutation averaging**: When `n_permutations > 1`, the a_k curve is averaged over random response orderings to reduce ordering sensitivity (Section 7.3 of paper). Per-permutation curves are preserved in `per_permutation_a_k_curves`.
- **Batching**: Unconditional surprises (n short sequences) and permutation forward passes are batched via left-padding with attention mask. `batch_size=1` (default) preserves sequential behavior.
- **Multi-temperature**: `temperature=list[float]` computes metrics for all temperatures from a single set of forward passes. Uses the identity `log_softmax(logits/T) = log_softmax(log_probs_T1/T)` — only T=1 full log-probs are needed. Unconditional surprises (C) are computed once at T=1 and shared; only the progressive curve (E) varies with T.
- **Model ensembling** (Section 7.5): `model` parameter accepts `PreTrainedModel | list[PreTrainedModel]`. For ensembles, softmax probabilities are averaged at each token position (Eq 27), forming a mixture distribution. All models must share the same tokenizer. Ensemble log-probs are accumulated on CPU to support models on different devices.
- **Conditioning format**: Responses are formatted as `"Response A: ...\n\nResponse B: ..."` in the context window (see `format_conditioning_context`).
- The `__init__.py` re-exports from `core.py` are intentional public API — ruff warns about unused imports but they are re-exports.
- **Fail fast, never silently skip**: Never use `continue` to hide errors. If input would cause a failure (e.g., exceeding context length), raise an error upfront rather than producing partial results. Crashing early is preferable to silently skipping bad configurations.
