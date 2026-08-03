# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Implementation of the ICL (in-context learning) diversity metric described in the paper (LaTeX source under `paper/`). The metric measures LLM output diversity by computing progressive conditional surprise under a base model θ — as θ sees more responses in-context, surprise decreases proportionally to how many distinct modes exist.

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

# Template-vs-SentBERT experiment (syntactic redundancy that SentBERT misses)
uv run python scripts/run_template_vs_sentbert.py --base-model gpt2 --device cuda:0 --batch-size 16 --n-draws 50 --output results/template_vs_sentbert/gpt2.json
uv run python scripts/plot_template_vs_sentbert.py --input results/template_vs_sentbert/gpt2.json --output-dir figures/template_vs_sentbert/gpt2

# POS-pattern sweep experiment (structural redundancy with zero lexical overlap; mixed result, see reports/POS_PATTERN_VS_BASELINES.md)
uv run python scripts/run_pos_pattern_vs_baselines.py --base-model Qwen/Qwen2.5-3B --device cuda:1 --sentbert-device cuda:1 --torch-dtype bfloat16 --batch-size 8 --output results/pos_pattern/qwen2.5-3b.json
uv run python scripts/plot_template_vs_sentbert.py --input results/pos_pattern/qwen2.5-3b.json --output-dir figures/pos_pattern/qwen2.5-3b

# Single-pattern detection (headline): canonical vs order-scrambled control; a_n beats SentBERT and distinct-n
uv run python scripts/run_pos_pattern_vs_baselines.py --base-model Qwen/Qwen2.5-3B --device cuda:1 --sentbert-device cuda:1 --torch-dtype bfloat16 --batch-size 8 --n-draws 20 --pattern-counts --include-scrambled --output results/pos_pattern/qwen2.5-3b_scrambled_control.json
uv run python scripts/plot_template_vs_sentbert.py --input results/pos_pattern/qwen2.5-3b_scrambled_control.json --output-dir figures/pos_pattern/qwen2.5-3b_scrambled_control
uv run python scripts/check_scrambled_distinctn_capitalization.py

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI for custom response files
uv run calculate-icl-diversity --input responses.jsonl --base-model gpt2 --n-permutations 3

# Regenerate paper/refs.bib from paper/refs_ids.toml (see Citation Pipeline below)
uv run python scripts/build_bib.py

# Lint: every \cite{} in the paper resolves to an entry in refs.bib (offline, no network)
uv run python scripts/verify_cites.py
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

## Paper Tables, Figures, and Inline Numbers

Paper tables are **machine-generated** by `scripts/analyze_c_ainf.py` and `\input{}`'d by the paper — no hand-transcribed numbers.

- **Paper table bodies:** `results/tables/contest_rho_oca.tex`, `results/tables/dectest_rho.tex`
- **Full metric summary (19 variants):** `figures/tevet_validation/c_ainf_analysis_v3/summary_table.txt`
- **Regenerate all:** `uv run python scripts/analyze_c_ainf.py --run-tag qwen25_completion_v3 --output-dir figures/tevet_validation/c_ainf_analysis_v3 --skip-fit`

**Inline scalars** (individual numbers cited in prose: abstract, captions, body text) are **also machine-generated**: every such number resolves through a `\newcommand` macro defined in `results/tables/paper_macros.tex`, produced by `scripts/build_paper_macros.py`. The paper `\input`s this file near the top and writes e.g. `\crossmodeQwenDiagMean` rather than `60.5`. Never hand-type a number into prose.

- **Generated:** `results/tables/paper_macros.tex` (101 macros; do NOT hand-edit)
- **Source:** reads pairwise JSONs under `investigations/cross_mode_surprise_drop/figures/`, `scaling_summary.json`, `results/mode_count/*.json`, Tevet sidecars, and parses `contest_rho_oca.tex` / `dectest_rho.tex` / `qwen3_comparison.tex` / `summary_table.txt`.
- **Regenerate:** `uv run python scripts/build_paper_macros.py`
- **Unit tests:** `uv run pytest tests/test_paper_macros.py` (6 tests; no network; asserts the script runs, every paper-referenced macro resolves, and forbidden hand-typed substrings are absent).

When reading, citing, or discussing table numbers, always read the `.tex` or `.txt` files directly. Cross-check any number that appears in the paper prose against the generating script's output — if it doesn't come from a macro or a `\input`'d table, it's a bug.

All figures referenced by the paper are also script-generated (in `figures/`). The paper compiles from the `paper/` directory (`cd paper && latexmk -pdf`).

## Citation Pipeline

The paper's bibliography is **machine-generated** by `scripts/build_bib.py`, following the same "identifier-first, never hand-type" principle as the tables. Author lists, titles, years, and venues are fetched from arXiv / Crossref / ACL Anthology rather than typed from memory — this prevents the LLM-typical citation-fabrication failure mode.

- **Source of truth:** `paper/refs_ids.toml` (identifier + claim per citation; the ONLY human-edited citation file).
- **Generated:** `paper/refs.bib` (do NOT hand-edit; each entry is annotated with `% source:` and `% claim:` comments).
- **Workflow doc:** `paper/CITATIONS.md` (project-local).
- **Regenerate:** `uv run python scripts/build_bib.py` (hits APIs; fails loudly on any unresolved identifier).
- **Lint:** `uv run python scripts/verify_cites.py` (offline; checks every `\cite{}` resolves to an entry in `refs.bib` and flags unused entries).
- **Unit tests:** `uv run pytest tests/test_bib_pipeline.py` (10 tests; no network).

When adding or editing a citation, edit `refs_ids.toml`, then `build_bib.py`, then `verify_cites.py`, then `latexmk -pdf`. Don't `\bibitem`-by-hand — it reintroduces the failure mode.

## Key Design Decisions

- **Single-pass is the only correct computation.** In a causal LM, pass n of any "multi-pass" sequence already contains all the information of passes 1..n-1 via causal attention, so running separate forward passes is a FLOPs tax for zero benefit. The `_single_pass` suffix on `compute_progressive_surprise_curve_single_pass` is historical — the multi-pass function was deleted because comparing SP to MP is tautological (when tokenization agrees they're bit-exact identical; when it diverges via BPE merges, SP is by definition the metric). Treat any "multi-pass" reference in code or reports as a flag, not as an alternative to validate against.
- **BPE-merged boundary tokens stay with the response.** `_find_response_boundaries` uses a character-span overlap rule: a token belongs to a response if its character span overlaps the response's. When Qwen merges a response's final `.` with the following `\n\n` into a single `.\n\n` token, the overlap rule keeps that token in the response, so the response's cross-entropy includes a small contribution from predicting the upcoming separator. The byte-count denominator is the response's literal byte length. Regression tests: `tests/test_response_boundaries.py::TestBoundaryRoundtrip::test_qwen_trailing_merge_attributed_to_response` and `test_no_separator_leaks_into_response`.
- **Base model requirement**: θ must be a base model (not instruction-tuned) to avoid confounding coherence-as-fluency with coherence-as-alignment.
- **Permutation averaging**: When `n_permutations > 1`, the a_k curve is averaged over random response orderings to reduce ordering sensitivity (Section 7.3 of paper). Per-permutation curves are preserved in `per_permutation_a_k_curves`.
- **Batching**: Unconditional surprises (n short sequences) and permutation forward passes are batched via left-padding with attention mask. `batch_size=1` (default) preserves sequential behavior.
- **Multi-temperature**: `temperature=list[float]` computes metrics for all temperatures from a single set of forward passes. Uses the identity `log_softmax(logits/T) = log_softmax(log_probs_T1/T)` — only T=1 full log-probs are needed. Unconditional surprises (C) are computed once at T=1 and shared; only the progressive curve (E) varies with T.
- **Model ensembling** (Section 7.5): `model` parameter accepts `PreTrainedModel | list[PreTrainedModel]`. For ensembles, softmax probabilities are averaged at each token position (Eq 27), forming a mixture distribution. All models must share the same tokenizer. Ensemble log-probs are accumulated on CPU to support models on different devices.
- **Conditioning format**: Responses are formatted as `"Response A: ...\n\nResponse B: ..."` in the context window (see `format_conditioning_context`).
- The `__init__.py` re-exports from `core.py` are intentional public API — ruff warns about unused imports but they are re-exports.
- **Fail fast, never silently skip**: Never use `continue` to hide errors. If input would cause a failure (e.g., exceeding context length), raise an error upfront rather than producing partial results. Crashing early is preferable to silently skipping bad configurations.

## Ongoing cleanup: the `diversity_score_D` → `C × a_n` naming migration

**The problem.** Historically, `compute_icl_diversity_metrics` in `src/icl_diversity/core.py` returned a dict key named `diversity_score_D` whose value is `coherence_C × excess_entropy_E` — the **Appendix-E alternative** scalar, **not** the paper's primary `D = C × a_n` defined in §6.3. This is a well-established naming trap: every new contributor reaches for `metrics["diversity_score_D"]` thinking it's the paper's headline D, and gets the wrong scalar. It caused a real bug on the RLHF-diversity experiment where the stage ordering came out inverted until we caught it, and caused a separate leak in §8.6's permutation-sensitivity macros where the "D-ranking" was secretly a C·E ranking.

**The interim convention.** `core.py` now exposes the two scalars under unambiguous formula-in-name keys:

- `diversity_score_D_C_an` — the paper's primary **D = C × a_n** (per-byte, §6.3). **Prefer this in all new code.**
- `a_n_per_byte`, `a_n_total` — convenience extracts of the last point of the per-byte / total-bits `a_k_curve`.
- `diversity_score_D_C_E`, `diversity_score_D_C_E_rate` — the explicit C·E variant from Appendix E, for when you actually want that.
- `diversity_score_D`, `diversity_score_D_rate` — **deprecated aliases** of the C·E variants. Retained so that ~24 pre-existing script / test references keep working. Do not use in new code.

**The rule for contributors.**

1. **New code must never write `metrics["diversity_score_D"]`.** If you see this in a diff review — even in a log message — flag it. Use `metrics["diversity_score_D_C_an"]` (paper's primary) or `metrics["diversity_score_D_C_E"]` (alternative) depending on which formula you actually want.
2. **When touching existing code that reads `diversity_score_D`**, migrate it to the formula-in-name key that matches what the *surrounding context* intends. A table labelled "D" in a paper figure almost always wants `C × a_n`; a `C × E` column wants `diversity_score_D_C_E`. If it's not obvious which the context wants, re-read the surrounding prose before renaming.
3. **Do not delete the bare `diversity_score_D` alias yet.** It's the only thing keeping the pre-migration scripts and tests running. Removal is the endpoint of the full overhaul (see below), not a drive-by change.

**The endpoint (full overhaul, separate PR when someone has a quiet week).** Rename every existing reader of `metrics["diversity_score_D"]` to one of the unambiguous keys, update every test that asserts on it (a few literally check `result["diversity_score_D"] == pytest.approx(C × E_value)` — those should move to the `_C_E` key to preserve their assertion semantics), then delete the bare `diversity_score_D` and `diversity_score_D_rate` aliases from `core.py`. Blast radius was last surveyed at ~54 occurrences across 20 files. Use a word-boundary regex: `rg -l '\bdiversity_score_D\b' src/ scripts/ tests/ | xargs sed -i -E 's/\bdiversity_score_D\b/diversity_score_D_C_E/g'`, but *read every test assertion* before committing — some expect specific numeric values, and the rename must preserve those. Run `uv run python -m pytest` + regenerate `results/tables/paper_macros.tex` and diff vs. the prior build to confirm zero numeric drift.

**Why document rather than do.** The bare-key alias is the cheap side; the full overhaul is surgical work across tests with specific numeric assertions. Doing it under time pressure re-introduces the exact bug class this migration is designed to prevent. Treat the rename as its own dedicated cleanup PR.
