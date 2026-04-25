# Length-matched RLHF re-run — verdict

**Date:** 2026-04-25
**Investigator:** Claude (under Matthew)
**Source spec:** `investigations/length_matched_rlhf/README.md`

## TL;DR

**Outcome 1 from the README applies: the monotone $D$ drop across `base → SFT → DPO → Instruct` survives length-matching on both prompt sets.** All three pre-registered H1 contrasts remain Bonferroni-significant ($p < 0.001$) with large effect sizes after every response is truncated to a per-prompt common UTF-8 byte length. The `Base → Instruct` per-byte $D$ drop is essentially unchanged — slightly *amplified* on AlpacaEval (raw 0.186 → length-matched 0.199, –7.5% attenuation, i.e. larger) and modestly attenuated on NB-curated (raw 0.202 → length-matched 0.179, 11.8% attenuation). The paper's tertiary post-training claim stands and can cite this re-run as a robustness check.

## Sanity-check finding (worth surfacing in the paper)

When we attempted to length-match every prompt to the minimum byte-length across its 40 responses, **111 of 300 prompts (37%)** had a common length below 50 UTF-8 bytes, including 8 AlpacaEval prompts where some sample slot is literally 0 bytes. The min was driven by the **base model on AlpacaEval** (136/200 prompts) and by the **SFT checkpoint on NB-curated** (62/100 prompts). Length-matching those prompts would mean comparing essentially-empty fragments — clearly not a meaningful diversity comparison — so we dropped them and ran the analysis on the surviving subset (**150/200 AlpacaEval, 39/100 NB-curated**). See `generations_length_matched/length_match_report.json`.

This is itself informative: the base model frequently emits a stop-token / newline almost immediately for an instruction-formatted prompt (it has no instruction-following prior), and the SFT model on the NB-curated short-answer prompts often emits very short replies. Both are consistent with stage-specific behavior and explain part of the raw byte-length non-monotonicity that motivated this audit.

## Numbers

`D = C × a_n` (per-byte; paper §6.3 primary scalar). Full table in `04_comparison.txt`.

### AlpacaEval (raw $n=200$, length-matched $n=150$)

| Stage | raw mean | length-matched mean |
|---|---:|---:|
| Base | 0.469 | 0.481 |
| SFT  | 0.320 | 0.329 |
| DPO  | 0.288 | 0.286 |
| Instruct | 0.283 | 0.281 |

Monotone decrease holds in both. H1 contrasts (paired Wilcoxon, Bonferroni $\alpha=0.05/3$):
- H1a Base > SFT:      raw $d_z=1.78$, $p<0.001$ → length-matched $d_z=1.62$, $p<0.001$
- H1b SFT > DPO:       raw $d_z=0.64$, $p<0.001$ → length-matched $d_z=0.68$, $p<0.001$
- H1c Base > Instruct: raw $d_z=2.39$, $p<0.001$ → length-matched $d_z=2.43$, $p<0.001$

### NB-curated (raw $n=100$, length-matched $n=39$)

| Stage | raw mean | length-matched mean |
|---|---:|---:|
| Base | 0.485 | 0.481 |
| SFT  | 0.330 | 0.369 |
| DPO  | 0.299 | 0.312 |
| Instruct | 0.283 | 0.303 |

Monotone decrease holds in both (note that on the surviving subset, SFT/DPO/Instruct sit modestly higher than on the full prompt set — the dropped prompts are systematically harder ones where post-training models give terse replies; on those that remain, every stage scores a bit higher but the ordering is preserved). H1 contrasts:
- H1a Base > SFT:      raw $d_z=1.64$, $p<0.001$ → length-matched $d_z=1.21$, $p<0.001$
- H1b SFT > DPO:       raw $d_z=0.44$, $p<0.001$ → length-matched $d_z=0.96$, $p<0.001$
- H1c Base > Instruct: raw $d_z=2.61$, $p<0.001$ → length-matched $d_z=1.89$, $p<0.001$

## Interpretation

The README laid out three possible outcomes. **Outcome 1 (monotone $D$ drop survives length-matching)** is what we observe:

> Paper can add a short Limitations paragraph citing the re-run as robustness evidence.

The drop is not a length-of-response artifact: under a per-prompt fixed truncation length, post-training still moves $D$ in the same direction with the same effect-size order of magnitude. On AlpacaEval the drop is even slightly *larger* under length-matching, which means the raw length pattern was, if anything, masking part of the diversity drop rather than driving it.

The single caveat the paper should flag is the **dropped prompt fraction** — length-matching is undefined for prompts where some stage emits ~0 bytes, and that situation is concentrated in (a) the base model on AlpacaEval (no instruction prior) and (b) SFT on NB-curated (short-answer prompts). The reported length-matched effect therefore conditions on prompts where length-matching is a well-defined comparison.

## Files produced

- `investigations/length_matched_rlhf/01_truncate.log`
- `investigations/length_matched_rlhf/02_score.log`
- `investigations/length_matched_rlhf/03_analyze.log`
- `investigations/length_matched_rlhf/04_comparison.txt`
- `investigations/length_matched_rlhf/04_comparison.json`
- `results/rlhf_experiment/generations_length_matched/*.jsonl` (truncated generations)
- `results/rlhf_experiment/generations_length_matched/length_match_report.json`
- `results/rlhf_experiment/icl_metrics_length_matched.jsonl` (756 scored groups)
- `results/rlhf_experiment/analysis_length_matched.json`
- `results/rlhf_experiment/tables_length_matched/rlhf_diversity.tex`
- `results/rlhf_experiment/paper_macros_rlhf_lenmatched.tex` (macros prefixed `\olmoLm…`)
- `figures/rlhf_experiment/{ak_curves_overlay,per_prompt_D_violin}_{alpacaeval,nbcurated}_lm.pdf`

## How to reproduce

```bash
# 1. Truncate generations to per-prompt common length, drop below-50-byte prompts
uv run python scripts/rlhf_experiment/3a_truncate_length_matched.py

# 2. Score the length-matched copy on GPU 1 (~25 min on RTX 8000)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rlhf_experiment/3_score_icl_diversity.py \
  --gen-dir results/rlhf_experiment/generations_length_matched \
  --out results/rlhf_experiment/icl_metrics_length_matched.jsonl

# 3. Analyze + write parallel macros / tables / figures
uv run python scripts/rlhf_experiment/5_analyze_and_figures.py \
  --icl results/rlhf_experiment/icl_metrics_length_matched.jsonl \
  --baselines results/rlhf_experiment/baseline_metrics_DOES_NOT_EXIST.jsonl \
  --analysis-name analysis_length_matched.json \
  --table-subdir tables_length_matched \
  --macros-name paper_macros_rlhf_lenmatched.tex \
  --macro-infix Lm \
  --fig-suffix _lm

# 4. Emit comparison report
uv run python scripts/rlhf_experiment/6_compare_raw_vs_lenmatched.py
```
