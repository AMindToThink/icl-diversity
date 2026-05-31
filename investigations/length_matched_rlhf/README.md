# Length-matched OLMo-2-7B RLHF re-run (open investigation)

## One-line task

Re-score the OLMo-2-7B four-stage RLHF experiment after truncating every response to a per-prompt common length, to check whether the reported monotone $D = C \times a_n$ drop across base $\to$ SFT $\to$ DPO $\to$ Instruct survives when response length is held fixed.

## Why this matters

The paper (`paper/rlhf_experiment.tex`) reports a monotone per-byte $D$ drop across the four post-training stages on AlpacaEval and NB-curated prompt sets. The concern, raised during the 2026-04-25 §1 checklist pass:

**Both $C$ and $a_n$ are length-sensitive even though they are per-byte quantities.** In a causal LM, longer responses let $\theta$ use more within-response context to predict later tokens, which:

- *raises $C$* (the geometric-mean per-byte surprise under the prompt drops as $\theta$ benefits from each response's own earlier tokens), and
- *lowers $a_n$* (same within-response effect, compounded with cross-response context).

So $D = C \times a_n$ is **not** invariant to response length by construction. The product's direction under a length change depends on the relative magnitudes of the $C$ rise and the $a_n$ fall, which in turn depend on how quickly per-byte surprise decays with context length for $\theta$.

A naïve length audit on the raw generations (`scripts/audit_response_lengths.py`, 2026-04-25) showed length is not monotone across stages on either prompt set. But the audit does not by itself prove length cannot explain $D$: it only shows the raw length pattern is non-monotone. The cleanest control is to **equalise response length per prompt across stages**, re-score, and compare trends.

## Experiment specification

### Inputs

- Generations: `results/rlhf_experiment/generations/{base,sft,dpo,instruct}_{alpacaeval,nbcurated}.jsonl`
- 10 responses per `prompt_id` per stage per prompt\_set. `max_new_tokens=100`, temperature=1.0, top-p=1.0, seed=42.
- Stage labels: `base` = pretrained OLMo-2-1124-7B, `sft` = SFT checkpoint, `dpo` = DPO checkpoint, `instruct` = final RLVR Instruct checkpoint.

### Procedure

1. **Group by prompt.** For each `(prompt_set, prompt_id)`, collect every response across all four stages and all 10 `sample_idx` slots — that's $4 \times 10 = 40$ responses per prompt.
2. **Compute the per-prompt truncation length.** Find the minimum UTF-8 byte length across those 40 responses. Call it $L_{\text{prompt}}$.
3. **Truncate.** Clip every response for that prompt (across all 4 stages, all 10 samples) to exactly $L_{\text{prompt}}$ bytes. Handle UTF-8: if the naive byte cut falls mid-character, back off to the largest valid UTF-8 prefix $\le L_{\text{prompt}}$ (use `response.encode("utf-8")[:L].decode("utf-8", errors="ignore")` or equivalent). **Use the identical truncation length for every response of that prompt** — the point is to equalise the length seen by $\theta$ at scoring time.
4. **Write truncated generations** to a parallel directory: `results/rlhf_experiment/generations_length_matched/{stage}_{set}.jsonl`. Preserve every field of the original record; overwrite only `response`. Add a field `original_bytes` for traceability.
5. **Re-score with the existing pipeline.** `scripts/rlhf_experiment/3_score_icl_diversity.py` reads from `GEN_DIR = REPO_ROOT / "results" / "rlhf_experiment" / "generations"`. Either:
    - (preferred) Add a `--gen-dir` CLI flag and point it at `generations_length_matched`.
    - (alternate) Duplicate the script as `3_score_icl_diversity_lenmatched.py` with the new path hard-coded.
   Write output to a parallel `results/rlhf_experiment/icl_metrics_length_matched.jsonl`.
6. **Regenerate the analysis table.** `scripts/rlhf_experiment/5_analyze_and_figures.py` produces the per-stage $D$ means and the pre-registered contrasts. Run it pointing at the length-matched metrics file; write outputs to `results/rlhf_experiment/tables_length_matched/` and `results/rlhf_experiment/paper_macros_rlhf_lenmatched.tex` (do not overwrite the original macros).

### What "success" looks like

The comparison is a single report: **raw vs length-matched per-stage $D$ means and contrast effect sizes, for both prompt sets.**

Three outcomes:

- **Monotone $D$ drop survives length-matching.** The "post-training reduces diversity" claim stands. Paper can add a short Limitations paragraph citing the re-run as robustness evidence.
- **Monotone $D$ drop disappears or weakens substantially.** Length was at least partially driving the effect. The paper's tertiary headline claim needs to be rewritten — likely kept in a weaker form ("we observe a $D$ drop on raw generations; under length-matched scoring the effect attenuates by X%, so the signal is partly length-mediated").
- **$D$ drop partially survives.** Report both numbers honestly and soften the claim to match the surviving magnitude.

### Notes

- Do not discard the raw-generation pipeline. The length-matched scoring is a robustness check, not a replacement.
- The 100-word truncation used for the cross-model table (`paper/rlhf_experiment.tex` "Cross-model comparison" paragraph) is a different, already-documented preprocessing step. Do not conflate the two.
- After the re-run, regenerate `paper_macros.tex` via `scripts/build_paper_macros.py` if any paper-referenced numbers change.
- Once the investigation is complete, update the paper's §Limitations with the outcome.

## Quick data primer

Each generation record looks like:
```json
{
  "prompt_id": "alpacaeval-000",
  "prompt": "What are the names of some famous actors...",
  "stage": "base",
  "model": "allenai/OLMo-2-1124-7B",
  "prompt_set": "alpacaeval",
  "sample_idx": 0,
  "response": " What's the most famous music venue...",
  "temperature": 1.0,
  "top_p": 1.0,
  "max_new_tokens": 100,
  "seed": 42,
  "finish_reason": "length"
}
```

Byte lengths summary from `scripts/audit_response_lengths.py`:

| Stage    | AlpacaEval mean bytes | NB-curated mean bytes |
|----------|----------------------:|----------------------:|
| base     | 395.5                 | 399.9                 |
| sft      | 410.5                 | **249.0**             |
| dpo      | 468.7                 | 401.0                 |
| instruct | 465.2                 | 373.2                 |

NB-curated SFT is the most extreme — for many prompts, $L_{\text{prompt}}$ (the min across 40 responses) will be dragged down by an SFT response. That's the correct behaviour; it's the length we need to match to in order to compare apples to apples.

## Handoff

Raise any blockers with the lead investigator. When the re-run finishes, report:

1. Per-stage $D$ means (raw vs length-matched) for both prompt sets.
2. Effect sizes ($d_z$) and Bonferroni-corrected $p$-values for the three pre-registered contrasts (H1a, H1b, H1c) under length-matched scoring.
3. A one-paragraph verdict.
