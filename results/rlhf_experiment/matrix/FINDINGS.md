# OLMo in-family self-scoring matrix — findings

**Date:** 2026-05-31/06-01 · **Branch:** `olmo-self-scoring`
**Question (reviewer):** does the OLMo post-training diversity drop replicate when θ
(the grader) is in the OLMo family — including a model grading its own outputs —
rather than the cross-family Qwen2.5-3B? I.e. *"how much does D = C·a_n just reward
responses that resemble the grader?"*

## Design

Re-scored the **fixed**, length-matched OLMo-2-1124-7B generations (4 stages: base,
sft, dpo, instruct; 200 AlpacaEval + 100 NoveltyBench prompts; K=10/prompt; n=150 /
n=39 after the ≥50-byte length-match floor) with **every pipeline stage as the grader
θ**, plus the published Qwen row. 5 grader rows × 4 gen columns × 2 prompt sets. Only θ
varies; responses, permutations (25, seed 42), and length-matching are held fixed.
Metric **D = C·a_n** (`diversity_score_D_C_an`), aggregated as mean of the per-prompt
product. The dpo→instruct step carries **no prediction** (near-tie); the confirmatory
chain is **base > sft > dpo** via adjacent one-sided paired Wilcoxon, Holm-corrected.

Refactor was validated bit-for-bit: re-scoring with θ=Qwen reproduces the committed
length-matched D to max 0.01% (mean 0.00%) per prompt-id; batch-size 8 vs 32 agree to
0.005%. Scorer CLI unit tests: 4/4 pass.

## Headline result — the drop is robust to the grader (H1, H7)

**D = C·a_n decreases base > sft > dpo in every grader row, on both prompt sets, with
every confirmatory contrast significant after Holm correction.** AlpacaEval D:

| θ (grader) \ gen | base | sft | dpo | instruct | base>sft>dpo |
|---|---|---|---|---|---|
| Qwen-3B (published) | 0.481 | 0.329 | 0.286 | 0.281 | ✓ |
| OLMo-base | 0.494 | 0.345 | 0.298 | 0.293 | ✓ |
| OLMo-sft | 0.487 | 0.356 | 0.313 | 0.308 | ✓ |
| OLMo-dpo | 0.494 | 0.398 | 0.354 | 0.349 | ✓ |
| **OLMo-instruct (RL)** | **0.494** | **0.404** | **0.359** | **0.354** | **✓** |

The **adversarial row** — OLMo-Instruct grading its own family's aligned outputs, the
configuration that maximally advantages the RLHF stage's coherence and violates the
paper's "θ must be a base model" rule — **still ranks base most-diverse, instruct
least** (0.494 vs 0.354; base>sft Holm p=8e-16, sft>dpo p=1.3e-8). The order never
flips. NoveltyBench shows the identical pattern (all 10 contrasts significant, n=39).
**The drop is not a cross-family scoring artifact and the metric does not merely reward
grader-similarity.**

## Mechanism — a_n carries the drop; self-similarity lives in C (H2, H8, H_diag, H10)

- **H2 / H8:** Across base→instruct, **a_n falls ~2.7×** while **C rises** in every row.
  C alone *increases* with alignment (it is not a diversity signal); the diversity drop
  is carried entirely by the a_n (mutual-predictability) term. AlpacaEval, θ=Qwen:
  a_n 1.14→0.42, C 0.44→0.70.
- **H_diag (home advantage in C):** present but **confined to the base column and the
  opposite of a confound.** On base-stage text, C is highest for the base grader and
  falls monotonically as the grader becomes more aligned (AlpacaEval base column:
  0.420→0.395→0.341→0.331 for base→sft→dpo→instruct θ). For the aligned columns the
  diagonal C bonus is weak/absent. So aligned graders find base text *less* coherent,
  not more — there is no mechanism by which self-similarity inflates an aligned stage's D.
- **H10 (a_n grader-invariant):** the a_n *ranking* across gen stages is identical in
  every row; aligned graders assign *higher* a_n to base text (1.14→1.58), reinforcing
  rather than eroding the diversity gap. Self-similarity does not contaminate the
  diversity term.

## Drop magnitude shrinks but survives (H6)

Fold-change R_θ = mean_D(base)/mean_D(dpo) shrinks monotonically as the grader becomes
more aligned, but stays well above 1 — exactly the predicted "lesser effect under the RL
grader":

| θ | R (AlpacaEval) | 95% CI | R (NB) | 95% CI |
|---|---|---|---|---|
| Qwen-3B | 1.68 | [1.61, 1.76] | 1.54 | [1.41, 1.68] |
| OLMo-base | 1.66 | [1.60, 1.72] | 1.63 | [1.51, 1.76] |
| OLMo-sft | 1.56 | [1.50, 1.62] | 1.53 | [1.41, 1.66] |
| OLMo-dpo | 1.40 | [1.34, 1.45] | 1.39 | [1.28, 1.49] |
| OLMo-instruct | 1.38 | [1.33, 1.43] | 1.37 | [1.27, 1.47] |

## Conclusion

The monotone post-training diversity drop replicates under in-family scoring at every
OLMo pipeline stage, including the RL model grading its own outputs. It is carried by
a_n (mutual predictability), not by the coherence term, and the only self-similarity
signal (a base-stage coherence bonus for the base grader) pushes *against* a spurious
drop rather than creating one. This rules out the cross-family scoring artifact the
reviewer raised.

## Reproduce

```bash
# score one grader row (length-matched); CUDA_VISIBLE_DEVICES picks the GPU
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rlhf_experiment/3_score_icl_diversity.py \
  --scorer-model allenai/OLMo-2-1124-7B-Instruct \
  --gen-dir results/rlhf_experiment/generations_length_matched --batch-size 32 \
  --out results/rlhf_experiment/matrix/icl_metrics_lm_theta-instruct.jsonl
# assemble matrix + stats + a_k overlays
uv run python scripts/rlhf_experiment/6_matrix_analysis.py --n-boot 10000 --seed 42
```

Outputs: `matrix_summary.{txt,json}` here; a_k overlays under
`figures/rlhf_experiment/matrix/`. Per-grader metric JSONLs:
`icl_metrics_lm_theta-{base,sft,dpo,instruct}.jsonl`.
