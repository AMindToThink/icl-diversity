# arXiv v2: deferred fixes

Changes intentionally **not** made to the camera-ready / accepted workshop version,
to be applied at the next arXiv build (`scripts/build_arxiv.sh`). The version on
OpenReview is frozen; do not edit it for any of these. Apply them when posting the
replacement (v2) on arXiv.

## 1. Conclusion: "single forward pass" should say "per permutation"

`paper/sections/conclusion_workshop.tex`: change "in a single forward pass" to
"in a single forward pass per permutation".

The abstract already states it correctly ("a single forward pass per permutation");
only the conclusion dropped the qualifier. The metric runs one forward pass *per
permutation* and averages over n_sigma random permutations, so the bare "single
forward pass" understates the per-ordering claim. The genuine claim is per-ordering:
one pass over the concatenated responses scores all n responses' conditional
surprises at once (causal LM; no need for n growing-prefix passes).

## 2. DPO/RLVR connector: arrow should be a comma list (no implied trend)

- `paper/sections/abstract_workshop.tex` (line ~6): "drops monotonically across the
  base $\to$ SFT $\to$ DPO $\to$ RLVR stages" becomes "drops monotonically across the
  base, SFT, DPO, and RLVR stages".
- `paper/sections/07_6_rlhf_workshop.tex` (line ~62): same "base $\to$ SFT $\to$ DPO
  $\to$ Instruct (RLVR)" arrow chain becomes a comma list.

Why: DPO-vs-RLVR is the paper's exploratory H1' two-sided contrast with **no
directional prediction** ("we have no directional prediction for which post-training
stage loses more diversity"). An arrow asserts an ordered drop between DPO (0.286) and
RLVR (0.281) that we do not claim. A comma list keeps "monotonically" (true, since
0.286 >= 0.281) without asserting pairwise direction. The poster already uses the
comma-list prose and uses approximately-equals in its numeric stage sequence
(`DPO 0.286 &asymp; RLVR 0.281`).

## 3. Candidate new appendix subsection: structural redundancy (drafted)

`paper/sections/07_9_structural_redundancy.tex` (drafted 2026-08-03) presents the
structural-redundancy experiments: syntactic frames vs SentBERT
(`reports/TEMPLATE_VS_SENTBERT.md`) and the canonical-vs-scrambled POS-pattern
detection with a known ground-truth entropy gap
(`reports/POS_PATTERN_VS_BASELINES.md`). A commented `\input` line sits after
`07_4_cross_mode` in `main_icml_workshop.tex`; uncomment it for v2 (decide final
placement then). All inline numbers are macros (`framesQwen*` / `posPattern*`)
emitted by `scripts/build_paper_macros.py` from the script-generated summaries.

## 4. B.1 "The formatting choice affects results" is ungrounded

`paper/sections/06_practical.tex` (App B.1) asserts "The formatting choice affects results"
with no supporting test — no format A/B comparison exists anywhere in the repo (all Tevet
runs are completion-format; `old_gpt2_instruct` is superseded, not a clean comparison).
Matthew expects format NOT to matter meaningfully (2026-08-03). Either soften to a
data-type-matching statement ("we match the conditioning format to the response style:
instruct-style responses use the instruct format, continuations the completion format")
or ground it with an actual A/B test. Do not restate the untested assertion.

## 5. Appendix F gap-direction wording slip

`paper/sections/appC_mcdiv_confounds.tex` says "the high-minus-low per-byte gap remains
positive," but both generated tables (`results/tables/confound_stats.tex`,
`confound_length.tex`) define Gap = low-minus-high (e.g., a_1 1.154 − 0.977 = +0.177;
mean bytes 46.5 − 52.7 = −6.3). Numbers and conclusion are correct; flip the prose
direction label to "low-minus-high" (found 2026-08-04 while preparing the NeurIPS rebuttal).

Also re-examine when rested: during the rebuttal Matthew was tired and did not think hard
about the "content-driven, not a length artifact" interpretation (the length-bin
stratification: gap +0.167/+0.141/+0.176 in the first three bins, +0.003 in the longest).
The rebuttal wording was checked against the tables, but the reasoning itself deserves a
fresh-eyes review before it goes into any paper revision.

## 6. arXiv paper title is out of date

arXiv v1 metadata (https://arxiv.org/abs/2606.01811, verified 2026-06-23) still carries
the OLD SHORT title:

> "I've Seen How This Goes": Characterizing Diversity via Progressive Conditional Surprise

The camera-ready PDF, slide, and poster all use the CURRENT LONG title:

> "I've Seen How This Goes": Characterizing the Diversity of LLM Generations and Human
> Writing via Progressive Conditional Surprise

When submitting the replacement version, update the arXiv title field to the long title
so arXiv matches the paper. OpenReview's listing also still shows the short title;
updating that is separate and optional.

## 7. Drop "each later stage's curve lies below the base curve at every $k \geq 2$"

Remove (or rewrite) the "lies below the base curve at every $k \geq 2$" claim in the
RLHF/OLMo section and its figure captions. Occurrences:

- `paper/sections/07_6_rlhf_workshop.tex:64` (prose)
- `paper/sections/07_6_rlhf_workshop.tex:76` (Figure caption)
- `paper/sections/07_6_rlhf_workshop.tex:80` (duplicate prose variant)
- `paper/sections/appF_rlhf_cross_metric.tex:20` (Figure caption)

Why: the observation is technically correct but not a useful, independent result. The
overall height of each $\bar{a}_k$ curve is largely determined by $a_1$ — the
unconditional per-byte entropy of that stage's responses. A stage whose responses have
lower unconditional entropy starts lower and simply stays lower across $k$, so "lies
below at every $k \geq 2$" mostly restates the $a_1$ (coherence/entropy) gap rather than
demonstrating a difference in the conditional-surprise *decay* that the diversity signal
is about. Keep the pre-registered contrasts (they test $D_{Ca_n}$/$a_n$ properly); only
the "curve-below-base-at-every-k" phrasing should go. The poster (`paper/poster/
icl_diversity_poster.html`) has already had this sentence removed from the
"Catches post-training mode collapse" caption.

## 8. RLHF $\bar{a}_k$ overlay figure: y-axis mislabeled "(bits/byte)" — should be "(total bits)"

`scripts/rlhf_experiment/5_analyze_and_figures.py:257` labeled the $\bar{a}_k$
overlay y-axis `(bits/byte)`, but that curve is the **total-bits** progressive surprise
(values run ~80–240; per-byte cross-entropy is ~1–2 bits/byte). The adjacent violin
(`C \times a_n`, line 278) is correctly per-byte. Two poster reviewers independently
flagged the same-panel scale mismatch.

**Status: already fixed in local source.** The script line was corrected to
`(total bits)` and the two length-matched overlay PDFs
(`figures/rlhf_experiment/ak_curves_overlay_{alpacaeval,nbcurated}_lm.pdf`) were
regenerated from `results/rlhf_experiment/icl_metrics_length_matched.jsonl`; the poster
asset `paper/poster/assets/fig2_ak.png` was re-exported. Only the frozen OpenReview v1
PDF still carries the old "(bits/byte)" label. Nothing further to do at the v2 build
beyond the normal figure regeneration, which now emits the correct label.
