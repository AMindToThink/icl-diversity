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

## 3. arXiv paper title is out of date

arXiv v1 metadata (https://arxiv.org/abs/2606.01811, verified 2026-06-23) still carries
the OLD SHORT title:

> "I've Seen How This Goes": Characterizing Diversity via Progressive Conditional Surprise

The camera-ready PDF, slide, and poster all use the CURRENT LONG title:

> "I've Seen How This Goes": Characterizing the Diversity of LLM Generations and Human
> Writing via Progressive Conditional Surprise

When submitting the replacement version, update the arXiv title field to the long title
so arXiv matches the paper. OpenReview's listing also still shows the short title;
updating that is separate and optional.

## 4. Drop "each later stage's curve lies below the base curve at every $k \geq 2$"

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
