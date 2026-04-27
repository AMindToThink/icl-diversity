# Workshop Choice: Human-AI Co-Creativity @ ICML 2026

## TL;DR

Submit to **Human-AI Co-Creativity @ ICML 2026** ([genaicreativity.org/icml2026](https://genaicreativity.org/icml2026/)). Deadline 1 May 2026 AOE, notification 15 May, non-archival, double-blind, ICML format, 4–8 pages main content.

## What the paper actually does (in one paragraph, since the previous version of this doc got it wrong)

We propose a per-byte diversity score $D = C \times a_n$ for a set of $n$ responses to a prompt $p$. Compute it like this: concatenate the prompt and the responses with `Response A:`, `Response B:`, ... labels, run **one forward pass** through a base language model $\theta$, and read off the per-token log-probs. The progressive conditional surprise curve $a_k$ is the per-byte cross-entropy of the $k$-th response given the previous $k{-}1$ responses; $a_n$ is its last point. The coherence term $C = 1/\mathrm{PPL}_\theta(\pi, p)$ is the geometric-mean per-byte probability $\theta$ assigns to the responses individually. Multiply. That is the whole pipeline. No embedding model, no reference corpus, no human labels, no parametric curve fit, no auxiliary classifier. The same pipeline scores AI outputs and human writing without modification. A working open-source reference implementation is at [github.com/AMindToThink/icl-diversity](https://github.com/AMindToThink/icl-diversity) (cited in paper §7.1).

## Why this venue, not the others considered

Several other workshops looked plausible on paper but failed on closer reading:

- **Foundations of Deep Generative Models (FoGen)** — about what diffusion/flow/AR models internally learn (memorization vs. generalization regimes, training dynamics). Our paper uses a generative model as a measurement instrument; it's not about the model's internals.
- **Structured Probabilistic Inference & Generative Modeling (SPIGM)** — classical probabilistic-inference machinery (graphical models, MCMC, variational methods). Our paper isn't an inference-methodology contribution.
- **Combining Theory and Benchmarks (CTB)** — explicitly aimed at *predictive science of foundation model performance*, with PAC-Bayes guarantees, scaling laws, and ex-ante capability prediction as core concepts. Our metric measures a property of output sets; it doesn't guarantee or predict capability. A reviewer there would ask "what does this guarantee?" and we'd have no clean answer.
- **Hypothesis Testing** — classical statistical testing (e-values, anytime-valid inference, kernel two-sample tests). We're not proposing a testing methodology.
- **Memorization workshop** — narrowly about training-data leakage and privacy, not output-distribution properties.
- **Pluralistic Alignment** — about value/cultural diversity, not output diversity in general. Our metric is a tool that community could use, but it's not the community's central concern.

## Why Co-Creativity is the right fit

Three bullets in their CFP map almost directly onto the paper:

1. *"Curating benchmarks and shared resources that focus on evaluating the creativity aspects of AI agents"* — the strongest hit. $D = C \times a_n$ is exactly an evaluation tool for the diversity of generated outputs, and the same pipeline applies to AI samples (validated on the OLMo-2-7B post-training stages and a frontier-model NoveltyBench ladder, paper §7.6) and to human writing (validated on Tevet and Berant's worker-written McDiv and ConTest, paper §7.5).
2. *"Papers focusing on artefacts or benchmarks"* — the metric (and its open implementation) is the artefact.
3. The motivating paragraph of the workshop names *"the risk of idea homogeneity"* as a core concern. That's the phenomenon $D_{Ca_n}$ is designed to quantify.

Plus: the OLMo-2-7B post-training-pipeline result lands cleanly in this audience's interests (mode collapse from RLHF / DPO is a known concern for creative-writing applications), and the metric being policy-agnostic — the same pipeline scoring AI samples and human writing — directly mirrors the workshop's "Human-AI" framing.

## Honest caveats

- The workshop's center of mass is HCI/applied/design. The paper will be among the more mathematically dense submissions there.
- This is more likely a feature than a bug — the workshop benefits from a rigorous-measurement anchor — but it argues for extra expository polish.
- The bullet *"developing novel methods to improve creativity/diversity of generative models"* is **not** ours to claim. The paper measures diversity; it doesn't propose a training/decoding method that improves it. Claiming that bullet would invite reviews asking for diversity-improvement experiments we don't have. Lead with the *measurement* contribution; mention diversity-aware training as future work.

## Framing for the paper body

The current abstract is already well-tuned for this audience — it leads with mode collapse and decoding-strategy comparison, which are the right hooks. The framing work is mostly about what to expand in the body:

### Lead with the use case, justify with the math

Many readers in this audience will skim looking for *"what can I now measure that I couldn't before, and how do I use it?"* That should be answered in the introduction, with the information-theoretic derivation arriving as principled justification rather than as the headline. The PMI decomposition and progressive-conditioning derivations are correct and important, but they support the claim — they aren't the claim.

### Two anchor experiments: OLMo (AI side) and Tevet (Human side)

The workshop title is "Human-AI Co-Creativity," not "AI Creativity," so the Human side deserves equal billing with the AI side. Use both anchors, framed so each carries the credibility for its own half of the framing.

- **AI side — the OLMo-2-7B base $\to$ SFT $\to$ DPO $\to$ RLVR case study** (paper §7.6). A concrete instance of "post-training collapses creative diversity": 10 samples per prompt at each of the four stages, on AlpacaEval (200 prompts) and curated NoveltyBench (100 prompts), with three pre-registered one-sided paired Wilcoxon tests, Bonferroni-corrected, all significant at $p<0.001$ on AlpacaEval, monotone $D$ drop on both prompt sets, length-matched re-run confirms the drop is not a length artefact, cross-metric agreement with EAD and SentBERT (per-prompt scatter, `rlhf_experiment.tex` Figure~\ref{fig:rlhf-metric-scatter}; computed by `scripts/rlhf_experiment/4_score_baselines.py`) confirms it's the same diversity signal those baselines see. Expand into a full case study — figures, table, the pre-registration story, the public sample release.
- **Human side — Tevet and Berant's McDiv and ConTest** (paper §7.5). The credibility anchor for the claim that $C \times a_n$ tracks human judgments of creative-output diversity. McDiv and ConTest pair human-grounded diversity labels with a standardized comparison protocol (Spearman ρ, OCA) for ranking metrics against those labels. The labels come from human workers (response sets written by Mechanical Turk workers; for McDiv\_nuggets the high/low label comes from the construction protocol itself — a worker writes 5 different continuations for the high-diversity set, then paraphrases one of them 5 ways for the low-diversity set; see paper App C for the full protocol and a construction-confound caveat). $C \times a_n$ matches SentBERT-class baselines on McDiv prompt\_gen (ROC AUC at the level of the strongest embedding baseline). Lead with this framing: a metric for creative-output diversity should be tested against a human-grounded standardized eval, and that's exactly what McDiv and ConTest provide.
- **Bridge between the two halves — the cross-model comparison on NoveltyBench-curated** (frontier instruct ladder vs. the OLMo four stages, `rlhf_experiment.tex` Table~\ref{tab:cross-model}). Good second figure for the AI-side case study — within-pipeline post-training drops (~0.2 bits/byte across OLMo's stages) are large compared to dispersion across the entire frontier instruct cluster.
- **Supporting only — synthetic scenarios** (§7.2) and cross-model scaling on Llama 1B/3B/8B/70B (§7.4) are sanity checks for readers who want to know the metric does what it says — appendix, or one-sentence mentions in the body.
- The cross-model comparison on NoveltyBench-curated (frontier instruct ladder vs. the OLMo four stages, `rlhf_experiment.tex` Table~\ref{tab:cross-model}) is a good second figure for the OLMo case study — within-pipeline post-training drops (~0.2 bits/byte across OLMo's stages) are large compared to dispersion across the entire frontier instruct cluster.
- Synthetic scenarios (§7.2) and cross-model scaling on Llama 1B/3B/8B/70B (§7.4) are sanity checks for readers who want to know the metric does what it says — appendix, or one-sentence mentions in the body.

### Position against embedding- and n-gram-based metrics

The abstract already does this — *"both of which can conflate genuine content diversity with lexical or representational artifacts."* Worth expanding in related work and probably in a figure or table that makes the conflation concrete. This audience has used embedding-based diversity metrics and felt their limits firsthand; explicit examples of where $D_{Ca_n}$ gives a different answer than embedding distance will resonate.

### The two unique angles to lead with: single forward pass, and policy-agnostic across AI and human outputs

The conceptual advantages that distinguish $D = C \times a_n$ from embedding- and $n$-gram-based diversity metrics are twofold, and these are the angles to anchor the introduction and abstract on:

- **No auxiliary apparatus.** No embedding model, no reference corpus, no human labels, no similarity classifier — only per-token log-probs from a base $\theta$, in a single forward pass per permutation. This audience has used embedding-based diversity metrics and felt the "which embedding? trained on what? validated against whose labels?" tax firsthand.
- **Same metric for AI outputs and human writing.** Diversity is treated as a property of (responses, prompt, scoring model), not of how the responses were produced. The paper validates on both populations with the same pipeline: human-written response sets (Tevet and Berant's worker-written McDiv and ConTest, paper §7.5) and AI samples (the OLMo-2-7B post-training stages and the frontier-model NoveltyBench-curated ladder, paper §7.6). For a "Human-AI Co-Creativity" audience this is the framing that matches the workshop's title — a creativity researcher can use the same tool to score a writing class's drafts, a model's samples, or a hybrid human-AI pipeline.

### What to keep out of the workshop variant

- **No sigmoid/exponential extrapolation for $a_\infty$.** The current method takes $a_\infty \approx a_n$ — the last observed point of the permutation-averaged per-byte curve (paper §5.3, Figure 1 caption). Earlier drafts fit a parametric curve; that machinery is gone. Workshop prose must not describe a fitting step.
- **No DecTest framing.** Tevet's DecTest sets are LM-sampled at varying temperatures; sampling-temperature labels don't map onto creative diversity in any interesting way for this audience. The paper still reports DecTest numbers (paper §7.5, Table 6) for consistency with Tevet's framework, but the workshop variant should not lean on them.
- **No $E$ / $C \times E$ as a foil.** The headline metric is $D = C \times a_n$. The excess-entropy alternative is one of many possible scalars derivable from the $a_k$ curve, found inferior on Tevet, and lives in an appendix. The previous version of this doc framed the contribution as "what we don't claim about $E$"; that's the wrong frame. Frame the contribution positively as what $D = C \times a_n$ does. E is uninteresting and the fact it is uninteresting is uninteresting.
- **No PMI / chain-rule motivation, no $k_0 \propto m \ln 2$ theory, no cross-mode pairwise-matrix analysis in the main body.** All interesting (paper §3, Appendix A, §7.4 respectively); none headline material at 4–8 pages. Cite into appendices or one-line mentions.

## Submission logistics

- **Deadline:** 1 May 2026 AOE
- **Notification:** 15 May 2026 AOE
- **Format:** ICML 2026 LaTeX style, 4–8 pages main + unlimited references/appendices
- **Review:** double-blind, anonymize
- **OpenReview:** [ICML.cc/2026/Workshop/GenAICreativity](https://openreview.net/group?id=ICML.cc/2026/Workshop/GenAICreativity)
- **Non-archival:** so a journal submission of the full paper remains possible afterward
- **Reciprocal reviewing:** likely required — one author should be willing to review

## Risks worth tracking

- Workshop paper acceptance is uncertain even with strong topic fit. Co-Creativity is a first-iteration workshop, so we have no historical acceptance rate to anchor on.
- Reviewer pool will be a mix of HCI researchers, generative-AI applied researchers, and creativity-domain practitioners. The math density should be calibrated for that mix — a long appendix carrying the heavy derivation, with the main text staying readable, is probably the right structure.
- Since the workshop is non-archival, this is mostly a feedback-and-signaling exercise; the substantive publication target is the benchmarks journal you've mentioned. The workshop submission shouldn't compromise the journal version's structure.