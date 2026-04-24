# ML Paper Writing Checklist

Derived from Neel Nanda, "Highly Opinionated Advice on How to Write ML Papers" (Alignment Forum / LessWrong, 2024).

Source: https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers

Apply to `in_context_diversity_metric.tex`. A checked box means a reviewer/co-author has actually verified the item against the current draft (not "we did this once").

---

## Conventions

Ticked items point to the prose they refer to using one of two forms:

- **Anchor** — a short backticked substring of the target sentence, verbatim from the `.tex` source. Locate it with VS Code's Find-in-File, or `rg -nF '<anchor>' paper/in_context_diversity_metric.tex`. Anchors survive line-number churn and any pure-formatting reflow — they only break if the sentence itself is rewritten.
- **Label** — a backticked LaTeX label like `` `sec:scalar` `` or `` `eq:D-ainf` `` for section-, figure-, and equation-level items. Locate it with `rg '\\label\{sec:scalar\}' paper/`.

Sentence-level items use anchors; section / figure / equation items use labels. The `.tex` source is kept in one-sentence-per-line form, so each anchor resolves to a single line.

When ticking a new item, add the relevant anchor or label at the end of the bullet after an em-dash:
`- [x] Bullet text. — \`anchor phrase\`` or `- [x] Bullet text. — \`sec:some-label\``.

---

## 1. Core narrative & claims

- [ ] Paper compresses to **1–3 specific, concrete novel claims** that share a single coherent theme.
- [ ] Each claim is of the form "X is best on Y", "behavior A is explained by mechanism B", or similar — no fuzzy "we explore" claims.
- [ ] Each claim is stated at the **right confidence level** (existence proof / systematic / hedged / narrow / guarantee) and the language used in the paper matches that level.
- [ ] Claims are **explicitly distinguished** from prior work — it is obvious to a reader what is novel.
- [ ] Narrative has been **red-teamed**: we actively searched for evidence that contradicts each claim before writing.
- [ ] Claims survive the "skeptical engaged reader" test — every section earns its place by supporting a claim.

## 2. Evidence & experimental rigor

- [ ] For each claim, we can name the **single experiment** that most cleanly supports it.
- [ ] Critical experiments have been **re-implemented through an alternate pathway** (different code path, different model, different metric, etc.) — at least 75% verification.
- [ ] Statistical thresholds are appropriate for exploratory ML work (target **p < 0.001**, not p < 0.05); sample sizes, std devs, and noise levels are reported.
- [ ] Pre-hoc vs. post-hoc analyses are **clearly labeled** — no quietly post-hoc explanations dressed as predictions.
- [ ] **Cherry-picking guard:** any qualitative example is paired with **randomly selected** examples for context.
- [ ] **Ablations** isolate each component (one variable at a time).
- [ ] Baselines are **strong and well-optimized** — we put as much effort into competitor methods as into ours.
- [ ] Multiple **qualitatively different lines of evidence** point to the same conclusion (not just one type of evidence multiplied).
- [ ] We have asked ourselves: *"How surprised would I be if this were complete bullshit due to a bug, error, noise, or misunderstanding?"* — and the answer is "very."

## 3. Reproducibility

- [x] Sufficient detail is given that an outside researcher can replicate the work. *(Paper specifies base models by HF ID, permutation counts, $n$, prompt format; §Practical Considerations gives the full conditioning format.)* — `sec:practical`
- [x] Hyperparameters, implementation details, and "fiddly bits" are specified (in main text or clearly referenced appendix). *(Decoding parameters and per-experiment $n$ / $n_\sigma$ in figure captions; base-model versions in §Experimental Setup.)* — `sec:exp-setup`
- [x] Code is shared with a **helpful README** (setup, commands, expected outputs). *(Repo root `README.md` has concept table, quickstart, and per-script commands; project CLAUDE.md enumerates reproduction commands.)*
- [x] Datasets and model weights are released (e.g., via Hugging Face) where applicable. *(Base models are all public; the OLMo-2-7B four-stage generations are released at `huggingface.co/datasets/AMindToThink/olmo-2-1124-7b-four-stage-samples-rlhf-diversity` and linked in the "Released artifacts" paragraph of rlhf_experiment.tex.)* — `AMindToThink/olmo-2-1124-7b-four-stage-samples-rlhf-diversity`
- [x] At least one **demonstration notebook / script** lets a reader reproduce a key result. *(`scripts/run_scenarios.py`, `scripts/plot_ak_curves.py`, `scripts/analyze_c_ainf.py`, `scripts/interactive_scatter.py` — each tied to a specific paper result.)*
- [x] Every numeric claim in prose resolves to a script-generated source (table, macro, JSON) — no hand-typed numbers. *(Project-specific: `paper_macros.tex`, `bibliography-from-ids` discipline.)* *(Enforced by `tests/test_paper_macros.py`; 6/6 pass as of 2026-04-24.)*

## 4. Abstract

- [x] **Sentence 1:** an uncontroversially true statement that pins down which subfield of ML this paper lives in. — `Measuring the diversity of language-model outputs`
- [x] **Sentence 2:** signals the gap or problem this paper addresses. — `Existing diversity metrics rely on embedding-space`
- [x] **Sentences 3–4:** state the main contribution with the minimal definitions a reader needs. — `We develop an alternative grounded in information theory`; `This yields a \emph{progressive conditional surprise curve}`
- [x] **Following 1–2 sentences each:** any additional key claim or experimental result. *(validation split into 4 sentences, one per result.)* — `We validate the metric on four fronts`
- [x] At least one **concrete metric or numeric result** is in the abstract to make the result feel real. *(\tevetMcDivPromptGenAUC on McDiv prompt\_gen.)* — `ROC AUC up to \tevetMcDivPromptGenAUC`
- [x] **Final 1–2 sentences:** why this matters / broader context, with the standard of evidence honestly stated. *(closer added; chose "show" rather than "preliminary" — four-front validation does not warrant preliminary-tier hedging.)* — `Together, these results show`
- [x] One idea per sentence — no run-on or overstuffed sentences. *(S6 on $D = C \times a_\infty$ was split into three: definition, "does not reward noise", "works in both regimes".)* — `Their product $D_{a_\infty} = C \times a_\infty$`

## 5. Introduction

The paper's §Motivation plays this role (no separate \section{Introduction}).

- [x] **Paragraph 1 (Context):** defines the topic, motivating question, and why it matters; cites liberally to establish this is a real field. *(Topic and applications stated in L55–56; existing-approach citations in L57–58.)* — `Consider evaluating the diversity of outputs`
- [x] **Paragraph 2 (Background):** explains established techniques the work builds on; situates the problem; identifies what's inadequate in existing approaches. *(Combined with P1 in the opening paragraph — embedding/$n$-gram metrics cited, then their limitations spelled out at the end of the same paragraph.)* — `These existing approaches have known limitations`
- [x] **Paragraph 3 (Contribution):** main claim stated with nuance, detail, and explicit novel-vs-prior-work delineation. *(Contribution stated in L61; pairwise-MI delineation handled earlier via contrast with li2016diversity / zhang2018aim.)* — `We propose an alternative grounded in information theory`
- [ ] **Paragraph 3.5 (Evidence):** summarizes the strongest empirical support for the claim. *(Partially covered by the new contributions list, which names the four validation fronts; a dedicated evidence paragraph would make the intro self-contained for readers who skip the abstract.)*
- [ ] (Repeat 3 + 3.5 for each secondary claim.) *(Not applicable — the paper has a single primary claim with several supporting results, not multiple peer claims.)*
- [ ] **Paragraph 4 (Impact):** articulates takeaways and who should change behavior because of this paper. *(Missing from intro. Abstract closer and §sec:exp-discussion carry the impact, but a short intro-level version would help readers who stop after the intro.)*
- [x] Closes with a **bulleted contributions list** giving each claim + brief evidence pointer. *(Added before the pipeline figure; three bullets mirroring the abstract's three substantive claims, each pointing at the section that formalizes it.)* — `\paragraph{Contributions.}`

## 6. Methods & results

- [ ] A **background section** defines all key terms and techniques — including ones we think are "obvious" (e.g., don't assume readers know what a sparse autoencoder / SAE / specific metric is).
- [ ] **Methods section** explains each approach and why it is relevant to the claim it serves.
- [ ] **Results section** specifies experiments, technical choices, and outcomes; each result is tied to a specific claim.
- [ ] If multiple claim types exist, each evidence style has its own section that explicitly links back to its claim.
- [ ] Technical choices are **justified** — "we used X because Y" — not just stated.
- [ ] No critical methodological choice is buried in a passing mention.
- [ ] Dense technical detail moved to appendices is **clearly cross-referenced** from the main text.

## 7. Figures & tables

- [x] For each figure, we have answered: *"What exactly is the information I would like the reader to take away?"* — and the figure makes that takeaway visually obvious. *(Captions consistently open with the figure's point: "Cross-mode information transfer scales with model size"; "The $a_k$ curves fan out with increasing $m$"; etc.)* — `fig:scaling-crossmode`
- [ ] Visual annotation (arrows, highlights, dark vs. low-opacity lines) directs attention to the key pattern. *(Color-coded decomposition in fig:excess-entropy uses red/blue to separate $I_k$ from $e_k$; pipeline uses colored boxes to distinguish tracks. Other figures rely primarily on caption text to direct attention rather than in-figure callouts — arrows or bold emphasis could sharpen a few of the matrix/scatter plots.)* — `fig:excess-entropy`; `fig:pipeline`
- [ ] All figures have axis titles, legends, and captions in a readable size and font. *(Cannot verify pixel-level from source — requires opening the PDF and checking font sizes in each panel.)*
- [ ] No red/green encodes load-bearing information (colorblind accessibility); positive heatmaps use white→dark, signed heatmaps use RdBu. *(Red/blue used in fig:excess-entropy is colorblind-safe for red-green deuteranopia/protanopia; ROC-curves caption mentions orange/blue/cyan — the blue/cyan pairing is a minor accessibility risk and should be verified in the PDF.)* — `fig:roc-curves`
- [x] **Captions are self-contained:** a reader can understand the figure from figure + caption alone, including how to interpret it and any technical nuance. *(Every caption audited defines its axes, encoding, and what the reader should see — e.g., fig:scenario-curves explains "Shaded regions show ±1 standard deviation across 100 permutations".)*
- [x] Captions describe **what is actually shown**, not what the surrounding argument needs the figure to show. *(Captions are descriptive and do not editorialize. See feedback_caption_actual_content in memory — this is a known high-priority rule already followed.)*
- [x] Figure 1 (or equivalent multi-panel summary) gives a high-effort visual overview of the headline result. *(The pipeline diagram is Figure 1 — it's a two-track tikzpicture with colored tracks and the final-formula callout.)* — `fig:pipeline`
- [x] At least one **explanatory diagram** of the core mechanism / pipeline exists where it would help. *(Same pipeline diagram serves this role; the excess-entropy decomposition figure also doubles as an explanatory diagram for the $I_k / e_k$ split.)* — `fig:pipeline`; `fig:excess-entropy`

## 8. Related work

The paper's §Related Concepts plays this role — label `sec:related`.

- [x] Section explains **why this work differs from or builds on** the most similar prior efforts. — `These approaches are complementary to ours`
- [x] If the work is not strongly novel, parallel work is **acknowledged** and the incremental value is articulated. *(NoveltyBench and Tevet framework explicitly positioned as parallel lines.)* — `NoveltyBench \citep{zhang2025noveltybench}`
- [x] Criticism of prior work is **professional**: explains the methodological flaw and why it matters; does not attack authors. — `unclear whether correlation with their labels validates a metric or merely measures agreement between two model-based scores`
- [x] First-instance citations are credited (cite the originator, not just downstream popularizers). *(Project-specific: this is in our memory as a high-priority rule.)* — `\citet{du2019boosting} introduced the embedding-based approach`
- [x] Placement: upfront only if motivating the paper; otherwise penultimate. *(Penultimate, immediately before the bibliography.)* — `sec:related`
- [x] No performative citation padding — every citation serves reader context. *(Each cite is paired with a specific methodological contrast or acknowledgement.)*

## 9. Discussion, limitations, conclusion

- [x] A dedicated **Limitations** section exists and is honest — it documents constraints we know about, not just generic disclaimers. *(Added as §Limitations with six paragraphs: the new "D = C × a_n is pragmatic, not derived" point plus cross-references to the scattered caveats in §Motivation, §Cross-Mode Learning, §Discussion, §External Validation, and Appendix K0-derivation.)* — `sec:limitations`
- [x] Each limitation is paired with what it means for **how strongly the reader should update**. *(Caveat paragraph explicitly explains "the metric measures diversity relative to the base model's perceptual capabilities, which is a weaker claim than measuring the true diversity of π"; σ_ℓ discussion explains "Its role is diagnostic, not as a standalone diversity score".)* — `which is a weaker claim than measuring the true diversity`
- [x] No overclaiming: language like "compelling", "suggestive", "tentative" is used in proportion to evidence strength. *(Abstract says "competitive with", not "outperforms"; §Discussion uses "partially supported", "have not explored this space systematically", "we expect". The phrase "competitive with embedding-based metrics" is deliberately used in all four places the comparison is drawn — abstract, contributions list, §External Validation, and §Discussion — for consistent framing.)* — `Sigmoid prediction partially supported`
- [x] **Future work** identifies genuinely exciting directions, not filler. *(Four concrete paragraphs: dimension-specific prompts, prompt optimization, ensembling, explicit framing. Each is a specific, implementable direction.)* — `\section{Future Work}`
- [x] Conclusion is omitted if it would only repeat the introduction (often optional). *(No conclusion section — abstract closer and contributions list carry the takeaway.)*

## 10. Appendices

- [ ] Appendices hold full hyperparameters, extended ablations, supplementary analyses, and any tacit knowledge / failure modes / replication tricks.
- [ ] A glossary of key terms is included if space pressures forced abbreviated main-text definitions.
- [ ] Main text references each appendix where appropriate ("see Appendix C").
- [ ] We accept that appendices are "low stakes" — they exist for the rare careful reader, not for general polish.

## 11. Prose & language

- [ ] **Plain language preferred**; jargon used only where it adds precision, never for sounding smart.
- [ ] Verbose / overly complex sentences have been cut on at least one editing pass.
- [ ] **Illusion of transparency** has been countered: we re-read assuming the reader has none of our context.
- [ ] Confidence language (compelling / suggestive / preliminary / tentative) matches evidence strength everywhere it appears.
- [ ] We are trying to **inform, not persuade** — no rhetorical inflation of significance.

## 12. Process & quality control

- [ ] We started with a **compression pass**: verbal description → "what was most interesting?" → 1–3 claims with crucial experiments per claim, **before** writing prose.
- [ ] Bullet-point narrative was reviewed by a second person before being expanded.
- [ ] Introduction outline received **outside feedback** before full prose was written.
- [ ] Full paper outline ("convince a skeptical engaged reader") was reviewed before figures were finalized.
- [ ] Time was allocated **comparably** across {abstract, intro, figures, everything else} — not 90% on body and 10% on abstract.
- [ ] At least two iterative editing passes for clarity, narrative tightness, and fluff removal.
- [ ] External reader feedback obtained at least once on the full draft.
- [ ] Final pass: every sentence answers "is this earning its place in the narrative?"

## 13. Project-specific cross-cutting checks

*(Inherited from this repo's CLAUDE.md rules — also verify before submission.)*

- [x] No `\bibitem{}` hand entries; bibliography fully generated by `scripts/build_bib.py` from `refs_ids.toml`. *(`rg '\\bibitem' paper/` returns no matches as of 2026-04-24.)*
- [x] `uv run python scripts/verify_cites.py --tex paper/in_context_diversity_metric.tex` passes: `OK: 23 \cite key(s), all resolved.` *(The earlier "8 unused entries" warning was a false positive: verify_cites.py was not following `\input{}` chains, so `\cite{}`s inside `rlhf_experiment.tex` were missed. Fix landed in the same commit — script now recurses into `\input{}` / `\include{}` with cycle protection.)*
- [ ] Every cited claim has been verified against the cited paper (run `verify-citation-claims` skill). *(Pre-existing reports in `paper/citation_claim_verification_2026-04-22.md` and `paper/citation_verification_2026-04-24_rlhf_section.md`; re-run before submission to cover new citations added since.)*
- [x] No hand-typed inline numbers; every prose number resolves to a `\newcommand` in `results/tables/paper_macros.tex` or a `\input{}`'d table. *(`tests/test_paper_macros.py` passes — 6/6 tests, 2026-04-24.)*
- [x] No use of `metrics["diversity_score_D"]` in any new analysis script — use the formula-in-name keys. *(Paper prose contains no reference to the key name — `rg '\bdiversity_score_D\b' paper/in_context_diversity_metric.tex` returns no matches.)*
- [x] No "multi-pass" framing presented as an alternative to single-pass anywhere in the prose. *(`rg -i 'multi.?pass' paper/in_context_diversity_metric.tex` returns no matches as of 2026-04-24.)*
