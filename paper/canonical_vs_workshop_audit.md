# Canonical vs Workshop Audit

Generated 2026-04-27. Compares `paper/in_context_diversity_metric.tex` (canonical
journal-track wrapper) against `paper/main_icml_workshop.tex` (Human-AI
Co-Creativity @ ICML 2026 workshop wrapper). Both built from the shared
`paper/sections/*.tex` source pool, with five workshop-specific variants
(`*_workshop.tex`) overriding their canonical counterparts.

## Snapshot

| Dimension | Canonical | Workshop |
|---|---|---|
| Document class | `article` (11pt, 1in margin) | `icml2026` (loaded over `article`) |
| Layout | single column | two column (via `\twocolumn[...]`) |
| Authors | named: Khoriaty / Williams-King / Feng | `Anonymous Authors` (double-blind) |
| Affiliations | ERA Cambridge / GW University | `Anonymous Institution` |
| Title | "A Diversity Metric for LLM Outputs via Progressive Conditional Surprise of a Base-Model" | "I've Seen How This Goes": Characterizing the Diversity of LLM Generations and Human Writing via Progressive Conditional Surprise |
| Page count | 44 pages | 25 pages (main body 7) |
| Cite keys cited | 23 unique | 23 unique (identical set) |
| Acknowledgements | included | omitted |
| Project URLs | real GitHub + HF dataset | placeholders pending anonymous mirrors (see `workshop_submission_todos.md`) |
| Bib style | `plainnat` | `icml2026` |
| PDF metadata title | (empty) | full Unicode-curly-quoted title |

## What both wrappers include identically

Both wrappers `\input{}` the same shared section files for these (no
content delta — byte-identical prose modulo float placement controlled
by the wrapper's column count):

- `02_setup.tex` (notation)
- `06_practical.tex` (practical considerations)
- `07_experiments_intro.tex`
- `07_2_scenario_validation.tex`
- `07_3_mode_count.tex`
- `07_4_cross_mode.tex`
- `07_7_practical_findings.tex`
- `07_8_discussion.tex`
- `08_limitations.tex`
- `09_future_work.tex`
- `10_related.tex`
- `appA_k0_derivation.tex`
- `appB_excess_entropy.tex`
- `appC_mcdiv_confounds.tex`
- `appD_aggregation.tex`
- `appE_qwen3_comparison.tex`

## Section-by-section: what's where

| Section file | Canonical | Workshop | Notes |
|---|---|---|---|
| `abstract` / `abstract_workshop` | main body | main body (variant) | workshop is 164w (canonical 303w, −46%) |
| `01_motivation` / `01_motivation_workshop` | main body | main body (variant) | workshop is 766w (canonical 1059w, −28%) |
| `02_setup` | main body | main body | shared |
| `03_progressive_conditioning` | main body | **dropped from main body** | folded into workshop's `03_method_workshop` |
| `04_coherence` | main body | **dropped from main body** | folded into workshop's `03_method_workshop` |
| `05_reporting` | main body | **dropped from main body** | partial fold into `03_method_workshop`; rest dropped |
| `03_method_workshop` (NEW) | not used | main body | combined +compressed §3+§4+§5 (706 words; the three canonical sources sum to 2352w, −70%) |
| `06_practical` | main body | **appendix** | |
| `07_experiments_intro` | main body | **appendix** | |
| `07_2_scenario_validation` | main body | **appendix** | |
| `07_3_mode_count` | main body | **appendix** | |
| `07_4_cross_mode` | main body | **appendix** | |
| `07_5_tevet` / `07_5_tevet_workshop` | main body | main body (variant) | workshop is 576w (canonical 1065w, −46%); demotes DecTest to a single paragraph; promoted to `\section{}` |
| `07_6_rlhf` / `07_6_rlhf_workshop` | main body | main body (variant) | workshop is 962w (canonical 1440w, −33%); promoted to `\section{}`; trims cross-model comparison |
| `07_7_practical_findings` | main body | **appendix** | |
| `07_8_discussion` | main body | **appendix** | |
| `08_limitations` | main body | main body | shared |
| `09_future_work` | main body | **appendix** | |
| `10_related` | main body | main body | shared |
| `acknowledgements` | main body | **omitted entirely** | double-blind requirement |
| `appA`–`appE` | appendix | appendix | shared |

Workshop drops 9 canonical main-body sections (or, more precisely, 7 of
them go to the workshop appendix; 3 are folded into the new
`03_method_workshop` variant; 1 is dropped entirely — acknowledgements).

## Per-variant content deltas

These are the only files where workshop prose differs from canonical
prose (i.e., the only places to audit for substantive content drift).

### `abstract_workshop` (vs `abstract`)

- **Lead changes**: opens with the use case ("creativity researchers
  routinely need to compare diversity") instead of the field-pinning
  sentence. Drops the canonical's "many- and few-draws regimes" framing
  (a holdover from the historical $E$-vs-$D$ comparison; workshop is
  $D_{Ca_n}$-only).
- **Anchor numbers**: same OLMo monotone-drop story and Tevet AUC, but
  reduced to one sentence each.
- **Word count**: 164w (canonical 303w, −46%).

### `01_motivation_workshop` (vs `01_motivation`)

- **Opening paragraph rewritten** to lead with creativity-relevant use
  cases (post-training stages, decoding strategies, prompting
  interventions, human-AI hybrid pipelines). Foregrounds the
  policy-agnostic claim ("scores AI samples and human-written response
  sets through the same pipeline").
- **Pipeline figure (`fig:pipeline`)**: same TikZ diagram, same caption.
- **Contributions list**: dropped (compressed into one sentence in the
  opening paragraph).
- **PMI/MMI prior-work contrast**: present in both (canonical has a full
  paragraph, workshop has a single-sentence pointer added in commit
  `e3782ab` to keep `li2016diversity` and `zhang2018aim` cited).
- **"Evidence preview" / "Who should care" paragraphs**: present in
  canonical, dropped in workshop.
- **Word count**: 766w (canonical 1059w, −28%).

### `03_method_workshop` (combines `03_progressive_conditioning` + `04_coherence` + `05_reporting`)

- **Structure**: three subsections (Progressive Conditional Surprise /
  The Coherence Term / The Diversity Score) instead of three separate
  top-level sections.
- **Equation labels preserved**: `eq:ak`, `eq:coherence`, `eq:D-Can`,
  plus section labels `sec:progressive-conditioning`, `sec:coherence`,
  `sec:scalar` — cross-references from shared files (e.g.
  `08_limitations`, `10_related`) still resolve.
- **Edge-case table**: included (the same 5-row content as canonical's
  edge-case enumeration, converted to a `table[t]`).
- **Pile scale-anchor footnote**: present (commit `636bf21`); canonical's
  parallel `\paragraph{Typical range of $C$}` was reframed identically
  in commit `acd038d`. Both wrappers carry the same corrected sentence
  inline.
- **Dropped from canonical sources**: PMI/chain-rule motivation
  (canonical §3.1, §3.3), the $(a_n, C, \sigma_\ell)$-triple discussion
  (canonical §5), the coherence-heterogeneity subsection (canonical
  §5.4), the diversity uncertainty band (canonical §5.4), the
  "What to Report" subsection (canonical §5.5).
- **Word count**: 706w; canonical sources sum to 2352w (−70%).

### `07_5_tevet_workshop` (vs `07_5_tevet`)

- **Promoted from `\subsection`** (under the canonical `\section{Experiments}`)
  to `\section{}` in the workshop wrapper, since the workshop has no
  parent §7 Experiments umbrella.
- **Lead reframed**: opens with what McDiv/ConTest *are* (human-grounded
  diversity labels + standardized comparison protocol) rather than with
  Tevet's eval-of-evals framework.
- **DecTest demoted**: kept as a one-paragraph "for completeness" pointer
  rather than a co-equal evidence stream.
- **ConTest table**: kept as `table*[t]` (workshop) to span both columns.
- **Construction-confound caveat**: kept as one-sentence pointer to
  Appendix C.
- **Word count**: 576w (canonical 1065w, −46%).

### `07_6_rlhf_workshop` (vs `07_6_rlhf`)

- **Promoted to `\section{}`** (same reason as Tevet variant).
- **One-paragraph framing intro** added that positions OLMo as the
  AI-side anchor parallel to Tevet's human-side anchor.
- **Pre-registered hypotheses (H1a/H1b/H1c with Bonferroni)**: kept.
- **Length-matched re-run + Pearson/Spearman**: kept as a `figure*[!htbp]`.
- **AlpacaFarm prompt-source citation**: present (commit `546d177` —
  was an oversight in the original workshop trim).
- **Cross-model comparison** (canonical's full table + discussion):
  trimmed to a one-sentence pointer ("we defer the full cross-model
  table to a longer venue") so the cross-model table doesn't have to
  appear in the workshop.
- **Released artefacts**: HF URL replaced with double-blind placeholder
  via `\projectHfDatasetUrl` macro (real URL kept in canonical wrapper).
- **Word count**: 962w (canonical 1440w, −33%).

## Citations

Both wrappers cite the **same 23 unique keys**, all resolved against
`refs.bib`. The set is identical because the workshop appendix
includes every shared section that introduces a unique key (e.g.
`crutchfield2003regularities` is cited in `10_related`, which is shared;
`du2019boosting` in motivation, which both wrappers cover via their
respective variants; `gao2020pile` in the Pile sentence, which is now
mirrored across both wrappers; `dubois2024alpacafarm` in `07_6_rlhf_workshop`
and `07_6_rlhf` for AlpacaEval).

`verify_cites.py --tex paper/<wrapper>.tex` reports `OK: 23 \cite key(s),
all resolved` for both, with zero unused entries in `refs.bib`.

## Anonymization deltas (workshop only)

- `\projectGithubUrl` macro:
  - Canonical: `\url{https://github.com/AMindToThink/icl-diversity}`
  - Workshop: `[code link withheld for double-blind review; an anonymous mirror is included in supplementary materials]`
- `\projectHfDatasetUrl` macro:
  - Canonical: `\url{https://huggingface.co/datasets/AMindToThink/olmo-2-1124-7b-four-stage-samples-rlhf-diversity}`
  - Workshop: `[dataset link withheld for double-blind review; an anonymous mirror is included in supplementary materials]`
- `acknowledgements.tex`: not `\input`'d by the workshop wrapper.
- Author block: replaced with `\icmlauthor{Anonymous Authors}` /
  `\icmlaffiliation{anon}{Anonymous Institution}`.
- `pdftotext` of the workshop PDF returns zero hits for any of:
  `Khoriaty`, `Williams-King`, `Shi Feng`, `ERA Cambridge`,
  `George Washington`, `AMindToThink`.

## Float-layout deltas

The same figures and tables appear in both wrappers, but the workshop
re-routes some of them between `figure`/`table` (single-column) and
`figure*`/`table*` (spanning both columns) based on natural width.
Concretely (from commit history `770c13e` + `ef344f8`):

- **Span both columns** in workshop: pipeline diagram (`fig:pipeline`),
  scenario-validation 15-col metrics table, side-by-side cross-mode
  pairwise heatmaps (Qwen / GPT-2), full-textwidth per-token figures,
  Qwen3-comparison 7-col table, RLHF four-stage subfigure pair.
- **Single column** in workshop (collapsed back from the over-aggressive
  initial sweep): small comparison tables, single-axis figures with
  width ≤ 0.6\textwidth, the pairwise-symmetry scatter, the small
  confound tables in Appendix C.

The canonical doesn't make this distinction (single column throughout).

## What to read first if you only have 10 minutes

1. **Workshop main body** (pages 1–7 of `main_icml_workshop.pdf`):
   abstract → motivation → method → Tevet → OLMo → limitations → related.
   This is the actual submission content.
2. **The 5 workshop-variant files** under `paper/sections/*_workshop.tex`:
   these are the only places where prose differs from canonical. Every
   other rendered word in the workshop PDF is byte-identical to canonical
   prose.
3. **`paper/workshop_submission_todos.md`**: the open admin items
   (anonymous code mirror, HF dataset anonymization strategy, OpenReview
   profile) that need to be completed before submission.

## What is NOT yet done (open follow-ups)

From `paper/workshop_submission_todos.md`:

- Anonymous code mirror at anonymous.4open.science (interactive sign-in
  required; not scriptable).
- HF dataset anonymization (decision pending on dataset size).
- OpenReview author profile + reciprocal-reviewing willingness flag.
