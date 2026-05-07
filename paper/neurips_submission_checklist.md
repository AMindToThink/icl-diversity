# NeurIPS 2026 Submission Checklist (Main Track)

Live tracker for adapting `paper/main_icml_workshop.tex` (ICML workshop variant) into a NeurIPS 2026 main-track submission.
Deadline: **Full paper + supplementary, May 6, 2026 AOE** (today).

## Sources (read these first before any edit)

- Call for Papers: <https://neurips.cc/Conferences/2026/CallForPapers>
- Main Track Handbook: <https://neurips.cc/Conferences/2026/MainTrackHandbook>
- Paper Checklist guidelines: <https://neurips.cc/public/guides/PaperChecklist>
- Reviewer Guidelines: <https://neurips.cc/Conferences/2026/ReviewerGuidelines>
- Ethics Guidelines: <https://neurips.cc/public/EthicsGuidelines>
- Dates: <https://neurips.cc/Conferences/2026/Dates>
- Evaluations & Datasets Track FAQ: <https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ>
- Evaluations & Datasets Call: <https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets>
- AI-Assisted Reviewing Experiment opt-out: <https://neurips.cc/Conferences/2026/ai-reviewing-experiment>
- Contribution-types blog: <https://blog.neurips.cc/2026/04/16/a-choice-of-contribution-types-at-neurips-2026/>
- April 2026 newsletter (OpenReview profile timing, Paper Assistant Tool): <https://blog.neurips.cc/2026/04/22/neurips-newsletter-april-2026/>
- Responsible Reviewing Initiative (reciprocal-review enforcement): <https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/>
- Style files (download): <https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip>
  - Contains `neurips_2026.sty`, `neurips_2026.tex` (template), `checklist.tex` (mandatory paper checklist).
- Overleaf mirror of the template: <https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc>
- Author FAQ (Google Doc): <https://docs.google.com/document/d/15vokOMKgjMyUr230BLfeBHkaTIyCarDY3X_4T83VqDc/edit>
- Underleaf summary: <https://www.underleaf.ai/templates/neurips>
- Serre-AI submission guide: <https://github.com/serre-ai/research/blob/main/docs/submission-guides/neurips-2026.md>

## Deadlines

- [x] Abstract submission: **May 4, 2026 AOE** (already submitted by the user before this conversation)
- [ ] Full paper + all supplementary: **May 6, 2026 AOE** (TODAY)
- Author response/decisions: notification September 24, 2026 AOE (post-submission, no action now)
- [x] All authors must have an OpenReview profile updated *by the abstract deadline*; after that, no author additions, only order changes. **Confirmed 2026-05-06.**

## Hard desk-rejection triggers (verify before submit)

- [x] **Page limit ≤ 9 pages.** Body ends mid-page-8 with Impact Statement; references start on page 8. ~1.5 pages of headroom under the 9-page limit. Verified 2026-05-06.
- [x] **No margin / font-size tweaks.** `paper/neurips_2026.sty` is the official upstream file copied verbatim from the NeurIPS 2026 zip; no edits.
- [x] **Anonymous.** `pdftotext paper/main_neurips.pdf - | grep -iE "matthew|khoriaty|northwestern|@gmail|@u\.northwestern|amindtothink"` returns empty. Citations to advisor's other-author work appear (Shi Feng on `qiu2026selfimprovement` and `wen2025unsupervised`) but the advisor is not a coauthor — not an anonymity violation.
- [x] **Self-citations in 3rd person.** Section files use `\citet{}` / `\citep{}` consistently and never write "our previous work"; this paper has no concurrent submissions to anonymize.
- [x] **PDF metadata stripped.** `pdfinfo paper/main_neurips.pdf` shows blank `Title:` and `Author:` (Creator/Producer are LaTeX/pdfTeX defaults, not author-identifying).
- [x] **Paper checklist appendix included with answers filled in.** `paper/checklist.tex` ends `\input{}`'d in `main_neurips.tex` after the appendix, all 16 questions answered (`grep -c answerTODO checklist.tex` = 0; "NeurIPS Paper Checklist" present in PDF).
- [x] **Single PDF, 3.6 MB** (≤ 50 MB limit).
- [ ] **Supplementary code/data zip ≤ 100 MB.** Decision needed: bundle `results/rlhf_experiment/` + code into a ZIP, or rely solely on the anonymous mirror URL `https://anonymous.4open.science/r/icl-diversity-67E6/`. The URL approach is fine but less robust because some reviewers don't click out.
- [x] **Not a dual archival submission.** ICL-diversity ICML workshop paper is a non-archival workshop submission (workshops in the human-AI co-creativity track are non-archival per ICML's own policy); does not block this NeurIPS submission.

## Track / contribution type to select on the OpenReview form

> **Decision: Main Track** (locked in by the already-filed Main Track abstract).
> The E&D track would also have been a defensible fit for a metric paper, but **the E&D abstract deadline was May 4, 2026 (also AOE), and is past as of today**. The E&D FAQ further states: "there will be no possibility to switch tracks or types and that papers cannot be submitted to multiple tracks or types simultaneously." So Main Track is the only option remaining.

- [x] Track confirmed: **Main Track** (abstract submitted 2026-05-04). Style file is `\usepackage{neurips_2026}` with no options.
- [ ] Pick contribution type on the OpenReview form: **General** (default for empirical work) or **Use-Inspired** (novel metric for real-world LLM creative-writing evaluation).

## Formatting / wrapper changes from ICML workshop variant

- [x] Style files copied: `paper/neurips_2026.sty`, `paper/neurips_2026.tex`, `paper/checklist.tex`, plus TinyTeX-missing dependencies `environ.sty`, `lineno.sty`, `trimspaces.sty`.
- [x] `paper/main_neurips.tex` created with:
  - [x] `\documentclass{article}` + `\usepackage{neurips_2026}` (no track option — gives anonymous + line numbers).
  - [x] No `\icmltitle{}` / `\icmlauthor{}` / `\twocolumn[...]` / etc. The only `icml` substring left is in the file header comment "Adapted from main_icml_workshop.tex".
  - [x] `\bibliographystyle{plainnat}`.
  - [x] Kept: `cmap`, `T1 fontenc`, `microtype`, `graphicx`, `subcaption`, `booktabs`, `hyperref`, `amsmath`, `amssymb`, `amsthm`, `enumitem`, `pgfplots`, `tikz`, `\input{../results/tables/paper_macros.tex}`.
  - [x] Anonymous author block (`\author{Anonymous Authors}`; `neurips_2026.sty` default anonymizes the title block at submission).
  - [x] `\input{checklist.tex}` after the appendix, on a new page.
- [x] Reuses existing `paper/sections/*.tex` files. Sections use `\citep`/`\citet` consistently. Cross-reference fixes applied: `Section~\ref{sec:related}` → `Appendix~\ref{sec:related}` in `01_motivation_workshop.tex` and `checklist.tex` (Related Work was moved to the NeurIPS appendix).

## Paper checklist (16 questions; answers in `checklist.tex`)

All 16 answered (zero `\answerTODO` remaining). Final answers as written into `paper/checklist.tex`:

- [x] **1. Claims** — `\answerYes{}` (Abstract + §1 Motivation).
- [x] **2. Limitations** — `\answerYes{}` (§Limitations covers θ-relativity, metric-selection bias on Tevet, length-matching, untested-reweighting honesty).
- [x] **3. Theory, assumptions, proofs** — `\answerNA{}` (empirical paper; informal derivations live in Appendix `app:excess-entropy`).
- [x] **4. Experimental result reproducibility** — `\answerYes{}` (§Method, run scripts, anonymous mirror).
- [x] **5. Open access to data and code** — `\answerYes{}` (`https://anonymous.4open.science/r/icl-diversity-67E6/`).
- [x] **6. Experimental setting / details** — `\answerYes{}` (Sec 4 Tevet, Sec 5 RLHF, length-matching protocol).
- [x] **7. Statistical significance** — `\answerYes{}` (permutation bands; Bonferroni-corrected paired tests in Table 4 / §5).
- [x] **8. Compute resources** — `\answerYes{}` (forward-passes-only, A100/H100-class for Qwen2.5-3B, multi-GPU `device_map="auto"` for Qwen3-30B-A3B-Base).
- [x] **9. Code of Ethics** — `\answerYes{}` (public models, public benchmark, no human subjects, no scraping, no PII).
- [x] **10. Broader impacts** — `\answerYes{}` (§Impact Statement).
- [x] **11. Safeguards** — `\answerNA{}` (no high-risk model release; the released artefacts are metric code + sampled OLMo-2 generations).
- [x] **12. Licenses** — `\answerYes{}` (canonical citations to GPT-2 / Qwen / OLMo-2 / Tevet, all permissive).
- [x] **13. New assets** — `\answerYes{}` (code + the 200+100 prompt × 4 stage × K=10 generation dataset under `results/rlhf_experiment/`, README-documented).
- [x] **14. Crowdsourcing / human subjects** — `\answerNA{}` (use Tevet & Berant's existing human ratings; no new collection).
- [x] **15. IRB approvals** — `\answerNA{}` (no human-subjects research conducted in this work).
- [x] **16. LLM usage in the method** — `\answerYes{}` (base model θ is the central object of the proposed method; per-experiment choice and ablations documented).

## Optional content choices (minor; discuss before applying)

- Keep "creativity" framing per user — explicitly: not a priority to remove for submission. NeurIPS audience cares more about the metric itself, but the framing is intelligible and may even broaden appeal.
- Consider relabeling §"Conclusion" → "Discussion and Conclusion" to satisfy reviewers who want explicit discussion (low-cost edit).
- Consider promoting §07_2_scenario_validation (synthetic-validation table) into the main body if space permits — reviewers like to see sanity checks before benchmark numbers.
- If main text overflows 9 pages: drop the synthetic-validation paragraph from the main body (it is already in the appendix), tighten the related-work paragraph, or move one of the secondary RLHF figures to appendix.

## Build / verification

- [x] `cd paper && latexmk -pdf main_neurips.tex` succeeds with zero unresolved citations and zero undefined references.
- [x] `pdfinfo paper/main_neurips.pdf` — 40 total pages; main body fits within 9 (body ends mid-page-8 before References).
- [x] Inline numbers in the body are macro-imported from `results/tables/paper_macros.tex` (now 136 macros including the new 10 `scenario*DCan` macros for Table 1). `uv run pytest tests/test_paper_macros.py` — 6/6 pass.
- [ ] **Open the PDF in pdf.js** (the renderer used by `anonymous.4open.science`) and visually verify text renders without kerning/glyph issues. Quick way: drag-drop into Firefox or Chrome which use pdf.js by default.
- [ ] `uv run python scripts/verify_cites.py` passes — *currently fails* because the script targets the archived `paper/main.tex` path. **Action:** either point the script at `paper/main_neurips.tex` or update it to scan all `paper/main_*.tex` files. Lower priority than the latexmk verification above (which already confirmed citation resolution at build time).

## Author obligations on the OpenReview form (don't miss)

- [ ] **Reciprocal reviewer nomination.** Every Main Track submission must nominate at least one author who will review for NeurIPS 2026; persistent low-quality reviewing can desk-reject the submitter's own paper at the meta-review stage (see Responsible Reviewing Initiative). Decide which coauthor will be the nominated reviewer before submitting.
- [ ] **OpenReview profiles** for every coauthor must be activated and up-to-date *before the abstract deadline (May 4, AOE)*; profiles created within ~2 weeks of the abstract deadline may not clear moderation in time. (User confirmed abstract is already filed, so all profiles should be live; verify each coauthor's profile shows the correct affiliation history.)
- [ ] **Conflict-of-interest declarations.** OpenReview tracks two CoI types: (1) Domain (last 3 years of education/career — public on profile), (2) Personal (family, advisors, all coauthors on original research articles in the last 3 years). False CoI declarations can cause removal from the system and rejection of all submitted papers; cross-check that every coauthor's CoI list is correct.
- [ ] **AI-assisted reviewing experiment.** NeurIPS 2026 runs an AI-assisted reviewing pilot. Authors can opt out at submission time via a checkbox on the OpenReview form; decide whether to participate before submit.
- [ ] **No sub-reviewers.** Reviewing duty cannot be delegated to non-author colleagues.

## Final submission steps (before pressing Submit)

- [x] Anonymity grep returns nothing.
- [x] PDF metadata `Author:` / `Title:` blank.
- [x] No "undefined reference" warnings in the LaTeX log.
- [x] Paper checklist in PDF; zero `\answerTODO{}` remain.
- [ ] **(Decide)** Bundle supplementary code+data ZIP from the anonymous mirror, or rely solely on the anonymous-URL link in the PDF? URL is currently in the body via `\projectGithubUrl`. ZIP is more robust (some reviewers won't click out); URL is simpler.
- [x] Dual-submission status confirmed: ICL-diversity ICML workshop is non-archival; does not block.
- [ ] **(User action)** Upload `paper/main_neurips.pdf` (+ supplementary ZIP if bundled) on OpenReview Main Track before **May 6, 2026 AOE** (today). No separate supplementary deadline.

## Pre-submission must-do list (single page; check these one by one)

1. [x] PDF builds clean. `paper/main_neurips.pdf` is current as of the last edit.
2. [x] Body ≤ 9 pages.
3. [x] Anonymity verified.
4. [x] Checklist filled in.
5. [x] Inline numbers macro-imported (no hand-typed digits).
6. [ ] **Visual pdf.js render check.** Open the PDF in Firefox/Chrome — confirm no broken kerning / missing glyphs.
7. [ ] **Pick a contribution type** on the OpenReview form (recommended: General; Use-Inspired is the alternative).
8. [ ] **Nominate the reciprocal reviewer** — pick which coauthor will review. Persistent low-quality reviewing can desk-reject the submitted paper.
9. [ ] **Confirm CoI declarations** for every coauthor (Domain + Personal CoI lists in OpenReview).
10. [ ] **Decide AI-assisted reviewing experiment** opt-in/out at submission.
11. [ ] **Decide supplementary packaging**: ZIP vs. anonymous URL only.
12. [ ] **Click Submit** before May 6, 2026 AOE.
