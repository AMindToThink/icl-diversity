# Workshop Submission TODOs (Human-AI Co-Creativity @ ICML 2026)

Deadline: **2026-05-01 AOE**. Double-blind, OpenReview, non-archival, 4–8 main pages.

These are administrative actions only — code/paper edits are tracked via git on the
`workshop-icml2026` branch. None of the items below can be done by Claude (they
require interactive web logins, account creation, or human judgment about a
service to use).

## Hard requirements (must be done before submission)

- [ ] **Anonymized code mirror.** Upload `github.com/AMindToThink/icl-diversity` to
      [anonymous.4open.science](https://anonymous.4open.science). The site requires
      an interactive Google/GitHub OAuth login + a ToS acceptance, so this has to
      be done from a browser. Once the mirror exists, replace the placeholder in
      `paper/main_icml_workshop.tex`:
      ```
      \newcommand{\projectGithubUrl}{[code link withheld …]}
      ```
      with `\newcommand{\projectGithubUrl}{\url{https://anonymous.4open.science/r/<id>}}`.
      The original (named-author) `paper/in_context_diversity_metric.tex` stays
      pointed at the real GitHub URL.

- [ ] **Anonymized HF dataset.** The OLMo-2-7B four-stage samples currently live at
      `huggingface.co/datasets/AMindToThink/olmo-2-1124-7b-four-stage-samples-rlhf-diversity`.
      HF has no first-party anonymization service (issues
      [#1924](https://github.com/huggingface/datasets/issues/1924) and
      [#7758](https://github.com/huggingface/datasets/issues/7758) still open as
      of 2026-04-27). Pick one of:
      1. **Throwaway HF account** (de facto practice in ML submissions, e.g.
         `anonymous-acl-submission/*`). Create with a fresh email, neutral
         username (`anon-icml2026-coc` or similar), re-upload the dataset.
      2. **Zenodo with anonymous-author deposit + secret reviewer URL**
         ([example](https://zenodo.org/records/15742903),
         [tutorial](https://github.com/dgraziotin/disclose-data-dbr-first-then-opendata)).
         Yields a DOI you keep after acceptance.
      3. **Zip in OpenReview supplementary** if dataset is under ~100 MB.
      Decision driver: dataset size. Run `du -sh` on the local copy before picking.
      After picking, replace `\projectHfDatasetUrl` in
      `paper/main_icml_workshop.tex` with the anonymous URL (or with
      "see supplementary materials" if going with option 3).

- [ ] **OpenReview author profile.** Matthew (and any co-authors who'll appear
      after camera-ready) needs an OpenReview profile by submission time.
      Workshop URL: see `paper/human-ai-creativity-workshop.md`.

- [ ] **Reciprocal-reviewing willingness flag.** The workshop expects authors
      who have submitted to be willing to review. Toggle this on the OpenReview
      submission form when uploading the PDF.

## Verification before clicking submit

- [ ] `pdftotext paper/main_icml_workshop.pdf - | rg -i 'khoriaty|williams-king|shi feng|ERA Cambridge|George Washington|AMindToThink'` → zero hits.
- [ ] `pdftotext paper/main_icml_workshop.pdf - | rg -nE 'github\.com/(AMindToThink|matthew)|huggingface\.co/datasets/(AMindToThink|matthew)'` → zero hits.
- [ ] Workshop main body is between 4 and 8 pages (count pages before the page on which `\appendix` produces output).
- [ ] `uv run python scripts/verify_cites.py --tex paper/main_icml_workshop.tex` passes.
- [ ] `uv run pytest tests/test_paper_macros.py` passes (5/6 currently — the 6th
      fails only inside the worktree because `diversity-eval/` is gitignored;
      run from main checkout if in doubt).

## Open paper investigations (for future agent)

- [ ] **Item 6 (edge-case caption hedge) is paused mid-edit.** The original
      plan-draft caption said "multi-mode coherent is the intended winner in
      the large-$n$ limit" — this is wrong, because at truly large $n$ every
      coherent policy has $a_n \to 0$ (every regularity gets learned).
      The empirical "smaller $a_n$ for multi-mode coherent than the table
      predicts" is most likely **mechanism 2: mode count too small relative to
      $n$**. The synthetic uses 3 modes and $n = 10$, i.e. each mode is seen
      ~3 times, plenty for $\theta$ to learn each one individually.
      The mode-count scaling experiment (§B.3, `paper/sections/07_3_mode_count.tex`,
      figure `figures/mode_count/qwen2.5-3b/ak_curves_overlay.png` and Table 6)
      already shows $a_n$ rising monotonically with $m$ (\modeCountAnMone\ bits
      at $m=1$ to \modeCountAnMten\ bits at $m=10$). That figure directly
      supports the "$a_n$ tracks mode count" story and validates mechanism 2.

      Two follow-ups for the next agent:
      1. **Rewrite Item 6's caption** in `paper/sections/03_method_workshop.tex`
         (lines 49–51) to hedge along the lines of:
         "Multi-mode coherent's $a_n$ scales with mode count
         (Appendix~\ref{sec:mode-count-scaling}); on the 3-mode 10-response
         synthetic the floor falls below the `high' label because $\theta$
         has effectively seen each mode several times. Practitioners with a
         clear preference for one factor over the other can reweight (e.g.,
         $C^{\alpha} \times a_n$)."
         Then mirror the practitioner-reweighting sentence to the §5.3
         enumerate in `paper/sections/05_reporting.tex`.
      2. **Decide whether to surface the mode-count figure to workshop main
         body.** It currently lives in the workshop appendix (§B.3). The user
         flagged it as "a beautiful image" they want findable. Options:
         (a) leave in appendix and add a forward-pointer from §3.3 caption,
         (b) move to main body (costs ~0.5 page; budget already at the cap).
         Decide after rewriting the caption.

## Post-acceptance follow-ups

- [ ] Mirror anonymized HF dataset back under `AMindToThink` (or wherever the
      camera-ready credit reads) and update the camera-ready URL.
- [ ] Replace anonymous-mirror code URL with canonical GitHub URL in the
      camera-ready.
- [ ] Merge `workshop-icml2026` branch back to `main` once the parallel
      original-paper work has settled.
