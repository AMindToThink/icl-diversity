# NOTE (2026-08-04): rebuttals SUBMITTED on OpenReview.
# rebuttal_submission.txt is the text as actually posted and is MORE UP-TO-DATE
# than the separate rebuttal_*.md working files; treat the .md files as drafts.

[x] Rearrange each section to associate it with the relevant reviewer's comments.
    -> rebuttal_general.md (shared opener), rebuttal_pAJM.md, rebuttal_2bo8.md, rebuttal_5AF8.md
[x] Add each thing that the reviewers asked for to this list, prioritize, and deduplicate
    -> tracked per-file via [claude-scaffold] blocks mapping each W/Q to its answer

# Before posting (Matthew)
[x] AI-marker format decided (2026-08-04): **[AI]** ... **[/AI]** bold markers
    (raw <AI> tags risk being stripped by OpenReview's HTML sanitizer). All four
    files converted; general response's disclosure paragraph rewritten by Matthew.
[x] Posting plan (2026-08-04): rebuttal_general.md is posted ONCE as a reply
    beneath the initial meta-review (or as a top-level Official Comment if no
    reply button exists there); each per-review thread opens with the
    "Please see our response to the meta-review" pointer instead of the opener.
[X] Review every [AI] block and [claude-scaffold] note; delete all scaffold lines.
[X] 2bo8 W3: pick Option A or B for the one mid-sentence AI block (SUGGEST marker
   in rebuttal_2bo8.md) so it can be separated like the others.
[] Approve/reword the Claude-drafted second sentence of each thread's top matter
   ("As there, any AI contributions are clearly marked...").
[] pAJM: decide whether to concede the question-length and template sweeps
   (do not exist) or omit; see the last scaffold block in rebuttal_pAJM.md.
[] pAJM: check the truncation block's "The general decision procedure:" sentence
   against the rejected "single explicit procedure" framing (flagged by Claude,
   Matthew's call).
[] Verify char counts <= 10,000 per posted comment (after stripping scaffolds).
[] Final sweep: no links/URLs, no identity leaks, DPO/RLVR never joined by an arrow.

# Deferred (post-discussion)
[] Push all rebuttal files + new scripts/results to GitHub (Matthew loses access
   to this machine soon; nothing should exist only in local worktrees).
[] arXiv v2 items live in paper/arxiv_v2_todo.md (incl. the Appendix F
   gap-direction fix and the rest-and-recheck note on the content-driven claim).
