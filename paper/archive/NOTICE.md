# Archived original paper

`in_context_diversity_metric.tex` and its single-paper-only section files
(`abstract`, `01_motivation`, `03_progressive_conditioning`, `04_coherence`,
`05_reporting`, `07_5_tevet`, `07_6_rlhf`, `08_limitations`,
`acknowledgements`) were the original journal-track wrapper. Archived
2026-04-30 in favour of `paper/main_icml_workshop.tex`, which is now the
canonical paper.

Future reframings should branch off the workshop wrapper, not this archive.
The wrapper here is preserved verbatim, but its `\input{sections/...}`
paths still point at `sections/` (i.e., `paper/archive/sections/`) for
files we moved into the archive, and at `paper/sections/` for files
shared with the workshop. If you ever need to recompile this archive,
those paths will need to be re-resolved (e.g., switch the shared-section
`\input` lines to `\input{../sections/...}`).

The cross-reference `\ref{app:c-normalization}` in `sections/05_reporting.tex`
will dangle on rebuild because that subsection was excluded from the
shared `appB_excess_entropy.tex` (wrapped in `\iffalse...\fi`) on the
same date. Re-enable the subsection there if you ever need this archive
to compile cleanly.
