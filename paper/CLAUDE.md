# paper/CLAUDE.md

Notes for Claude sessions editing or rebuilding the LaTeX papers.

For paper-tables / inline-macros / citation-pipeline conventions, see the
root `CLAUDE.md` ("Paper Tables, Figures, and Inline Numbers" and
"Citation Pipeline" sections). This file covers only the rebuild workflow.

## Rebuilding the PDF

Use `.claude-tools/rebuild-latex.py <paper.tex> [<paper2.tex> ...]` instead
of writing out `latexmk` commands by hand. It runs
`latexmk -pdf -interaction=nonstopmode` with `SOURCE_DATE_EPOCH=1700000000`
/ `FORCE_SOURCE_DATE=1` for byte-stable PDFs, judges success by output
markers + PDF presence (not the noisy exit code), auto-retries with
`latexmk -C` once if the build genuinely failed, and takes a per-paper
`flock` so concurrent agent invocations on the same paper serialize
instead of corrupting each other's `.aux` files. Different papers build
in parallel.

- **Default (no flag)** is the right mode for "I edited a section, give me
  a new PDF" — preserves the prior `.bbl` so citations stay resolved.
- **`--force`** cleans first; use only if you suspect aux corruption that
  the auto-retry didn't catch. It can leave unresolved `?`-refs because
  `latexmk` doesn't always run enough passes after wiping `.bbl`; re-run
  without `--force` to settle them.
- An OK line ending in `[latexmk warnings; see log]` means a PDF was
  produced but pdflatex emitted warnings (typically undefined refs or
  missing fonts) — grep the `.log` for `! ` or `LaTeX Warning:` to
  investigate.
- Not on the allowlist, so the first call per session will prompt; safe
  to approve.
- Full docstring at the top of the script explains every edge case.

### Worktree note

The script lives at `.claude-tools/rebuild-latex.py` from the repo root.
Git worktrees (e.g. `.worktrees/workshop-icml2026/`) have independent
working trees, so the script only appears inside a worktree once this
file's commit reaches that branch. Until then, invoke it via its
absolute path: `/home/cs29824/matthew/icl-diversity/.claude-tools/rebuild-latex.py`.
