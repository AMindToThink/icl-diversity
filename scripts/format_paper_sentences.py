#!/usr/bin/env python3
"""
Format LaTeX paper source as one-sentence-per-line ("semantic linebreaks").

Inserts a newline between sentence-terminating punctuation (`.`, `?`, `!`)
and the following sentence, but only at safe positions:

- Skips math mode (`$...$`, `\\(...\\)`, `\\[...\\]`, `$$...$$`).
- Skips math/code/figure environments (verbatim, tikzpicture, equation,
  align, tabular, etc.).
- Skips command arguments where breaking would be wrong
  (`\\texttt{...}`, `\\textcolor{...}`, `\\cite{...}`, `\\ref{...}`, etc.).
- Skips comments.
- Recognizes common abbreviations (`e.g.`, `i.e.`, `i.i.d.`, `vs.`,
  `Fig.`, `etc.`, ...) so their internal periods don't trigger breaks.
- Only breaks before a capital letter (so `.''`, `.\\d` (footnote markers),
  and `etc.,` don't trigger).

Why a custom script: `latexindent --m oneSentencePerLine` and `tex-fmt`
both have known limitations on math-heavy LaTeX papers (factorials,
color specs `orange!70!black`, `\\texttt{.}`, footnote `.\\d`,
quote-attached `?''`). See paper/.latexindent.yaml history for the
attempt that didn't pan out.

Usage:
    uv run python scripts/format_paper_sentences.py INPUT OUTPUT

The output renders identically to the input — single newlines in LaTeX
source become spaces in the PDF.
"""

import re
import sys
from pathlib import Path

# Environments inside which we never insert sentence breaks.
SKIP_ENVS: set[str] = {
    "verbatim", "lstlisting", "tikzpicture",
    "equation", "equation*", "align", "align*",
    "gather", "gather*", "eqnarray", "eqnarray*",
    "multline", "multline*", "array",
    "matrix", "pmatrix", "bmatrix", "vmatrix",
    "cases", "pgfplots", "axis",
    "tabular", "tabular*", "tabularx",
}

# Commands whose first {...} argument should be treated as opaque.
SKIP_CMD_ARGS: set[str] = {
    "texttt", "textcolor", "cite", "citep", "citet", "citeyear",
    "ref", "label", "url", "href", "verb",
}

# Abbreviations whose terminating period must not end a sentence.
# Listed without the trailing dot — the regex adds it.
ABBREVS: list[str] = [
    "e.g", "i.e", "i.i.d", "cf", "vs", "et al", "etc",
    "Fig", "Eq", "Sec", "Tab", "Ref", "Refs",
    "Eqs", "Figs", "Tabs", "Secs", "App", "Apps", "Alg", "No",
]
ABBREV_RE = re.compile(
    r"(?:" + "|".join(re.escape(a) for a in ABBREVS) + r")\.\s*$"
)


def find_skip_regions(text: str) -> list[tuple[int, int]]:
    """Return [start, end) char ranges where sentence breaks must not be inserted."""
    regions: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        c = text[i]

        # Escaped backslash `\\` (LaTeX linebreak) — skip both chars so
        # the second backslash isn't misread as the start of `\(`, `\[`,
        # or a control word. Without this, `\date{...\\[1em]...}` mis-
        # interprets `\[1em]` as display math and creates a runaway
        # skip region until the next real `\]` later in the file.
        if c == "\\" and i + 1 < n and text[i + 1] == "\\":
            i += 2
            continue

        # Comment: `%` to end of line (unless escaped as `\%`).
        if c == "%" and (i == 0 or text[i - 1] != "\\"):
            j = text.find("\n", i)
            j = n if j == -1 else j
            regions.append((i, j))
            i = j
            continue

        # Display math `$$...$$`.
        if c == "$" and i + 1 < n and text[i + 1] == "$":
            j = text.find("$$", i + 2)
            j = n if j == -1 else j + 2
            regions.append((i, j))
            i = j
            continue

        # Inline math `$...$`.
        if c == "$" and (i == 0 or text[i - 1] != "\\"):
            j = i + 1
            while j < n:
                if text[j] == "$" and text[j - 1] != "\\":
                    j += 1
                    break
                j += 1
            regions.append((i, j))
            i = j
            continue

        # `\(...\)` and `\[...\]`.
        if c == "\\" and i + 1 < n and text[i + 1] in "([":
            close = ")" if text[i + 1] == "(" else "]"
            j = text.find("\\" + close, i + 2)
            j = n if j == -1 else j + 2
            regions.append((i, j))
            i = j
            continue

        # `\begin{ENV}...\end{ENV}` for SKIP_ENVS.
        m = re.match(r"\\begin\{(\w+\*?)\}", text[i:])
        if m:
            env = m.group(1)
            if env in SKIP_ENVS:
                end_marker = f"\\end{{{env}}}"
                j = text.find(end_marker, i + len(m.group(0)))
                j = n if j == -1 else j + len(end_marker)
                regions.append((i, j))
                i = j
                continue

        # `\cmd{...}` for commands whose argument is opaque.
        m = re.match(r"\\(\w+)\b", text[i:])
        if m and m.group(1) in SKIP_CMD_ARGS:
            j = i + len(m.group(0))
            # Optional argument `[...]` first.
            if j < n and text[j] == "[":
                close = text.find("]", j)
                if close != -1:
                    j = close + 1
            # Mandatory argument `{...}` with brace counting.
            if j < n and text[j] == "{":
                depth = 1
                k = j + 1
                while k < n and depth > 0:
                    if text[k] == "\\" and k + 1 < n:
                        k += 2
                        continue
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                    k += 1
                regions.append((i, k))
                i = k
                continue

        i += 1
    # Sort + merge (defensive — should already be in order).
    regions.sort()
    merged: list[tuple[int, int]] = []
    for s, e in regions:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def in_skip_region(pos: int, regions: list[tuple[int, int]]) -> bool:
    """O(log n) check via binary search; regions must be sorted, non-overlapping."""
    lo, hi = 0, len(regions)
    while lo < hi:
        mid = (lo + hi) // 2
        s, e = regions[mid]
        if pos < s:
            hi = mid
        elif pos >= e:
            lo = mid + 1
        else:
            return True
    return False


def reflow(text: str) -> str:
    """Insert sentence breaks at safe positions; never removes existing newlines.

    A sentence boundary is `[.?!]` optionally followed by a closing quote
    (`''` or `"`), then whitespace, then a capital letter, all outside a
    skip region (math/comment/skip-env/skip-cmd-arg). The newline is
    inserted between the closing quote (if any) and the whitespace, so
    `?''` stays attached to its sentence.
    """
    regions = find_skip_regions(text)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in ".?!" and not in_skip_region(i, regions):
            j = i + 1
            # Allow an attached closing quote: `''` or `"`.
            if j + 1 < n and text[j : j + 2] == "''":
                j += 2
            elif j < n and text[j] in "'\"":
                j += 1
            # Require a space immediately after.
            if j < n and text[j] == " ":
                k = j
                while k < n and text[k] == " ":
                    k += 1
                if k < n and text[k].isupper() and text[k].isalpha():
                    # Reject if preceded by an abbreviation.
                    prefix = text[max(0, i - 12) : i + 1]
                    if not ABBREV_RE.search(prefix):
                        out.append(text[i:j])
                        out.append("\n")
                        i = k
                        continue
        out.append(c)
        i += 1
    return "".join(out)


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: format_paper_sentences.py INPUT OUTPUT",
            file=sys.stderr,
        )
        sys.exit(2)
    src = Path(sys.argv[1]).read_text()
    Path(sys.argv[2]).write_text(reflow(src))


if __name__ == "__main__":
    main()
