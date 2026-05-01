#!/usr/bin/env python3
"""Rebuild LaTeX papers reproducibly, retrying with --clean on failure.

Usage:
  .claude-tools/rebuild-latex.py [--force] <tex-file> [<tex-file> ...]

Behavior:
  For each .tex file, cd into its parent directory and run
  `latexmk -pdf -interaction=nonstopmode <name.tex>` with
  SOURCE_DATE_EPOCH=1700000000 / FORCE_SOURCE_DATE=1 set so the resulting
  PDFs are byte-stable across rebuilds.

  Success is judged by (a) the absence of fatal-error markers in the latexmk
  output ("Emergency stop", "Fatal error occurred", "no output PDF file
  produced"), (b) the presence of a positive marker ("Output written on" or
  "are up-to-date"), (c) the PDF file existing on disk, AND (d) a post-build
  check that the rendered PDF contains zero `??` placeholders and that the
  .log has zero "LaTeX Warning: Reference/Citation ... undefined" lines.
  If (d) fails, the script automatically runs one additional latexmk pass
  (no clean) to let the now-populated .aux settle outstanding cross-refs;
  if the issue persists, the build is judged FAILED and the user is told
  to re-run with --force.

  The latexmk *exit code* is intentionally NOT used as the primary signal,
  because `pdflatex -interaction=nonstopmode` returns 1 on benign warnings
  (undefined refs, missing fonts) even when the PDF is produced fine. When
  the build is judged successful but rc != 0, the script appends
  "[latexmk warnings; see log]" to the OK line so the caller knows to look
  at <name>.log.

  If the build is judged unsuccessful, the script runs `latexmk -C <name.tex>`
  to clean auxiliary files and retries the build once. With --force, the
  clean step runs first (see Caveats).

  After each attempt, prints the PDF page count and size (or, on failure,
  the last 20 lines of latexmk output). Exits 0 only if every requested
  rebuild succeeded.

Caveats / known limitations:
  - The default (no-flag) incremental mode is the reliable mode: it
    preserves the prior .bbl, so all citation resolution from earlier
    builds carries over. This is the right thing to use for "I just edited
    a section, give me a new PDF."
  - --force is a "nuke and pray" escape hatch. `latexmk -C` wipes the .bbl
    along with everything else, and latexmk does not always run enough
    pdflatex passes after rebuilding the .bbl to fully resolve all
    cross-references — the resulting PDF may have ?-placeholder citations
    or refs. The post-build resolution check (item (d) above) now catches
    this and runs an extra pass automatically. If even the extra pass
    can't resolve everything, the build is judged FAILED rather than
    silently shipping a broken PDF. Use --force only when you actually
    suspect aux corruption that the default-mode retry path didn't catch.
  - "[latexmk warnings; see log]" means a PDF was produced but pdflatex
    emitted warnings on its final pass — typically undefined refs/cites,
    missing characters, or hbox overfulls. Investigate by greping the
    .log for `! ` (errors), `LaTeX Warning:`, or `undefined`. The PDF is
    usable but may not be what you intended.

Operational notes:
  - `latexmk -C` only deletes latexmk's own aux files (.aux, .log, .fls,
    .fdb_latexmk, .bbl, .blg, .out, .synctex.gz, etc). It does NOT delete
    .tex sources, .bib files, or .pdf outputs.
  - This script never writes outside the .tex file's parent directory and
    never touches files other than what latexmk itself produces (plus a
    per-paper `.<name>.tex.rebuild.lock` lock file in the same directory;
    gitignored at the repo root).
  - Concurrency-safe: takes a per-paper exclusive `flock` so that two
    parallel invocations on the same paper serialize instead of corrupting
    each other's aux files. Different papers build in parallel as normal.
    Default lock timeout is 600s; override with REBUILD_LOCK_TIMEOUT env var.
    If the lock times out (likely a stuck/crashed sibling), the script
    fails loudly rather than auto-killing the holder — investigate
    manually.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOCK_TIMEOUT_SECONDS = int(os.environ.get("REBUILD_LOCK_TIMEOUT", "600"))

REPRODUCIBLE_ENV = {
    "SOURCE_DATE_EPOCH": "1700000000",
    "FORCE_SOURCE_DATE": "1",
}

FATAL_LOG_MARKERS = (
    "Emergency stop",
    "Fatal error occurred",
    "==> Fatal error",
    "no output PDF file produced",
)

# latexmk prints one of these when the run actually produced (or kept) a PDF.
# We use these as the positive success signal because latexmk's exit code is
# noisy: pdflatex returns 1 on non-fatal warnings (undefined refs etc.)
# even when the PDF is produced fine.
SUCCESS_LOG_MARKERS = (
    "Output written on",          # fresh build produced a PDF
    "are up-to-date",             # nothing to do; existing PDF kept
)

# Matches lines like:
#   LaTeX Warning: Reference `eq:foo' on input line 42 undefined on input line 99.
#   LaTeX Warning: Citation `bar2024' on page 3 undefined on input line 99.
# pdflatex emits one of these for every \ref/\cite that didn't resolve. If the
# .log has any of these we know the rendered PDF has at least one `??` (or
# bibliography style equivalent) regardless of what pdftotext can extract.
UNDEFINED_LOG_RE = re.compile(
    r"^LaTeX Warning: (?:Reference|Citation) `[^']+' .* undefined",
    re.MULTILINE,
)


def run_latexmk_pdf(tex_dir: Path, tex_name: str) -> tuple[int, str]:
    """Run `latexmk -pdf -interaction=nonstopmode tex_name` in tex_dir.

    Returns (returncode, combined_stdout_stderr).
    """
    env = {**os.environ, **REPRODUCIBLE_ENV}
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_name],
        cwd=tex_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def run_latexmk_clean(tex_dir: Path, tex_name: str) -> None:
    """Run `latexmk -C tex_name` in tex_dir. Returns nothing; failures are non-fatal."""
    env = {**os.environ, **REPRODUCIBLE_ENV}
    subprocess.run(
        ["latexmk", "-C", tex_name],
        cwd=tex_dir,
        env=env,
        capture_output=True,
        text=True,
    )


def looks_fatal(output: str) -> bool:
    return any(marker in output for marker in FATAL_LOG_MARKERS)


def looks_successful(output: str, pdf_path: Path) -> bool:
    """latexmk-success heuristic that ignores the noisy exit code.

    True iff: the output contains a positive success marker, no fatal marker,
    AND the PDF file actually exists on disk.
    """
    if looks_fatal(output):
        return False
    if not pdf_path.exists():
        return False
    return any(marker in output for marker in SUCCESS_LOG_MARKERS)


def count_pdf_unresolved(pdf_text: str) -> int:
    """Count `??` placeholder pairs in extracted PDF text.

    LaTeX renders unresolved `\\ref{}` (and some `\\cite{}` styles) as `??`.
    A clean prose `?` is unaffected; this counts only consecutive `??`.
    """
    return pdf_text.count("??")


def count_log_undefined(log_text: str) -> int:
    """Count `LaTeX Warning: Reference/Citation '...' undefined` lines in a .log."""
    return len(UNDEFINED_LOG_RE.findall(log_text))


def extract_pdf_text(pdf_path: Path) -> str:
    """Run `pdftotext <pdf> -` and return stdout. Empty string on failure.

    pdftotext ships with poppler-utils alongside pdfinfo, which the script
    already requires. If pdftotext is missing we fall back to empty text
    (the log-based check still catches undefined refs); the startup probe
    in main() warns the user when pdftotext is absent.
    """
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def check_resolution(pdf_path: Path, log_path: Path) -> tuple[int, int]:
    """Post-build verification.

    Returns (pdf_qq_count, log_undefined_count). Either > 0 means the build
    shipped a PDF with unresolved `\\ref`/`\\cite`. Both checks are run because
    they catch overlapping but not identical failure modes:
      - pdftotext + `??`: catches anything LaTeX rendered as `??`, including
        the case where a stale .aux made latexmk decide "no rerun needed"
        and pdflatex consequently emitted no warning on the final pass.
      - log scan: catches undefined cites that some bib styles render as
        `[?]` rather than `??`, plus any pass-where-pdflatex-did-warn.
    """
    pdf_text = extract_pdf_text(pdf_path)
    pdf_qq = count_pdf_unresolved(pdf_text)
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    log_undef = count_log_undefined(log_text)
    return pdf_qq, log_undef


def pdf_status(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return "MISSING"
    size_kb = pdf_path.stat().st_size // 1024
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Pages:"):
                    pages = line.split()[1]
                    return f"{pages} pages, {size_kb} KB"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return f"{size_kb} KB"


def tail(text: str, n: int = 20) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


@contextlib.contextmanager
def acquire_paper_lock(tex_path: Path, timeout: int):
    """Acquire an exclusive flock on a per-paper lock file.

    Polls every 2s up to `timeout` seconds. Raises TimeoutError if the lock
    is still held when the timeout expires (likely a stuck/crashed sibling
    build — investigate manually rather than auto-killing).
    """
    lock_path = tex_path.parent / f".{tex_path.name}.rebuild.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + timeout
    waited = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if not waited:
                    print(
                        f"  waiting for sibling rebuild to finish "
                        f"(lock: {lock_path.name}) ...",
                        flush=True,
                    )
                    waited = True
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Lock {lock_path} held by another process for >{timeout}s"
                    )
                time.sleep(2.0)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def rebuild_one(tex_path: Path, force_clean: bool) -> bool:
    tex_dir = tex_path.parent
    tex_name = tex_path.name
    pdf_path = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".log")

    print(f"→ {tex_path}", flush=True)

    try:
        with acquire_paper_lock(tex_path, LOCK_TIMEOUT_SECONDS):
            if force_clean:
                print("  cleaning aux files (--force) ...", flush=True)
                run_latexmk_clean(tex_dir, tex_name)

            rc, output = run_latexmk_pdf(tex_dir, tex_name)
            ok = looks_successful(output, pdf_path)

            if not ok and not force_clean:
                print("  build did not produce PDF; cleaning aux files and retrying ...", flush=True)
                run_latexmk_clean(tex_dir, tex_name)
                rc, output = run_latexmk_pdf(tex_dir, tex_name)
                ok = looks_successful(output, pdf_path)

            if not ok:
                print("  FAILED. Last 20 lines of latexmk output:", file=sys.stderr)
                print(tail(output, 20), file=sys.stderr)
                return False

            # Post-build cross-reference resolution check. latexmk's "rerun if
            # .aux changed" heuristic occasionally ships a PDF with `??`
            # placeholders even though the build appears successful (the bug
            # that produced anon-submission's 6b40e79: 53 `??` across 17 pages
            # with no warning suffix). Run an extra latexmk pass to let any
            # outstanding refs settle; if they don't, treat as build failure.
            pdf_qq, log_undef = check_resolution(pdf_path, log_path)
            if pdf_qq or log_undef:
                print(
                    f"  unresolved cross-refs after first build: "
                    f"{pdf_qq} '??' in PDF, {log_undef} 'undefined' warning(s) in log; "
                    f"running another latexmk pass ...",
                    flush=True,
                )
                rc, output = run_latexmk_pdf(tex_dir, tex_name)
                pdf_qq, log_undef = check_resolution(pdf_path, log_path)
                if pdf_qq or log_undef:
                    print(
                        f"  FAILED: extra pass did not resolve cross-refs. "
                        f"PDF still has {pdf_qq} '??' placeholder(s); "
                        f"log still has {log_undef} 'Reference/Citation undefined' warning(s).",
                        file=sys.stderr,
                    )
                    print(
                        f"  Inspect: grep -nE \"LaTeX Warning: (Reference|Citation)\" {log_path}",
                        file=sys.stderr,
                    )
                    print(
                        f"  Workaround: re-run with --force to wipe .aux/.bbl and rebuild from scratch:",
                        file=sys.stderr,
                    )
                    print(f"    {sys.argv[0]} --force {tex_path}", file=sys.stderr)
                    return False

            # latexmk rc!=0 with a produced PDF means non-fatal warnings
            # (missing fonts, hbox overfulls, etc). Surface this so the user
            # can investigate, but treat the build as successful.
            warn_suffix = "  [latexmk warnings; see log]" if rc != 0 else ""
            print(f"  OK  {pdf_status(pdf_path)}{warn_suffix}", flush=True)
            return True
    except TimeoutError as e:
        print(f"  FAILED to acquire lock: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild LaTeX papers reproducibly with auto-retry on failure.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clean aux files before building (forces a fresh rebuild).",
    )
    parser.add_argument(
        "tex_files",
        nargs="+",
        help="Path(s) to .tex file(s) to rebuild.",
    )
    args = parser.parse_args()

    if not shutil.which("latexmk"):
        print("Error: latexmk is not installed or not on PATH", file=sys.stderr)
        sys.exit(2)
    if not shutil.which("pdftotext"):
        print(
            "Warning: pdftotext not on PATH; the post-build `??`-in-PDF check will "
            "be skipped. Install poppler-utils to enable it.",
            file=sys.stderr,
        )

    tex_paths: list[Path] = []
    for arg in args.tex_files:
        p = Path(arg).resolve()
        if not p.is_file():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(2)
        if p.suffix != ".tex":
            print(f"Error: not a .tex file: {p}", file=sys.stderr)
            sys.exit(2)
        tex_paths.append(p)

    start = time.time()
    results = [rebuild_one(p, force_clean=args.force) for p in tex_paths]
    elapsed = time.time() - start

    n_ok = sum(results)
    n_total = len(results)
    print(f"\n{n_ok}/{n_total} rebuilds succeeded in {elapsed:.1f}s")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
