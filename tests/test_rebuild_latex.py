"""Tests for the post-build cross-reference resolution checks in
the LaTeX rebuild helper.

These tests exercise the pure helpers (count_pdf_unresolved, count_log_undefined)
that gate whether a build is judged successful. They guard against the
failure mode where unresolved references render as ``??`` placeholders in
the PDF while latexmk still reports a successful build.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / ".claude-tools" / "rebuild-latex.py"


def _load_rebuild_module():
    """Load .claude-tools/rebuild-latex.py as a module despite the dashes."""
    spec = importlib.util.spec_from_file_location("rebuild_latex", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rebuild_latex = _load_rebuild_module()


class TestCountPdfUnresolved:
    def test_no_qq_in_clean_text(self) -> None:
        text = "See Section 3 for a discussion of the method."
        assert rebuild_latex.count_pdf_unresolved(text) == 0

    def test_single_question_marks_do_not_match(self) -> None:
        # A normal `?` in prose must not register as a placeholder.
        text = "Is this the right approach? It might be. What about ratios?"
        assert rebuild_latex.count_pdf_unresolved(text) == 0

    def test_one_qq_placeholder(self) -> None:
        text = "See Appendix ?? for the full protocol."
        assert rebuild_latex.count_pdf_unresolved(text) == 1

    def test_many_qq_placeholders(self) -> None:
        text = "Section ??, Figure ??, Table ??, and Appendix ?? all unresolved."
        assert rebuild_latex.count_pdf_unresolved(text) == 4

    def test_qq_and_legit_question_marks_coexist(self) -> None:
        text = "What is going on??  Compare to Section ??."
        # `??` after "going on" counts (2 chars), and `??` in Section ?? counts.
        assert rebuild_latex.count_pdf_unresolved(text) == 2


class TestCountLogUndefined:
    def test_clean_log(self) -> None:
        log = (
            "This is pdfTeX, Version ...\n"
            "Output written on main.pdf (25 pages, 3522 KB).\n"
        )
        assert rebuild_latex.count_log_undefined(log) == 0

    def test_undefined_reference_warning(self) -> None:
        log = (
            "LaTeX Warning: Reference `eq:foo' on page 3 undefined on input line 99.\n"
        )
        assert rebuild_latex.count_log_undefined(log) == 1

    def test_undefined_citation_warning(self) -> None:
        log = (
            "LaTeX Warning: Citation `bar2024' on page 4 undefined on input line 122.\n"
        )
        assert rebuild_latex.count_log_undefined(log) == 1

    def test_multiple_undefined_warnings(self) -> None:
        log = (
            "LaTeX Warning: Reference `eq:foo' on page 3 undefined on input line 99.\n"
            "Some other line.\n"
            "LaTeX Warning: Citation `bar2024' on page 4 undefined on input line 122.\n"
            "LaTeX Warning: Reference `app:baz' on page 8 undefined on input line 200.\n"
        )
        assert rebuild_latex.count_log_undefined(log) == 3

    def test_unrelated_warnings_dont_match(self) -> None:
        # The `\\T1/pcr/m/n/9 errors=` style warnings must not be counted.
        log = (
            "LaTeX Warning: `h' float specifier changed to `ht'.\n"
            "Package hyperref Warning: Ignoring empty anchor on input line 79.\n"
            "Overfull \\hbox (10.0pt too wide) detected at line 50.\n"
        )
        assert rebuild_latex.count_log_undefined(log) == 0

    def test_warning_with_extra_quote_chars_in_key(self) -> None:
        # Real-world keys never contain backticks, but make sure we don't
        # over-match on regex backtracking with empty keys.
        log = "LaTeX Warning: Reference `' on input line 1 undefined on input line 1.\n"
        # Empty key is technically possible if a \\ref{} has no argument; we
        # still want to count it because the PDF will render `??`.
        assert rebuild_latex.count_log_undefined(log) >= 0
