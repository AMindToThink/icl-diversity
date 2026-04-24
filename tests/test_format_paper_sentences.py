"""Unit tests for scripts/format_paper_sentences.py.

Each test asserts that `reflow()` either inserts a newline at a real
sentence boundary, or leaves the input unchanged in a context where a
break would corrupt the rendered LaTeX. The cases mirror the false
breaks observed in `latexindent --m oneSentencePerLine` on this paper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "format_paper_sentences.py"

# Load the script as a module without making `scripts/` a package.
_spec = importlib.util.spec_from_file_location("format_paper_sentences", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["format_paper_sentences"] = _mod
_spec.loader.exec_module(_mod)
reflow = _mod.reflow


def test_basic_break_between_sentences() -> None:
    assert reflow("Hello world. Next sentence.") == "Hello world.\nNext sentence."


def test_break_after_question_mark() -> None:
    assert reflow("Is it true? Yes it is.") == "Is it true?\nYes it is."


def test_break_after_exclamation_mark() -> None:
    assert reflow("Wow! What a result.") == "Wow!\nWhat a result."


def test_no_break_at_lowercase_continuation() -> None:
    # Period followed by lowercase = not a sentence boundary.
    assert reflow("This. is one sentence.") == "This. is one sentence."


def test_abbreviation_eg_does_not_break() -> None:
    assert reflow("Models, e.g. GPT-2 and BERT.") == "Models, e.g. GPT-2 and BERT."


def test_abbreviation_ie_does_not_break() -> None:
    assert reflow("That is, i.e. the metric is fine.") == "That is, i.e. the metric is fine."


def test_abbreviation_iid_does_not_break() -> None:
    src = "We sample i.i.d. Draws from the policy."
    assert reflow(src) == src


def test_abbreviation_vs_does_not_break() -> None:
    src = "We compare A vs. B in Table 1."
    assert reflow(src) == src


def test_abbreviation_etc_does_not_break() -> None:
    src = "(haiku, code, recipe, etc.) all responding."
    assert reflow(src) == src


def test_no_break_inside_inline_math() -> None:
    # `!` in `$1/n!$` is the factorial operator, not a sentence end.
    src = r"The probability is $1/n! \approx 0.01$. The next sentence."
    expected = r"The probability is $1/n! \approx 0.01$." + "\n" + r"The next sentence."
    assert reflow(src) == expected


def test_no_break_inside_display_math() -> None:
    # Math env terminator `\]` is not a sentence terminator. With nothing
    # after the math but space + Capital, we leave the line as-is.
    src = r"\[ a! = a \cdot (a-1)! \] Next sentence."
    assert reflow(src) == src


def test_break_after_display_math_when_period_precedes() -> None:
    # When prose punctuation (.) sits before the math env terminator,
    # reflow correctly fires on that period.
    src = r"We compute $a! = b$. Then we continue."
    expected = r"We compute $a! = b$." + "\n" + r"Then we continue."
    assert reflow(src) == expected


def test_no_break_inside_texttt_arg() -> None:
    # Period inside \texttt{} should not trigger a break.
    src = r"Tokenizer emits \texttt{.\textbackslash n} for separators. Continued."
    expected = (
        r"Tokenizer emits \texttt{.\textbackslash n} for separators."
        + "\n"
        + r"Continued."
    )
    assert reflow(src) == expected


def test_no_break_inside_textcolor_color_spec() -> None:
    # `orange!70!black` color spec — `!` should not be treated as sentence end.
    src = r"The \textcolor{orange!70!black}{conditional track} feeds context. Done."
    expected = (
        r"The \textcolor{orange!70!black}{conditional track} feeds context."
        + "\n"
        + r"Done."
    )
    assert reflow(src) == expected


def test_no_break_before_closing_quote() -> None:
    # `?''` is a quoted question; the closing `''` must not be split off.
    src = r"He asked ``why?'' Then he left."
    expected = r"He asked ``why?''" + "\n" + r"Then he left."
    assert reflow(src) == expected


def test_no_break_before_footnote_digit() -> None:
    # `bits.1` has no space — already not a candidate, but verify.
    src = "Units: bits.1 This footnote applies."
    assert reflow(src) == src


def test_no_break_inside_comment() -> None:
    src = "% This. Should not break.\nReal text. Real next."
    expected = "% This. Should not break.\nReal text.\nReal next."
    assert reflow(src) == expected


def test_no_break_inside_tikzpicture() -> None:
    src = (
        r"\begin{tikzpicture}"
        + "\n"
        + r"  \node {orange!70!black. Label.};"
        + "\n"
        + r"\end{tikzpicture} Then prose. More prose."
    )
    expected = (
        r"\begin{tikzpicture}"
        + "\n"
        + r"  \node {orange!70!black. Label.};"
        + "\n"
        + r"\end{tikzpicture} Then prose."
        + "\n"
        + r"More prose."
    )
    assert reflow(src) == expected


def test_no_break_inside_equation_env() -> None:
    src = (
        r"\begin{equation}"
        + "\n"
        + r"  a! = b. \quad c! = d."
        + "\n"
        + r"\end{equation} Then prose. More."
    )
    expected = (
        r"\begin{equation}"
        + "\n"
        + r"  a! = b. \quad c! = d."
        + "\n"
        + r"\end{equation} Then prose."
        + "\n"
        + r"More."
    )
    assert reflow(src) == expected


def test_existing_newlines_preserved() -> None:
    # Blank lines (paragraph breaks) and existing single newlines remain.
    src = "First paragraph.\n\nSecond paragraph. Continues here."
    expected = "First paragraph.\n\nSecond paragraph.\nContinues here."
    assert reflow(src) == expected


def test_idempotent() -> None:
    once = reflow("Hello. World. Third sentence.")
    twice = reflow(once)
    assert once == twice


def test_paragraph_with_mixed_content() -> None:
    src = (
        "We measure $1/n!$ and find it small. "
        "Likewise, e.g. on Tevet, results agree. "
        "More text continues."
    )
    expected = (
        "We measure $1/n!$ and find it small.\n"
        "Likewise, e.g. on Tevet, results agree.\n"
        "More text continues."
    )
    assert reflow(src) == expected


def test_escaped_backslash_does_not_open_math() -> None:
    # `\\[1em]` in a \date{...} argument is the LaTeX linebreak command
    # plus an optional spacing arg — NOT the start of `\[ ... \]` display
    # math. A buggy formatter treats the second `\` as opening display
    # math and creates a runaway skip region until the next real `\]`,
    # which causes downstream prose to be skipped.
    src = (
        r"\date{\today\\[1em]\small $^1$ERA Cambridge}"
        + "\n\n"
        + r"\begin{abstract}"
        + "\n"
        + r"First sentence. Second sentence."
        + "\n"
        + r"\end{abstract}"
    )
    expected = (
        r"\date{\today\\[1em]\small $^1$ERA Cambridge}"
        + "\n\n"
        + r"\begin{abstract}"
        + "\n"
        + r"First sentence."
        + "\n"
        + r"Second sentence."
        + "\n"
        + r"\end{abstract}"
    )
    assert reflow(src) == expected


def test_no_break_inside_cite() -> None:
    src = r"As shown by \citep{smith2020a. b} this works. Next."
    expected = r"As shown by \citep{smith2020a. b} this works." + "\n" + r"Next."
    assert reflow(src) == expected
