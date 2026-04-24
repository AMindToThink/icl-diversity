"""Tests for `scripts/build_paper_macros.py`.

These tests guarantee that `results/tables/paper_macros.tex` — the single source
of truth for every inline numeric scalar in the paper — stays well-formed and
that each section of the paper prose finds the macros it relies on.

No network and no heavy computation: we read the generated file and run the
script end-to-end against local data files (pairwise JSONs, mode-count JSONs,
summary_table.txt, contest/dectest/qwen3 .tex tables).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "build_paper_macros.py"
OUTPUT = PROJECT_ROOT / "results" / "tables" / "paper_macros.tex"
PAPER_TEX = PROJECT_ROOT / "paper" / "in_context_diversity_metric.tex"


@pytest.fixture(scope="module")
def macros() -> dict[str, str]:
    """Parse the generated paper_macros.tex into a dict {name: value}."""
    assert OUTPUT.exists(), (
        f"{OUTPUT} missing. Run: uv run python scripts/build_paper_macros.py"
    )
    text = OUTPUT.read_text()
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
    return dict(pattern.findall(text))


def test_script_runs_end_to_end() -> None:
    """The script must run without error on the committed data."""
    result = subprocess.run(
        ["uv", "run", "python", str(SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"build_paper_macros.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert OUTPUT.exists()


def test_output_is_well_formed(macros: dict[str, str]) -> None:
    """Every line (after the header) is a single \\newcommand with non-empty value."""
    for name, value in macros.items():
        assert value.strip(), f"Macro {name} has empty value"
        # LaTeX identifier names are letters only.
        assert re.fullmatch(r"[A-Za-z]+", name), f"Invalid macro name: {name}"


def test_expected_section_coverage(macros: dict[str, str]) -> None:
    """Every paper section that cites inline numbers has at least one macro."""
    prefixes_needed = [
        "crossmode",  # Sec 8.4 pairwise matrices + fractional model
        "scaling",  # Sec 8.7 scaling paragraph + App E
        "qwenThree",  # App E Qwen3 comparison
        "tevet",  # Sec 7.5 + Abstract + App B
        "modeCount",  # Sec 8.3
    ]
    for p in prefixes_needed:
        matching = [n for n in macros if n.startswith(p)]
        assert matching, f"No macros found for prefix {p!r}"


def test_numeric_macros_are_parseable(macros: dict[str, str]) -> None:
    """All primary numeric macros should parse as floats (after stripping % sign / sign chars)."""
    # Integer-valued counts and mixed-type macros are excluded.
    numeric_prefixes = (
        "crossmodeQwenDiag",
        "crossmodeGPTDiag",
        "crossmodeQwenOff",
        "crossmodeGPTOff",
        "crossmodeAsymmetry",
        "crossmodeAinfFloor",
        "scalingOffDiag",
        "scalingFracPos",
        "scalingRC",
        "scalingSym",
        "tevetMcDiv",
        "tevetCxAn",
        "tevetAnVs",
        "modeCountAn",
        "modeCountCxAn",
        "qwenThreeBinaryMean",
        "qwenThreeDecTestMean",
    )
    for name, raw in macros.items():
        if not any(name.startswith(p) for p in numeric_prefixes):
            continue
        val = raw.replace("\\%", "").replace("%", "").strip()
        # Allow a leading + sign.
        if val.startswith("+"):
            val = val[1:]
        try:
            float(val)
        except ValueError as e:
            raise AssertionError(f"Macro {name}={raw!r} is not a valid number") from e


def test_paper_macros_all_resolve() -> None:
    """Every \\macroName referenced in the paper is defined in paper_macros.tex.

    Catches drift where the paper references a macro that doesn't exist (most
    common after renaming a macro in build_paper_macros.py but forgetting the paper).
    """
    paper = PAPER_TEX.read_text()
    # Strip out comments (lines starting with % or inline % not preceded by \)
    paper_no_comments = re.sub(r"(?<!\\)%.*", "", paper)
    # Find macros our script defines (used as the allowlist so we don't trip on
    # LaTeX built-ins like \small, \textbf, etc.).
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", OUTPUT.read_text()))
    # Our macros follow specific prefixes; look for references.
    prefixes = ("crossmode", "scaling", "qwenThree", "tevet", "modeCount")
    referenced = set(
        m.group(1)
        for prefix in prefixes
        for m in re.finditer(rf"\\({prefix}[A-Za-z]+)", paper_no_comments)
    )
    missing = referenced - defined
    assert not missing, (
        f"Paper references macros that don't exist in paper_macros.tex: {sorted(missing)}"
    )


def test_paper_has_no_forgotten_hand_typed_numbers_in_edited_clusters() -> None:
    """Specific regression guard: the Tevet prose and cross-mode cluster should
    no longer contain the hand-typed strings that were there before this pass."""
    paper = PAPER_TEX.read_text()
    forbidden_substrings = [
        # Sec 7.5 prose that was factually wrong.
        "$\\rho = +0.73$ and OCA = 0.85 on McDiv prompt",
        "SentBERT is the strongest baseline across all tasks",
        "$C \\times a_n$ consistently outperforms distinct-$n$",
        "is within 5--15\\% of SentBERT's OCA",
        "is 5--17\\% lower in OCA than",
        "(OCA near chance)",
        # Misinterpretation: a_1's wrong sign was attributed to "residual signal"
        # hypothesis, but it actually reflects the McDiv confound — same as C's
        # positive correlation. Keep both explanations unified.
        "confirming that the diversity signal is in the \\emph{residual} surprise",
        # Sec 8.6 permutation-sensitivity: the original "≥50" recommendation
        # was not backed by data between 3 and 100 perms. Keep it softened.
        "We recommend $n_{\\mathrm{permutations}} \\geq 50$.",
        "At 3 permutations, scenario rankings are unreliable; at 100 permutations, rankings stabilize.",
        # Sec 8.6 BPE: the old paragraph framed the single-pass/multi-pass
        # difference as "bias". There is no "multi-pass ground truth" — see
        # the global CLAUDE.md entry. Keep the replacement "Boundary handling"
        # wording instead.
        "Single-pass computation introduces a small systematic bias",
        "residual effect is small relative to the total curve decline",
        # Sec 8.4 / 8.7 hand-typed numbers we replaced.
        "only $+0.9$ bits",
        "overpredicts by $10.6\\times$ (622 bits predicted",
        "$r_{\\mathrm{off}} = \\bar{M}_{\\mathrm{off}} / a_1 = 1.4\\%",
        "an overprediction of $2.2\\times$ at $m=10$",
        "is 5.6 bits (2.9$\\times$ the off-diagonal mean)",
        # Sec 8.7 scaling paragraph hand-typed percentages.
        "25\\%, 41\\%, 44\\%, 63\\%",
        "$-4.9$ (1B), $-1.4$ (3B)",
        # Sec 8.7 discussion recap of Tevet 5--15% claim.
        "competitive with embedding-based metrics (within 5--15\\% OCA of SentBERT)",
        # Qwen3 -0.014 that should be -0.013 via macro.
        "(mean $-0.014$ AUC)",
        # Abstract understating.
        "achieves ROC AUC 0.87,",
    ]
    present = [s for s in forbidden_substrings if s in paper]
    assert not present, (
        f"Paper still contains these hand-typed strings (should be macro references):\n"
        + "\n".join(f"  - {s!r}" for s in present)
    )
