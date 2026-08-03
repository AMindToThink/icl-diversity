"""Tests for scripts/rlhf_experiment/7_matrix_paper_outputs.py.

Covers the variance-decomposition math on synthetic data with known answers,
the FINDINGS.md marker-splice mechanics, an end-to-end run against the
committed matrix JSONLs (no model, no network), and the paper-integrity
guards: every \\olmoMx macro referenced in the appendix section resolves in
the generated macros file, and no headline number is hand-typed in the
section prose.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rlhf_experiment" / "7_matrix_paper_outputs.py"
SECTION_PATH = REPO_ROOT / "paper" / "sections" / "appG_olmo_self_scoring.tex"
MACROS_PATH = REPO_ROOT / "results" / "rlhf_experiment" / "paper_macros_matrix.tex"


def _load_script():
    spec = importlib.util.spec_from_file_location("matrix_paper_outputs", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MOD = _load_script()


class TestTwoWayDecomposition:
    def test_pure_column_variation(self):
        """A matrix whose rows are identical has all variance in the column factor."""
        M = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))
        out = MOD.two_way_decomposition(M)
        assert out["frac"]["col"] == pytest.approx(1.0)
        assert out["frac"]["row"] == pytest.approx(0.0, abs=1e-12)
        assert out["frac"]["interaction"] == pytest.approx(0.0, abs=1e-12)

    def test_pure_row_variation(self):
        M = np.tile(np.array([[1.0], [2.0], [5.0]]), (1, 4))
        out = MOD.two_way_decomposition(M)
        assert out["frac"]["row"] == pytest.approx(1.0)
        assert out["frac"]["col"] == pytest.approx(0.0, abs=1e-12)

    def test_additive_effects_have_zero_interaction(self):
        row = np.array([0.0, 1.0, 2.0])
        col = np.array([0.0, 10.0, 20.0, 30.0])
        M = row[:, None] + col[None, :]
        out = MOD.two_way_decomposition(M)
        assert out["frac"]["interaction"] == pytest.approx(0.0, abs=1e-12)
        # Analytic SS: rows: 4 * sum((row - mean)^2); cols: 3 * sum((col - mean)^2).
        ss_row = 4 * ((row - row.mean()) ** 2).sum()
        ss_col = 3 * ((col - col.mean()) ** 2).sum()
        assert out["ss"]["row"] == pytest.approx(ss_row)
        assert out["ss"]["col"] == pytest.approx(ss_col)

    def test_fractions_sum_to_one_on_random_matrix(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(5, 4))
        out = MOD.two_way_decomposition(M)
        assert sum(out["frac"].values()) == pytest.approx(1.0)

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError):
            MOD.two_way_decomposition(np.zeros(5))


class TestThreeWayDecomposition:
    def test_components_sum_to_total(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(5, 4, 7))
        out = MOD.three_way_decomposition(X)
        assert sum(out["ss"].values()) == pytest.approx(out["ss_total"])
        assert sum(out["within_prompt_frac"].values()) == pytest.approx(1.0)

    def test_pure_stage_variation(self):
        """Variance along axis 1 only -> the stage factor owns all within-prompt SS."""
        stage = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.tile(stage[None, :, None], (5, 1, 7))
        out = MOD.three_way_decomposition(X)
        assert out["within_prompt_frac"]["stage"] == pytest.approx(1.0)
        assert out["within_prompt_frac"]["grader"] == pytest.approx(0.0, abs=1e-12)
        assert out["prompt_frac_of_total"] == pytest.approx(0.0, abs=1e-12)

    def test_grader_share_includes_grader_prompt_interaction(self):
        """Grader effect with prompt-dependent sign lands in the grader share."""
        rng = np.random.default_rng(2)
        grader = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prompt_sign = rng.choice([-1.0, 1.0], size=9)
        X = grader[:, None, None] * prompt_sign[None, None, :] * np.ones((5, 4, 9))
        out = MOD.three_way_decomposition(X)
        assert out["within_prompt_frac"]["grader"] == pytest.approx(1.0)

    def test_rejects_non_tensor(self):
        with pytest.raises(ValueError):
            MOD.three_way_decomposition(np.zeros((5, 4)))


class TestExtendedRowMissing:
    def _cells(self, present: dict[str, list[str]]):
        """theta_cells stub for one extended grader over one prompt set."""
        return {
            "GPT-2": {
                (gen, "alpacaeval"): {pid: {"diversity_score_D_C_an": 0.5} for pid in pids}
                for gen, pids in present.items()
            }
        }

    def test_sidecar_backed_missing_is_allowed(self, tmp_path):
        canonical = {"p1", "p2", "p3"}
        cells = self._cells({
            "base": ["p1", "p2", "p3"], "sft": ["p1", "p2"],
            "dpo": ["p1", "p2", "p3"], "instruct": ["p1", "p2", "p3"],
        })
        jsonl = tmp_path / "icl_metrics_lm_theta-gpt2.jsonl"
        jsonl.write_text("")
        sidecar = tmp_path / "icl_metrics_lm_theta-gpt2.skipped_context_overflow.json"
        sidecar.write_text(json.dumps({"skipped": [
            {"stage": "sft", "prompt_set": "alpacaeval", "prompt_id": "p3"},
        ]}))
        missing = MOD.extended_row_missing(cells, "GPT-2", jsonl, canonical, "alpacaeval")
        assert missing == {"p3"}

    def test_unaccounted_missing_raises(self, tmp_path):
        canonical = {"p1", "p2", "p3"}
        cells = self._cells({
            "base": ["p1", "p2"], "sft": ["p1", "p2", "p3"],
            "dpo": ["p1", "p2", "p3"], "instruct": ["p1", "p2", "p3"],
        })
        jsonl = tmp_path / "icl_metrics_lm_theta-gpt2.jsonl"
        jsonl.write_text("")
        with pytest.raises(ValueError, match="NOT in the skip sidecar"):
            MOD.extended_row_missing(cells, "GPT-2", jsonl, canonical, "alpacaeval")


class TestSpliceFindings:
    def test_replaces_marked_block(self, tmp_path):
        f = tmp_path / "FINDINGS.md"
        f.write_text(
            "intro\n<!-- BEGIN GENERATED: d-matrix-alpacaeval -->\nstale\n"
            "<!-- END GENERATED: d-matrix-alpacaeval -->\noutro\n"
        )
        MOD.splice_findings(f, {"d-matrix-alpacaeval": "| fresh |"})
        text = f.read_text()
        assert "| fresh |" in text
        assert "stale" not in text
        assert text.startswith("intro\n") and text.endswith("outro\n")

    def test_missing_marker_raises(self, tmp_path):
        f = tmp_path / "FINDINGS.md"
        f.write_text("no markers here\n")
        with pytest.raises(ValueError, match="missing markers"):
            MOD.splice_findings(f, {"variance": "x"})


class TestEndToEnd:
    """Runs on the committed matrix JSONLs; no model, no network."""

    @pytest.fixture(scope="class")
    def outputs(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("matrix_out")
        variance_out = MOD.run(output_root=root, update_findings=False)
        return root, variance_out

    def test_all_output_files_written(self, outputs):
        root, _ = outputs
        for rel in [
            "results/rlhf_experiment/matrix/variance_decomposition.json",
            "results/rlhf_experiment/matrix/generated_tables.md",
            "results/rlhf_experiment/tables_matrix/olmo_matrix_D.tex",
            "results/rlhf_experiment/tables_matrix/olmo_matrix_R.tex",
            "results/rlhf_experiment/tables_matrix/olmo_matrix_variance.tex",
            "results/rlhf_experiment/paper_macros_matrix.tex",
        ]:
            assert (root / rel).exists(), rel

    def test_variance_fractions_are_sane(self, outputs):
        _, variance_out = outputs
        for pset_out in variance_out["prompt_sets"].values():
            for key in ("cell_means_all_graders", "cell_means_olmo_only"):
                frac = pset_out[key]["frac"]
                assert sum(frac.values()) == pytest.approx(1.0)
                # The generating stage must dominate grader identity by design of
                # the finding; a flip here means the inputs changed materially.
                assert frac["stage"] > frac["grader"]

    def test_extended_tier_is_sane(self, outputs):
        _, variance_out = outputs
        for pset, e in variance_out["extended"].items():
            assert e["n_prompts_common"] <= e["n_prompts_canonical"]
            # Llama has a 128k context; it must be complete.
            assert e["dropped_per_grader"]["Llama-3.1-8B"]["n_missing"] == 0
            frac = e["all_graders_common"]["cell_means"]["frac"]
            assert sum(frac.values()) == pytest.approx(1.0)
            assert frac["stage"] > frac["grader"]
            # Only sidecar-backed GPT-2 drops may shrink the common set.
            n_drop = e["dropped_per_grader"]["GPT-2"]["n_missing"]
            assert e["n_prompts_common"] == e["n_prompts_canonical"] - n_drop

    def test_macros_unique_and_match_committed_file(self, outputs):
        root, _ = outputs
        fresh = (root / "results/rlhf_experiment/paper_macros_matrix.tex").read_text()
        names = re.findall(r"\\newcommand\{\\(olmoMx[A-Za-z]+)\}", fresh)
        assert len(names) == len(set(names)), "duplicate macro names"
        assert MACROS_PATH.exists(), "committed macros file missing; run the script"
        assert fresh == MACROS_PATH.read_text(), (
            "committed paper_macros_matrix.tex is stale; re-run "
            "scripts/rlhf_experiment/7_matrix_paper_outputs.py"
        )


class TestPaperIntegrity:
    def test_every_referenced_macro_is_defined(self):
        section = SECTION_PATH.read_text()
        defined = set(
            re.findall(r"\\newcommand\{\\(olmoMx[A-Za-z]+)\}", MACROS_PATH.read_text())
        )
        referenced = set(re.findall(r"\\(olmoMx[A-Za-z]+)", section))
        missing = referenced - defined
        assert not missing, f"section references undefined macros: {sorted(missing)}"

    def test_every_referenced_tevet_macro_is_defined(self):
        tevet_macros = (
            REPO_ROOT / "results" / "tevet" / "grader_invariance" / "tevet_invariance_macros.tex"
        )
        assert tevet_macros.exists(), "run analyze_tevet_grader_invariance.py first"
        section = SECTION_PATH.read_text()
        defined = set(
            re.findall(r"\\newcommand\{\\(tevetInv[A-Za-z]+)\}", tevet_macros.read_text())
        )
        referenced = set(re.findall(r"\\(tevetInv[A-Za-z]+)", section))
        missing = referenced - defined
        assert not missing, f"section references undefined Tevet macros: {sorted(missing)}"

    def test_no_hand_typed_headline_numbers_in_section(self):
        """Headline results must flow through macros, never be typed in prose."""
        section = SECTION_PATH.read_text()
        forbidden = [
            "0.494", "0.354", "0.481",  # D cells
            "1.68", "1.38", "1.37",     # fold-changes
            "88.0", "90.1", "9.6", "8.0",  # variance shares
            "10^{-16}", "10^{-8}",      # p-values
        ]
        hits = [s for s in forbidden if s in section]
        assert not hits, f"hand-typed numbers found in section: {hits}"
