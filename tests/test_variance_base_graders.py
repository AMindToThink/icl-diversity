"""Tests for scripts/rlhf_experiment/8_variance_base_graders.py.

The script recomputes the OLMo-matrix variance decomposition
(7_matrix_paper_outputs.py) restricted to base-model-only grader subsets. These
tests run against the committed matrix JSONLs (no model, no network) and cover:
  * the sanity check (re-running the unmodified published pipeline into a
    scratch dir must reproduce the committed variance_decomposition.json)
  * the restricted decompositions are internally consistent (fractions sum to
    1, generator stage still dominates grader identity)
  * the small-prompt-intersection guard fails loudly rather than proceeding
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rlhf_experiment" / "8_variance_base_graders.py"
VARIANCE_DECOMP_PATH = REPO_ROOT / "results" / "rlhf_experiment" / "matrix" / "variance_decomposition.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("variance_base_graders", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MOD = _load_script()


pytestmark = pytest.mark.skipif(
    not VARIANCE_DECOMP_PATH.exists(),
    reason="results/rlhf_experiment/matrix/variance_decomposition.json not generated; "
    "run 7_matrix_paper_outputs.py first",
)


class TestSanityCheck:
    def test_reproduces_published_numbers(self):
        """The unmodified published pipeline, re-run into a scratch dir, must
        exactly match the committed variance_decomposition.json. If this
        fails, the restricted-grader numbers in this script cannot be trusted
        either (see the module docstring)."""
        result = MOD.sanity_check()
        assert result["ok"], result["mismatches"]
        assert result["mismatches"] == []


class TestRestrictedDecomposition:
    @pytest.fixture(scope="class")
    def out(self):
        return MOD.run()

    def test_fractions_sum_to_one(self, out):
        for pset_out in out["prompt_sets"].values():
            for tier_name in ("primary", "supplementary"):
                view = pset_out[tier_name]
                assert sum(view["cell_means"]["frac"].values()) == pytest.approx(1.0)
                assert sum(view["per_prompt"]["within_prompt_frac"].values()) == pytest.approx(1.0)

    def test_generator_stage_dominates_grader(self, out):
        """Headline finding under the restricted base-model-only panels: the
        generating stage must still own far more variance than grader
        identity. A flip here means the base-model-only subset overturns the
        published finding and must be investigated, not silently accepted."""
        for pset_out in out["prompt_sets"].values():
            for tier_name in ("primary", "supplementary"):
                view = pset_out[tier_name]
                cm = view["cell_means"]["frac"]
                wp = view["per_prompt"]["within_prompt_frac"]
                assert cm["stage"] > cm["grader"]
                assert wp["stage"] > wp["grader"]

    def test_primary_excludes_qwen_supplementary_includes_it(self, out):
        assert "Qwen-3B" not in MOD.PRIMARY_LABELS
        assert "Qwen-3B" in MOD.SUPPLEMENTARY_LABELS
        assert set(MOD.PRIMARY_LABELS) < set(MOD.SUPPLEMENTARY_LABELS)

    def test_no_olmo_aligned_stage_graders(self, out):
        """Neither tier may include an OLMo SFT/DPO/Instruct grader row."""
        forbidden = {"OLMo-sft", "OLMo-dpo", "OLMo-instruct"}
        assert forbidden.isdisjoint(MOD.PRIMARY_LABELS)
        assert forbidden.isdisjoint(MOD.SUPPLEMENTARY_LABELS)

    def test_common_prompt_subset_matches_gpt2_context_limit(self, out):
        """AlpacaEval's common subset should equal canonical minus GPT-2's
        context-overflow drops (Llama and OLMo-base have full coverage by
        construction of the canonical set)."""
        alpaca = out["prompt_sets"]["alpacaeval"]
        n_canonical = alpaca["n_prompts_canonical"]
        n_dropped_gpt2 = alpaca["dropped_n"]["GPT-2"]
        assert alpaca["primary"]["n_prompts"] == n_canonical - n_dropped_gpt2
        assert alpaca["supplementary"]["n_prompts"] == n_canonical - n_dropped_gpt2
        # NoveltyBench: no drops recorded for either extended grader.
        nb = out["prompt_sets"]["nbcurated"]
        assert nb["dropped_n"]["GPT-2"] == 0
        assert nb["dropped_n"]["Llama-3.1-8B"] == 0
        assert nb["primary"]["n_prompts"] == nb["n_prompts_canonical"]


class TestDecomposeForLabelsGuard:
    def test_small_intersection_raises_without_touching_data(self):
        """A near-empty prompt intersection must fail loudly, not silently
        proceed on a handful of prompts. theta_cells is left empty to prove
        the guard fires before any data access."""
        canonical = {f"p{i}" for i in range(5)}
        dropped = {label: set() for label in MOD.PRIMARY_LABELS}
        with pytest.raises(ValueError, match="unexpectedly small"):
            MOD.decompose_for_labels({}, "alpacaeval", MOD.PRIMARY_LABELS, canonical, dropped)
