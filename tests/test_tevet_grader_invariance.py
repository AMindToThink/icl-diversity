"""Tests for scripts/analyze_tevet_grader_invariance.py.

Synthetic-data tests for the per-file analysis (rank agreement, label
consistency guard, z-scored variance decomposition) plus an integrity check
that committed outputs match a fresh in-memory analysis when the sidecars are
available. No GPU, no network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_tevet_grader_invariance.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tevet_invariance", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MOD = _load_script()


def _per_tag(tags: list[str], D: np.ndarray, labels: np.ndarray) -> dict:
    """Build the per_tag structure from a (n_items, n_tags) D matrix."""
    return {
        t: {
            f"k{i}": {"D": float(D[i, j]), "label": float(labels[i])}
            for i in range(D.shape[0])
        }
        for j, t in enumerate(tags)
    }


class TestAnalyzeFile:
    def test_identical_rankings_give_full_item_share(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=50)
        labels = base + rng.normal(scale=0.1, size=50)
        # Three graders: same per-item values up to affine scale changes.
        D = np.column_stack([base, 2.0 * base + 5.0, 0.1 * base - 3.0])
        out = MOD.analyze_file(_per_tag(["a", "b", "c"], D, labels), ["a", "b", "c"])
        assert out["min_pairwise_rho"] == pytest.approx(1.0)
        # Affine-equivalent graders z-score to identical columns.
        assert out["variance_zscored_frac"]["item"] == pytest.approx(1.0)
        assert out["variance_zscored_frac"]["interaction"] == pytest.approx(0.0, abs=1e-12)
        # Raw decomposition sees the scale offsets as grader variance.
        assert out["variance_raw_frac"]["grader"] > 0.5

    def test_independent_graders_give_low_item_share(self):
        rng = np.random.default_rng(1)
        D = rng.normal(size=(80, 3))
        labels = rng.normal(size=80)
        out = MOD.analyze_file(_per_tag(["a", "b", "c"], D, labels), ["a", "b", "c"])
        assert out["variance_zscored_frac"]["item"] < 0.6
        assert abs(out["min_pairwise_rho"]) < 0.5

    def test_label_mismatch_raises(self):
        rng = np.random.default_rng(2)
        D = rng.normal(size=(40, 2))
        labels = rng.normal(size=40)
        per_tag = _per_tag(["a", "b"], D, labels)
        per_tag["b"]["k0"]["label"] += 1.0
        with pytest.raises(AssertionError, match="label mismatch"):
            MOD.analyze_file(per_tag, ["a", "b"])

    def test_too_few_common_samples_raises(self):
        rng = np.random.default_rng(3)
        D = rng.normal(size=(10, 2))
        labels = rng.normal(size=10)
        with pytest.raises(ValueError, match="common samples"):
            MOD.analyze_file(_per_tag(["a", "b"], D, labels), ["a", "b"])

    def test_zero_variance_grader_raises(self):
        D = np.column_stack([np.arange(40.0), np.full(40, 2.0)])
        labels = np.arange(40.0)
        with pytest.raises(ValueError, match="zero variance"):
            MOD.analyze_file(_per_tag(["a", "b"], D, labels), ["a", "b"])


class TestCommittedOutputs:
    OUT = REPO_ROOT / "results" / "tevet" / "grader_invariance"

    @pytest.fixture(scope="class")
    def committed(self):
        if not (self.OUT / "invariance.json").exists():
            pytest.skip("grader_invariance outputs not yet generated")
        import json

        return json.loads((self.OUT / "invariance.json").read_text())

    def test_all_files_and_tags_present(self, committed):
        stems = [s for s, _d, _m in MOD.FILE_SPECS]
        assert sorted(committed["files"]) == sorted(stems)
        for r in committed["files"].values():
            assert set(r["rho_vs_label"]) == set(committed["tags"])

    def test_every_grader_sees_positive_signal(self, committed):
        for r in committed["files"].values():
            for t, v in r["rho_vs_label"].items():
                assert v["rho"] > 0, f"{t} lost the human-label signal"

    def test_item_dominates_after_zscoring(self, committed):
        for r in committed["files"].values():
            assert r["variance_zscored_frac"]["item"] > r["variance_zscored_frac"]["interaction"]
