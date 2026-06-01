"""Unit tests for the parameterized theta (scorer) in the RLHF scoring script.

The OLMo self-scoring matrix re-scores fixed generations under different scorer
models via --scorer-model. These tests verify the flag is parsed and threaded
all the way into the per-prompt output record's `scorer_model` field, without
loading any real model or touching a GPU (the heavy calls are mocked).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make `src/` importable (mirrors the repo conftest) so the lazy
# `from icl_diversity.core import ...` inside score_all resolves, and so
# mock.patch("icl_diversity.core....") can find its target.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import icl_diversity.core  # noqa: E402,F401  (force into sys.modules before patching)

# The script lives at scripts/rlhf_experiment/3_score_icl_diversity.py — a path
# that is not a valid module name (leading digit), so load it by file path.
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rlhf_experiment"
    / "3_score_icl_diversity.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("rlhf_scorer", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scorer_mod():
    return _load_module()


def test_default_scorer_is_qwen(scorer_mod):
    assert scorer_mod.DEFAULT_SCORER_MODEL == "Qwen/Qwen2.5-3B"


def test_argparse_scorer_model_override(scorer_mod, monkeypatch):
    argv = [
        "prog",
        "--scorer-model",
        "allenai/OLMo-2-1124-7B",
        "--stages",
        "base",
        "--prompt-sets",
        "alpacaeval",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}
    monkeypatch.setattr(scorer_mod, "score_all", lambda **kw: captured.update(kw))
    scorer_mod.main()
    assert captured["scorer_model"] == "allenai/OLMo-2-1124-7B"
    assert captured["stages"] == ["base"]
    assert captured["prompt_sets"] == ["alpacaeval"]


def _write_gen_file(gen_dir: Path, stage: str, pset: str, n_prompts: int, k: int):
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / f"{stage}_{pset}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in range(n_prompts):
            for s in range(k):
                f.write(
                    json.dumps(
                        {
                            "prompt_id": f"p{p}",
                            "prompt": f"prompt {p}",
                            "stage": stage,
                            "prompt_set": pset,
                            "sample_idx": s,
                            "response": f"response {p}-{s}",
                        }
                    )
                    + "\n"
                )
    return path


def test_scorer_model_written_into_records(scorer_mod, tmp_path):
    gen_dir = tmp_path / "generations"
    _write_gen_file(gen_dir, "base", "alpacaeval", n_prompts=2, k=3)
    out_path = tmp_path / "out.jsonl"

    fake_model = mock.MagicMock()
    fake_model.to.return_value = fake_model  # .to("cuda:0") -> model
    fake_model.eval.return_value = fake_model
    fake_metrics = {
        "coherence_C": 0.5,
        "a_n_per_byte": 0.4,
        "diversity_score_D_C_an": 0.2,
        "diversity_score_D": 0.1,
    }

    with mock.patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model
    ), mock.patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock.MagicMock()
    ), mock.patch(
        "icl_diversity.core.compute_icl_diversity_metrics", return_value=fake_metrics
    ):
        scorer_mod.score_all(
            stages=["base"],
            prompt_sets=["alpacaeval"],
            n_permutations=3,
            batch_size=2,
            out_path=out_path,
            limit=None,
            gen_dir=gen_dir,
            scorer_model="allenai/OLMo-2-1124-7B-Instruct",
        )

    rows = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(rows) == 2  # 2 prompts, 1 group each
    for r in rows:
        assert r["scorer_model"] == "allenai/OLMo-2-1124-7B-Instruct"
        assert r["n_responses"] == 3
        assert r["diversity_score_D_C_an"] == 0.2


def test_idempotent_skips_already_scored(scorer_mod, tmp_path):
    """Re-running with the same (stage,set,prompt,perms) keys scores nothing new."""
    gen_dir = tmp_path / "generations"
    _write_gen_file(gen_dir, "base", "alpacaeval", n_prompts=2, k=3)
    out_path = tmp_path / "out.jsonl"

    fake_model = mock.MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_metrics = {"diversity_score_D_C_an": 0.2, "diversity_score_D": 0.1}

    call_count = {"n": 0}

    def counting_metrics(**kwargs):
        call_count["n"] += 1
        return fake_metrics

    with mock.patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model
    ), mock.patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock.MagicMock()
    ), mock.patch(
        "icl_diversity.core.compute_icl_diversity_metrics", side_effect=counting_metrics
    ):
        common = dict(
            stages=["base"],
            prompt_sets=["alpacaeval"],
            n_permutations=3,
            batch_size=2,
            out_path=out_path,
            limit=None,
            gen_dir=gen_dir,
            scorer_model="Qwen/Qwen2.5-3B",
        )
        scorer_mod.score_all(**common)
        first = call_count["n"]
        scorer_mod.score_all(**common)  # second run: all keys already present
        second = call_count["n"]

    assert first == 2
    assert second == 2  # no new scoring calls on the re-run
