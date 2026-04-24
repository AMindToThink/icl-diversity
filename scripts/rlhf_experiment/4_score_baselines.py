"""Baseline diversity metrics: EAD, distinct-n, SentBERT-sim→diversity.

Ports Kirk et al.'s rlvsil/diversity formulas as standalone ~100 LOC so
we don't depend on their package (which has an unpinned transformers and
mandatory wandb). Outputs one row per (stage, prompt_set, prompt_id)
to results/rlhf_experiment/baseline_metrics.jsonl.

Metrics:
  - EAD      = expectation-adjusted distinct n-grams (Kirk's Table 1),
               averaged over n=1..5, vocab V=50257 (Kirk's value).
               Formula: unique_ngrams / (V * (1 - ((V-1)/V)**total_ngrams))
  - distinct = len(set(ngrams)) / len(ngrams), averaged over n=1..5.
  - sentbert_diversity = 1 - mean(pairwise cosine sim of sentence-
               transformers embeddings). Model: all-MiniLM-L6-v2
               (cached). Matches Kirk's 'sent_bert_from_sim' metric in
               direction.

Idempotent: skips (stage, prompt_set, prompt_id) keys already scored.

GPU 1 only: CUDA_VISIBLE_DEVICES=1 set before any torch import.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rlhf_experiment"
GEN_DIR = RESULTS_DIR / "generations"

SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KIRK_VOCAB = 50257
NGRAM_RANGE = range(1, 6)  # n=1..5 averaged, matching Kirk


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add((rec["stage"], rec["prompt_set"], rec["prompt_id"]))
    return keys


def _tokenize_ws(text: str) -> list[str]:
    """Whitespace tokenisation — matches Kirk's simple per-word approach."""
    return text.split()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n_averaged(responses: list[str]) -> tuple[float, dict[int, float]]:
    """Return (mean over n=1..5 of distinct_n, per-n dict)."""
    per_n = {}
    for n in NGRAM_RANGE:
        all_grams: list[tuple[str, ...]] = []
        for r in responses:
            toks = _tokenize_ws(r)
            all_grams.extend(_ngrams(toks, n))
        if not all_grams:
            per_n[n] = 0.0
        else:
            per_n[n] = len(set(all_grams)) / len(all_grams)
    mean = sum(per_n.values()) / len(per_n)
    return mean, per_n


def ead_averaged(responses: list[str], vocab_size: int = KIRK_VOCAB) -> tuple[float, dict[int, float]]:
    """Expectation-adjusted distinct n-grams, Kirk's exact formula.

    EAD_n = len(set(ngrams)) / (V * (1 - ((V-1)/V)**C))
    where C = len(ngrams), V = vocab size.
    """
    per_n = {}
    for n in NGRAM_RANGE:
        all_grams: list[tuple[str, ...]] = []
        for r in responses:
            toks = _tokenize_ws(r)
            all_grams.extend(_ngrams(toks, n))
        uniq = len(set(all_grams))
        total = len(all_grams)
        if total == 0:
            per_n[n] = 0.0
            continue
        # Adjustment term: fraction of vocab that would be hit under random draws
        denom = vocab_size * (1.0 - ((vocab_size - 1) / vocab_size) ** total)
        if denom == 0:
            per_n[n] = 0.0
        else:
            per_n[n] = uniq / denom
    mean = sum(per_n.values()) / len(per_n)
    return mean, per_n


def sentbert_diversity(responses: list[str], encoder) -> float:
    import numpy as np

    embs = encoder.encode(responses, convert_to_numpy=True, show_progress_bar=False)
    # Pairwise cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.clip(norms, 1e-8, None)
    sim = normed @ normed.T
    # Off-diagonal mean
    n = sim.shape[0]
    if n < 2:
        return 0.0
    off = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1.0 - off)


def group_by_prompt(rows: list[dict]) -> dict[str, list[dict]]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    for pid in by_prompt:
        by_prompt[pid].sort(key=lambda r: r["sample_idx"])
    return by_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["base", "sft", "dpo", "instruct"])
    ap.add_argument("--prompt-sets", nargs="+", default=["alpacaeval", "nbcurated"])
    ap.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "baseline_metrics.jsonl"
    )
    ap.add_argument(
        "--skip-sentbert",
        action="store_true",
        help="Skip the SentBERT pass (fast smoke-test mode).",
    )
    args = ap.parse_args()

    encoder = None
    if not args.skip_sentbert:
        from sentence_transformers import SentenceTransformer

        print(f"[baselines] loading {SENTENCE_MODEL}")
        encoder = SentenceTransformer(SENTENCE_MODEL, device="cuda:0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_keys(args.out)
    print(f"[baselines] {len(seen)} keys already scored")

    with args.out.open("a", encoding="utf-8") as f:
        for stage in args.stages:
            for pset in args.prompt_sets:
                gen_path = GEN_DIR / f"{stage}_{pset}.jsonl"
                if not gen_path.exists():
                    print(f"[baselines] skipping missing {gen_path}")
                    continue
                rows = read_jsonl(gen_path)
                by_prompt = group_by_prompt(rows)
                todo = [pid for pid in by_prompt if (stage, pset, pid) not in seen]
                print(
                    f"[baselines] stage={stage} set={pset}: "
                    f"{len(todo)}/{len(by_prompt)} prompts to score"
                )
                for pid in todo:
                    group = by_prompt[pid]
                    responses = [g["response"] for g in group]

                    ead_mean, ead_per_n = ead_averaged(responses)
                    dist_mean, dist_per_n = distinct_n_averaged(responses)
                    sb = sentbert_diversity(responses, encoder) if encoder else None

                    rec = {
                        "stage": stage,
                        "prompt_set": pset,
                        "prompt_id": pid,
                        "n_responses": len(responses),
                        "ead": ead_mean,
                        "ead_per_n": ead_per_n,
                        "distinct_n": dist_mean,
                        "distinct_per_n": dist_per_n,
                        "sentbert_diversity": sb,
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
