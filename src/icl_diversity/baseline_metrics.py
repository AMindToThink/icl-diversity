"""Embedding- and n-gram-based diversity baselines, following Tevet & Berant.

These replicate the conventions of the diversity-eval reference
implementation (vendored under ``diversity-eval/``):

- **SentBERT diversity**: mean pairwise cosine similarity over sentence
  embeddings, negated. Similarity per pair is ``1 - cosine_distance``
  (``diversity-eval/diversity_metrics.py``, class ``SentBert``); the
  similarity-to-diversity reduction is ``-mean(similarities)``
  (``diversity-eval/metric.py``, ``similarity2diversity_function``).
  Embedding encoding itself happens in the calling script (it needs a
  sentence-transformers model); this module only does the pure math so it
  can be unit-tested without a network.

- **Averaged distinct-n**: fraction of unique word n-grams among all word
  n-grams, averaged over n in [n_min, n_max] (default [1, 5]), matching
  ``DistinctNgrams`` / ``AveragedDistinctNgrams`` in
  ``diversity-eval/diversity_metrics.py``. Word tokenization matches
  ``diversity-eval/utils.py::lines_to_ngrams``: strip '.' and newlines,
  split on spaces (other punctuation is intentionally kept, as in Tevet's
  code).
"""

from __future__ import annotations

import numpy as np


def mean_pairwise_cosine_similarity(embeddings: np.ndarray) -> float:
    """Mean cosine similarity over all unordered pairs of embedding rows.

    Args:
        embeddings: array of shape (n, dim) with n >= 2.

    Raises:
        ValueError: on wrong shape, fewer than 2 rows, or a zero-norm row.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 embeddings, got {n}")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding row; cannot compute cosine similarity")
    unit = embeddings / norms[:, None]
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sims[iu]))


def sentbert_diversity(embeddings: np.ndarray) -> float:
    """Tevet's SentBERT diversity: negated mean pairwise cosine similarity."""
    return -mean_pairwise_cosine_similarity(embeddings)


def _line_to_words(line: str) -> list[str]:
    """Tokenize as in diversity-eval/utils.py::lines_to_ngrams."""
    return [w for w in line.replace(".", "").replace("\n", "").split(" ") if w != ""]


def distinct_ngrams(responses: list[str], n: int) -> float:
    """Unique-ngram fraction over the pooled word n-grams of all responses."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ngrams: list[tuple[str, ...]] = []
    for line in responses:
        words = _line_to_words(line)
        ngrams.extend(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def averaged_distinct_ngrams(
    responses: list[str], n_min: int = 1, n_max: int = 5
) -> float:
    """Mean of distinct-n over n in [n_min, n_max] (Tevet's default 1..5)."""
    if not n_max >= n_min >= 1:
        raise ValueError(f"need n_max >= n_min >= 1, got [{n_min}, {n_max}]")
    return float(
        np.mean([distinct_ngrams(responses, n) for n in range(n_min, n_max + 1)])
    )
