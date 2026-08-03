"""Tests for baseline_metrics: SentBERT reduction math and distinct-n."""

import numpy as np
import pytest

from icl_diversity.baseline_metrics import (
    averaged_distinct_ngrams,
    distinct_ngrams,
    mean_pairwise_cosine_similarity,
    sentbert_diversity,
)


class TestMeanPairwiseCosine:
    def test_identical_vectors(self):
        e = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
        assert mean_pairwise_cosine_similarity(e) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        e = np.eye(3)
        assert mean_pairwise_cosine_similarity(e) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        e = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert mean_pairwise_cosine_similarity(e) == pytest.approx(-1.0)

    def test_mixed_pairs_average(self):
        # pairs: (e0,e1)=0, (e0,e2)=1, (e1,e2)=0 -> mean 1/3
        e = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
        assert mean_pairwise_cosine_similarity(e) == pytest.approx(1.0 / 3.0)

    def test_scale_invariance(self):
        rng = np.random.default_rng(0)
        e = rng.normal(size=(5, 8))
        scaled = e * rng.uniform(0.1, 10.0, size=(5, 1))
        assert mean_pairwise_cosine_similarity(e) == pytest.approx(
            mean_pairwise_cosine_similarity(scaled)
        )

    def test_diversity_is_negated_similarity(self):
        rng = np.random.default_rng(1)
        e = rng.normal(size=(6, 4))
        assert sentbert_diversity(e) == pytest.approx(
            -mean_pairwise_cosine_similarity(e)
        )

    def test_errors(self):
        with pytest.raises(ValueError):
            mean_pairwise_cosine_similarity(np.zeros((3,)))
        with pytest.raises(ValueError):
            mean_pairwise_cosine_similarity(np.ones((1, 4)))
        with pytest.raises(ValueError):
            mean_pairwise_cosine_similarity(np.array([[1.0, 0.0], [0.0, 0.0]]))


class TestDistinctNgrams:
    def test_tevet_doc_example_unigrams(self):
        # From diversity-eval/diversity_metrics.py __main__ example set.
        # 1-grams: [i, am, going] * 2 + [lets, go, i, i] = 10 tokens,
        # unique = {i, am, going, lets, go} = 5 -> 0.5
        resp_set = ["i am going", "i am going", "lets go i i"]
        assert distinct_ngrams(resp_set, 1) == pytest.approx(0.5)

    def test_tevet_doc_example_trigrams(self):
        # 3-grams: (i,am,going) * 2 + (lets,go,i), (go,i,i) = 4 total,
        # unique = 3 -> 0.75
        resp_set = ["i am going", "i am going", "lets go i i"]
        assert distinct_ngrams(resp_set, 3) == pytest.approx(0.75)

    def test_period_stripped_like_tevet(self):
        # utils.lines_to_ngrams strips '.' before splitting
        assert distinct_ngrams(["the cat.", "the cat"], 2) == pytest.approx(0.5)

    def test_all_unique(self):
        assert distinct_ngrams(["a b", "c d"], 2) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert distinct_ngrams(["a", "b"], 3) == pytest.approx(0.0)

    def test_averaged_matches_manual_mean(self):
        resp_set = ["i am going", "i am going", "lets go i i"]
        manual = np.mean([distinct_ngrams(resp_set, n) for n in range(1, 6)])
        assert averaged_distinct_ngrams(resp_set) == pytest.approx(manual)

    def test_errors(self):
        with pytest.raises(ValueError):
            distinct_ngrams(["a"], 0)
        with pytest.raises(ValueError):
            averaged_distinct_ngrams(["a"], n_min=3, n_max=2)
