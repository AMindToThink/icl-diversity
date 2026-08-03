"""Tests for template_scenarios: word lists, frames, and the generator."""

import random

import pytest

from icl_diversity.template_scenarios import (
    ADJECTIVES,
    FRAME_LABELS,
    FRAMES,
    NOUNS,
    PARAPHRASE_RESPONSES,
    VERBS,
    generate_template_responses,
)


class TestWordLists:
    def test_list_sizes(self):
        assert len(ADJECTIVES) >= 150
        assert len(NOUNS) >= 150
        assert len(VERBS) >= 100

    def test_no_duplicates(self):
        assert len(set(ADJECTIVES)) == len(ADJECTIVES)
        assert len(set(NOUNS)) == len(NOUNS)
        assert len(set(v.base for v in VERBS)) == len(VERBS)

    def test_all_lowercase_single_words(self):
        for w in ADJECTIVES + NOUNS:
            assert w == w.lower() and " " not in w, w
        for v in VERBS:
            for form in v:
                assert form == form.lower() and " " not in form, v

    def test_article_heuristic_safe(self):
        # The _an helper uses first-letter-vowel; curated lists must avoid
        # exceptions (you-sounding u words, silent h). Spot-check known traps.
        traps = {
            "unique",
            "useful",
            "usual",
            "uniform",
            "ukulele",
            "unicycle",
            "honest",
            "hour",
            "hourly",
            "heir",
            "european",
            "one",
        }
        for w in ADJECTIVES + NOUNS:
            assert w not in traps, w


class TestFrames:
    def test_frame_count_matches_labels(self):
        assert len(FRAMES) == len(FRAME_LABELS) == 20

    def test_every_frame_uses_all_slots(self):
        v = VERBS[0]
        for i, frame in enumerate(FRAMES):
            s = frame("gloomy", "walrus", v, "amber", "pretzel")
            assert "gloomy" in s and "walrus" in s, FRAME_LABELS[i]
            assert "amber" in s and "pretzel" in s, FRAME_LABELS[i]
            assert v.base in s or v.past in s or v.part in s, FRAME_LABELS[i]

    def test_frames_are_structurally_distinct(self):
        v = VERBS[0]
        rendered = [f("gloomy", "walrus", v, "amber", "pretzel") for f in FRAMES]
        assert len(set(rendered)) == len(rendered)

    def test_frames_end_with_sentence_punctuation(self):
        v = VERBS[0]
        for i, frame in enumerate(FRAMES):
            s = frame("gloomy", "walrus", v, "amber", "pretzel")
            assert s[-1] in ".?!", FRAME_LABELS[i]

    def test_canonical_frame_zero(self):
        v = VERBS[0]
        s = FRAMES[0]("gloomy", "walrus", v, "amber", "pretzel")
        assert s == f"The gloomy walrus {v.past} the amber pretzel."


class TestGenerator:
    def test_deterministic(self):
        r1, f1 = generate_template_responses(5, n=20, seed=7)
        r2, f2 = generate_template_responses(5, n=20, seed=7)
        assert r1 == r2 and f1 == f2

    def test_seeds_differ(self):
        r1, _ = generate_template_responses(5, n=20, seed=7)
        r2, _ = generate_template_responses(5, n=20, seed=8)
        assert r1 != r2

    def test_frame_count_respected(self):
        for m in [1, 2, 5, 10, 20]:
            responses, frame_ids = generate_template_responses(m, n=20, seed=3)
            assert len(responses) == 20
            assert len(set(frame_ids)) == m
            # even distribution: each frame used 20/m times when m divides 20
            counts = {fid: frame_ids.count(fid) for fid in set(frame_ids)}
            assert all(c == 20 // m for c in counts.values())

    def test_frame_pool_restriction(self):
        responses, frame_ids = generate_template_responses(
            1, n=10, seed=0, frame_pool=[0]
        )
        assert set(frame_ids) == {0}
        for r in responses:
            assert r.startswith("The ") and r.endswith(".")

    def test_no_unfilled_braces(self):
        responses, _ = generate_template_responses(20, n=20, seed=1)
        for r in responses:
            assert "{" not in r and "}" not in r

    def test_capitalized_sentences(self):
        responses, _ = generate_template_responses(20, n=20, seed=2)
        for r in responses:
            assert r[0].isupper()

    def test_all_unique(self):
        responses, _ = generate_template_responses(1, n=100, seed=4, frame_pool=[0])
        assert len(set(responses)) == 100

    def test_invalid_n_frames_raises(self):
        with pytest.raises(ValueError):
            generate_template_responses(0, n=10, seed=0)
        with pytest.raises(ValueError):
            generate_template_responses(21, n=10, seed=0)
        with pytest.raises(ValueError):
            generate_template_responses(2, n=10, seed=0, frame_pool=[0])

    def test_invalid_pool_raises(self):
        with pytest.raises(ValueError):
            generate_template_responses(1, n=10, seed=0, frame_pool=[99])

    def test_no_bad_articles(self):
        # "a" must precede consonant-initial words, "an" vowel-initial ones.
        rng = random.Random(0)
        seeds = [rng.randint(0, 10**6) for _ in range(20)]
        for seed in seeds:
            responses, _ = generate_template_responses(20, n=20, seed=seed)
            for r in responses:
                words = r.replace("!", "").replace("?", "").replace(".", "").split()
                for article, nxt in zip(words, words[1:]):
                    if article.lower() == "a":
                        assert nxt[0].lower() not in "aeiou", r
                    elif article.lower() == "an":
                        assert nxt[0].lower() in "aeiou", r


class TestParaphrases:
    def test_twenty_unique(self):
        assert len(PARAPHRASE_RESPONSES) == 20
        assert len(set(PARAPHRASE_RESPONSES)) == 20

    def test_shared_meaning_tokens(self):
        # every paraphrase mentions the committee, the meeting, and next week
        for p in PARAPHRASE_RESPONSES:
            low = p.lower()
            assert "committee" in low and "meeting" in low and "next week" in low
