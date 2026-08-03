"""Tests for pos_pattern_scenarios: word lists, patterns, and the generator."""

import random
import re

import pytest

from icl_diversity.pos_pattern_scenarios import (
    ADVERBS,
    CANONICAL_CLASS_MULTISET,
    INTRANSITIVE_PAST,
    PLURAL_NOUNS,
    POS_PATTERN_LABELS,
    POS_PATTERNS,
    PREPOSITIONS,
    TRANSITIVE_PAST,
    generate_pos_pattern_responses,
    generate_scrambled_canonical_responses,
    render_pattern,
)


class TestWordLists:
    def test_list_sizes(self):
        assert len(PLURAL_NOUNS) >= 150
        assert len(INTRANSITIVE_PAST) >= 90
        assert len(PREPOSITIONS) >= 30
        assert len(ADVERBS) >= 50
        assert len(TRANSITIVE_PAST) >= 100

    def test_no_duplicates(self):
        for lst in [
            PLURAL_NOUNS,
            INTRANSITIVE_PAST,
            PREPOSITIONS,
            ADVERBS,
            TRANSITIVE_PAST,
        ]:
            assert len(set(lst)) == len(lst)

    def test_all_lowercase_single_words(self):
        for lst in [PLURAL_NOUNS, INTRANSITIVE_PAST, PREPOSITIONS, ADVERBS]:
            for w in lst:
                assert w == w.lower() and " " not in w, w

    def test_plural_nouns_look_plural(self):
        # explicit forms; everything shipped ends in 's' (curated: no
        # irregular zero-plurals like "bison")
        for w in PLURAL_NOUNS:
            assert w.endswith("s"), w


class TestPatterns:
    def test_pattern_count_matches_labels(self):
        assert len(POS_PATTERNS) == len(POS_PATTERN_LABELS) == 12

    def test_canonical_pattern_zero(self):
        assert POS_PATTERNS[0] == "{N1} {Vi1} {P1} {N2} {P2} {N3}."
        assert POS_PATTERN_LABELS[0] == "n_vi_p_n_p_n"

    def test_no_fixed_lexical_material(self):
        # THE invariant of this experiment: outside {Tag} placeholders a
        # pattern may contain only spaces, commas, and periods, so repeated
        # patterns share zero literal words for distinct-n to detect.
        for pattern in POS_PATTERNS:
            stripped = re.sub(r"\{[A-Za-z]+\d+\}", "", pattern)
            assert re.fullmatch(r"[ ,.]*", stripped), pattern

    def test_all_patterns_six_words(self):
        rng = random.Random(0)
        for i, pattern in enumerate(POS_PATTERNS):
            sentence = render_pattern(pattern, rng)
            words = sentence.replace(",", "").rstrip(".").split()
            assert len(words) == 6, (POS_PATTERN_LABELS[i], sentence)

    def test_pos_sequences_distinct(self):
        # the class sequence (including punctuation) differs across patterns
        sequences = []
        for pattern in POS_PATTERNS:
            seq = re.findall(r"\{([A-Za-z]+)\d+\}|([,.])", pattern)
            sequences.append(tuple(seq))
        assert len(set(sequences)) == len(sequences)

    def test_labels_match_class_sequence(self):
        for pattern, label in zip(POS_PATTERNS, POS_PATTERN_LABELS):
            classes = [c.lower() for c in re.findall(r"\{([A-Za-z]+)\d+\}", pattern)]
            assert label == "_".join(classes), (pattern, label)

    def test_render_deterministic_and_capitalized(self):
        s1 = render_pattern(POS_PATTERNS[0], random.Random(5))
        s2 = render_pattern(POS_PATTERNS[0], random.Random(5))
        assert s1 == s2
        assert s1[0].isupper()
        assert s1.endswith(".")

    def test_render_samples_correct_classes(self):
        rng = random.Random(1)
        sentence = render_pattern(POS_PATTERNS[0], rng)
        words = sentence.rstrip(".").split()
        # N Vi P N P N; first word is capitalized, lowercase for lookup
        assert words[0].lower() in PLURAL_NOUNS
        assert words[1] in INTRANSITIVE_PAST
        assert words[2] in PREPOSITIONS
        assert words[3] in PLURAL_NOUNS
        assert words[4] in PREPOSITIONS
        assert words[5] in PLURAL_NOUNS

    def test_same_class_slots_distinct_words(self):
        # canonical has three N slots and two P slots; within one sentence
        # they must all be distinct
        for seed in range(20):
            sentence = render_pattern(POS_PATTERNS[0], random.Random(seed))
            words = [w.lower() for w in sentence.rstrip(".").split()]
            nouns = [words[0], words[3], words[5]]
            preps = [words[2], words[4]]
            assert len(set(nouns)) == 3, sentence
            assert len(set(preps)) == 2, sentence


class TestGenerator:
    def test_deterministic(self):
        r1, p1 = generate_pos_pattern_responses(4, n=40, seed=7)
        r2, p2 = generate_pos_pattern_responses(4, n=40, seed=7)
        assert r1 == r2 and p1 == p2

    def test_seeds_differ(self):
        r1, _ = generate_pos_pattern_responses(4, n=40, seed=7)
        r2, _ = generate_pos_pattern_responses(4, n=40, seed=8)
        assert r1 != r2

    def test_pattern_count_respected(self):
        for m in [1, 2, 4, 8, 12]:
            responses, pattern_ids = generate_pos_pattern_responses(m, n=40, seed=3)
            assert len(responses) == 40
            assert len(set(pattern_ids)) == m
            # near-even distribution: counts differ by at most 1
            counts = [pattern_ids.count(pid) for pid in set(pattern_ids)]
            assert max(counts) - min(counts) <= 1

    def test_pattern_pool_restriction(self):
        responses, pattern_ids = generate_pos_pattern_responses(
            1, n=10, seed=0, pattern_pool=[0]
        )
        assert set(pattern_ids) == {0}
        for r in responses:
            words = r.rstrip(".").split()
            assert len(words) == 6
            assert words[0].lower() in PLURAL_NOUNS

    def test_all_unique(self):
        responses, _ = generate_pos_pattern_responses(
            1, n=100, seed=4, pattern_pool=[0]
        )
        assert len(set(responses)) == 100

    def test_no_unfilled_braces(self):
        responses, _ = generate_pos_pattern_responses(12, n=40, seed=1)
        for r in responses:
            assert "{" not in r and "}" not in r

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            generate_pos_pattern_responses(0, n=10, seed=0)
        with pytest.raises(ValueError):
            generate_pos_pattern_responses(13, n=10, seed=0)
        with pytest.raises(ValueError):
            generate_pos_pattern_responses(2, n=10, seed=0, pattern_pool=[0])
        with pytest.raises(ValueError):
            generate_pos_pattern_responses(1, n=10, seed=0, pattern_pool=[99])


class TestScrambledControl:
    def test_multiset_matches_canonical_pattern(self):
        # CANONICAL_CLASS_MULTISET must equal the class counts parsed from
        # POS_PATTERNS[0], or the control is not composition-matched.
        classes = re.findall(r"\{([A-Za-z]+)\d+\}", POS_PATTERNS[0])
        counts = {cls: classes.count(cls) for cls in set(classes)}
        assert counts == dict(CANONICAL_CLASS_MULTISET)

    def test_deterministic(self):
        assert generate_scrambled_canonical_responses(
            n=40, seed=7
        ) == generate_scrambled_canonical_responses(n=40, seed=7)

    def test_seeds_differ(self):
        assert generate_scrambled_canonical_responses(
            n=40, seed=7
        ) != generate_scrambled_canonical_responses(n=40, seed=8)

    def test_class_composition_per_sentence(self):
        responses = generate_scrambled_canonical_responses(n=40, seed=3)
        for r in responses:
            words = [w.lower() for w in r.rstrip(".").split()]
            assert len(words) == 6, r
            nouns = [w for w in words if w in PLURAL_NOUNS]
            verbs = [w for w in words if w in INTRANSITIVE_PAST]
            preps = [w for w in words if w in PREPOSITIONS]
            assert len(nouns) == 3, r
            assert len(verbs) == 1, r
            assert len(preps) == 2, r
            assert len(set(words)) == 6, r

    def test_orders_are_not_consistent(self):
        # the whole point of the control: no shared word-order pattern
        responses = generate_scrambled_canonical_responses(n=40, seed=1)

        def class_order(r: str) -> tuple[str, ...]:
            order = []
            for w in [w.lower() for w in r.rstrip(".").split()]:
                if w in PLURAL_NOUNS:
                    order.append("N")
                elif w in INTRANSITIVE_PAST:
                    order.append("Vi")
                else:
                    order.append("P")
            return tuple(order)

        orders = {class_order(r) for r in responses}
        assert len(orders) > 10

    def test_capitalized_with_period(self):
        for r in generate_scrambled_canonical_responses(n=20, seed=2):
            assert r[0].isupper() and r.endswith(".")

    def test_all_unique(self):
        responses = generate_scrambled_canonical_responses(n=100, seed=4)
        assert len(set(responses)) == 100
