"""Unit tests for mode_count_scenarios.py."""

import pytest

from icl_diversity.mode_count_scenarios import (
    MODE_NAMES,
    PROMPT,
    get_format_modes,
    generate_mode_count_responses,
    _ALL_RAIN_MODES,
)


class TestGetFormatModes:
    def test_returns_correct_count(self) -> None:
        for m in [1, 5, 15, 25, 50]:
            modes = get_format_modes(m)
            assert len(modes) == m

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            get_format_modes(0)
        with pytest.raises(ValueError):
            get_format_modes(51)

    def test_modes_are_callable(self) -> None:
        import random
        rng = random.Random(42)
        for m in [1, 15, 50]:
            modes = get_format_modes(m)
            for mode in modes:
                result = mode(rng)
                assert isinstance(result, str)
                assert len(result) > 0


class TestGenerateResponses:
    def test_correct_count(self) -> None:
        for m in [1, 3, 5, 10]:
            responses = generate_mode_count_responses(m, n_per_mode=4, seed=42)
            assert len(responses) == m * 4

    def test_deterministic(self) -> None:
        r1 = generate_mode_count_responses(3, n_per_mode=4, seed=42)
        r2 = generate_mode_count_responses(3, n_per_mode=4, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        r1 = generate_mode_count_responses(3, n_per_mode=4, seed=42)
        r2 = generate_mode_count_responses(3, n_per_mode=4, seed=99)
        assert r1 != r2

    def test_responses_are_nonempty_strings(self) -> None:
        responses = generate_mode_count_responses(5, n_per_mode=4, seed=42)
        for r in responses:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_modes_produce_distinct_formats(self) -> None:
        """Check that different modes produce structurally different outputs."""
        import random
        rng = random.Random(42)
        modes = get_format_modes(15)
        samples = [mode(rng) for mode in modes]
        # All should be unique strings (different formats)
        assert len(set(samples)) == 15


class TestModeNames:
    def test_50_mode_names(self) -> None:
        assert len(MODE_NAMES) == 50

    def test_matches_all_rain_modes(self) -> None:
        assert len(_ALL_RAIN_MODES) == 50


class TestPrompt:
    def test_prompt_exists(self) -> None:
        assert isinstance(PROMPT, str)
        assert "rain" in PROMPT.lower()
