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
            modes, names = get_format_modes(m, seed=42)
            assert len(modes) == m
            assert len(names) == m

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            get_format_modes(0)
        with pytest.raises(ValueError):
            get_format_modes(51)

    def test_modes_are_callable(self) -> None:
        import random
        rng = random.Random(42)
        for m in [1, 15, 50]:
            modes, _ = get_format_modes(m, seed=42)
            for mode in modes:
                result = mode(rng)
                assert isinstance(result, str)
                assert len(result) > 0

    def test_different_seeds_select_different_modes(self) -> None:
        """m=1 should not always be the same mode."""
        _, names1 = get_format_modes(1, seed=42)
        _, names2 = get_format_modes(1, seed=99)
        _, names3 = get_format_modes(1, seed=256)
        # At least two of three seeds should pick different modes
        assert len({names1[0], names2[0], names3[0]}) >= 2

    def test_seed_deterministic(self) -> None:
        modes1, names1 = get_format_modes(5, seed=42)
        modes2, names2 = get_format_modes(5, seed=42)
        assert names1 == names2

    def test_m50_returns_all_modes(self) -> None:
        """With m=50, all modes are selected regardless of seed."""
        _, names = get_format_modes(50, seed=42)
        assert set(names) == set(MODE_NAMES)


class TestGenerateResponses:
    def test_correct_count(self) -> None:
        """All m values produce exactly n responses."""
        for m in [1, 3, 5, 10]:
            responses, _ = generate_mode_count_responses(m, n=20, seed=42)
            assert len(responses) == 20

    def test_fixed_n_across_m(self) -> None:
        """Different m values with same n produce same-length response lists."""
        n = 12
        for m in [1, 2, 3, 4, 6, 12]:
            responses, _ = generate_mode_count_responses(m, n=n, seed=42)
            assert len(responses) == n, f"m={m} produced {len(responses)} responses, expected {n}"

    def test_n_less_than_m_raises(self) -> None:
        """n < m should raise ValueError."""
        with pytest.raises(ValueError, match="n must be >= m"):
            generate_mode_count_responses(m=5, n=3, seed=42)

    def test_deterministic(self) -> None:
        r1, n1 = generate_mode_count_responses(3, n=20, seed=42)
        r2, n2 = generate_mode_count_responses(3, n=20, seed=42)
        assert r1 == r2
        assert n1 == n2

    def test_different_seeds_differ(self) -> None:
        r1, _ = generate_mode_count_responses(3, n=20, seed=42)
        r2, _ = generate_mode_count_responses(3, n=20, seed=99)
        assert r1 != r2

    def test_responses_are_nonempty_strings(self) -> None:
        responses, _ = generate_mode_count_responses(5, n=20, seed=42)
        for r in responses:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_modes_produce_distinct_formats(self) -> None:
        """Check that different modes produce structurally different outputs."""
        import random
        rng = random.Random(42)
        modes, _ = get_format_modes(15, seed=42)
        samples = [mode(rng) for mode in modes]
        # All should be unique strings (different formats)
        assert len(set(samples)) == 15

    def test_returns_mode_names(self) -> None:
        _, names = generate_mode_count_responses(3, n=20, seed=42)
        assert len(names) == 3
        for name in names:
            assert name in MODE_NAMES

    def test_n_equals_m_gives_one_per_mode(self) -> None:
        """Edge case: n == m means exactly 1 response per mode."""
        responses, names = generate_mode_count_responses(m=5, n=5, seed=42)
        assert len(responses) == 5
        assert len(names) == 5


class TestModeNames:
    def test_50_mode_names(self) -> None:
        assert len(MODE_NAMES) == 50

    def test_matches_all_rain_modes(self) -> None:
        assert len(_ALL_RAIN_MODES) == 50


class TestPrompt:
    def test_prompt_exists(self) -> None:
        assert isinstance(PROMPT, str)
        assert "rain" in PROMPT.lower()
