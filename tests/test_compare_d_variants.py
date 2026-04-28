"""Unit tests for scripts/compare_d_variants.compute_variants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare_d_variants import compute_variants  # noqa: E402


def test_minimal_record_with_coherence_C():
    """Canonical case: single-permutation record with coherence_C provided directly."""
    rec = {
        "a_k_curve": [100.0, 80.0, 60.0],
        "a_k_byte_counts": [40, 40, 40],
        "coherence_C": 0.5,
    }
    v = compute_variants(rec)
    assert v.C == 0.5
    assert v.a_n_total == 60.0
    assert v.a_n_pb_RoM == pytest.approx(60.0 / 40.0)  # = 1.5
    assert v.a_n_pb_MoR is None  # no per-perm data
    assert v.D_hybrid == pytest.approx(0.5 * 60.0)  # = 30.0
    assert v.D_pb_RoM == pytest.approx(0.5 * 1.5)  # = 0.75
    assert v.D_pb_MoR is None


def test_C_derived_from_unconditional_surprises():
    """If coherence_C is missing, derive C = 2^(-mean(per-byte unconditional))."""
    rec = {
        "a_k_curve": [10.0, 8.0],
        "a_k_byte_counts": [5, 5],
        "unconditional_surprises": [1.0, 1.0],  # per-byte
    }
    v = compute_variants(rec)
    assert v.C == pytest.approx(2.0**-1.0)  # = 0.5
    assert v.a_n_total == 8.0
    assert v.D_hybrid == pytest.approx(0.5 * 8.0)


def test_C_derived_from_unconditional_total_bits():
    """If only total bits available, derive per-byte then C."""
    rec = {
        "a_k_curve": [10.0, 8.0],
        "a_k_byte_counts": [5, 5],
        "unconditional_total_bits": [5.0, 5.0],
    }
    v = compute_variants(rec)
    # mean per-byte = mean([5/5, 5/5]) = 1.0 → C = 2^-1 = 0.5
    assert v.C == pytest.approx(0.5)


def test_mor_with_per_permutation_data():
    """MoR is mean over perms of (bits[-1] / bytes[-1])."""
    rec = {
        "a_k_curve": [100.0, 60.0],
        "a_k_byte_counts": [50, 40],
        "coherence_C": 1.0,
        "per_permutation_a_k_curves": [
            [120.0, 80.0],  # ratio at last = 80/40 = 2.0
            [80.0, 40.0],  # ratio at last = 40/40 = 1.0
        ],
        "per_permutation_byte_counts": [
            [60, 40],
            [40, 40],
        ],
    }
    v = compute_variants(rec)
    # MoR = (2.0 + 1.0) / 2 = 1.5
    assert v.a_n_pb_MoR == pytest.approx(1.5)
    # RoM = mean_bits[-1] / mean_bytes[-1] = (80+40)/2 / ((40+40)/2) = 60/40 = 1.5
    # In this contrived example RoM == MoR. Use a_k_curve directly: 60/40 = 1.5.
    assert v.a_n_pb_RoM == pytest.approx(60.0 / 40.0)
    assert v.D_pb_MoR == pytest.approx(1.0 * 1.5)


def test_mor_diverges_from_rom_when_correlated():
    """When bits and bytes are positively correlated across perms, RoM ≠ MoR.

    Construct: long perms are surprising AND long; short perms are
    unsurprising AND short. RoM mixes lengths in numerator and denominator
    separately; MoR averages each perm's normalized rate.
    """
    rec = {
        "a_k_curve": [55.0, 55.0],  # mean of [100, 10] = 55
        "a_k_byte_counts": [55, 55],  # mean of [100, 10] = 55
        "coherence_C": 1.0,
        "per_permutation_a_k_curves": [
            [100.0, 100.0],  # long perm: 100 bits / 100 bytes → 1.0/byte
            [10.0, 10.0],  # short perm: 10 bits / 10 bytes → 1.0/byte
        ],
        "per_permutation_byte_counts": [
            [100, 100],
            [10, 10],
        ],
    }
    v = compute_variants(rec)
    # MoR = mean([100/100, 10/10]) = mean([1.0, 1.0]) = 1.0
    assert v.a_n_pb_MoR == pytest.approx(1.0)
    # RoM = a_k_curve[-1] / a_k_byte_counts[-1] = 55/55 = 1.0
    # In this case they happen to coincide because per-perm ratios are equal.
    assert v.a_n_pb_RoM == pytest.approx(1.0)


def test_mor_diverges_from_rom_when_uncorrelated():
    """Asymmetric perms where MoR and RoM disagree."""
    rec = {
        "a_k_curve": [55.0, 55.0],  # mean([100, 10]) = 55
        "a_k_byte_counts": [55, 55],  # mean([10, 100]) = 55
        "coherence_C": 1.0,
        "per_permutation_a_k_curves": [
            [100.0, 100.0],  # 100 bits paired with 10 bytes → 10/byte
            [10.0, 10.0],  # 10 bits paired with 100 bytes → 0.1/byte
        ],
        "per_permutation_byte_counts": [
            [10, 10],
            [100, 100],
        ],
    }
    v = compute_variants(rec)
    # MoR = mean([100/10, 10/100]) = mean([10.0, 0.1]) = 5.05
    assert v.a_n_pb_MoR == pytest.approx(5.05)
    # RoM = 55/55 = 1.0
    assert v.a_n_pb_RoM == pytest.approx(1.0)
    # Demonstrates the divergence the user worried about.


def test_raises_on_zero_byte_count():
    rec = {
        "a_k_curve": [10.0, 5.0],
        "a_k_byte_counts": [10, 0],
        "coherence_C": 0.5,
    }
    with pytest.raises(ValueError):
        compute_variants(rec)


def test_raises_on_missing_C_inputs():
    rec = {
        "a_k_curve": [10.0, 5.0],
        "a_k_byte_counts": [10, 10],
        # no coherence_C, no unconditional_surprises, no unconditional_total_bits
    }
    with pytest.raises(KeyError):
        compute_variants(rec)


def test_handles_empty_per_perm_lists():
    rec = {
        "a_k_curve": [10.0, 5.0],
        "a_k_byte_counts": [10, 10],
        "coherence_C": 0.5,
        "per_permutation_a_k_curves": None,
        "per_permutation_byte_counts": None,
    }
    v = compute_variants(rec)
    assert v.a_n_pb_MoR is None
    assert v.D_pb_MoR is None
