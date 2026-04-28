"""Canonical mean-of-ratios per-byte aggregation.

This module exists as the single source of truth for one specific question:
**how do we compute a per-byte progressive surprise curve when we have
per-permutation data?**

Two estimators are possible (see the global ``implement-math`` skill):

* **mean-of-ratios (MoR)** — the formula the paper specifies::

      a_k_pb_MoR := (1 / |Σ|) · Σ_σ ( bits_σ[k] / ‖r_{σ(k)}‖ )

  i.e. each permutation contributes its own per-byte rate, then we
  average.

* **ratio-of-means (RoM)** — what an earlier implementation revision
  did under the same name::

      a_k_pb_RoM := mean_σ(bits_σ[k]) / mean_σ(‖r_{σ(k)}‖)

  i.e. average bits and bytes separately, then divide.

These are *not* equivalent when ``bits_σ[k]`` and ``‖r_{σ(k)}‖`` are
correlated across permutations (the typical case: long responses tend
to have more total bits).  After perm-averaging the byte counts at
each position k approach the *mean response length* (roughly constant
in k), so RoM degenerates into "total-bits curve rescaled by
1 / mean_response_length" — *not* what the paper specifies.

The paper specifies MoR.  This module implements MoR.

For the historical bug postmortem and a list of related traps, see
``~/.claude/skills/implement-math/SKILL.md``.
"""

from __future__ import annotations

from typing import Sequence


def compute_a_k_curve_mor(
    per_permutation_a_k_curves: Sequence[Sequence[float]],
    per_permutation_byte_counts: Sequence[Sequence[int]],
) -> list[float]:
    """Mean-of-ratios per-byte progressive surprise curve.

    Formula:
        a_k_pb_MoR[k] = (1 / |Σ|) · Σ_σ ( bits_σ[k] / ‖r_{σ(k)}‖ )

    Code correspondence:
        - Σ_σ ... (sum over permutations):  the inner ``sum(...)`` per k
        - 1 / |Σ| (outer mean):              ``/ n_perms``
        - per-permutation per-byte ratio:    ``bits / bc`` inside the sum

    Args:
        per_permutation_a_k_curves: Outer index is permutation σ, inner index
            is position k.  Values are total bits.  Shape (P, n).
        per_permutation_byte_counts: Same shape.  Values are byte counts of
            the response that sat at slot k in permutation σ.

    Returns:
        List of length n: the per-byte progressive surprise curve, averaged
        across permutations as a mean-of-ratios.  Slots with zero byte count
        are skipped within their permutation (they cannot contribute a
        meaningful rate).

    Raises:
        ValueError: if the inputs are empty, or if every permutation at
            some position k has zero byte count.  In practice the latter
            only fires on a degenerate input where the response sitting
            at slot k under every permutation is the empty string, e.g.
            every response in the input list is ``""``.  We hit this
            exactly once in the wild (Tevet decTest sample
            ``test.enc.prompt.txt::output.cmdc.uni.sample10.temp1.01::00955``,
            an upstream-corrupted row whose 10 ``resp_*`` cells are all
            empty); ``analyze_c_ainf.py`` catches the ValueError and
            drops the sample.
    """
    if not per_permutation_a_k_curves or not per_permutation_byte_counts:
        raise ValueError(
            "per_permutation_a_k_curves and ..._byte_counts must be non-empty"
        )
    if len(per_permutation_a_k_curves) != len(per_permutation_byte_counts):
        raise ValueError(
            f"length mismatch: {len(per_permutation_a_k_curves)} curves vs "
            f"{len(per_permutation_byte_counts)} byte-count rows"
        )

    n_positions = len(per_permutation_a_k_curves[0])
    curve: list[float] = []
    for k in range(n_positions):
        ratios: list[float] = []
        for cur, bcs in zip(per_permutation_a_k_curves, per_permutation_byte_counts):
            bc = bcs[k]
            if bc > 0:
                ratios.append(cur[k] / bc)
        if not ratios:
            raise ValueError(
                f"position k={k} has zero byte count in every permutation "
                "(every response landing at this slot is the empty string)"
            )
        curve.append(sum(ratios) / len(ratios))
    return curve


def compute_a_n_per_byte_mor(
    per_permutation_a_k_curves: Sequence[Sequence[float]],
    per_permutation_byte_counts: Sequence[Sequence[int]],
) -> float:
    """Last-position mean-of-ratios per-byte surprise.

    Convenience wrapper for ``compute_a_k_curve_mor(...)[-1]``.  Most
    consumers (RLHF, scenarios, Tevet) only need this scalar.

    Formula:
        a_n_pb_MoR = (1 / |Σ|) · Σ_σ ( bits_σ[n-1] / ‖r_{σ(n-1)}‖ )
    """
    return compute_a_k_curve_mor(
        per_permutation_a_k_curves, per_permutation_byte_counts
    )[-1]
