"""Keyed thinning of reported take-up anchors to a residual count.

A take-up stage that preserves reported recipients as anchors can meet a
group, such as a state, where the anchors alone weigh more than the count
left for that group. Thinning holds the anchors to that count in
expectation: every anchor of such a group stays an anchor with one
probability, (residual - fixed mass) / anchored mass, clipped to [0, 1], on
its own keyed uniform. A record's result depends only on the seed, the
stream and its own key, never on the row order. Fixed rows, such as a stored
take-up ``True``, count toward the residual and are never thinned.

The caller chooses the groups, the mass each row counts toward the residual,
the rows whose mass counts, the fixed rows, the anchors and the stream, so
the rule holds no country's choices. The ACS local lane thins its Medicaid
``HINS4`` anchors (microcosm#1060) and its ACA subsidized-coverage anchors
(microcosm#1720) with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms

__all__ = ["AnchorThinning", "anchor_keep_uniforms", "thin_anchors_to_residuals"]


@dataclass(frozen=True)
class AnchorThinning:
    """The kept anchors and, per group, the terms of the thinning.

    ``kept`` is aligned with the rows and is ``True`` only on kept anchors.
    ``keep_probability`` has one entry per residual group. ``fixed_mass`` and
    ``anchor_mass`` hold the counted mass of the fixed rows and of the
    anchors in every group the rows carry.
    """

    kept: np.ndarray
    keep_probability: dict[str, float]
    fixed_mass: dict[str, float]
    anchor_mass: dict[str, float]

    @property
    def thinned_groups(self) -> list[str]:
        """The residual groups whose anchors are kept with probability < 1."""

        return sorted(
            group for group, value in self.keep_probability.items() if value < 1.0
        )


def _group_sums(
    groups: np.ndarray, weights: np.ndarray, selected: np.ndarray
) -> dict[str, float]:
    """The weight of the selected rows in each group (every group listed)."""

    table = pd.DataFrame({"group": groups, "weight": np.where(selected, weights, 0.0)})
    sums = table.groupby("group", sort=True)["weight"].sum()
    return {str(group): float(value) for group, value in sums.items()}


def anchor_keep_uniforms(
    keys: Sequence[object] | np.ndarray, *, seed: int, stream: str
) -> np.ndarray:
    """One seeded blake2b uniform per record key on the keep draw's stream."""

    return stable_identity_uniforms(
        [str(key) for key in keys], seed=int(seed), salt=stream
    )


def thin_anchors_to_residuals(
    *,
    groups: Sequence[object] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    counted: Sequence[bool] | np.ndarray,
    fixed: Sequence[bool] | np.ndarray,
    anchors: Sequence[bool] | np.ndarray,
    residuals: Mapping[str, float],
    keys: Sequence[object] | np.ndarray,
    seed: int,
    stream: str,
) -> AnchorThinning:
    """Keep each group's anchors with the probability that meets its residual.

    Args:
        groups: Each row's group, compared as text with the keys of
            ``residuals``.
        weights: Each row's mass toward its group's residual.
        counted: The rows whose mass counts: those the residual's own
            measure can select, such as the eligible ones.
        fixed: Rows that take up already and are never thinned.
        anchors: Rows that may be thinned.
        residuals: The count left to fill in each group.
        keys: Each row's record key for the keep draw.
        seed: The build seed.
        stream: The keep draw's own stream, apart from any take-up draw.

    Returns:
        The kept anchors and the terms of the thinning. In a group whose
        counted fixed and anchored mass fit its residual, or that has no
        residual, every anchor is kept. In any other group each anchor,
        counted or not, is kept where its keyed uniform is below
        (residual - fixed mass) / anchored mass, clipped to [0, 1].

    Raises:
        ValueError: If the row arrays are not aligned and one-dimensional.
    """

    labels = np.asarray(groups).astype(str)
    weights = np.asarray(weights, dtype=np.float64)
    counted = np.asarray(counted, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool)
    anchors = np.asarray(anchors, dtype=bool)
    keys = np.asarray(keys, dtype=object)
    if labels.ndim != 1 or any(
        array.shape != labels.shape
        for array in (weights, counted, fixed, anchors, keys)
    ):
        raise ValueError(
            "Anchor thinning needs aligned one-dimensional groups, weights, "
            "counted, fixed, anchor and key arrays."
        )
    fixed_mass = _group_sums(labels, weights, counted & fixed)
    anchor_mass = _group_sums(labels, weights, counted & anchors)
    keep: dict[str, float] = {}
    for group, target in residuals.items():
        stored = fixed_mass.get(str(group), 0.0)
        anchored = anchor_mass.get(str(group), 0.0)
        keep[str(group)] = (
            1.0
            if anchored <= 0.0 or stored + anchored <= target
            else float(np.clip((target - stored) / anchored, 0.0, 1.0))
        )
    probability = pd.Series(labels).map(keep).fillna(1.0).to_numpy(dtype=np.float64)
    draws = np.zeros(len(labels), dtype=np.float64)
    thinning = anchors & (probability < 1.0)
    if thinning.any():
        draws[thinning] = anchor_keep_uniforms(keys[thinning], seed=seed, stream=stream)
    return AnchorThinning(
        kept=anchors & (draws < probability),
        keep_probability=keep,
        fixed_mass=fixed_mass,
        anchor_mass=anchor_mass,
    )
