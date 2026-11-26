"""The US rows-level target-loss formula exactly as it stood before #1729.

A verbatim copy of the four functions ``microcosm.build.us_runtime.
target_loss_weights`` defined at ac7ab99ae (origin/main when the formula
became the shared rule ``sqrt_value_concept_budget`` of
``microcosm.calibrate.target_loss_rules``), under their original names, so
``git diff`` against that commit's module shows them unchanged. The
differential tests hold the shared rule to this copy, bit for bit. Do not
edit it; it is the "before" of a pure move.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np

US_FISCAL_TARGET_VALUE_WEIGHT_POWER = 0.5


def target_loss_weights_from_rows(
    values: Sequence[float],
    bases: Sequence[str],
    concept_budget_keys: Sequence[Hashable],
    families: Sequence[str | None],
    family_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Steps 2-6 of the formula, over row-aligned per-target inputs.

    Returns float64 weights, row-aligned with the inputs. With at least one
    row they are positive and of mean 1, given positive finite multipliers
    whose products stay within float64 range (the CLI parser admits only
    positive finite ones; this function, like the national helper it came
    from, does not check them).
    """

    n_rows = len(values)
    if not len(bases) == len(concept_budget_keys) == len(families) == n_rows:
        raise ValueError(
            "Target-loss inputs are not row-aligned: "
            f"{n_rows} values, {len(bases)} bases, "
            f"{len(concept_budget_keys)} concept keys, {len(families)} families."
        )
    basis_array = np.asarray(list(bases), dtype=object)
    weights = _concept_budget_weights(
        _value_basis_weights(values, basis_array), concept_budget_keys
    )
    unique_bases = sorted(set(basis_array.tolist()))
    if not unique_bases:
        return weights
    basis_total = len(weights) / len(unique_bases)
    for basis in unique_bases:
        mask = basis_array == basis
        current_total = weights[mask].sum()
        if current_total > 0:
            weights[mask] *= basis_total / current_total
    weights = weights / weights.mean()
    if not family_multipliers:
        return weights
    family_array = np.asarray(list(families), dtype=object)
    for family, multiplier in sorted(family_multipliers.items()):
        mask = family_array == family
        if not mask.any():
            raise ValueError(
                f"--target-family-loss-multiplier family {family!r} matches "
                "no compiled target."
            )
        weights[mask] *= multiplier
    return weights / weights.mean()


def _value_basis_weights(values: Sequence[float], bases: np.ndarray) -> np.ndarray:
    """Step 2: ``max(|value|, 1) ** 0.5`` over its basis mean."""

    weights = np.ones(len(bases), dtype=np.float64)
    magnitudes = np.asarray(
        [max(abs(float(value)), 1.0) for value in values],
        dtype=np.float64,
    )
    raw_weights = magnitudes**US_FISCAL_TARGET_VALUE_WEIGHT_POWER
    for basis in sorted(set(bases.tolist())):
        mask = bases == basis
        mean_value = raw_weights[mask].mean()
        if mean_value > 0:
            weights[mask] = raw_weights[mask] / mean_value
    return weights


def _concept_budget_weights(
    weights: np.ndarray, concept_budget_keys: Sequence[Hashable]
) -> np.ndarray:
    """Step 3: each group sums to its largest member's weight."""

    for indices in concept_budget_groups(concept_budget_keys):
        group_weights = weights[indices]
        group_total = float(group_weights.sum())
        group_budget = float(group_weights.max(initial=0.0))
        if group_total > 0 and group_budget > 0:
            weights[indices] *= group_budget / group_total
    return weights


def concept_budget_groups(
    concept_budget_keys: Sequence[Hashable],
) -> list[list[int]]:
    """Row indices sharing each concept-budget key, in first-row order."""

    group_indices: dict[Hashable, list[int]] = {}
    for index, key in enumerate(concept_budget_keys):
        group_indices.setdefault(key, []).append(index)
    return list(group_indices.values())
