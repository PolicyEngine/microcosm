"""The UK loss-weight arithmetic exactly as it stood before #1729.

Verbatim copies of ``uk_target_loss_weights_for_rows``
(``uk_runtime.target_weights``) and ``uk_national_target_loss_weights``
(``uk_runtime.national_doctrine``) at fccd9a68b (origin/main when both moved
onto the shared rules of ``microcosm.calibrate.target_loss_rules``), under
their original names, so ``git diff`` against that commit's modules shows them
unchanged. The helpers they call say what a UK row is; those did not move, and
are imported from the live modules. The differential tests hold the shared
rules to these copies, bit for bit. Do not edit them; they are the "before" of
a pure move.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from microcosm.build.uk_runtime.national_doctrine import (
    _ALLOWED_TARGET_WEIGHT_RULES,
    UK_NATIONAL_TARGET_WEIGHT_RULE,
)
from microcosm.build.uk_runtime.target_weights import (
    _AMOUNT,
    _COUNT,
    UKTargetRows,
    _check_row_rule,
    _rule_grains,
    uk_target_value_basis,
)


def uk_target_loss_weights_for_rows(rows: UKTargetRows, *, rule: str) -> np.ndarray:
    """The loss-weight vector of a row-metadata rule (sums to one)."""

    _check_row_rule(rule)
    if not len(rows):
        raise ValueError("target weighting requires at least one target.")
    nation = rule.startswith("nation_")
    sqrt_count = rule.endswith("_sqrt_count")
    grains = np.asarray(_rule_grains(rows, nation=nation), dtype=object)
    families = np.asarray(rows.family, dtype=object)
    bases = (
        np.asarray(uk_target_value_basis(rows), dtype=object) if sqrt_count else None
    )
    magnitudes = np.maximum(np.abs(rows.values), 1.0) ** 0.5 if sqrt_count else None
    grain_order = list(dict.fromkeys(grains.tolist()))
    weights = np.zeros(len(rows), dtype=np.float64)
    for grain in grain_order:
        in_grain = grains == grain
        family_order = list(dict.fromkeys(families[in_grain].tolist()))
        budget = 1.0 / (len(grain_order) * len(family_order))
        for family in family_order:
            cell = np.flatnonzero(in_grain & (families == family))
            if bases is None:
                weights[cell] = budget / cell.size
                continue
            counts = cell[bases[cell] == _COUNT]
            amounts = cell[bases[cell] == _AMOUNT]
            weights[amounts] = budget / cell.size
            if counts.size:
                share = magnitudes[counts]
                weights[counts] = (
                    (budget * counts.size / cell.size) * share / share.sum()
                )
    return weights


def uk_national_target_loss_weights(
    families: Sequence[str],
    *,
    rule: str = UK_NATIONAL_TARGET_WEIGHT_RULE,
) -> np.ndarray | None:
    """The declared target-weight vector for one compiled register.

    ``uniform`` returns ``None`` — the kernel's own equal weighting, and the
    identity this doctrine shipped with. ``family_equal`` gives each declared
    family one equal share of the objective, divided evenly among its
    members, so an over-supplied family cannot outvote the rest of the
    surface by sheer count.

    The vector is derived from the compiled register's declared families and
    nothing else: it is a rule, not a per-target vector, which is the
    property the local doctrine requires of any weighting.
    """

    if rule not in _ALLOWED_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {_ALLOWED_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    if rule == "uniform":
        return None
    labels = [str(family) for family in families]
    if not labels:
        raise ValueError("family_equal weighting requires a non-empty register.")
    if any(not label for label in labels):
        raise ValueError("every target must declare a family for family_equal.")
    counts = Counter(labels)
    family_count = len(counts)
    return np.asarray(
        [1.0 / (family_count * counts[label]) for label in labels],
        dtype=np.float64,
    )
