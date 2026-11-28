"""The row-level pass and the allocators beside the cross-grain operator.

:mod:`microcosm.build.cross_grain` reconciles a surface that is a data frame,
and its receipt names legs, not rows. A caller that holds target rows wants
the answer for each row: its new value, what reconciled it, with which
factor, and under which control row. :func:`reconcile_rows` runs the operator
and gives that answer. It holds no country: the caller supplies the rows,
the measurements and the rule.

The module also holds:

- :func:`sum_of_lower_rows`, the arithmetic of a derived control that is the
  sum of its lower rows;
- three allocators for ``CrossGrainRule.allocator``. They keep, bit for bit,
  rows that were first made outside the operator. They go when the feed that
  holds those rows is pinned again, and the operator's own rule then divides
  every control (microcosm#1760).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.cross_grain import (
    EXACT_SIGNATURE_CONTROL,
    CrossGrainRule,
    _contract_target_id,
    _control_grains,
    _inconsistency_identity,
    _measurement_signature,
    _row_legs,
    apply_cross_grain_reconciliation,
)

# ---------------------------------------------------------------------------
# The row-level pass
# ---------------------------------------------------------------------------

RECONCILED = "reconciled"
UNBOUND = "unbound"
UNTOUCHED = "untouched"


@dataclass(frozen=True)
class SurfaceRow:
    """One row of a reconciliation surface."""

    grain: str
    geography_id: str
    target_id: str
    value: float


@dataclass(frozen=True)
class RowOutcome:
    """What a pass did with one surface row.

    ``status`` is :data:`RECONCILED` (the row was divided from a control),
    :data:`UNBOUND` (its leg was out of the bound of a bridge that unbinds, so
    the row keeps its value and the caller takes it off its surface) or
    :data:`UNTOUCHED` (a control row, or a row that no group covers).
    ``control`` is the bridge id, or ``exact_signature``. ``factor`` and
    ``lower_total`` are those of the row's leg. ``parent`` is the position of
    the control row, when the control is one row.
    """

    value: float
    status: str = UNTOUCHED
    control: str | None = None
    factor: float | None = None
    lower_total: float | None = None
    parent: int | None = None


def reconcile_rows(
    rows: Sequence[SurfaceRow],
    bound_higher_targets: Iterable[str],
    signatures: Mapping[str, Mapping[str, Any]],
    rule: CrossGrainRule,
    *,
    expected_members: Mapping[str, Mapping[str, frozenset[str]]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
) -> tuple[list[RowOutcome], dict[str, Any]]:
    """Run the operator on ``rows`` and say what it did with each row.

    Returns one :class:`RowOutcome` for each row, in the order of ``rows``,
    and the operator's pass receipt.

    Two shapes that the operator accepts are refused here, because the
    answer for a row would be wrong:

    - a leg whose id holds ``+``: the operator's receipt joins the legs of
      one control with ``+``, so such a leg cannot be found again;
    - a row that two groups reconcile (a lower target under a bridge and
      under a control of its own measurement): a row names one control and
      one factor.
    """

    for row in rows:
        joined = [leg for leg in _row_legs(row.geography_id, rule) if "+" in leg]
        if joined:
            raise ValueError(
                f"cross-grain row at {row.geography_id!r} sits on leg "
                f"{joined[0]!r}; a leg id that holds '+' cannot be named for "
                "each row."
            )
    frame = pd.DataFrame(
        [
            (row.grain, row.geography_id, row.target_id, float(row.value))
            for row in rows
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        frame,
        bound_higher_targets,
        signatures,
        rule,
        expected_members=expected_members,
        licensed_empty_legs=licensed_empty_legs,
    )
    values = reconciled["value"].to_numpy(dtype=np.float64).tolist()
    control_grains = set(_control_grains(rule))
    target_ids = [_contract_target_id(row.target_id) for row in rows]
    controls: dict[tuple[str, str], int] = {}
    lower: dict[tuple[str, str, str], list[int]] = {}
    for position, (row, target_id) in enumerate(zip(rows, target_ids, strict=True)):
        if row.grain in control_grains:
            controls.setdefault((target_id, row.geography_id), position)
        for leg in _row_legs(row.geography_id, rule):
            lower.setdefault((target_id, row.grain, leg), []).append(position)
    identities: dict[str, list[str]] = {}
    for target_id in dict.fromkeys(target_ids):
        if target_id in signatures:
            identity = _inconsistency_identity(
                "signature",
                "",
                _measurement_signature(signatures[target_id], rule.signature_fields),
            )
            identities.setdefault(identity, []).append(target_id)
    lower_side = {
        bridge.bridge_id: _contract_target_id(bridge.lower_side)
        for bridge in rule.bridges
    }

    def parent_of(leg: Mapping[str, Any]) -> int | None:
        higher = leg["higher_target_ids"]
        if len(higher) != 1:
            return None
        return controls.get((str(higher[0]), str(leg["parent_geography_id"])))

    outcomes = [RowOutcome(value=value) for value in values]
    for group in receipt["groups"]:
        bridge_id = group["bridge_id"]
        if bridge_id is None:
            control = EXACT_SIGNATURE_CONTROL
            lower_targets = identities.get(
                group["inconsistency_id"].rsplit(":", 1)[0], []
            )
        else:
            control = str(bridge_id)
            lower_targets = [lower_side[bridge_id]]
        for leg in group["legs"]:
            parent = parent_of(leg)
            for target_id in lower_targets:
                for name in str(leg["leg"]).split("+"):
                    for position in lower.get(
                        (target_id, group["lower_grain"], name), ()
                    ):
                        if outcomes[position].status != UNTOUCHED:
                            raise ValueError(
                                f"cross-grain row {position} of target "
                                f"{target_id!r} is reconciled by "
                                f"{outcomes[position].control!r} and by "
                                f"{control!r}; a row names one control."
                            )
                        outcomes[position] = RowOutcome(
                            value=values[position],
                            status=RECONCILED,
                            control=control,
                            factor=float(leg["declared_factor"]),
                            lower_total=float(leg["old_total"]),
                            parent=parent,
                        )
    for leg in receipt.get("unbound_legs", ()):
        parent = parent_of(leg)
        for position in leg["positions"]:
            outcomes[position] = RowOutcome(
                value=values[position],
                status=UNBOUND,
                control=str(leg["bridge_id"]),
                factor=float(leg["factor_on_input_totals"]),
                parent=parent,
            )
    return outcomes, receipt


# ---------------------------------------------------------------------------
# A derived control
# ---------------------------------------------------------------------------


def sum_of_lower_rows(values: Iterable[float]) -> float:
    """A total that is the exactly rounded sum of its own lower rows."""

    return math.fsum(float(value) for value in values)


# ---------------------------------------------------------------------------
# Allocators (until the feed that holds their rows is pinned again)
# ---------------------------------------------------------------------------

Allocation = tuple[np.ndarray, float, float]
Allocator = Callable[[np.ndarray, float], Allocation]


def proportional_fsum(raw_values: np.ndarray, parent_value: float) -> Allocation:
    """Each row times the control over the rows' exactly rounded sum.

    The operator's standing rule with ``math.fsum`` in place of the numpy
    sum, so the factor does not depend on the order of the rows. A zero total
    keeps the rows (factor one), as the standing rule does.
    """

    raw_sum = math.fsum(raw_values.tolist())
    factor = 1.0 if raw_sum == 0.0 else parent_value / raw_sum
    return raw_values * factor, factor, raw_sum


def residual_shares(raw_values: Iterable[float]) -> tuple[list[float], float, int]:
    """Each value's share of the exactly rounded total, summing to one.

    Returns ``(shares, total, largest)``. The shares' rounding residual goes
    to the largest share (the first of equal largest shares), never to a small
    row. The total must not be zero.
    """

    values = [float(value) for value in raw_values]
    total = math.fsum(values)
    shares = [value / total for value in values]
    largest = max(range(len(values)), key=lambda index: shares[index])
    shares[largest] += 1.0 - math.fsum(shares)
    return shares, total, largest


def share_residual_to_largest(
    raw_values: np.ndarray, parent_value: float
) -> Allocation:
    """The control times each row's share, with two residual steps.

    The shares come from :func:`residual_shares`; the products' rounding
    residual goes to the same largest row, so the rows sum to the control
    exactly under ``math.fsum``. The recorded factor is the control over the
    rows' total; no single factor reproduces the rows.

    This allocator and :func:`scale_residual_to_largest` reproduce, bit for
    bit, rows that were first made outside the operator.
    """

    if math.fsum(raw_values.tolist()) == 0.0:
        return raw_values.copy(), 1.0, 0.0
    shares, total, largest = residual_shares(raw_values.tolist())
    values = [parent_value * share for share in shares]
    values[largest] += parent_value - math.fsum(values)
    return np.asarray(values, dtype=np.float64), parent_value / total, total


def scale_residual_to_largest(
    raw_values: np.ndarray, parent_value: float
) -> Allocation:
    """The control times each row over the rows' total, with one residual step.

    Multiply first, then divide; the rounding residual goes to the largest
    scaled row (the first of equal largest rows). The recorded factor is the
    control over the rows' total.

    See :func:`share_residual_to_largest` for why this allocator exists.
    """

    raw = raw_values.tolist()
    total = math.fsum(raw)
    if total == 0.0:
        return raw_values.copy(), 1.0, 0.0
    values = [parent_value * value / total for value in raw]
    largest = max(range(len(values)), key=lambda index: values[index])
    values[largest] += parent_value - math.fsum(values)
    return np.asarray(values, dtype=np.float64), parent_value / total, total


#: The allocators a declaration can name.
ALLOCATORS: Mapping[str, Allocator] = {
    allocator.__name__: allocator
    for allocator in (
        proportional_fsum,
        share_residual_to_largest,
        scale_residual_to_largest,
    )
}


def allocator_named(name: str) -> Allocator:
    """The allocator a declaration names; an unknown name is refused."""

    try:
        return ALLOCATORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown cross-grain allocator {name!r}; expected one of "
            f"{sorted(ALLOCATORS)}."
        ) from None
