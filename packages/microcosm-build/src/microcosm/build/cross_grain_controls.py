"""Controls, allocators and checks that sit beside the cross-grain operator.

:mod:`microcosm.build.cross_grain` groups the rows of one quantity, finds the
control of each lower leg, divides the control among the leg's rows, checks
the sum and writes the receipt. A country's sources can need more than that
around the same job (microcosm#1760):

- **allocators**: another way to divide a control among its rows than the
  operator's standing rule. A country names one in its declarations and passes
  it as ``CrossGrainRule.allocator``;
- **derived controls**: a control that no source publishes as one row, computed
  before the operator runs and placed on the surface as the control row;
- **the factor band**: how far one control's ratio to its lower rows may sit
  from the median ratio across the controls of the same quantity;
- **completeness**: whether a leg holds exactly the members it must, with
  declared combined cells;
- **self-consistency**: a source's own total against the sum of its own lower
  rows.

Every function here is plain arithmetic on values and keys. None knows a
country, a source or a geography: the caller's declarations supply them. The
operator does not import this module, so the graphs that hash the operator do
not hash it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

#: ``(values, factor, raw_sum)``, as ``CrossGrainRule.allocator`` returns it.
Allocation = tuple[np.ndarray, float, float]
Allocator = Callable[[np.ndarray, float], Allocation]


# ---------------------------------------------------------------------------
# Allocators
# ---------------------------------------------------------------------------


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
    bit, rows that were first made outside the operator. Stage B of
    microcosm#1760 replaces the two with one allocator.
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


# ---------------------------------------------------------------------------
# Derived controls
# ---------------------------------------------------------------------------


def sum_of_lower_rows(values: Iterable[float]) -> float:
    """A total that is the exactly rounded sum of its own lower rows."""

    return math.fsum(float(value) for value in values)


def sibling_ratio(sibling_higher: float, sibling_lower: float) -> float:
    """A sibling quantity's level in the higher source over the lower one's."""

    return float(sibling_higher) / float(sibling_lower)


def scaled_by_sibling_ratio(
    value: float, sibling_higher: float, sibling_lower: float
) -> tuple[float, float]:
    """A lower-source level lifted onto the higher source's basis.

    Only the lower source publishes this quantity. A sibling quantity that
    both sources publish at the same geography estimates the gap between
    them. Returns ``(value * factor, factor)`` with the sibling's ratio as
    the factor. The caller refuses a missing or zero sibling first.
    """

    factor = sibling_ratio(sibling_higher, sibling_lower)
    return float(value) * factor, factor


# ---------------------------------------------------------------------------
# The factor band
# ---------------------------------------------------------------------------


def level_ratio(higher: float, lower: float) -> float:
    """Two levels of one quantity as a ratio; ``NaN`` where none is defined.

    No ratio is defined against a zero lower level, or when the two levels
    disagree on being positive.
    """

    higher, lower = float(higher), float(lower)
    if lower != 0.0 and (higher > 0) == (lower > 0):
        return higher / lower
    return math.nan


def control_ratio(
    control_value: float, lower_values: Iterable[float]
) -> tuple[float | None, float]:
    """``(control / lower sum, lower sum)``, with the sum exactly rounded.

    The ratio is the factor :func:`proportional_fsum` gives the same rows. It
    is ``None`` when the rows sum to zero.
    """

    lower_sum = sum_of_lower_rows(lower_values)
    if lower_sum == 0.0:
        return None, lower_sum
    return float(control_value) / lower_sum, lower_sum


def median_by_group[Key: Hashable, Group: Hashable](
    ratios: Mapping[Key, float],
    group_of: Callable[[Key], Group],
    *,
    positive_only: bool = False,
) -> dict[Group, float]:
    """The median finite ratio of each group of keys.

    ``positive_only`` also leaves out ratios that are zero or negative.
    """

    grouped: dict[Group, list[float]] = {}
    for key, ratio in ratios.items():
        if math.isfinite(ratio) and (ratio > 0 or not positive_only):
            grouped.setdefault(group_of(key), []).append(ratio)
    return {group: float(np.median(values)) for group, values in grouped.items()}


@dataclass(frozen=True)
class BandVerdict:
    """One ratio judged against its group's median."""

    ok: bool
    ratio: float
    median: float
    relative: float


def band_verdict(ratio: float, median: float, band: float) -> BandVerdict:
    """Whether ``ratio`` sits within ``band`` of ``median``, either way.

    ``relative`` is the ratio over the median. It is ``NaN``, and the verdict
    fails, when the ratio is not finite or the median is zero or unknown.
    """

    relative = ratio / median if math.isfinite(ratio) and median else math.nan
    ok = math.isfinite(relative) and 1 / band <= relative <= band
    return BandVerdict(ok=ok, ratio=ratio, median=median, relative=relative)


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegCompleteness:
    """A leg's present members against the members it must hold."""

    missing: tuple
    unexpected: tuple
    duplicated: tuple
    #: Absent member -> the present host whose row covers it.
    merged: Mapping

    @property
    def complete(self) -> bool:
        return not (self.missing or self.unexpected or self.duplicated)


def leg_completeness[Member: Hashable](
    present: Iterable[Member],
    expected: Iterable[Member],
    *,
    combined_cells: Mapping[Member, Member] | None = None,
) -> LegCompleteness:
    """Compare the members a leg holds with the members it must hold.

    ``combined_cells`` maps a member that may be absent to its declared host.
    An absent member is covered, not missing, when its host is present and
    expected: the host's row then stands for both.
    """

    held = list(present)
    held_set, expected_set = set(held), set(expected)
    combined = dict(combined_cells or {})
    absent = expected_set - held_set
    merged = {
        member: combined[member]
        for member in sorted(absent)
        if member in combined
        and combined[member] in held_set
        and combined[member] in expected_set
    }
    seen: set = set()
    duplicated = sorted(
        {member for member in held if member in seen or seen.add(member)}
    )
    return LegCompleteness(
        missing=tuple(sorted(absent - merged.keys())),
        unexpected=tuple(sorted(held_set - expected_set)),
        duplicated=tuple(duplicated),
        merged=merged,
    )


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SumGap:
    """The distance between a total and the sum of its lower rows."""

    gap: float
    scale: float

    @property
    def relative(self) -> float:
        return self.gap / self.scale

    def within(self, rtol: float) -> bool:
        """The gap is at most ``rtol`` of the scale."""

        return self.gap <= rtol * self.scale

    def exceeds(self, rtol: float) -> bool:
        """The gap is more than ``rtol`` of the scale."""

        return self.gap > rtol * self.scale


def sum_gap(total: float, lower_sum: float, *, floor: float = 1.0) -> SumGap:
    """A total against the sum of its lower rows.

    The scale is the total's size, and at least ``floor``, so a total near
    zero is judged on an absolute gap.
    """

    total = float(total)
    return SumGap(gap=abs(total - lower_sum), scale=max(abs(total), floor))
