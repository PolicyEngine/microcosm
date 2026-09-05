"""Pure benchmark metrics: point relative error and the categorical partition.

Arithmetic contract (``/categorical_partition/numeric_representation``): every
threshold is an exact :class:`~decimal.Decimal` parsed from the document's
decimal string, and every measurement is converted to ``Decimal`` exactly —
``Decimal(float)`` is the float's exact binary value, so no tolerance is
inflated on the way in. All ratios are evaluated in :data:`ARITHMETIC`, a
120-digit context, and compared with the document's inclusive ``<=``.

These functions measure. They do not establish that a denominator is *known*,
that a mapping is approved, or that lineage is authenticated: those are separate
axes in :mod:`microcosm.build.cd_benchmark.universe` and
:mod:`microcosm.build.cd_benchmark.evaluation`, and a caller cannot obtain them
by passing a number here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Context, Decimal, localcontext
from types import MappingProxyType

from microcosm.build.cd_benchmark.protocol import (
    DistributionBlock,
    PartitionRules,
    PointBlock,
    ProtocolError,
)
from microcosm.build.cd_benchmark.reasons import Reason, dedupe

__all__ = [
    "ARITHMETIC",
    "BinDiscrepancy",
    "CategoricalResult",
    "MetricError",
    "PointResult",
    "categorical_check",
    "measure",
    "point_check",
]

#: The fixed arithmetic context for every ratio in this package. Wide enough
#: that sums of exactly converted float masses do not round, and every
#: hand-calculable boundary value compares exactly.
ARITHMETIC = Context(prec=120)

_ZERO = Decimal(0)
_HUNDRED = Decimal(100)
_HALF = Decimal("0.5")


class MetricError(ValueError):
    """A measurement is not a finite real number the reducer contract allows."""


def measure(value: object, label: str) -> Decimal:
    """Convert a measurement to an exact ``Decimal``.

    Booleans are refused outright: ``True`` is not a mass of one. Integers,
    floats and ``Decimal`` values convert exactly; nonfinite values are refused.

    Raises:
        MetricError: On a boolean, a nonfinite value or an unsupported type.
    """
    if isinstance(value, bool):
        raise MetricError(f"{label} must be a number, not a boolean.")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise MetricError(f"{label} must be finite.")
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetricError(f"{label} must be finite.")
        return Decimal(value)
    raise MetricError(f"{label} must be an int, float or Decimal, got {type(value)!r}.")


def _nonnegative(value: object, label: str, problems: list[Reason]) -> Decimal | None:
    try:
        parsed = measure(value, label)
    except MetricError:
        problems.append(Reason.INVALID_REDUCER_INPUT)
        return None
    if parsed < 0:
        problems.append(Reason.INVALID_REDUCER_INPUT)
        return None
    return parsed


@dataclass(frozen=True)
class PointResult:
    """One mandatory point block's measured discrepancy against its bound."""

    block_id: str
    geography_id: str
    bound: Decimal
    candidate: Decimal | None
    reference: Decimal | None
    absolute_discrepancy: Decimal | None
    relative_discrepancy: Decimal | None
    within_bound: bool | None
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)


def point_check(
    block: PointBlock,
    *,
    candidate: object,
    reference: object,
    geography_id: str = "",
) -> PointResult:
    """Compare a candidate scalar with its published reference.

    ``abs(candidate - reference) / reference <= bound`` when ``reference > 0``.
    A zero reference requires ``candidate == 0`` for point conformity and leaves
    the relative discrepancy unavailable rather than substituting a ratio.
    Negative or nonfinite masses are ``invalid_reducer_input``: no absolute
    value, clipping or drop is applied.
    """
    problems: list[Reason] = []
    measured = _nonnegative(candidate, "candidate", problems)
    published = _nonnegative(reference, "reference", problems)
    evidence: dict[str, object] = {
        "count_once": block.count_once,
        "duplicate_reference_aliases": block.duplicate_reference_aliases,
        "reference_variables": block.variables,
    }
    if measured is None or published is None:
        return PointResult(
            block_id=block.id,
            geography_id=geography_id,
            bound=block.point_relative_bound,
            candidate=measured,
            reference=published,
            absolute_discrepancy=None,
            relative_discrepancy=None,
            within_bound=None,
            reasons=dedupe(problems),
            evidence=MappingProxyType(evidence),
        )
    with localcontext(ARITHMETIC):
        absolute = abs(measured - published)
        if published == 0:
            evidence["zero_reference"] = True
            within = measured == 0
            relative = None
        else:
            relative = absolute / published
            within = relative <= block.point_relative_bound
    if not within:
        problems.append(Reason.PUBLISHED_BENCHMARK_NONCONFORMITY)
    return PointResult(
        block_id=block.id,
        geography_id=geography_id,
        bound=block.point_relative_bound,
        candidate=measured,
        reference=published,
        absolute_discrepancy=absolute,
        relative_discrepancy=relative,
        within_bound=within,
        reasons=dedupe(problems),
        evidence=MappingProxyType(evidence),
    )


@dataclass(frozen=True)
class BinDiscrepancy:
    """One bin's candidate and reference shares, and their gap in points."""

    bin_id: str
    unclassified: bool
    candidate_mass: Decimal
    candidate_share: Decimal
    reference_count: Decimal | None
    reference_share: Decimal
    discrepancy_pp: Decimal
    within_bound: bool


@dataclass(frozen=True)
class CategoricalResult:
    """A complete categorical comparison, residual bin included."""

    block_id: str
    geography_id: str
    candidate_denominator: Decimal | None
    reference_total: Decimal | None
    unclassified_mass: Decimal | None
    unclassified_fraction: Decimal | None
    bins: tuple[BinDiscrepancy, ...]
    max_bin_discrepancy_pp: Decimal | None
    total_variation: Decimal | None
    within_bound: bool | None
    independent_votes: bool | None
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)


def _empty(
    block: DistributionBlock,
    geography_id: str,
    problems: list[Reason],
    evidence: dict[str, object],
    *,
    denominator: Decimal | None = None,
    reference_total: Decimal | None = None,
    unclassified: Decimal | None = None,
) -> CategoricalResult:
    return CategoricalResult(
        block_id=block.id,
        geography_id=geography_id,
        candidate_denominator=denominator,
        reference_total=reference_total,
        unclassified_mass=unclassified,
        unclassified_fraction=None,
        bins=(),
        max_bin_discrepancy_pp=None,
        total_variation=None,
        within_bound=None,
        independent_votes=block.independent_votes,
        reasons=dedupe(problems),
        evidence=MappingProxyType(evidence),
    )


def categorical_check(
    block: DistributionBlock,
    *,
    partition: PartitionRules,
    candidate_bin_masses: Mapping[str, object],
    unclassified_mass: object,
    candidate_denominator: object,
    reference_counts: Mapping[str, object],
    reference_total: object,
    geography_id: str = "",
) -> CategoricalResult:
    """Compare a complete candidate partition with its published distribution.

    ``p_i = m_i / D`` and ``p_unclassified = U / D`` against
    ``q_i = published_i / published_total`` with ``q_unclassified = 0``; per-bin
    discrepancy is ``100 * abs(p_i - q_i)`` including the residual bin, and
    total variation is ``0.5 * sum(abs(p_i - q_i))`` over every bin including
    the residual. Known bins are never renormalized to absorb residual mass.

    ``sum(m_i) + U`` must equal ``D`` exactly; a partition that does not close
    is ``invalid_reducer_input`` rather than a silently rescaled result. A zero
    candidate or reference denominator yields the matching ``zero_*_denominator``
    reason with no synthesized shares, and a residual fraction above the
    declared maximum adds ``unclassified_fraction_exceeded`` while the measured
    discrepancies are retained.
    """
    if block.id not in partition.applies_to:
        raise ProtocolError(
            f"The categorical partition does not apply to block {block.id!r}."
        )
    if partition.renormalize_known_bins:
        raise ProtocolError("This partition renormalizes known bins; refusing.")
    if partition.comparison != "<=":
        raise ProtocolError(
            f"Unsupported residual-fraction comparison {partition.comparison!r}."
        )
    problems: list[Reason] = []
    evidence: dict[str, object] = {
        "renormalize_known_bins": partition.renormalize_known_bins,
        "maximum_unclassified_fraction": str(partition.maximum_unclassified_fraction),
        "residual_bin_id": partition.unknown_bin_id,
    }
    known = block.known_bins
    expected = {item.id for item in known}
    if set(candidate_bin_masses) != expected:
        problems.append(Reason.INVALID_REDUCER_INPUT)
        evidence["candidate_bins_missing"] = tuple(
            sorted(expected - set(candidate_bin_masses))
        )
        evidence["candidate_bins_unknown"] = tuple(
            sorted(set(candidate_bin_masses) - expected)
        )
    if set(reference_counts) != expected:
        problems.append(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING)
        evidence["reference_cells_missing"] = tuple(
            sorted(expected - set(reference_counts))
        )
        evidence["reference_cells_unknown"] = tuple(
            sorted(set(reference_counts) - expected)
        )
    denominator = _nonnegative(candidate_denominator, "candidate denominator", problems)
    published_total = _nonnegative(reference_total, "reference total", problems)
    residual = _nonnegative(unclassified_mass, "unclassified mass", problems)
    masses = {
        item.id: _nonnegative(candidate_bin_masses.get(item.id), item.id, problems)
        for item in known
        if item.id in candidate_bin_masses
    }
    counts = {
        item.id: _nonnegative(reference_counts.get(item.id), item.id, problems)
        for item in known
        if item.id in reference_counts
    }
    if problems or None in masses.values() or None in counts.values():
        return _empty(
            block,
            geography_id,
            problems,
            evidence,
            denominator=denominator,
            reference_total=published_total,
            unclassified=residual,
        )

    with localcontext(ARITHMETIC):
        closure = sum(masses.values(), _ZERO) + residual
        if closure != denominator:
            problems.append(Reason.INVALID_REDUCER_INPUT)
            evidence["closure_residual"] = str(closure - denominator)
            return _empty(
                block,
                geography_id,
                problems,
                evidence,
                denominator=denominator,
                reference_total=published_total,
                unclassified=residual,
            )
        if published_total == 0:
            problems.append(Reason.ZERO_REFERENCE_DENOMINATOR)
        if denominator == 0:
            problems.append(Reason.ZERO_CANDIDATE_DENOMINATOR)
            evidence["totals_remain_separately_evaluable"] = True
        if problems:
            return _empty(
                block,
                geography_id,
                problems,
                evidence,
                denominator=denominator,
                reference_total=published_total,
                unclassified=residual,
            )

        fraction = residual / denominator
        rows: list[BinDiscrepancy] = []
        total_variation = _ZERO
        maximum = _ZERO
        reference_share_sum = _ZERO
        for item in block.bins:
            mass = residual if item.unclassified else masses[item.id]
            count = None if item.unclassified else counts[item.id]
            candidate_share = mass / denominator
            reference_share = _ZERO if count is None else count / published_total
            reference_share_sum += reference_share
            gap = abs(candidate_share - reference_share)
            points = _HUNDRED * gap
            total_variation += gap
            maximum = max(maximum, points)
            rows.append(
                BinDiscrepancy(
                    bin_id=item.id,
                    unclassified=item.unclassified,
                    candidate_mass=mass,
                    candidate_share=candidate_share,
                    reference_count=count,
                    reference_share=reference_share,
                    discrepancy_pp=points,
                    within_bound=points <= block.max_each_bin_discrepancy_pp,
                )
            )
        total_variation *= _HALF
        within = (
            maximum <= block.max_each_bin_discrepancy_pp
            and total_variation <= block.max_total_variation
        )
        evidence["reference_share_sum"] = str(reference_share_sum)
        if fraction > partition.maximum_unclassified_fraction:
            problems.append(Reason.UNCLASSIFIED_FRACTION_EXCEEDED)
    if not within:
        problems.append(Reason.PUBLISHED_BENCHMARK_NONCONFORMITY)
    return CategoricalResult(
        block_id=block.id,
        geography_id=geography_id,
        candidate_denominator=denominator,
        reference_total=published_total,
        unclassified_mass=residual,
        unclassified_fraction=fraction,
        bins=tuple(rows),
        max_bin_discrepancy_pp=maximum,
        total_variation=total_variation,
        within_bound=within,
        independent_votes=block.independent_votes,
        reasons=dedupe(problems),
        evidence=MappingProxyType(evidence),
    )
