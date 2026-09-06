"""Origin support: clone aggregation, Kish ESS, concentration and carrier counts.

``/support`` measures how few authenticated original source households carry a
district's mass. Contributions are summed by original source-household key
*before* any count, effective-sample-size or maximum-share calculation, so
clones, aliases and pruned descendants of one source household count once
(``/source_household_origin/aliases_and_clones``).

Two limits of this axis are the document's, not this module's, and are kept
visible rather than argued away. ``/support/independence_or_bias_guarantee`` is
false: a satisfied floor is not an independence or unbiasedness result, and a
breached one is not proof of bias. ``/source_household_origin/shared_donor_limit``
adds that an origin key and an ESS do not establish independent households
across surveys or independent imputation donors.

The effective sample size is Kish's ``(sum c)^2 / sum(c^2)``, the same
concentration statistic :func:`microcosm.build.quality_aggregates.population_support`
reports; it is computed here in exact decimal arithmetic over origin-aggregated
contributions rather than over raw rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import StrEnum
from types import MappingProxyType

from microcosm.build.cd_benchmark.metrics import ARITHMETIC, MetricError, measure
from microcosm.build.cd_benchmark.protocol import SupportBounds
from microcosm.build.cd_benchmark.reasons import Reason, SupportVerdict, dedupe

__all__ = [
    "OriginTotals",
    "SourceRecord",
    "SupportCounts",
    "SupportMeasure",
    "SupportResult",
    "SupportScope",
    "aggregate_origins",
    "bin_support",
    "denominator_support",
]

_ZERO = Decimal(0)


class SupportMeasure(StrEnum):
    """What a contribution counts, so a substitution has to be declared.

    ``/support/person_measure_contribution`` forbids silently substituting
    household count weights for person mass. This module cannot detect that
    substitution inside a supplied number, so the measure is named on every
    result and the caller owns the claim.
    """

    PERSONS = "persons"
    OCCUPIED_HOUSING_UNITS = "occupied_housing_units"


class SupportScope(StrEnum):
    """Whether a support reading covers a denominator or a single bin."""

    DENOMINATOR = "denominator"
    BIN = "bin"


@dataclass(frozen=True)
class SourceRecord:
    """One in-scope candidate record, before clone aggregation.

    ``origin_key`` is ``None`` when the record's original-source lineage is not
    authenticated. Zero-weight records are in scope and must still carry one
    (``/support/origin_record_coverage``).
    """

    origin_key: str | None
    weight: object
    contribution: object


@dataclass(frozen=True)
class SupportCounts:
    """``/support/distinct_counts_required``."""

    actual_records: int
    positive_weight_records: int
    authenticated_original_origins: int
    positive_contribution_origins: int


@dataclass(frozen=True)
class OriginTotals:
    """Contributions summed by authenticated original source-household key."""

    totals: Mapping[str, Decimal]
    counts: SupportCounts
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class SupportResult:
    """One support reading for a district denominator or one bin."""

    scope: SupportScope
    block_id: str
    geography_id: str
    bin_id: str | None
    measure: SupportMeasure
    counts: SupportCounts
    contribution_total: Decimal | None
    effective_sample_size: Decimal | None
    largest_origin_share: Decimal | None
    verdict: SupportVerdict
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)
    bounds: SupportBounds | None = None


def aggregate_origins(records: Iterable[SourceRecord]) -> OriginTotals:
    """Sum contributions by original source key and count the required distincts.

    Records sharing an authenticated original source-household key merge into
    one origin regardless of how many clone or alias rows carry them. A record
    without an authenticated origin — zero weight included — yields
    ``unresolved_source_lineage``; a negative or nonfinite weight or
    contribution yields ``invalid_reducer_input``. Neither is silently dropped,
    clipped or absolute-valued.
    """
    problems: list[Reason] = []
    totals: dict[str, Decimal] = {}
    actual = 0
    positive_weight = 0
    with localcontext(ARITHMETIC):
        for record in records:
            actual += 1
            if record.origin_key is None:
                problems.append(Reason.UNRESOLVED_SOURCE_LINEAGE)
            elif not isinstance(record.origin_key, str) or not record.origin_key:
                problems.append(Reason.UNRESOLVED_SOURCE_LINEAGE)
            try:
                weight = measure(record.weight, "weight")
                contribution = measure(record.contribution, "contribution")
            except MetricError:
                problems.append(Reason.INVALID_REDUCER_INPUT)
                continue
            if weight < 0 or contribution < 0:
                problems.append(Reason.INVALID_REDUCER_INPUT)
                continue
            if weight > 0:
                positive_weight += 1
            if isinstance(record.origin_key, str) and record.origin_key:
                totals[record.origin_key] = (
                    totals.get(record.origin_key, _ZERO) + contribution
                )
    positive_origins = sum(1 for value in totals.values() if value > 0)
    return OriginTotals(
        totals=MappingProxyType(dict(totals)),
        counts=SupportCounts(
            actual_records=actual,
            positive_weight_records=positive_weight,
            authenticated_original_origins=len(totals),
            positive_contribution_origins=positive_origins,
        ),
        reasons=dedupe(problems),
    )


def _concentration(
    totals: Mapping[str, Decimal],
) -> tuple[Decimal, Decimal | None, Decimal | None]:
    with localcontext(ARITHMETIC):
        total = sum(totals.values(), _ZERO)
        if total <= 0:
            return total, None, None
        squares = sum((value * value for value in totals.values()), _ZERO)
        ess = (total * total) / squares
        largest = max(totals.values()) / total
    return total, ess, largest


def _reading(
    *,
    scope: SupportScope,
    bounds: SupportBounds,
    records: Iterable[SourceRecord],
    block_id: str,
    geography_id: str,
    bin_id: str | None,
    measure_kind: SupportMeasure,
    minimum_origins: int,
    maximum_share: Decimal,
    apply_ess_floor: bool,
    applicable: bool,
) -> SupportResult:
    aggregated = aggregate_origins(records)
    problems = list(aggregated.reasons)
    evidence: dict[str, object] = {
        "clone_aggregation": "contributions summed by original source key first",
        "independence_or_bias_guarantee": False,
        "minimum_positive_origins": minimum_origins,
        "maximum_origin_share": str(maximum_share),
        "ess_floor_applied": apply_ess_floor,
    }
    total, ess, largest = _concentration(aggregated.totals)
    if not applicable:
        evidence["carrier_floor"] = "not_applicable_zero_reference_share_bin"
        return SupportResult(
            scope=scope,
            bounds=bounds,
            block_id=block_id,
            geography_id=geography_id,
            bin_id=bin_id,
            measure=measure_kind,
            counts=aggregated.counts,
            contribution_total=total,
            effective_sample_size=ess,
            largest_origin_share=largest,
            verdict=SupportVerdict.NOT_APPLICABLE,
            reasons=dedupe(problems),
            evidence=MappingProxyType(evidence),
        )
    if problems:
        return SupportResult(
            scope=scope,
            bounds=bounds,
            block_id=block_id,
            geography_id=geography_id,
            bin_id=bin_id,
            measure=measure_kind,
            counts=aggregated.counts,
            contribution_total=total,
            effective_sample_size=ess,
            largest_origin_share=largest,
            verdict=SupportVerdict.UNDETERMINED,
            reasons=dedupe(problems),
            evidence=MappingProxyType(evidence),
        )
    breaches: list[str] = []
    if aggregated.counts.positive_contribution_origins < minimum_origins:
        breaches.append("positive_contribution_origins")
    if total <= 0:
        # No division by zero: the carrier floor above already records the
        # breach, and concentration stays undefined rather than infinite.
        evidence["zero_candidate_mass"] = True
    else:
        if apply_ess_floor and ess < bounds.minimum_origin_cluster_concentration_ess:
            breaches.append("effective_sample_size")
        if largest > maximum_share:
            breaches.append("largest_origin_share")
    if breaches:
        problems.append(Reason.SUPPORT_BOUND_EXCEEDED)
        evidence["breached_bounds"] = tuple(breaches)
    return SupportResult(
        scope=scope,
        bounds=bounds,
        block_id=block_id,
        geography_id=geography_id,
        bin_id=bin_id,
        measure=measure_kind,
        counts=aggregated.counts,
        contribution_total=total,
        effective_sample_size=ess,
        largest_origin_share=largest,
        verdict=(
            SupportVerdict.BOUND_EXCEEDED if breaches else SupportVerdict.WITHIN_BOUNDS
        ),
        reasons=dedupe(problems),
        evidence=MappingProxyType(evidence),
    )


def denominator_support(
    bounds: SupportBounds,
    records: Iterable[SourceRecord],
    *,
    block_id: str,
    geography_id: str,
    measure_kind: SupportMeasure,
) -> SupportResult:
    """Support for one district's relevant denominator.

    All three declared bounds apply: at least
    ``minimum_positive_source_household_origins`` positive-contribution origins,
    a Kish effective sample size at or above
    ``minimum_origin_cluster_concentration_ESS``, and a largest origin share at
    or below ``maximum_origin_denominator_share``. A known value below a floor
    is ``support_bound_exceeded``; it is not proof of bias.
    """
    return _reading(
        scope=SupportScope.DENOMINATOR,
        bounds=bounds,
        records=records,
        block_id=block_id,
        geography_id=geography_id,
        bin_id=None,
        measure_kind=measure_kind,
        minimum_origins=bounds.minimum_positive_source_household_origins,
        maximum_share=bounds.maximum_origin_denominator_share,
        apply_ess_floor=True,
        applicable=True,
    )


def bin_support(
    bounds: SupportBounds,
    records: Iterable[SourceRecord],
    *,
    block_id: str,
    geography_id: str,
    bin_id: str,
    measure_kind: SupportMeasure,
    reference_share_positive: bool,
) -> SupportResult:
    """Support for one published bin, using the same origin grouping.

    The carrier and concentration bounds apply to every *positive* published
    reference bin: at least
    ``minimum_positive_origins_per_positive_reference_bin`` positive origins and
    a largest origin share at or below ``maximum_origin_positive_bin_share``. A
    zero candidate bin mass fails the carrier floor without dividing by zero.
    A bin whose published reference share is zero — the protocol-constructed
    residual bin — has no carrier floor; its mass and origin diagnostics are
    still returned.

    The ``minimum_origin_cluster_concentration_ESS`` floor is *not* applied per
    bin: ``/support`` requires it "for every district and relevant denominator",
    and names only the carrier count and the maximum share for bins. The per-bin
    effective sample size is still computed and returned as a diagnostic.
    """
    return _reading(
        scope=SupportScope.BIN,
        bounds=bounds,
        records=records,
        block_id=block_id,
        geography_id=geography_id,
        bin_id=bin_id,
        measure_kind=measure_kind,
        minimum_origins=bounds.minimum_positive_origins_per_positive_reference_bin,
        maximum_share=bounds.maximum_origin_positive_bin_share,
        apply_ess_floor=False,
        applicable=reference_share_positive,
    )
