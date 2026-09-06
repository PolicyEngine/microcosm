"""Assemble the six independent axes into one scope verdict, and nothing more.

``/status_contract/independent_axes_required`` names six axes that are reported
separately: benchmark metric, reference precision, mapping, support, exposure
and scope completeness. This module keeps them separate — no axis is collapsed
into another, and there is no single boolean that stands for all six.

``/status_contract/scope_reduction_order`` then gives the only permitted
reduction: an unapproved protocol prevents scoring at all; any incomplete
mandatory block or district makes the overall scope incomplete while every
already-known nonconformity is retained; otherwise any mandatory nonconformity
is nonconformity; otherwise conformity.

What a conforming verdict is not. ``/claim`` states that conformity does not
prove population truth or policy accuracy, that this is not a hypothesis test
of truth error, and that a true population may be nonconforming purely from
reference noise. ``/status_contract/no_rescue`` forbids rescuing a block with a
good national average, overlapping diagnostic votes, incumbent improvement, a
large reference margin of error, or a high clone count or effective sample size.
Accordingly :class:`BenchmarkEvaluation` exposes no release, certification or
eligibility field, and :attr:`BenchmarkEvaluation.claims_not_established` spells
out what a conforming result still does not establish.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from microcosm.build.cd_benchmark.approval import (
    ApprovalResolution,
    ApprovalScope,
)
from microcosm.build.cd_benchmark.binding import BindingResult
from microcosm.build.cd_benchmark.metrics import CategoricalResult, PointResult
from microcosm.build.cd_benchmark.protocol import (
    CompiledProtocol,
    DistributionBlock,
    PointBlock,
    ScopeRules,
)
from microcosm.build.cd_benchmark.reasons import (
    AXES,
    BenchmarkVerdict,
    ExposureStatus,
    MappingStatus,
    PrecisionStatus,
    Reason,
    SupportVerdict,
    dedupe,
)
from microcosm.build.cd_benchmark.reference import PrecisionResult
from microcosm.build.cd_benchmark.support import SupportResult

__all__ = [
    "BenchmarkEvaluation",
    "BlockEvaluation",
    "CLAIMS_NOT_ESTABLISHED",
    "district_scope_reasons",
    "evaluate_block",
    "evaluate_scope",
]

#: What a ``benchmark_conforming`` verdict still does not establish.
CLAIMS_NOT_ESTABLISHED = (
    "population truth",
    "policy accuracy",
    "a hypothesis test of truth error",
    "independent-source evidence",
    "national published reference conformity",
    "launch or release approval",
)

_SCOPE_REASONS = frozenset(
    {
        Reason.CANDIDATE_IDENTITY_UNAVAILABLE,
        Reason.DENOMINATOR_UNKNOWN,
        Reason.EXPOSURE_HISTORY_UNKNOWN,
        Reason.GQ_WEIGHT_BRIDGE_MISSING,
        Reason.IDENTITY_MISMATCH,
        Reason.INVALID_REDUCER_INPUT,
        Reason.LEDGER_BUSY,
        Reason.LEDGER_INTEGRITY_UNKNOWN,
        Reason.MAPPING_UNAPPROVED,
        Reason.PROTOCOL_UNAPPROVED,
        Reason.REFERENCE_INVALID,
        Reason.REFERENCE_PRECISION_INADEQUATE,
        Reason.REQUIRED_CELL_OR_DISTRICT_MISSING,
        Reason.SOURCE_DOMAIN_REFUSED,
        Reason.UNCLASSIFIED_FRACTION_EXCEEDED,
        Reason.UNRESOLVED_SOURCE_LINEAGE,
        Reason.VACANCY_BRIDGE_MISSING,
        Reason.ZERO_CANDIDATE_DENOMINATOR,
        Reason.ZERO_REFERENCE_DENOMINATOR,
    }
)
_NONCONFORMITY_REASONS = frozenset(
    {Reason.PUBLISHED_BENCHMARK_NONCONFORMITY, Reason.SUPPORT_BOUND_EXCEEDED}
)


@dataclass(frozen=True)
class BlockEvaluation:
    """One mandatory block's axes for one district, kept apart."""

    block_id: str
    geography_id: str
    metric: PointResult | CategoricalResult | None
    precision: tuple[PrecisionResult, ...]
    support: tuple[SupportResult, ...]
    exposure: ExposureStatus
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)

    @property
    def metric_conforming(self) -> bool | None:
        """The metric axis alone: ``None`` when it could not be measured."""
        return None if self.metric is None else self.metric.within_bound

    @property
    def scope_complete(self) -> bool:
        """Whether this block carries no scope-incompleteness reason."""
        return not (set(self.reasons) & _SCOPE_REASONS)


def evaluate_block(
    block: PointBlock | DistributionBlock,
    *,
    geography_id: str,
    metric: PointResult | CategoricalResult | None,
    precision: Iterable[PrecisionResult] = (),
    support: Iterable[SupportResult] = (),
    exposure: ExposureStatus,
    extra_reasons: Iterable[Reason] = (),
) -> BlockEvaluation:
    """Collect one block's axes without letting any of them overwrite another.

    Precision never widens the metric bound and support never rescues it; the
    reasons simply accumulate, deduplicated, from every axis that produced one.

    A mandatory block with an axis carrying *no evidence at all* is incomplete,
    not conforming: ``/status_contract/scope_incomplete`` covers unavailable
    precision, lineage and cell evidence, and
    ``/support/required_for_every_district_and_relevant_denominator`` makes
    support mandatory per district and denominator. An absent measurement is
    ``required_cell_or_district_missing``, absent precision evidence is
    ``reference_precision_inadequate``, and absent support evidence is
    ``unresolved_source_lineage`` — the code ``/support/unknown_lineage`` names
    when origin coverage cannot be established.
    """
    precision = tuple(precision)
    support = tuple(support)
    problems: list[Reason] = []
    if metric is not None:
        problems.extend(metric.reasons)
    for item in precision:
        problems.extend(item.reasons)
    for item in support:
        problems.extend(item.reasons)
    problems.extend(extra_reasons)
    if exposure is ExposureStatus.EXPOSURE_HISTORY_UNKNOWN:
        problems.append(Reason.EXPOSURE_HISTORY_UNKNOWN)
    absent: list[str] = []
    if block.mandatory:
        if metric is None:
            absent.append("metric")
            problems.append(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING)
        if not precision:
            absent.append("reference_precision")
            problems.append(Reason.REFERENCE_PRECISION_INADEQUATE)
        if not support:
            absent.append("support")
            problems.append(Reason.UNRESOLVED_SOURCE_LINEAGE)
    return BlockEvaluation(
        block_id=block.id,
        geography_id=geography_id,
        metric=metric,
        precision=precision,
        support=support,
        exposure=exposure,
        reasons=dedupe(problems),
        evidence=MappingProxyType(
            {
                "mandatory": block.mandatory,
                "absent_axis_evidence": tuple(absent),
                "precision_statuses": tuple(str(item.status) for item in precision),
                "support_verdicts": tuple(str(item.verdict) for item in support),
                "precision_does_not_widen_point_bound": True,
                "support_does_not_rescue_a_metric": True,
            }
        ),
    )


def district_scope_reasons(
    scope: ScopeRules,
    *,
    required_geography_ids: Iterable[str],
    present_geography_ids: Iterable[str],
    published_geography_ids: Mapping[str, str],
) -> tuple[tuple[Reason, ...], Mapping[str, object]]:
    """Check the district set, retaining both canonical and published identities.

    ``allowed_missing_mandatory_district_fraction`` is zero, so one missing
    district is enough. The published and canonical geography identities are
    both retained (``/scope/published_and_canonical_geography_ids_retained``),
    and the DC alias pair is required to be present as the document states it.
    """
    required = set(required_geography_ids)
    present = set(present_geography_ids)
    missing = tuple(sorted(required - present))
    unexpected = tuple(sorted(present - required))
    problems: list[Reason] = []
    notes: dict[str, object] = {
        "required_district_count": len(required),
        "declared_district_count": scope.district_count,
        "missing_districts": missing,
        "unexpected_districts": unexpected,
        "allowed_missing_fraction": str(
            scope.allowed_missing_mandatory_district_fraction
        ),
    }
    if missing or unexpected or len(required) != scope.district_count:
        problems.append(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING)
    for canonical, published in published_geography_ids.items():
        if canonical not in present:
            continue
        if canonical == scope.dc_canonical_id and published != scope.dc_published_id:
            notes["dc_alias_mismatch"] = True
            problems.append(Reason.IDENTITY_MISMATCH)
    if scope.dc_canonical_id in present and scope.dc_canonical_id not in (
        published_geography_ids
    ):
        notes["dc_published_identity_missing"] = True
        problems.append(Reason.IDENTITY_MISMATCH)
    return dedupe(problems), MappingProxyType(notes)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """The six axes plus the one permitted reduction. Never a release verdict."""

    protocol_sha256: str
    action: ApprovalScope
    scoring_permitted: bool
    verdict: BenchmarkVerdict
    axes: Mapping[str, object]
    blocks: tuple[BlockEvaluation, ...]
    reasons: tuple[Reason, ...]
    claims_not_established: tuple[str, ...] = CLAIMS_NOT_ESTABLISHED
    evidence: Mapping[str, object] = field(default_factory=dict)


def evaluate_scope(
    protocol: CompiledProtocol,
    *,
    approval: ApprovalResolution,
    binding: BindingResult,
    blocks: Sequence[BlockEvaluation],
    scope_reasons: Iterable[Reason] = (),
    ledger_reasons: Iterable[Reason] = (),
    ledger_writer_available: bool = True,
) -> BenchmarkEvaluation:
    """Reduce block axes to one scope verdict in the document's order.

    ``approval`` must come from
    :func:`~microcosm.build.cd_benchmark.approval.resolve_action` — which checks
    canonical bytes and a matching registration, and is the only thing that sets
    ``registration_resolved`` — and must authorize
    :data:`ApprovalScope.CANDIDATE_SCORING`. Implementation-only approval never
    permits scoring, so a fixture run reaches ``scope_incomplete`` with
    ``protocol_unapproved`` and every measured block result is still retained
    for inspection.

    ``allowed_missing_mandatory_block_fraction`` is zero, so every mandatory
    block the contract declares must appear for each supplied geography; a
    missing one is ``required_cell_or_district_missing``. A block from another
    district cannot fill that gap. District coverage is the caller's to
    establish with :func:`district_scope_reasons` and pass in through
    ``scope_reasons``.

    This package takes no ledger lock and creates no ledger.
    ``ledger_writer_available=False`` is a caller-declared state that yields
    ``ledger_busy``; it is not an observation this module makes.
    """
    problems: list[Reason] = []
    mandatory = {item.id for item in protocol.blocks if item.mandatory}
    missing_blocks = tuple(sorted(mandatory - {item.block_id for item in blocks}))
    present_by_geography: dict[str, set[str]] = {}
    for item in blocks:
        present_by_geography.setdefault(item.geography_id, set()).add(item.block_id)
    missing_block_geographies = tuple(
        (geography, block_id)
        for geography, present in sorted(present_by_geography.items())
        for block_id in sorted(mandatory - present)
    )
    if missing_blocks or missing_block_geographies:
        problems.append(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING)
    scoring = (
        approval.authorized
        and approval.registration_resolved
        and approval.action is ApprovalScope.CANDIDATE_SCORING
    )
    if not scoring:
        problems.append(Reason.PROTOCOL_UNAPPROVED)
    problems.extend(approval.reasons)
    problems.extend(binding.reasons)
    problems.extend(scope_reasons)
    problems.extend(ledger_reasons)
    if not ledger_writer_available:
        problems.append(Reason.LEDGER_BUSY)
    for block in blocks:
        problems.extend(block.reasons)
    reasons = dedupe(problems)
    known = set(reasons)
    if known & _SCOPE_REASONS:
        verdict = BenchmarkVerdict.SCOPE_INCOMPLETE
    elif known & _NONCONFORMITY_REASONS:
        verdict = BenchmarkVerdict.BENCHMARK_NONCONFORMING
    else:
        verdict = BenchmarkVerdict.BENCHMARK_CONFORMING
    axes = {
        "benchmark_metric_status": tuple(
            (item.block_id, item.geography_id, item.metric_conforming)
            for item in blocks
        ),
        "reference_precision_status": tuple(
            sorted(
                {str(reading.status) for item in blocks for reading in item.precision}
            )
        )
        or (str(PrecisionStatus.UNAVAILABLE),),
        "mapping_status": str(binding.status),
        "support_status": tuple(
            sorted(
                {str(reading.verdict) for item in blocks for reading in item.support}
            )
        )
        or (str(SupportVerdict.UNDETERMINED),),
        "exposure_status": tuple(
            (item.block_id, str(item.exposure)) for item in blocks
        ),
        "scope_completeness": verdict is not BenchmarkVerdict.SCOPE_INCOMPLETE,
    }
    if set(axes) != set(AXES):
        raise ValueError("The reported axes must be exactly the required six.")
    return BenchmarkEvaluation(
        protocol_sha256=protocol.sha256,
        action=approval.action,
        scoring_permitted=scoring,
        verdict=verdict,
        axes=MappingProxyType(axes),
        blocks=tuple(blocks),
        reasons=reasons,
        evidence=MappingProxyType(
            {
                "scope_reduction_order": protocol.scope_reduction_order,
                "missing_mandatory_blocks": missing_blocks,
                "missing_mandatory_block_geographies": missing_block_geographies,
                "allowed_missing_mandatory_block_fraction": str(
                    protocol.scope.allowed_missing_mandatory_block_fraction
                ),
                "mapping_status_is_unapproved_without_authentication": (
                    binding.status is MappingStatus.UNAPPROVED
                ),
                "nonconformities_retained": tuple(
                    item.block_id
                    for item in blocks
                    if Reason.PUBLISHED_BENCHMARK_NONCONFORMITY in item.reasons
                ),
                "ledger_writer_available": ledger_writer_available,
                "no_ledger_created_or_appended": True,
            }
        ),
    )
