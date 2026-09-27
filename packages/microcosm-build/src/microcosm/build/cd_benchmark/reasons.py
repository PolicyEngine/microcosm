"""Closed reason, exposure and status vocabularies for the CD benchmark protocol.

Every member here is checked against the canonical protocol document by
:func:`microcosm.build.cd_benchmark.protocol.assert_vocabulary_closed`, so the
code's vocabulary is verified equal to the approved document's rather than
merely resembling it. Per ``/status_contract/reason_encoding`` the primary
reason namespace is closed: unknown codes refuse validation, several
independently established reasons may coexist, and block/bin/metric identifiers
and raw source detail are separate bounded evidence, never a replacement code.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

__all__ = [
    "AXES",
    "BenchmarkVerdict",
    "ExposureStatus",
    "MappingStatus",
    "PrecisionStatus",
    "Reason",
    "SupportVerdict",
    "dedupe",
]


class Reason(StrEnum):
    """The 21 primary reason codes at ``/status_contract/reason_codes``."""

    CANDIDATE_IDENTITY_UNAVAILABLE = "candidate_identity_unavailable"
    DENOMINATOR_UNKNOWN = "denominator_unknown"
    EXPOSURE_HISTORY_UNKNOWN = "exposure_history_unknown"
    GQ_WEIGHT_BRIDGE_MISSING = "gq_weight_bridge_missing"
    IDENTITY_MISMATCH = "identity_mismatch"
    INVALID_REDUCER_INPUT = "invalid_reducer_input"
    LEDGER_BUSY = "ledger_busy"
    LEDGER_INTEGRITY_UNKNOWN = "ledger_integrity_unknown"
    MAPPING_UNAPPROVED = "mapping_unapproved"
    PROTOCOL_UNAPPROVED = "protocol_unapproved"
    PUBLISHED_BENCHMARK_NONCONFORMITY = "published_benchmark_nonconformity"
    REFERENCE_INVALID = "reference_invalid"
    REFERENCE_PRECISION_INADEQUATE = "reference_precision_inadequate"
    REQUIRED_CELL_OR_DISTRICT_MISSING = "required_cell_or_district_missing"
    SOURCE_DOMAIN_REFUSED = "source_domain_refused"
    SUPPORT_BOUND_EXCEEDED = "support_bound_exceeded"
    UNCLASSIFIED_FRACTION_EXCEEDED = "unclassified_fraction_exceeded"
    UNRESOLVED_SOURCE_LINEAGE = "unresolved_source_lineage"
    VACANCY_BRIDGE_MISSING = "vacancy_bridge_missing"
    ZERO_CANDIDATE_DENOMINATOR = "zero_candidate_denominator"
    ZERO_REFERENCE_DENOMINATOR = "zero_reference_denominator"


class ExposureStatus(StrEnum):
    """The four values at ``/exposure_ledger/exposure_status_values``."""

    EXPOSURE_HISTORY_UNKNOWN = "exposure_history_unknown"
    FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE = "fresh_same_source_holdout_eligible"
    NOT_RESERVED_CONFORMITY = "not_reserved_conformity"
    POST_EXPOSURE_CONFORMITY = "post_exposure_conformity"


class BenchmarkVerdict(StrEnum):
    """Scope-level outcomes named by ``/status_contract``.

    There is deliberately no "release", "certified" or "passing" member: a
    conforming verdict is operational published-benchmark conformity for the
    stated scope and nothing else (``/claim``).
    """

    BENCHMARK_CONFORMING = "benchmark_conforming"
    BENCHMARK_NONCONFORMING = "benchmark_nonconforming"
    SCOPE_INCOMPLETE = "scope_incomplete"


class PrecisionStatus(StrEnum):
    """The reference-precision axis, independent of the metric axis.

    ``CONTROLLED_UNKNOWN`` is the ``/reference_status/controlled_estimate``
    class: the point comparison stays valid, no variance is claimed, and no
    adequacy is asserted. It is neither an adequacy verdict nor a failure.
    """

    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"
    CONTROLLED_UNKNOWN = "controlled_unknown"
    UNAVAILABLE = "unavailable"


class MappingStatus(StrEnum):
    """Whether an authenticated producer bound the required mapping slots."""

    APPROVED = "approved"
    UNAPPROVED = "unapproved"


class SupportVerdict(StrEnum):
    """Origin-support outcome, independent of the benchmark metric.

    ``NOT_APPLICABLE`` is the ``/support/reference_zero_unknown_bin_carrier_floor``
    case: the carrier floor does not apply to a bin whose published reference
    share is zero, so mass and origin diagnostics are retained without a bound.
    """

    WITHIN_BOUNDS = "within_bounds"
    BOUND_EXCEEDED = "bound_exceeded"
    UNDETERMINED = "undetermined"
    NOT_APPLICABLE = "not_applicable"


#: ``/status_contract/independent_axes_required`` — reported separately and
#: never collapsed into one scalar.
AXES = (
    "benchmark_metric_status",
    "reference_precision_status",
    "mapping_status",
    "support_status",
    "exposure_status",
    "scope_completeness",
)


def dedupe(items: Iterable[Reason]) -> tuple[Reason, ...]:
    """Deduplicate reasons, preserving first-seen order.

    Raises:
        TypeError: If any item is outside the closed :class:`Reason` namespace.
    """
    seen: dict[Reason, None] = {}
    for item in items:
        if not isinstance(item, Reason):
            raise TypeError(f"Unknown primary reason code {item!r}.")
        seen.setdefault(item)
    return tuple(seen)
