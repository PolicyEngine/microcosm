"""Published reference cells: validity classes and the precision axis.

Classification is not reimplemented here. Sentinel values, annotation
precedence and annotation/sentinel conflicts come from the shipped adapter,
:func:`microcosm.build.us_runtime.cd_reference.classify_acs_value`, imported
lazily so the pure core of this package never pulls the eager ``us_runtime``
package in. This module only maps that adapter's class names onto the closed
``/reference_status/invalid_classes`` vocabulary and keeps the adapter's own
class string verbatim as evidence, because ``/reference_status/invalid_treatment``
requires the exact reason and raw values to survive.

Precision is a separate axis from the benchmark metric.
``/claim/precision_does_not_widen_point_bound`` is true, so nothing here is ever
an input to :func:`microcosm.build.cd_benchmark.metrics.point_check`; a wide
margin of error never loosens a point bound, and it is not a test of an unknown
true value. A controlled estimate is its own precision class: the point
comparison stays valid and no zero-variance or zero-nonsampling-error claim is
made from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from enum import StrEnum
from types import MappingProxyType

from microcosm.build.cd_benchmark.metrics import ARITHMETIC, MetricError, measure
from microcosm.build.cd_benchmark.reasons import PrecisionStatus, Reason

__all__ = [
    "ADAPTER_CLASSES",
    "ClassifiedValue",
    "INVALID_CLASSES",
    "PrecisionResult",
    "REQUIRED_RAW_FIELDS",
    "ReferenceCell",
    "ReferenceError",
    "ReferenceValidity",
    "classify_reference_value",
    "discrepancy_over_moe",
    "published_percentage_precision",
    "reference_cell",
    "reference_validity_reasons",
    "relative_moe_precision",
    "subset_share_precision",
]

#: ``/reference_status/raw_fields_required`` — every one is mandatory; a cell
#: missing any of them is not a usable published observation.
REQUIRED_RAW_FIELDS = (
    "E",
    "M",
    "EA",
    "MA",
    "published_GEO_ID",
    "canonical_geography_id",
    "dataset",
    "table",
    "variable",
    "year",
    "congress",
    "units",
    "universe",
)


class ReferenceValidity(StrEnum):
    """Protocol validity classes, plus the two usable ones."""

    NUMERIC = "numeric"
    CONTROLLED_ESTIMATE = "controlled_estimate"
    MISSING = "missing"
    INSUFFICIENT_ESTIMATE = "insufficient_estimate"
    INSUFFICIENT_GEOGRAPHIC_SAMPLE = "insufficient_geographic_sample"
    INAPPLICABLE = "inapplicable"
    MOE_UNAVAILABLE = "moe_unavailable"
    OPEN_INTERVAL = "open_interval"
    UNKNOWN_ANNOTATION = "unknown_annotation"
    ANNOTATION_CONFLICT = "annotation_conflict"


#: ``/reference_status/invalid_classes``.
INVALID_CLASSES = frozenset(
    {
        ReferenceValidity.MISSING,
        ReferenceValidity.INSUFFICIENT_ESTIMATE,
        ReferenceValidity.INSUFFICIENT_GEOGRAPHIC_SAMPLE,
        ReferenceValidity.INAPPLICABLE,
        ReferenceValidity.MOE_UNAVAILABLE,
        ReferenceValidity.OPEN_INTERVAL,
        ReferenceValidity.UNKNOWN_ANNOTATION,
        ReferenceValidity.ANNOTATION_CONFLICT,
    }
)

#: Adapter class name -> protocol validity class. The adapter separates a
#: ``***`` margin-of-error open interval from a ``123+``-style open-interval
#: annotation; the protocol names one ``open_interval`` class for both, so the
#: adapter's own string is retained on every :class:`ClassifiedValue`.
ADAPTER_CLASSES = MappingProxyType(
    {
        "numeric": ReferenceValidity.NUMERIC,
        "controlled_estimate": ReferenceValidity.CONTROLLED_ESTIMATE,
        "missing": ReferenceValidity.MISSING,
        "insufficient_sample": ReferenceValidity.INSUFFICIENT_ESTIMATE,
        "insufficient_geographic_sample": (
            ReferenceValidity.INSUFFICIENT_GEOGRAPHIC_SAMPLE
        ),
        "not_applicable_or_available": ReferenceValidity.INAPPLICABLE,
        "moe_insufficient_sample": ReferenceValidity.MOE_UNAVAILABLE,
        "moe_open_interval": ReferenceValidity.OPEN_INTERVAL,
        "open_interval": ReferenceValidity.OPEN_INTERVAL,
        "unknown_annotation": ReferenceValidity.UNKNOWN_ANNOTATION,
    }
)


class ReferenceError(ValueError):
    """A published cell cannot be read as the protocol requires."""


@dataclass(frozen=True)
class ClassifiedValue:
    """One published estimate or margin of error, with its raw evidence kept."""

    raw: object
    annotation: object
    adapter_class: str
    validity: ReferenceValidity
    numeric_value: Decimal | None
    annotation_conflict: bool

    @property
    def usable(self) -> bool:
        """Whether this value carries an ordinary finite numeric reading."""
        return self.validity is ReferenceValidity.NUMERIC


def classify_reference_value(raw: object, annotation: object) -> ClassifiedValue:
    """Classify one raw published cell through the shipped ACS adapter.

    Raises:
        ReferenceError: If the adapter refuses the value, or returns a class
            this package has no protocol mapping for.
    """
    from microcosm.build.us_runtime.cd_reference import classify_acs_value

    try:
        result = classify_acs_value(raw, annotation)
    except ValueError as error:
        raise ReferenceError(f"Adapter refused the published cell: {error}") from None
    adapter_class = result["class"]
    if result["annotation_conflict"]:
        validity = ReferenceValidity.ANNOTATION_CONFLICT
    else:
        validity = ADAPTER_CLASSES.get(adapter_class)
    if validity is None:
        raise ReferenceError(f"Unmapped adapter class {adapter_class!r}.")
    numeric = result["numeric_value"]
    return ClassifiedValue(
        raw=raw,
        annotation=annotation,
        adapter_class=adapter_class,
        validity=validity,
        numeric_value=(
            None
            if numeric is None
            else measure(
                Decimal(raw.strip()) if isinstance(raw, str) else numeric,
                "reference value",
            )
        ),
        annotation_conflict=bool(result["annotation_conflict"]),
    )


@dataclass(frozen=True)
class ReferenceCell:
    """A published cell with the complete raw field set the protocol requires."""

    variable: str
    table: str
    dataset: str
    year: int
    congress: int
    units: str
    universe: str
    published_geo_id: str
    canonical_geography_id: str
    estimate: ClassifiedValue
    moe: ClassifiedValue


def reference_cell(fields: Mapping[str, object]) -> ReferenceCell:
    """Build a :class:`ReferenceCell` from the complete required raw field set.

    Raises:
        ReferenceError: If any required raw field is absent, or the estimate
            carries a negative ordinary numeric count (never a valid count).
    """
    missing = [name for name in REQUIRED_RAW_FIELDS if name not in fields]
    if missing:
        raise ReferenceError(f"Reference cell is missing raw fields {missing}.")
    for name in ("year", "congress"):
        if (
            isinstance(fields[name], bool)
            or not isinstance(fields[name], int)
            or fields[name] <= 0
        ):
            raise ReferenceError(f"{name} must be a positive integer.")
    for name in (
        "variable",
        "table",
        "dataset",
        "units",
        "universe",
        "published_GEO_ID",
        "canonical_geography_id",
    ):
        if not isinstance(fields[name], str) or not fields[name].strip():
            raise ReferenceError(f"{name} must be a nonempty string.")
    estimate = classify_reference_value(fields["E"], fields["EA"])
    moe = classify_reference_value(fields["M"], fields["MA"])
    for value, label in ((estimate, "estimate"), (moe, "margin of error")):
        if value.usable and value.numeric_value < 0:
            raise ReferenceError(f"Negative ordinary numeric {label} is not a count.")
    return ReferenceCell(
        variable=str(fields["variable"]),
        table=str(fields["table"]),
        dataset=str(fields["dataset"]),
        year=int(fields["year"]),
        congress=int(fields["congress"]),
        units=str(fields["units"]),
        universe=str(fields["universe"]),
        published_geo_id=str(fields["published_GEO_ID"]),
        canonical_geography_id=str(fields["canonical_geography_id"]),
        estimate=estimate,
        moe=moe,
    )


def reference_validity_reasons(cell: ReferenceCell) -> tuple[Reason, ...]:
    """Return ``reference_invalid`` when the estimate class is unusable.

    The margin of error is handled by the precision axis, not here: an invalid
    MOE beside an ordinary estimate does not invalidate the estimate.
    """
    if cell.estimate.validity in INVALID_CLASSES:
        return (Reason.REFERENCE_INVALID,)
    return ()


@dataclass(frozen=True)
class PrecisionResult:
    """One reference-precision reading, independent of the metric axis."""

    kind: str
    status: PrecisionStatus
    measured: Decimal | None
    ceiling: Decimal
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)
    cells: tuple[ReferenceCell, ...] = ()


def _unavailable(
    kind: str,
    ceiling: Decimal,
    evidence: dict[str, object],
    cells: tuple[ReferenceCell, ...],
) -> PrecisionResult:
    return PrecisionResult(
        kind=kind,
        status=PrecisionStatus.UNAVAILABLE,
        measured=None,
        ceiling=ceiling,
        reasons=(Reason.REFERENCE_PRECISION_INADEQUATE,),
        evidence=MappingProxyType(evidence),
        cells=cells,
    )


def _controlled(kind: str, ceiling: Decimal, cell: ReferenceCell) -> PrecisionResult:
    return PrecisionResult(
        kind=kind,
        status=PrecisionStatus.CONTROLLED_UNKNOWN,
        measured=None,
        ceiling=ceiling,
        reasons=(),
        cells=(cell,),
        evidence=MappingProxyType(
            {
                "raw_moe": cell.moe.raw,
                "moe_annotation": cell.moe.annotation,
                "numeric_moe": None,
                "treatment": (
                    "point comparison remains valid; separate controlled "
                    "precision class; no variance or nonsampling-error claim"
                ),
            }
        ),
    )


def _verdict(
    kind: str,
    measured: Decimal,
    ceiling: Decimal,
    evidence: dict[str, object],
    cells: tuple[ReferenceCell, ...],
) -> PrecisionResult:
    within = measured <= ceiling
    return PrecisionResult(
        kind=kind,
        status=PrecisionStatus.ADEQUATE if within else PrecisionStatus.INADEQUATE,
        measured=measured,
        ceiling=ceiling,
        reasons=() if within else (Reason.REFERENCE_PRECISION_INADEQUATE,),
        evidence=MappingProxyType(evidence),
        cells=cells,
    )


def relative_moe_precision(cell: ReferenceCell, *, ceiling: Decimal) -> PrecisionResult:
    """Relative margin of error ``M / E`` against a point block's ceiling.

    A controlled margin of error returns the controlled class rather than any
    adequacy claim. A zero estimate leaves the ratio undefined; that is recorded
    as unavailable precision, never as a perfect one.
    """
    kind = "relative_moe"
    cells = (cell,)
    if (
        cell.estimate.usable
        and cell.moe.validity is ReferenceValidity.CONTROLLED_ESTIMATE
    ):
        return _controlled(kind, ceiling, cell)
    evidence: dict[str, object] = {
        "estimate_class": cell.estimate.adapter_class,
        "moe_class": cell.moe.adapter_class,
    }
    if not (cell.estimate.usable and cell.moe.usable):
        return _unavailable(kind, ceiling, evidence, cells)
    if cell.estimate.numeric_value == 0:
        evidence["zero_reference_estimate"] = True
        return _unavailable(kind, ceiling, evidence, cells)
    with localcontext(ARITHMETIC):
        measured = cell.moe.numeric_value / cell.estimate.numeric_value
    return _verdict(kind, measured, ceiling, evidence, cells)


def published_percentage_precision(
    cell: ReferenceCell, *, ceiling: Decimal
) -> PrecisionResult:
    """A published percentage margin of error, already in percentage points.

    This is the age block's rule: the published percentage MOE column is used
    as printed, retained with its rounded published percentage. Benchmark shares
    still come from the count column, not from this percentage.
    """
    kind = "published_percentage_moe_pp"
    cells = (cell,)
    if (
        cell.estimate.usable
        and cell.moe.validity is ReferenceValidity.CONTROLLED_ESTIMATE
    ):
        return _controlled(kind, ceiling, cell)
    evidence: dict[str, object] = {
        "published_percentage": cell.estimate.raw,
        "moe_class": cell.moe.adapter_class,
    }
    if not (cell.estimate.usable and cell.moe.usable):
        return _unavailable(kind, ceiling, evidence, cells)
    return _verdict(kind, cell.moe.numeric_value, ceiling, evidence, cells)


def subset_share_precision(
    numerator: ReferenceCell,
    denominator: ReferenceCell,
    *,
    ceiling: Decimal,
) -> PrecisionResult:
    """The frozen descriptive subset-share approximation, in percentage points.

    ``100 * sqrt(Mx^2 - (X / Y)^2 * My^2) / Y``, with the documented plus
    fallback when the radicand is negative. A missing or controlled input has no
    approximation at all: nothing is fabricated, and the result is unavailable.
    """
    kind = "subset_share_moe_pp"
    cells = (numerator, denominator)
    evidence: dict[str, object] = {
        "numerator_variable": numerator.variable,
        "denominator_variable": denominator.variable,
        "numerator_estimate_class": numerator.estimate.adapter_class,
        "numerator_moe_class": numerator.moe.adapter_class,
        "denominator_estimate_class": denominator.estimate.adapter_class,
        "denominator_moe_class": denominator.moe.adapter_class,
        "formula": "100*sqrt(Mx^2-(X/Y)^2*My^2)/Y",
    }
    inputs = (
        numerator.estimate,
        numerator.moe,
        denominator.estimate,
        denominator.moe,
    )
    if not all(item.usable for item in inputs):
        evidence["no_fabricated_approximation"] = True
        return _unavailable(kind, ceiling, evidence, cells)
    estimate_x = numerator.estimate.numeric_value
    moe_x = numerator.moe.numeric_value
    estimate_y = denominator.estimate.numeric_value
    moe_y = denominator.moe.numeric_value
    if estimate_y == 0:
        evidence["zero_reference_denominator"] = True
        return _unavailable(kind, ceiling, evidence, cells)
    with localcontext(ARITHMETIC):
        share = estimate_x / estimate_y
        radicand = moe_x**2 - share**2 * moe_y**2
        fallback = radicand < 0
        if fallback:
            radicand = moe_x**2 + share**2 * moe_y**2
        try:
            measured = Decimal(100) * radicand.sqrt() / estimate_y
        except (InvalidOperation, DivisionByZero, MetricError):
            return _unavailable(kind, ceiling, evidence, cells)
    evidence["negative_radicand_plus_fallback"] = fallback
    return _verdict(kind, measured, ceiling, evidence, cells)


def discrepancy_over_moe(
    absolute_discrepancy: Decimal, cell: ReferenceCell
) -> Decimal | None:
    """Diagnostic ratio of an absolute discrepancy to a numeric margin of error.

    Defined only when the margin of error is ordinary and strictly positive.
    ``/reference_status/discrepancy_over_MOE`` makes this diagnostic only: it
    never changes a benchmark status, and no caller of this package feeds it
    into a verdict.
    """
    if not cell.moe.usable or cell.moe.numeric_value <= 0:
        return None
    with localcontext(ARITHMETIC):
        return absolute_discrepancy / cell.moe.numeric_value
