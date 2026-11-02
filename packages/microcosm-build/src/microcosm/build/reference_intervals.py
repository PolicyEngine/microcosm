"""Exact income-band intervals with declared semantics and rational cuts.

Country data declares every interval table: its entity, currency, input unit,
income and universe columns, closure, nil band, negative-income policy, first
lower cut, source locator and diagnostics window. This module supplies no
default for any of them. It validates the declarations, scales cuts exactly,
classifies signed integer-cent amounts and reports fixed-weight boundary
diagnostics.

Each cut is stored once: a positive band's lower cut is its predecessor's
upper cut, and the first positive band starts at the declared first lower cut.
Source cuts are integer cents. Scaling by a positive rational factor produces
reduced rational cents through integer cross-products and never rounds a
scaled cut. A scaled table keeps its source cuts and the digest of its source
table, and cannot be scaled again.

Classification validates every record before computing any band. A universe
status is ``included``, ``excluded`` or ``unresolved``; an unresolved record,
an invalid status or an invalid amount on an included record refuses with a
summary of counts by reason. Excluded records never enter a band. Amounts are
signed integers; booleans, floats, strings, missing and nonfinite values are
invalid, and nothing converts a float to cents.

Diagnostics weigh records with ``frame.resolve_weights(entity)``. They report
weighted mass near every finite cut and each positive band's weighted count
and amount under the declared closure and the other closure, computed
exactly. They never change a reference value or target. A published histogram
does not identify the publisher's mass exactly on a cut, so publisher
reclassification is unavailable: the report gives only the fixed-population
model measurement change. Records below the first cut remain visible, and
floored negative records retain their signed pre-floor amounts and their
destination under each closure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from microcosm.graph.canonical import canonical_json

__all__ = [
    "BELOW_FIRST_CUT",
    "EXCLUDED",
    "INTERVAL_DOCUMENT_SCHEMA_VERSION",
    "IntervalPreparationError",
    "IntervalTable",
    "alternative_closure",
    "boundary_diagnostics",
    "classify_records",
    "frame_boundary_diagnostics",
    "interval_document_sha256",
    "interval_indicators",
    "interval_input_fingerprint",
    "interval_table",
    "scale_interval_document",
    "scale_interval_table",
    "source_interval_table",
    "validate_interval_document",
]

INTERVAL_DOCUMENT_SCHEMA_VERSION = 1
SCALING_OPERATION = "exact_cut_multiplication"
BELOW_FIRST_CUT = -1
EXCLUDED = -2
CLOSURES = ("left", "right")
NIL_BAND_STATES = ("present", "absent")
NEGATIVE_INCOME_POLICIES = ("floor_zero", "refuse")
INPUT_UNITS = ("integer_cent",)
UNIVERSE_STATES = ("included", "excluded", "unresolved")
PUBLISHER_RECLASSIFICATION = "unavailable"
LIMITATION = (
    "A published histogram does not identify the publisher's mass exactly on a "
    "cut, so the publisher's reclassification under the alternative closure "
    "cannot be recovered. These are fixed-weight model measurement changes; "
    "reference values and calibration targets are unchanged."
)

_DOCUMENT_FIELDS = frozenset(
    {"schema_version", "country", "description", "disclosure", "tables"}
)
_TABLE_FIELDS = frozenset(
    {
        "id",
        "entity",
        "currency",
        "input_unit",
        "income_column",
        "universe_column",
        "source_period",
        "target_period",
        "positive_band_closure",
        "nil_band",
        "negative_income_policy",
        "first_lower_cut_cents",
        "source_locator",
        "diagnostics",
        "bands",
    }
)
_SCALED_TABLE_FIELDS = frozenset({"scaling", "source_first_lower_cut_cents"})
_OPTIONAL_TABLE_FIELDS = frozenset({"description"})
_BAND_FIELDS = frozenset({"id", "kind", "indicator"})
_OPTIONAL_BAND_FIELDS = frozenset(
    {"count_reference", "amount_reference", "source_label"}
)
_SCALING_FIELDS = frozenset(
    {
        "operation",
        "factor_id",
        "operational_factor",
        "factor_declaration_sha256",
        "factor_receipt_sha256",
        "source_table_sha256",
        "source_cuts_sha256",
    }
)
_DIAGNOSTIC_FIELDS = frozenset(
    {"boundary_window_cents", "alternative_positive_band_closure"}
)


class IntervalPreparationError(ValueError):
    """Records cannot be classified; ``summary`` counts every reason."""

    def __init__(self, summary: Mapping[str, object]) -> None:
        self.summary = dict(summary)
        super().__init__(
            "Interval preparation refused before emitting indicators: "
            + json.dumps(self.summary, sort_keys=True)
        )


@dataclass(frozen=True)
class IntervalTable:
    """One validated table. ``cuts`` holds the first lower cut then each finite upper cut."""

    id: str
    entity: str
    income_column: str
    universe_column: str
    closure: str
    nil_present: bool
    negative_income_policy: str
    window_cents: int
    band_ids: tuple[str, ...]
    indicators: tuple[str, ...]
    nil_index: int | None
    positive_indices: tuple[int, ...]
    cuts: tuple[Fraction, ...]
    scaled: bool
    declaration: Mapping[str, object]


def interval_document_sha256(document: object) -> str:
    """Hash canonical document bytes."""

    return hashlib.sha256(canonical_json(document)).hexdigest()


def alternative_closure(closure: str) -> str:
    """The other positive-band closure, whichever is declared."""

    if closure not in CLOSURES:
        raise ValueError("Positive band closure must be 'left' or 'right'.")
    return "left" if closure == "right" else "right"


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return dict(value)


def _closed(value: object, required: frozenset, optional: frozenset, label: str):
    row = _mapping(value, label)
    unknown = row.keys() - required - optional
    missing = required - row.keys()
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}.")
    if missing:
        raise ValueError(f"{label} is missing declared fields: {sorted(missing)}.")
    return row


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text.")
    return value


def _choice(value: object, choices: tuple[str, ...], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be declared as one of {list(choices)}.")
    return value


def _period(value: object, label: str):
    if isinstance(value, bool) or type(value) not in {int, str}:
        raise ValueError(f"{label} must be an integer or nonempty text.")
    if isinstance(value, str) and not value:
        raise ValueError(f"{label} must be an integer or nonempty text.")
    return value


def _integer_cents(value: object, label: str) -> int:
    # ``type`` rather than ``isinstance`` keeps booleans out; JSON floats,
    # infinity and NaN are floats and refuse here too.
    if type(value) is not int:
        raise ValueError(f"{label} must be integer cents, never a float or boolean.")
    return value


def _rational_cents(value: object, label: str) -> Fraction:
    row = _closed(value, frozenset({"numerator", "denominator"}), frozenset(), label)
    numerator, denominator = row["numerator"], row["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError(
            f"{label} requires integer numerator and positive integer denominator."
        )
    if math.gcd(numerator, denominator) != 1:
        raise ValueError(f"{label} must be a reduced rational.")
    return Fraction(numerator, denominator)


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _positive_factor(value: object) -> Fraction:
    row = _closed(
        value, frozenset({"numerator", "denominator"}), frozenset(), "scaling factor"
    )
    numerator, denominator = row["numerator"], row["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator <= 0
        or denominator <= 0
    ):
        raise ValueError("A scaling factor must be a positive integer rational.")
    return Fraction(numerator, denominator)


def _source_table(table: Mapping) -> dict:
    """Reconstruct the source declaration a scaled table was derived from."""

    source = {
        key: copy.deepcopy(value)
        for key, value in table.items()
        if key not in _SCALED_TABLE_FIELDS
    }
    source["first_lower_cut_cents"] = table["source_first_lower_cut_cents"]
    bands = []
    for band in table["bands"]:
        row = {key: copy.deepcopy(value) for key, value in band.items()}
        if "source_upper_cent" in row:
            row["upper_cent"] = row.pop("source_upper_cent")
        bands.append(row)
    source["bands"] = bands
    return source


def _source_cuts(source: Mapping) -> dict:
    return {
        "first_lower_cut_cents": source["first_lower_cut_cents"],
        "upper_cents": [
            band["upper_cent"] for band in source["bands"] if band["kind"] == "positive"
        ],
    }


def _cut_payload(table: IntervalTable) -> dict:
    return {
        "first_lower_cut_cents": _rational(table.cuts[0]),
        "upper_cents": [_rational(cut) for cut in table.cuts[1:]] + [None],
    }


def interval_table(value: object) -> IntervalTable:
    """Validate one source or scaled table declaration."""

    raw = _mapping(value, "interval table")
    scaled = "scaling" in raw
    required = _TABLE_FIELDS | (_SCALED_TABLE_FIELDS if scaled else frozenset())
    row = _closed(raw, required, _OPTIONAL_TABLE_FIELDS, "interval table")
    table_id = _text(row["id"], "Interval table id")
    entity = _text(row["entity"], "Interval table entity")
    _text(row["currency"], "Interval table currency")
    _choice(row["input_unit"], INPUT_UNITS, "Interval table input_unit")
    income = _text(row["income_column"], "Interval income column")
    universe = _text(row["universe_column"], "Interval universe column")
    if income == universe:
        raise ValueError("Interval income and universe columns must differ.")
    _period(row["source_period"], "Interval source period")
    _period(row["target_period"], "Interval target period")
    closure = _choice(
        row["positive_band_closure"], CLOSURES, "Interval positive_band_closure"
    )
    nil_present = (
        _choice(row["nil_band"], NIL_BAND_STATES, "Interval nil_band") == "present"
    )
    policy = _choice(
        row["negative_income_policy"],
        NEGATIVE_INCOME_POLICIES,
        "Interval negative_income_policy",
    )
    _text(row["source_locator"], "Interval source locator")
    if "description" in row:
        _text(row["description"], "Interval table description")
    diagnostics = _closed(
        row["diagnostics"], _DIAGNOSTIC_FIELDS, frozenset(), "interval diagnostics"
    )
    window = diagnostics["boundary_window_cents"]
    if type(window) is not int or window < 0:
        raise ValueError("boundary_window_cents must be a nonnegative integer.")
    if diagnostics["alternative_positive_band_closure"] != alternative_closure(closure):
        raise ValueError(
            "The diagnostic alternative must be the other positive band closure."
        )
    bands = row["bands"]
    if not isinstance(bands, list) or not bands:
        raise ValueError("An interval table needs a nonempty band list.")
    if scaled:
        first = _rational_cents(row["first_lower_cut_cents"], "Scaled first lower cut")
    else:
        first = Fraction(
            _integer_cents(row["first_lower_cut_cents"], "First lower cut")
        )
    if first < 0:
        raise ValueError("The first lower cut must be nonnegative.")
    band_ids, indicators, cuts, positive = [], [], [first], []
    nil_index, open_tail = None, False
    companions = set()
    for index, raw_band in enumerate(bands):
        optional = _OPTIONAL_BAND_FIELDS
        if scaled:
            optional = optional | {"source_upper_cent"}
        band = _mapping(raw_band, "interval band")
        kind = band.get("kind")
        required_band = _BAND_FIELDS | (
            {"upper_cent"} if kind == "positive" else frozenset()
        )
        band = _closed(band, frozenset(required_band), optional, "interval band")
        band_ids.append(_text(band["id"], "Interval band id"))
        indicators.append(_text(band["indicator"], "Interval band indicator"))
        if "source_label" in band:
            _text(band["source_label"], "Interval band source label")
        has_companions = tuple(
            key in band for key in ("count_reference", "amount_reference")
        )
        if has_companions not in {(True, True), (False, False)}:
            raise ValueError("A band names both companion references or neither.")
        companions.add(has_companions[0])
        if has_companions[0]:
            _text(band["count_reference"], "Band count reference")
            _text(band["amount_reference"], "Band amount reference")
            if band["count_reference"] == band["amount_reference"]:
                raise ValueError("A band's count and amount references must differ.")
        if kind == "nil":
            if not nil_present:
                raise ValueError("A table declaring nil_band absent has a nil band.")
            if index != 0:
                raise ValueError("A nil band, when present, comes first.")
            if "source_upper_cent" in band:
                raise ValueError("A nil band has no cut.")
            nil_index = index
            continue
        if kind != "positive":
            raise ValueError("Interval band kind must be 'nil' or 'positive'.")
        if open_tail:
            raise ValueError("Exactly one final open tail ends the band list.")
        positive.append(index)
        upper = band["upper_cent"]
        if upper is None:
            open_tail = True
            if scaled and band.get("source_upper_cent", 0) is not None:
                raise ValueError("A scaled open tail keeps a null source cut.")
            continue
        if scaled:
            value = _rational_cents(upper, "Scaled upper cut")
            if "source_upper_cent" not in band:
                raise ValueError("A scaled band keeps its source cut.")
        else:
            value = Fraction(_integer_cents(upper, "Upper cut"))
        if value <= 0 or value <= cuts[-1]:
            raise ValueError("Upper cuts must be positive and strictly increasing.")
        cuts.append(value)
    if nil_present and nil_index is None:
        raise ValueError("A table declaring nil_band present needs a first nil band.")
    if not open_tail:
        raise ValueError(
            "Exactly one final open tail with upper_cent null is required."
        )
    if not positive:
        raise ValueError("An interval table needs positive bands.")
    if len(companions) != 1:
        raise ValueError(
            "A table's bands either all name companion references or none do."
        )
    if len(set(band_ids)) != len(band_ids):
        raise ValueError("Interval band ids must be unique within a table.")
    if len(set(indicators)) != len(indicators) or {income, universe} & set(indicators):
        raise ValueError("Interval band indicators must be unique output columns.")
    table = IntervalTable(
        id=table_id,
        entity=entity,
        income_column=income,
        universe_column=universe,
        closure=closure,
        nil_present=nil_present,
        negative_income_policy=policy,
        window_cents=window,
        band_ids=tuple(band_ids),
        indicators=tuple(indicators),
        nil_index=nil_index,
        positive_indices=tuple(positive),
        cuts=tuple(cuts),
        scaled=scaled,
        declaration=copy.deepcopy(row),
    )
    if scaled:
        _validate_scaling(table, row)
    return table


def _validate_scaling(table: IntervalTable, row: Mapping) -> None:
    scaling = _closed(row["scaling"], _SCALING_FIELDS, frozenset(), "interval scaling")
    if scaling["operation"] != SCALING_OPERATION:
        raise ValueError("Unknown interval scaling operation.")
    _text(scaling["factor_id"], "Scaling factor id")
    factor = _positive_factor(scaling["operational_factor"])
    for key in (
        "factor_declaration_sha256",
        "factor_receipt_sha256",
        "source_table_sha256",
        "source_cuts_sha256",
    ):
        digest = scaling[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError(f"Scaling {key} must be a lowercase SHA-256.")
    source = _source_table(row)
    parsed = interval_table(source)
    if parsed.scaled:
        raise ValueError("A scaled table's source cannot itself be scaled.")
    if (
        interval_document_sha256(source) != scaling["source_table_sha256"]
        or interval_document_sha256(_source_cuts(source))
        != scaling["source_cuts_sha256"]
    ):
        raise ValueError("Scaled interval source table identity mismatch.")
    expected = tuple(cut * factor for cut in parsed.cuts)
    if expected != table.cuts:
        raise ValueError(
            "Scaled interval cuts must equal their source cuts times the factor "
            "exactly once."
        )


def validate_interval_document(document: object) -> dict[str, IntervalTable]:
    """Validate a whole document; return its tables by id in declared order."""

    row = _closed(
        document,
        frozenset({"schema_version", "tables"}),
        _DOCUMENT_FIELDS - {"schema_version", "tables"},
        "interval document",
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != INTERVAL_DOCUMENT_SCHEMA_VERSION
    ):
        raise ValueError("Interval documents require schema_version 1.")
    for key in ("country", "description", "disclosure"):
        if key in row:
            _text(row[key], f"Interval document {key}")
    tables = row["tables"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("An interval document needs a nonempty table list.")
    result: dict[str, IntervalTable] = {}
    outputs, references = set(), []
    for raw in tables:
        table = interval_table(raw)
        if table.id in result:
            raise ValueError("Interval table ids must be unique.")
        result[table.id] = table
        if outputs & set(table.indicators):
            raise ValueError("Interval indicator ids must be unique.")
        outputs.update(table.indicators)
        for band in table.declaration["bands"]:
            references.extend(
                band[key]
                for key in ("count_reference", "amount_reference")
                if key in band
            )
    if len(set(references)) != len(references):
        raise ValueError("A companion reference belongs to at most one band.")
    return result


def scale_interval_table(
    table: Mapping,
    factor: Mapping,
    *,
    factor_id: str,
    factor_declaration_sha256: str,
    factor_receipt_sha256: str,
) -> dict:
    """Multiply every cut once by a positive rational; never round a cut."""

    source = interval_table(table)
    if source.scaled:
        raise ValueError(f"Interval table {source.id!r} is already scaled.")
    rational = _positive_factor(factor)
    raw = copy.deepcopy(source.declaration)
    scaled = {
        key: value for key, value in raw.items() if key != "first_lower_cut_cents"
    }
    scaled["source_first_lower_cut_cents"] = raw["first_lower_cut_cents"]
    first = Fraction(raw["first_lower_cut_cents"]) * rational
    scaled["first_lower_cut_cents"] = _rational(first)
    bands = []
    for band in raw["bands"]:
        band = dict(band)
        if band["kind"] == "positive":
            cut = band["upper_cent"]
            band["source_upper_cent"] = cut
            band["upper_cent"] = None if cut is None else _rational(cut * rational)
        bands.append(band)
    scaled["bands"] = bands
    scaled["scaling"] = {
        "operation": SCALING_OPERATION,
        "factor_id": factor_id,
        # Recorded as declared (for example p / 10**digits), not reduced.
        "operational_factor": {
            "numerator": factor["numerator"],
            "denominator": factor["denominator"],
        },
        "factor_declaration_sha256": factor_declaration_sha256,
        "factor_receipt_sha256": factor_receipt_sha256,
        "source_table_sha256": interval_document_sha256(raw),
        "source_cuts_sha256": interval_document_sha256(_source_cuts(raw)),
    }
    interval_table(scaled)
    return scaled


def scale_interval_document(
    document: Mapping, scalings: Mapping[str, Mapping[str, object]]
) -> dict:
    """Return a document with only the named source tables scaled."""

    tables = validate_interval_document(document)
    if any(table.scaled for table in tables.values()):
        raise ValueError("A base interval document cannot contain scaled tables.")
    unknown = set(scalings) - set(tables)
    if unknown:
        raise ValueError(f"Unknown interval tables {sorted(unknown)}.")
    effective = copy.deepcopy(dict(document))
    effective["tables"] = [
        scale_interval_table(raw, **scalings[raw["id"]])
        if raw["id"] in scalings
        else raw
        for raw in effective["tables"]
    ]
    validate_interval_document(effective)
    return effective


def source_interval_table(table: Mapping) -> dict:
    """The source declaration of a scaled table, or the table itself."""

    parsed = interval_table(table)
    return _source_table(parsed.declaration) if parsed.scaled else dict(table)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _amount_reason(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return "boolean_amount"
    if isinstance(value, Integral):
        return None
    if value is None:
        return "missing_amount"
    if isinstance(value, float):
        if math.isnan(value):
            return "missing_amount"
        return "non_integer_amount" if math.isfinite(value) else "nonfinite_amount"
    if isinstance(value, str):
        return "text_amount"
    try:
        if value != value:  # pandas NA, numpy NaN scalars
            return "missing_amount"
    except TypeError:
        return "missing_amount"
    return "non_integer_amount"


def _validated(
    table: IntervalTable, amounts: Sequence[object], universe: Sequence[object]
) -> list[int | None]:
    """Validate every record, then return floored cents or None when excluded."""

    amounts, universe = list(amounts), list(universe)
    if len(amounts) != len(universe):
        raise ValueError("Interval amounts and universe statuses must align.")
    reasons, statuses, floored = Counter(), Counter(), []
    for amount, status in zip(amounts, universe, strict=True):
        if not isinstance(status, str) or status not in UNIVERSE_STATES:
            reasons["invalid_universe_status"] += 1
            statuses["invalid"] += 1
            floored.append(None)
            continue
        statuses[status] += 1
        if status == "excluded":
            floored.append(None)
            continue
        if status == "unresolved":
            reasons["unresolved_universe_status"] += 1
            floored.append(None)
            continue
        reason = _amount_reason(amount)
        if reason is not None:
            reasons[reason] += 1
            floored.append(None)
            continue
        value = int(amount)
        if value < 0 and table.negative_income_policy == "refuse":
            reasons["negative_amount_refused"] += 1
            floored.append(None)
            continue
        floored.append(max(0, value))
    if reasons:
        raise IntervalPreparationError(
            {
                "table": table.id,
                "entity": table.entity,
                "records": len(amounts),
                "refused_records": sum(reasons.values()),
                "reasons": dict(sorted(reasons.items())),
                "universe_statuses": dict(sorted(statuses.items())),
            }
        )
    return floored


def _cuts_at_or_below(z: int, cuts: Sequence[Fraction], inclusive: bool) -> int:
    """Count cuts ``c`` with ``c < z`` (or ``c <= z``) by cross-multiplication."""

    low, high = 0, len(cuts)
    while low < high:
        middle = (low + high) // 2
        cut = cuts[middle]
        scaled = z * cut.denominator
        if cut.numerator < scaled or (inclusive and cut.numerator == scaled):
            low = middle + 1
        else:
            high = middle
    return low


def _position(table: IntervalTable, z: int, closure: str) -> int:
    if table.nil_present and z == 0:
        return table.nil_index
    # Right-closed bands are ``lower < z <= upper`` and the tail ``z > last``;
    # left-closed bands are ``lower <= z < upper`` and the tail ``z >= last``.
    count = _cuts_at_or_below(z, table.cuts, inclusive=closure == "left")
    return BELOW_FIRST_CUT if count == 0 else table.positive_indices[count - 1]


def classify_records(
    table: IntervalTable,
    amounts: Sequence[object],
    universe: Sequence[object],
    *,
    closure: str | None = None,
) -> tuple[int, ...]:
    """Band index per record, ``BELOW_FIRST_CUT`` or ``EXCLUDED``.

    Every record is validated first; any refusal raises before a band is
    computed. ``closure`` defaults to the table's own declaration.
    """

    closure = (
        table.closure if closure is None else _choice(closure, CLOSURES, "closure")
    )
    floored = _validated(table, amounts, universe)
    return tuple(
        EXCLUDED if z is None else _position(table, z, closure) for z in floored
    )


def interval_indicators(
    table: IntervalTable, amounts: Sequence[object], universe: Sequence[object]
) -> dict[str, np.ndarray]:
    """One Boolean indicator per band, produced only after validation passes."""

    positions = np.asarray(classify_records(table, amounts, universe), dtype=np.int64)
    return {
        indicator: positions == index
        for index, indicator in enumerate(table.indicators)
    }


def _fingerprint_value(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        return value
    return [type(value).__name__, repr(value)]


def interval_input_fingerprint(values: Sequence[object]) -> str:
    """Digest of an input column that keeps value types, including invalid ones."""

    return interval_document_sha256([_fingerprint_value(value) for value in values])


# ---------------------------------------------------------------------------
# Boundary diagnostics
# ---------------------------------------------------------------------------


def _record_order(record_ids: Sequence[object]) -> list:
    result = []
    for value in record_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (str, Integral)
        ):
            raise ValueError("Diagnostic record identities are strings or integers.")
        result.append(value if isinstance(value, str) else int(value))
    if len(set(result)) != len(result):
        raise ValueError("Diagnostic record identities must be unique.")
    return result


def boundary_diagnostics(
    table: IntervalTable,
    record_ids: Sequence[object],
    amounts: Sequence[object],
    universe: Sequence[object],
    weights: Sequence[float],
    *,
    weight_record_ids: Sequence[object],
    weight_kind: str,
) -> dict[str, object]:
    """Fixed-weight boundary sensitivity, computed with exact weight arithmetic."""

    order = _record_order(record_ids)
    if _record_order(weight_record_ids) != order:
        raise ValueError("Diagnostic weights are not aligned with the record order.")
    values = np.asarray(weights, dtype="<f8")
    if values.shape != (len(order),) or not np.isfinite(values).all():
        raise ValueError("Diagnostic weights must be finite and record-aligned.")
    amounts = list(amounts)
    if len(amounts) != len(order):
        raise ValueError("Diagnostic amounts must be record-aligned.")
    alternative = alternative_closure(table.closure)
    floored = _validated(table, amounts, universe)
    zero = Fraction(0)
    slots = (*range(len(table.band_ids)), BELOW_FIRST_CUT)
    declared = {slot: [zero, zero, 0] for slot in slots}
    other = {slot: [zero, zero, 0] for slot in slots}
    near = [[zero, 0] for _ in table.cuts]
    floored_negative = [zero, zero, 0, 0]
    floored_destinations = {
        closure: {
            "nil": [zero, zero, 0, 0],
            "below_first_cut": [zero, zero, 0, 0],
            "none": [zero, zero, 0, 0],
            "positive_bands": {
                table.band_ids[index]: [zero, zero, 0, 0]
                for index in table.positive_indices
            },
        }
        for closure in (table.closure, alternative)
    }
    window = table.window_cents
    included = 0
    for weight, amount, z in zip(values.tolist(), amounts, floored, strict=True):
        if z is None:
            continue
        included += 1
        exact = Fraction(weight)
        raw_negative = int(amount) if int(amount) < 0 else None
        if raw_negative is not None:
            floored_negative[0] += exact
            floored_negative[1] += exact * raw_negative
            floored_negative[2] += 1
            floored_negative[3] += raw_negative
        for closure, sums in ((table.closure, declared), (alternative, other)):
            position = _position(table, z, closure)
            sums[position][0] += exact
            sums[position][1] += exact * z
            sums[position][2] += 1
            if raw_negative is not None:
                destinations = floored_destinations[closure]
                if position == table.nil_index:
                    destination = destinations["nil"]
                elif position == BELOW_FIRST_CUT:
                    destination = destinations[
                        "below_first_cut" if table.cuts[0] > 0 else "none"
                    ]
                else:
                    destination = destinations["positive_bands"][
                        table.band_ids[position]
                    ]
                destination[0] += exact
                destination[1] += exact * raw_negative
                destination[2] += 1
                destination[3] += raw_negative
        first = _cuts_at_or_below(z - window, table.cuts, inclusive=False)
        last = _cuts_at_or_below(z + window, table.cuts, inclusive=True)
        for index in range(first, last):
            near[index][0] += exact
            near[index][1] += 1
    cut_names = ["first_lower_cut"] + [
        table.band_ids[index] for index in table.positive_indices[:-1]
    ]
    source = _source_table(table.declaration) if table.scaled else table.declaration
    source_cuts = _source_cuts(source)
    source_values = [source_cuts["first_lower_cut_cents"], *source_cuts["upper_cents"]]
    cuts = [
        {
            "cut": name,
            "cut_cents": _rational(cut),
            "source_cut_cents": source_values[index],
            "near_cut_weighted_count": _rational(near[index][0]),
            "near_cut_records": near[index][1],
        }
        for index, (name, cut) in enumerate(zip(cut_names, table.cuts, strict=True))
    ]

    def measures(sums):
        count, amount, records = sums
        return {
            "weighted_count": _rational(count),
            "weighted_amount_cents": _rational(amount),
            "records": records,
        }

    def delta(slot):
        return {
            "weighted_count": _rational(other[slot][0] - declared[slot][0]),
            "weighted_amount_cents": _rational(other[slot][1] - declared[slot][1]),
            "records": other[slot][2] - declared[slot][2],
        }

    def floored_measures(sums):
        count, weighted_amount, records, amount = sums
        return {
            "records": records,
            "weighted_count": _rational(count),
            "pre_floor_amount_cents": amount,
            "weighted_pre_floor_amount_cents": _rational(weighted_amount),
        }

    def floored_by_destination(closure):
        destinations = floored_destinations[closure]
        return {
            "nil": floored_measures(destinations["nil"]),
            "below_first_cut": floored_measures(destinations["below_first_cut"]),
            "none": floored_measures(destinations["none"]),
            "positive_bands": {
                band_id: floored_measures(sums)
                for band_id, sums in destinations["positive_bands"].items()
            },
        }

    if table.nil_index is not None and (
        declared[table.nil_index] != other[table.nil_index]
    ):
        raise ValueError("The nil band must be identical under both closures.")
    bands = []
    for index, band_id in enumerate(table.band_ids):
        row = {
            "band": band_id,
            "kind": "nil" if index == table.nil_index else "positive",
            "declared": measures(declared[index]),
        }
        if index != table.nil_index:
            row["alternative"] = measures(other[index])
            row["delta"] = delta(index)
        bands.append(row)
    scaling = table.declaration.get("scaling")
    return {
        "schema_version": 1,
        "kind": "interval_boundary_diagnostics",
        "table": table.id,
        "entity": table.entity,
        "currency": table.declaration["currency"],
        "input_unit": table.declaration["input_unit"],
        "positive_band_closure": table.closure,
        "alternative_positive_band_closure": alternative,
        "nil_band": "present" if table.nil_present else "absent",
        "negative_income_policy": table.negative_income_policy,
        "boundary_window_cents": window,
        "weight_kind": weight_kind,
        "records": len(order),
        "included_records": included,
        "digests": {
            "table_sha256": interval_document_sha256(table.declaration),
            "source_cuts_sha256": interval_document_sha256(source_cuts),
            "effective_cuts_sha256": interval_document_sha256(_cut_payload(table)),
            "factor": None
            if scaling is None
            else {
                key: scaling[key]
                for key in (
                    "factor_id",
                    "operational_factor",
                    "factor_declaration_sha256",
                    "factor_receipt_sha256",
                )
            },
            "weights_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "record_order_sha256": interval_document_sha256(order),
        },
        "cuts": cuts,
        "bands": bands,
        "below_first_cut": {
            "declared": measures(declared[BELOW_FIRST_CUT]),
            "alternative": measures(other[BELOW_FIRST_CUT]),
            "delta": delta(BELOW_FIRST_CUT),
        },
        "floored_negative_income": {
            **floored_measures(floored_negative),
            "by_destination": {
                "declared": floored_by_destination(table.closure),
                "alternative": floored_by_destination(alternative),
            },
        },
        "publisher_reclassification": PUBLISHER_RECLASSIFICATION,
        "limitation": LIMITATION,
    }


def frame_boundary_diagnostics(table: IntervalTable, frame) -> dict[str, object]:
    """Diagnostics on a frame, weighted through ``resolve_weights(entity)``."""

    data = frame.table(table.entity)
    for column in (table.income_column, table.universe_column):
        if column not in data:
            raise ValueError(
                f"Interval diagnostics need {table.entity}.{column} on the frame."
            )
    weights = frame.resolve_weights(table.entity)
    ids = data[frame.schema.entity_id_column(table.entity)].tolist()
    return boundary_diagnostics(
        table,
        ids,
        data[table.income_column].tolist(),
        data[table.universe_column].tolist(),
        weights.values,
        weight_record_ids=ids,
        weight_kind=weights.kind.value,
    )
