"""The first monetary amounts typed out of the 2015 PUF delivery bytes.

The reviewed raw source slice
(:mod:`microcosm.build.us_runtime.puf_raw_source`) types six of the 222
delivered main columns and the six demographic codes, reports
``amounts_decoded: False``, and drops the other 216 main columns after
checking their header name, order and record width. This module is the next
bounded increment: it authenticates the **whole** delivered layout against the
retained IRS booklets and decodes a small, explicitly enumerated set of
monetary columns at source-return grain.

Two things it produces, and one it refuses.

*The full layout is authenticated, not merely counted.* The packaged document
``puf_2015_monetary_source_projection.json`` locates every one of the 222 main
columns and 7 demographic columns in the September 2022 booklet by the
publisher's own position number and name, retains that record-layout row
verbatim with its exact character span and span digest, and corroborates each
against the February 2023 edition. Nothing about a column's role is inferred
from an ``E``, ``P``, ``S`` or ``T`` prefix or from membership of the
amount-field list: the publisher's own "Misc Codes" block sits inside that list
and holds ``RECID``, ``S006``, ``S008``, ``S009``, ``WSAMP`` and ``TXRT``, none
of which is money and one of which (``TXRT``) carries an implied decimal point.

*Twelve monetary columns are decoded, each with its own declaration.* Wages,
taxable and tax-exempt interest, ordinary and qualified dividends, the
Schedule C and Schedule F net results, pensions received and pensions in AGI,
and the three Schedule D quantities the booklets define. Each declares exact
units, sign evidence, missingness policy, return grain, source-period status
and source pin identity, and each is refused for target-year use.

*No amount is grown, restated or combined.* The row period is unresolved — the
cover says Tax Year 2015, the code definitions print FLPDYR 2011-2014, the
disclosure section removed pre-2012 returns, and the reviewed header audit
observed 2012-2015 — so every column carries
``target_year_use: refused_pending_independent_growth_decision``. Component
relationships the booklets do not state (qualified dividends inside ordinary
dividends, pensions in AGI inside pensions received, ``E01000`` out of
``P22250`` and ``P23250``) are recorded as unresolved and never computed.
``E22250`` is absent from the delivered layout and from both booklets and is
never aliased onto ``P22250``. An uncapped self-employment earnings derivation
from ``E30400``/``E30500`` is refused with the publisher text that refuses it.

Three properties are load-bearing.

*Every delivered main record is projected.* All 207,696 rows, including the
88,021 with no demographic row and the four disclosure aggregate records. The
aggregate rows are tagged and every derived fact is reported for three
disjoint row classes — aggregate, demographic-matched and
demographic-unmatched — so an aggregate can never be averaged into a
per-return statistic and an unmatched row is never a zero.

*The individual-return amount universe is explicit.* A row is inside it if and
only if its authenticated disclosure-aggregate flag is 0, and the decoded
object exposes that rule as a read-only ``amount_known`` mask derived from the
flag on every access. Individual returns keep the unchanged integer,
whole-dollar, twelve-digit grammar, and a fractional token on one of them
still refuses. The four disclosure aggregates are outside that universe: their
typed slots hold the minimum-int64 out-of-universe sentinel — never a zero,
which is a delivered value in these columns — and their delivered tokens are
retained exactly under a separate, explicitly bounded lexical grammar that is
an operational input bound for this pinned file rather than a publisher
specification. Nothing converts an aggregate token to a number, rounds it,
grows it or drops its row. The aggregate row class carries only token-class
counts and a token width, so no header, receipt or summary can state an
aggregate amount, and the exact row/token pairs are available only through a
local accessor on the decoded object.

*The projection is bound to the reviewed status artifact.* The node takes
``microcosm.us.puf_2015_raw_return_status`` as an artifact input and refuses
unless its own ``RECID`` sequence and aggregate flags are element-wise
identical to that artifact's. ``demographic_status`` is carried from it rather
than re-joined, so the two artifacts cannot disagree about which returns
matched.

*The raw producer has an explicit packaging revision.* This module imports
:mod:`puf_raw_source` for its definition, codec and artifact decoder. Moving
the unchanged definition bytes into ``us_runtime`` changed the raw resource
loader and its implementation identity. New runs use that revised identity;
previously accepted raw artifacts retain their original keys and evidence.
The CSV reader helpers here remain local to this monetary projection.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType

import numpy as np

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry, load_source_bytes

from .puf_raw_source import (
    PUF_AGGREGATE_RECIDS,
    PUF_RAW_SOURCE_NODE,
    RETURN_STATUS_TYPE,
    PufRawSourceDefinition,
    csv_acceptance_profile,
    decode_return_status,
    puf_raw_source_codecs,
)
from .puf_raw_source import packaged_definition as packaged_raw_definition

__all__ = [
    "AGGREGATE_ROW_CLASS",
    "AGGREGATE_SENTINEL",
    "AMOUNT_KNOWN_RULE",
    "BODY_MAX_BYTES",
    "HEADER_MAX_BYTES",
    "LEXICAL_WIDTH_MAX",
    "MONETARY_PROJECTION_NODE",
    "MONETARY_PROJECTION_STAGE",
    "MONETARY_PROJECTION_SUMMARY_NODE",
    "MONETARY_PROJECTION_TYPE",
    "PufMonetaryProjection",
    "PufMonetaryRefusalError",
    "PufMonetarySourceDocument",
    "ProjectedColumn",
    "RETURN_ROW_CLASSES",
    "SUPERSEDED_MAGIC",
    "USPufMonetaryProjectionSummaryKernel",
    "USPufMonetarySourceKernel",
    "decode_monetary_projection",
    "decode_puf_monetary_source",
    "encode_monetary_projection",
    "fixture_projection_document",
    "packaged_projection",
    "projection_document_json",
    "register_us_puf_monetary_source_kernels",
    "us_puf_monetary_projection_summary_node",
    "us_puf_monetary_source_node",
]

MONETARY_PROJECTION_TYPE = ArtifactType(
    "microcosm.us.puf_2015_monetary_source_projection", 3
)
MONETARY_PROJECTION_MAGIC = b"microcosm.us.puf_2015_monetary_source_projection/3\n"

#: Magic lines this reader refuses by name instead of reinterpreting. No
#: genuine schema-1 or schema-2 monetary artifact was ever completed — the
#: first genuine run refused before writing one — but invented artifacts of
#: both do exist in test evidence, and the amount universe differs, so an old
#: payload gets its own refusal rather than falling through the generic magic
#: check as an unrecognised blob.
SUPERSEDED_MAGIC = (
    b"microcosm.us.puf_2015_monetary_source_projection/1\n",
    b"microcosm.us.puf_2015_monetary_source_projection/2\n",
)

MONETARY_PROJECTION_STAGE = "us_puf_monetary_source"
MONETARY_PROJECTION_NODE = f"{MONETARY_PROJECTION_STAGE}.projection"
MONETARY_PROJECTION_SUMMARY_NODE = f"{MONETARY_PROJECTION_STAGE}.projection_summary"

#: Bound on the encoded artifact's column bodies, matching the raw slice's own
#: private bound. Twelve projected columns over the delivered 207,696 rows is
#: 262 bytes a row, so about 51.9 MiB; a thirteenth column of the same shape
#: would still fit and a much wider selection would not.
BODY_MAX_BYTES = 64 * 1024 * 1024

#: Bound on the canonical JSON header. The header carries per-column facts for
#: three row classes, so it is larger than the status artifact's.
HEADER_MAX_BYTES = 64 * 1024

#: The widest lexical allocation a projected column may declare.
LEXICAL_WIDTH_MAX = 64

_PROJECTION_RESOURCE = "puf_2015_monetary_source_projection.json"
_PROJECTION_PACKAGE = "microcosm.build.us_runtime"
_PROJECTION_SCHEMA = "microcosm.us.puf_2015_monetary_source_projection"
_FIXTURE_AUTHORITY = "invented_fixture_nonauthority"

_STRUCTURAL_COLUMNS = ("RECID", "disclosure_aggregate", "demographic_status")
_STRUCTURAL_DTYPES = {
    "RECID": "<i8",
    "disclosure_aggregate": "<u1",
    "demographic_status": "<i1",
}
_AMOUNT_DTYPE = "<i8"
_DECLARED_DTYPE = MappingProxyType({"int64": "<i8"})

_TYPED_KIND = "typed"
_LEXICAL_KIND = "lexical_fixed_ascii_nul_padded"

#: The three disjoint row classes every derived fact is reported for. An
#: aggregate row is an aggregate first, whether or not a demographic row was
#: keyed to it, so the classes partition the rows exactly once.
ROW_CLASSES = ("disclosure_aggregate", "demographic_matched", "demographic_unmatched")

#: The aggregate class, and the two classes numeric per-return statistics are
#: reported for. Nothing numeric is ever reported for the aggregate class.
AGGREGATE_ROW_CLASS = "disclosure_aggregate"
RETURN_ROW_CLASSES = ("demographic_matched", "demographic_unmatched")

#: The closed vocabulary the packaged document may use.
_ADMISSIONS = frozenset({"monetary_source_projection", "not_selected"})
_FIT_ADMISSIONS = frozenset(
    {"not_admitted_pending_basis_review", "unresolved_retained"}
)
_NEGATIVE_STATES = frozenset(
    {
        "documented_by_record_layout_marker",
        "documented_by_publisher_definition_text",
        "not_documented_unresolved",
    }
)
_TARGET_YEAR_USE = "refused_pending_independent_growth_decision"
_PERIOD_SEMANTICS = "unresolved_raw_FLPDYR"
_GRAIN = "source_return"
_AMOUNT_UNITS = "whole_usd_source_year"
_GRAMMAR = "ascii_decimal_signed_integer"

#: The individual-return amount universe rule, carried in the header facts as
#: a declared string rather than left as a code habit.
AMOUNT_KNOWN_RULE = "amount_known == (disclosure_aggregate == 0)"

#: The out-of-universe storage sentinel for a typed amount slot on a
#: disclosure-aggregate row. It is storage, not a value: minimum int64 is
#: unreachable for a token of at most twelve digits, so no individual return
#: can legitimately hold it and readback can assert it exactly. A physical
#: zero is refused here, because zero is a delivered amount in these columns
#: and would invent a known value with semantics.
AGGREGATE_SENTINEL = -(2**63)

#: The whole-dollar integer semantics the packaged booklet-derived projection
#: document states apply to individual returns. The document's bytes are
#: unchanged; this encoded metadata scopes them.
_AMOUNT_UNITS_SCOPE = "individual_return_rows_only"
_AGGREGATE_UNIVERSE = "outside_individual_return_amount_universe"
_AGGREGATE_TREATMENT = "retained_lexical_only_never_parsed_to_a_number"
_AGGREGATE_AUTHORITY = "delivered_token_exact_for_every_row"
_AGGREGATE_INTERPRETATION = "refused"

#: The aggregate lexical grammar is an explicit operational input bound for
#: this pinned delivered file, not a newly discovered IRS specification.
_AGGREGATE_GRAMMAR = "ascii_decimal_optional_single_fraction_bounded"
_AGGREGATE_GRAMMAR_AUTHORITY = (
    "operational_input_bound_for_pinned_delivery_not_publisher_specification"
)

#: Keys a projection column may never carry: this increment computes no
#: derived quantity at all, so a document that declares one is a refusal
#: rather than a silently ignored field.
_FORBIDDEN_COLUMN_KEYS = frozenset({"derived_from", "derived_inputs", "formula"})

_ENVELOPE_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "type",
        "rows",
        "projection_sha256",
        "projection_route",
        "raw_source_definition_sha256",
        "status_artifact_sha256",
        "sources",
        "header_length_endianness",
        "body_endianness",
        "columns",
        "facts",
    }
)
_ENVELOPE_TYPED_COLUMN_KEYS = frozenset({"name", "kind", "dtype", "bytes", "sha256"})
_ENVELOPE_LEXICAL_COLUMN_KEYS = frozenset({"name", "kind", "width", "bytes", "sha256"})
_ENVELOPE_SOURCE_KEYS = frozenset(
    {"sha256", "git_blob_sha1", "bytes", "header_record_canonical_sha256"}
)
_ENVELOPE_FACT_KEYS = frozenset(
    {
        "rows",
        "main_records",
        "projected_columns",
        "layout_columns_authenticated",
        "demographic_layout_columns",
        "aggregate_records",
        "demographic_matched_rows",
        "demographic_unmatched_rows",
        "row_class_rows",
        "column_facts",
        "column_metadata",
        "grain",
        "period_semantics",
        "amounts_decoded",
        "amount_units",
        "amount_units_scope",
        "amount_known_rule",
        "amount_known_rows",
        "aggregate_amount_universe",
        "aggregate_row_treatment",
        "aggregate_typed_sentinel",
        "aggregate_token_grammar",
        "aggregate_lexical_authority",
        "aggregate_numeric_interpretation",
        "target_year_use",
    }
)
#: The per-return classes keep the existing numeric census.
_RETURN_COLUMN_FACT_KEYS = frozenset(
    {"rows", "zero", "negative", "positive", "minimum", "maximum", "sum"}
)
#: The aggregate class has its own closed key set with no zero, sign, extremum
#: or sum key, so a header cannot spell a number for the aggregate rows at
#: all: only token classes and a token width.
_AGGREGATE_COLUMN_FACT_KEYS = frozenset(
    {
        "rows",
        "integer_tokens",
        "fractional_tokens",
        "negative_tokens",
        "max_token_width",
    }
)

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)
_DIGITS = frozenset("0123456789")
#: The publisher's amount fields are twelve digits wide.
_MAX_AMOUNT_DIGITS = 12


class PufMonetaryRefusalError(ValueError):
    """A refusal. The reason is a code and aggregates, never a source token."""

    def __init__(self, reason: str, *detail: object) -> None:
        super().__init__(" ".join((reason, *(str(part) for part in detail))).strip())
        self.reason = reason


def _refuse(reason: str, *detail: object) -> PufMonetaryRefusalError:
    return PufMonetaryRefusalError(reason, *detail)


# --------------------------------------------------------------------------
# The closed projection document
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectedColumn:
    """One enumerated monetary column, with everything the parse needs."""

    field: str
    concept: str
    concept_group: str
    delivered_index: int
    publisher_position: int
    publisher_label: str
    dtype: str
    lexical_width: int
    grammar: str
    max_digits: int
    publisher_signed_marker: bool
    negative_values: str
    fit_admission: str
    unresolved: tuple[str, ...]

    @property
    def numpy_dtype(self) -> str:
        return _DECLARED_DTYPE[self.dtype]


@dataclass(frozen=True)
class PufMonetaryProjection:
    """A closed layout authentication plus the enumerated projection."""

    route: str
    document: Mapping[str, object]
    canonical: bytes
    sha256: str
    columns: tuple[ProjectedColumn, ...]
    raw_definition_sha256: str
    main_delivered_header: tuple[str, ...]
    demographic_delivered_header: tuple[str, ...]

    @property
    def params_text(self) -> str:
        """The canonical JSON the node carries verbatim as a parameter."""

        return self.canonical.decode("utf-8")

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(column.field for column in self.columns)


#: Kept as a separate public name so a caller can talk about the document
#: without holding a parsed projection.
PufMonetarySourceDocument = Mapping[str, object]


def _frozen(value: object) -> object:
    """A deeply immutable view, as the raw source definition already does."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _thawed(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value


def _hex64(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _refuse("PROJECTION_DIGEST", label)
    return value


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _refuse("PROJECTION_INTEGER", label)
    return value


def _layout_names(
    entry: Mapping[str, object], label: str, reasons: frozenset[str]
) -> tuple[str, ...]:
    columns = entry["columns"]
    if not isinstance(columns, (list, tuple)) or not columns:
        raise _refuse("PROJECTION_LAYOUT", label)
    names: list[str] = []
    for position, column in enumerate(columns):
        if not isinstance(column, Mapping):
            raise _refuse("PROJECTION_LAYOUT", label)
        name = column["name"]
        if not isinstance(name, str) or not name:
            raise _refuse("PROJECTION_LAYOUT_NAME", label)
        if column["delivered_index"] != position:
            raise _refuse("PROJECTION_LAYOUT_INDEX", name)
        admission = column["admission"]
        if admission not in _ADMISSIONS:
            raise _refuse("PROJECTION_LAYOUT_ADMISSION", name)
        if admission == "not_selected":
            # An unselected column carries a reason from the document's own
            # closed list, so nothing is dropped by omission.
            if column.get("not_selected_reason") not in reasons:
                raise _refuse("PROJECTION_LAYOUT_NOT_SELECTED_REASON", name)
        elif "not_selected_reason" in column:
            raise _refuse("PROJECTION_LAYOUT_NOT_SELECTED_REASON", name)
        row = column["layout_row"]
        if not isinstance(row, Mapping):
            raise _refuse("PROJECTION_LAYOUT_EVIDENCE", name)
        text = row["text"]
        if not isinstance(text, str) or not text:
            raise _refuse("PROJECTION_LAYOUT_EVIDENCE", name)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["text_sha256"]:
            raise _refuse("PROJECTION_LAYOUT_EVIDENCE_DIGEST", name)
        if name not in text:
            raise _refuse("PROJECTION_LAYOUT_EVIDENCE_NAME", name)
        names.append(name)
    if entry["delivered_header_width"] != len(names):
        raise _refuse("PROJECTION_LAYOUT_WIDTH", label)
    if len(set(names)) != len(names):
        raise _refuse("PROJECTION_LAYOUT_DUPLICATE", label)
    return tuple(names)


def _projected_column(
    entry: Mapping[str, object], layout: Mapping[str, Mapping[str, object]]
) -> ProjectedColumn:
    field = entry["field"]
    if not isinstance(field, str) or field not in layout:
        raise _refuse("PROJECTION_UNDELIVERED_FIELD", field)
    for key in _FORBIDDEN_COLUMN_KEYS:
        if key in entry:
            raise _refuse("PROJECTION_DERIVATION_NOT_SUPPORTED", field)
    column = layout[field]
    if column["admission"] != "monetary_source_projection":
        raise _refuse("PROJECTION_ADMISSION_DISAGREEMENT", field)
    if entry["delivered_index"] != column["delivered_index"]:
        raise _refuse("PROJECTION_INDEX_DISAGREEMENT", field)
    if entry["publisher_position"] != column["publisher_position"]:
        raise _refuse("PROJECTION_POSITION_DISAGREEMENT", field)
    if entry["publisher_label"] != column["publisher_label"]:
        raise _refuse("PROJECTION_LABEL_DISAGREEMENT", field)
    if entry["dtype"] not in _DECLARED_DTYPE:
        raise _refuse("PROJECTION_DTYPE", field)
    if entry["grammar"] != _GRAMMAR:
        raise _refuse("PROJECTION_GRAMMAR", field)
    width = _positive_int(entry["lexical_width"], f"{field}.lexical_width")
    if width > LEXICAL_WIDTH_MAX:
        raise _refuse("PROJECTION_LEXICAL_WIDTH", field)
    units = entry["units"]
    digits = _positive_int(units["field_width_digits"], f"{field}.field_width_digits")
    if digits > _MAX_AMOUNT_DIGITS or width < digits + 1:
        raise _refuse("PROJECTION_WIDTH_BELOW_GRAMMAR", field)
    if units["currency"] != "USD" or units["scale"] != "whole_dollars":
        raise _refuse("PROJECTION_UNITS", field)
    if units["implied_decimals"] != 0:
        raise _refuse("PROJECTION_IMPLIED_DECIMALS", field)
    sign = entry["sign"]
    marker = sign["publisher_signed_marker"]
    if not isinstance(marker, bool) or marker != column["publisher_signed_marker"]:
        raise _refuse("PROJECTION_SIGN_MARKER", field)
    negative = sign["negative_values"]
    if negative not in _NEGATIVE_STATES:
        raise _refuse("PROJECTION_SIGN_STATE", field)
    if marker and negative != "documented_by_record_layout_marker":
        raise _refuse("PROJECTION_SIGN_STATE", field)
    if sign["negative_zero"] != "refused":
        raise _refuse("PROJECTION_NEGATIVE_ZERO_POLICY", field)
    missing = entry["missingness"]
    if missing["empty_token"] != "refused":
        raise _refuse("PROJECTION_EMPTY_TOKEN_POLICY", field)
    if missing["zero_meaning"] != "delivered_zero_in_edited_public_file":
        raise _refuse("PROJECTION_ZERO_POLICY", field)
    if entry["return_grain"]["grain"] != _GRAIN:
        raise _refuse("PROJECTION_GRAIN", field)
    if entry["source_period_status"]["status"] != _PERIOD_SEMANTICS:
        raise _refuse("PROJECTION_PERIOD_STATUS", field)
    if entry["target_year_use"] != _TARGET_YEAR_USE:
        raise _refuse("PROJECTION_TARGET_YEAR_USE", field)
    if entry["derived_combinations"] != "refused":
        raise _refuse("PROJECTION_DERIVATION_NOT_SUPPORTED", field)
    admission = entry["fit_admission"]
    if admission not in _FIT_ADMISSIONS:
        raise _refuse("PROJECTION_FIT_ADMISSION", field)
    unresolved = tuple(item["code"] for item in entry["unresolved"])
    return ProjectedColumn(
        field=field,
        concept=entry["concept"],
        concept_group=entry["concept_group"],
        delivered_index=entry["delivered_index"],
        publisher_position=entry["publisher_position"],
        publisher_label=entry["publisher_label"],
        dtype=entry["dtype"],
        lexical_width=width,
        grammar=entry["grammar"],
        max_digits=digits,
        publisher_signed_marker=marker,
        negative_values=negative,
        fit_admission=admission,
        unresolved=unresolved,
    )


def _projection_from_document(
    document: Mapping[str, object],
    *,
    route: str,
    definition: PufRawSourceDefinition,
) -> PufMonetaryProjection:
    """Validate a projection document against a raw source definition."""

    if not isinstance(definition, PufRawSourceDefinition):
        raise _refuse("PROJECTION_DEFINITION_TYPE")
    if document.get("schema") != _PROJECTION_SCHEMA:
        raise _refuse("PROJECTION_SCHEMA")
    if document.get("schema_version") != 1:
        raise _refuse("PROJECTION_SCHEMA_VERSION")
    if document.get("route") != route:
        raise _refuse("PROJECTION_ROUTE")

    raw = document["raw_source_definition"]
    raw_sha = _hex64(raw["sha256"], "raw_source_definition.sha256")
    if raw_sha != definition.sha256:
        raise _refuse("PROJECTION_RAW_DEFINITION_MISMATCH")

    layout = document["layout"]
    reasons = layout["not_selected_reasons"]
    if not isinstance(reasons, Mapping) or not reasons:
        raise _refuse("PROJECTION_NOT_SELECTED_REASONS")
    allowed = frozenset(reasons)
    main_names = _layout_names(layout["main"], "main", allowed)
    demographic_names = _layout_names(layout["demographic"], "demographic", allowed)
    if main_names != definition.main.delivered_header:
        raise _refuse("PROJECTION_MAIN_LAYOUT_HEADER")
    if demographic_names != definition.demographic.delivered_header:
        raise _refuse("PROJECTION_DEMOGRAPHIC_LAYOUT_HEADER")

    for name, entry in (
        ("main", definition.main),
        ("demographic", definition.demographic),
    ):
        pin = document["sources"][name]
        if (
            pin["sha256"] != entry.sha256
            or pin["git_blob_sha1"] != entry.git_blob_sha1
            or pin["bytes"] != entry.bytes
            or pin["header_record_canonical_sha256"]
            != entry.header_record_canonical_sha256
            or pin["delivered_header_width"] != len(entry.delivered_header)
            or pin["data_records"] != entry.data_records
        ):
            raise _refuse("PROJECTION_SOURCE_PIN", name)

    by_name = {column["name"]: column for column in layout["main"]["columns"]}
    for refused in document["refused_fields"]:
        if refused in main_names or refused in demographic_names:
            raise _refuse("PROJECTION_REFUSED_FIELD_DELIVERED", refused)

    projection = document["projection"]
    if projection["grain"] != _GRAIN:
        raise _refuse("PROJECTION_GRAIN")
    if projection["period_semantics"] != _PERIOD_SEMANTICS:
        raise _refuse("PROJECTION_PERIOD_STATUS")
    if projection["target_year_use"] != _TARGET_YEAR_USE:
        raise _refuse("PROJECTION_TARGET_YEAR_USE")
    if projection["amounts_decoded"] is not True:
        raise _refuse("PROJECTION_AMOUNTS_DECODED")

    entries = projection["columns"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise _refuse("PROJECTION_EMPTY")
    columns = tuple(_projected_column(entry, by_name) for entry in entries)
    fields = [column.field for column in columns]
    if len(set(fields)) != len(fields):
        raise _refuse("PROJECTION_DUPLICATE_FIELD")
    for field in fields:
        if field in document["refused_fields"]:
            raise _refuse("PROJECTION_REFUSED_FIELD", field)
    admitted = {
        name
        for name, column in by_name.items()
        if column["admission"] == "monetary_source_projection"
    }
    if admitted != set(fields):
        raise _refuse("PROJECTION_ADMISSION_SET")
    if projection["column_count"] != len(columns):
        raise _refuse("PROJECTION_COLUMN_COUNT")
    if sorted(fields, key=lambda name: by_name[name]["delivered_index"]) != fields:
        raise _refuse("PROJECTION_COLUMN_ORDER")

    if any(value is not False for value in document["non_claims"].values()):
        raise _refuse("PROJECTION_NON_CLAIMS")

    canonical = canonical_json(_thawed(document))
    return PufMonetaryProjection(
        route=route,
        document=_frozen(json.loads(canonical.decode("utf-8"))),
        canonical=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
        columns=columns,
        raw_definition_sha256=raw_sha,
        main_delivered_header=main_names,
        demographic_delivered_header=demographic_names,
    )


def _packaged_bytes() -> bytes:
    return (
        resources.files(_PROJECTION_PACKAGE).joinpath(_PROJECTION_RESOURCE).read_bytes()
    )


_PACKAGED: PufMonetaryProjection | None = None


def packaged_projection() -> PufMonetaryProjection:
    """The one closed projection document a production run may use."""

    global _PACKAGED
    if _PACKAGED is None:
        document = json.loads(_packaged_bytes().decode("utf-8"))
        _PACKAGED = _projection_from_document(
            document, route="packaged", definition=packaged_raw_definition()
        )
    return _PACKAGED


def projection_document_json(
    projection: PufMonetaryProjection | None = None,
) -> dict[str, object]:
    """A fresh, plain, mutable JSON copy of a projection document."""

    resolved = packaged_projection() if projection is None else projection
    return json.loads(resolved.canonical.decode("utf-8"))


def fixture_projection_document(
    document: Mapping[str, object], definition: PufRawSourceDefinition
) -> PufMonetaryProjection:
    """Build a projection on the explicit non-production test route.

    A fixture projection must declare ``route="test_fixture"`` and the
    invented-fixture authority, and it may not restate either packaged source
    pin. It still has to authenticate the whole delivered layout, because a
    fixture that dropped the layout would be testing a different contract.
    """

    projection = _projection_from_document(
        document, route="test_fixture", definition=definition
    )
    if document.get("authority") != _FIXTURE_AUTHORITY:
        raise _refuse("FIXTURE_AUTHORITY")
    packaged_raw = packaged_raw_definition()
    genuine = {packaged_raw.main.sha256, packaged_raw.demographic.sha256}
    stated = {
        document["sources"]["main"]["sha256"],
        document["sources"]["demographic"]["sha256"],
    }
    if stated & genuine:
        raise _refuse("FIXTURE_RESTATES_PACKAGED_PIN")
    return projection


# --------------------------------------------------------------------------
# The structural decoder
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PufMonetarySourceProjection:
    """Decoded source values with explicit, immutable fitting refusals."""

    typed: Mapping[str, np.ndarray]
    lexical: Mapping[str, tuple[str, ...]]
    facts: Mapping[str, object]
    column_metadata: Mapping[str, object]

    @property
    def rows(self) -> int:
        return int(self.facts["rows"])

    @property
    def amount_known(self) -> np.ndarray:
        """Whether a row is inside the individual-return amount universe.

        Derived from the authenticated disclosure flag on every access and
        handed back read-only, so there is no stored mask to tamper with and
        nothing a caller holds can contradict the flags. A false entry means
        *outside the universe*: the typed slot holds the sentinel and the
        delivered token is the authority.
        """

        return _amount_known(self.typed["disclosure_aggregate"])

    def aggregate_lexical_rows(
        self, field: str | None = None
    ) -> Mapping[str, tuple[tuple[int, str], ...]]:
        """Exact ``(RECID, token)`` pairs for the disclosure-aggregate rows.

        A local accessor, for diagnosis at the console or in a test. Public
        summaries, receipts and reports stay aggregate-only: nothing in the
        kernels calls this. No token is parsed, totalled, ordered by value or
        compared with another token.
        """

        if field is None:
            fields = tuple(self.lexical)
        elif field in self.lexical:
            fields = (field,)
        else:
            raise _refuse("AGGREGATE_LEXICAL_FIELD", field)
        aggregate = np.flatnonzero(~self.amount_known).tolist()
        recids = self.typed["RECID"]
        return MappingProxyType(
            {
                name: tuple(
                    (int(recids[position]), self.lexical[name][position])
                    for position in aggregate
                )
                for name in fields
            }
        )


def _check_csv_profile(profile: Mapping[str, object]) -> None:
    """Refuse when the effective acceptance state is not the pinned one."""

    declared = profile["field_size_limit"]
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        raise _refuse("DEFINITION_CSV_FIELD_SIZE_LIMIT")
    if csv.field_size_limit() != declared:
        raise _refuse("CSV_FIELD_SIZE_LIMIT")
    if profile["quoting"] != "QUOTE_MINIMAL" or profile["strict"] is not True:
        raise _refuse("DEFINITION_CSV_PROFILE")
    if profile["encoding"] != "utf-8-strict" or profile["newline"] != "":
        raise _refuse("DEFINITION_CSV_PROFILE")


def _reader(payload: bytes, profile: Mapping[str, object]):
    if not isinstance(payload, bytes):
        raise _refuse("DECODE_PAYLOAD_TYPE")
    stream = io.TextIOWrapper(
        io.BytesIO(payload), encoding="utf-8", errors="strict", newline=""
    )
    return csv.reader(
        stream,
        delimiter=profile["delimiter"],
        quotechar=profile["quotechar"],
        doublequote=bool(profile["doublequote"]),
        quoting=csv.QUOTE_MINIMAL,
        strict=True,
    )


def _canonical_header_digest(header: Sequence[str]) -> str:
    payload = json.dumps(
        list(header),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _key_token(value: str, *, width: int) -> str:
    """The unsigned decimal ``RECID`` token, as the raw slice reads it."""

    if not isinstance(value, str):
        raise _refuse("KEY_TOKEN_TYPE")
    if not value.isascii():
        raise _refuse("KEY_TOKEN_NON_ASCII")
    if "\x00" in value:
        raise _refuse("KEY_TOKEN_EMBEDDED_NUL")
    if not value:
        raise _refuse("KEY_TOKEN_EMPTY")
    if len(value) > width:
        raise _refuse("KEY_TOKEN_OVER_WIDTH")
    if any(character not in _DIGITS for character in value):
        raise _refuse("KEY_TOKEN_GRAMMAR")
    number = int(value)
    if number > _INT64_MAX:
        raise _refuse("KEY_TOKEN_INT64_OVERFLOW")
    return value


def _amount(value: str, column: ProjectedColumn) -> int:
    """Parse one delivered amount token, or refuse.

    The grammar is exactly what the retained booklets support: an optional
    leading minus, then decimal digits, no wider than the publisher's declared
    amount-field width. There is no missing-value sentinel anywhere in either
    booklet, so an empty token is a refusal to be preserved and reviewed
    rather than a zero to be invented, and a plus sign, a decimal point,
    surrounding whitespace or an exponent is a refusal rather than a silently
    widened grammar. ``-0`` is refused outright: a signed zero would compare
    equal to zero while spelling differently.
    """

    label = column.field
    if not isinstance(value, str):
        raise _refuse("AMOUNT_TOKEN_TYPE", label)
    if not value.isascii():
        raise _refuse("AMOUNT_TOKEN_NON_ASCII", label)
    if "\x00" in value:
        raise _refuse("AMOUNT_TOKEN_EMBEDDED_NUL", label)
    if not value:
        raise _refuse("AMOUNT_TOKEN_EMPTY", label)
    if len(value) > column.lexical_width:
        raise _refuse("AMOUNT_TOKEN_OVER_WIDTH", label)
    digits = value[1:] if value.startswith("-") else value
    if not digits:
        raise _refuse("AMOUNT_TOKEN_GRAMMAR", label)
    if any(character not in _DIGITS for character in digits):
        raise _refuse("AMOUNT_TOKEN_GRAMMAR", label)
    if len(digits) > column.max_digits:
        raise _refuse("AMOUNT_TOKEN_OVER_DIGITS", label)
    number = -int(digits) if value.startswith("-") else int(digits)
    if number == 0 and value.startswith("-"):
        raise _refuse("AMOUNT_TOKEN_NEGATIVE_ZERO", label)
    if not _INT64_MIN <= number <= _INT64_MAX:
        raise _refuse("AMOUNT_TOKEN_INT64_OVERFLOW", label)
    return number


def _aggregate_token(value: str, column: ProjectedColumn) -> str:
    """Validate one disclosure-aggregate token, and return it unchanged.

    This is a **separate, explicitly bounded operational grammar** for the
    delivered aggregate rows of this pinned file. It is not a newly discovered
    publisher specification, and it is never applied to an individual return.
    Optional leading minus, one or more digits, at most one point followed by
    one or more digits: no plus sign, exponent, whitespace, embedded NUL,
    empty token or negative zero in either spelling, at most the column's
    declared digit bound and at most its declared width.

    The token is returned exactly as delivered. Nothing here converts it to a
    number, rounds it, or compares it against another token.
    """

    label = column.field
    if not isinstance(value, str):
        raise _refuse("AGGREGATE_TOKEN_TYPE", label)
    if not value.isascii():
        raise _refuse("AGGREGATE_TOKEN_NON_ASCII", label)
    if "\x00" in value:
        raise _refuse("AGGREGATE_TOKEN_EMBEDDED_NUL", label)
    if not value:
        raise _refuse("AGGREGATE_TOKEN_EMPTY", label)
    if len(value) > column.lexical_width:
        raise _refuse("AGGREGATE_TOKEN_OVER_WIDTH", label)
    negative = value.startswith("-")
    body = value[1:] if negative else value
    if not body:
        raise _refuse("AGGREGATE_TOKEN_GRAMMAR", label)
    if body.count(".") > 1:
        raise _refuse("AGGREGATE_TOKEN_GRAMMAR", label)
    whole, point, fraction = body.partition(".")
    if point and (not whole or not fraction):
        raise _refuse("AGGREGATE_TOKEN_GRAMMAR", label)
    digits = whole + fraction
    if not digits or any(character not in _DIGITS for character in digits):
        raise _refuse("AGGREGATE_TOKEN_GRAMMAR", label)
    if len(digits) > column.max_digits:
        raise _refuse("AGGREGATE_TOKEN_OVER_DIGITS", label)
    if negative and set(digits) == {"0"}:
        raise _refuse("AGGREGATE_TOKEN_NEGATIVE_ZERO", label)
    return value


def _amount_known(flags: np.ndarray) -> np.ndarray:
    """The individual-return amount universe, derived from the flag alone.

    Returned read-only and recomputed at every call, so no caller can hold a
    mask that contradicts the disclosure flags it came from.
    """

    known = np.asarray(flags, dtype=_STRUCTURAL_DTYPES["disclosure_aggregate"]) == 0
    # A read-only flag on an owning ndarray can be reversed with setflags.
    # An immutable bytes backing store makes that reversal impossible.
    return np.frombuffer(known.tobytes(), dtype=np.bool_)


def _typed_column(
    tokens: Sequence[str], known: np.ndarray, column: ProjectedColumn
) -> np.ndarray:
    """Parse by universe: an amount where known, the sentinel where not."""

    if len(tokens) != int(known.shape[0]):
        raise _refuse("COLUMN_LEXICAL_ROWS", column.field)
    values: list[int] = []
    for position, token in enumerate(tokens):
        if bool(known[position]):
            values.append(_amount(token, column))
        else:
            _aggregate_token(token, column)
            values.append(AGGREGATE_SENTINEL)
    return np.asarray(values, dtype=column.numpy_dtype)


def _check_header(
    reader, projection: PufMonetaryProjection, definition: PufRawSourceDefinition
) -> list[str]:
    pin = definition.main
    profile = definition.document["csv_profile"]
    try:
        header = next(reader)
    except StopIteration as error:
        raise _refuse("HEADER_MISSING") from error
    except (csv.Error, UnicodeError) as error:
        raise _refuse("HEADER_UNREADABLE") from error
    if len(header) > profile["header_field_cap"]:
        raise _refuse("HEADER_WIDTH_LIMIT")
    if tuple(header) != pin.delivered_header:
        raise _refuse("HEADER_MISMATCH")
    if tuple(header) != projection.main_delivered_header:
        raise _refuse("HEADER_NOT_AUTHENTICATED_LAYOUT")
    if _canonical_header_digest(header) != pin.header_record_canonical_sha256:
        raise _refuse("HEADER_DIGEST")
    refused = set(projection.document["refused_fields"]) & set(header)
    if refused:
        raise _refuse("HEADER_REFUSED_FIELD", sorted(refused))
    return header


def _records(reader, definition: PufRawSourceDefinition):
    profile = definition.document["csv_profile"]
    width = len(definition.main.delivered_header)
    cap = profile["logical_record_character_cap"]
    limit = profile["record_cap_per_file"]
    count = 0
    while True:
        try:
            record = next(reader)
        except StopIteration:
            return
        except (csv.Error, UnicodeError) as error:
            raise _refuse("RECORD_UNREADABLE") from error
        count += 1
        if count > limit:
            raise _refuse("RECORD_CAP")
        if len(record) != width:
            raise _refuse("RECORD_WIDTH")
        if sum(len(cell) for cell in record) > cap:
            raise _refuse("RECORD_CHARACTER_CAP")
        yield record


def decode_puf_monetary_source(
    main_bytes: bytes,
    status: object,
    projection: PufMonetaryProjection,
    definition: PufRawSourceDefinition,
) -> PufMonetarySourceProjection:
    """Decode the enumerated monetary columns, bound to the status artifact.

    ``status`` is the decoded ``puf_2015_raw_return_status``. Its ``RECID``
    sequence and aggregate flags must be element-wise identical to this
    parse's, and its ``demographic_status`` is carried through rather than
    re-joined, so the two artifacts cannot disagree about which returns
    matched. Every delivered main record is projected: nothing is filtered,
    and the four disclosure aggregates stay tagged rather than dropped.

    Amount tokens are collected before that alignment runs and parsed after
    it, by universe: the unchanged integer grammar on individual returns, the
    bounded aggregate lexical grammar plus the out-of-universe sentinel on
    disclosure aggregates. A parse therefore cannot decide which grammar a row
    answers to before the status artifact has proved which row it is.
    """

    if not isinstance(projection, PufMonetaryProjection):
        raise _refuse("DECODE_PROJECTION_TYPE")
    if projection.raw_definition_sha256 != definition.sha256:
        raise _refuse("DECODE_DEFINITION_MISMATCH")
    profile = definition.document["csv_profile"]
    _check_csv_profile(profile)
    key_width = next(
        field["lexical_width"]
        for field in definition.document["fields"]
        if field["name"] == "RECID"
    )

    reader = _reader(main_bytes, profile)
    header = _check_header(reader, projection, definition)
    positions = {
        column.field: header.index(column.field) for column in projection.columns
    }
    for column in projection.columns:
        if positions[column.field] != column.delivered_index:
            raise _refuse("DELIVERED_INDEX_DISAGREEMENT", column.field)
    recid_position = header.index("RECID")

    keys: list[int] = []
    seen: set[str] = set()
    aggregates = frozenset(PUF_AGGREGATE_RECIDS)
    flags: list[int] = []
    tokens: dict[str, list[str]] = {column.field: [] for column in projection.columns}

    # Tokens are collected first and parsed afterwards. Which grammar a row's
    # amount answers to is decided only once the status artifact has proved
    # the RECID sequence and the aggregate flags, so "authenticate the
    # alignment before interpreting a row's amount universe" is the literal
    # order of operations rather than a claim about one.
    for record in _records(reader, definition):
        key = _key_token(record[recid_position], width=key_width)
        if key in seen:
            raise _refuse("MAIN_DUPLICATE_LEXICAL_KEY")
        seen.add(key)
        recid = int(key)
        keys.append(recid)
        flags.append(1 if recid in aggregates else 0)
        for column in projection.columns:
            tokens[column.field].append(record[positions[column.field]])
    rows = len(keys)
    if rows == 0:
        raise _refuse("MAIN_NO_RECORDS")
    if rows != definition.main.data_records:
        raise _refuse("MAIN_RECORD_COUNT")

    typed: dict[str, np.ndarray] = {
        "RECID": np.asarray(keys, dtype=_STRUCTURAL_DTYPES["RECID"]),
        "disclosure_aggregate": np.asarray(
            flags, dtype=_STRUCTURAL_DTYPES["disclosure_aggregate"]
        ),
    }
    if len(np.unique(typed["RECID"])) != rows:
        raise _refuse("MAIN_NUMERIC_KEY_COLLISION")
    typed["demographic_status"] = _status_alignment(status, typed, rows)
    known = _amount_known(typed["disclosure_aggregate"])
    for column in projection.columns:
        typed[column.field] = _typed_column(tokens[column.field], known, column)

    lexical = {name: tuple(items) for name, items in tokens.items()}
    facts = _facts_from_arrays(typed, lexical, projection)
    # Match envelope readback's immutable byte-backed arrays. In particular,
    # disclosure flags cannot change after an amount-known mask is returned.
    typed = {
        name: np.frombuffer(array.tobytes(), dtype=array.dtype)
        for name, array in typed.items()
    }
    return PufMonetarySourceProjection(
        typed=MappingProxyType(typed),
        lexical=MappingProxyType(lexical),
        facts=MappingProxyType(facts),
        column_metadata=_frozen(facts["column_metadata"]),
    )


def _status_alignment(
    status: object, typed: Mapping[str, np.ndarray], rows: int
) -> np.ndarray:
    """Bind this parse to the reviewed status artifact, element by element."""

    try:
        status_typed = status.typed
    except AttributeError as error:
        raise _refuse("STATUS_ARTIFACT_TYPE") from error
    for name in ("RECID", "disclosure_aggregate", "demographic_status"):
        if name not in status_typed:
            raise _refuse("STATUS_ARTIFACT_COLUMN", name)
    if int(status_typed["RECID"].shape[0]) != rows:
        raise _refuse("STATUS_ROW_COUNT")
    if not np.array_equal(
        np.asarray(status_typed["RECID"], dtype="<i8"), typed["RECID"]
    ):
        raise _refuse("STATUS_RECID_SEQUENCE")
    if not np.array_equal(
        np.asarray(status_typed["disclosure_aggregate"], dtype="<u1"),
        typed["disclosure_aggregate"],
    ):
        raise _refuse("STATUS_AGGREGATE_FLAGS")
    carried = np.asarray(
        status_typed["demographic_status"],
        dtype=_STRUCTURAL_DTYPES["demographic_status"],
    )
    if not np.isin(carried, (0, 1)).all():
        raise _refuse("STATUS_DEMOGRAPHIC_STATE")
    return carried


# --------------------------------------------------------------------------
# Facts, derived from the arrays and only from the arrays
# --------------------------------------------------------------------------


def _row_masks(typed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The three disjoint row classes, which must partition the rows exactly."""

    aggregate = typed["disclosure_aggregate"] == 1
    status = typed["demographic_status"] == 1
    masks = {
        "disclosure_aggregate": aggregate,
        "demographic_matched": (~aggregate) & status,
        "demographic_unmatched": (~aggregate) & (~status),
    }
    total = sum(int(np.count_nonzero(mask)) for mask in masks.values())
    if total != int(typed["RECID"].shape[0]):
        raise _refuse("ROW_CLASS_PARTITION")
    return masks


def _column_facts(values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = values[mask]
    count = int(selected.shape[0])
    if count == 0:
        return {
            "rows": 0,
            "zero": 0,
            "negative": 0,
            "positive": 0,
            "minimum": None,
            "maximum": None,
            "sum": 0,
        }
    total = int(sum(int(item) for item in selected.tolist()))
    if not _INT64_MIN <= total <= _INT64_MAX:
        raise _refuse("COLUMN_SUM_OVERFLOW")
    return {
        "rows": count,
        "zero": int(np.count_nonzero(selected == 0)),
        "negative": int(np.count_nonzero(selected < 0)),
        "positive": int(np.count_nonzero(selected > 0)),
        "minimum": int(selected.min()),
        "maximum": int(selected.max()),
        "sum": total,
    }


def _aggregate_column_facts(
    tokens: Sequence[str], mask: np.ndarray
) -> dict[str, object]:
    """Token classes for the aggregate rows, and nothing that reads as money.

    Counts of token shapes and one token width. There is no zero, sign census,
    minimum, maximum or sum key here, because the delivered token is the
    authority for these rows and this module never interprets it. A diagnostic
    may count token classes; it may not state an aggregate amount.
    """

    selected = [
        token for token, taken in zip(tokens, mask.tolist(), strict=True) if taken
    ]
    if not selected:
        return {
            "rows": 0,
            "integer_tokens": 0,
            "fractional_tokens": 0,
            "negative_tokens": 0,
            "max_token_width": None,
        }
    return {
        "rows": len(selected),
        "integer_tokens": sum(1 for token in selected if "." not in token),
        "fractional_tokens": sum(1 for token in selected if "." in token),
        "negative_tokens": sum(1 for token in selected if token.startswith("-")),
        "max_token_width": max(len(token) for token in selected),
    }


def _facts_from_arrays(
    typed: Mapping[str, np.ndarray],
    lexical: Mapping[str, Sequence[str]],
    projection: PufMonetaryProjection,
) -> dict[str, object]:
    """Derive every reported fact from the arrays themselves.

    The producer builds its facts here and the envelope readback rebuilds them
    here, so a header can never report a count, a sum or a sign census the
    arrays do not carry. Layout counts and column semantics come from the
    authenticated projection document, rather than being inferred from rows.
    The lexical arrays are an input because the aggregate class is described
    from its delivered tokens and from nothing else.
    """

    rows = int(typed["RECID"].shape[0])
    masks = _row_masks(typed)
    known = _amount_known(typed["disclosure_aggregate"])
    for column in projection.columns:
        if len(lexical[column.field]) != rows:
            raise _refuse("FACTS_LEXICAL_ROWS", column.field)
    return {
        "rows": rows,
        "main_records": rows,
        "projected_columns": len(projection.columns),
        "layout_columns_authenticated": len(projection.main_delivered_header),
        "demographic_layout_columns": len(projection.demographic_delivered_header),
        "aggregate_records": int(np.count_nonzero(masks["disclosure_aggregate"])),
        "demographic_matched_rows": int(
            np.count_nonzero(typed["demographic_status"] == 1)
        ),
        "demographic_unmatched_rows": int(
            np.count_nonzero(typed["demographic_status"] == 0)
        ),
        "row_class_rows": {
            name: int(np.count_nonzero(masks[name])) for name in ROW_CLASSES
        },
        "column_facts": {
            column.field: {
                AGGREGATE_ROW_CLASS: _aggregate_column_facts(
                    lexical[column.field], masks[AGGREGATE_ROW_CLASS]
                ),
                **{
                    name: _column_facts(typed[column.field], masks[name])
                    for name in RETURN_ROW_CLASSES
                },
            }
            for column in projection.columns
        },
        "column_metadata": {
            column.field: {
                "concept": column.concept,
                "grain": _GRAIN,
                # The packaged projection document's integer, whole-dollar,
                # twelve-digit semantics describe individual returns. That
                # document's bytes are unchanged; this metadata says who they
                # apply to, and what happens to the rows they do not.
                "amount_units": _AMOUNT_UNITS,
                "amount_units_scope": _AMOUNT_UNITS_SCOPE,
                "amount_universe": _AMOUNT_UNITS_SCOPE,
                "amount_known_rule": AMOUNT_KNOWN_RULE,
                "aggregate_amount_universe": _AGGREGATE_UNIVERSE,
                "aggregate_row_treatment": _AGGREGATE_TREATMENT,
                "aggregate_typed_sentinel": AGGREGATE_SENTINEL,
                "aggregate_token_grammar": _AGGREGATE_GRAMMAR,
                "aggregate_token_max_digits": column.max_digits,
                "aggregate_token_max_width": column.lexical_width,
                "aggregate_token_grammar_authority": _AGGREGATE_GRAMMAR_AUTHORITY,
                "aggregate_lexical_authority": _AGGREGATE_AUTHORITY,
                "aggregate_numeric_interpretation": _AGGREGATE_INTERPRETATION,
                "period_semantics": _PERIOD_SEMANTICS,
                "fit_admission": column.fit_admission,
                "unresolved": list(column.unresolved),
                "negative_values": column.negative_values,
                "zero_meaning": "delivered_zero_in_edited_public_file",
                "target_year_use": _TARGET_YEAR_USE,
                "projection_sha256": projection.sha256,
            }
            for column in projection.columns
        },
        "grain": _GRAIN,
        "period_semantics": _PERIOD_SEMANTICS,
        "amounts_decoded": True,
        "amount_units": _AMOUNT_UNITS,
        "amount_units_scope": _AMOUNT_UNITS_SCOPE,
        "amount_known_rule": AMOUNT_KNOWN_RULE,
        "amount_known_rows": int(np.count_nonzero(known)),
        "aggregate_amount_universe": _AGGREGATE_UNIVERSE,
        "aggregate_row_treatment": _AGGREGATE_TREATMENT,
        "aggregate_typed_sentinel": AGGREGATE_SENTINEL,
        "aggregate_token_grammar": _AGGREGATE_GRAMMAR,
        "aggregate_lexical_authority": _AGGREGATE_AUTHORITY,
        "aggregate_numeric_interpretation": _AGGREGATE_INTERPRETATION,
        "target_year_use": _TARGET_YEAR_USE,
    }


# --------------------------------------------------------------------------
# The typed envelope
# --------------------------------------------------------------------------


def _fixed_ascii(values: Sequence[str], width: int, label: str) -> bytes:
    body = bytearray(len(values) * width)
    for position, value in enumerate(values):
        raw = value.encode("ascii")
        if len(raw) > width:
            raise _refuse("LEXICAL_OVER_WIDTH", label)
        if b"\x00" in raw:
            raise _refuse("LEXICAL_EMBEDDED_NUL", label)
        start = position * width
        body[start : start + len(raw)] = raw
    return bytes(body)


def _read_fixed_ascii(
    body: bytes, rows: int, width: int, label: str
) -> tuple[str, ...]:
    values = []
    for position in range(rows):
        cell = body[position * width : (position + 1) * width]
        stripped = cell.rstrip(b"\x00")
        if b"\x00" in stripped:
            raise _refuse("LEXICAL_EMBEDDED_NUL", label)
        try:
            values.append(stripped.decode("ascii"))
        except UnicodeDecodeError as error:
            raise _refuse("LEXICAL_NON_ASCII", label) from error
    return tuple(values)


def payload_bound(
    rows: int, projection: PufMonetaryProjection
) -> tuple[int, dict[str, int]]:
    """The exact body size this artifact allocates, and its per-column parts."""

    sizes = {
        name: rows * np.dtype(_STRUCTURAL_DTYPES[name]).itemsize
        for name in _STRUCTURAL_COLUMNS
    }
    for column in projection.columns:
        sizes[column.field] = rows * np.dtype(column.numpy_dtype).itemsize
        sizes[f"{column.field}_lexical"] = rows * column.lexical_width
    return sum(sizes.values()), sizes


def _envelope_column_order(projection: PufMonetaryProjection) -> list[str]:
    return [
        *_STRUCTURAL_COLUMNS,
        *(column.field for column in projection.columns),
        *(f"{column.field}_lexical" for column in projection.columns),
    ]


def encode_monetary_projection(
    decoded: PufMonetarySourceProjection,
    projection: PufMonetaryProjection,
    definition: PufRawSourceDefinition,
    *,
    status_artifact_sha256: str,
) -> bytes:
    """Encode the projection into the bounded, self-describing envelope."""

    rows = decoded.rows
    total, sizes = payload_bound(rows, projection)
    if total > BODY_MAX_BYTES:
        raise _refuse("BODY_OVER_BOUND", total)
    dtypes = {
        **_STRUCTURAL_DTYPES,
        **{column.field: column.numpy_dtype for column in projection.columns},
    }
    widths = {column.field: column.lexical_width for column in projection.columns}
    bodies: list[bytes] = []
    columns: list[dict[str, object]] = []
    for name in _envelope_column_order(projection):
        if name.endswith("_lexical"):
            field = name.removesuffix("_lexical")
            body = _fixed_ascii(decoded.lexical[field], widths[field], field)
            entry: dict[str, object] = {
                "name": name,
                "kind": _LEXICAL_KIND,
                "width": widths[field],
            }
        else:
            array = np.ascontiguousarray(decoded.typed[name], dtype=dtypes[name])
            if array.shape != (rows,):
                raise _refuse("COLUMN_SHAPE", name)
            body = array.tobytes()
            entry = {"name": name, "kind": _TYPED_KIND, "dtype": dtypes[name]}
        if len(body) != sizes[name]:
            raise _refuse("COLUMN_SIZE", name)
        bodies.append(body)
        entry["bytes"] = len(body)
        entry["sha256"] = hashlib.sha256(body).hexdigest()
        columns.append(entry)
    header = {
        "schema_version": MONETARY_PROJECTION_TYPE.schema_version,
        "type": [
            MONETARY_PROJECTION_TYPE.name,
            MONETARY_PROJECTION_TYPE.schema_version,
        ],
        "rows": rows,
        "projection_sha256": projection.sha256,
        "projection_route": projection.route,
        "raw_source_definition_sha256": definition.sha256,
        "status_artifact_sha256": _hex64(
            status_artifact_sha256, "status_artifact_sha256"
        ),
        "sources": {
            "main": {
                "sha256": definition.main.sha256,
                "git_blob_sha1": definition.main.git_blob_sha1,
                "bytes": definition.main.bytes,
                "header_record_canonical_sha256": (
                    definition.main.header_record_canonical_sha256
                ),
            },
            "demographic": {
                "sha256": definition.demographic.sha256,
                "git_blob_sha1": definition.demographic.git_blob_sha1,
                "bytes": definition.demographic.bytes,
                "header_record_canonical_sha256": (
                    definition.demographic.header_record_canonical_sha256
                ),
            },
        },
        "header_length_endianness": "big",
        "body_endianness": "little",
        "columns": columns,
        "facts": dict(decoded.facts),
    }
    encoded = canonical_json(header)
    if len(encoded) > HEADER_MAX_BYTES:
        raise _refuse("HEADER_OVER_BOUND", len(encoded))
    return b"".join(
        (MONETARY_PROJECTION_MAGIC, len(encoded).to_bytes(4, "big"), encoded, *bodies)
    )


def _exact_keys(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise _refuse("ENVELOPE_SCHEMA", label)
    return value


def _envelope_int(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _refuse("ENVELOPE_INTEGER", label)
    return value


def _resolve_envelope_projection(
    header: Mapping[str, object],
    projection: PufMonetaryProjection | None,
) -> PufMonetaryProjection | None:
    """The projection this payload may be checked against, or ``None``."""

    route = header["projection_route"]
    if route not in ("packaged", "test_fixture"):
        raise _refuse("ENVELOPE_PROJECTION_ROUTE")
    digest = _hex64(header["projection_sha256"], "envelope.projection_sha256")
    if projection is not None:
        if not isinstance(projection, PufMonetaryProjection):
            raise _refuse("DECODE_PROJECTION_TYPE")
        if projection.route != route or projection.sha256 != digest:
            raise _refuse("ENVELOPE_PROJECTION_MISMATCH")
        return projection
    if route == "packaged":
        packaged = packaged_projection()
        if packaged.sha256 != digest:
            raise _refuse("ENVELOPE_PROJECTION_NOT_PACKAGED")
        return packaged
    return None


def _check_envelope_sources(
    header: Mapping[str, object], projection: PufMonetaryProjection | None
) -> None:
    sources = _exact_keys(
        header["sources"], frozenset({"main", "demographic"}), "sources"
    )
    for name in ("main", "demographic"):
        entry = _exact_keys(sources[name], _ENVELOPE_SOURCE_KEYS, f"sources.{name}")
        _hex64(entry["sha256"], f"sources.{name}.sha256")
        _envelope_int(entry["bytes"], f"sources.{name}.bytes", minimum=1)
        if projection is None:
            continue
        pin = projection.document["sources"][name]
        if (
            entry["sha256"] != pin["sha256"]
            or entry["git_blob_sha1"] != pin["git_blob_sha1"]
            or entry["bytes"] != pin["bytes"]
            or entry["header_record_canonical_sha256"]
            != pin["header_record_canonical_sha256"]
        ):
            raise _refuse("ENVELOPE_SOURCE_PIN", name)


def _envelope_columns(
    header: Mapping[str, object], rows: int, projection: PufMonetaryProjection
) -> list[Mapping[str, object]]:
    """Close the column schema: exact order, kind, dtype, width and size."""

    columns = header["columns"]
    if not isinstance(columns, list):
        raise _refuse("ENVELOPE_COLUMNS")
    declared = _envelope_column_order(projection)
    if len(columns) != len(declared):
        raise _refuse("ENVELOPE_COLUMNS")
    dtypes = {
        **_STRUCTURAL_DTYPES,
        **{column.field: column.numpy_dtype for column in projection.columns},
    }
    widths = {column.field: column.lexical_width for column in projection.columns}
    resolved: list[Mapping[str, object]] = []
    total = 0
    for column, name in zip(columns, declared, strict=True):
        if not isinstance(column, dict) or column.get("name") != name:
            raise _refuse("ENVELOPE_COLUMNS")
        if name.endswith("_lexical"):
            field = name.removesuffix("_lexical")
            entry = _exact_keys(column, _ENVELOPE_LEXICAL_COLUMN_KEYS, name)
            if entry["kind"] != _LEXICAL_KIND:
                raise _refuse("ENVELOPE_COLUMN_KIND", name)
            width = _envelope_int(entry["width"], f"{name}.width", minimum=1)
            if width > LEXICAL_WIDTH_MAX or width != widths[field]:
                raise _refuse("ENVELOPE_COLUMN_WIDTH", name)
            expected = rows * width
        else:
            entry = _exact_keys(column, _ENVELOPE_TYPED_COLUMN_KEYS, name)
            if entry["kind"] != _TYPED_KIND:
                raise _refuse("ENVELOPE_COLUMN_KIND", name)
            # An exact dtype string closes width, signedness and endianness in
            # one comparison, so ``<f8`` can never stand in for ``<i8``.
            if entry["dtype"] != dtypes[name]:
                raise _refuse("ENVELOPE_COLUMN_DTYPE", name)
            expected = rows * np.dtype(dtypes[name]).itemsize
        _hex64(entry["sha256"], f"{name}.sha256")
        size = _envelope_int(entry["bytes"], f"{name}.bytes")
        if size != expected:
            raise _refuse("ENVELOPE_COLUMN_SIZE", name)
        total += size
        if total > BODY_MAX_BYTES:
            raise _refuse("ENVELOPE_BODY_OVER_BOUND", total)
        resolved.append(entry)
    return resolved


def _verify_readback(
    typed: Mapping[str, np.ndarray],
    lexical: Mapping[str, tuple[str, ...]],
    projection: PufMonetaryProjection,
) -> None:
    """Re-prove the projection's own invariants from the arrays read back."""

    rows = int(typed["RECID"].shape[0])
    status = typed["demographic_status"]
    aggregate = typed["disclosure_aggregate"]
    if not np.isin(status, (0, 1)).all():
        raise _refuse("READBACK_DEMOGRAPHIC_STATUS")
    if not np.isin(aggregate, (0, 1)).all():
        raise _refuse("READBACK_AGGREGATE_FLAG")
    recids = typed["RECID"]
    if len(np.unique(recids)) != rows:
        raise _refuse("READBACK_NUMERIC_KEY_COLLISION")
    expected = np.isin(recids, np.asarray(PUF_AGGREGATE_RECIDS, dtype="<i8"))
    if not np.array_equal(aggregate == 1, expected):
        raise _refuse("READBACK_AGGREGATE_RECID")
    known = _amount_known(aggregate)
    for column in projection.columns:
        tokens = lexical[column.field]
        if len(tokens) != rows:
            raise _refuse("READBACK_LEXICAL_ROWS", column.field)
        values = typed[column.field]
        for position, token in enumerate(tokens):
            if bool(known[position]):
                # An individual return re-parses under the unchanged integer
                # grammar and must equal its typed value. The sentinel is
                # unreachable for that grammar, so a sentinel parked on an
                # ordinary row is caught here as a lexical disagreement.
                if _amount(token, column) != int(values[position]):
                    raise _refuse("READBACK_LEXICAL_DISAGREEMENT", column.field)
            else:
                # An aggregate row validates under the aggregate grammar and
                # must hold the sentinel exactly. A zero, or any other stored
                # number, would be a value this artifact never observed.
                _aggregate_token(token, column)
                if int(values[position]) != AGGREGATE_SENTINEL:
                    raise _refuse("READBACK_AGGREGATE_SLOT", column.field)


def decode_monetary_projection(
    payload: bytes,
    projection: PufMonetaryProjection | None = None,
    *,
    expected_status_artifact_sha256: str | None = None,
) -> PufMonetarySourceProjection:
    """Decode an artifact payload and re-prove it, never trusting its header.

    The header is a closed schema; every reported fact is recomputed from the
    arrays actually read back and must agree exactly, so a payload cannot
    report a row count, a negative census, a class partition or a column sum
    it does not carry.
    """

    if not isinstance(payload, bytes):
        raise _refuse("ENVELOPE_MAGIC")
    if not payload.startswith(MONETARY_PROJECTION_MAGIC):
        # A superseded magic is named, not reinterpreted: those payloads
        # declare a different amount universe, and silently reading one under
        # this contract would report aggregates as individual amounts.
        if any(payload.startswith(magic) for magic in SUPERSEDED_MAGIC):
            raise _refuse("ENVELOPE_SUPERSEDED_MAGIC")
        raise _refuse("ENVELOPE_MAGIC")
    start = len(MONETARY_PROJECTION_MAGIC)
    if len(payload) < start + 4:
        raise _refuse("ENVELOPE_TRUNCATED")
    length = int.from_bytes(payload[start : start + 4], "big")
    if not 0 < length <= HEADER_MAX_BYTES or len(payload) < start + 4 + length:
        raise _refuse("ENVELOPE_HEADER_LENGTH")
    encoded = payload[start + 4 : start + 4 + length]
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise _refuse("ENVELOPE_HEADER_JSON") from error
    try:
        recanonical = canonical_json(header)
    except (TypeError, ValueError, RecursionError) as error:
        raise _refuse("ENVELOPE_HEADER_JSON") from error
    if recanonical != encoded:
        raise _refuse("ENVELOPE_HEADER_NOT_CANONICAL")
    _exact_keys(header, _ENVELOPE_HEADER_KEYS, "header")
    if header["schema_version"] != MONETARY_PROJECTION_TYPE.schema_version or header[
        "type"
    ] != [
        MONETARY_PROJECTION_TYPE.name,
        MONETARY_PROJECTION_TYPE.schema_version,
    ]:
        raise _refuse("ENVELOPE_TYPE")
    if (
        header["header_length_endianness"] != "big"
        or header["body_endianness"] != "little"
    ):
        raise _refuse("ENVELOPE_ENDIANNESS")
    _hex64(header["raw_source_definition_sha256"], "raw_source_definition_sha256")
    _hex64(header["status_artifact_sha256"], "status_artifact_sha256")
    if expected_status_artifact_sha256 is not None:
        _hex64(expected_status_artifact_sha256, "expected_status_artifact_sha256")
        if header["status_artifact_sha256"] != expected_status_artifact_sha256:
            raise _refuse("ENVELOPE_STATUS_ARTIFACT_MISMATCH")
    rows = _envelope_int(header["rows"], "rows", minimum=1)
    resolved = _resolve_envelope_projection(header, projection)
    if resolved is None:
        # A fixture-route payload has no reconstructible projection document,
        # so there is nothing to close its column set against. That is a
        # caller error here rather than a weaker structural pass: the
        # projection is exactly what makes the columns meaningful.
        raise _refuse("ENVELOPE_PROJECTION_REQUIRED")
    if resolved.raw_definition_sha256 != header["raw_source_definition_sha256"]:
        raise _refuse("ENVELOPE_RAW_DEFINITION_MISMATCH")
    _check_envelope_sources(header, resolved)
    columns = _envelope_columns(header, rows, resolved)

    cursor = start + 4 + length
    typed: dict[str, np.ndarray] = {}
    lexical: dict[str, tuple[str, ...]] = {}
    for column in columns:
        name = column["name"]
        size = column["bytes"]
        body = payload[cursor : cursor + size]
        if len(body) != size:
            raise _refuse("ENVELOPE_BODY_TRUNCATED", name)
        if hashlib.sha256(body).hexdigest() != column["sha256"]:
            raise _refuse("ENVELOPE_BODY_DIGEST", name)
        cursor += size
        if column["kind"] == _TYPED_KIND:
            typed[name] = np.frombuffer(body, dtype=np.dtype(column["dtype"]))
        else:
            field = name.removesuffix("_lexical")
            lexical[field] = _read_fixed_ascii(body, rows, column["width"], field)
    if cursor != len(payload):
        raise _refuse("ENVELOPE_TRAILING_BYTES")

    _verify_readback(typed, lexical, resolved)
    facts = header["facts"]
    _exact_keys(facts, _ENVELOPE_FACT_KEYS, "facts")
    for field, classes in facts["column_facts"].items():
        if field not in resolved.fields:
            raise _refuse("ENVELOPE_FACTS", "column_facts")
        _exact_keys(classes, frozenset(ROW_CLASSES), f"column_facts.{field}")
        for name, entry in classes.items():
            keys = (
                _AGGREGATE_COLUMN_FACT_KEYS
                if name == AGGREGATE_ROW_CLASS
                else _RETURN_COLUMN_FACT_KEYS
            )
            _exact_keys(entry, keys, f"column_facts.{field}.{name}")
    recomputed = _facts_from_arrays(typed, lexical, resolved)
    if canonical_json(recomputed) != canonical_json(dict(facts)):
        raise _refuse("ENVELOPE_FACTS_INCONSISTENT")
    if recomputed["rows"] != rows:
        raise _refuse("ENVELOPE_ROWS")
    return PufMonetarySourceProjection(
        typed=MappingProxyType(typed),
        lexical=MappingProxyType(lexical),
        facts=MappingProxyType(recomputed),
        column_metadata=_frozen(recomputed["column_metadata"]),
    )


# --------------------------------------------------------------------------
# The graph producer, consumer and registry
# --------------------------------------------------------------------------

_IMPLEMENTATION_MODULES = (
    "microcosm.build.us_runtime.puf_raw_source",
    "microcosm.build.us_runtime.puf_monetary_source",
)
#: The implementation identity moves with the contract: a node written
#: against the version-2 amount universe must not bind to this kernel.
_IMPLEMENTATION_IDENTITY_MAGIC = b"us.puf.monetary_source/implementation/3\n"

_STATUS_ALIAS = "return_status"
_PROJECTION_ALIAS = "monetary_projection"


class USPufMonetarySourceKernel(KernelBase):
    """Read the pinned main delivery and produce the monetary projection."""

    ref = "us.puf.monetary_source.projection@3"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy",),
    )

    def __init__(
        self,
        *,
        projection: PufMonetaryProjection | None = None,
        definition: PufRawSourceDefinition | None = None,
        source_codecs: SourceCodecRegistry | None = None,
    ) -> None:
        self.definition = (
            packaged_raw_definition() if definition is None else definition
        )
        self.projection = packaged_projection() if projection is None else projection
        if self.projection.route not in ("packaged", "test_fixture"):
            raise _refuse("KERNEL_PROJECTION_ROUTE")
        if self.projection.route != self.definition.route:
            raise _refuse("KERNEL_ROUTE_DISAGREEMENT")
        if self.projection.raw_definition_sha256 != self.definition.sha256:
            raise _refuse("KERNEL_DEFINITION_MISMATCH")
        if self.projection.route == "packaged":
            # The production wrapper, not the bare decoder, is the authority:
            # the projection must be the packaged file's own canonical bytes,
            # and the document it carries must still be those exact bytes.
            packaged = packaged_projection()
            if (
                self.projection.canonical != packaged.canonical
                or self.projection.sha256 != packaged.sha256
                or canonical_json(_thawed(self.projection.document))
                != self.projection.canonical
            ):
                raise _refuse("KERNEL_PROJECTION_NOT_PACKAGED")
        self.source_codecs = (
            puf_raw_source_codecs(self.definition)
            if source_codecs is None
            else source_codecs
        )

    def implementation_hash(self) -> str:
        """Module source, dependency versions and the effective CSV profile."""

        from importlib import import_module

        base = source_hash(
            *(import_module(name) for name in _IMPLEMENTATION_MODULES),
            dependencies=self.capabilities.dependencies,
        )
        digest = hashlib.sha256(_IMPLEMENTATION_IDENTITY_MAGIC)
        digest.update(base.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(csv_acceptance_profile()))
        return digest.hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        definition = self.definition
        projection = self.projection
        if (
            context.node.kernel != self.ref
            or context.node.structural is not StructuralDelta.NONE
            or context.node.sources != (definition.main.source_name,)
            or context.node.inputs
            or context.node.outputs
            or len(context.node.artifact_inputs) != 1
            or context.node.artifact_inputs[0].type != RETURN_STATUS_TYPE
            or context.node.artifact_outputs
            != (ArtifactOutput(_PROJECTION_ALIAS, MONETARY_PROJECTION_TYPE),)
            or set(context.params) != {"definition", "projection"}
        ):
            raise _refuse("NODE_DECLARATION")
        if context.params["definition"] != definition.params_text:
            raise _refuse("NODE_DEFINITION_PARAM")
        if context.params["projection"] != projection.params_text:
            raise _refuse("NODE_PROJECTION_PARAM")
        alias = context.node.artifact_inputs[0].name
        value = context.artifacts[alias]
        if value.type != RETURN_STATUS_TYPE:
            raise _refuse("STATUS_ARTIFACT_TYPE")
        status = decode_return_status(value.payload, definition)
        main_bytes = load_source_bytes(
            definition.main.codec,
            context.sources[definition.main.source_name],
            registry=self.source_codecs,
        )
        decoded = decode_puf_monetary_source(main_bytes, status, projection, definition)
        payload = encode_monetary_projection(
            decoded,
            projection,
            definition,
            status_artifact_sha256=hashlib.sha256(value.payload).hexdigest(),
        )
        return KernelResult(
            artifacts={_PROJECTION_ALIAS: payload},
            receipt={
                "puf_monetary_source_projection": {
                    "projection_sha256": projection.sha256,
                    "projection_route": projection.route,
                    "definition_sha256": definition.sha256,
                    "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                    "artifact_bytes": len(payload),
                    "status_artifact_sha256": hashlib.sha256(value.payload).hexdigest(),
                    "main_sha256": definition.main.sha256,
                    "csv_acceptance_profile": csv_acceptance_profile(),
                    "projected_fields": list(projection.fields),
                    **{
                        key: value
                        for key, value in decoded.facts.items()
                        if key != "column_facts"
                    },
                }
            },
        )


class USPufMonetaryProjectionSummaryKernel(KernelBase):
    """A downstream consumer: decode the projection and report its census."""

    ref = "us.puf.monetary_source.projection_summary@3"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy",),
    )

    def __init__(self, *, projection: PufMonetaryProjection | None = None) -> None:
        self.projection = packaged_projection() if projection is None else projection

    def implementation_hash(self) -> str:
        from importlib import import_module

        return source_hash(
            *(import_module(name) for name in _IMPLEMENTATION_MODULES),
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        if (
            context.node.kernel != self.ref
            or context.node.structural is not StructuralDelta.NONE
            or context.node.sources
            or context.node.inputs
            or context.node.outputs
            or context.node.artifact_outputs
            or len(context.node.artifact_inputs) != 1
            or context.node.artifact_inputs[0].type != MONETARY_PROJECTION_TYPE
            or context.params
        ):
            raise _refuse("SUMMARY_NODE_DECLARATION")
        alias = context.node.artifact_inputs[0].name
        value = context.artifacts[alias]
        if value.type != MONETARY_PROJECTION_TYPE:
            raise _refuse("SUMMARY_ARTIFACT_TYPE")
        decoded = decode_monetary_projection(value.payload, self.projection)
        facts = decoded.facts
        # The per-return negative census covers the two return classes only.
        # Summing it across all three would have counted a token class as a
        # monetary sign, which is exactly the conflation this contract ends.
        per_return_negatives = {
            field: sum(int(classes[name]["negative"]) for name in RETURN_ROW_CLASSES)
            for field, classes in facts["column_facts"].items()
        }
        # Aggregate-only: token-class counts and a width, never a token, a
        # RECID or anything that could be read as an aggregate amount. The
        # exact row/token pairs stay behind the local decoded accessor.
        aggregate_token_classes = {
            field: dict(classes[AGGREGATE_ROW_CLASS])
            for field, classes in facts["column_facts"].items()
        }
        return KernelResult(
            receipt={
                "puf_monetary_projection_summary": {
                    "producer_key": value.producer_key,
                    "rows": decoded.rows,
                    "projected_columns": facts["projected_columns"],
                    "row_class_rows": dict(facts["row_class_rows"]),
                    "amount_known_rows": facts["amount_known_rows"],
                    "amount_known_rule": facts["amount_known_rule"],
                    "per_return_negative_values_by_field": per_return_negatives,
                    "aggregate_rows_are_lexical_only": True,
                    "aggregate_typed_sentinel": facts["aggregate_typed_sentinel"],
                    "aggregate_token_grammar": facts["aggregate_token_grammar"],
                    "aggregate_lexical_authority": facts["aggregate_lexical_authority"],
                    "aggregate_numeric_interpretation": facts[
                        "aggregate_numeric_interpretation"
                    ],
                    "aggregate_lexical_token_classes": aggregate_token_classes,
                    "period_semantics": facts["period_semantics"],
                    "amount_units": facts["amount_units"],
                    "amount_units_scope": facts["amount_units_scope"],
                    "target_year_use": facts["target_year_use"],
                    "column_metadata": _thawed(decoded.column_metadata),
                }
            }
        )


def us_puf_monetary_source_node(
    *,
    population: str,
    producer: str = PUF_RAW_SOURCE_NODE,
    producer_output: str = "return_status",
    projection: PufMonetaryProjection | None = None,
    definition: PufRawSourceDefinition | None = None,
    stage: str = MONETARY_PROJECTION_STAGE,
) -> Node:
    """Declare the producer node inside an explicitly named host population."""

    if not isinstance(population, str) or not population:
        raise _refuse("NODE_POPULATION_REQUIRED")
    resolved_definition = (
        packaged_raw_definition() if definition is None else definition
    )
    resolved = packaged_projection() if projection is None else projection
    if resolved.raw_definition_sha256 != resolved_definition.sha256:
        raise _refuse("NODE_DEFINITION_MISMATCH")
    return Node(
        f"{stage}.projection",
        USPufMonetarySourceKernel.ref,
        population=population,
        sources=(resolved_definition.main.source_name,),
        params={
            "definition": resolved_definition.params_text,
            "projection": resolved.params_text,
        },
        artifact_inputs=(
            ArtifactInput(_STATUS_ALIAS, producer, producer_output, RETURN_STATUS_TYPE),
        ),
        artifact_outputs=(ArtifactOutput(_PROJECTION_ALIAS, MONETARY_PROJECTION_TYPE),),
        description=(
            "Decode the enumerated 2015 PUF monetary source columns at return "
            "grain, bound to the reviewed status artifact. Owns no population "
            "columns and admits no target year."
        ),
    )


def us_puf_monetary_projection_summary_node(
    *,
    population: str,
    producer: str = MONETARY_PROJECTION_NODE,
    stage: str = MONETARY_PROJECTION_STAGE,
) -> Node:
    """Declare an ordinary consumer of the projection artifact."""

    if not isinstance(population, str) or not population:
        raise _refuse("NODE_POPULATION_REQUIRED")
    return Node(
        f"{stage}.projection_summary",
        USPufMonetaryProjectionSummaryKernel.ref,
        population=population,
        artifact_inputs=(
            ArtifactInput(
                _PROJECTION_ALIAS,
                producer,
                _PROJECTION_ALIAS,
                MONETARY_PROJECTION_TYPE,
            ),
        ),
        description="Read the monetary projection artifact and report its census.",
    )


def register_us_puf_monetary_source_kernels(
    registry: KernelRegistry,
    *,
    projection: PufMonetaryProjection | None = None,
    definition: PufRawSourceDefinition | None = None,
    source_codecs: SourceCodecRegistry | None = None,
) -> KernelRegistry:
    """Register the producer and consumer kernels into ``registry``."""

    registry.register(
        USPufMonetarySourceKernel(
            projection=projection,
            definition=definition,
            source_codecs=source_codecs,
        )
    )
    registry.register(USPufMonetaryProjectionSummaryKernel(projection=projection))
    return registry
