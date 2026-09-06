"""Typed PAW transport and reported-income accounting, never source authority.

The six results are source observations and an accounting residual. They are
not simulated SSI/TANF, ACS other income, or a completed eight-category bridge.
"""

from __future__ import annotations

import struct
from dataclasses import InitVar, dataclass

import numpy as np
import pandas as pd

from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json

from . import asec_income_observations as source
from . import graph_context
from .asec_current_money import MoneyRefusalError, _json, _parse, _require, _sha
from .asec_current_money_selection import (
    US_ASEC_PREPARED_RECEIPT_TYPE,
    US_ASEC_SELECTED_MONEY_TYPE,
    US_ASEC_SELECTION_TYPE,
    SelectedCurrentMoney,
    decode_selected_current_money,
    encode_selected_current_money,
)
from .graph_context import US_FRAME_CONTEXT_TYPE
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)

US_ASEC_INCOME_OBSERVATIONS_TYPE = ArtifactType(
    "microcosm.us.asec_income_observations", 1
)
US_ASEC_REPORTED_INCOME_TYPE = ArtifactType("microcosm.us.asec_reported_income", 3)
STAGE = "asec_prepared_v3"
REPORTED_INCOME_NODE = "asec_prepared.reported_income"
RESULT_COLUMNS = tuple(
    "asec_reported_" + name + "_2024_price"
    for name in (
        "wage_income",
        "net_self_employment_income",
        "ssi",
        "cash_public_assistance",
        "total_income",
        "income_unallocated",
    )
)
CONTRIBUTING_FIELDS = ("WSAL_VAL", "SEMP_VAL", "FRSE_VAL", "SSI_VAL", "PTOTVAL")
COORDINATES = ("person_id", "income_year", "A_AGE")
EVIDENCE_COLUMNS = tuple(
    f"{name}.{axis}"
    for name in CONTRIBUTING_FIELDS
    for axis in ("status", "validity", "zero_origin")
) + (
    "PAW_VAL.status",
    "PAW_VAL.validity",
    "PAW_VAL.zero_origin",
    "PAW_VAL.source_encoding_class",
)
ACCOUNTING_COLUMNS = COORDINATES + RESULT_COLUMNS + EVIDENCE_COLUMNS
ACCOUNTING_DTYPES = tuple(["<i8"] * 3 + ["<f8"] * 6 + ["u1"] * len(EVIDENCE_COLUMNS))
ACCOUNTING_MAGIC = b"MCAINACC\x03"
ACCOUNTING_KIND = "microcosm.us.asec_reported_income.v3"
ACCOUNTING_HEADER_MAX = 65536
ACCOUNTING_ROW_BYTES = 8 * 9 + len(EVIDENCE_COLUMNS)
ACCOUNTING_PAYLOAD_MAX_BYTES = (
    len(ACCOUNTING_MAGIC)
    + 4
    + ACCOUNTING_HEADER_MAX
    + source._MAX_PERSONS * ACCOUNTING_ROW_BYTES
    + 32
)
ARITHMETIC_ORDER = "W=WSAL_VAL;E=SEMP_VAL+FRSE_VAL;S=SSI_VAL;P=PAW_VAL_2024_price;T=PTOTVAL;U=T-(((W+E)+S)+P)"
_TOKEN = object()
_ACCOUNTING_TOKEN = object()
_BINDING_KEYS = frozenset(
    {
        "parser_profile",
        "header_sha256",
        "content_sha256",
        "payload_sha256",
        "rows",
        "columns",
        "kind",
        "scope_sha256",
    }
)


@dataclass(frozen=True)
class BoundIncomeObservations:
    """Full native source buffers bound to CREATE, without source/Frame authority."""

    header: bytes
    body: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "BOUND_INCOME_CONSTRUCTOR")

    def array(self, name):
        _require(name in source.COLUMNS, "BOUND_INCOME_COLUMN")
        n = _parse(self.header)["rows"]
        return np.frombuffer(
            self.body,
            dtype="<f8" if name == source.COLUMNS[-1] else "<i8",
            count=n,
            offset=source.COLUMNS.index(name) * n * 8,
        )


def bind_income_observations(
    payload: bytes, *, prepared_receipt: dict
) -> BoundIncomeObservations:
    """Admit graph transport only against the independently issued CREATE receipt."""
    try:
        _require(
            type(payload) is bytes
            and len(source.MAGIC) + 4 + 32
            < len(payload)
            <= source.INCOME_PAYLOAD_MAX_BYTES,
            "BOUND_INCOME_SIZE",
        )
        binding = prepared_receipt["income_observations"]
        _require(
            set(binding) == _BINDING_KEYS
            and prepared_receipt["schema"] == "microcosm.us.asec_prepared_receipt.v3"
            and prepared_receipt["source_kind"] == "us_asec_prepared_current_money_v3"
            and prepared_receipt["release_eligible"] is False
            and binding["kind"] == source.ARTIFACT_KIND
            and binding["columns"] == list(source.COLUMNS),
            "BOUND_INCOME_PREPARATION",
        )
        _require(_sha(payload) == binding["payload_sha256"], "BOUND_INCOME_PAYLOAD")
        _require(
            payload.startswith(source.MAGIC)
            and _sha(payload[:-32]) == payload[-32:].hex(),
            "BOUND_INCOME_FRAMING",
        )
        length = struct.unpack_from("<I", payload, len(source.MAGIC))[0]
        start = len(source.MAGIC) + 4
        _require(
            0 < length <= source._HEADER_MAX and start + length < len(payload) - 32,
            "BOUND_INCOME_HEADER_SIZE",
        )
        header, body = payload[start : start + length], payload[start + length : -32]
        data = _parse(header)
        _require(
            set(data) == source._HEADER_KEYS and _json(data) == header,
            "BOUND_INCOME_HEADER_SCHEMA",
        )
        _require(
            _sha(header) == binding["header_sha256"]
            and _sha(header + body) == binding["content_sha256"],
            "BOUND_INCOME_CONTENT",
        )
        rows = data["rows"]
        _require(
            type(rows) is int
            and 0 < rows <= source._MAX_PERSONS
            and rows == binding["rows"] == prepared_receipt["entity_rows"]["person"]
            and len(body) == rows * len(source.COLUMNS) * 8,
            "BOUND_INCOME_ROWS",
        )
        contract = source._contract()
        factors = source.price_factors(contract)
        _require(
            type(data["schema_version"]) is int
            and data["schema_version"] == 1
            and data["artifact_kind"] == source.ARTIFACT_KIND
            and data["columns"] == list(source.COLUMNS)
            and data["column_dtypes"] == source.COLUMN_DTYPES
            and data["release_eligible"] is False
            and data["monetary_basis"] == contract["monetary_basis"]
            and data["price"]
            == {k: contract["price"][k] for k in ("algorithm", "target_year", "cells")}
            | {"factors": [list(f) for f in factors]}
            and data["join_keys"] == ["source_year", "PERIDNUM"]
            and data["crosschecks"] == ["source_household_id", "A_LINENO", "A_AGE"]
            and data["reference_period"]
            == "interview_household_one_year_after_income_year"
            and data["encoding_contract"]
            == "independently_reconstructed_canonical_numeric_bytes_v1"
            and _json(data["implementation"]) == _json(source._implementation())
            and binding["parser_profile"] == data["implementation"]["parser_profile"],
            "BOUND_INCOME_DEFINITION",
        )
        _require(
            _sha(data["source_identity"].encode())
            == prepared_receipt["source_evidence_sha256"]
            and data["t1_sha256"] == prepared_receipt["t1_sha256"]
            and data["scope_sha256"]
            == binding["scope_sha256"]
            == prepared_receipt["scope_sha256"]
            and data["money_header_sha256"] == prepared_receipt["money_header_sha256"]
            and data["money_content_sha256"]
            == prepared_receipt["money_content_sha256"],
            "BOUND_INCOME_SOURCE",
        )
        evidence = _parse(data["source_identity"].encode())
        _require(
            data["source_frame_sha256"] == evidence["frame_sha256"]
            and data["t1_sha256"] == evidence["person_income_attachment_sha256"],
            "BOUND_INCOME_SOURCE_FIELDS",
        )
        _require(
            data["body_sha256"] == _sha(body)
            and data["column_sha256"]
            == {
                name: _sha(body[i * rows * 8 : (i + 1) * rows * 8])
                for i, name in enumerate(source.COLUMNS)
            },
            "BOUND_INCOME_BUFFERS",
        )
        result = BoundIncomeObservations(header, body, _token=_TOKEN)
        ids, years = result.array("person_id"), result.array("income_year")
        _require(
            len(np.unique(ids)) == rows
            and bool(np.isin(years, [2022, 2023, 2024]).all()),
            "BOUND_INCOME_COORDINATES",
        )
        age, yn, nominal = (result.array(n) for n in ("A_AGE", "PAW_YN", "PAW_VAL"))
        source._validate_observations(age, yn, nominal)
        _require(
            source._restate_amounts(nominal, years, factors).tobytes()
            == result.array("PAW_VAL_2024_price").tobytes(),
            "BOUND_INCOME_RESTATEMENT",
        )
        classes = source.zero_origin_classes(age, yn, nominal)
        counts = {
            str(y): {
                name: int(((years == y) & (classes == i)).sum())
                for i, name in enumerate(source.ZERO_CLASSES)
            }
            for y in (2022, 2023, 2024)
        }
        _require(
            data["zero_origin_derivation"]
            == {
                "rule": "PAW_VAL_A_AGE_PAW_YN_source_encoding_v1",
                "classes": list(source.ZERO_CLASSES),
                "cohort_counts": counts,
            },
            "BOUND_INCOME_ZERO_CLASSES",
        )
        _require(
            type(data["sources"]) is list and len(data["sources"]) == 3,
            "BOUND_INCOME_JOINS",
        )
        for joined, pins in zip(data["sources"], source._MEMBER_PINS, strict=True):
            year, member, archive_pin, pin, expected_rows, _size = pins
            compared = joined["incumbent_compared_rows"]
            _require(
                type(compared) is dict
                and set(compared) == {"PAW_VAL"}
                and type(compared["PAW_VAL"]) is int
                and 0 <= compared["PAW_VAL"] <= expected_rows,
                "BOUND_INCOME_INCUMBENT_COVERAGE",
            )
            _require(
                joined
                == {
                    "income_year": year,
                    "survey_year": year + 1,
                    "member": member,
                    "archive_sha256": archive_pin,
                    "member_sha256": pin,
                    "source_rows": expected_rows,
                    "joined_rows": int((years == year).sum()),
                    "unreferenced_source_rows": 0,
                    "incumbent_compared_rows": compared,
                    "incumbent_conflicts": 0,
                    "native_key_or_age_conflicts": 0,
                    "zero_origin_counts": counts[str(year)],
                    "paw_yn_zero_at_age_15_plus": counts[str(year)][
                        "niu_in_age_universe"
                    ],
                }
                and int((years == year).sum()) == expected_rows,
                "BOUND_INCOME_JOINS",
            )
        return result
    except MoneyRefusalError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        AttributeError,
        struct.error,
    ):
        raise MoneyRefusalError("BOUND_INCOME_CONTRACT") from None


@dataclass(frozen=True)
class ReportedIncomeAccounting:
    """Derived buffers: their only reader reconstructs from the same typed inputs."""

    header: bytes
    buffers: tuple[bytes, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _ACCOUNTING_TOKEN, "ACCOUNTING_CONSTRUCTOR")

    def array(self, name):
        _require(name in ACCOUNTING_COLUMNS, "ACCOUNTING_COLUMN")
        index = ACCOUNTING_COLUMNS.index(name)
        return np.frombuffer(self.buffers[index], dtype=ACCOUNTING_DTYPES[index])


def derive_reported_income(
    selected,
    income,
    *,
    person_ids,
    income_years,
    person_positions_sha256,
    prepared_receipt_sha256,
    selection_sha256,
    selected_money_sha256,
    income_payload_sha256,
    source_producer_key,
    selection_producer_key,
    frame_context_sha256,
):
    """Reconstruct six observations in fixed float64 order, preserving source axes."""
    _require(
        type(selected) is SelectedCurrentMoney
        and type(income) is BoundIncomeObservations,
        "ACCOUNTING_INPUT_TYPES",
    )
    _require(
        all(
            type(a) is np.ndarray and a.dtype == np.dtype("int64") and a.ndim == 1
            for a in (person_ids, income_years)
        )
        and person_ids.shape == income_years.shape == (selected.person_rows,),
        "ACCOUNTING_COORDINATES",
    )
    selected_header, income_header = selected.header_data, _parse(income.header)
    identity = graph_context._row_identity(
        pd.DataFrame({"person_id": person_ids}), "person"
    )
    _require(
        selected_header["person_identity_sha256"] == identity["ordered_ids_sha256"]
        and selected_header["parent_header_sha256"]
        == income_header["money_header_sha256"]
        and selected_header["parent_content_sha256"]
        == income_header["money_content_sha256"]
        and selected_header["prepared_receipt_sha256"] == prepared_receipt_sha256
        and selected_header["selection_sha256"] == selection_sha256,
        "ACCOUNTING_MONEY_BINDING",
    )
    source_payload = (
        source.MAGIC
        + struct.pack("<I", len(income.header))
        + income.header
        + income.body
    )
    _require(
        _sha(encode_selected_current_money(selected)) == selected_money_sha256
        and _sha(source_payload + bytes.fromhex(_sha(source_payload)))
        == income_payload_sha256,
        "ACCOUNTING_INPUT_PAYLOADS",
    )
    positions = (
        pd.Index(income.array("person_id")).get_indexer(person_ids).astype("int64")
    )
    _require(
        bool((positions >= 0).all())
        and len(np.unique(positions)) == len(positions)
        and (len(positions) < 2 or bool((np.diff(positions) > 0).all()))
        and _sha(positions.tobytes()) == person_positions_sha256,
        "ACCOUNTING_SOURCE_POSITIONS",
    )
    _require(
        np.array_equal(income_years, income.array("income_year")[positions]),
        "ACCOUNTING_SOURCE_COHORTS",
    )
    _require(
        all(selected.entity_of(name) == "person" for name in CONTRIBUTING_FIELDS),
        "ACCOUNTING_FIELD_ENTITY",
    )
    w = selected.field("WSAL_VAL").amounts
    e = selected.field("SEMP_VAL").amounts + selected.field("FRSE_VAL").amounts
    s = selected.field("SSI_VAL").amounts
    p = income.array("PAW_VAL_2024_price")[positions]
    t = selected.field("PTOTVAL").amounts
    u = t - (((w + e) + s) + p)
    results = (w, e, s, p, t, u)
    _require(all(np.isfinite(a).all() for a in results), "ACCOUNTING_NONFINITE")
    age = income.array("A_AGE")[positions]
    nominal, yn = income.array("PAW_VAL")[positions], income.array("PAW_YN")[positions]
    buffers = [
        a.astype("<i8", copy=False).tobytes() for a in (person_ids, income_years, age)
    ]
    buffers.extend(a.astype("<f8", copy=False).tobytes() for a in results)
    for name in CONTRIBUTING_FIELDS:
        field = selected.field(name)
        buffers.extend(
            (field.status_bytes, field.validity_bytes, field.zero_origin_bytes)
        )
    zero = nominal == 0
    buffers.extend(
        (
            np.where(zero, 2, 1).astype("u1").tobytes(),
            np.ones(len(p), dtype="u1").tobytes(),
            (zero.astype("u1") * 2).tobytes(),
            source.zero_origin_classes(age, yn, nominal).tobytes(),
        )
    )
    cursor, columns = 0, []
    for name, dtype, buffer in zip(
        ACCOUNTING_COLUMNS, ACCOUNTING_DTYPES, buffers, strict=True
    ):
        columns.append(
            {
                "name": name,
                "dtype": dtype,
                "offset": cursor,
                "bytes": len(buffer),
                "sha256": _sha(buffer),
            }
        )
        cursor += len(buffer)
    rows = len(person_ids)
    _require(
        0 < rows <= source._MAX_PERSONS and cursor == rows * ACCOUNTING_ROW_BYTES,
        "ACCOUNTING_LENGTH",
    )
    header = {
        "schema_version": 3,
        "artifact_kind": ACCOUNTING_KIND,
        "rows": rows,
        "columns": columns,
        "body_sha256": _sha(b"".join(buffers)),
        "inputs": {
            "prepared_receipt_sha256": prepared_receipt_sha256,
            "selection_sha256": selection_sha256,
            "selected_money_sha256": selected_money_sha256,
            "income_payload_sha256": income_payload_sha256,
            "source_producer_key": source_producer_key,
            "selection_producer_key": selection_producer_key,
            "person_positions_sha256": person_positions_sha256,
            "frame_context_sha256": frame_context_sha256,
        },
        "source": {
            name: _parse(income.header)[name]
            for name in (
                "source_identity",
                "t1_sha256",
                "scope_sha256",
                "money_header_sha256",
                "money_content_sha256",
            )
        },
        "arithmetic_order": ARITHMETIC_ORDER,
        "monetary_basis": _parse(income.header)["monetary_basis"],
        "old_evidence_axes": "retained_exactly_no_origin_upgrade",
        "residual_meaning": "signed_source_accounting_residual_not_acs_other_income",
        "release_eligible": False,
    }
    encoded = _json(header)
    _require(len(encoded) <= ACCOUNTING_HEADER_MAX, "ACCOUNTING_HEADER_SIZE")
    return ReportedIncomeAccounting(encoded, tuple(buffers), _token=_ACCOUNTING_TOKEN)


def encode_reported_income(value):
    _require(type(value) is ReportedIncomeAccounting, "ACCOUNTING_TYPE")
    payload = (
        ACCOUNTING_MAGIC
        + struct.pack("<I", len(value.header))
        + value.header
        + b"".join(value.buffers)
    )
    _require(len(payload) + 32 <= ACCOUNTING_PAYLOAD_MAX_BYTES, "ACCOUNTING_SIZE")
    return payload + bytes.fromhex(_sha(payload))


def read_reported_income(payload, selected, income, **bindings):
    """Never parse candidate authority: reconstruct then compare its canonical bytes."""
    _require(
        type(payload) is bytes and len(payload) <= ACCOUNTING_PAYLOAD_MAX_BYTES,
        "ACCOUNTING_SIZE",
    )
    expected = derive_reported_income(selected, income, **bindings)
    _require(payload == encode_reported_income(expected), "ACCOUNTING_RECONSTRUCTION")
    return expected


def reported_income_declarations():
    return tuple(Owned("person", name, "float64") for name in RESULT_COLUMNS)


class USAsecReportedIncomeKernel(KernelBase):
    """Pure fifth branch; declared source_year maps to the artifact's income_year."""

    ref = "us.asec_prepared.reported_income@3"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=STAGE_DEPENDENCIES[STAGE],
    )

    def implementation_hash(self):
        return implementation_hash(STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        from . import graph_asec_prepared as bridge

        bridge._phase(context)
        node = context.node
        _require(
            node.kernel == self.ref
            and node.structural is StructuralDelta.NONE
            and not node.sources,
            "ACCOUNTING_NODE",
        )
        _require(
            node.inputs == (Slice("person", ("source_year",)),)
            and node.outputs == reported_income_declarations(),
            "ACCOUNTING_DECLARATIONS",
        )
        _require(
            set(context.artifacts)
            == {
                "frame_context",
                "selection",
                "selected_current_money",
                "prepared_receipt",
                "income_observations",
            },
            "ACCOUNTING_ARTIFACT_INPUTS",
        )
        _require(
            node.artifact_outputs
            == (
                ArtifactOutput("reported_income", US_ASEC_REPORTED_INCOME_TYPE),
                ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ),
            "ACCOUNTING_ARTIFACT_OUTPUTS",
        )
        values = {
            name: bridge._artifact(context, name, kind)
            for name, kind in (
                ("frame_context", US_FRAME_CONTEXT_TYPE),
                ("selection", US_ASEC_SELECTION_TYPE),
                ("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
                ("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
                ("income_observations", US_ASEC_INCOME_OBSERVATIONS_TYPE),
            )
        }
        selection_producer = bridge._same_producer(
            context, ("frame_context", "selection", "selected_current_money")
        )
        source_producer = bridge._same_producer(
            context, ("prepared_receipt", "income_observations")
        )
        document = bridge._document(values["frame_context"])
        person = context.tables["person"]
        bridge._check_identity(document, "person", person)
        _require(
            not set(RESULT_COLUMNS) & set(document["entities"]["person"]["columns"]),
            "OWNED_LEAF_INCUMBENT",
        )
        prepared = _parse(values["prepared_receipt"].payload)
        selection = _parse(values["selection"].payload)
        _require(
            canonical_json(prepared) == values["prepared_receipt"].payload
            and canonical_json(selection) == values["selection"].payload
            and selection["schema"] == bridge.SELECTION_SCHEMA,
            "ACCOUNTING_RECEIPTS",
        )
        _require(
            selection["parent"]["producer_key"] == source_producer
            and selection["parent"]["prepared_receipt_sha256"]
            == _sha(values["prepared_receipt"].payload),
            "ACCOUNTING_SELECTION_PARENT",
        )
        for entity, declared in document["entities"].items():
            _require(
                all(
                    selection["entities"][entity][key] == declared[key]
                    for key in ("rows", "ordered_ids_sha256")
                ),
                "ACCOUNTING_SELECTION_IDENTITY",
            )
        selected = decode_selected_current_money(
            values["selected_current_money"].payload,
            expected_parent_header_sha256=prepared["money_header_sha256"],
            expected_parent_content_sha256=prepared["money_content_sha256"],
            expected_prepared_receipt_sha256=_sha(values["prepared_receipt"].payload),
            expected_selection_sha256=_sha(values["selection"].payload),
        )
        _require(
            selected.header_data["person_identity_sha256"]
            == document["entities"]["person"]["ordered_ids_sha256"]
            and selected.header_data["household_identity_sha256"]
            == document["entities"]["household"]["ordered_ids_sha256"],
            "ACCOUNTING_MONEY_COORDINATES",
        )
        income = bind_income_observations(
            values["income_observations"].payload, prepared_receipt=prepared
        )
        _require(
            person.source_year.dtype == np.dtype("int64"),
            "ACCOUNTING_SOURCE_YEAR_DTYPE",
        )
        result = derive_reported_income(
            selected,
            income,
            person_ids=person.person_id.to_numpy(),
            income_years=person.source_year.to_numpy(),
            person_positions_sha256=selection["person_positions_sha256"],
            prepared_receipt_sha256=_sha(values["prepared_receipt"].payload),
            selection_sha256=_sha(values["selection"].payload),
            selected_money_sha256=_sha(values["selected_current_money"].payload),
            income_payload_sha256=_sha(values["income_observations"].payload),
            source_producer_key=source_producer,
            selection_producer_key=selection_producer,
            frame_context_sha256=_sha(values["frame_context"].payload),
        )
        index = pd.Index(person.person_id.to_numpy(), name="person_id")
        columns = {
            ("person", name): pd.Series(
                result.array(name), index=index, dtype="float64"
            )
            for name in RESULT_COLUMNS
        }
        document["entities"]["person"]["columns"].extend(RESULT_COLUMNS)
        payload = encode_reported_income(result)
        return KernelResult(
            columns=columns,
            artifacts={
                "reported_income": payload,
                "frame_context": canonical_json(document),
            },
            receipt={
                "phase": bridge.ASEC_PREPARED_PHASE,
                "implementation": implementation_manifest(STAGE),
                "accounting_sha256": _sha(payload),
                "arithmetic_order": ARITHMETIC_ORDER,
                "person_rows": len(person),
                "release_eligible": False,
            },
        )
