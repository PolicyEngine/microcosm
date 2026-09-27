"""Fixture-only PUF price-result import and combined-host feature placement.

The local envelope is transport, not scientific admission. A caller must bind
its exact producer result and source/price identities before using its arrays.
Genuine source definitions remain refused. No amount is repaired or allocated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit.model_input import encode_recipient_matrix
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights

from . import native_household_origin as origin
from . import puf_growth as growth
from . import puf_monetary_agi_projection as agi
from . import puf_raw_source as raw
from . import support_provenance as provenance

TARGET = "E00900"
OUTPUT = "dev_puf_schedule_c_profit_loss_2024_price"
MASK = "dev_puf_detail_recipient"
FEATURES = ("reported_wage_proxy", "mars_1", "mars_2", "mars_3", "mars_4")
SCOPE = "invented_fixture_machinery_only"
SOURCE_ADMISSION = "not_admitted_pending_basis_review"
MAGIC = b"local-puf-price-restatement-arrays-v1\n"
MAX_ROWS = 4096
MAX_BYTES = 128 * 1024 * 1024
MONEY_FIELDS = (
    "E00100",
    "E00200",
    "E00300",
    "E00400",
    "E00600",
    "E00650",
    "E00900",
    "E01000",
    "E01500",
    "E01700",
    "E02100",
    "P22250",
    "P23250",
)
ARRAY_DTYPES = {
    **{name: "<f8" for name in MONEY_FIELDS},
    "RECID": "<i8",
    growth.PROVENANCE_RECID_COLUMN: "<i8",
    growth.PROVENANCE_SOURCE_AGI_COLUMN: "<f8",
    "S006": "<f8",
    "FLPDYR": "<i4",
    "demographic_status": "<i4",
    "S006_delivered_int64": "<i8",
    "FLPDYR_delivered_int16": "<i2",
    "demographic_status_delivered_int8": "|i1",
}
HEADER_FIELDS = {
    "schema",
    "input_binding_sha256",
    "contract_sha256",
    "source_head",
    "helper_head",
    "release_eligible",
}
MARS = {
    "SINGLE": 1,
    "JOINT": 2,
    "SURVIVING_SPOUSE": 2,
    "SEPARATE": 3,
    "HEAD_OF_HOUSEHOLD": 4,
}


def require(condition: bool, reason: str) -> None:
    """Refuse using a stable reason, without reporting source row values."""
    if not condition:
        raise ValueError(reason)


def fixture_sources(definition_text: str, projection_text: str):
    """Validate actual fixture definitions; packaged source pins cannot enter."""
    # Source documents and fit packets use distinct JSON canonical conventions. Validate each document against its own producer's exact bytes.
    definition = raw.fixture_definition(json.loads(definition_text))
    projection = agi.fixture_agi_projection_document(
        json.loads(projection_text), definition
    )
    require(
        definition.params_text == definition_text
        and projection.params_text == projection_text,
        "FIXTURE_SOURCE_CANONICAL",
    )
    require(tuple(sorted(projection.fields)) == MONEY_FIELDS, "FIXTURE_MONEY_FIELDS")
    require(
        all(c.fit_admission == SOURCE_ADMISSION for c in projection.columns),
        "SOURCE_ADMISSION_CHANGED",
    )
    return definition, projection


def decode_price_arrays(
    payload: bytes, *, expected: Mapping, expected_sha256: str, max_rows=MAX_ROWS
):
    """Read the exact reviewed local helper envelope, with external identity pins.

    This parser does not create a GrownPufTable, repeat growth, or admit a fit.
    The returned arrays are read-only views of the authenticated body bytes.
    """
    require(type(max_rows) is int and 0 < max_rows <= 250_000, "PRICE_ROW_BOUND")
    require(
        isinstance(payload, bytes)
        and len(payload) <= MAX_BYTES
        and payload.startswith(MAGIC),
        "PRICE_ENVELOPE",
    )
    require(
        codec._hash(expected_sha256) and codec.sha(payload) == expected_sha256,
        "PRICE_PAYLOAD_SHA",
    )
    offset = len(MAGIC)
    require(len(payload) >= offset + 8, "PRICE_HEADER_LENGTH")
    size = int.from_bytes(payload[offset : offset + 8], "big")
    start = offset + 8
    require(0 < size <= 65536 and len(payload) >= start + size, "PRICE_HEADER_BOUND")
    header = codec.decode_json(payload[start : start + size])
    require(
        set(expected) == HEADER_FIELDS and set(header) == HEADER_FIELDS | {"columns"},
        "PRICE_HEADER_FIELDS",
    )
    require({k: header[k] for k in HEADER_FIELDS} == dict(expected), "PRICE_IDENTITY")
    require(
        header["schema"] == "local.puf.price_restatement.arrays.v1"
        and header["release_eligible"] is False,
        "PRICE_SCOPE",
    )
    for field in ("input_binding_sha256", "contract_sha256"):
        require(codec._hash(header[field]), "PRICE_HEADER_SHA")
    for field in ("source_head", "helper_head"):
        require(
            isinstance(header[field], str)
            and len(header[field]) == 40
            and all(c in "0123456789abcdef" for c in header[field]),
            "PRICE_CODE_ID",
        )
    columns = header["columns"]
    require(
        isinstance(columns, dict) and set(columns) == set(ARRAY_DTYPES),
        "PRICE_COLUMN_ROSTER",
    )
    cursor, rows, result = start + size, None, {}
    for name in sorted(ARRAY_DTYPES):
        entry = columns[name]
        require(
            isinstance(entry, dict) and set(entry) == {"dtype", "rows", "sha256"},
            "PRICE_COLUMN_FIELDS",
        )
        require(
            entry["dtype"] == ARRAY_DTYPES[name]
            and type(entry["rows"]) is int
            and 0 < entry["rows"] <= max_rows,
            "PRICE_COLUMN_TYPE_ROWS",
        )
        rows = entry["rows"] if rows is None else rows
        require(entry["rows"] == rows, "PRICE_ROW_ALIGNMENT")
        end = cursor + rows * np.dtype(entry["dtype"]).itemsize
        require(end <= len(payload), "PRICE_BODY_TRUNCATED")
        body = payload[cursor:end]
        require(codec.sha(body) == entry["sha256"], "PRICE_BODY_SHA")
        values = np.frombuffer(body, dtype=entry["dtype"])
        require(np.isfinite(values).all(), "PRICE_NONFINITE")
        result[name] = values
        cursor = end
    require(cursor == len(payload), "PRICE_TRAILING_BYTES")
    return result


def donor_frame(arrays: Mapping, status, decoded, projection) -> Frame:
    """One computational person/return wrapper, with explicit raw S006 DESIGN weights."""
    require(projection.route == "test_fixture", "GENUINE_FIT_NOT_ADMITTED")
    return _donor_frame(arrays, status, decoded, projection, scope=SCOPE)


def _donor_frame(arrays, status, decoded, projection, *, scope):
    """Shared mechanical checks; the entry point must qualify its source/recipe."""
    known = np.asarray(decoded.amount_known, dtype=bool)
    require(
        np.array_equal(status.typed["RECID"], decoded.typed["RECID"]),
        "DONOR_STATUS_JOIN",
    )
    require(
        np.array_equal(status.typed["disclosure_aggregate"], ~known), "DONOR_UNIVERSE"
    )
    ids = status.typed["RECID"][known]
    require(
        len(ids) == len(np.unique(ids))
        and np.array_equal(ids, arrays["RECID"])
        and np.array_equal(ids, arrays[growth.PROVENANCE_RECID_COLUMN]),
        "DONOR_RECID_ORDER",
    )
    aggregate_ids = status.typed["RECID"][~known]
    require(
        len(aggregate_ids) == 4
        and len(np.unique(aggregate_ids)) == 4
        and set(aggregate_ids) == set(raw.PUF_AGGREGATE_RECIDS),
        "DONOR_AGGREGATE_ROSTER",
    )
    for field, delivered in (
        ("S006", "S006_delivered_int64"),
        ("FLPDYR", "FLPDYR_delivered_int16"),
        ("demographic_status", "demographic_status_delivered_int8"),
    ):
        original = status.typed[field][known]
        require(
            np.array_equal(original, arrays[delivered])
            and np.array_equal(original, arrays[field]),
            "DONOR_CARRIED_STATUS",
        )
    require(
        np.array_equal(
            decoded.typed["E00100"][known], arrays[growth.PROVENANCE_SOURCE_AGI_COLUMN]
        ),
        "DONOR_SOURCE_AGI",
    )
    weights = arrays["S006_delivered_int64"].astype("float64")
    require(
        np.all(arrays["S006_delivered_int64"] >= 0)
        and np.all(arrays["S006_delivered_int64"] <= 2**53)
        and np.array_equal(weights.astype("int64"), arrays["S006_delivered_int64"])
        and weights.sum() > 0,
        "DONOR_DESIGN_WEIGHTS",
    )
    wage = arrays["E00200"]
    require(np.isfinite(wage).all() and np.all(wage >= 0), "DONOR_WAGE_FEATURE")
    mars = status.typed["MARS"][known]
    require(np.isin(mars, (1, 2, 3, 4)).all(), "DONOR_MARS")
    table = pd.DataFrame(
        {
            "tax_unit_id": ids.copy(),
            FEATURES[0]: wage.copy(),
            **{f"mars_{i}": (mars == i).astype("float64") for i in range(1, 5)},
            TARGET: arrays[TARGET].copy(),
        }
    )
    return Frame(
        {
            "tax_unit": table,
            "person": pd.DataFrame(
                {"person_id": ids.copy(), "person_tax_unit_id": ids.copy()}
            ),
        },
        EntitySchema(group_entities=("tax_unit",)),
        {"tax_unit": Weights(weights, WeightKind.DESIGN)},
        metadata={
            "scope": scope,
            "source_projection_sha256": projection.sha256,
            "source_projection_document": projection.canonical.decode("utf-8"),
            "carrier": "one technical person per return; not a taxpayer count",
            "weight_units": "delivered_S006_integer_hundredths",
        },
    )


def population_content(frame: Frame) -> str:
    """Full materialized cells/axes/effective weights/strata; no metadata shortcuts."""
    return origin._population_content(frame)


def recipient_matrix(frame: Frame, *, role_column=None, person_wages=None):
    """Use every detail unit, including zero-weight rows, and actual filing roles."""
    require(frame.schema == US_SCHEMA, "HOST_SCHEMA")
    # Build a manifest from the current native roster only for validation. The
    # separate fixture source edge binds that roster's authority and full bytes.
    tables = {e: frame.table(e) for e in frame.entities}
    for entity, table in tables.items():
        role = table[provenance.support_clone_index_column(entity)]
        require(
            role.dtype == np.dtype("int64") and role.isin((0, 1)).all(),
            "HOST_CLONE_TYPE",
        )
    manifest = provenance.spine_assembly_manifest(
        {
            e: t.loc[t[provenance.support_clone_index_column(e)].eq(0)]
            for e, t in tables.items()
        },
        channels=("acs", "asec"),
    )
    checked = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        metadata=manifest,
    )
    provenance.validate_assembly_provenance(checked, boundary="invented_puf_detail")
    person, units = tables["person"], tables["tax_unit"]
    if person_wages is not None:
        require(
            type(person_wages) is pd.Series
            and person_wages.dtype == np.dtype("float64")
            and person_wages.index.dtype == np.dtype("int64")
            and person_wages.index.is_unique
            and np.array_equal(person_wages.index.to_numpy(), person.person_id),
            "HOST_WAGE_ROW_AXIS",
        )
        clone_role = person[provenance.support_clone_index_column("person")]
        wage_values = person_wages.to_numpy()
        require(
            np.array_equal(
                wage_values[clone_role.eq(0)].view("uint64"),
                wage_values[clone_role.eq(1)].view("uint64"),
            ),
            "HOST_WAGE_PAIR_BYTES",
        )
    pairs = {}
    for entity, table in tables.items():
        src, role = (
            provenance.support_source_id_column(entity),
            provenance.support_clone_index_column(entity),
        )
        require(table[src].dtype == np.dtype("int64"), "HOST_SOURCE_ID_TYPE")
        require(not table.duplicated([src, role]).any(), "HOST_ROLE_PAIR_DUPLICATE")
        native, detail = table.loc[table[role].eq(0)], table.loc[table[role].eq(1)]
        require(
            len(native) == len(detail) and np.array_equal(native[src], detail[src]),
            "HOST_ROLE_PAIR_COVERAGE",
        )
        structural = {US_SCHEMA.entity_id_column(entity), role}
        if entity == "person":
            structural.update(
                US_SCHEMA.membership_column(e) for e in US_SCHEMA.group_entities
            )
        for name in table.columns:
            if name not in structural:
                require(
                    native[name]
                    .reset_index(drop=True)
                    .equals(detail[name].reset_index(drop=True)),
                    "HOST_COPIED_SOURCE_CELL",
                )
        weights = frame.resolve_weights(entity).values
        require(
            np.array_equal(
                weights[table[role].eq(0)].view("uint64"),
                weights[table[role].eq(1)].view("uint64"),
            ),
            "HOST_PAIR_WEIGHTS",
        )
        pairs[entity] = dict(
            zip(
                table[US_SCHEMA.entity_id_column(entity)],
                zip(table[src], table[role], strict=True),
                strict=True,
            )
        )
    for group in US_SCHEMA.group_entities:
        member = US_SCHEMA.membership_column(group)
        for pid, gid in zip(person.person_id, person[member], strict=True):
            require(pairs["person"][pid][1] == pairs[group][gid][1], "HOST_MEMBER_ROLE")
        by_source = {}
        for pid, gid in zip(person.person_id, person[member], strict=True):
            source, role = pairs["person"][pid]
            group_source = pairs[group][gid][0]
            if role == 0:
                by_source[source] = group_source
            else:
                require(by_source.get(source) == group_source, "HOST_CLONE_MEMBERSHIP")
    require(
        set(units[provenance.support_channel_column("tax_unit")]) == {"acs", "asec"},
        "HOST_BOTH_ARMS",
    )
    mask = units[provenance.support_clone_index_column("tax_unit")].eq(1).to_numpy()
    selected = units.loc[mask]
    values = []
    for row in selected.itertuples(index=False):
        members = person.loc[person.person_tax_unit_id.eq(row.tax_unit_id)]
        status = row.filing_status_input
        require(status in MARS, "HOST_FILING_STATUS")
        if role_column is None:
            for name in (
                "is_tax_unit_head",
                "is_tax_unit_spouse",
                "is_tax_unit_dependent",
            ):
                require(
                    name in members
                    and members[name].dtype == np.dtype("bool")
                    and not members[name].isna().any(),
                    "HOST_FILING_ROLE",
                )
            head, spouse, dependent = (
                members[name].to_numpy()
                for name in (
                    "is_tax_unit_head",
                    "is_tax_unit_spouse",
                    "is_tax_unit_dependent",
                )
            )
        else:
            require(
                role_column == "tax_unit_role_input" and role_column in members,
                "HOST_ROLE_INPUT_COLUMN",
            )
            roles = members[role_column]
            require(
                pd.api.types.is_string_dtype(roles.dtype)
                and not roles.isna().any()
                and roles.isin(("HEAD", "SPOUSE", "DEPENDENT")).all(),
                "HOST_ROLE_INPUT_VALUES",
            )
            head, spouse, dependent = (
                roles.eq(name).to_numpy(dtype=bool)
                for name in ("HEAD", "SPOUSE", "DEPENDENT")
            )
        require(
            np.all(head.astype(int) + spouse.astype(int) + dependent.astype(int) == 1)
            and head.sum() == 1
            and spouse.sum() == int(status == "JOINT"),
            "HOST_FILING_STRUCTURE",
        )
        channel = getattr(row, provenance.support_channel_column("tax_unit"))
        name = (
            "asec_reported_wage_income_2024_price"
            if channel == "asec"
            else "employment_income_before_lsr"
        )
        earners = head | spouse
        if person_wages is None:
            require(name in members, "HOST_MISSING_FEATURE")
            wage = members.loc[earners, name]
        else:
            wage = person_wages.loc[members.loc[earners, "person_id"]]
        require(
            pd.api.types.is_numeric_dtype(wage)
            and not pd.api.types.is_bool_dtype(wage)
            and not pd.api.types.is_complex_dtype(wage)
            and not wage.isna().any(),
            "HOST_MISSING_FEATURE",
        )
        a = wage.to_numpy(dtype="float64")
        require(np.isfinite(a).all() and (a >= 0).all(), "HOST_NONFINITE_FEATURE")
        amount = float(a.sum())
        require(np.isfinite(amount), "HOST_FEATURE_OVERFLOW")
        values.append([amount, *(float(MARS[status] == i) for i in range(1, 5))])
    ids = selected.tax_unit_id.to_numpy(copy=True)
    features = pd.DataFrame(
        values, index=pd.Index(ids), columns=FEATURES, dtype="float64"
    )
    return encode_recipient_matrix(features, entity="tax_unit", entity_ids=ids), mask
