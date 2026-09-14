"""Observed child support received, paid and required-to-pay source answers.

Obligation is not payment. Returned values are descriptive and issue no source
authority; a consuming host retains/requalifies the actual preparation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from . import asec_current_money_source as physical
from . import current_asec_income_routing_source as routing

PROTOCOL = "microcosm.us.current-asec-child-support-source.v1"
AMOUNT_FIELDS = ("CSP_VAL", "CHSP_VAL")
# Width, position, PDF page, printed page, printed universe, zero label.
RESPONSE_ENTRIES = {
    "CHELSEW_YN": (1, 711, 52, "6C-31", "All Persons aged 15+", "Niu"),
    "CHSP_YN": (1, 717, 52, "6C-31", "CHELSEW_YN", "Niu"),
    "CSP_YN": (1, 723, 52, "6C-31", "All Persons aged 15+", "Niu"),
}
RESPONSE_MEANINGS = {
    "CHELSEW_YN": "has a child living outside the household",
    "CHSP_YN": "required to pay child support; not whether payment occurred",
    "CSP_YN": "received child support payments",
}
ALLOCATION_ENTRIES = {
    "I_CHELSEWYN": (1, 807, 54, "6C-33", "CHELSEW_YN > 0"),
    "I_CHSPVAL": (1, 808, 54, "6C-33", "CHSP_YN = 1"),
    "I_CHSPYN": (1, 809, 54, "6C-33", "CHELSEW_YN = 1"),
    "I_CSPVAL": (1, 810, 54, "6C-33", "CSP_YN = 1"),
    "I_CSPYN": (1, 811, 54, "6C-33", "CSP_YN > 0"),
}
TOPCODE_ENTRIES = {
    "TCHSP_VAL": (1, 899, 59, "6C-38", "CHSP_VAL > 0"),
    "TCSP_VAL": (1, 901, 59, "6C-38", "CSP_VAL > 0"),
}
READ_COLUMNS = (
    *routing.COORDINATE_COLUMNS,
    *AMOUNT_FIELDS,
    *RESPONSE_ENTRIES,
    *ALLOCATION_ENTRIES,
    *TOPCODE_ENTRIES,
)
OBLIGATION_UNIVERSE_NOTE = (
    "CHSP_YN prints bare CHELSEW_YN, whereas I_CHSPYN prints CHELSEW_YN = 1. "
    "The bare field is not silently compiled as a general predicate. Positive "
    "paid amounts are qualified only on the observed intersection of age15+, "
    "CHELSEW_YN=1 and CHSP_YN=1; other routes remain explicit and unresolved."
)


def require(condition, reason):
    if not condition:
        raise ValueError("ASEC_CHILD_SUPPORT_SOURCE_" + reason)


@lru_cache(maxsize=1)
def _cached_amount_entries_json():
    payload = (
        resources.files(__package__).joinpath(routing.DOMAINS_RESOURCE).read_bytes()
    )
    require(routing._sha(payload) == routing.DOMAINS_SHA256, "DOMAINS_HASH")
    entries = {}
    for field in json.loads(payload)["fields"]:
        if field["name"] not in AMOUNT_FIELDS:
            continue
        require(field["name"] not in entries, "DOMAIN_FIELD_DUPLICATE:" + field["name"])
        vintages = [
            v
            for v in field["vintages"]
            if v["income_year"] == routing.CURRENT_INCOME_YEAR
        ]
        require(len(vintages) == 1, "DOMAIN_VINTAGE:" + field["name"])
        vintage = vintages[0]
        require(
            vintage["dictionary_spelling"] == field["name"]
            and vintage["pdf_sha256"] == routing.DICTIONARY_SHA256
            and vintage["source_url"] == routing.DICTIONARY_URL,
            "DICTIONARY_PIN",
        )
        routing._require_nonnegative_amount_domain(
            field,
            vintage,
            zero_semantics="niu"
            if field["name"] == "CHSP_VAL"
            else "none_or_niu_not_distinguishable_from_amount_alone",
            valid_minimum=1 if field["name"] == "CHSP_VAL" else 0,
        )
        entries[field["name"]] = {
            "width": vintage["ascii_length_as_printed"],
            "position": vintage["ascii_position_as_printed"],
            "pdf_page": vintage["pdf_page_1based"],
            "printed_page": vintage["printed_page"],
            "universe": vintage["universe_as_printed"],
            "domain": field["domain"],
            "period": field["period"],
            "temporal_authority": field["temporal_authority"],
        }
    require(set(entries) == set(AMOUNT_FIELDS), "AMOUNT_ROSTER")
    return json.dumps(entries, separators=(",", ":")).encode()


def amount_entries():
    """Return deeply detached evidence from privately cached immutable bytes."""
    return json.loads(_cached_amount_entries_json())


amount_entries.cache_clear = _cached_amount_entries_json.cache_clear


def _domain_agreement(ready):
    routing._require_live_amount_domains(
        ready,
        {
            name: (
                entry["domain"]["encoded_range_inclusive"]["minimum"],
                entry["domain"]["encoded_range_inclusive"]["maximum"],
                entry["domain"]["zero_semantics"],
            )
            for name, entry in amount_entries().items()
        },
    )


def _amount_pair(token, entry):
    bounds = entry["domain"]["encoded_range_inclusive"]
    return routing.literal_code(
        token, range(bounds["minimum"], bounds["maximum"] + 1), width=entry["width"]
    )


def _paid_status(age, elsewhere, obligation, amount):
    value, parse_status = amount
    if parse_status != "in_printed_range":
        return (
            "missing_amount" if parse_status == "missing" else "invalid_amount_literal"
        )
    if age < 15:
        return (
            "outside_reporting_universe"
            if elsewhere[0] == obligation[0] == 0 and value == 0
            else "contradictory_outside_reporting_universe"
        )
    if value == 0:
        return "declared_niu_amount"
    if elsewhere[1] != "in_printed_range":
        return "unresolved_child_elsewhere_literal"
    if obligation[1] != "in_printed_range":
        return (
            "missing_obligation_literal"
            if obligation[1] == "missing"
            else "invalid_obligation_literal"
        )
    if elsewhere[0] != 1:
        return "unresolved_child_elsewhere_route"
    if obligation[0] != 1:
        return "payment_outside_published_obligation_universe"
    return "observed_positive_payment"


def project_child_support_literals(ordered):
    """Descriptive projection only; arbitrary literals grant no authority."""
    require(
        type(ordered) is pd.DataFrame and set(READ_COLUMNS) <= set(ordered), "COLUMNS"
    )
    ages = ordered.A_AGE.to_numpy(dtype=np.int64)
    require(((ages >= 0) & (ages <= 99)).all(), "AGE_RANGE")
    out = pd.DataFrame(index=range(len(ordered)))
    parsed, codes = {}, {}
    for name, entry in amount_entries().items():
        pairs = [_amount_pair(token, entry) for token in ordered[name]]
        parsed[name] = pairs
        out[name + "_literal"] = pd.array(ordered[name].tolist(), dtype="string")
        out[name + "_literal_status"] = pd.array(
            [status for _, status in pairs], dtype="string"
        )
        out[name + "_published_amount"] = pd.array(
            [value for value, _ in pairs], dtype="Float64"
        )
    for name, entry in RESPONSE_ENTRIES.items():
        frame, values, statuses = routing._codes_frame(
            name,
            ordered[name],
            routing.RECEIPT_CODE_DOMAIN,
            {0: entry[5], 1: "Yes", 2: "No"},
            width=entry[0],
        )
        out = pd.concat([out, frame], axis=1)
        codes[name] = list(zip(values, statuses, strict=True))
    for name in AMOUNT_FIELDS:
        labels = []
        for i, age in enumerate(ages):
            value, status = parsed[name][i]
            if name == "CHSP_VAL":
                label = _paid_status(
                    age, codes["CHELSEW_YN"][i], codes["CHSP_YN"][i], parsed[name][i]
                )
            elif status not in ("missing", "in_printed_range"):
                label = "invalid_amount_literal"
            else:
                kind = (
                    "missing"
                    if value is None
                    else ("zero" if value == 0 else "nonzero")
                )
                label = routing.receipt_status(
                    bool(age >= 15),
                    codes["CSP_YN"][i],
                    kind,
                    net_measure=False,
                    zero_is_dollars=False,
                )
            labels.append(label)
        known = np.array(
            [
                label in (*routing.KNOWN_AMOUNT_STATUSES, "observed_positive_payment")
                for label in labels
            ]
        )
        out[name + "_reporting_status"] = pd.array(labels, dtype="string")
        out[name + "_amount_known"] = known
        out[name + "_amount"] = out[name + "_published_amount"].where(known)
    flags = {}
    for name, entry in ALLOCATION_ENTRIES.items():
        frame, values, statuses = routing._codes_frame(
            name, ordered[name], routing.ALLOCATION_ANNVAL_CODES, width=entry[0]
        )
        out = pd.concat([out, frame], axis=1)
        flags[name] = (values, statuses)
    out["received_allocation_origin"] = routing._allocation_origin(
        {name: flags[name] for name in ("I_CSPVAL", "I_CSPYN")}, unflagged=False
    )
    out["paid_allocation_origin"] = routing._allocation_origin(
        {name: flags[name] for name in ("I_CHELSEWYN", "I_CHSPVAL", "I_CHSPYN")},
        unflagged=False,
    )
    for name, entry in TOPCODE_ENTRIES.items():
        frame, _, _ = routing._codes_frame(name, ordered[name], (0, 1), width=entry[0])
        out = pd.concat([out, frame], axis=1)
    out["source_age"] = ages
    out["obligation_implies_payment"] = False
    out["voluntary_payment_absence_known"] = False
    return out


@dataclass(frozen=True)
class CurrentAsecChildSupportValues:
    person: pd.DataFrame
    asec_literals: pd.DataFrame
    evidence: dict


def child_support_values_seal(values):
    """Physical value seal, including nullable backing storage and exact bits."""
    require(type(values) is CurrentAsecChildSupportValues, "VALUES_TYPE")
    digest = hashlib.sha256(PROTOCOL.encode())
    for table in (values.person, values.asec_literals):
        require(type(table) is pd.DataFrame and table.columns.is_unique, "TABLE_TYPE")
        digest.update(
            physical._json(
                {
                    "columns": list(table.columns),
                    "columns_axis": physical.checkpoint._index_spec(
                        table.columns, label="child support columns"
                    ),
                    "index": physical.checkpoint._index_spec(
                        table.index, label="child support"
                    ),
                }
            )
        )
        physical._series_digest(
            digest, pd.Series(table.index.to_numpy(copy=False), dtype=table.index.dtype)
        )
        for column in table:
            series = table[column]
            if isinstance(series.dtype, pd.Float64Dtype):
                # The checkpoint digest supports NumPy floats, but deliberately
                # excludes Float64 extension arrays. Keep the nullable tag and
                # hash both physical arrays, including values under null masks.
                digest.update(physical._json({"dtype": "Float64", "nullable": True}))
                physical._series_digest(digest, pd.Series(series.array._data))
                physical._series_digest(digest, pd.Series(series.array._mask))
            else:
                physical._series_digest(digest, series)
    digest.update(physical._json(values.evidence))
    return digest.hexdigest()


def _capture_member(root, pin):
    _, member, _, digest, rows, size = pin
    with tempfile.TemporaryDirectory(prefix="microcosm-asec-child-support-") as tmp:
        path = Path(tmp) / member
        identity = routing.coverage._capture(
            root / "asec" / member,
            path,
            size=size,
            digest=digest,
            budget=[routing.coverage._BODY_MAX],
        )
        raw = routing._read_capture(path, rows=rows, columns=READ_COLUMNS, patterns={})
        require(
            routing.coverage._identity(path.stat(follow_symlinks=False)) == identity
            and routing._file_sha(path) == digest,
            "CAPTURE_CHANGED",
        )
    return raw


def _compare_amount(ready, positions, name, literals):
    field = ready.field(name)
    entry = amount_entries()[name]
    pairs = [_amount_pair(t, entry) for t in literals]
    require(
        all(s in ("missing", "in_printed_range") for _, s in pairs), "AMOUNT_LITERAL"
    )
    valid = field.validity[positions] == 1
    require(np.array_equal(valid, [v is not None for v, _ in pairs]), "AMOUNT_VALIDITY")
    expected = np.array(
        [np.nan if v is None else v for v, _ in pairs], dtype=np.float64
    )
    require(
        np.array_equal(
            field.amounts[positions][valid].view("uint64"),
            expected[valid].view("uint64"),
        ),
        "AMOUNT_BITS",
    )
    return field


def qualify_current_asec_child_support(preparation):
    """Borrow the original preparation, capture once, and requalify before return."""
    source = routing.source
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state, native = entry[2], entry[2].native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "NATIVE_ISSUANCE",
    )
    parent = issued[2].parent
    ready = parent.ready()
    _domain_agreement(ready)
    header, native_document = json.loads(ready.header), json.loads(issued[1])
    require(
        header["target_year"] == routing.CURRENT_INCOME_YEAR
        and header["semantic"] == "annual_current_money"
        and native_document["source_year"]
        == native_document["income_year"]
        == routing.CURRENT_INCOME_YEAR
        and native_document["survey_year"] == (routing.CURRENT_INCOME_YEAR + 1),
        "PERIOD",
    )
    pins = [
        p for p in routing.coverage._MEMBER_PINS if p[0] == routing.CURRENT_INCOME_YEAR
    ]
    require(len(pins) == 1, "SOURCE_REGISTRY")
    year, member, archive, digest, rows, size = pins[0]
    retained = [
        s for s in issued[2].coverage.receipt["sources"] if s["source_year"] == year
    ]
    require(
        len(retained) == 1
        and all(
            retained[0][k] == v
            for k, v in (
                ("member", member),
                ("archive_sha256", archive),
                ("member_sha256", digest),
                ("rows", rows),
                ("member_bytes", size),
            )
        ),
        "MEMBER_BINDING",
    )
    raw = _capture_member(state.root, pins[0])
    positions = np.flatnonzero(
        np.asarray(parent.scope.person_years) == routing.CURRENT_INCOME_YEAR
    )
    keys = np.asarray(parent.scope.person_native_keys)[positions]
    require(
        len(keys) == rows and len(set(keys)) == rows and set(keys) == set(raw.index),
        "COMPLETE_SOURCE_JOIN",
    )
    ordered = raw.loc[keys]
    for raw_name, parent_name in (
        ("PH_SEQ", "source_household_id"),
        ("A_LINENO", "A_LINENO"),
        ("A_AGE", "A_AGE"),
    ):
        require(
            np.array_equal(
                ordered[raw_name].astype("int64").to_numpy(),
                parent.frame.person.iloc[positions][parent_name].to_numpy(),
            ),
            "PARENT_COORDINATE",
        )
    fields = {
        name: _compare_amount(ready, positions, name, ordered[name])
        for name in AMOUNT_FIELDS
    }
    basis = project_child_support_literals(ordered)
    basis.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    for field_name, field in fields.items():
        for name in ("statuses", "validity", "zero_origin"):
            basis[field_name + "_parent_" + name] = getattr(field, name)[positions]
    selected = state.frame.person.loc[
        state.frame.person[routing.support_channel_column("person")].eq("asec")
    ]
    native_ids = selected[routing.spine_source_id_column("person")].to_numpy()
    require(
        len(set(native_ids)) == len(native_ids) and set(native_ids) <= set(basis.index),
        "SELECTED_NATIVE_JOIN",
    )
    out = basis.loc[native_ids].copy()
    out["native_person_id"] = native_ids
    out.index = pd.Index(selected.person_id.to_numpy(), name="person_id")
    literals = ordered.copy()
    literals.index = basis.index.copy()
    evidence = {
        "protocol": PROTOCOL,
        "dictionary_url": routing.DICTIONARY_URL,
        "dictionary_sha256": routing.DICTIONARY_SHA256,
        "amount_entries": amount_entries(),
        "response_entries": RESPONSE_ENTRIES,
        "response_meanings": RESPONSE_MEANINGS,
        "obligation_universe_note": OBLIGATION_UNIVERSE_NOTE,
        "response_period_note": (
            "CHSP_YN and CHELSEW_YN do not state an independent reference year "
            "in their printed entries. Their 2025 source context is preserved; "
            "no new current-at-interview or policy eligibility input is issued."
        ),
        "allocation_entries": ALLOCATION_ENTRIES,
        "topcode_entries": TOPCODE_ENTRIES,
        "preparation_sha256": routing._sha(entry[1]),
        "asec_native_sha256": routing._sha(issued[1]),
        "money_header_sha256": routing._sha(ready.header),
        "source_member_sha256": digest,
        "source_year": routing.CURRENT_INCOME_YEAR,
        "survey_year": (routing.CURRENT_INCOME_YEAR + 1),
        "complete_source_rows": rows,
        "selected_rows": len(out),
        "read_columns": READ_COLUMNS,
        "obligation_implies_payment": False,
        "voluntary_payment_absence_known": False,
        "paid_niu_completed_with_zero": False,
        "tax_treatment_assigned": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    # Freeze detached JSON-compatible metadata before its seal; do not retain the
    # module's mutable constant dictionaries inside a returned descriptive view.
    result = CurrentAsecChildSupportValues(
        out, literals, json.loads(json.dumps(evidence))
    )
    seal = child_support_values_seal(result)
    require(
        preparation._checked() is entry and parent.ready().header == ready.header,
        "SOURCE_CHANGED",
    )
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued
        and issued[2].parent is parent
        and native.payload == issued[1],
        "FINAL_OWNER",
    )
    require(child_support_values_seal(result) == seal, "FINAL_VALUES_CHANGED")
    return result
