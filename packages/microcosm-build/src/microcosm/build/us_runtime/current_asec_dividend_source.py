"""Current ASEC dividend observations and possible survivor-property routes.

Source flags are provenance, never donor filters. No survivor amount, tax
treatment, ACS imputation or new source authority is issued by this module.
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

PROTOCOL = "microcosm.us.current-asec-dividend-source.v1"
AMOUNT_FIELDS = ("DIV_VAL",)
# Width, position, PDF page, printed page, printed universe, zero wording.
RECEIPT_ENTRIES = {
    "DIV_YN": (1, 484, 45, "6C-24", "All Persons aged 15+", "niu"),
    "SUR_YN": (1, 664, 50, "6C-29", "All Persons aged 15+", "niu"),
}
SURVIVOR_ENTRIES = {
    "SUR_SC1": (2, 648, 50, "6C-29", "SUR_YN = 1"),
    "SUR_SC2": (2, 650, 50, "6C-29", "SUR_YN = 1"),
}
SURVIVOR_CODES = {
    0: "none or niu",
    1: "company or union survivor pension",
    2: "federal government",
    3: "US military retirement survivor pension",
    4: "state or local government survivor pension",
    5: "US railroad retirement survivor pension",
    6: "worker compensation survivor",
    7: "black lung",
    8: "regular payments from estates or trusts",
    9: "regular payments from annuities or paid-up life insurance",
    10: "other or don't know",
}
ALLOCATION_ENTRIES = {
    "I_DIVVAL": (1, 819, 54, "6C-33", "DIV_YN = 1", "0:9", "See I_ANNVAL"),
    "I_DIVYN": (1, 820, 54, "6C-33", "All Persons 15+", "0:1", "See I_ANNVAL"),
}
ALLOCATION_CONFLICT_NOTE = (
    "I_DIVYN prints header range0:1 but references I_ANNVAL values0:9. "
    "Both axes are preserved; a code2..9 is not silently admitted or discarded. "
    "Flag values do not alter dividend amount knownness or select donors."
)
TOPCODE_ENTRIES = {
    "TDIV_VAL": (1, 905, 59, "6C-38", "DIV_VAL > 0"),
    "TRNT_VAL": (1, 917, 60, "6C-39", "RNT_VAL > 0"),
}
READ_COLUMNS = (
    *routing.COORDINATE_COLUMNS,
    *AMOUNT_FIELDS,
    *RECEIPT_ENTRIES,
    *SURVIVOR_ENTRIES,
    *ALLOCATION_ENTRIES,
    *TOPCODE_ENTRIES,
)


def require(condition, reason):
    if not condition:
        raise ValueError("ASEC_DIVIDEND_SOURCE_" + reason)


def _decode_amount_entry(data):
    fields = [f for f in data["fields"] if f["name"] == "DIV_VAL"]
    require(len(fields) == 1, "DOMAIN_FIELD_ROSTER")
    field = fields[0]
    vintages = [
        v for v in field["vintages"] if v["income_year"] == routing.CURRENT_INCOME_YEAR
    ]
    require(len(vintages) == 1, "DOMAIN_VINTAGE")
    vintage, domain = vintages[0], field["domain"]
    require(
        vintage["dictionary_spelling"] == "DIV_VAL"
        and vintage["survey_year"] == 2025
        and vintage["pdf_sha256"] == routing.DICTIONARY_SHA256
        and vintage["source_url"] == routing.DICTIONARY_URL,
        "DICTIONARY_PIN",
    )
    require(
        field["entity"] == "person"
        and domain["encoded_range_inclusive"] == {"minimum": 0, "maximum": 999999}
        and domain["valid_dollar_range_inclusive"] == {"minimum": 0, "maximum": 999999}
        and domain["negative_dollars_permitted"] is False
        and domain["declared_negative_nonmoney_codes"] == []
        and domain["declared_other_missing_codes"] == []
        and domain["declared_niu_codes"] == [0]
        and domain["valid_dollar_range_excludes"] == []
        and domain["zero_semantics"]
        == "none_or_niu_not_distinguishable_from_amount_alone",
        "DOMAIN_UNSUPPORTED",
    )
    return (
        vintage["ascii_length_as_printed"],
        vintage["ascii_position_as_printed"],
        vintage["pdf_page_1based"],
        vintage["printed_page"],
        domain["encoded_range_inclusive"]["maximum"],
        vintage["universe_as_printed"],
        domain["zero_semantics"],
    )


@lru_cache(maxsize=1)
def _amount_entry():
    payload = (
        resources.files(__package__).joinpath(routing.DOMAINS_RESOURCE).read_bytes()
    )
    require(routing._sha(payload) == routing.DOMAINS_SHA256, "DOMAINS_HASH")
    # Every cached value is immutable; callers never receive a shared mapping.
    return _decode_amount_entry(json.loads(payload))


def amount_entries():
    return {"DIV_VAL": _amount_entry()}


def _domain_agreement(ready):
    domains = [d for d in ready.bindings.spec.fields if d.name == "DIV_VAL"]
    require(len(domains) == 1, "MONEY_DOMAIN_ROSTER")
    domain, entry = domains[0], amount_entries()["DIV_VAL"]
    require(
        domain.entity == "person"
        and domain.minimum == 0
        and domain.maximum == entry[4]
        and domain.zero_semantics == entry[6],
        "MONEY_DOMAIN_DISAGREEMENT",
    )


def _survivor_route(age, receipt, slots):
    """Clear only resolved non-property source routes; never infer absent amounts."""
    code, status = receipt
    values = [pair[0] for pair in slots]
    readable = all(pair[1] == "in_printed_range" for pair in slots)
    if age < 15:
        if status != "in_printed_range" or not readable:
            return "unresolved_outside_reporting_universe", None
        return (
            "outside_reporting_universe"
            if code == 0 and readable and values == [0, 0]
            else "contradictory_outside_reporting_universe",
            None,
        )
    if status != "in_printed_range":
        return (
            "missing_receipt_literal"
            if status == "missing"
            else "unrecognized_receipt_literal",
            None,
        )
    if code == 0:
        return (
            "niu" if readable and values == [0, 0] else "unresolved_niu_route",
            None,
        )
    if code == 2:
        return (
            ("known_nonreceipt", True)
            if readable and values == [0, 0]
            else ("unresolved_nonreceipt_source_route", None)
        )
    if 8 in values:
        return "possible_estate_or_trust_route", False
    if not readable:
        return "unresolved_source_literal", None
    if 10 in values:
        return "source_type_unspecified", None
    if values == [0, 0]:
        return "unreported_source_slots", None
    return "known_other_survivor_sources", True


def _dividend_allocation_conflict(tokens):
    # Width1 admits only one decimal digit. Parsing physical syntax against
    # 0..9 does not decide which conflicting codebook range is authoritative.
    frame, values, statuses = routing._codes_frame("I_DIVYN", tokens, range(10))
    frame["I_DIVYN_literal_status"] = pd.array(
        [
            "well_formed" if status == "in_printed_range" else status
            for status in statuses
        ],
        dtype="string",
    )
    header = [routing.literal_code(t, (0, 1), width=1)[1] for t in tokens]
    frame["I_DIVYN_header_range_status"] = pd.array(header, dtype="string")
    frame["I_DIVYN_referenced_values_status"] = pd.array(statuses, dtype="string")
    frame["I_DIVYN_codebook_status"] = pd.array(
        [
            "literal_unresolved"
            if status != "in_printed_range"
            else (
                "header_reference_conflict"
                if value not in (0, 1)
                else "header_reference_agree"
            )
            for value, status in zip(values, statuses, strict=True)
        ],
        dtype="string",
    )
    return frame


def project_dividend_literals(ordered):
    """Descriptive source projection; an arbitrary DataFrame grants no authority."""
    require(
        type(ordered) is pd.DataFrame and set(READ_COLUMNS) <= set(ordered), "COLUMNS"
    )
    ages = ordered.A_AGE.to_numpy(dtype=np.int64)
    require(((ages >= 0) & (ages <= 99)).all(), "AGE_RANGE")
    out = pd.DataFrame(index=range(len(ordered)))
    entry = amount_entries()["DIV_VAL"]
    amounts = [
        routing.literal_code(t, range(entry[4] + 1), width=entry[0])
        for t in ordered.DIV_VAL
    ]
    out["DIV_VAL_literal"] = pd.array(ordered.DIV_VAL.tolist(), dtype="string")
    out["DIV_VAL_published_amount"] = pd.array([v for v, _ in amounts], dtype="Float64")
    out["DIV_VAL_literal_status"] = pd.array([s for _, s in amounts], dtype="string")
    codes = {}
    for name, item in (*RECEIPT_ENTRIES.items(), *SURVIVOR_ENTRIES.items()):
        allowed = (
            routing.RECEIPT_CODE_DOMAIN if name in RECEIPT_ENTRIES else SURVIVOR_CODES
        )
        labels = (
            {0: "niu", 1: "yes", 2: "no"} if name in RECEIPT_ENTRIES else SURVIVOR_CODES
        )
        frame, values, statuses = routing._codes_frame(
            name, ordered[name], allowed, labels, width=item[0]
        )
        out = pd.concat([out, frame], axis=1)
        codes[name] = list(zip(values, statuses, strict=True))
    labels = []
    for age, receipt, (value, status) in zip(
        ages, codes["DIV_YN"], amounts, strict=True
    ):
        kind = "missing" if value is None else ("zero" if value == 0 else "nonzero")
        label = routing.receipt_status(
            bool(age >= 15), receipt, kind, net_measure=False, zero_is_dollars=False
        )
        labels.append(
            label
            if status in ("missing", "in_printed_range")
            else "invalid_amount_literal"
        )
    known = np.array([s in routing.KNOWN_AMOUNT_STATUSES for s in labels])
    out["DIV_VAL_reporting_status"] = pd.array(labels, dtype="string")
    out["DIV_VAL_amount_known"] = known
    out["DIV_VAL_amount"] = out.DIV_VAL_published_amount.where(known)
    routes = [
        _survivor_route(age, receipt, (one, two))
        for age, receipt, one, two in zip(
            ages, codes["SUR_YN"], codes["SUR_SC1"], codes["SUR_SC2"], strict=True
        )
    ]
    out["survivor_property_route_status"] = pd.array(
        [s for s, _ in routes], dtype="string"
    )
    out["survivor_property_route_clear"] = pd.array(
        [v for _, v in routes], dtype="boolean"
    )
    out["survivor_estate_or_trust_code_present"] = [
        one[0] == 8 or two[0] == 8
        for one, two in zip(codes["SUR_SC1"], codes["SUR_SC2"], strict=True)
    ]
    out["survivor_unspecified_source_code_present"] = [
        one[0] == 10 or two[0] == 10
        for one, two in zip(codes["SUR_SC1"], codes["SUR_SC2"], strict=True)
    ]
    flags, _, _ = routing._codes_frame(
        "I_DIVVAL", ordered.I_DIVVAL, routing.ALLOCATION_ANNVAL_CODES
    )
    out = pd.concat(
        [out, flags, _dividend_allocation_conflict(ordered.I_DIVYN)], axis=1
    )
    for name, item in TOPCODE_ENTRIES.items():
        flags, _, _ = routing._codes_frame(
            name,
            ordered[name],
            (0, 1),
            {0: "not topcoded", 1: "topcoded"},
            width=item[0],
        )
        out = pd.concat([out, flags], axis=1)
    out["source_age"] = ages
    return out


@dataclass(frozen=True)
class CurrentAsecDividendValues:
    person: pd.DataFrame
    asec_literals: pd.DataFrame
    evidence: dict


def dividend_values_seal(values):
    """Physical value seal, including nullable backing storage and exact bits."""
    require(type(values) is CurrentAsecDividendValues, "VALUES_TYPE")
    digest = hashlib.sha256(PROTOCOL.encode())
    for table in (values.person, values.asec_literals):
        require(type(table) is pd.DataFrame and table.columns.is_unique, "TABLE_TYPE")
        digest.update(
            physical._json(
                {
                    "columns": list(table.columns),
                    "columns_axis": physical.checkpoint._index_spec(
                        table.columns, label="dividend columns"
                    ),
                    "index": physical.checkpoint._index_spec(
                        table.index, label="dividend"
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
    with tempfile.TemporaryDirectory(prefix="microcosm-asec-dividend-") as tmp:
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


def _compare_total(ready, positions, literals):
    field = ready.field("DIV_VAL")
    entry = amount_entries()["DIV_VAL"]
    pairs = [
        routing.literal_code(t, range(entry[4] + 1), width=entry[0]) for t in literals
    ]
    require(
        all(s in ("missing", "in_printed_range") for _, s in pairs), "TOTAL_LITERAL"
    )
    valid = field.validity[positions] == 1
    require(np.array_equal(valid, [v is not None for v, _ in pairs]), "TOTAL_VALIDITY")
    expected = np.array(
        [np.nan if v is None else v for v, _ in pairs], dtype=np.float64
    )
    require(
        np.array_equal(
            field.amounts[positions][valid].view("uint64"),
            expected[valid].view("uint64"),
        ),
        "TOTAL_BITS",
    )
    return field


def qualify_current_asec_dividend(preparation):
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
        header["target_year"] == 2024
        and header["semantic"] == "annual_current_money"
        and native_document["source_year"] == native_document["income_year"] == 2024
        and native_document["survey_year"] == 2025,
        "PERIOD",
    )
    pins = [p for p in routing.coverage._MEMBER_PINS if p[0] == 2024]
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
    positions = np.flatnonzero(np.asarray(parent.scope.person_years) == 2024)
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
    field = _compare_total(ready, positions, ordered.DIV_VAL)
    basis = project_dividend_literals(ordered)
    basis.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    for name in ("statuses", "validity", "zero_origin"):
        basis["DIV_VAL_parent_" + name] = getattr(field, name)[positions]
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
        "receipt_entries": RECEIPT_ENTRIES,
        "survivor_entries": SURVIVOR_ENTRIES,
        "survivor_codes": SURVIVOR_CODES,
        "allocation_entries": ALLOCATION_ENTRIES,
        "allocation_conflict_note": ALLOCATION_CONFLICT_NOTE,
        "topcode_entries": TOPCODE_ENTRIES,
        "preparation_sha256": routing._sha(entry[1]),
        "asec_native_sha256": routing._sha(issued[1]),
        "money_header_sha256": routing._sha(ready.header),
        "source_member_sha256": digest,
        "source_year": 2024,
        "survey_year": 2025,
        "complete_source_rows": rows,
        "selected_rows": len(out),
        "read_columns": READ_COLUMNS,
        "source_flags_used_as_donor_filter": False,
        "survivor_amounts_read": False,
        "survivor_code8_is_possible_property_scope_only": True,
        "survivor_code10_property_clearance_known": False,
        "trnt_flag_universe_evaluated": False,
        "tax_treatment_assigned": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    # Freeze detached JSON-compatible metadata before its seal; do not retain the
    # module's mutable constant dictionaries inside a returned descriptive view.
    result = CurrentAsecDividendValues(out, literals, json.loads(json.dumps(evidence)))
    seal = dividend_values_seal(result)
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
    require(dividend_values_seal(result) == seal, "FINAL_VALUES_CHANGED")
    return result
