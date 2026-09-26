"""Source observations of ordinary and retirement-account ASEC interest.

No tax treatment, model, component balancing or source issuer lives here.
The consuming host retains and requalifies the actual preparation around I/O.
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

PROTOCOL = "microcosm.us.current-asec-interest-source.v1"
AMOUNT_FIELDS = ("INT_VAL", "TRDINT_VAL", "RINT_VAL1", "RINT_VAL2")
# Width, position, PDF page, printed page, maximum, universe, zero wording.
# INT_VAL is read from the already pinned current-money domain resource.
ADDITIONAL_AMOUNT_ENTRIES = {
    "TRDINT_VAL": (5, 665, 50, "6C-29", 99999, "INT_YN = 1", "dollar value"),
    "RINT_VAL1": (6, 608, 49, "6C-28", 999999, "RINT_SC1 > 0", "none or niu"),
    "RINT_VAL2": (6, 614, 49, "6C-28", 999999, "RINT_SC2 > 0", "none or niu"),
}
RECEIPT_ENTRIES = {
    "INT_YN": (1, 543, 46, "6C-25", "All Persons aged 15+", "niu"),
    "RINT_YN": (1, 620, 49, "6C-28", "All Persons aged 15+", "niu"),
}
ACCOUNT_ENTRIES = {
    "RINT_SC1": (1, 606, 49, "6C-28", "RINT_YN = 1"),
    "RINT_SC2": (1, 607, 49, "6C-28", "RINT_YN = 1"),
}
# These published component and composite flags have different code systems.
ALLOCATION_ENTRIES = {
    "I_INTVAL": (2, 838, 56, "6C-35", "INT_VAL> 0", (0, 11, 12, 13, 14, 15)),
    "I_INTYN": (2, 840, 56, "6C-35", "INT_YN > 0", routing.ALLOCATION_COMPOSITE_CODES),
    "I_RINTSC": (1, 857, 57, "6C-36", "RINT_SC1 > 0", routing.ALLOCATION_ANNVAL_CODES),
    "I_RINTVAL1": (
        1,
        858,
        57,
        "6C-36",
        "RINT_VAL1 > 0",
        routing.ALLOCATION_ANNVAL_CODES,
    ),
    "I_RINTVAL2": (
        1,
        859,
        57,
        "6C-36",
        "RINT_VAL2 > 0",
        routing.ALLOCATION_ANNVAL_CODES,
    ),
    "I_RINTYN": (1, 860, 57, "6C-36", "RINT_YN > 0", routing.ALLOCATION_ANNVAL_CODES),
}
TOPCODE_ENTRIES = {
    "TRINT_VAL1": (1, 915, 60, "6C-39", "RINT_VAL1 > 0"),
    "TRINT_VAL2": (1, 916, 60, "6C-39", "RINT_VAL2 > 0"),
    "TTRDINT_VAL": (1, 918, 60, "6C-39", "TRDINT_VAL > 0"),
}
READ_COLUMNS = (
    *routing.COORDINATE_COLUMNS,
    *AMOUNT_FIELDS,
    *RECEIPT_ENTRIES,
    *ACCOUNT_ENTRIES,
    *ALLOCATION_ENTRIES,
    *TOPCODE_ENTRIES,
)


def require(condition, reason):
    if not condition:
        raise ValueError("ASEC_INTEREST_SOURCE_" + reason)


@lru_cache(maxsize=1)
def _cached_amount_entries():
    payload = (
        resources.files(__package__).joinpath(routing.DOMAINS_RESOURCE).read_bytes()
    )
    require(routing._sha(payload) == routing.DOMAINS_SHA256, "DOMAINS_HASH")
    fields = [f for f in json.loads(payload)["fields"] if f["name"] == "INT_VAL"]
    require(len(fields) == 1, "DOMAIN_FIELD_ROSTER:INT_VAL")
    field = fields[0]
    vintages = [
        v for v in field["vintages"] if v["income_year"] == routing.CURRENT_INCOME_YEAR
    ]
    require(len(vintages) == 1, "DOMAIN_VINTAGE:INT_VAL")
    vintage = vintages[0]
    require(
        vintage["dictionary_spelling"] == "INT_VAL"
        and vintage["pdf_sha256"] == routing.DICTIONARY_SHA256
        and vintage["source_url"] == routing.DICTIONARY_URL,
        "DICTIONARY_PIN",
    )
    routing._require_nonnegative_amount_domain(
        field,
        vintage,
        zero_semantics="none_or_niu_not_distinguishable_from_amount_alone",
        valid_minimum=0,
    )
    return tuple(
        {
            "INT_VAL": (
                vintage["ascii_length_as_printed"],
                vintage["ascii_position_as_printed"],
                vintage["pdf_page_1based"],
                vintage["printed_page"],
                field["domain"]["encoded_range_inclusive"]["maximum"],
                vintage["universe_as_printed"],
                field["domain"]["zero_semantics"],
            ),
            **ADDITIONAL_AMOUNT_ENTRIES,
        }.items()
    )


def amount_entries():
    """Return a detached mapping over privately cached immutable tuples."""
    return dict(_cached_amount_entries())


amount_entries.cache_clear = _cached_amount_entries.cache_clear


def _domain_agreement(ready):
    entry = amount_entries()["INT_VAL"]
    routing._require_live_amount_domains(ready, {"INT_VAL": (0, entry[4], entry[6])})


def _slot_status(age, receipt, account, amount):
    code, status = account
    value, amount_status = amount
    receipt_code, receipt_state = receipt
    # Reuse the shared universe/receipt classifier before refining account slots.
    kind = "missing" if value is None else ("zero" if value == 0 else "nonzero")
    base = routing.receipt_status(
        bool(age >= 15), receipt, kind, net_measure=False, zero_is_dollars=False
    )
    if age < 15:
        return base if code == 0 else "contradictory_outside_reporting_universe"
    if receipt_state != "in_printed_range":
        return base
    if amount_status != "in_printed_range":
        return (
            "missing_amount" if amount_status == "missing" else "invalid_amount_literal"
        )
    if status != "in_printed_range":
        return (
            "missing_account_literal"
            if status == "missing"
            else "invalid_account_literal"
        )
    if receipt_code != 1:
        return base if code == 0 else "account_without_receipt"
    if code == 0:
        return "unreported_account_slot" if value == 0 else "amount_without_account"
    return base


def project_interest_literals(ordered):
    """Descriptive projection; a DataFrame argument grants no source authority."""
    require(
        type(ordered) is pd.DataFrame and set(READ_COLUMNS) <= set(ordered), "COLUMNS"
    )
    out = pd.DataFrame(index=range(len(ordered)))
    ages = ordered.A_AGE.to_numpy(dtype=np.int64)
    require(((ages >= 0) & (ages <= 99)).all(), "AGE_RANGE")
    parsed = {}
    for name, entry in amount_entries().items():
        pairs = [
            routing.literal_code(t, range(entry[4] + 1), width=entry[0])
            for t in ordered[name]
        ]
        parsed[name] = pairs
        out[name + "_published_amount"] = pd.array(
            [v for v, _ in pairs], dtype="Float64"
        )
        out[name + "_literal_status"] = pd.array([s for _, s in pairs], dtype="string")
    codes = {}
    for name in (*RECEIPT_ENTRIES, *ACCOUNT_ENTRIES):
        allowed = (
            routing.RECEIPT_CODE_DOMAIN
            if name in RECEIPT_ENTRIES
            else routing.ACCOUNT_CODES
        )
        labels = (
            {0: RECEIPT_ENTRIES[name][5], 1: "yes", 2: "no"}
            if name in RECEIPT_ENTRIES
            else routing.ACCOUNT_CODES
        )
        frame, values, statuses = routing._codes_frame(
            name, ordered[name], allowed, labels
        )
        out = pd.concat([out, frame], axis=1)
        codes[name] = list(zip(values, statuses, strict=True))
    for name in AMOUNT_FIELDS:
        labels = []
        for i, age in enumerate(ages):
            value, status = parsed[name][i]
            if name.startswith("RINT_VAL"):
                label = _slot_status(
                    age,
                    codes["RINT_YN"][i],
                    codes["RINT_SC" + name[-1]][i],
                    parsed[name][i],
                )
            else:
                kind = (
                    "missing"
                    if value is None
                    else ("zero" if value == 0 else "nonzero")
                )
                label = routing.receipt_status(
                    bool(age >= 15),
                    codes["INT_YN"][i],
                    kind,
                    net_measure=False,
                    zero_is_dollars=name == "TRDINT_VAL",
                )
                if status not in ("missing", "in_printed_range"):
                    label = "invalid_amount_literal"
                # Unlike INT_VAL, TRDINT_VAL prints 'dollar value', not 'none or niu'.
                if name == "TRDINT_VAL" and label == "known_recipient_zero":
                    label = "observed_zero_component"
            labels.append(label)
        known = np.array(
            [
                s in (*routing.KNOWN_AMOUNT_STATUSES, "observed_zero_component")
                for s in labels
            ]
        )
        out[name + "_reporting_status"] = pd.array(labels, dtype="string")
        out[name + "_amount_known"] = known
        out[name + "_amount"] = out[name + "_published_amount"].where(known)
    for name in ACCOUNT_ENTRIES:
        out[name + "_account_known"] = [
            bool(age >= 15 and receipt[0] == 1 and code is not None and code > 0)
            for age, receipt, (code, _) in zip(
                ages, codes["RINT_YN"], codes[name], strict=True
            )
        ]
    flags = {}
    for name, entry in ALLOCATION_ENTRIES.items():
        frame, values, statuses = routing._codes_frame(
            name, ordered[name], entry[5], width=entry[0]
        )
        out = pd.concat([out, frame], axis=1)
        flags[name] = (values, statuses)
    out["allocation_origin"] = routing._allocation_origin(
        flags, ("TRDINT_VAL", "RINT_SC2")
    )
    for name, entry in TOPCODE_ENTRIES.items():
        frame, _, _ = routing._codes_frame(name, ordered[name], (0, 1), width=entry[0])
        out = pd.concat([out, frame], axis=1)
    published = [out[name + "_published_amount"] for name in AMOUNT_FIELDS]
    out["published_components_sum"] = published[1] + published[2] + published[3]
    out["combined_minus_published_components"] = (
        published[0] - out.published_components_sum
    )
    out["component_discrepancy_known"] = out.combined_minus_published_components.notna()
    out["all_component_amounts_observed"] = out[
        [name + "_amount_known" for name in AMOUNT_FIELDS[1:]]
    ].all(axis=1)
    out["source_age"] = ages
    return out


@dataclass(frozen=True)
class CurrentAsecInterestValues:
    person: pd.DataFrame
    asec_literals: pd.DataFrame
    evidence: dict


def interest_values_seal(values):
    """Physical value seal, including nullable backing storage and exact bits."""
    require(type(values) is CurrentAsecInterestValues, "VALUES_TYPE")
    digest = hashlib.sha256(PROTOCOL.encode())
    for table in (values.person, values.asec_literals):
        require(type(table) is pd.DataFrame and table.columns.is_unique, "TABLE_TYPE")
        digest.update(
            physical._json(
                {
                    "columns": list(table.columns),
                    "columns_axis": physical.checkpoint._index_spec(
                        table.columns, label="interest columns"
                    ),
                    "index": physical.checkpoint._index_spec(
                        table.index, label="interest"
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
    with tempfile.TemporaryDirectory(prefix="microcosm-asec-interest-") as tmp:
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
    field = ready.field("INT_VAL")
    entry = amount_entries()["INT_VAL"]
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


def qualify_current_asec_interest(preparation):
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
    field = _compare_total(ready, positions, ordered.INT_VAL)
    basis = project_interest_literals(ordered)
    basis.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    for name in ("statuses", "validity", "zero_origin"):
        basis["INT_VAL_parent_" + name] = getattr(field, name)[positions]
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
        "account_entries": ACCOUNT_ENTRIES,
        "account_codes": routing.ACCOUNT_CODES,
        "allocation_entries": ALLOCATION_ENTRIES,
        "topcode_entries": TOPCODE_ENTRIES,
        "fields_without_individual_allocation_flag": ["TRDINT_VAL", "RINT_SC2"],
        "preparation_sha256": routing._sha(entry[1]),
        "asec_native_sha256": routing._sha(issued[1]),
        "money_header_sha256": routing._sha(ready.header),
        "source_member_sha256": digest,
        "source_year": 2024,
        "survey_year": 2025,
        "complete_source_rows": rows,
        "selected_rows": len(out),
        "read_columns": READ_COLUMNS,
        "component_discrepancy_is_diagnostic_only": True,
        "unreported_slot_imputed_zero": False,
        "tax_treatment_assigned": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    # Freeze detached JSON-compatible metadata before its seal; do not retain the
    # module's mutable constant dictionaries inside a returned descriptive view.
    result = CurrentAsecInterestValues(out, literals, json.loads(json.dumps(evidence)))
    seal = interest_values_seal(result)
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
    require(interest_values_seal(result) == seal, "FINAL_VALUES_CHANGED")
    return result
