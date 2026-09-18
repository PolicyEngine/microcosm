"""Receipt-universe projection from the retained current ASEC original.

UC_VAL's numeric zero is not itself evidence of nonreceipt. This module keeps
the source amount separate from a canonical amount whose reporting basis is
known. Detached projection values grant no source authority: consuming hosts
requalify the retained preparation and compare these values before/after I/O.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import asec_coverage_authentication as coverage
from . import asec_current_money as money
from . import source_csv_builtin
from . import survey_population_preparation as source
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.current-asec-unemployment-source.v1"
READ_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "UC_VAL", "UC_YN")
DICTIONARY = {
    "url": "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf",
    "sha256": "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f",
    "pdf_page_1based": 50,
    "printed_page": "6C-29",
    "receipt_field": "UC_YN",
    "receipt_codes": {"0": "niu", "1": "yes", "2": "no"},
    "receipt_universe": "All Persons aged 15+",
    "amount_field": "UC_VAL",
    "amount_universe": "UC_YN = 1",
    "zero_code": "none or niu",
}


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_ASEC_UNEMPLOYMENT_" + reason)


def reporting_basis(amount, age, receipt_tokens):
    """Classify invented or observed arrays without issuing source authority.

    Invalid/missing receipt literals are unresolved, rather than recoded to no.
    Contradictions remain inspectable source observations, excluded from the
    modeled donor and canonical amount. Age only gates the printed universe;
    it never creates an income zero.
    """
    amounts = np.asarray(amount)
    ages = np.asarray(age)
    tokens = tuple(receipt_tokens)
    require(
        amounts.dtype == np.dtype("float64")
        and ages.dtype == np.dtype("float64")
        and amounts.ndim == ages.ndim == 1
        and len(amounts) == len(ages) == len(tokens),
        "ARRAY_CONTRACT",
    )
    require(
        np.isfinite(ages).all()
        and ((ages >= 0) & (ages <= 99) & (ages == np.floor(ages))).all(),
        "AGE_DOMAIN",
    )
    require(
        all(type(t) is str and len(t) <= 64 for t in tokens)
        and not np.isinf(amounts).any()
        and (
            (amounts[np.isfinite(amounts)] >= 0)
            & (amounts[np.isfinite(amounts)] <= 99999)
        ).all(),
        "AMOUNT_OR_TOKEN_DOMAIN",
    )
    result = pd.DataFrame({"source_amount": amounts.copy(), "receipt_literal": tokens})
    result["source_reporting_universe"] = ages >= 15
    result["receipt_code_known"] = [t in ("0", "1", "2") for t in tokens]
    result["receipt_code"] = pd.array(
        [int(t) if t in ("0", "1", "2") else pd.NA for t in tokens], dtype="Int8"
    )
    canonical = np.full(len(amounts), np.nan, dtype=np.float64)
    labels = []
    for i, (value, years, token) in enumerate(zip(amounts, ages, tokens, strict=True)):
        if years < 15:
            label = (
                "outside_reporting_universe"
                if token == "0" and (np.isnan(value) or value == 0)
                else "contradictory_outside_reporting_universe"
            )
        elif token == "":
            label = "missing_receipt_literal"
        elif token not in ("0", "1", "2"):
            label = "unrecognized_receipt_literal"
        elif np.isnan(value):
            label = "missing_amount"
        elif token == "0":
            label = "niu" if value == 0 else "contradictory_niu_positive"
        elif token == "1":
            label = "known_receipt" if value > 0 else "ambiguous_recipient_zero"
            if value > 0:
                canonical[i] = value
        else:
            label = "known_nonreceipt" if value == 0 else "contradictory_no_positive"
            if value == 0:
                canonical[i] = value
        labels.append(label)
    result["reporting_status"] = labels
    result["canonical_amount"] = canonical
    result["canonical_amount_known"] = np.isfinite(canonical)
    return result


def _read_capture(path, *, rows):
    """Bounded literal reader; its path and row count do not establish authority."""
    reader = source_csv_builtin.capture_csv_reader(csv)
    require(reader is not None, "CSV_READER_CHANGED")
    records, keys, coordinates = [], set(), set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        stream = reader(handle, strict=True)
        header = next(stream, [])
        require(
            bool(header)
            and all(header)
            and len(header) == len(set(header))
            and set(READ_COLUMNS) <= set(header),
            "HEADER",
        )
        positions = [header.index(c) for c in READ_COLUMNS]
        for row in stream:
            require(len(row) == len(header) and len(records) < rows, "ROW_SHAPE")
            record = dict(zip(READ_COLUMNS, (row[i] for i in positions), strict=True))
            key = record["PERIDNUM"]
            require(re.fullmatch(r"[0-9]{22}", key, re.ASCII) is not None, "PERSON_KEY")
            for name, width in (("PH_SEQ", 5), ("A_LINENO", 2), ("A_AGE", 2)):
                require(
                    re.fullmatch(r"[0-9]{1," + str(width) + "}", record[name], re.ASCII)
                    is not None,
                    "COORDINATE:" + name,
                )
            pair = (int(record["PH_SEQ"]), int(record["A_LINENO"]))
            require(
                min(pair) > 0 and pair not in coordinates and key not in keys,
                "DUPLICATE_OR_INVALID_COORDINATE",
            )
            require(
                record["UC_VAL"] == ""
                or re.fullmatch(r"[0-9]{1,5}", record["UC_VAL"], re.ASCII) is not None,
                "AMOUNT_TOKEN",
            )
            require(len(record["UC_YN"]) <= 64, "RECEIPT_TOKEN_BOUND")
            records.append(record)
            keys.add(key)
            coordinates.add(pair)
    require(len(records) == rows, "ROW_COUNT")
    return pd.DataFrame(records).set_index("PERIDNUM", drop=False)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _amount_observations(field, positions, literals):
    """Join literal validity; missing backing storage never becomes a survey zero.

    Today's retained parent requires complete ready money before issuing a
    preparation. This helper keeps that prerequisite separate from the literal
    join, so a missing field is represented correctly if that scope later grows.
    """
    require(type(field) is money.MoneyField, "MONEY_FIELD_TYPE")
    positions = np.asarray(positions)
    require(
        positions.dtype == np.dtype("int64") and positions.ndim == 1, "MONEY_POSITIONS"
    )
    tokens = tuple(literals)
    require(
        len(tokens) == len(positions)
        and all(
            type(t) is str and (t == "" or re.fullmatch(r"[0-9]{1,5}", t, re.ASCII))
            for t in tokens
        ),
        "AMOUNT_TOKEN",
    )
    valid = field.validity[positions] == 1
    require(
        np.array_equal(valid, np.array([t != "" for t in tokens])),
        "CURRENT_AMOUNT_SOURCE_VALIDITY",
    )
    amounts = field.amounts[positions].copy()
    literal_values = np.asarray([float(t) if t else np.nan for t in tokens])
    require(
        np.array_equal(amounts[valid], literal_values[valid]),
        "CURRENT_AMOUNT_SOURCE_IDENTITY",
    )
    amounts[~valid] = np.nan
    return amounts


@dataclass(frozen=True)
class CurrentAsecUnemploymentValues:
    person: pd.DataFrame
    evidence: dict


def qualify_current_asec_unemployment(preparation):
    """Capture the exact current source member retained by the original owner."""
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    native = state.native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "NATIVE_ISSUANCE",
    )
    parent = issued[2].parent
    ready = parent.ready()
    header = json.loads(ready.header)
    require(
        header["target_year"] == 2024 and header["semantic"] == "annual_current_money",
        "MONEY_PERIOD",
    )
    native_document = json.loads(issued[1])
    require(
        native_document["source_year"] == native_document["income_year"] == 2024
        and native_document["survey_year"] == 2025,
        "NATIVE_PERIOD",
    )
    pins = [p for p in coverage._MEMBER_PINS if p[0] == 2024]
    require(len(pins) == 1, "SOURCE_REGISTRY")
    year, member, archive, digest, rows, size = pins[0]
    retained = [
        s for s in issued[2].coverage.receipt["sources"] if s["source_year"] == year
    ]
    require(
        len(retained) == 1
        and retained[0]["member"] == member
        and retained[0]["archive_sha256"] == archive
        and retained[0]["member_sha256"] == digest
        and retained[0]["rows"] == rows
        and retained[0]["member_bytes"] == size,
        "NATIVE_MEMBER_BINDING",
    )
    with tempfile.TemporaryDirectory(prefix="microcosm-current-uc-") as tmp:
        captured = Path(tmp) / member
        identity = coverage._capture(
            state.root / "asec" / member,
            captured,
            size=size,
            digest=digest,
            budget=[coverage._BODY_MAX],
        )
        raw = _read_capture(captured, rows=rows)
        require(
            coverage._identity(captured.stat(follow_symlinks=False)) == identity
            and _file_sha(captured) == digest,
            "CAPTURE_CHANGED",
        )
    positions = np.flatnonzero(np.asarray(parent.scope.person_years) == 2024)
    keys = np.asarray(parent.scope.person_native_keys)[positions]
    require(
        len(keys) == rows and len(set(keys)) == rows and set(keys) == set(raw.index),
        "COMPLETE_CURRENT_SOURCE_JOIN",
    )
    ordered = raw.loc[keys]
    parent_people = parent.frame.person.iloc[positions]
    for raw_name, parent_name in (
        ("PH_SEQ", "source_household_id"),
        ("A_LINENO", "A_LINENO"),
        ("A_AGE", "A_AGE"),
    ):
        require(
            np.array_equal(
                ordered[raw_name].astype("int64").to_numpy(),
                parent_people[parent_name].to_numpy(),
            ),
            "PARENT_COORDINATE_IDENTITY",
        )
    field = ready.field("UC_VAL")
    amounts = _amount_observations(field, positions, ordered.UC_VAL)
    basis = reporting_basis(
        amounts, ordered.A_AGE.to_numpy(dtype="float64"), ordered.UC_YN
    )
    basis.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    basis["amount_status"] = field.statuses[positions]
    basis["amount_validity"] = field.validity[positions]
    basis["zero_origin"] = field.zero_origin[positions]
    people = state.frame.person
    selected = people.loc[people[support_channel_column("person")].eq("asec")]
    native_ids = selected[spine_source_id_column("person")].to_numpy()
    require(
        len(set(native_ids)) == len(native_ids) and set(native_ids) <= set(basis.index),
        "SELECTED_NATIVE_JOIN",
    )
    out = basis.loc[native_ids].copy()
    out["native_person_id"] = native_ids
    out.index = pd.Index(selected.person_id.to_numpy(), name="person_id")
    evidence = {
        "protocol": PROTOCOL,
        "dictionary": json.loads(json.dumps(DICTIONARY)),
        "preparation_sha256": hashlib.sha256(entry[1]).hexdigest(),
        "asec_native_sha256": hashlib.sha256(issued[1]).hexdigest(),
        "money_header_sha256": hashlib.sha256(ready.header).hexdigest(),
        "source_member_sha256": digest,
        "read_columns": list(READ_COLUMNS),
        "complete_current_source_rows": rows,
        "selected_rows": len(out),
        "projection_sha256": hashlib.sha256(
            out.to_json(orient="table").encode()
        ).hexdigest(),
        "policy": "known_yes_positive_or_no_zero_only; ambiguous_NIU_missing_contradictions_retained_unknown",
        "unallocated_observation_claim": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    require(
        preparation._checked()[1] == entry[1] and parent.ready().header == ready.header,
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
    require(
        hashlib.sha256(out.to_json(orient="table").encode()).hexdigest()
        == evidence["projection_sha256"],
        "FINAL_PROJECTION_CHANGED",
    )
    return CurrentAsecUnemploymentValues(out, evidence)
