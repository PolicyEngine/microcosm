"""Observed ACS income anchors from the retained original source archive.

Values and projections are descriptive, not source issuers. Consumers retain the
original preparation and requalify after their last relevant I/O. This module
does not decompose income, create tax inputs, or fill any analytical zeros.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import acs_housing_universe_source as housing
from . import acs_person_coverage_authentication as records
from . import graph_full_puf_enrichment as physical
from . import source_csv_builtin
from . import support_provenance as provenance
from . import survey_population_preparation as preparation

PROTOCOL = "microcosm.us.current-acs-income-anchors.v1"
COLUMNS = ("SERIALNO", "SPORDER", "INTP", "RETP", "ADJINC", "AGEP", "FINTP", "FRETP")
ANCHORS = (
    ("INTP", "FINTP", "property_income", "acs_interest_dividend_rental_income"),
    ("RETP", "FRETP", "retirement_income", "acs_retirement_income"),
)
STRING_DTYPE = pd.StringDtype(storage="python", na_value=pd.NA)


def require(condition, reason):
    if not condition:
        raise ValueError("ACS_INCOME_ANCHOR_" + reason)


def _literal(value):
    require(type(value) is str and len(value) <= 64, "LITERAL_TYPE_OR_BOUND")
    return value


def parse_anchor(
    token: str, *, age: str, adjustment: str, allocation: str, field: str
) -> dict:
    """Classify source literals separately from native numeric storage.

    Allocation is provenance, not amount validity. Published domains describe
    released records; their disclosure bounds are not latent income bounds.
    """
    require(field in ("INTP", "RETP"), "FIELD")
    token, age, adjustment, allocation = map(
        _literal, (token, age, adjustment, allocation)
    )
    require(re.fullmatch(r"[0-9]{1,2}", age, re.ASCII) is not None, "AGE_LITERAL")
    parsed_age = int(age)
    factor_valid = (
        re.fullmatch(r"[0-9]{1,7}", adjustment, re.ASCII) is not None
        and int(adjustment) > 0
    )
    status, amount = "observed", None
    if token == "":
        status = (
            "outside_universe_blank" if parsed_age < 15 else "missing_source_amount"
        )
    elif re.fullmatch(r"-?[0-9]+", token, re.ASCII) is None:
        status = "malformed_source_amount"
    else:
        raw = int(token)
        in_domain = (
            raw == 0 or 4 <= raw <= 999999 or (field == "INTP" and -10000 <= raw <= -4)
        )
        if not in_domain:
            status = "outside_published_domain"
        elif parsed_age < 15:
            status = "outside_universe_observation"
        elif not factor_valid:
            status = "invalid_adjustment"
        else:
            amount = float(
                np.float64(raw) * (np.float64(int(adjustment)) / 1_000_000.0)
            )
    return {
        "amount": amount,
        "known": status == "observed",
        "status": status,
        "allocation_status": {
            "0": "not_allocated",
            "1": "allocated",
            "": "allocation_missing",
        }.get(allocation, "allocation_unrecognized"),
        "adjustment_known": factor_valid,
        "in_income_universe": parsed_age >= 15,
    }


def _key(row):
    require(
        re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", row["SERIALNO"], re.ASCII) is not None,
        "SOURCE_HOUSEHOLD_KEY",
    )
    require(
        re.fullmatch(r"[0-9]{1,2}", row["SPORDER"], re.ASCII) is not None
        and 1 <= int(row["SPORDER"]) <= 20,
        "SOURCE_PERSON_KEY",
    )
    return row["SERIALNO"], int(row["SPORDER"])


def _scan(stream, wanted, selected, *, maximum):
    require(source_csv_builtin.capture_csv_reader(csv) is not None, "CSV_BINDING")
    header, count = None, 0
    for raw in records._records(stream):
        values = records._decode_record(raw, first=header is None)
        if header is None:
            header = values
            require(
                bool(header)
                and all(header)
                and len(set(header)) == len(header)
                and set(COLUMNS) <= set(header),
                "SOURCE_HEADER",
            )
            positions = [header.index(c) for c in COLUMNS]
            continue
        count += 1
        require(count <= maximum and len(values) == len(header), "SOURCE_ROW_SHAPE")
        row = dict(zip(COLUMNS, (values[p] for p in positions), strict=True))
        require(all(len(v) <= 64 for v in row.values()), "SOURCE_TOKEN_BOUND")
        key = _key(row)
        if key in wanted:
            require(key not in selected, "DUPLICATE_SELECTED_SOURCE_KEY")
            selected[key] = row
    require(header is not None, "SOURCE_HEADER")
    return count


def _origins(state, document):
    payload = document["origins"]["persons"]
    full = pd.DataFrame(payload["rows"], columns=payload["columns"])
    people = state.frame.person
    require(
        full.person_id.dtype == np.dtype("int64")
        and full.person_id.is_unique
        and np.array_equal(full.person_id.to_numpy(), people.person_id.to_numpy()),
        "ORIGIN_AXIS",
    )
    channel = provenance.support_channel_column("person")
    native_id = provenance.spine_source_id_column("person")
    require(
        np.array_equal(full.source.to_numpy(), people[channel].to_numpy())
        and np.array_equal(
            full.selected_receiving_person_id.to_numpy(), people[native_id].to_numpy()
        ),
        "ORIGIN_FRAME_IDENTITY",
    )
    origins = full.loc[full.source.eq("acs")].copy().set_index("person_id", drop=True)
    require(
        len(origins) > 0
        and origins.source_year.eq(2024).all()
        and origins.survey_year.eq(2024).all(),
        "ORIGIN_PERIOD",
    )
    native = state.source_frames[0].person
    require(
        np.array_equal(
            origins.selected_receiving_person_id.to_numpy(), native.person_id.to_numpy()
        ),
        "ORIGIN_NATIVE_AXIS",
    )
    keys = []
    for _, row in origins.iterrows():
        require(
            all(
                type(v) is str
                for v in (
                    row.raw_native_household_id,
                    row.raw_native_person_id,
                    row.native_line_numeric_original,
                )
            ),
            "ORIGIN_LITERAL_TYPE",
        )
        key = _key(
            {
                "SERIALNO": row.raw_native_household_id,
                "SPORDER": row.native_line_numeric_original,
            }
        )
        require(row.raw_native_person_id == str(key[1]), "ORIGIN_NATIVE_PERSON_KEY")
        keys.append(key)
    require(len(set(keys)) == len(keys), "ORIGIN_DUPLICATE")
    origins["anchor_source_key"] = keys
    return origins


def _raw_table(origins, selected):
    require(set(selected) == set(origins.anchor_source_key), "SELECTED_SOURCE_ROSTER")
    raw = pd.DataFrame(
        [selected[k] for k in origins.anchor_source_key],
        index=origins.index.copy(),
        columns=COLUMNS,
        dtype=object,
    )
    require(
        all(
            _key(row) == key
            for row, key in zip(
                raw.to_dict("records"), origins.anchor_source_key, strict=True
            )
        ),
        "SOURCE_COORDINATE_CHANGED",
    )
    return raw


def _native_numbers(tokens):
    # Match the existing mapper's numeric-coercion semantics for identity only;
    # this operation does not determine source amount knownness.
    return pd.to_numeric(pd.Series(tokens, dtype=object), errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _same_number(left, right):
    if pd.isna(left):
        return bool(pd.isna(right))
    if isinstance(left, (bool, np.bool_)) or not isinstance(
        left, (int, float, np.integer, np.floating)
    ):
        return False
    return np.float64(left).view("uint64") == np.float64(right).view("uint64")


def _compare_retained(raw, origins, native):
    people = native.person
    require(
        raw.index.equals(origins.index)
        and np.array_equal(
            origins.selected_receiving_person_id.to_numpy(), people.person_id.to_numpy()
        ),
        "RETAINED_AXIS",
    )
    households = native.table("household").set_index("household_id", drop=False)
    require(
        households.index.is_unique
        and np.array_equal(
            origins.selected_receiving_household_id.to_numpy(),
            people.person_household_id.to_numpy(),
        ),
        "RETAINED_HOUSEHOLD_ID",
    )
    serials = households.SERIALNO.reindex(people.person_household_id.to_numpy())
    require(
        np.array_equal(raw.SERIALNO.to_numpy(), serials.to_numpy())
        and np.array_equal(raw.SPORDER.map(int).to_numpy(), people.SPORDER.to_numpy()),
        "RETAINED_SOURCE_ID",
    )
    numbers = {c: _native_numbers(raw[c]) for c in ("INTP", "RETP", "ADJINC", "AGEP")}
    for column, expected_numbers in numbers.items():
        for literal, stored, expected in zip(
            raw[column], people[column], expected_numbers, strict=True
        ):
            same = (
                stored == literal
                if type(stored) is str
                else _same_number(stored, expected)
            )
            require(bool(same), "RETAINED_RAW_" + column)
    adjustment = numbers["ADJINC"]
    for column, _flag, _prefix, output in ANCHORS:
        require(people[output].dtype == np.dtype("float64"), "RETAINED_ADJUSTED_DTYPE")
        expected = numbers[column] * (adjustment / 1_000_000.0)
        actual = people[output].to_numpy(copy=False)
        require(
            np.array_equal(np.isnan(expected), np.isnan(actual))
            and np.array_equal(
                expected[~np.isnan(expected)].view("uint64"),
                actual[~np.isnan(actual)].view("uint64"),
            ),
            "RETAINED_ADJUSTED_BITS",
        )


def _parsed_table(raw, origins):
    result = raw.copy(deep=True)
    result["native_person_id"] = origins.selected_receiving_person_id.to_numpy(
        copy=True
    )
    result["source_year"] = 2024
    result["dollar_year"] = 2024
    for column, flag, prefix, _output in ANCHORS:
        values = [
            parse_anchor(
                r[column],
                age=r["AGEP"],
                adjustment=r["ADJINC"],
                allocation=r[flag],
                field=column,
            )
            for r in raw.to_dict("records")
        ]
        parsed = pd.DataFrame(values, index=raw.index)
        for name in parsed:
            dtype = (
                "Float64"
                if name == "amount"
                else (
                    bool
                    if name in ("known", "adjustment_known", "in_income_universe")
                    else STRING_DTYPE
                )
            )
            result[prefix + "_" + name] = pd.array(parsed[name], dtype=dtype)
    return result


@dataclass(frozen=True)
class QualifiedAcsIncomeAnchors:
    """Detached source observations, not a substitute for the preparation."""

    anchors: pd.DataFrame
    projection: bytes
    evidence: dict


def income_anchor_seal(qualified: QualifiedAcsIncomeAnchors) -> tuple:
    require(type(qualified) is QualifiedAcsIncomeAnchors, "QUALIFIED_TYPE")
    require(
        qualified.projection
        == qualified.anchors.reset_index().to_json(orient="table", index=False).encode()
        and qualified.evidence["projection_sha256"]
        == hashlib.sha256(qualified.projection).hexdigest(),
        "PROJECTION_BINDING",
    )
    return (
        physical._table_stamp(qualified.anchors),
        qualified.projection,
        json.dumps(
            qualified.evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )


def _capture_person(root, pin, wanted, total_rows):
    """Return literal values from an owned capture, after cleanup I/O completes.

    Caller-supplied pins grant no authority here; the qualifier binds them to
    the retained catalogue and rechecks that original owner after this returns.
    """
    _role, name, digest, size = pin
    selected, count = {}, 0
    with tempfile.TemporaryDirectory(
        prefix="microcosm-acs-income-anchor-"
    ) as directory:
        captured = Path(directory) / name
        require(
            housing._copy(root / "acs" / name, captured, size, exact_size=size)
            == digest,
            "ACS_CAPTURE_DIGEST",
        )
        with zipfile.ZipFile(captured) as archive:
            members, prefix = records._members(archive, "person")
            for item in members:
                with archive.open(item) as stream:
                    if item.filename.casefold().startswith(prefix):
                        count += _scan(
                            stream,
                            wanted,
                            selected,
                            maximum=total_rows - count,
                        )
                    else:
                        while stream.read(65536):
                            pass
        require(
            count == total_rows and housing._persisted_sha(captured, size) == digest,
            "ACS_CAPTURE_CHANGED",
        )
    return selected


def qualify_current_acs_income_anchors(source_preparation) -> QualifiedAcsIncomeAnchors:
    """Read exact original ACS members and recheck the real owner before return."""
    require(
        type(source_preparation)
        is preparation.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = source_preparation._checked()
    state, document = entry[2], json.loads(entry[1])
    origins = _origins(state, document)
    owned = preparation.acs_catalogue._lookup(state.catalogues[0])
    catalogue = json.loads(owned.receipt)
    pins = tuple(p for p in owned.pins if p[0] == "person")
    require(
        len(pins) == 1 and catalogue["source_year"] == catalogue["survey_year"] == 2024,
        "ACS_PIN_OR_PERIOD",
    )
    selected = _capture_person(
        state.root,
        pins[0],
        set(origins.anchor_source_key),
        catalogue["counts"]["people"],
    )
    raw = _raw_table(origins, selected)
    _compare_retained(raw, origins, state.source_frames[0])
    anchors = _parsed_table(raw, origins)
    projection = anchors.reset_index().to_json(orient="table", index=False).encode()
    evidence = {
        "protocol": PROTOCOL,
        "preparation_sha256": hashlib.sha256(entry[1]).hexdigest(),
        "acs_catalogue_sha256": hashlib.sha256(owned.receipt).hexdigest(),
        "acs_person_archive_sha256": pins[0][2],
        "selected_persons": len(anchors),
        "source_year": 2024,
        "dollar_year": 2024,
        "income_period": "rolling_12_months",
        "projection_sha256": hashlib.sha256(projection).hexdigest(),
        "source_admission_issued": False,
        "unallocated_observation_claim": False,
        "analytical_universe_zeros_supplied": False,
        "decomposition_performed": False,
        "release_eligible": False,
    }
    result = QualifiedAcsIncomeAnchors(anchors, projection, evidence)
    seal = income_anchor_seal(result)
    require(source_preparation._checked() is entry, "PREPARATION_CHANGED")
    preparation._pure_final(state)
    require(
        preparation._ISSUED.get(id(source_preparation)) is entry
        and preparation.acs_catalogue._lookup(state.catalogues[0]) is owned,
        "FINAL_OWNER",
    )
    require(income_anchor_seal(result) == seal, "FINAL_VALUES_CHANGED")
    return result
