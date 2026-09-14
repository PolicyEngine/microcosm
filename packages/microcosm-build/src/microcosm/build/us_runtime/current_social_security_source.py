"""Current survey Social Security totals, reason literals and unknown shares.

An additive projection from a live authenticated survey preparation. The
current ASEC original member is captured against the maintained closed member
registry; carried RESNSS columns are never source evidence. ACS SSP remains a
combined source amount. This module fits no model and issues no admission.
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
from . import source_csv_builtin
from . import survey_population_preparation as source
from . import survey_social_security as basis_owner
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.current-social-security-source.v1"
READ_COLUMNS = (
    "PERIDNUM",
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "SS_VAL",
    "SS_YN",
    "RESNSS1",
    "RESNSS2",
    "RESNSSA",
    "I_SSVAL",
    "I_SSYN",
)
WIDTHS = {
    "PH_SEQ": 5,
    "A_LINENO": 2,
    "A_AGE": 2,
    "SS_VAL": 5,
    "SS_YN": 1,
    "RESNSS1": 1,
    "RESNSS2": 1,
    "RESNSSA": 1,
    "I_SSVAL": 2,
    "I_SSYN": 2,
}
DICTIONARY = {
    "url": "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf",
    "sha256": "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f",
    "pdf_pages_1based": {
        "reasons": 48,
        "total_and_recipiency": 49,
        "allocation_values": [53, 56],
        "allocation_fields": [57, 59],
    },
}


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SOCIAL_SECURITY_" + reason)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(series):
    require(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype),
        "PHYSICAL_NUMERIC_TYPE",
    )
    return series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)


def _read_capture(path, *, rows):
    """Bounded literal parser; a path or row count confers no source authority."""
    reader = source_csv_builtin.capture_csv_reader(csv)
    require(reader is not None, "CSV_READER_CHANGED")
    records, seen, coordinates = [], set(), set()
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
        indices = [header.index(name) for name in READ_COLUMNS]
        for row in stream:
            require(len(row) == len(header) and len(records) < rows, "ROW_SHAPE")
            selected = [row[i] for i in indices]
            require(
                re.fullmatch(r"[0-9]{22}", selected[0], re.ASCII) is not None,
                "PERSON_KEY",
            )
            record = dict(zip(READ_COLUMNS, selected, strict=True))
            for name, width in WIDTHS.items():
                token = record[name]
                # Missing measurements remain literal unknowns. Coordinates
                # cannot be missing because they establish the source join.
                require(
                    token == ""
                    or re.fullmatch(r"[0-9]{1," + str(width) + "}", token, re.ASCII)
                    is not None,
                    "TOKEN_WIDTH:" + name,
                )
            require(
                all(record[n] != "" for n in ("PH_SEQ", "A_LINENO", "A_AGE")),
                "COORDINATE_MISSING",
            )
            pair = (int(record["PH_SEQ"]), int(record["A_LINENO"]))
            require(
                min(pair) > 0 and pair not in coordinates and selected[0] not in seen,
                "COORDINATE_DUPLICATE",
            )
            coordinates.add(pair)
            seen.add(selected[0])
            records.append(record)
    require(len(records) == rows, "ROW_COUNT")
    return pd.DataFrame(records, columns=READ_COLUMNS).set_index("PERIDNUM", drop=False)


def _codes(tokens, allowed):
    values = np.full(len(tokens), np.nan, dtype=np.float64)
    for i, token in enumerate(tokens):
        require(type(token) is str, "CODE_TOKEN_TYPE")
        if token != "":
            require(re.fullmatch(r"[0-9]+", token, re.ASCII) is not None, "CODE_TOKEN")
            require(int(token) in allowed, "CODE_DOMAIN")
            values[i] = int(token)
    return values


def _allocation_labels(raw):
    total = _codes(raw.I_SSVAL, {0, 11, 12, 13, 14, 15})
    receipt = _codes(raw.I_SSYN, {0, 10, 11})
    reason = _codes(raw.RESNSSA, set(range(10)))
    return pd.DataFrame(
        {
            "amount_allocation_code": total,
            "recipiency_allocation_code": receipt,
            "reason_allocation_code": reason,
            "allocation_origin": [
                "unresolved_allocation_provenance"
                if not np.isfinite([a, b, c]).all()
                else "publisher_allocated"
                if any(v != 0 for v in (a, b, c))
                else "publisher_no_allocation"
                for a, b, c in zip(total, receipt, reason, strict=True)
            ],
        },
        index=raw.index,
    )


@dataclass(frozen=True)
class CurrentSocialSecurityProjection:
    """Computed data, requalified by the host before execution and replay."""

    person: pd.DataFrame
    asec_literals: pd.DataFrame
    evidence: dict


def qualify_current_social_security(preparation):
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    acs_owned = source.acs_native._owned(state.native[0])
    acs_receipt = json.loads(acs_owned.payload)
    require(acs_receipt["vintage"] == 2024, "ACS_NATIVE_PERIOD")
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
    selected_pins = [pin for pin in coverage._MEMBER_PINS if pin[0] == 2024]
    require(len(selected_pins) == 1, "SOURCE_REGISTRY")
    year, member, archive, digest, rows, size = selected_pins[0]
    native_sources = issued[2].coverage.receipt["sources"]
    retained = [s for s in native_sources if s["source_year"] == year]
    require(
        len(retained) == 1
        and retained[0]["member"] == member
        and retained[0]["archive_sha256"] == archive
        and retained[0]["member_sha256"] == digest
        and retained[0]["rows"] == rows
        and retained[0]["member_bytes"] == size,
        "NATIVE_MEMBER_BINDING",
    )
    with tempfile.TemporaryDirectory(prefix="microcosm-current-ss-") as tmp:
        captured = Path(tmp) / member
        identity = coverage._capture(
            state.root / "asec" / member,
            captured,
            size=size,
            digest=digest,
            # _capture's budget bounds its encoded coverage projection, not
            # CSV bytes. Short valid source rows can expand in that format.
            # Use the source owner's existing closed projection bound; the
            # exact captured CSV size and digest remain separate checks.
            budget=[coverage._BODY_MAX],
        )
        raw = _read_capture(captured, rows=rows)
        require(
            coverage._identity(captured.stat(follow_symlinks=False)) == identity,
            "CAPTURE_CHANGED",
        )
        require(_file_sha(captured) == digest, "CAPTURE_DIGEST")

    scope = parent.scope
    current_positions = np.flatnonzero(np.asarray(scope.person_years) == 2024)
    keys = np.asarray(scope.person_native_keys)[current_positions]
    require(
        len(keys) == rows and len(set(keys)) == rows and set(keys) == set(raw.index),
        "COMPLETE_CURRENT_SOURCE_JOIN",
    )
    ordered = raw.loc[keys].copy()
    pp = parent.frame.person.iloc[current_positions]
    for native_name, parent_name in (
        ("PH_SEQ", "source_household_id"),
        ("A_LINENO", "A_LINENO"),
        ("A_AGE", "A_AGE"),
    ):
        require(
            np.array_equal(
                ordered[native_name].astype("int64").to_numpy(),
                pp[parent_name].to_numpy(),
            ),
            "PARENT_COORDINATE_IDENTITY",
        )
    field = ready.field("SS_VAL")
    require((field.validity[current_positions] == 1).all(), "CURRENT_AMOUNT_UNKNOWN")
    amount = field.amounts[current_positions].copy()
    require(
        amount.dtype == np.dtype("float64")
        and np.isfinite(amount).all()
        and (amount >= 0).all(),
        "CURRENT_AMOUNT_DOMAIN",
    )
    literal_amount = _codes(ordered.SS_VAL, range(100000))
    require(np.array_equal(literal_amount, amount), "CURRENT_AMOUNT_SOURCE_IDENTITY")
    reason_1 = _codes(ordered.RESNSS1, range(9))
    reason_2 = _codes(ordered.RESNSS2, range(9))
    recipiency = _codes(ordered.SS_YN, {0, 1, 2})
    report_amount, components, allowed, labels = basis_owner.asec_reporting_basis(
        amount,
        ordered.A_AGE.to_numpy(dtype=np.float64),
        recipiency,
        reason_1,
        reason_2,
    )
    allocation = _allocation_labels(ordered)

    people = state.frame.person
    require(
        people.person_id.dtype == np.dtype("int64")
        and people.person_id.is_unique
        and (people.person_id >= 0).all(),
        "PERSON_AXIS",
    )
    channel = people[support_channel_column("person")]
    require(set(channel) == {"acs", "asec"}, "CHANNEL_ROSTER")
    out = pd.DataFrame(index=pd.Index(people.person_id, name="person_id"))
    out["native_person_id"] = people[spine_source_id_column("person")].to_numpy()
    out["source"] = channel.to_numpy()
    out["social_security_source_total"] = np.nan
    out["source_reporting_universe"] = False
    out["source_reporting_unit"] = "person_report_record"
    for c in basis_owner.COMPONENTS:
        out[c] = np.nan
        out["allowed_" + c] = True
    out["basis_origin"] = "unresolved"
    out["allocation_origin"] = "unresolved_allocation_provenance"
    original_ids = np.asarray(scope.person_ids)[current_positions]
    lookup = {int(pid): i for i, pid in enumerate(original_ids)}
    asec_ids = out.index[out.source.eq("asec")]
    require(
        all(int(v) in lookup for v in out.loc[asec_ids, "native_person_id"]),
        "SELECTED_NATIVE_JOIN",
    )
    take = np.asarray([lookup[int(v)] for v in out.loc[asec_ids, "native_person_id"]])
    require(len(set(take)) == len(take), "SELECTED_NATIVE_DUPLICATE")
    out.loc[asec_ids, "social_security_source_total"] = report_amount[take]
    out.loc[asec_ids, "source_reporting_universe"] = (
        ordered.A_AGE.to_numpy(dtype=np.float64)[take] >= 15
    )
    out.loc[asec_ids, "source_reporting_unit"] = (
        "person_report_may_combine_family_payments"
    )
    for j, c in enumerate(basis_owner.COMPONENTS):
        out.loc[asec_ids, c] = components[take, j]
        out.loc[asec_ids, "allowed_" + c] = allowed[take, j]
    out.loc[asec_ids, "basis_origin"] = np.asarray(labels)[take]
    out.loc[asec_ids, "allocation_origin"] = allocation.allocation_origin.to_numpy()[
        take
    ]

    acs = people.loc[channel.eq("acs")]
    require(
        {"SSP", "AGEP", "ADJINC", "acs_social_security_income"} <= set(acs),
        "ACS_SOURCE_COLUMNS",
    )
    raw_ss = _numeric(acs.SSP)
    adjusted = _numeric(acs.acs_social_security_income)
    age = _numeric(acs.AGEP)
    factor = _numeric(acs.ADJINC) / 1_000_000
    require(
        np.isfinite(age).all()
        and (age == np.floor(age)).all()
        and ((age >= 0) & (age <= 99)).all(),
        "ACS_AGE",
    )
    require(np.isfinite(factor).all() and (factor > 0).all(), "ACS_PRICE_FACTOR")
    eligible = age >= 15
    out.loc[acs.person_id, "source_reporting_universe"] = eligible
    require(
        np.isnan(raw_ss[~eligible]).all() and np.isnan(adjusted[~eligible]).all(),
        "ACS_OUTSIDE_REPORTING_UNIVERSE_OBSERVATION",
    )
    require(
        np.isfinite(raw_ss[eligible]).all()
        and (
            (raw_ss[eligible] == 0)
            | ((raw_ss[eligible] >= 4) & (raw_ss[eligible] <= 55000))
        ).all()
        and (raw_ss[eligible] == np.floor(raw_ss[eligible])).all(),
        "ACS_SSP_DOMAIN",
    )
    require(
        np.array_equal(adjusted[eligible], raw_ss[eligible] * factor[eligible]),
        "ACS_SSP_ADJUSTMENT_IDENTITY",
    )
    # Below-age-15 SSP is explicitly out of the question universe. Keep it
    # unknown here; a separate reviewed reporting-unit convention is needed.
    out.loc[acs.person_id, "social_security_source_total"] = adjusted
    zero_ids = acs.person_id.to_numpy()[eligible & (raw_ss == 0)]
    out.loc[zero_ids, list(basis_owner.COMPONENTS)] = 0.0
    out.loc[zero_ids, ["allowed_" + c for c in basis_owner.COMPONENTS]] = False
    out.loc[zero_ids, "basis_origin"] = "known_total_zero"
    out.loc[acs.person_id.to_numpy()[eligible & (raw_ss > 0)], "basis_origin"] = (
        "acs_combined_positive_requires_model"
    )
    out.loc[acs.person_id.to_numpy()[~eligible], "basis_origin"] = (
        "acs_below15_outside_reporting_universe"
    )
    evidence = {
        "protocol": PROTOCOL,
        "dictionary": json.loads(json.dumps(DICTIONARY)),
        "preparation_sha256": _sha(entry[1]),
        "asec_native_sha256": _sha(issued[1]),
        "acs_native_sha256": _sha(acs_owned.payload),
        "acs_survey_year": 2024,
        "money_header_sha256": _sha(ready.header),
        "source_member_sha256": digest,
        "read_columns": list(READ_COLUMNS),
        "complete_current_source_rows": rows,
        "asec_income_year": 2024,
        "asec_interview_year": 2025,
        "price_basis_year": 2024,
        "acs_income_window": "rolling_prior_12_months_at_2024_interview",
        "acs_price_adjustment": "SSP*(ADJINC/1000000)",
        "acs_dictionary": {
            "url": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf",
            "sha256": "929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff",
            "ssp_pdf_page_1based": 45,
            "fssp_pdf_page_1based": 130,
        },
        "acs_allocation_flag": "FSSP_not_in_current_native_projection; provenance_unresolved",
        "asec_component_mapping": "unique_reported_component_only; no priority or age fallback",
        "knownness": "unknown components and outside-universe SSP retained as unknown",
        "reporting_grain": "source person report; ASEC may combine family payments",
        "individual_beneficiary_assignment_claim": False,
        "below15_convention": "both surveys retain unknown beneficiary totals; ASEC zero literal is NIU",
        "observed_component_amounts_claim": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    require(
        preparation._checked()[1] == entry[1] and parent.ready().header == ready.header,
        "SOURCE_CHANGED",
    )
    return CurrentSocialSecurityProjection(out, ordered, evidence)
