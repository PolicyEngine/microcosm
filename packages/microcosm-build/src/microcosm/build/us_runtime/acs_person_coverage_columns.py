"""Additive, literal ACS person coverage fields for an exact native roster.

This reader preserves MIL and ESR separately, including their different age
universes. It performs no population-domain allocation and authenticates no
source file. A source owner must bind the archive and requested native keys
before these observations may support a genuine graph. The existing ACS
loader, its columns and its implementation identity are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .acs_pums import AcsPumsSource
from .source_csv_builtin import capture_csv_reader

PROTOCOL = "microcosm.acs-person-coverage-columns.v1"
DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/"
    "PUMS_Data_Dictionary_2024.pdf"
)
DICTIONARY_SHA256 = "929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff"
KEYS = ("SERIALNO", "SPORDER")
READ_COLUMNS = (*KEYS, "AGEP", "MIL", "ESR")
MAX_ROWS = 6_000_000
MAX_SELECTED_ROWS = 1_000_000
MAX_CSV_RECORD_CHARS = 100_000
_NON_CSV_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def coverage_field_contract() -> dict:
    """A fresh source-definition document, suitable for provenance consumers."""
    return {
        "protocol": PROTOCOL,
        "survey": "ACS",
        "survey_year": 2024,
        "record_grain": "person",
        "read_columns": list(READ_COLUMNS),
        "dictionary": {"url": DICTIONARY_URL, "sha256": DICTIONARY_SHA256},
        "fields": {
            "MIL": {
                "pdf_page": 41,
                "minimum_age": 17,
                "blank_meaning": "outside age universe",
                "codes": {
                    "1": "active_duty",
                    "2": "past_active_duty",
                    "3": "training_only",
                    "4": "never_served",
                },
            },
            "ESR": {
                "pdf_page": 56,
                "minimum_age": 16,
                "blank_meaning": "outside age universe",
                "codes": {
                    "1": "civilian_working",
                    "2": "civilian_job_absent",
                    "3": "unemployed",
                    "4": "armed_forces_working",
                    "5": "armed_forces_job_absent",
                    "6": "outside_labor_force",
                },
            },
        },
        "cross_survey_residence_equivalence_established": False,
    }


def _json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _keys(table: pd.DataFrame) -> pd.DataFrame:
    """Canonical native identities; no row-position join or key imputation."""
    if (
        not isinstance(table, pd.DataFrame)
        or not table.columns.is_unique
        or not set(KEYS).issubset(table)
    ):
        raise ValueError("ACS coverage requires native person keys")
    result = table.loc[:, list(KEYS)].copy()
    if result.isna().any().any():
        raise ValueError("ACS coverage native keys are missing")
    serial = result.SERIALNO
    if not serial.map(
        lambda x: (
            isinstance(x, str) and re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", x) is not None
        )
    ).all():
        raise ValueError("ACS coverage native serial is not a 2024 HU/GQ identity")
    order = result.SPORDER.astype("string")
    if not order.str.fullmatch(r"[0-9]{1,2}").all():
        raise ValueError("ACS coverage native person order is not an integer")
    result["SPORDER"] = order.astype("int64")
    if not result.SPORDER.between(1, 20).all():
        raise ValueError("ACS coverage native person order is outside 1..20")
    if result.duplicated(list(KEYS)).any():
        raise ValueError("ACS coverage native person keys repeat")
    return result


def _field_state(age: str, raw: str, *, minimum_age: int, codes: dict) -> str:
    if not re.fullmatch(r"[0-9]{1,2}", age):
        return "age_unresolved"
    if int(age) < minimum_age:
        return "outside_age_universe" if raw == "" else "value_below_age_universe"
    if raw == "":
        return "missing_in_universe"
    return "observed_code" if raw in codes else "unlabelled_code"


def _literal_csv_records(stream):
    """Check controls before CSV parsing, bounding each complete logical record."""
    csv_reader = capture_csv_reader(csv)
    if csv_reader is None:
        raise ValueError("SOURCE_CSV_READER_CHANGED")
    record_chars = 0

    def lines():
        nonlocal record_chars
        while line := stream.readline(MAX_CSV_RECORD_CHARS - record_chars + 1):
            record_chars += len(line)
            if record_chars > MAX_CSV_RECORD_CHARS:
                raise ValueError("ACS coverage CSV record character bound exceeded")
            # HT is literal whitespace; CR/LF delimit records or remain literal
            # inside quotes. newline="" prevents universal-newline rewriting.
            if _NON_CSV_CONTROLS.search(line):
                raise ValueError("ACS coverage CSV contains a forbidden control")
            yield line

    reader = csv_reader(lines(), strict=True)
    try:
        while True:
            record_chars = 0
            record = next(reader, None)
            if record is None:
                return
            yield record
    except (csv.Error, UnicodeError) as exc:
        raise ValueError("ACS coverage invalid literal CSV") from exc


def _scan_acs_person_coverage(source, consume_row):
    """Exhaust the literal source once before returning its member/row counts.

    The private consumer receives only exact READ_COLUMNS strings. It cannot
    signal successful early completion: all members, records and widths are
    checked unless an exception refuses the scan. Source custody and requested
    roster checks remain the caller's responsibility.
    """
    if type(source) is not AcsPumsSource or source.vintage != 2024:
        raise ValueError("ACS coverage projection requires the 2024 source type")
    rows = 0
    with ZipFile(source.person_zip) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.lower().startswith("psam_pus")
            and name.lower().endswith(".csv")
        )
        if not members or len(members) != len(set(members)):
            raise ValueError(
                "ACS coverage person member roster is missing or duplicated"
            )
        for name in members:
            with (
                archive.open(name) as member,
                io.TextIOWrapper(member, encoding="utf-8-sig", newline="") as stream,
            ):
                reader = _literal_csv_records(stream)
                header = next(reader, [])
                if len(header) != len(set(header)) or not set(READ_COLUMNS).issubset(
                    header
                ):
                    raise ValueError(
                        "ACS coverage source columns are missing or duplicated"
                    )
                positions = [header.index(column) for column in READ_COLUMNS]
                for record in reader:
                    rows += 1
                    if rows > MAX_ROWS:
                        raise ValueError("ACS coverage source row bound exceeded")
                    if len(record) != len(header):
                        raise ValueError(
                            "ACS coverage CSV record width differs from header"
                        )
                    consume_row(tuple(record[position] for position in positions))
    return members, rows


def read_acs_person_coverage_columns(
    source: AcsPumsSource,
    *,
    person_keys: pd.DataFrame,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, dict]:
    """Stream an additive person projection and align exact selected households.

    ``person_keys`` must include every person in each requested source
    household. The read refuses missing, duplicate or extra native person keys
    in those households, including differences discovered in later members.
    Other households are streamed past. The separate max_households option on
    the legacy source is irrelevant here: the exact native roster is authority.

    AGEP, MIL and ESR remain literal strings, including empty Census cells.
    Each member is parsed once, with full record widths checked before column
    or household selection. Non-CSV C0/C1 controls (including NUL) refuse;
    tabs and quoted CR/LF remain literal. Records, including their physical
    line endings, are bounded to MAX_CSV_RECORD_CHARS decoded characters.
    State columns distinguish printed codes, out-of-universe blanks, missing
    in-universe values, unexpected codes and age conflicts. They are evidence
    states, not a combined military predicate or survey membership decision.

    Only invented archives have been executed in development. Genuine use
    requires source authentication and a separately bounded source-read plan.
    """
    if type(source) is not AcsPumsSource or source.vintage != 2024:
        raise ValueError("ACS coverage projection requires the 2024 source type")
    if type(chunksize) is not int or not 0 < chunksize <= 100_000:
        raise ValueError("ACS coverage chunksize must be in 1..100000")
    expected = _keys(person_keys).reset_index(drop=True)
    if not 0 < len(expected) <= MAX_SELECTED_ROWS:
        raise ValueError("ACS coverage selected person count is outside the bound")
    retained = frozenset(expected.SERIALNO)
    pieces, batch, selected_rows = [], [], 0

    def consume_row(cells):
        nonlocal batch, selected_rows
        if cells[0] not in retained:
            return
        selected_rows += 1
        if selected_rows > len(expected):
            raise ValueError("ACS coverage source has extra selected-household people")
        batch.append(cells)
        if len(batch) == chunksize:
            pieces.append(pd.DataFrame(batch, columns=READ_COLUMNS, dtype="string"))
            batch = []

    members, rows = _scan_acs_person_coverage(source, consume_row)
    if batch:
        pieces.append(pd.DataFrame(batch, columns=READ_COLUMNS, dtype="string"))
    if not pieces:
        raise ValueError("ACS coverage requested persons are missing")
    observed = pd.concat(pieces, ignore_index=True)
    observed_keys = _keys(observed)
    observed["SPORDER"] = observed_keys.SPORDER
    wanted = pd.MultiIndex.from_frame(expected)
    actual = pd.MultiIndex.from_frame(observed_keys)
    if set(wanted) != set(actual):
        raise ValueError("ACS coverage source and requested native roster differ")
    result = observed.set_index(list(KEYS)).loc[wanted].reset_index()
    contract = coverage_field_contract()
    for field, definition in contract["fields"].items():
        result[field + "_state"] = pd.Series(
            [
                _field_state(
                    age,
                    raw,
                    minimum_age=definition["minimum_age"],
                    codes=definition["codes"],
                )
                for age, raw in zip(result.AGEP, result[field], strict=True)
            ],
            dtype="string",
        )
    receipt = {
        "protocol": PROTOCOL,
        "contract_sha256": hashlib.sha256(_json(contract)).hexdigest(),
        "source_authenticated": False,
        "coverage_status": "literal_source_fields_only",
        "release_eligible": False,
        "member_names": members,
        "source_rows_streamed": rows,
        "selected_person_rows": len(result),
        "native_roster_sha256": hashlib.sha256(
            _json(expected.to_dict("records"))
        ).hexdigest(),
        "projection_sha256": hashlib.sha256(
            _json(result.to_dict("records"))
        ).hexdigest(),
        "legacy_max_households_applied": False,
        "read_columns": list(READ_COLUMNS),
    }
    return result, receipt
