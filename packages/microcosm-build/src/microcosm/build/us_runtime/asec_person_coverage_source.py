"""Bounded, non-authoritative original-CSV ASEC PRPERTYP projection.

Only invented cohorts have been exercised. This reader neither authenticates
source bytes nor attaches observations to a Frame. Genuine use requires a
separately reviewed authentication/binding wrapper and source-read approval.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from . import source_csv_builtin as _csv_builtin

PROTOCOL = "microcosm.asec_person_coverage_source.v1"
# Existing ASEC adapters use income/source year as the cohort key. This is an
# identity convention only, not a decision about the observation's period.
SOURCE_YEARS = (2022, 2023, 2024)
KEYS = ("source_year", "PERIDNUM")
CROSSCHECKS = ("source_household_id", "A_LINENO", "A_AGE")
ROSTER_COLUMNS = KEYS + CROSSCHECKS
READ_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "PRPERTYP")
MAX_PERSONS = 600_000
MAX_MEMBER_BYTES = 1_000_000_000
_INT64_MAX = 2**63 - 1


class CoverageSourceRefusalError(ValueError):
    """Static refusal codes only; no path, key, row or source token."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CoverageSourceRefusalError(reason)


def coverage_field_contract() -> dict:
    """Fresh metadata from the public dictionaries reviewed on 2026-09-07.

    The review's receipt.json and field-pages.json establish these definitions;
    their document hashes are metadata, not original CSV authentication pins.
    """
    documents = (
        (2023, 26, "66bd6e3fe516233ab63b75c60573451222b3b3d3235d61cfed96f64608de2117"),
        (2024, 27, "761c67ea53f5c3264329b3e9ddbdd802826ba4a859122b8a8a2f29ab85b3a840"),
        (2025, 28, "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f"),
    )
    return {
        "protocol": PROTOCOL,
        "survey": "CPS ASEC",
        "grain": "person",
        "field": "PRPERTYP",
        "concept_as_printed": "Type of person record recode",
        "universe_as_printed": "All Persons",
        "printed_length": 1,
        "printed_position": 155,
        "printed_range": [-4, 3],
        "codes": {
            "1": "Child household member",
            "2": "Adult civilian household member",
            "3": "Adult Armed Forces household member",
        },
        "unlabelled_in_range_codes": [-4, -3, -2, -1, 0],
        "blank_meaning": "unresolved",
        "authority": [
            {
                "survey_year": year,
                "source_year": year - 1,
                "url": "https://www2.census.gov/programs-surveys/cps/datasets/"
                f"{year}/march/asec{year}_ddl_pub_full.pdf",
                "sha256": digest,
                "pdf_page_1based": page,
                "printed_page": "6C-7",
            }
            for year, page, digest in documents
        ],
        "join_keys": list(KEYS),
        "crosschecks": dict(zip(CROSSCHECKS, READ_COLUMNS[1:4], strict=True)),
        "roster_scope": "all_person_rows_in_each_of_the_three_source_cohorts",
        "period_harmonized": False,
        "cross_survey_coverage_equivalence_established": False,
    }


def _state(token: str) -> str:
    if token in ("1", "2", "3"):
        return "observed_code"
    if token == "":
        return "blank_unresolved"
    if token in ("-4", "-3", "-2", "-1", "0"):
        return "unlabelled_in_range"
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", token) and token != "-0":
        return "out_of_range"
    return "malformed_token"


def _roster(person_roster: pd.DataFrame) -> pd.DataFrame:
    _require(
        isinstance(person_roster, pd.DataFrame)
        and person_roster.columns.is_unique
        and set(ROSTER_COLUMNS) <= set(person_roster.columns),
        "ROSTER_COLUMNS",
    )
    _require(0 < len(person_roster) <= MAX_PERSONS, "ROSTER_ROWS")
    result = person_roster.loc[:, list(ROSTER_COLUMNS)].copy().reset_index(drop=True)
    _require(not result.isna().any().any(), "ROSTER_MISSING")
    _require(
        all(str(result[name].dtype) == "int64" for name in (KEYS[0], *CROSSCHECKS)),
        "ROSTER_INTEGER_DTYPE",
    )
    _require(set(result.source_year) == set(SOURCE_YEARS), "ROSTER_COHORTS")
    _require(
        result.PERIDNUM.map(
            lambda value: (
                isinstance(value, str) and re.fullmatch(r"[0-9]{22}", value) is not None
            )
        ).all(),
        "ROSTER_PERSON_KEY",
    )
    _require(
        (result.source_household_id > 0).all()
        and (result.A_LINENO > 0).all()
        and (result.A_AGE >= 0).all(),
        "ROSTER_COORDINATES",
    )
    _require(not result.duplicated(list(KEYS)).any(), "ROSTER_DUPLICATE_KEY")
    _require(
        not result.duplicated([KEYS[0], *CROSSCHECKS[:2]]).any(),
        "ROSTER_DUPLICATE_COORDINATES",
    )
    return result


def _coordinate(token: str) -> int:
    _require(re.fullmatch(r"[0-9]{1,19}", token) is not None, "MEMBER_INTEGER")
    number = int(token)
    _require(number <= _INT64_MAX, "MEMBER_INTEGER")
    return number


def _digest(table: pd.DataFrame) -> str:
    # Stream canonical records into the hash; never log native values or retain
    # a second full JSON copy. Row order is part of the projection identity.
    digest = hashlib.sha256()
    digest.update(json.dumps(list(table.columns), separators=(",", ":")).encode())
    for row in table.itertuples(index=False, name=None):
        digest.update(b"\n")
        digest.update(json.dumps(row, separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


def _regular_member_opener(path, flags):
    # Check the opened descriptor, not a race-prone path snapshot. Nonblocking
    # open lets us refuse a FIFO even when no writer has opened its other end.
    descriptor = os.open(path, flags | os.O_NONBLOCK)
    try:
        snapshot = os.fstat(descriptor)
        _require(stat.S_ISREG(snapshot.st_mode), "MEMBER_FILE_TYPE")
        _require(snapshot.st_size <= MAX_MEMBER_BYTES, "MEMBER_BYTES")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class _BoundedMember(io.RawIOBase):
    """Bound raw reads beneath buffering/decoding; the caller owns the raw file."""

    def __init__(self, raw):
        super().__init__()
        self._raw = raw
        self._remaining = MAX_MEMBER_BYTES

    def readable(self):
        return True

    def readinto(self, buffer):
        # One excess byte distinguishes exact-limit EOF from a growing member.
        # Never pass that probe byte to the decoder or issue an unbounded read.
        count = self._raw.readinto(memoryview(buffer)[: self._remaining + 1])
        self._remaining -= count
        _require(self._remaining >= 0, "MEMBER_BYTES")
        return count


def read_asec_person_coverage_source(
    member_paths: Mapping[int, str | Path], *, person_roster: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Read exact whole cohorts, returning literals in requested native order.

    Paths are keyed by source/income years 2022, 2023, 2024 (survey years
    2023, 2024, 2025), following the existing ASEC source adapters. The caller
    supplies PERIDNUM and int64 source_year/source_household_id/A_LINENO/A_AGE.
    Every CSV row must match exactly one requested person in its own cohort;
    PH_SEQ, A_LINENO and A_AGE crosscheck the requested native coordinates.
    Age is only a crosscheck: it never supplies or overrides PRPERTYP meaning.

    All PRPERTYP cells remain strings, including blanks and malformed tokens.
    The state column describes lexical/printed-code evidence, not coverage or
    eligibility. Receipt digests bind this local projection, not source custody.
    """
    csv_reader = _csv_builtin.capture_csv_reader(csv)
    _require(csv_reader is not None, "SOURCE_CSV_READER_CHANGED")
    _require(
        isinstance(member_paths, Mapping)
        and set(member_paths) == set(SOURCE_YEARS)
        and all(type(year) is int for year in member_paths)
        and all(isinstance(path, (str, Path)) for path in member_paths.values()),
        "MEMBER_PATHS",
    )
    expected = _roster(person_roster)
    lookup = {
        row[:2]: (position, row[2:])
        for position, row in enumerate(expected.itertuples(index=False, name=None))
    }
    tokens = [""] * len(expected)
    seen = set()
    cohort_counts = []
    for year in SOURCE_YEARS:
        coordinates = set()
        rows = 0
        try:
            with (
                open(
                    member_paths[year],
                    "rb",
                    buffering=0,
                    opener=_regular_member_opener,
                ) as raw,
                io.BufferedReader(_BoundedMember(raw)) as bounded,
                io.TextIOWrapper(bounded, encoding="utf-8", newline="") as handle,
            ):
                reader = csv_reader(handle, strict=True)
                header = next(reader, [])
                _require(
                    len(header) == len(set(header))
                    and set(READ_COLUMNS) <= set(header),
                    "MEMBER_HEADER",
                )
                indices = [header.index(name) for name in READ_COLUMNS]
                for cells in reader:
                    rows += 1
                    _require(len(seen) < MAX_PERSONS, "MEMBER_ROWS")
                    _require(len(cells) == len(header), "MEMBER_ROW_WIDTH")
                    key, household, line, age, token = (cells[i] for i in indices)
                    _require(
                        re.fullmatch(r"[0-9]{22}", key) is not None, "MEMBER_PERSON_KEY"
                    )
                    native_key = (year, key)
                    _require(native_key not in seen, "MEMBER_DUPLICATE_KEY")
                    _require(native_key in lookup, "MEMBER_EXTRA_KEY")
                    actual = tuple(
                        _coordinate(value) for value in (household, line, age)
                    )
                    _require(actual[0] > 0 and actual[1] > 0, "MEMBER_COORDINATES")
                    _require(
                        actual[:2] not in coordinates, "MEMBER_DUPLICATE_COORDINATES"
                    )
                    position, wanted = lookup[native_key]
                    _require(actual == wanted, "MEMBER_CROSSCHECK")
                    coordinates.add(actual[:2])
                    seen.add(native_key)
                    tokens[position] = token
        except CoverageSourceRefusalError:
            raise
        except (OSError, UnicodeError, csv.Error, ValueError, OverflowError):
            raise CoverageSourceRefusalError("MEMBER_READ") from None
        cohort_counts.append(
            {"source_year": year, "survey_year": year + 1, "rows": rows}
        )
    _require(len(seen) == len(expected), "MEMBER_MISSING_KEY")
    result = expected.copy()
    result["PRPERTYP"] = pd.array(tokens, dtype="string")
    result["PRPERTYP_state"] = pd.array(
        [_state(token) for token in tokens], dtype="string"
    )
    receipt = {
        "protocol": PROTOCOL,
        "source_authenticated": False,
        "population_binding_authenticated": False,
        "coverage_status": "literal_source_fields_only",
        "release_eligible": False,
        "contract": coverage_field_contract(),
        "read_columns": list(READ_COLUMNS),
        "cohorts": cohort_counts,
        "person_rows": len(result),
        "native_roster_sha256": _digest(expected),
        "projection_sha256": _digest(result),
    }
    return result, receipt
