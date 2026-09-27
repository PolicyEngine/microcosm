"""Closed original ASEC household observations, without coverage inference.

The byte/CSV primitives belong to the separate original-weight owner. This
module composes them with a new literal field contract; it never projects HSUP_WGT,
assigns survey domains, binds a Frame or decides cross-survey residence rules.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass, field
from importlib import resources
from pathlib import Path

from microcosm.build.cd_benchmark.origin import (
    AsecHouseholdOrigin,
    SourceMember,
    origin_key,
)

from . import asec_original_household_weights as shared
from . import source_csv_builtin as _csv_builtin

COLUMNS = ("H_SEQ", "H_HHTYPE", "HRHTYPE", "H_LIVQRT", "H_NUMPER")
_TOKEN = object()
_SCHEMA = "microcosm.us.asec-household-coverage-fields.v1"


def _field(concept, length, position, page, universe, labels, **extra):
    return {
        "concept": concept,
        "printed_length": length,
        "position_by_survey_year": {
            str(year): offset
            for year, offset in zip(
                (2023, 2024, 2025),
                (position,) * 3 if isinstance(position, int) else position,
                strict=True,
            )
        },
        "printed_page": f"6A-{page}",
        "pdf_page_1based_by_survey_year": {
            str(year): page if year < 2025 else page + 7 for year in (2023, 2024, 2025)
        },
        "universe_as_printed": universe,
        "labels": {str(code): label for code, label in labels.items()},
        "literal_preserved": True,
        **extra,
    }


_FIELDS = {
    "H_SEQ": _field(
        "Household sequence number",
        5,
        29,
        1,
        "All Households",
        {},
        printed_range="00001:99999",
        role="native_key",
    ),
    "H_HHTYPE": _field(
        "Type of household interview",
        1,
        61,
        2,
        "All Households",
        {1: "Interview", 2: "Type A non-interview", 3: "Type B/C non-interview"},
        printed_range="1:3",
    ),
    "HRHTYPE": _field(
        "Household type",
        2,
        72,
        3,
        "H_HHTYPE = 1",
        {
            0: "Non-interview household",
            1: "Married couple primary family (neither spouse in Armed Forces)",
            2: "Married couple primary family (one spouse in Armed Forces)",
            3: "Unmarried civilian male primary family householder",
            4: "Unmarried civilian female primary family householder",
            5: "Primary family household - reference person in Armed Forces and unmarried",
            6: "Civilian male nonfamily householder",
            7: "Civilian female nonfamily householder",
            8: "Nonfamily householder household - reference person in Armed Forces",
            9: "Group quarters with actual families (This is new in 1994)",
            10: "Group quarters with secondary individuals only",
        },
        printed_range="00:10",
        universe_and_noninterview_label_both_preserved=True,
    ),
    "H_LIVQRT": _field(
        "Type of living quarters (recode)",
        2,
        62,
        2,
        "All Households",
        {
            1: "House, apt., flat",
            2: "HU in nontransient hotel, etc.",
            3: "HU, perm, in trans. hotel, motel, etc.",
            4: "HU in rooming house",
            5: "Mobile home or trailer with no permanent room added",
            6: "Mobile home or trailer with 1 or more perm rooms added",
            7: "HU not specified above",
            8: "Qtrs not hu in rooming or boarding house",
            9: "Unit not perm in trans. hotel, motel, etc.",
            10: "Tent or trailer site",
            11: "Student quarters in college dormitory",
            12: "Other not HU",
        },
        printed_range="01:12",
        printed_groupings={"01-07": "Housing unit", "08-12": "Other Unit"},
    ),
    "H_NUMPER": _field(
        "Number of persons in household",
        2,
        (82, 82, 84),
        3,
        "H_HHTYPE = 1",
        {
            0: "Noninterview household",
            **{n: "Number of persons in HHLD" for n in range(1, 17)},
        },
        printed_range="0:16",
        printed_member_count_codes="01-16",
        universe_and_noninterview_label_both_preserved=True,
    ),
}
_INTERPRETATION_NOTES = {
    "student_quarters": {
        "dictionary_code_retained": "H_LIVQRT = 11",
        "methodology_statement": "Student quarters excluded from the ASEC GQ sample since 2018",
        "methodology_url": "https://www.census.gov/topics/income-poverty/guidance/group-quarters.html",
        "methodology_page_last_revised": "2023-06-23",
        "interpretation": "unresolved",
        "observed_presence_asserted": False,
    },
    "household_and_quarters": "Different field labels do not establish identical or exclusive cross-survey frames",
    "legacy_universe_names": "H_TYPE in adjacent dictionary universe text is not silently substituted for HRHTYPE",
    "person_count": "H_NUMPER is an original reported household field, not an authenticated person-roster count",
}


class HouseholdCoverageSourceError(ValueError):
    """Static refusal reason without source rows, native identifiers or paths."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HouseholdCoverageSourceError(reason)


def _producer() -> dict:
    _require(_csv_builtin.csv_reader_bound(csv), "SOURCE_CSV_READER_CHANGED")
    return {
        "protocol": _SCHEMA,
        "owner_sha256": shared._sha(
            resources.files(__package__)
            .joinpath("asec_household_coverage_fields.py")
            .read_bytes()
        ),
        "bounded_reader_dependency": shared._implementation(),
        "csv_builtin_sha256": shared._sha(
            resources.files(__package__).joinpath("source_csv_builtin.py").read_bytes()
        ),
        "columns": COLUMNS,
        "fields": _FIELDS,
        "interpretation_notes": _INTERPRETATION_NOTES,
    }


def _code_state(name: str, token: str, interview_code: int | None) -> dict:
    contract = _FIELDS[name]
    code, label = None, None
    if token == "":
        status = "missing"
    elif re.fullmatch(rf"[0-9]{{1,{contract['printed_length']}}}", token) is None:
        status = "malformed"
    else:
        code = int(token)
        label = contract["labels"].get(str(code))
        status = "valid" if label is not None else "unlabelled"
    universe = (
        "in"
        if contract["universe_as_printed"] == "All Households" or interview_code == 1
        else "outside"
        if interview_code in (2, 3)
        else "unresolved"
    )
    return {
        "code": code,
        "label": label,
        "code_status": status,
        "universe_status": universe,
    }


def _project(row: Sequence[str]) -> dict:
    _require(
        all(len(token) <= shared._MAX_FIELD_CHARS for token in row), "MEMBER_FIELD_SIZE"
    )
    native = row[0]
    _require(
        re.fullmatch(r"[0-9]+", native) is not None and 1 <= int(native) <= 99999,
        "MEMBER_NATIVE_KEY",
    )
    literals = dict(zip(COLUMNS, row, strict=True))
    interview = _code_state("H_HHTYPE", literals["H_HHTYPE"], None)
    interview_code = interview["code"] if interview["code_status"] == "valid" else None
    states = {
        name: _code_state(name, literals[name], interview_code) for name in COLUMNS[1:]
    }
    hr = (
        states["HRHTYPE"]["code"]
        if states["HRHTYPE"]["code_status"] == "valid"
        else None
    )
    count = (
        states["H_NUMPER"]["code"]
        if states["H_NUMPER"]["code_status"] == "valid"
        else None
    )
    diagnostics = []
    if interview_code == 1:
        if hr == 0:
            diagnostics.append("interview_with_noninterview_household_type")
        if count == 0:
            diagnostics.append("interview_with_noninterview_person_count")
    elif interview_code in (2, 3):
        if hr is not None and hr > 0:
            diagnostics.append("noninterview_with_interview_household_type")
        if count is not None and count > 0:
            diagnostics.append("noninterview_with_positive_person_count")
    if states["H_LIVQRT"]["code"] == 11:
        diagnostics.append("student_quarters_dictionary_methodology_unresolved")
    return {
        **literals,
        "native_household_id": int(native),
        "field_states": states,
        "reported_person_count": count
        if interview_code == 1 and count is not None and count > 0
        else None,
        "diagnostics": diagnostics,
        "coverage_interpretation": "unresolved",
    }


def _read_capture(capture: Path, pin) -> list[dict]:
    """Compose existing bounded byte/record primitives with this field projection."""
    csv_reader = _csv_builtin.capture_csv_reader(csv)
    _require(csv_reader is not None, "SOURCE_CSV_READER_CHANGED")
    result, seen = [], set()
    member = SourceMember(pin.canonical_member_id, pin.member_sha256)
    with open(capture, "rb", buffering=0, opener=shared._regular_opener) as raw:
        before = os.fstat(raw.fileno())
        _require(before.st_size == pin.size_bytes, "CAPTURE_SIZE")
        digest = shared._DigestReader(raw, pin.size_bytes)
        with io.TextIOWrapper(
            io.BufferedReader(digest), encoding="utf-8-sig", newline=""
        ) as text:
            lines = shared._RecordLines(text)
            reader = csv_reader(lines, strict=True)
            header = next(reader, [])
            _require(
                bool(header)
                and all(header)
                and len(set(header)) == len(header)
                and set(COLUMNS) <= set(header),
                "MEMBER_HEADER",
            )
            positions = [header.index(column) for column in COLUMNS]
            while True:
                lines.characters = 0
                try:
                    row = next(reader)
                except StopIteration:
                    break
                _require(len(row) == len(header), "MEMBER_ROW_WIDTH")
                _require(len(result) < pin.rows, "MEMBER_ROWS")
                record = _project([row[i] for i in positions])
                native = record["native_household_id"]
                _require(native not in seen, "MEMBER_DUPLICATE_KEY")
                seen.add(native)
                record.update(
                    income_year=pin.income_year,
                    survey_year=pin.survey_year,
                    member_id=pin.canonical_member_id,
                    member_sha256=pin.member_sha256,
                    member_row_1based=len(result) + 1,
                    origin_key=origin_key(
                        AsecHouseholdOrigin(member, pin.income_year, native)
                    ),
                )
                result.append(record)
            _require(len(result) == pin.rows, "MEMBER_ROWS")
            _require(
                digest.count == pin.size_bytes
                and digest.digest.hexdigest() == pin.member_sha256,
                "CAPTURE_CHANGED",
            )
            after = os.fstat(raw.fileno())
            _require(
                all(
                    getattr(before, key) == getattr(after, key)
                    for key in (
                        "st_dev",
                        "st_ino",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                ),
                "CAPTURE_CHANGED",
            )
    return result


@dataclass(frozen=True, slots=True, weakref_slot=True)
class AuthenticatedAsecHouseholdCoverageFields:
    """Reader-issued immutable original observations, without Frame authority."""

    payload: bytes
    _token: InitVar[object] = None
    _issued_sha256: str = field(init=False, repr=False)

    def __post_init__(self, _token: object) -> None:
        _require(_token is _TOKEN, "SOURCE_CONSTRUCTOR")
        try:
            digest = shared._issue_capsule(self, self.payload)
        except shared.HouseholdWeightSourceError as error:
            raise HouseholdCoverageSourceError(str(error)) from None
        object.__setattr__(self, "_issued_sha256", digest)

    @property
    def document(self) -> dict:
        """Return a defensive view after checking issuance and current producer."""
        return verify_asec_household_coverage_fields(self)

    def households_for(
        self, native_roster: Sequence[tuple[int, int]]
    ) -> tuple[dict, ...]:
        """Look up exact cohort/native keys; preserve unresolved observations."""
        document = self.document
        _require(
            isinstance(native_roster, (list, tuple))
            and 0 < len(native_roster) <= shared._MAX_ROWS,
            "LOOKUP_ROSTER",
        )
        keys = []
        for key in native_roster:
            _require(
                isinstance(key, (list, tuple))
                and len(key) == 2
                and all(type(value) is int for value in key),
                "LOOKUP_ROSTER",
            )
            keys.append(tuple(key))
        _require(len(set(keys)) == len(keys), "LOOKUP_ROSTER")
        records = {
            (row["income_year"], row["native_household_id"]): row
            for row in document["records"]
        }
        _require(all(key in records for key in keys), "LOOKUP_UNKNOWN")
        return tuple(records[key] for key in keys)


def verify_asec_household_coverage_fields(
    source: AuthenticatedAsecHouseholdCoverageFields,
) -> dict:
    """Validate a reader-issued capsule; JSON is not an admission credential."""
    _require(type(source) is AuthenticatedAsecHouseholdCoverageFields, "SOURCE_TYPE")
    try:
        payload = shared._checked_capsule_payload(source)
        document = json.loads(payload)
        _require(
            shared._encode(document["producer"]) == shared._encode(_producer()),
            "PRODUCER_CHANGED",
        )
    except shared.HouseholdWeightSourceError as error:
        raise HouseholdCoverageSourceError(str(error)) from None
    return document


def load_authenticated_asec_household_coverage_fields(
    member_paths: Mapping[int, str | Path], *, candidate: bytes | None = None
) -> AuthenticatedAsecHouseholdCoverageFields:
    """Reconstruct whole requested original cohorts, never caller-provided rows."""
    try:
        pins = shared._registry()
        paths = shared._member_path_snapshot(member_paths, pins)
        _require(
            candidate is None
            or (
                type(candidate) is bytes and len(candidate) <= shared._MAX_PAYLOAD_BYTES
            ),
            "CANDIDATE_SIZE",
        )
        producer_document = _producer()
        _require(
            shared._encode(producer_document["bounded_reader_dependency"]["registry"])
            == shared._encode([asdict(pin) for pin in pins]),
            "REGISTRY_TRANSITION",
        )
        producer = shared._encode(producer_document)
        selected = [pin for pin in pins if pin.income_year in paths]
        _require(bool(selected), "MEMBER_PATHS")
        _require(sum(pin.rows for pin in selected) <= shared._MAX_ROWS, "REQUEST_ROWS")
        records = []
        with tempfile.TemporaryDirectory(
            prefix="asec-household-coverage-fields-"
        ) as tmp:
            for pin in selected:
                capture = Path(tmp) / f"{pin.income_year}.csv"
                try:
                    digest = shared._capture_owner._snapshot(
                        paths[pin.income_year], capture, size=pin.size_bytes
                    )
                except (OSError, ValueError, TypeError):
                    raise HouseholdCoverageSourceError("MEMBER_CAPTURE") from None
                _require(digest == pin.member_sha256, "MEMBER_SHA256")
                try:
                    records.extend(_read_capture(capture, pin))
                except (
                    HouseholdCoverageSourceError,
                    shared.HouseholdWeightSourceError,
                ):
                    raise
                except (OSError, UnicodeError, csv.Error, ValueError, OverflowError):
                    raise HouseholdCoverageSourceError("MEMBER_ENCODING") from None
        _require(shared._encode(_producer()) == producer, "PRODUCER_CHANGED")
        payload = shared._encode(
            {
                "schema": _SCHEMA,
                "producer": json.loads(producer),
                "members": [asdict(pin) for pin in selected],
                "fields": _FIELDS,
                "dictionary_authorities": [
                    {
                        "survey_year": year,
                        "sha256": digest,
                        "url": "https://www2.census.gov/programs-surveys/cps/datasets/"
                        f"{year}/march/asec{year}_ddl_pub_full.pdf",
                    }
                    for year, _page, digest in shared._DICTIONARIES
                ],
                "interpretation_notes": _INTERPRETATION_NOTES,
                "native_roster_sha256": shared._sha(
                    shared._encode(
                        [
                            [row["income_year"], row["native_household_id"]]
                            for row in records
                        ]
                    )
                ),
                "projection_sha256": shared._sha(shared._encode(records)),
                "records": records,
                "source_authenticated": True,
                "population_binding_authenticated": False,
                "release_eligible": False,
                "coverage_classification_performed": False,
            }
        )
        _require(candidate is None or candidate == payload, "CANDIDATE_MISMATCH")
        return AuthenticatedAsecHouseholdCoverageFields(payload, _token=_TOKEN)
    except shared.HouseholdWeightSourceError as error:
        raise HouseholdCoverageSourceError(str(error)) from None
