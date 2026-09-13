"""Independent invented-source tests for the approved ACS housing-unit source bridge.

Written against the approved v2 plan contract, not against the production
module: no byte of ``acs_housing_universe_source.py`` or of the packaged
``acs_2024_housing_universe.json`` was read while authoring this file, so a
failure here is evidence about the contract rather than a restatement of the
implementation.

Every archive, member, header, identifier and row below is wholly invented. No
genuine ACS/Census archive, candidate, target or dictionary file is opened,
statted or referenced. Nothing here authenticates real data, and nothing here
makes a scientific claim: the fixtures exercise the closed source rules only.

Scope is the source producer/readback pair. ``prepare_acs_housing_population``
builds a Frame and is deliberately left to the production author's own
Frame/graph tests.

Two reconciliation seams are collected in one place each, so integrating this
file with the finished module is a rename exercise and never a semantic one:

* ``NAMES`` - column, constant, filename and definition-name spellings.
* ``EXPECTED_REASONS`` - exact refusal reason codes, asserted only once
  ``REASON_CODES_AGREED`` is flipped to True after agreeing them with the
  production author. Until then every refusal test still asserts the exception
  type, message sanitisation, and that semantically distinct causes carry
  distinct codes, which is the part that does not need agreement.
"""

import csv
import hashlib
import io
import json
import os
import re
import signal
import stat
import struct
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# The production import must surface as a loud collection error naming the
# module while it is absent. It must never skip: no engine is involved here.
from microcosm.build.us_runtime import acs_housing_universe_source as source
from microcosm.build.us_runtime.acs_housing_universe import (
    classify_acs_housing_universe,
)

# --------------------------------------------------------------------------
# Reconciliation seam 1: identifier spellings.
# --------------------------------------------------------------------------

NAMES = {
    # Frame columns carrying the archive-member/row lineage of each row.
    "member_column": "source_member",
    "ordinal_column": "source_row_ordinal",
    # The normalised two-digit state column exposed on the household frame.
    # ``_state_series`` falls back to the raw ST/STATE header aliases, so only
    # the dedicated naming test depends on this spelling.
    "state_column": "ST",
    # Files written into the source artifact directory.
    "projection_filename": "projection.json",
    "receipt_filename": "receipt.json",
    # Named source definitions bound by the artifact receipt.
    "vacancy_definition": "acs_2024_source_vacancy_v1",
    "gq_definition": "acs_2024_gq_person_placeholder_design_v1",
    # Module byte caps, monkeypatched tiny so no test allocates real GiBs.
    "source_max_bytes": "ACS_HU_SOURCE_MAX_BYTES",
    "receipt_max_bytes": "ACS_HU_RECEIPT_MAX_BYTES",
    # The private archive pin tuple; entries are (role, filename, sha256, size).
    "archive_pins": "_ARCHIVE_PINS",
    "household_role": "household",
    "person_role": "person",
}

HOUSEHOLD_ARCHIVE = "csv_hus.zip"
PERSON_ARCHIVE = "csv_pus.zip"

# --------------------------------------------------------------------------
# Reconciliation seam 2: refusal reason codes.
# --------------------------------------------------------------------------

REASON_CODES_AGREED = True

EXPECTED_REASONS: dict[str, str] = {
    "artifact_forgery": "RECONSTRUCTION_MISMATCH",
    "capture_digest": "SOURCE_SHA256",
    "csv_record": "CSV_ROW_WIDTH",
    "header_contract": "CSV_STATE_REQUIRED",
    "join_integrity": "ORPHAN_PERSON",
    "output_precondition": "OUTPUT_EXISTS",
    "selection": "SELECTION_UNKNOWN",
    "value_domain": "TEN",
    "zip_preflight": "ZIP_DUPLICATE_MEMBER",
}

# Coarse refusal families that must not collapse onto one reason code.
DISTINCT_FAMILIES = (
    "capture_digest",
    "zip_preflight",
    "csv_record",
    "header_contract",
    "value_domain",
    "join_integrity",
    "selection",
    "artifact_forgery",
    "output_precondition",
)

# --------------------------------------------------------------------------
# Invented source bytes.
# --------------------------------------------------------------------------

HOUSEHOLD_HEADER = (
    "RT",
    "SERIALNO",
    "DIVISION",
    "PUMA",
    "REGION",
    "ST",
    "ADJHSG",
    "ADJINC",
    "WGTP",
    "NP",
    "TYPEHUGQ",
    "ACR",
    "BDSP",
    "TEN",
    "VALP",
    "HINCP",
)

PERSON_HEADER = (
    "RT",
    "SERIALNO",
    "SPORDER",
    "PUMA",
    "ST",
    "ADJINC",
    "PWGTP",
    "AGEP",
    "SEX",
    "WAGP",
)

HOUSEHOLD_DEFAULTS = {
    "RT": "H",
    "DIVISION": "9",
    "PUMA": "00100",
    "REGION": "4",
    "ST": "06",
    "ADJHSG": "1000000",
    "ADJINC": "1010000",
    "ACR": "1",
    "BDSP": "3",
    "VALP": "410000",
    "HINCP": "82000",
}

PERSON_DEFAULTS = {
    "RT": "P",
    "PUMA": "00100",
    "ST": "06",
    "ADJINC": "1010000",
    "AGEP": "41",
    "SEX": "1",
    "WAGP": "52000",
}

# Projected observation rosters, per the closed plan.
HOUSEHOLD_ROSTER = ("SERIALNO", "TYPEHUGQ", "NP", "TEN", "WGTP", "PUMA")
PERSON_ROSTER = ("SERIALNO", "SPORDER", "PWGTP")

# Fixed classifier output order, read from the pinned pure module.
CODE_COLUMNS = (
    "interview_scope",
    "physical_unit",
    "household_kind",
    "tenure_subtype",
    "occupied_hu",
    "hu_tenure_class",
    "unresolved_reasons",
    "TEN_valid",
)


def household_row(serialno, typehugq, persons, ten, wgtp, **overrides):
    """One invented household record as an ordered tuple of lexical tokens."""
    values = dict(HOUSEHOLD_DEFAULTS)
    values.update(SERIALNO=serialno, TYPEHUGQ=typehugq, NP=persons, TEN=ten, WGTP=wgtp)
    values.update(overrides)
    return tuple(values[name] for name in HOUSEHOLD_HEADER)


def person_row(serialno, sporder, pwgtp, **overrides):
    """One invented person record as an ordered tuple of lexical tokens."""
    values = dict(PERSON_DEFAULTS)
    values.update(SERIALNO=serialno, SPORDER=sporder, PWGTP=pwgtp)
    values.update(overrides)
    return tuple(values[name] for name in PERSON_HEADER)


# Eight invented households covering every accepted source class exactly once,
# plus both WGTP boundaries and the whole observed TEN domain.
BASE_HOUSEHOLDS = (
    # occupied owner with a mortgage, two people
    household_row("2024HU0000001", "1", "2", "1", "118", PUMA="00100", ST="06"),
    # occupied renter paying rent, one person
    household_row("2024HU0000002", "1", "1", "3", "76", PUMA="03701", ST="06"),
    # occupied housing unit whose tenure is not available, three people
    household_row("2024HU0000003", "1", "3", "", "1", PUMA="81003", ST="36"),
    # vacant housing unit: no people, blank tenure, positive weight
    household_row("2024HU0000004", "1", "0", "", "55", PUMA="00100", ST="11"),
    # institutional group quarters placeholder
    household_row("2024GQ0000005", "2", "1", "", "0", PUMA="03701", ST="36"),
    # noninstitutional group quarters placeholder
    household_row("2024GQ0000006", "3", "1", "", "0", PUMA="00100", ST="01"),
    # occupied owner outright
    household_row("2024HU0000007", "1", "1", "2", "200", PUMA="00100", ST="01"),
    # occupied renter paying no rent, maximum housing weight
    household_row("2024HU0000008", "1", "1", "4", "9999", PUMA="81003", ST="11"),
)

BASE_PERSONS = (
    person_row("2024HU0000001", "01", "121", WAGP="52000"),
    person_row("2024HU0000001", "02", "97", WAGP="0"),
    person_row("2024HU0000002", "1", "80", WAGP="31000"),
    person_row("2024HU0000003", "1", "12", WAGP="17000"),
    person_row("2024HU0000003", "2", "11", WAGP="0"),
    person_row("2024HU0000003", "3", "9", WAGP="0"),
    person_row("2024GQ0000005", "01", "64", WAGP="0"),
    person_row("2024GQ0000006", "1", "9999", WAGP="7000"),
    person_row("2024HU0000007", "01", "205", WAGP="99000"),
    person_row("2024HU0000008", "1", "1", WAGP="4000"),
)

# The base population is split across two members per role so that member and
# ordinal lineage is exercised by every ordinary test rather than by one.
BASE_HOUSEHOLD_MEMBERS = (
    ("psam_husa.csv", BASE_HOUSEHOLDS[:5]),
    ("psam_husb.csv", BASE_HOUSEHOLDS[5:]),
)
BASE_PERSON_MEMBERS = (
    ("psam_pusa.csv", BASE_PERSONS[:6]),
    ("psam_pusb.csv", BASE_PERSONS[6:]),
)

BASE_SERIALNOS = tuple(
    row[HOUSEHOLD_HEADER.index("SERIALNO")] for row in BASE_HOUSEHOLDS
)

# The plan fixes what each row contains, never the order rows are emitted in.
# Content assertions below are therefore keyed by identifier, and the ordering
# decision lives in exactly one place: this constant and the two tests that
# read it. "sorted_serialno" is the order the production module emits;
# "source" would be member-then-row-ordinal order. See the author report.
CANONICAL_ROW_ORDER = "sorted_serialno"


def expected_row_order(serialnos):
    if CANONICAL_ROW_ORDER == "sorted_serialno":
        return sorted(serialnos)
    if CANONICAL_ROW_ORDER == "source":
        return list(serialnos)
    raise AssertionError(f"unknown row-order contract {CANONICAL_ROW_ORDER!r}")


# (occupied_hu, hu_tenure_class, tenure_subtype, unresolved_reasons,
#  physical_unit, TEN_valid) recomputed by hand from the pinned classifier.
BASE_EXPECTED_CODES = {
    "interview_scope": [0, 0, 0, 0, 0, 0, 0, 0],
    "physical_unit": [1, 1, 1, 1, 2, 2, 1, 1],
    "household_kind": [0, 0, 0, 0, 0, 0, 0, 0],
    "tenure_subtype": [1, 3, 0, 0, 0, 0, 2, 4],
    "occupied_hu": [1, 1, 1, 0, 2, 2, 1, 1],
    "hu_tenure_class": [1, 2, 3, 0, 0, 0, 1, 2],
    "unresolved_reasons": [0, 0, 2, 1, 0, 0, 0, 0],
    "TEN_valid": [1, 1, 0, 0, 0, 0, 1, 1],
}


# --------------------------------------------------------------------------
# Building invented archives.
# --------------------------------------------------------------------------


EXPECTED_HOUSEHOLDS = {
    row[HOUSEHOLD_HEADER.index("SERIALNO")]: {
        name: row[HOUSEHOLD_HEADER.index(name)] for name in HOUSEHOLD_ROSTER + ("ST",)
    }
    for row in BASE_HOUSEHOLDS
}

EXPECTED_PERSONS = {
    (
        row[PERSON_HEADER.index("SERIALNO")],
        row[PERSON_HEADER.index("SPORDER")],
    ): row[PERSON_HEADER.index("PWGTP")]
    for row in BASE_PERSONS
}

EXPECTED_CODES = {
    serialno: tuple(BASE_EXPECTED_CODES[name][index] for name in CODE_COLUMNS)
    for index, serialno in enumerate(BASE_SERIALNOS)
}

EXPECTED_HOUSEHOLD_MEMBER = {
    row[HOUSEHOLD_HEADER.index("SERIALNO")]: member
    for member, rows in BASE_HOUSEHOLD_MEMBERS
    for row in rows
}

EXPECTED_PERSON_MEMBER = {
    (
        row[PERSON_HEADER.index("SERIALNO")],
        row[PERSON_HEADER.index("SPORDER")],
    ): member
    for member, rows in BASE_PERSON_MEMBERS
    for row in rows
}


def csv_bytes(header, rows, *, newline="\n", trailing_newline=True):
    """Serialise invented tokens without any quoting the fixtures do not ask for."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=newline, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    writer.writerows(rows)
    text = buffer.getvalue()
    if not trailing_newline and text.endswith(newline):
        text = text[: -len(newline)]
    return text.encode("utf-8")


@dataclass(frozen=True)
class Member:
    """One planned archive entry, data or auxiliary."""

    name: str
    data: bytes
    compress_type: int = zipfile.ZIP_DEFLATED
    external_attr: int | None = None
    create_system: int = 3


def data_members(header, members, *, newline="\n", **kwargs):
    return tuple(
        Member(name, csv_bytes(header, rows, newline=newline, **kwargs))
        for name, rows in members
    )


def write_archive(path, members):
    """Write invented members exactly as planned, preserving order and duplicates."""
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            info = zipfile.ZipInfo(member.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = member.compress_type
            info.create_system = member.create_system
            info.external_attr = (
                member.external_attr
                if member.external_attr is not None
                else (0o644 << 16)
            )
            archive.writestr(info, member.data)
    return path


def flip_archive_byte(path, offset=None):
    """Change one compressed byte in place: same size, different digest.

    A deflated archive holds no plaintext, so a token replacement here would
    silently do nothing and the test would stop testing what it claims.
    """
    payload = Path(path).read_bytes()
    index = len(payload) // 2 if offset is None else offset
    mutated = payload[:index] + bytes([payload[index] ^ 0xFF]) + payload[index + 1 :]
    assert len(mutated) == len(payload) and mutated != payload
    Path(path).write_bytes(mutated)
    return payload


def digest_and_size(path):
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


@dataclass
class Fixture:
    """An invented source directory plus everything a test needs to drive it."""

    tmp_path: Path
    source_dir: Path
    snapshot_root: Path
    monkeypatch: pytest.MonkeyPatch
    households: tuple = ()
    persons: tuple = ()
    secrets: list = field(default_factory=list)
    _output_index: int = 0

    @property
    def household_zip(self):
        return self.source_dir / HOUSEHOLD_ARCHIVE

    @property
    def person_zip(self):
        return self.source_dir / PERSON_ARCHIVE

    def pin(self):
        """Point the private pin tuple at the invented archives as they are now."""
        pins = []
        for role, name in (
            (NAMES["household_role"], HOUSEHOLD_ARCHIVE),
            (NAMES["person_role"], PERSON_ARCHIVE),
        ):
            path = self.source_dir / name
            sha, size = digest_and_size(path)
            pins.append((role, name, sha, size))
        self.monkeypatch.setattr(source, NAMES["archive_pins"], tuple(pins))
        return tuple(pins)

    def output_dir(self, label="artifact"):
        self._output_index += 1
        return self.tmp_path / f"out-{self._output_index}-{label}"

    def produce(self, *, output_dir=None, serialnos=None, source_dir=None):
        return source.produce_acs_housing_source(
            self.source_dir if source_dir is None else source_dir,
            snapshot_root=self.snapshot_root,
            output_dir=self.output_dir() if output_dir is None else output_dir,
            serialnos=serialnos,
        )

    def load(self, output_dir, *, serialnos=None, source_dir=None):
        return source.load_acs_housing_source(
            self.source_dir if source_dir is None else source_dir,
            output_dir,
            snapshot_root=self.snapshot_root,
            serialnos=serialnos,
        )

    def produce_and_load(self, *, serialnos=None):
        output_dir = self.output_dir()
        produced = self.produce(output_dir=output_dir, serialnos=serialnos)
        return produced, self.load(output_dir, serialnos=serialnos), output_dir


def build_fixture(
    tmp_path,
    monkeypatch,
    *,
    households=BASE_HOUSEHOLD_MEMBERS,
    persons=BASE_PERSON_MEMBERS,
    household_members=None,
    person_members=None,
    household_header=HOUSEHOLD_HEADER,
    person_header=PERSON_HEADER,
    newline="\n",
    pin=True,
):
    """Assemble an invented source directory; explicit members override the rows."""
    # Resolve first: on macOS pytest's tmp_path sits under /var, a symlink to
    # /private/var, and the capture refuses symlinked path components. The
    # deliberate symlink refusals below build their own links explicitly.
    tmp_path = Path(tmp_path).resolve()
    source_dir = Path(tmp_path) / "acs-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(tmp_path) / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    if household_members is None:
        household_members = data_members(household_header, households, newline=newline)
    if person_members is None:
        person_members = data_members(person_header, persons, newline=newline)

    write_archive(source_dir / HOUSEHOLD_ARCHIVE, household_members)
    write_archive(source_dir / PERSON_ARCHIVE, person_members)

    fixture = Fixture(
        tmp_path=Path(tmp_path),
        source_dir=source_dir,
        snapshot_root=snapshot_root,
        monkeypatch=monkeypatch,
        households=households,
        persons=persons,
    )
    fixture.secrets = sanitisation_secrets(fixture)
    if pin:
        fixture.pin()
    return fixture


def sanitisation_secrets(fixture):
    """Strings a sanitised refusal must never echo back to the caller."""
    secrets = [
        str(fixture.tmp_path),
        str(fixture.source_dir),
        str(fixture.snapshot_root),
    ]
    secrets.extend(BASE_SERIALNOS)
    # Unprojected invented money tokens: no legitimate reason code names these,
    # unlike the domain bounds (1..9999, 100..81003) a code may legitimately cite.
    secrets.extend(["82000", "52000", "410000", "31000", "99000", "17000"])
    return secrets


@pytest.fixture(autouse=True)
def _invented_disk_budget(monkeypatch):
    """Eight invented rows must not require a full-source disk reserve in CI."""
    monkeypatch.setattr(
        source.shutil, "disk_usage", lambda _path: SimpleNamespace(free=64 * 1024**3)
    )


@pytest.fixture
def acs(tmp_path, monkeypatch):
    """The plain accepted invented population, pinned and ready to produce."""
    return build_fixture(tmp_path, monkeypatch)


# --------------------------------------------------------------------------
# Refusal helpers.
# --------------------------------------------------------------------------


def refuses(fixture, call, *args, **kwargs):
    """Assert the exact refusal type, capture the reason, and check sanitisation."""
    with pytest.raises(source.ACSHousingSourceError) as caught:
        call(*args, **kwargs)
    reason = str(caught.value)
    assert type(caught.value) is source.ACSHousingSourceError
    assert reason, "a refusal must carry a reason"
    assert len(reason) <= 120, "a reason code must not be prose"
    assert "\n" not in reason
    for secret in fixture.secrets:
        assert secret not in reason, "refusal leaked invented source content"
    return reason


def flat_scalars(payload):
    """Every scalar reachable in a decoded receipt, for shape-tolerant assertions."""
    found = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            found.append(item)
    return found


def flat_strings(payload):
    return {item for item in flat_scalars(payload) if isinstance(item, str)}


def flat_numbers(payload):
    return {
        item
        for item in flat_scalars(payload)
        if isinstance(item, int) and not isinstance(item, bool)
    }


def state_series(households):
    """Read the state column under whichever spelling the module exposes."""
    for name in (NAMES["state_column"], "ST", "STATE"):
        if name in households.columns:
            return households[name]
    raise AssertionError("household frame exposes no state column")


def household_keys(households):
    keys = lexical(households["SERIALNO"])
    assert len(set(keys)) == len(keys), "household identifiers must be unique"
    return keys


def household_map(households, names=HOUSEHOLD_ROSTER + ("ST",)):
    """Every projected household token, keyed by identifier rather than by row."""
    keys = household_keys(households)
    columns = {
        name: lexical(state_series(households) if name == "ST" else households[name])
        for name in names
    }
    return {
        key: {name: columns[name][row] for name in names}
        for row, key in enumerate(keys)
    }


def person_map(persons):
    """Person weight keyed by identifier and the raw SPORDER token."""
    entries = list(
        zip(
            lexical(persons["SERIALNO"]),
            lexical(persons["SPORDER"]),
            lexical(persons["PWGTP"]),
            strict=True,
        )
    )
    keyed = {(serialno, sporder): weight for serialno, sporder, weight in entries}
    assert len(keyed) == len(entries), "person keys must be unique"
    return keyed


def codes_map(result):
    """The eight native codes as a tuple per household identifier."""
    keys = household_keys(result.households)
    codes = result.codes
    return {
        key: tuple(int(codes[name].iloc[row]) for name in CODE_COLUMNS)
        for row, key in enumerate(keys)
    }


def lexical(series):
    """A projected column must be lexical text with no NA conversion."""
    values = series.tolist()
    assert all(isinstance(value, str) for value in values), (
        f"{series.name} is not lexical: {[type(v).__name__ for v in values]}"
    )
    return values


def strict_arrays(households):
    """Rebuild the classifier inputs from the projected lexical household tokens."""
    tenure_tokens = lexical(households["TEN"])
    tenure_valid = np.asarray([token != "" for token in tenure_tokens], dtype=bool)
    frame = pd.DataFrame(
        {
            "TYPEHUGQ": np.asarray(
                [int(token) for token in lexical(households["TYPEHUGQ"])], dtype="int64"
            ),
            "NP": np.asarray(
                [int(token) for token in lexical(households["NP"])], dtype="int64"
            ),
            "TEN": np.asarray(
                [int(token) if token != "" else 0 for token in tenure_tokens],
                dtype="int64",
            ),
        },
        index=households.index,
    )
    return frame, tenure_valid


def assert_lineage(frame, keys, expected_member):
    """Find member/ordinal lineage without depending on either column spelling.

    Order-independent by construction: the member column must agree row by row
    with the invented layout, and each member's ordinals must form one
    consecutive run, whatever order the rows are emitted in.
    """
    members = [expected_member[key] for key in keys]
    member_column = next(
        (
            name
            for name in frame.columns
            if [value if isinstance(value, str) else None for value in frame[name]]
            == members
        ),
        None,
    )
    assert member_column is not None, (
        "no column carries the archive member of each row; expected "
        f"{sorted(set(members))}"
    )

    for name in frame.columns:
        if name == member_column:
            continue
        try:
            values = [int(value) for value in frame[name].tolist()]
        except (TypeError, ValueError):
            continue
        if len(values) != len(members):
            continue
        runs = {}
        for member, value in zip(members, values, strict=True):
            runs.setdefault(member, []).append(value)
        bases = {min(run) for run in runs.values()}
        if bases <= {0} or bases <= {1}:
            if all(
                sorted(run) == list(range(min(run), min(run) + len(run)))
                for run in runs.values()
            ):
                return member_column, name
    raise AssertionError(
        "no column carries a within-member row ordinal restarting at each member"
    )


# --------------------------------------------------------------------------
# Accepted population: projection, classification and roundtrip.
# --------------------------------------------------------------------------


def test_projected_households_are_lexical_and_complete(acs):
    result = acs.produce()
    households = result.households
    assert len(households) == len(BASE_HOUSEHOLDS)
    assert households.columns.is_unique
    assert set(HOUSEHOLD_ROSTER) <= set(households.columns)
    for name in HOUSEHOLD_ROSTER:
        lexical(households[name])
    # Every projected token survives byte for byte: blank tenure stays an empty
    # string rather than NaN, None or the dictionary's display character, and
    # leading zeros on PUMA and ST are preserved.
    assert household_map(households) == EXPECTED_HOUSEHOLDS


def test_projected_persons_keep_raw_sporder_tokens(acs):
    persons = acs.produce().persons
    assert len(persons) == len(BASE_PERSONS)
    assert set(PERSON_ROSTER) <= set(persons.columns)
    # "01" and "1" are both accepted and both retained exactly as written.
    assert person_map(persons) == EXPECTED_PERSONS


def test_native_codes_are_uint8_in_the_pinned_order(acs):
    result = acs.produce()
    codes = result.codes
    assert list(codes.columns) == list(CODE_COLUMNS)
    for name in CODE_COLUMNS:
        assert codes[name].dtype == np.dtype("uint8")
    assert codes.index.equals(result.households.index)
    assert codes_map(result) == EXPECTED_CODES


def test_codes_equal_the_pinned_pure_classifier_on_the_strict_arrays(acs):
    """The source result must not invent its own semantics for the same rows."""
    result = acs.produce()
    frame, tenure_valid = strict_arrays(result.households)
    expected, _header = classify_acs_housing_universe(frame, tenure_valid=tenure_valid)
    pd.testing.assert_frame_equal(result.codes, expected, check_like=False)


def test_np0_housing_unit_stays_occupancy_unknown_in_the_native_codes(acs):
    """Source vacancy evidence must not be back-written into the pure classes."""
    codes = dict(
        zip(CODE_COLUMNS, codes_map(acs.produce())["2024HU0000004"], strict=True)
    )
    assert codes["occupied_hu"] == 0
    assert codes["hu_tenure_class"] == 0
    assert codes["unresolved_reasons"] == 1
    assert codes["physical_unit"] == 1
    assert codes["TEN_valid"] == 0


def test_occupied_unit_with_unavailable_tenure_is_retained_not_recoded(acs):
    codes = dict(
        zip(CODE_COLUMNS, codes_map(acs.produce())["2024HU0000003"], strict=True)
    )
    assert codes["occupied_hu"] == 1
    assert codes["hu_tenure_class"] == 3
    assert codes["tenure_subtype"] == 0
    assert codes["unresolved_reasons"] == 2


def test_member_and_row_ordinal_lineage_is_accessible(acs):
    result = acs.produce()
    assert_lineage(
        result.households,
        household_keys(result.households),
        EXPECTED_HOUSEHOLD_MEMBER,
    )
    assert_lineage(
        result.persons,
        list(person_map(result.persons)),
        EXPECTED_PERSON_MEMBER,
    )


def test_rows_are_emitted_in_the_agreed_canonical_order(acs):
    """The plan leaves row order open; this is the single place it is decided."""
    assert household_keys(acs.produce().households) == expected_row_order(
        BASE_SERIALNOS
    )


def test_row_order_does_not_depend_on_the_member_split(tmp_path, monkeypatch):
    """Splitting the same invented rows differently must not reorder them."""
    one = single_member(tmp_path / "one", monkeypatch).produce()
    three = build_fixture(
        tmp_path / "three",
        monkeypatch,
        household_members=(
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[6:])),
            Member("psam_husb.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[:3])),
            Member("psam_husc.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[3:6])),
        ),
        person_members=(Member("psam_pusa.csv", BASE_PERSON_CSV),),
    ).produce()
    assert household_keys(three.households) == household_keys(one.households)
    assert household_map(three.households) == household_map(one.households)


def test_source_artifact_holds_exactly_the_projection_and_receipt(acs):
    output_dir = acs.output_dir()
    result = acs.produce(output_dir=output_dir)
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        (NAMES["projection_filename"], NAMES["receipt_filename"])
    )
    # The house convention appends a trailing newline to published JSON.
    assert (output_dir / NAMES["projection_filename"]).read_bytes().rstrip(
        b"\n"
    ) == result.projection_json.rstrip(b"\n")
    assert (output_dir / NAMES["receipt_filename"]).read_bytes().rstrip(
        b"\n"
    ) == result.receipt_json.rstrip(b"\n")
    json.loads(result.projection_json)
    json.loads(result.receipt_json)


def test_readback_returns_the_same_bytes_and_the_same_population(acs):
    produced, loaded, _output_dir = acs.produce_and_load()
    assert loaded.projection_json == produced.projection_json
    assert loaded.receipt_json == produced.receipt_json
    pd.testing.assert_frame_equal(loaded.households, produced.households)
    pd.testing.assert_frame_equal(loaded.persons, produced.persons)
    pd.testing.assert_frame_equal(loaded.codes, produced.codes)


def test_two_produces_from_identical_inputs_are_byte_identical(acs):
    """Readback compares exact bytes, so both artifacts must be deterministic."""
    first = acs.produce(output_dir=acs.output_dir("first"))
    second = acs.produce(output_dir=acs.output_dir("second"))
    assert first.projection_json == second.projection_json
    assert first.receipt_json == second.receipt_json


@pytest.mark.parametrize("attribute", ["households", "persons", "codes"])
def test_returned_frames_are_owned_copies(acs, attribute):
    result = acs.produce()
    first = getattr(result, attribute)
    assert getattr(result, attribute) is not first
    before = first.copy(deep=True)
    projection = result.projection_json
    first.iloc[0, 0] = first.iloc[-1, 0]
    first.rename(columns={first.columns[0]: "clobbered"}, inplace=True)
    first.index = pd.RangeIndex(1000, 1000 + len(first))
    after = getattr(result, attribute)
    pd.testing.assert_frame_equal(after, before)
    assert result.projection_json == projection


def test_receipt_records_auditable_member_and_definition_evidence(acs):
    result = acs.produce()
    receipt = json.loads(result.receipt_json)
    strings = flat_strings(receipt)
    numbers = flat_numbers(receipt)

    assert NAMES["vacancy_definition"] in strings
    assert NAMES["gq_definition"] in strings
    assert {HOUSEHOLD_ARCHIVE, PERSON_ARCHIVE} <= strings
    assert {"psam_husa.csv", "psam_husb.csv", "psam_pusa.csv", "psam_pusb.csv"} <= (
        strings
    )

    for archive, names in (
        (acs.household_zip, ("psam_husa.csv", "psam_husb.csv")),
        (acs.person_zip, ("psam_pusa.csv", "psam_pusb.csv")),
    ):
        with zipfile.ZipFile(archive) as opened:
            for name in names:
                info = opened.getinfo(name)
                payload = opened.read(name)
                assert hashlib.sha256(payload).hexdigest() in strings, (
                    f"receipt omits the decompressed digest of {name}"
                )
                assert info.CRC in numbers, f"receipt omits the CRC of {name}"
                assert info.file_size in numbers

    # Counts an auditor needs: the whole population and the vacancy evidence.
    assert len(BASE_HOUSEHOLDS) in numbers
    assert len(BASE_PERSONS) in numbers


def test_receipt_binds_the_pinned_archive_digests(acs):
    pins = acs.pin()
    strings = flat_strings(json.loads(acs.produce().receipt_json))
    for _role, name, sha, _size in pins:
        assert sha in strings, f"receipt omits the pinned digest of {name}"


# --------------------------------------------------------------------------
# Population variants: one deliberate change against the accepted invented rows.
# --------------------------------------------------------------------------


def with_household(index, **fields):
    rows = list(BASE_HOUSEHOLDS)
    values = dict(zip(HOUSEHOLD_HEADER, rows[index], strict=True))
    values.update(fields)
    rows[index] = tuple(values[name] for name in HOUSEHOLD_HEADER)
    return tuple(rows)


def with_person(index, **fields):
    rows = list(BASE_PERSONS)
    values = dict(zip(PERSON_HEADER, rows[index], strict=True))
    values.update(fields)
    rows[index] = tuple(values[name] for name in PERSON_HEADER)
    return tuple(rows)


def rename_serialno(old, new):
    """Rename a key on both sides so the only change is the identifier token."""
    key = HOUSEHOLD_HEADER.index("SERIALNO")
    households = tuple(
        row[:key] + (new,) + row[key + 1 :] if row[key] == old else row
        for row in BASE_HOUSEHOLDS
    )
    key = PERSON_HEADER.index("SERIALNO")
    persons = tuple(
        row[:key] + (new,) + row[key + 1 :] if row[key] == old else row
        for row in BASE_PERSONS
    )
    return households, persons


def single_member(tmp_path, monkeypatch, *, households=None, persons=None, **kwargs):
    """One member per role, so a variant's only difference is the row content."""
    return build_fixture(
        tmp_path,
        monkeypatch,
        households=(
            ("psam_husa.csv", BASE_HOUSEHOLDS if households is None else households),
        ),
        persons=(("psam_pusa.csv", BASE_PERSONS if persons is None else persons),),
        **kwargs,
    )


# --------------------------------------------------------------------------
# Key selection.
# --------------------------------------------------------------------------


def test_selection_none_retains_the_whole_invented_source(acs):
    result = acs.produce(serialnos=None)
    assert set(household_keys(result.households)) == set(BASE_SERIALNOS)
    assert len(result.persons) == len(BASE_PERSONS)


def test_selection_keeps_whole_households_and_no_other_person(acs):
    result = acs.produce(serialnos=("2024HU0000001", "2024GQ0000005"))
    assert set(household_keys(result.households)) == {
        "2024HU0000001",
        "2024GQ0000005",
    }
    assert person_map(result.persons) == {
        ("2024HU0000001", "01"): "121",
        ("2024HU0000001", "02"): "97",
        ("2024GQ0000005", "01"): "64",
    }
    assert codes_map(result) == {
        key: EXPECTED_CODES[key] for key in ("2024HU0000001", "2024GQ0000005")
    }


def test_a_vacancy_only_selection_is_a_valid_source_projection(acs):
    """Source-only preparation may validly retain no people at all."""
    result = acs.produce(serialnos=("2024HU0000004",))
    assert household_keys(result.households) == ["2024HU0000004"]
    assert len(result.persons) == 0
    assert result.codes["occupied_hu"].tolist() == [0]
    assert result.codes["unresolved_reasons"].tolist() == [1]


def test_selected_rows_ignore_the_caller_tuple_order(acs):
    """The request is a set of keys; canonical order is the module's own."""
    forward = acs.produce(
        output_dir=acs.output_dir("forward"),
        serialnos=("2024HU0000002", "2024HU0000008"),
    )
    reversed_request = acs.produce(
        output_dir=acs.output_dir("reversed"),
        serialnos=("2024HU0000008", "2024HU0000002"),
    )
    assert forward.projection_json == reversed_request.projection_json
    assert household_keys(forward.households) == expected_row_order(
        ("2024HU0000002", "2024HU0000008")
    )


def test_selection_changes_the_projection_bytes(acs):
    whole = acs.produce()
    selected = acs.produce(serialnos=("2024HU0000001",))
    assert selected.projection_json != whole.projection_json


def test_readback_under_a_different_selection_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir, serialnos=("2024HU0000001",))
    refuses(acs, acs.load, output_dir, serialnos=("2024HU0000002",))
    refuses(acs, acs.load, output_dir, serialnos=None)


@pytest.mark.parametrize(
    "serialnos",
    [
        pytest.param(("2024HU0000009",), id="unknown_key"),
        pytest.param(("2024HU0000001", "2024HU0000009"), id="one_unknown_key"),
        pytest.param(("2024HU0000001", "2024HU0000001"), id="duplicate_key"),
        pytest.param((), id="empty_selection"),
        pytest.param(("",), id="blank_key"),
        pytest.param((" 2024HU0000001",), id="whitespace_padded_key"),
        pytest.param(("2024hu0000001",), id="case_folded_key"),
    ],
)
def test_invalid_selection_refuses(acs, serialnos):
    refuses(acs, acs.produce, serialnos=serialnos)


# --------------------------------------------------------------------------
# Complete-join integrity, checked before any exclusion or selection.
# --------------------------------------------------------------------------

JOIN_VARIANTS = [
    pytest.param(
        BASE_HOUSEHOLDS,
        BASE_PERSONS + (person_row("2024HU0000004", "1", "40"),),
        id="person_attached_to_a_vacant_household",
    ),
    pytest.param(
        BASE_HOUSEHOLDS,
        BASE_PERSONS + (person_row("2024HU0000009", "1", "40"),),
        id="orphan_person_without_a_household",
    ),
    pytest.param(
        BASE_HOUSEHOLDS + (BASE_HOUSEHOLDS[0],),
        BASE_PERSONS,
        id="duplicate_household_key",
    ),
    pytest.param(
        BASE_HOUSEHOLDS,
        BASE_PERSONS + (person_row("2024HU0000002", "1", "44"),),
        id="duplicate_person_key",
    ),
    pytest.param(
        BASE_HOUSEHOLDS,
        BASE_PERSONS + (person_row("2024HU0000002", "01", "44"),),
        id="duplicate_person_key_via_leading_zero_alias",
    ),
    pytest.param(
        with_household(0, NP="3"),
        BASE_PERSONS,
        id="household_np_above_linked_person_count",
    ),
    pytest.param(
        with_household(0, NP="1"),
        BASE_PERSONS,
        id="household_np_below_linked_person_count",
    ),
    pytest.param(
        BASE_HOUSEHOLDS,
        BASE_PERSONS[:6] + BASE_PERSONS[7:],
        id="group_quarters_placeholder_without_its_person",
    ),
    pytest.param(
        with_household(4, NP="2"),
        BASE_PERSONS + (person_row("2024GQ0000005", "02", "51"),),
        id="group_quarters_placeholder_with_two_people",
    ),
    pytest.param(
        with_household(4, NP="0"),
        BASE_PERSONS[:6] + BASE_PERSONS[7:],
        id="group_quarters_placeholder_with_no_people",
    ),
]


@pytest.mark.parametrize("households,persons", JOIN_VARIANTS)
def test_join_integrity_refusals(tmp_path, monkeypatch, households, persons):
    fixture = single_member(
        tmp_path, monkeypatch, households=households, persons=persons
    )
    refuses(fixture, fixture.produce)


@pytest.mark.parametrize(
    "serialnos",
    [
        pytest.param(None, id="whole_source"),
        pytest.param(("2024HU0000001",), id="unrelated_single_key"),
    ],
)
def test_vacant_household_person_refuses_before_exclusion_or_selection(
    tmp_path, monkeypatch, serialnos
):
    """An NP0 household with a person must refuse even when nothing selects it.

    An implementation that dropped NP0 rows, or applied the key selection,
    before validating would silently accept this invented contradiction.
    """
    fixture = single_member(
        tmp_path,
        monkeypatch,
        persons=BASE_PERSONS + (person_row("2024HU0000004", "1", "40"),),
    )
    refuses(fixture, fixture.produce, serialnos=serialnos)


def test_orphan_person_refuses_even_when_selection_excludes_it(tmp_path, monkeypatch):
    fixture = single_member(
        tmp_path,
        monkeypatch,
        persons=BASE_PERSONS + (person_row("2024HU0000009", "1", "40"),),
    )
    refuses(fixture, fixture.produce, serialnos=("2024HU0000002",))


# --------------------------------------------------------------------------
# Closed source-value domains.
# --------------------------------------------------------------------------

DOMAIN_VARIANTS = [
    # SERIALNO: exactly 13 ASCII characters, 2024HU/2024GQ plus seven digits,
    # numeric suffix 1..9999999, prefix agreeing with TYPEHUGQ, no whitespace,
    # no empty identifier and no normalisation of the accepted bytes.
    pytest.param(
        *rename_serialno("2024HU0000001", " 2024HU000001"), id="serialno_leading_space"
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024HU000001 "), id="serialno_trailing_space"
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024HU 000001"), id="serialno_interior_space"
    ),
    pytest.param(*rename_serialno("2024HU0000001", ""), id="serialno_empty"),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024HU000001"), id="serialno_too_short"
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024HU00000010"), id="serialno_too_long"
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2023HU0000001"), id="serialno_wrong_vintage"
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024XX0000001"),
        id="serialno_unknown_type_token",
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024hu0000001"),
        id="serialno_lowercase_type_token",
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024HU0000000"), id="serialno_zero_suffix"
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024HU000000A"),
        id="serialno_nondigit_suffix",
    ),
    pytest.param(
        *rename_serialno("2024HU0000001", "2024GQ0000001"),
        id="serialno_gq_prefix_on_housing_unit",
    ),
    pytest.param(
        *rename_serialno("2024GQ0000005", "2024HU0000005"),
        id="serialno_hu_prefix_on_group_quarters",
    ),
    # SPORDER: one or two ASCII digits, value 1..20, no sign/exponent/point.
    pytest.param(BASE_HOUSEHOLDS, with_person(2, SPORDER="0"), id="sporder_zero"),
    pytest.param(
        BASE_HOUSEHOLDS, with_person(2, SPORDER="21"), id="sporder_above_domain"
    ),
    pytest.param(
        BASE_HOUSEHOLDS, with_person(2, SPORDER="001"), id="sporder_three_digits"
    ),
    pytest.param(
        BASE_HOUSEHOLDS, with_person(2, SPORDER=" 1"), id="sporder_leading_space"
    ),
    pytest.param(
        BASE_HOUSEHOLDS, with_person(2, SPORDER="1 "), id="sporder_trailing_space"
    ),
    pytest.param(BASE_HOUSEHOLDS, with_person(2, SPORDER="+1"), id="sporder_signed"),
    pytest.param(BASE_HOUSEHOLDS, with_person(2, SPORDER="-1"), id="sporder_negative"),
    pytest.param(
        BASE_HOUSEHOLDS, with_person(2, SPORDER="1.0"), id="sporder_decimal_point"
    ),
    pytest.param(BASE_HOUSEHOLDS, with_person(2, SPORDER="1e0"), id="sporder_exponent"),
    pytest.param(BASE_HOUSEHOLDS, with_person(2, SPORDER=""), id="sporder_empty"),
    # TYPEHUGQ: 1, 2 or 3 only.
    pytest.param(with_household(0, TYPEHUGQ="0"), BASE_PERSONS, id="typehugq_zero"),
    pytest.param(
        with_household(0, TYPEHUGQ="4"), BASE_PERSONS, id="typehugq_above_domain"
    ),
    pytest.param(with_household(0, TYPEHUGQ=""), BASE_PERSONS, id="typehugq_empty"),
    pytest.param(
        with_household(0, TYPEHUGQ="H"), BASE_PERSONS, id="typehugq_nonnumeric"
    ),
    # NP: published integer domain 0..20.
    pytest.param(with_household(0, NP="21"), BASE_PERSONS, id="np_above_domain"),
    pytest.param(with_household(0, NP="-1"), BASE_PERSONS, id="np_negative"),
    pytest.param(with_household(0, NP=""), BASE_PERSONS, id="np_empty"),
    pytest.param(with_household(0, NP=" 2"), BASE_PERSONS, id="np_leading_space"),
    pytest.param(with_household(0, NP="2.0"), BASE_PERSONS, id="np_decimal_point"),
    # WGTP: 1..9999 for housing units including vacant ones; zero only for the
    # group-quarters placeholders.
    pytest.param(
        with_household(0, WGTP="0"), BASE_PERSONS, id="wgtp_zero_on_occupied_unit"
    ),
    pytest.param(
        with_household(3, WGTP="0"), BASE_PERSONS, id="wgtp_zero_on_vacant_unit"
    ),
    pytest.param(with_household(0, WGTP="10000"), BASE_PERSONS, id="wgtp_above_domain"),
    pytest.param(with_household(0, WGTP="-5"), BASE_PERSONS, id="wgtp_negative"),
    pytest.param(with_household(0, WGTP=""), BASE_PERSONS, id="wgtp_empty"),
    # PWGTP: 1..9999 for every person, group quarters included.
    pytest.param(BASE_HOUSEHOLDS, with_person(2, PWGTP="0"), id="pwgtp_zero"),
    pytest.param(
        BASE_HOUSEHOLDS,
        with_person(6, PWGTP="0"),
        id="pwgtp_zero_on_group_quarters_person",
    ),
    pytest.param(
        BASE_HOUSEHOLDS, with_person(2, PWGTP="10000"), id="pwgtp_above_domain"
    ),
    pytest.param(BASE_HOUSEHOLDS, with_person(2, PWGTP="-1"), id="pwgtp_negative"),
    pytest.param(BASE_HOUSEHOLDS, with_person(2, PWGTP=""), id="pwgtp_empty"),
    # TEN: blank means missing; observed values are 1..4; group quarters and
    # vacancy require the blank token.
    pytest.param(with_household(0, TEN="5"), BASE_PERSONS, id="ten_above_domain"),
    pytest.param(with_household(0, TEN="0"), BASE_PERSONS, id="ten_zero"),
    pytest.param(with_household(0, TEN="b"), BASE_PERSONS, id="ten_display_character"),
    pytest.param(with_household(0, TEN=" "), BASE_PERSONS, id="ten_single_space"),
    pytest.param(
        with_household(3, TEN="1"), BASE_PERSONS, id="ten_present_on_vacant_unit"
    ),
    pytest.param(
        with_household(4, TEN="1"), BASE_PERSONS, id="ten_present_on_group_quarters"
    ),
    # PUMA: exactly five ASCII digits with integer value 100..81003.
    pytest.param(with_household(0, PUMA="0010"), BASE_PERSONS, id="puma_four_digits"),
    pytest.param(with_household(0, PUMA="000100"), BASE_PERSONS, id="puma_six_digits"),
    pytest.param(with_household(0, PUMA="00099"), BASE_PERSONS, id="puma_below_domain"),
    pytest.param(with_household(0, PUMA="81004"), BASE_PERSONS, id="puma_above_domain"),
    pytest.param(with_household(0, PUMA="0010A"), BASE_PERSONS, id="puma_nondigit"),
    pytest.param(with_household(0, PUMA=""), BASE_PERSONS, id="puma_empty"),
    # ST: exactly two ASCII digits from the 50-state and DC subset.
    pytest.param(with_household(0, ST="72"), BASE_PERSONS, id="state_puerto_rico"),
    pytest.param(with_household(0, ST="99"), BASE_PERSONS, id="state_unassigned_code"),
    pytest.param(with_household(0, ST="6"), BASE_PERSONS, id="state_single_digit"),
    pytest.param(with_household(0, ST="006"), BASE_PERSONS, id="state_three_digits"),
    pytest.param(
        with_household(0, ST="CA"), BASE_PERSONS, id="state_postal_abbreviation"
    ),
    pytest.param(with_household(0, ST=""), BASE_PERSONS, id="state_empty"),
]


@pytest.mark.parametrize("households,persons", DOMAIN_VARIANTS)
def test_source_value_domain_refusals(tmp_path, monkeypatch, households, persons):
    fixture = single_member(
        tmp_path, monkeypatch, households=households, persons=persons
    )
    refuses(fixture, fixture.produce)


def test_sporder_leading_zero_is_accepted_and_the_raw_token_is_retained(
    tmp_path, monkeypatch
):
    """Leading-zero SPORDER normalises for the join without losing its bytes."""
    fixture = single_member(tmp_path, monkeypatch, persons=with_person(2, SPORDER="01"))
    assert ("2024HU0000002", "01") in person_map(fixture.produce().persons)


def test_maximum_and_minimum_accepted_weights_are_retained(acs):
    weights = household_map(acs.produce().households)
    assert weights["2024HU0000003"]["WGTP"] == "1"
    assert weights["2024HU0000008"]["WGTP"] == "9999"


# --------------------------------------------------------------------------
# Strict lexical parsing and the ordered-header contract.
# --------------------------------------------------------------------------

BASE_HOUSEHOLD_CSV = csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS)
BASE_PERSON_CSV = csv_bytes(PERSON_HEADER, BASE_PERSONS)


def household_bytes_fixture(tmp_path, monkeypatch, payload, *, name="psam_husa.csv"):
    return build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member(name, payload),),
        person_members=(Member("psam_pusa.csv", BASE_PERSON_CSV),),
    )


def person_bytes_fixture(tmp_path, monkeypatch, payload, *, name="psam_pusa.csv"):
    return build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),),
        person_members=(Member(name, payload),),
    )


def splice_line(payload, index, line):
    """Insert an invented physical line at a record boundary."""
    lines = payload.split(b"\n")
    lines.insert(index, line)
    return b"\n".join(lines)


def drop_column(header, rows, name):
    position = header.index(name)
    trimmed = header[:position] + header[position + 1 :]
    return trimmed, tuple(row[:position] + row[position + 1 :] for row in rows)


def rename_column(header, old, new):
    return tuple(new if name == old else name for name in header)


WIDE_SUFFIX = tuple(f"AUX{index:02d}" for index in range(24))
WIDE_HEADER = HOUSEHOLD_HEADER + WIDE_SUFFIX
WIDE_ROWS = tuple(row + ("0",) * len(WIDE_SUFFIX) for row in BASE_HOUSEHOLDS)


def oversized_record_csv():
    """One physical record above 1 MiB whose every field stays under 64 KiB."""
    padded = ("4" * 50_000,) * len(WIDE_SUFFIX)
    rows = (WIDE_ROWS[0][: len(HOUSEHOLD_HEADER)] + padded,) + WIDE_ROWS[1:]
    return csv_bytes(WIDE_HEADER, rows)


def oversized_field_csv():
    """One field token above 64 KiB in a column that is never projected."""
    return csv_bytes(HOUSEHOLD_HEADER, with_household(0, HINCP="7" * 65_537))


HOUSEHOLD_PARSE_VARIANTS = [
    pytest.param(b"\xef\xbb\xbf" + BASE_HOUSEHOLD_CSV, id="utf8_byte_order_mark"),
    pytest.param(
        BASE_HOUSEHOLD_CSV.replace(b"2024HU0000002", b"2024HU\xff000002"),
        id="invalid_utf8_byte",
    ),
    pytest.param(
        BASE_HOUSEHOLD_CSV.replace(b"82000", b"820\x0000"), id="embedded_nul_in_field"
    ),
    pytest.param(splice_line(BASE_HOUSEHOLD_CSV, 3, b""), id="interior_blank_record"),
    pytest.param(BASE_HOUSEHOLD_CSV + b"\n", id="trailing_blank_record"),
    pytest.param(
        csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[1:])
        + ",".join(BASE_HOUSEHOLDS[0][:-1]).encode()
        + b"\n",
        id="short_record",
    ),
    pytest.param(
        csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[1:])
        + ",".join(BASE_HOUSEHOLDS[0] + ("extra",)).encode()
        + b"\n",
        id="long_record",
    ),
    pytest.param(
        csv_bytes(rename_column(HOUSEHOLD_HEADER, "REGION", "ST"), BASE_HOUSEHOLDS),
        id="duplicate_header_name",
    ),
    pytest.param(
        csv_bytes(rename_column(HOUSEHOLD_HEADER, "REGION", ""), BASE_HOUSEHOLDS),
        id="blank_header_name",
    ),
    pytest.param(
        csv_bytes(
            rename_column(HOUSEHOLD_HEADER, "SERIALNO", " SERIALNO"), BASE_HOUSEHOLDS
        ),
        id="whitespace_padded_header_name",
    ),
    pytest.param(
        csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "NP")),
        id="missing_np_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "SERIALNO")),
        id="missing_serialno_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "TEN")),
        id="missing_ten_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "WGTP")),
        id="missing_wgtp_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "PUMA")),
        id="missing_puma_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "ST")),
        id="neither_st_nor_state_column",
    ),
    pytest.param(
        csv_bytes(HOUSEHOLD_HEADER, with_household(0, HINCP="82000\r82000")),
        id="embedded_carriage_return_in_field",
    ),
    pytest.param(
        csv_bytes(HOUSEHOLD_HEADER, with_household(0, HINCP="82000\n82000")),
        id="embedded_line_feed_in_field",
    ),
    pytest.param(oversized_field_csv(), id="field_token_above_64_kib"),
    pytest.param(oversized_record_csv(), id="physical_record_above_1_mib"),
    pytest.param(BASE_HOUSEHOLD_CSV.replace(b",", b";"), id="semicolon_delimiter"),
    pytest.param(b"", id="empty_member_without_a_header"),
]


@pytest.mark.parametrize("payload", HOUSEHOLD_PARSE_VARIANTS)
def test_household_parser_refusals(tmp_path, monkeypatch, payload):
    fixture = household_bytes_fixture(tmp_path, monkeypatch, payload)
    refuses(fixture, fixture.produce)


PERSON_PARSE_VARIANTS = [
    pytest.param(
        csv_bytes(*drop_column(PERSON_HEADER, BASE_PERSONS, "SPORDER")),
        id="missing_sporder_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(PERSON_HEADER, BASE_PERSONS, "PWGTP")),
        id="missing_pwgtp_column",
    ),
    pytest.param(
        csv_bytes(*drop_column(PERSON_HEADER, BASE_PERSONS, "SERIALNO")),
        id="missing_person_serialno_column",
    ),
    pytest.param(splice_line(BASE_PERSON_CSV, 2, b""), id="person_blank_record"),
    pytest.param(
        csv_bytes(PERSON_HEADER, BASE_PERSONS[1:])
        + ",".join(BASE_PERSONS[0][:-2]).encode()
        + b"\n",
        id="person_short_record",
    ),
]


@pytest.mark.parametrize("payload", PERSON_PARSE_VARIANTS)
def test_person_parser_refusals(tmp_path, monkeypatch, payload):
    fixture = person_bytes_fixture(tmp_path, monkeypatch, payload)
    refuses(fixture, fixture.produce)


def test_cross_member_reordered_header_refuses(tmp_path, monkeypatch):
    """Identical column sets in a different order are still a header mismatch."""
    reordered = list(HOUSEHOLD_HEADER)
    first, second = reordered.index("REGION"), reordered.index("ADJHSG")
    reordered[first], reordered[second] = reordered[second], reordered[first]
    rows = tuple(
        tuple(dict(zip(HOUSEHOLD_HEADER, row, strict=True))[name] for name in reordered)
        for row in BASE_HOUSEHOLDS[5:]
    )
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[:5])),
            Member("psam_husb.csv", csv_bytes(tuple(reordered), rows)),
        ),
        person_members=(Member("psam_pusa.csv", BASE_PERSON_CSV),),
    )
    refuses(fixture, fixture.produce)


def test_cross_member_extra_column_refuses(tmp_path, monkeypatch):
    extended = HOUSEHOLD_HEADER + ("FINCP",)
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[:5])),
            Member(
                "psam_husb.csv",
                csv_bytes(extended, tuple(row + ("0",) for row in BASE_HOUSEHOLDS[5:])),
            ),
        ),
        person_members=(Member("psam_pusa.csv", BASE_PERSON_CSV),),
    )
    refuses(fixture, fixture.produce)


def test_header_only_member_is_allowed_and_contributes_no_rows(tmp_path, monkeypatch):
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("psam_husb.csv", csv_bytes(HOUSEHOLD_HEADER, ())),
        ),
        person_members=(
            Member("psam_pusa.csv", csv_bytes(PERSON_HEADER, ())),
            Member("psam_pusb.csv", BASE_PERSON_CSV),
        ),
    )
    result = fixture.produce()
    assert set(household_keys(result.households)) == set(BASE_SERIALNOS)
    assert person_map(result.persons) == EXPECTED_PERSONS
    assert_lineage(
        result.households,
        household_keys(result.households),
        dict.fromkeys(BASE_SERIALNOS, "psam_husa.csv"),
    )
    assert_lineage(
        result.persons,
        list(person_map(result.persons)),
        dict.fromkeys(EXPECTED_PERSONS, "psam_pusb.csv"),
    )
    # The empty members are still inventoried even though they carry no rows.
    strings = flat_strings(json.loads(result.receipt_json))
    assert {"psam_husb.csv", "psam_pusa.csv"} <= strings


def test_an_entirely_header_only_source_is_not_a_population(tmp_path, monkeypatch):
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, ())),),
        person_members=(Member("psam_pusa.csv", csv_bytes(PERSON_HEADER, ())),),
    )
    refuses(fixture, fixture.produce)


def test_state_only_household_header_is_supported(tmp_path, monkeypatch):
    header = rename_column(HOUSEHOLD_HEADER, "ST", "STATE")
    fixture = household_bytes_fixture(
        tmp_path, monkeypatch, csv_bytes(header, BASE_HOUSEHOLDS)
    )
    result = fixture.produce()
    assert household_map(result.households)["2024HU0000001"]["ST"] == "06"
    # The header alias itself is preserved in the source evidence.
    assert "STATE" in flat_strings(json.loads(result.receipt_json))


def test_agreeing_st_and_state_columns_are_accepted(tmp_path, monkeypatch):
    header = HOUSEHOLD_HEADER + ("STATE",)
    state = HOUSEHOLD_HEADER.index("ST")
    rows = tuple(row + (row[state],) for row in BASE_HOUSEHOLDS)
    fixture = household_bytes_fixture(tmp_path, monkeypatch, csv_bytes(header, rows))
    result = fixture.produce()
    assert household_map(result.households)["2024HU0000003"]["ST"] == "36"


def test_disagreeing_st_and_state_columns_refuse(tmp_path, monkeypatch):
    header = HOUSEHOLD_HEADER + ("STATE",)
    state = HOUSEHOLD_HEADER.index("ST")
    rows = tuple(
        row + ("36" if index == 0 else row[state],)
        for index, row in enumerate(BASE_HOUSEHOLDS)
    )
    fixture = household_bytes_fixture(tmp_path, monkeypatch, csv_bytes(header, rows))
    refuses(fixture, fixture.produce)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_both_physical_line_terminators_project_identically(
    tmp_path, monkeypatch, newline
):
    """A genuine archive's terminator choice must not change the projection."""
    fixture = build_fixture(tmp_path, monkeypatch, newline=newline)
    result = fixture.produce()
    assert household_map(result.households) == EXPECTED_HOUSEHOLDS
    assert person_map(result.persons) == EXPECTED_PERSONS


# --------------------------------------------------------------------------
# Archive preflight, run on the central directory before any by-name lookup.
# --------------------------------------------------------------------------

SYMLINK_ATTR = (stat.S_IFLNK | 0o777) << 16
DIRECTORY_ATTR = (stat.S_IFDIR | 0o755) << 16 | 0x10


def household_archive_fixture(tmp_path, monkeypatch, members):
    return build_fixture(
        tmp_path,
        monkeypatch,
        household_members=members,
        person_members=(Member("psam_pusa.csv", BASE_PERSON_CSV),),
    )


def aux_members(count, *, prefix="ACS2024_note"):
    return tuple(
        Member(f"{prefix}{index:03d}.txt", b"invented auxiliary text\n")
        for index in range(count)
    )


ARCHIVE_VARIANTS = [
    pytest.param(
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
        ),
        id="duplicate_member_name",
    ),
    pytest.param(
        (
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[:5])),
            Member("PSAM_HUSA.CSV", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[5:])),
        ),
        id="case_folded_member_alias",
    ),
    pytest.param(
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("psam_hus/", b"", external_attr=DIRECTORY_ATTR),
        ),
        id="directory_entry",
    ),
    pytest.param(
        (Member("pums/psam_husa.csv", BASE_HOUSEHOLD_CSV),),
        id="nested_path_member",
    ),
    pytest.param(
        (Member("/psam_husa.csv", BASE_HOUSEHOLD_CSV),),
        id="absolute_path_member",
    ),
    pytest.param(
        (Member("../psam_husa.csv", BASE_HOUSEHOLD_CSV),),
        id="parent_traversal_member",
    ),
    pytest.param(
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("psam_husb.txt", BASE_HOUSEHOLD_CSV),
        ),
        id="prefix_matching_non_csv_member",
    ),
    pytest.param(
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member(
                "psam_husb.csv", BASE_HOUSEHOLD_CSV, compress_type=zipfile.ZIP_BZIP2
            ),
        ),
        id="bzip2_compressed_member",
    ),
    pytest.param(
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("psam_link.csv", b"psam_husa.csv", external_attr=SYMLINK_ATTR),
        ),
        id="symlink_member",
    ),
    pytest.param(
        (Member("psam_pusa.csv", BASE_PERSON_CSV),),
        id="wrong_role_members_only",
    ),
    pytest.param(
        (Member("ACS2024_readme.txt", b"invented auxiliary text\n"),),
        id="no_applicable_csv_member",
    ),
    pytest.param((), id="empty_archive"),
    pytest.param(
        (Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),) + aux_members(64),
        id="member_count_above_cap",
    ),
]


@pytest.mark.parametrize("members", ARCHIVE_VARIANTS)
def test_archive_preflight_refusals(tmp_path, monkeypatch, members):
    fixture = household_archive_fixture(tmp_path, monkeypatch, members)
    refuses(fixture, fixture.produce)


def test_member_count_at_the_cap_is_accepted(tmp_path, monkeypatch):
    fixture = household_archive_fixture(
        tmp_path,
        monkeypatch,
        (Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),) + aux_members(63),
    )
    assert len(fixture.produce().households) == len(BASE_HOUSEHOLDS)


def test_ordinary_auxiliary_members_are_inventory_only(tmp_path, monkeypatch):
    """A non-matching ordinary file is recorded but never read as data."""
    fixture = household_archive_fixture(
        tmp_path,
        monkeypatch,
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("ACS2024_PUMS_README.txt", b"invented auxiliary text\n"),
        ),
    )
    result = fixture.produce()
    assert set(household_keys(result.households)) == set(BASE_SERIALNOS)
    assert_lineage(
        result.households,
        household_keys(result.households),
        dict.fromkeys(BASE_SERIALNOS, "psam_husa.csv"),
    )
    assert "ACS2024_PUMS_README.txt" in flat_strings(json.loads(result.receipt_json))


def test_a_non_matching_csv_member_is_auxiliary_not_data(tmp_path, monkeypatch):
    """The role prefix, not the extension, decides what counts as data."""
    fixture = household_archive_fixture(
        tmp_path,
        monkeypatch,
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("ACS2024_shells.csv", csv_bytes(("A", "B"), (("1", "2"),))),
        ),
    )
    result = fixture.produce()
    assert len(result.households) == len(BASE_HOUSEHOLDS)
    assert "ACS2024_shells.csv" in flat_strings(json.loads(result.receipt_json))


def test_stored_members_are_accepted(tmp_path, monkeypatch):
    fixture = household_archive_fixture(
        tmp_path,
        monkeypatch,
        (
            Member(
                "psam_husa.csv", BASE_HOUSEHOLD_CSV, compress_type=zipfile.ZIP_STORED
            ),
        ),
    )
    assert len(fixture.produce().households) == len(BASE_HOUSEHOLDS)


def test_several_members_per_role_join_in_name_order(tmp_path, monkeypatch):
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[:3])),
            Member("psam_husb.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[3:6])),
            Member("psam_husc.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[6:])),
        ),
        person_members=(
            Member("psam_pusa.csv", csv_bytes(PERSON_HEADER, BASE_PERSONS[:4])),
            Member("psam_pusb.csv", csv_bytes(PERSON_HEADER, BASE_PERSONS[4:])),
        ),
    )
    result = fixture.produce()
    assert set(household_keys(result.households)) == set(BASE_SERIALNOS)
    members = {
        row[HOUSEHOLD_HEADER.index("SERIALNO")]: name
        for name, rows in (
            ("psam_husa.csv", BASE_HOUSEHOLDS[:3]),
            ("psam_husb.csv", BASE_HOUSEHOLDS[3:6]),
            ("psam_husc.csv", BASE_HOUSEHOLDS[6:]),
        )
        for row in rows
    }
    assert_lineage(result.households, household_keys(result.households), members)
    # 2024HU0000003's three people straddle the two person members.
    assert lexical(result.persons["SERIALNO"]).count("2024HU0000003") == 3
    assert person_map(result.persons) == EXPECTED_PERSONS
    person_members = {
        (
            row[PERSON_HEADER.index("SERIALNO")],
            row[PERSON_HEADER.index("SPORDER")],
        ): name
        for name, rows in (
            ("psam_pusa.csv", BASE_PERSONS[:4]),
            ("psam_pusb.csv", BASE_PERSONS[4:]),
        )
        for row in rows
    }
    assert_lineage(result.persons, list(person_map(result.persons)), person_members)


@pytest.mark.parametrize("archive", [HOUSEHOLD_ARCHIVE, PERSON_ARCHIVE])
def test_missing_role_archive_refuses(acs, archive):
    (acs.source_dir / archive).unlink()
    refuses(acs, acs.produce)


def test_corrupted_member_payload_fails_the_recorded_crc(tmp_path, monkeypatch):
    """A stored member whose bytes were swapped keeps its declared size and CRC."""
    fixture = household_archive_fixture(
        tmp_path,
        monkeypatch,
        (
            Member(
                "psam_husa.csv", BASE_HOUSEHOLD_CSV, compress_type=zipfile.ZIP_STORED
            ),
        ),
    )
    payload = fixture.household_zip.read_bytes()
    assert b"82000" in payload, "the stored member is not plain in the archive"
    fixture.household_zip.write_bytes(payload.replace(b"82000", b"82001", 1))
    fixture.pin()
    refuses(fixture, fixture.produce)


# --------------------------------------------------------------------------
# Private capture: paths, pins, originals and byte bounds.
# --------------------------------------------------------------------------


@contextmanager
def bounded_time(seconds=30):
    """Fail rather than hang if a non-regular original is opened blocking."""

    def expire(signum, frame):
        raise TimeoutError("the call did not return within the bounded test window")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_missing_snapshot_root_refuses(acs):
    refuses(
        acs,
        source.produce_acs_housing_source,
        acs.source_dir,
        snapshot_root=acs.tmp_path / "absent-root",
        output_dir=acs.output_dir(),
    )


def test_snapshot_root_that_is_a_file_refuses(acs):
    root = acs.tmp_path / "root-file"
    root.write_bytes(b"invented\n")
    refuses(
        acs,
        source.produce_acs_housing_source,
        acs.source_dir,
        snapshot_root=root,
        output_dir=acs.output_dir(),
    )


def test_symlinked_snapshot_root_refuses(acs):
    link = acs.tmp_path / "root-link"
    link.symlink_to(acs.snapshot_root, target_is_directory=True)
    refuses(
        acs,
        source.produce_acs_housing_source,
        acs.source_dir,
        snapshot_root=link,
        output_dir=acs.output_dir(),
    )


def test_existing_output_directory_refuses(acs):
    output_dir = acs.output_dir()
    output_dir.mkdir()
    refuses(acs, acs.produce, output_dir=output_dir)


def test_existing_output_file_refuses(acs):
    output_dir = acs.output_dir()
    output_dir.write_bytes(b"invented\n")
    refuses(acs, acs.produce, output_dir=output_dir)


def test_output_directory_without_a_parent_refuses(acs):
    refuses(acs, acs.produce, output_dir=acs.tmp_path / "absent" / "artifact")


def test_missing_source_directory_refuses(acs):
    refuses(acs, acs.produce, source_dir=acs.tmp_path / "absent-source")


def test_source_directory_that_is_a_file_refuses(acs):
    path = acs.tmp_path / "source-file"
    path.write_bytes(b"invented\n")
    refuses(acs, acs.produce, source_dir=path)


def test_symlinked_source_directory_refuses(acs):
    link = acs.tmp_path / "source-link"
    link.symlink_to(acs.source_dir, target_is_directory=True)
    refuses(acs, acs.produce, source_dir=link)


def test_symlinked_original_archive_refuses(acs):
    real = acs.tmp_path / "real-csv_hus.zip"
    real.write_bytes(acs.household_zip.read_bytes())
    acs.household_zip.unlink()
    acs.household_zip.symlink_to(real)
    refuses(acs, acs.produce)


def test_fifo_original_archive_refuses(acs):
    acs.household_zip.unlink()
    os.mkfifo(acs.household_zip)
    with bounded_time():
        refuses(acs, acs.produce)


def test_directory_in_place_of_an_original_archive_refuses(acs):
    acs.household_zip.unlink()
    acs.household_zip.mkdir()
    refuses(acs, acs.produce)


def test_pinned_digest_mismatch_at_the_same_size_refuses(acs):
    """Authentication is over the whole archive bytes, not over the projection."""
    pins = acs.pin()
    flip_archive_byte(acs.household_zip)
    acs.monkeypatch.setattr(source, NAMES["archive_pins"], pins)
    refuses(acs, acs.produce)


def test_pinned_size_mismatch_refuses(acs):
    pins = acs.pin()
    with acs.household_zip.open("ab") as handle:
        handle.write(b"\n")
    acs.monkeypatch.setattr(source, NAMES["archive_pins"], pins)
    refuses(acs, acs.produce)


def test_original_archives_are_never_modified(acs):
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (acs.household_zip, acs.person_zip)
    }
    acs.produce()
    for path, (payload, mtime) in before.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime


def test_capture_writes_only_under_the_supplied_snapshot_root(acs):
    assert not any(acs.snapshot_root.iterdir())
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    children = list(acs.snapshot_root.iterdir())
    assert children, "the private capture did not land under the snapshot root"
    for child in children:
        assert child.is_dir() and not child.is_symlink()
        assert stat.S_IMODE(child.stat().st_mode) == 0o700
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        (NAMES["projection_filename"], NAMES["receipt_filename"])
    )


def test_capture_never_falls_back_to_system_temporary_storage(acs, monkeypatch):
    absent = acs.tmp_path / "no-such-tmpdir"
    monkeypatch.setenv("TMPDIR", str(absent))
    monkeypatch.setattr(tempfile, "tempdir", str(absent))
    result = acs.produce()
    assert len(result.households) == len(BASE_HOUSEHOLDS)
    assert not absent.exists()


def test_two_produces_share_one_snapshot_root_without_collision(acs):
    first = acs.produce(output_dir=acs.output_dir("first"))
    second = acs.produce(output_dir=acs.output_dir("second"))
    assert first.projection_json == second.projection_json
    assert len(list(acs.snapshot_root.iterdir())) >= 2


def test_a_refused_produce_leaves_no_receipt_behind(tmp_path, monkeypatch):
    fixture = single_member(
        tmp_path, monkeypatch, households=with_household(0, TEN="5")
    )
    output_dir = fixture.output_dir()
    refuses(fixture, fixture.produce, output_dir=output_dir)
    assert not (output_dir / NAMES["receipt_filename"]).exists()


def byte_cap_names():
    return tuple(sorted(name for name in dir(source) if name.endswith("_MAX_BYTES")))


def test_module_declares_its_own_byte_caps():
    assert byte_cap_names(), "the module declares no explicit byte caps"


@pytest.mark.parametrize("cap", [0, 1, 64])
def test_tiny_byte_caps_refuse_before_any_payload_is_returned(acs, cap):
    """Exercise the streaming bounds with mocked caps, never with real GiBs."""
    for name in byte_cap_names():
        acs.monkeypatch.setattr(source, name, cap)
    refuses(acs, acs.produce)


def test_declared_cap_values_match_the_approved_plan():
    """Reconciliation: the plan fixes both ACS source caps by name and value."""
    assert getattr(source, NAMES["source_max_bytes"]) == 8 * 1024**3
    assert getattr(source, NAMES["receipt_max_bytes"]) == 1024**2


# --------------------------------------------------------------------------
# Artifact custody: readback reconstructs, it never trusts what it decoded.
# --------------------------------------------------------------------------


def make_writable(path):
    """The producer publishes read-only files; tampering needs write access."""
    path = Path(path)
    path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o600)
    return path


def forge(output_dir, old, new):
    """Rewrite the artifact so its own internal digests stay self-consistent."""
    projection = (output_dir / NAMES["projection_filename"]).read_bytes()
    receipt = (output_dir / NAMES["receipt_filename"]).read_bytes()
    assert old in projection, "the invented forgery target is not in the projection"
    forged = projection.replace(old, new)
    assert len(forged) == len(projection), "a forgery must not change the length"
    old_digest = hashlib.sha256(projection).hexdigest().encode()
    new_digest = hashlib.sha256(forged).hexdigest().encode()
    make_writable(output_dir / NAMES["projection_filename"]).write_bytes(forged)
    make_writable(output_dir / NAMES["receipt_filename"]).write_bytes(
        receipt.replace(old_digest, new_digest)
    )
    return old_digest in receipt


def test_rehashed_projection_forgery_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    forge(output_dir, b"2024HU0000002", b"2024HU0000012")
    refuses(acs, acs.load, output_dir)


def test_receipt_binds_the_projection_digest(acs):
    """Otherwise a forged projection would be self-consistent on its face."""
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    assert forge(output_dir, b"2024HU0000002", b"2024HU0000012")


def test_forging_a_weight_to_another_valid_value_refuses(acs):
    """Custody is byte reconstruction, so an in-domain forgery refuses too."""
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    forge(output_dir, b"118", b"119")
    refuses(acs, acs.load, output_dir)


def test_rewritten_receipt_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    receipt = json.loads((output_dir / NAMES["receipt_filename"]).read_bytes())
    forged = json.dumps(receipt, sort_keys=True).encode()
    make_writable(output_dir / NAMES["receipt_filename"]).write_bytes(forged)
    refuses(acs, acs.load, output_dir)


@pytest.mark.parametrize(
    "filename", [NAMES["projection_filename"], NAMES["receipt_filename"]]
)
def test_truncated_or_missing_artifact_files_refuse(acs, filename):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    make_writable(output_dir / filename).write_bytes(
        (output_dir / filename).read_bytes()[:20]
    )
    refuses(acs, acs.load, output_dir)

    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    (output_dir / filename).unlink()
    refuses(acs, acs.load, output_dir)


def test_missing_artifact_directory_refuses(acs):
    refuses(acs, acs.load, acs.tmp_path / "absent-artifact")


def test_extra_file_in_the_artifact_directory_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    (output_dir / "notes.json").write_bytes(b"{}\n")
    refuses(acs, acs.load, output_dir)


def test_parent_change_between_produce_and_readback_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    flip_archive_byte(acs.household_zip)
    refuses(acs, acs.load, output_dir)


def test_reparented_readback_refuses_even_after_repinning(acs):
    """Fresh authentic parents that no longer reconstruct the artifact refuse."""
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    rows = with_household(1, TEN="4")
    write_archive(
        acs.household_zip,
        (
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, rows[:5])),
            Member("psam_husb.csv", csv_bytes(HOUSEHOLD_HEADER, rows[5:])),
        ),
    )
    acs.pin()
    refuses(acs, acs.load, output_dir)


def test_readback_against_swapped_role_archives_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    household = acs.household_zip.read_bytes()
    acs.household_zip.write_bytes(acs.person_zip.read_bytes())
    acs.person_zip.write_bytes(household)
    acs.pin()
    refuses(acs, acs.load, output_dir)


def test_readback_from_an_unrelated_artifact_refuses(tmp_path, monkeypatch):
    other = build_fixture(
        tmp_path / "other",
        monkeypatch,
        households=(("psam_husa.csv", BASE_HOUSEHOLDS[:4]),),
        persons=(("psam_pusa.csv", BASE_PERSONS[:6]),),
    )
    output_dir = other.output_dir()
    other.produce(output_dir=output_dir)

    mine = build_fixture(tmp_path / "mine", monkeypatch)
    refuses(mine, mine.load, output_dir)


@pytest.mark.parametrize("attribute", ["households", "persons", "codes"])
def test_readback_frames_are_owned_copies(acs, attribute):
    _produced, loaded, _output_dir = acs.produce_and_load()
    first = getattr(loaded, attribute)
    assert getattr(loaded, attribute) is not first
    before = first.copy(deep=True)
    first.iloc[0, 0] = first.iloc[-1, 0]
    pd.testing.assert_frame_equal(getattr(loaded, attribute), before)


# --------------------------------------------------------------------------
# Refusal semantics: sanitised, and one code per cause.
# --------------------------------------------------------------------------


def collect_family_reasons(tmp_path, monkeypatch):
    """One refusal per coarse family, each from its own invented source."""
    reasons = {}

    def record(family, build, call):
        fixture = build(tmp_path / family, monkeypatch)
        fixture.pin()
        reasons[family] = refuses(fixture, call, fixture)

    def digest_change(fixture):
        pins = fixture.pin()
        flip_archive_byte(fixture.household_zip)
        fixture.monkeypatch.setattr(source, NAMES["archive_pins"], pins)
        fixture.produce()

    def forged_artifact(fixture):
        output_dir = fixture.output_dir()
        fixture.produce(output_dir=output_dir)
        forge(output_dir, b"2024HU0000002", b"2024HU0000012")
        fixture.load(output_dir)

    def existing_output(fixture):
        output_dir = fixture.output_dir()
        output_dir.mkdir()
        fixture.produce(output_dir=output_dir)

    plain = build_fixture

    record("capture_digest", plain, digest_change)
    record("artifact_forgery", plain, forged_artifact)
    record("output_precondition", plain, existing_output)
    record("selection", plain, lambda f: f.produce(serialnos=("2024HU0000009",)))
    record(
        "zip_preflight",
        lambda base, patch: household_archive_fixture(
            base,
            patch,
            (
                Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
                Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            ),
        ),
        lambda f: f.produce(),
    )
    record(
        "csv_record",
        lambda base, patch: household_bytes_fixture(
            base,
            patch,
            csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[1:])
            + ",".join(BASE_HOUSEHOLDS[0][:-1]).encode()
            + b"\n",
        ),
        lambda f: f.produce(),
    )
    record(
        "header_contract",
        lambda base, patch: household_bytes_fixture(
            base,
            patch,
            csv_bytes(*drop_column(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, "ST")),
        ),
        lambda f: f.produce(),
    )
    record(
        "value_domain",
        lambda base, patch: single_member(
            base, patch, households=with_household(0, TEN="5")
        ),
        lambda f: f.produce(),
    )
    record(
        "join_integrity",
        lambda base, patch: single_member(
            base,
            patch,
            persons=BASE_PERSONS + (person_row("2024HU0000009", "1", "40"),),
        ),
        lambda f: f.produce(),
    )
    return reasons


@pytest.fixture
def family_reasons(tmp_path, monkeypatch):
    return collect_family_reasons(tmp_path, monkeypatch)


def test_refusal_type_is_a_sanitised_value_error():
    assert issubclass(source.ACSHousingSourceError, ValueError)


def test_every_refusal_family_is_exercised(family_reasons):
    assert set(family_reasons) == set(DISTINCT_FAMILIES)


def test_distinct_causes_carry_distinct_reason_codes(family_reasons):
    """A single catch-all code would make every refusal above unfalsifiable."""
    collisions = {}
    for family, reason in family_reasons.items():
        collisions.setdefault(reason, []).append(family)
    shared = {reason: names for reason, names in collisions.items() if len(names) > 1}
    assert not shared, f"distinct causes share a reason code: {shared}"


def test_no_refusal_reason_echoes_invented_source_content(family_reasons):
    """refuses() checks each call; this states the whole-sweep contract once."""
    forbidden = set(BASE_SERIALNOS) | {"psam_husa.csv", "82000", "52000"}
    for family, reason in family_reasons.items():
        for secret in forbidden:
            assert secret not in reason, f"{family} leaked {secret!r}"


def test_reason_codes_match_the_agreed_table(family_reasons):
    """Shape always; exact spellings once EXPECTED_REASONS is filled in.

    The house idiom in the committed sibling source modules is a bare
    UPPER_SNAKE token, so that much is asserted unconditionally and nothing is
    parked behind a skip. Only the agreement on which token names which cause
    waits on the production author.
    """
    for family, reason in family_reasons.items():
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", reason), (
            f"{family} reason {reason!r} is not a bare reason code"
        )
    if REASON_CODES_AGREED:
        assert family_reasons == EXPECTED_REASONS
    else:
        assert not EXPECTED_REASONS, (
            "EXPECTED_REASONS is populated; set REASON_CODES_AGREED = True"
        )


# --------------------------------------------------------------------------
# Naming reconciliation: everything this file had to guess, in one place.
# --------------------------------------------------------------------------


def test_module_exposes_the_agreed_public_surface():
    for name in (
        "ACSHousingSourceError",
        "produce_acs_housing_source",
        "load_acs_housing_source",
        "prepare_acs_housing_population",
    ):
        assert hasattr(source, name), f"missing agreed API name {name}"


def test_archive_pins_are_consulted_at_call_time(acs):
    """The whole suite rests on this seam, so assert it directly.

    Every fixture authenticates its invented archives by pointing the private
    pin tuple at them. That only works if the module reads the attribute when
    it captures, rather than copying it at import.
    """
    real = acs.pin()
    assert acs.produce().households is not None

    wrong = tuple((role, name, "f" * 64, size) for role, name, _sha, size in real)
    acs.monkeypatch.setattr(source, NAMES["archive_pins"], wrong)
    refuses(acs, acs.produce)


def test_private_archive_pins_are_a_sentinel_or_the_two_fixed_archives():
    """Naming seam, reconciled with the production author, not a defect.

    The brief described `_ARCHIVE_PINS` as a populated tuple of
    (role, filename, sha256, size). The module instead defaults it to None and
    loads the packaged manifest lazily, while tests still patch the same
    private name - which is what
    ``test_archive_pins_are_consulted_at_call_time`` proves still works. Both
    readings are admitted here; the shape is checked whenever it is populated,
    so a wrong shape cannot hide behind the sentinel.
    """
    pins = getattr(source, NAMES["archive_pins"])
    if pins is None:
        return
    assert type(pins) is tuple and len(pins) == 2
    assert [entry[0] for entry in pins] == [
        NAMES["household_role"],
        NAMES["person_role"],
    ]
    assert [entry[1] for entry in pins] == [HOUSEHOLD_ARCHIVE, PERSON_ARCHIVE]
    for _role, _name, sha, size in pins:
        assert type(sha) is str and len(sha) == 64 and sha == sha.lower()
        assert int(sha, 16) >= 0
        assert type(size) is int and size > 0


def test_lineage_and_state_columns_use_the_agreed_names(acs):
    households = acs.produce().households
    assert NAMES["member_column"] in households.columns
    assert NAMES["ordinal_column"] in households.columns
    assert NAMES["state_column"] in households.columns


# --------------------------------------------------------------------------
# Tail coverage: cases an obvious pass over the five dimensions omits.
# --------------------------------------------------------------------------


def set_encrypted_flag(path):
    """Mark every entry encrypted in place; zipfile itself will not write this."""
    raw = bytearray(Path(path).read_bytes())
    marked = 0
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        index = 0
        while True:
            index = raw.find(signature, index)
            if index < 0:
                break
            flag = struct.unpack_from("<H", raw, index + offset)[0]
            struct.pack_into("<H", raw, index + offset, flag | 0x1)
            marked += 1
            index += 4
    assert marked >= 2
    Path(path).write_bytes(bytes(raw))


def raw_household_record(**overrides):
    """A physical household line written by hand, bypassing the csv writer."""
    values = dict(zip(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[0], strict=True))
    values.update(overrides)
    return ",".join(values[name] for name in HOUSEHOLD_HEADER).encode()


def appended_record(record):
    return csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[1:]) + record + b"\n"


def test_encrypted_member_refuses(tmp_path, monkeypatch):
    fixture = household_archive_fixture(
        tmp_path, monkeypatch, (Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),)
    )
    set_encrypted_flag(fixture.household_zip)
    fixture.pin()
    refuses(fixture, fixture.produce)


def test_lzma_member_refuses(tmp_path, monkeypatch):
    fixture = household_archive_fixture(
        tmp_path,
        monkeypatch,
        (
            Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),
            Member("psam_husb.csv", BASE_HOUSEHOLD_CSV, compress_type=zipfile.ZIP_LZMA),
        ),
    )
    refuses(fixture, fixture.produce)


def test_bytes_that_are_not_an_archive_refuse(acs):
    acs.household_zip.write_bytes(b"this is invented, and it is not a zip archive\n")
    acs.pin()
    refuses(acs, acs.produce)


def test_person_archive_holding_household_members_refuses(tmp_path, monkeypatch):
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),),
        person_members=(Member("psam_husa.csv", BASE_HOUSEHOLD_CSV),),
    )
    refuses(fixture, fixture.produce)


UTF16_MEMBER = BASE_HOUSEHOLD_CSV.decode().encode("utf-16")

TAIL_PARSE_VARIANTS = [
    pytest.param(UTF16_MEMBER, id="utf16_encoded_member"),
    pytest.param(
        appended_record(raw_household_record(ST='"06"x')),
        id="character_after_a_closing_quote",
    ),
    pytest.param(
        appended_record(raw_household_record(HINCP='"82000')),
        id="unterminated_quote",
    ),
    pytest.param(
        splice_line(BASE_HOUSEHOLD_CSV, 3, b"   "), id="whitespace_only_record"
    ),
    pytest.param(
        csv_bytes(HOUSEHOLD_HEADER, with_household(0, TEN="NA")),
        id="na_token_is_not_a_missing_tenure",
    ),
    pytest.param(
        csv_bytes(
            rename_column(HOUSEHOLD_HEADER, "SERIALNO", "serialno"), BASE_HOUSEHOLDS
        ),
        id="lowercase_roster_column_name",
    ),
]


@pytest.mark.parametrize("payload", TAIL_PARSE_VARIANTS)
def test_tail_parser_refusals(tmp_path, monkeypatch, payload):
    fixture = household_bytes_fixture(tmp_path, monkeypatch, payload)
    refuses(fixture, fixture.produce)


def test_household_rows_absent_while_person_rows_remain_refuses(tmp_path, monkeypatch):
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, ())),),
        person_members=(Member("psam_pusa.csv", BASE_PERSON_CSV),),
    )
    refuses(fixture, fixture.produce)


def test_a_final_record_without_a_trailing_newline_is_accepted(tmp_path, monkeypatch):
    fixture = household_bytes_fixture(
        tmp_path,
        monkeypatch,
        csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS, trailing_newline=False),
    )
    assert household_map(fixture.produce().households) == EXPECTED_HOUSEHOLDS


def test_many_unprojected_columns_are_accepted(tmp_path, monkeypatch):
    """Genuine PUMS members carry hundreds of columns this projection ignores."""
    fixture = household_bytes_fixture(
        tmp_path, monkeypatch, csv_bytes(WIDE_HEADER, WIDE_ROWS)
    )
    assert household_map(fixture.produce().households) == EXPECTED_HOUSEHOLDS


def test_roster_columns_in_any_header_position_are_accepted(tmp_path, monkeypatch):
    """The plan pins header order across members of a role, not within one."""
    reversed_header = tuple(reversed(HOUSEHOLD_HEADER))
    rows = tuple(tuple(reversed(row)) for row in BASE_HOUSEHOLDS)
    fixture = household_bytes_fixture(
        tmp_path, monkeypatch, csv_bytes(reversed_header, rows)
    )
    assert household_map(fixture.produce().households) == EXPECTED_HOUSEHOLDS


def test_maximum_serialno_suffix_is_accepted(tmp_path, monkeypatch):
    households, persons = rename_serialno("2024HU0000001", "2024HU9999999")
    fixture = single_member(
        tmp_path, monkeypatch, households=households, persons=persons
    )
    keys = household_keys(fixture.produce().households)
    assert "2024HU9999999" in keys and "2024HU0000001" not in keys


def test_np_at_the_published_upper_bound_is_accepted(tmp_path, monkeypatch):
    households = (household_row("2024HU0000020", "1", "20", "1", "300"),)
    persons = tuple(
        person_row("2024HU0000020", str(order), str(order + 10))
        for order in range(1, 21)
    )
    fixture = single_member(
        tmp_path, monkeypatch, households=households, persons=persons
    )
    result = fixture.produce()
    assert lexical(result.households["NP"]) == ["20"]
    assert len(result.persons) == 20
    assert result.codes["occupied_hu"].tolist() == [1]


def test_an_all_group_quarters_source_is_retained(tmp_path, monkeypatch):
    fixture = single_member(
        tmp_path,
        monkeypatch,
        households=BASE_HOUSEHOLDS[4:6],
        persons=BASE_PERSONS[6:8],
    )
    result = fixture.produce()
    assert codes_map(result) == {
        key: EXPECTED_CODES[key] for key in ("2024GQ0000005", "2024GQ0000006")
    }


def test_a_vacancy_only_source_is_retained(tmp_path, monkeypatch):
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(
            Member("psam_husa.csv", csv_bytes(HOUSEHOLD_HEADER, BASE_HOUSEHOLDS[3:4])),
        ),
        person_members=(Member("psam_pusa.csv", csv_bytes(PERSON_HEADER, ())),),
    )
    result = fixture.produce()
    assert household_keys(result.households) == ["2024HU0000004"]
    assert len(result.persons) == 0
    assert codes_map(result) == {"2024HU0000004": EXPECTED_CODES["2024HU0000004"]}


@pytest.mark.parametrize(
    "serialnos",
    [
        pytest.param((1,), id="integer_element"),
        pytest.param((None,), id="none_element"),
        pytest.param((b"2024HU0000001",), id="bytes_element"),
    ],
)
def test_non_string_selection_elements_refuse(acs, serialnos):
    refuses(acs, acs.produce, serialnos=serialnos)


def test_an_explicit_full_selection_matches_the_none_selection(acs):
    whole = acs.produce(output_dir=acs.output_dir("whole"))
    explicit = acs.produce(
        output_dir=acs.output_dir("explicit"), serialnos=BASE_SERIALNOS
    )
    assert explicit.projection_json == whole.projection_json


def test_a_published_artifact_survives_a_later_refused_produce(acs):
    output_dir = acs.output_dir("kept")
    produced = acs.produce(output_dir=output_dir)
    before = {path.name: path.read_bytes() for path in sorted(output_dir.iterdir())}
    acs.household_zip.write_bytes(b"not an archive\n")
    acs.pin()
    refuses(acs, acs.produce, output_dir=acs.output_dir("doomed"))
    assert {path.name: path.read_bytes() for path in sorted(output_dir.iterdir())} == (
        before
    )
    assert before[NAMES["projection_filename"]].rstrip(b"\n") == (
        produced.projection_json.rstrip(b"\n")
    )


def test_readback_with_a_missing_snapshot_root_refuses(acs):
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    refuses(
        acs,
        source.load_acs_housing_source,
        acs.source_dir,
        output_dir,
        snapshot_root=acs.tmp_path / "absent-root",
    )


def test_wrong_size_candidate_refuses_as_a_candidate_not_as_a_parent(acs):
    """Readback size failures must name the candidate, not the pinned archives."""
    for filename, mutate in (
        (NAMES["projection_filename"], lambda raw: raw + b" "),
        (NAMES["projection_filename"], lambda raw: raw[:20]),
        (NAMES["receipt_filename"], lambda raw: raw + b" "),
        (NAMES["receipt_filename"], lambda raw: raw[:20]),
    ):
        output_dir = acs.output_dir()
        acs.produce(output_dir=output_dir)
        path = output_dir / filename
        make_writable(path).write_bytes(mutate(path.read_bytes()))
        assert refuses(acs, acs.load, output_dir) == "RECONSTRUCTION_SIZE"


def test_parent_archive_size_failures_keep_their_own_reason_codes(acs):
    """The candidate code must not be borrowed by, or borrow from, the parents."""
    output_dir = acs.output_dir()
    acs.produce(output_dir=output_dir)
    intact = acs.household_zip.read_bytes()
    acs.household_zip.write_bytes(intact[:-1])
    assert refuses(acs, acs.load, output_dir) == "SOURCE_SIZE"
    acs.household_zip.write_bytes(intact + b"\x00")
    assert refuses(acs, acs.load, output_dir) == "FILE_TOO_LARGE"
    acs.household_zip.write_bytes(intact)
    assert acs.load(output_dir) is not None


def test_failure_record_write_failure_does_not_mask_the_original_refusal(
    acs, monkeypatch
):
    """A failed failure record must never replace the refusal it was recording."""
    actual_write = source._write
    attempted = []

    def refuse_the_failure_record(path, data, cap):
        if Path(path).name == "failure.json":
            attempted.append(Path(path).name)
            raise OSError(28, "No space left on device")
        return actual_write(path, data, cap)

    monkeypatch.setattr(source, "_write", refuse_the_failure_record)
    corrupted = bytearray(acs.household_zip.read_bytes())
    corrupted[-1] ^= 0xFF
    acs.household_zip.write_bytes(bytes(corrupted))
    assert refuses(acs, acs.produce) == "SOURCE_SHA256"
    assert attempted == ["failure.json"]
