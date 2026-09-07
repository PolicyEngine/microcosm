"""Closed Census interview-week controls, separate from the T1 money source.

These observations do not establish annual or five-month tax-student status.
Candidate bytes never supply parsed content or authority: replay reconstructs S
from the authenticated T1 parent and the code-owned official CSV members.
"""

from __future__ import annotations

import csv
import hashlib
import os
import stat
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.frame import Frame

from . import asec_current_money_source as legacy
from . import asec_person_income_source as restoration
from ._asec_current_money_codec import current_money_content_sha256
from .asec_current_money import (
    RESTORED_SOURCE_KIND,
    MoneyRefusalError,
    ReadyCurrentMoney,
    _json,
    _parse,
    _require,
    _sha,
    _validate_money,
)
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

CONTROLS = ("A_ENRLW", "A_FTPT")
ALIASES = ("asec_A_ENRLW", "asec_A_FTPT")
COORDINATES = ("person_id", "income_year", "source_household_id", "A_LINENO", "A_AGE")
COLUMNS = COORDINATES + CONTROLS
_READ_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE") + CONTROLS
_MEMBER_PINS = tuple(
    (year, p.member, p.zip_sha256, p.member_sha256, p.rows, p.member_size_bytes)
    for year, p in sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items())
)
MAGIC = b"MCASTUD\x01"
FILENAME = "asec_student_controls.bin"
_HEADER_MAX = 65536
_MAX_PERSONS = 600_000
_TOKEN = object()
_COMPOSE_TOKEN = object()


def _implementation():
    package = resources.files(__package__)
    return {
        "schema": 1,
        "t1_verification_sha256": restoration._implementation(),
        "modules": {
            name: _sha(package.joinpath(name).read_bytes())
            for name in ("asec_student_controls.py", "_asec_current_money_codec.py")
        },
        "dependencies": {n: metadata.version(n) for n in ("numpy", "pandas")},
        "member_pins": _MEMBER_PINS,
        "aliases": dict(zip(CONTROLS, ALIASES, strict=True)),
    }


def _parent(source, ready, *, reconstruct_money=False):
    _require(
        type(source) is legacy.AuthenticatedCurrentMoneySource
        and type(ready) is ReadyCurrentMoney,
        "STUDENT_AUTHENTICATED_PARENT",
    )
    source.validate()
    _require(
        _parse(source.source.identity)["source_kind"] == RESTORED_SOURCE_KIND,
        "STUDENT_RESTORED_PARENT_REQUIRED",
    )
    _require(ready.bindings.spec == source.spec, "STUDENT_MONEY_PARENT")
    _validate_money(ready, source.spec, nominal=False)
    if reconstruct_money:
        expected = source.ready()
        _require(
            ready.bindings == expected.bindings and ready.fields == expected.fields,
            "STUDENT_MONEY_PARENT",
        )


def _snapshot(path, destination, *, size):
    """Exact-size private capture; no candidate parser and no unbounded copy."""
    digest = hashlib.sha256()
    # Opening a FIFO in ordinary blocking mode would wait before fstat could
    # reject it. Use one nonblocking descriptor, then admit only regular files.
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "STUDENT_SOURCE_FILE_KIND")
        _require(before.st_size == size, "STUDENT_SOURCE_SIZE")
        count = 0
        with (
            os.fdopen(descriptor, "rb", closefd=False) as src,
            destination.open("xb") as dst,
        ):
            while chunk := src.read(min(1024 * 1024, size - count + 1)):
                count += len(chunk)
                _require(count <= size, "STUDENT_SOURCE_SIZE")
                digest.update(chunk)
                dst.write(chunk)
        after = os.fstat(descriptor)
        attrs = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(
            count == size
            and all(getattr(before, a) == getattr(after, a) for a in attrs),
            "STUDENT_SOURCE_CHANGED",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_member(path, *, rows):
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    _require(
        len(header) == len(set(header)) and set(_READ_COLUMNS) <= set(header),
        "STUDENT_MEMBER_HEADER",
    )
    table = pd.read_csv(
        path, usecols=list(_READ_COLUMNS), dtype="string", na_filter=False
    )
    _require(len(table) == rows, "STUDENT_MEMBER_ROWS")
    _require(
        bool(table.PERIDNUM.str.fullmatch(r"[0-9]{22}").all()),
        "STUDENT_MEMBER_PERSON_KEY",
    )
    for name in _READ_COLUMNS[1:]:
        _require(
            bool(table[name].str.fullmatch(r"[0-9]+").all()),
            "STUDENT_MEMBER_INTEGER",
            name,
        )
        table[name] = table[name].astype("int64")
    _require(
        not table.PERIDNUM.duplicated().any()
        and not table.duplicated(["PH_SEQ", "A_LINENO"]).any()
        and bool((table.PH_SEQ > 0).all())
        and bool((table.A_LINENO > 0).all()),
        "STUDENT_MEMBER_COORDINATES",
    )
    for name in CONTROLS:
        _require(bool(table[name].isin([0, 1, 2]).all()), "STUDENT_CODE_DOMAIN", name)
    _require(
        bool((table.A_ENRLW[~table.A_AGE.between(16, 54)] == 0).all())
        and bool((table.A_FTPT[table.A_ENRLW == 1] != 0).all())
        and bool((table.A_FTPT[table.A_ENRLW != 1] == 0).all()),
        "STUDENT_SOURCE_UNIVERSE",
    )
    return table


@dataclass(frozen=True)
class AuthenticatedStudentControls:
    """Immutable owned numeric buffers; only closed source reconstruction issues S."""

    _header: bytes
    _body: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "STUDENT_CONSTRUCTOR_UNAVAILABLE")
        self.validate()

    @property
    def receipt(self):
        return _parse(self._header)

    @property
    def content_sha256(self):
        return _sha(self._header + self._body)

    def validate(self):
        _require(
            type(self._header) is bytes and 0 < len(self._header) <= _HEADER_MAX,
            "STUDENT_HEADER_SIZE",
        )
        header = self.receipt
        rows = header["rows"]
        _require(type(rows) is int and 0 < rows <= _MAX_PERSONS, "STUDENT_ROWS")
        _require(
            type(self._body) is bytes
            and len(self._body) == rows * len(COLUMNS) * 8
            and _sha(self._body) == header["body_sha256"]
            and header["columns"] == list(COLUMNS)
            and header["aliases"] == dict(zip(CONTROLS, ALIASES, strict=True))
            and _json(header["implementation"]) == _json(_implementation()),
            "STUDENT_CONTENT_CHANGED",
        )

    def array(self, name):
        self.validate()
        _require(name in COLUMNS, "STUDENT_COLUMN")
        n = self.receipt["rows"]
        offset = COLUMNS.index(name) * n * 8
        # A bytes-backed buffer cannot be made writeable by the caller.
        return np.frombuffer(self._body, dtype="<i8", count=n, offset=offset)


def _encode(value):
    value.validate()
    payload = (
        MAGIC + struct.pack("<I", len(value._header)) + value._header + value._body
    )
    return payload + hashlib.sha256(payload).digest()


def _reconstruct(source, ready, member_paths: Mapping[int, str | Path]):
    restoration._paths(member_paths)
    _parent(source, ready, reconstruct_money=True)
    before = _implementation()
    person = source.frame.person
    n = len(person)
    _require(0 < n <= _MAX_PERSONS, "STUDENT_ROWS")
    _require(not set(ALIASES) & set(person), "STUDENT_ALREADY_ATTACHED")
    _require(set(CONTROLS + COORDINATES[2:]) <= set(person), "STUDENT_PARENT_ROSTER")
    years = np.asarray(source.scope.person_years, dtype=np.int64)
    keys = np.asarray(source.scope.person_native_keys)
    output = {
        "person_id": np.asarray(source.scope.person_ids, dtype=np.int64),
        "income_year": years,
    }
    for name in COORDINATES[2:]:
        _require(
            person[name].dtype == np.dtype("int64"), "STUDENT_COORDINATE_DTYPE", name
        )
        output[name] = person[name].to_numpy(copy=True)
    for name in CONTROLS:
        _require(
            pd.api.types.is_numeric_dtype(person[name].dtype)
            and not pd.api.types.is_bool_dtype(person[name].dtype),
            "STUDENT_INCUMBENT_DTYPE",
            name,
        )
        output[name] = np.empty(n, dtype=np.int64)
    joins = []
    with tempfile.TemporaryDirectory(prefix="microcosm-student-members-") as directory:
        for year, member, archive_pin, pin, rows, size in _MEMBER_PINS:
            staged = Path(directory) / member
            _require(
                _snapshot(member_paths[year], staged, size=size) == pin,
                "STUDENT_SOURCE_BYTES",
            )
            raw = _read_member(staged, rows=rows)
            staged.unlink()
            positions = np.flatnonzero(years == year)
            _require(len(positions) == rows, "STUDENT_COHORT_ROWS")
            recipient = pd.Index(keys[positions])
            _require(not recipient.has_duplicates, "STUDENT_PERSON_KEY")
            indices = pd.Index(raw.PERIDNUM).get_indexer(recipient)
            _require(
                bool((indices >= 0).all()) and len(np.unique(indices)) == rows,
                "STUDENT_KEY_COVERAGE",
            )
            joined = raw.iloc[indices]
            for name, raw_name in (
                ("source_household_id", "PH_SEQ"),
                ("A_LINENO", "A_LINENO"),
                ("A_AGE", "A_AGE"),
            ):
                _require(
                    np.array_equal(
                        output[name][positions], joined[raw_name].to_numpy()
                    ),
                    "STUDENT_NATIVE_KEY_OR_AGE",
                )
            compared = {}
            for name in CONTROLS:
                incumbent = person[name].iloc[positions]
                known = ~incumbent.isna().to_numpy()
                observed = joined[name].to_numpy(dtype=np.int64)
                _require(
                    np.array_equal(
                        incumbent.to_numpy(dtype=np.float64, na_value=np.nan)[known],
                        observed[known],
                    ),
                    "STUDENT_INCUMBENT_CONFLICT",
                    name,
                )
                if year == 2024:
                    _require(bool(known.all()), "STUDENT_INCUMBENT_INCOMPLETE", name)
                output[name][positions] = observed
                compared[name] = int(known.sum())
            joins.append(
                {
                    "income_year": year,
                    "survey_year": year + 1,
                    "member": member,
                    "archive_sha256": archive_pin,
                    "member_sha256": pin,
                    "source_rows": rows,
                    "joined_rows": len(positions),
                    "unreferenced_source_rows": 0,
                    "incumbent_compared_rows": compared,
                    "incumbent_conflicts": 0,
                    "native_key_or_age_conflicts": 0,
                }
            )
    _parent(source, ready)
    _require(
        _json(_implementation()) == _json(before), "STUDENT_IMPLEMENTATION_CHANGED"
    )
    buffers = [output[name].astype("<i8", copy=False).tobytes() for name in COLUMNS]
    body = b"".join(buffers)
    evidence = _parse(source.source.identity)
    header = {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_student_controls.v1",
        "rows": n,
        "columns": COLUMNS,
        "dtype": "little_endian_int64",
        "aliases": dict(zip(CONTROLS, ALIASES, strict=True)),
        "source_identity": source.source.identity.decode(),
        "t1_sha256": evidence["person_income_attachment_sha256"],
        "source_frame_sha256": evidence["frame_sha256"],
        "scope_sha256": _parse(ready.header)["scope_sha256"],
        "native_person_keys_sha256": _sha(_json(source.scope.person_native_keys)),
        "money_header_sha256": _sha(ready.header),
        "money_content_sha256": current_money_content_sha256(ready),
        "sources": joins,
        "join_keys": ["source_year", "PERIDNUM"],
        "crosschecks": ["source_household_id", "A_LINENO", "A_AGE"],
        "reference_period": "preceding_survey_week_income_cohort_plus_one",
        "annual_five_month_student_status_validated": False,
        "zero_origin": "authenticated_census_csv_encoded_not_in_universe",
        "release_eligible": False,
        "implementation": before,
        "body_sha256": _sha(body),
        "column_sha256": dict(zip(COLUMNS, map(_sha, buffers), strict=True)),
        "encoding_contract": "independently_reconstructed_canonical_numeric_bytes_v1",
    }
    return AuthenticatedStudentControls(_json(header), body, _token=_TOKEN)


def load_authenticated_student_controls(
    source, ready, *, member_paths, candidate_path=None
):
    """Issue S from exact sources; optional replay admits only canonical candidate bytes."""
    try:
        expected = _reconstruct(source, ready, member_paths)
        if candidate_path is not None:
            payload = _encode(expected)
            with tempfile.TemporaryDirectory(
                prefix="microcosm-student-candidate-"
            ) as directory:
                digest = _snapshot(
                    candidate_path, Path(directory) / FILENAME, size=len(payload)
                )
            _require(digest == _sha(payload), "STUDENT_CANONICAL_BYTES")
        _parent(source, ready)
        expected.validate()
        return expected
    except MoneyRefusalError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        csv.Error,
    ):
        raise MoneyRefusalError("STUDENT_SOURCE_CONTRACT") from None


def write_student_controls(source, ready, *, member_paths, output_dir):
    """Produce a new immutable local S bundle; never replace any existing parent."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    value = load_authenticated_student_controls(
        source, ready, member_paths=member_paths
    )
    payload = _encode(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".asec-student-", dir=destination.parent
    ) as directory:
        staging = Path(directory)
        (staging / FILENAME).write_bytes(payload)
        receipt = {
            **value.receipt,
            "content_sha256": value.content_sha256,
            "output_file": FILENAME,
            "output_sha256": _sha(payload),
        }
        (staging / "receipt.json").write_bytes(_json(receipt) + b"\n")
        destination.mkdir()
        for path in staging.iterdir():
            os.link(path, destination / path.name)
    return receipt


@dataclass(frozen=True)
class StudentControlsAttachedAsec:
    """Owned additive Frame with the unchanged authenticated T1/money parents."""

    frame: Frame
    parent: legacy.AuthenticatedCurrentMoneySource
    money: ReadyCurrentMoney
    controls: AuthenticatedStudentControls
    _receipt: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _COMPOSE_TOKEN, "STUDENT_ATTACHMENT_CONSTRUCTOR_UNAVAILABLE")

    @property
    def receipt(self):
        return _parse(self._receipt)

    @property
    def source(self):
        return self.parent.source

    @property
    def spec(self):
        return self.parent.spec

    def ready(self):
        self.validate()
        return self.money

    def validate(self):
        try:
            _check_attachment_parents(self.parent, self.money, self.controls)
            _require(type(self.frame) is Frame, "STUDENT_ATTACHMENT_FRAME")
            for control, alias in zip(CONTROLS, ALIASES, strict=True):
                values = self.frame.person[alias]
                _require(
                    values.dtype == np.dtype("int64")
                    and np.array_equal(values.to_numpy(), self.controls.array(control)),
                    "STUDENT_ATTACHMENT_OUTPUT",
                )
            recovered = restoration._owned_frame(self.frame)
            recovered.person.drop(columns=list(ALIASES), inplace=True)
            _require(
                legacy._frame_signature(recovered)
                == _parse(self.parent.source.identity)["frame_sha256"]
                and self.receipt
                == _attachment_receipt(
                    self.frame, self.parent, self.money, self.controls
                ),
                "STUDENT_ATTACHMENT_CHANGED",
            )
        except MoneyRefusalError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, AssertionError):
            raise MoneyRefusalError("STUDENT_ATTACHMENT_CHANGED") from None


def _check_attachment_parents(source, ready, controls):
    _parent(source, ready)
    _require(
        type(controls) is AuthenticatedStudentControls, "STUDENT_AUTHENTICATED_CONTROLS"
    )
    controls.validate()
    header = controls.receipt
    _require(
        header["source_identity"].encode() == source.source.identity
        and header["money_header_sha256"] == _sha(ready.header)
        and header["money_content_sha256"] == current_money_content_sha256(ready),
        "STUDENT_ATTACHMENT_PARENT",
    )


def _attachment_receipt(frame, source, ready, controls):
    return {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_student_controls_attachment.v1",
        "source_identity_sha256": _sha(source.source.identity),
        "money_header_sha256": _sha(ready.header),
        "money_content_sha256": current_money_content_sha256(ready),
        "controls_content_sha256": controls.content_sha256,
        "aliases": dict(zip(CONTROLS, ALIASES, strict=True)),
        "output_frame_sha256": legacy._frame_signature(frame),
        "release_eligible": False,
    }


def attach_student_controls(source, ready, controls):
    """Append exactly two aliases; preserve old missing columns and monetary authority."""
    _check_attachment_parents(source, ready, controls)
    _require(not set(ALIASES) & set(source.frame.person), "STUDENT_ALREADY_ATTACHED")
    frame = restoration._owned_frame(source.frame)
    for name, alias in zip(CONTROLS, ALIASES, strict=True):
        frame.person[alias] = controls.array(name).copy()
    result = StudentControlsAttachedAsec(
        frame,
        source,
        ready,
        controls,
        _json(_attachment_receipt(frame, source, ready, controls)),
        _token=_COMPOSE_TOKEN,
    )
    result.validate()
    return result
