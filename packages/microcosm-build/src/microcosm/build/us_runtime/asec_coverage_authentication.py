"""Closed, source-only PRPERTYP evidence for an exact current-money parent.

No attachment, public pin override, population preparation or publication API.
The low-level reader remains non-authoritative inside this owned envelope.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import struct
import sys
import tempfile
import weakref
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path
from types import CodeType, FunctionType

import pandas as pd

from . import asec_current_money_source as money
from . import asec_person_coverage_source as literal
from . import source_csv_builtin as _csv_builtin
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

ARTIFACT_KIND = "microcosm.asec_coverage_authentication.v1"
MAGIC = b"MCASCOV\x01"
_MEMBER_PINS = tuple(
    (year, p.member, p.zip_sha256, p.member_sha256, p.rows, p.member_size_bytes)
    for year, p in sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items())
)
_HEADER_MAX = 262_144
_BODY_MAX = 268_435_456
_ROW_MAX = 1_048_576
_TOKEN_MAX = 65_536
_CSV_HEADER_MAX = 65_536
_MAX_PERSONS = 600_000
_CHUNK = 65_536
_INT_COLUMNS = (
    "person_id",
    "person_household_id",
    "source_year",
    "source_household_id",
    "A_LINENO",
    "A_AGE",
)
_STRING_COLUMNS = ("PERIDNUM", "PRPERTYP", "PRPERTYP_state")
COLUMNS = (
    "person_id",
    "person_household_id",
    "source_year",
    "source_household_id",
    "A_LINENO",
    "A_AGE",
    "PERIDNUM",
    "PRPERTYP",
    "PRPERTYP_state",
)
_PARENT_COLUMNS = (
    "person_id",
    "person_household_id",
    "source_year",
    "source_household_id",
    "A_LINENO",
    "A_AGE",
    "PERIDNUM",
)
_NUMBERS = struct.Struct("<6q")
_LENGTH = struct.Struct("<I")
_TOKEN = object()
# Identity-based, weak issuance seals also reject object.__setattr__ rewrites
# with recomputed self-checksums. Private Python internals are not a sandbox.
_ISSUED = weakref.WeakKeyDictionary()


class AsecCoverageAuthenticationError(ValueError):
    """Static codes only, with source paths and exception chains suppressed."""


def _require(condition, code):
    if not condition:
        raise AsecCoverageAuthenticationError(code)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _constant_document(value):
    """Typed, deterministic identities for Python code constants."""
    if isinstance(value, CodeType):
        return {"code": _code_document(value)}
    if type(value) is tuple:
        return {"tuple": [_constant_document(item) for item in value]}
    if type(value) is slice:
        return {
            "slice": [
                _constant_document(value.start),
                _constant_document(value.stop),
                _constant_document(value.step),
            ]
        }
    if type(value) is frozenset:
        return {
            "frozenset": sorted((_constant_document(item) for item in value), key=_json)
        }
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is float:
        return {"float": value.hex()}
    if type(value) is complex:
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if value is Ellipsis:
        return {"ellipsis": True}
    if value is None or type(value) in (str, int, bool):
        return {type(value).__name__: value}
    raise AsecCoverageAuthenticationError("COVERAGE_CODE_CONSTANT")


def _code_document(code):
    # Marshal's reference/intern flags can change when cold code is exercised.
    # Bind the semantic code attributes instead of that serialization state.
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_constant_document(c) for c in code.co_consts],
        "names": [code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars],
        "arguments": [
            code.co_argcount,
            code.co_posonlyargcount,
            code.co_kwonlyargcount,
        ],
        "flags": code.co_flags,
        "exceptions": code.co_exceptiontable.hex(),
    }


def _runtime_code(module):
    result = {}
    for name, value in vars(module).items():
        if isinstance(value, FunctionType):
            result[name] = _sha(_json(_code_document(value.__code__)))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            for method, function in vars(value).items():
                if isinstance(function, property):
                    function = function.fget
                if isinstance(function, FunctionType):
                    result[name + "." + method] = _sha(
                        _json(_code_document(function.__code__))
                    )
    return result


def _implementation():
    _require(
        _csv_builtin.csv_reader_bound(csv)
        and _csv_builtin.csv_reader_bound(literal.csv),
        "SOURCE_CSV_READER_CHANGED",
    )
    package = resources.files(__package__)
    _require(
        _runtime_code(literal) == _LITERAL_AUTHORITY,
        "COVERAGE_LITERAL_IMPLEMENTATION_CHANGED",
    )
    return {
        "producer": ARTIFACT_KIND,
        "modules": {
            name: _sha(package.joinpath(name).read_bytes())
            for name in (
                "asec_coverage_authentication.py",
                "source_csv_builtin.py",
                "asec_person_coverage_source.py",
                "asec_current_money_source.py",
                "education_assistance_source.py",
            )
        },
        "runtime_code": {
            m.__name__: _runtime_code(m) for m in (sys.modules[__name__], literal)
        },
        "dependencies": {name: metadata.version(name) for name in ("numpy", "pandas")},
        "python": sys.version,
        "csv_field_size_limit": csv.field_size_limit(),
        "member_pins": _MEMBER_PINS,
        "columns": COLUMNS,
        "encoding": {
            "integer_columns": _INT_COLUMNS,
            "string_columns": _STRING_COLUMNS,
            "parent_columns": _PARENT_COLUMNS,
            "numbers": _NUMBERS.format,
            "length": _LENGTH.format,
            "magic": MAGIC.hex(),
        },
        "limits": [
            _HEADER_MAX,
            _BODY_MAX,
            _ROW_MAX,
            _TOKEN_MAX,
            _CSV_HEADER_MAX,
            _MAX_PERSONS,
            _CHUNK,
        ],
        "reader_contract": [
            literal.PROTOCOL,
            literal.SOURCE_YEARS,
            literal.ROSTER_COLUMNS,
            literal.READ_COLUMNS,
            literal.MAX_PERSONS,
            literal.MAX_MEMBER_BYTES,
        ],
        "field_contract": literal.coverage_field_contract(),
    }


def _identity(snapshot):
    return (
        snapshot.st_dev,
        snapshot.st_ino,
        snapshot.st_size,
        snapshot.st_mtime_ns,
        snapshot.st_ctime_ns,
    )


class _CsvBounds:
    """Raw CSV framing only, integrated into capture before any text decoding.

    Quotes open at field start, doubled quotes continue quoted fields, and
    CR/LF only end records outside quotes. The original strict reader owns CSV
    syntax and coordinate parsing. Counts include quotes/delimiters: a bounded
    conservative limit on decoded token size, not a normalization.
    """

    def __init__(self, budget):
        self.row = self.field = 0
        self.header = self.start = True
        self.quoted = self.after_quote = self.after_cr = False
        self.column = self.token_column = 0
        self.header_bytes = bytearray()
        self.budget = budget

    def _end_header(self):
        # Only a bounded header is decoded here to locate the literal column.
        # No raw coordinate/age is parsed twice. Quoted source bytes give a
        # conservative upper bound on the eventual UTF-8 token length.
        csv_reader = _csv_builtin.capture_csv_reader(csv)
        _require(csv_reader is not None, "SOURCE_CSV_READER_CHANGED")
        with io.StringIO(self.header_bytes.decode("utf-8"), newline="") as handle:
            header = next(csv_reader(handle, strict=True), [])
        _require(header.count("PRPERTYP") == 1, "COVERAGE_CSV_HEADER")
        self.token_column = header.index("PRPERTYP")
        self.header_bytes.clear()

    def feed(self, chunk):
        for byte in chunk:
            if self.after_cr and byte == 10:
                self.after_cr = False
                continue
            self.after_cr = False
            self.row += 1
            self.field += 1
            _require(
                self.row <= (_CSV_HEADER_MAX if self.header else _ROW_MAX),
                "COVERAGE_CSV_RECORD_BYTES",
            )
            _require(self.field <= _TOKEN_MAX, "COVERAGE_CSV_TOKEN_BYTES")
            if self.header:
                self.header_bytes.append(byte)
            else:
                if self.row == 1:
                    # Row length + six integers + three lengths + fixed native
                    # key + a conservative 32 bytes for any closed status.
                    self.budget[0] -= 4 + _NUMBERS.size + 12 + 22 + 32
                if self.column == self.token_column:
                    self.budget[0] -= 1
                _require(self.budget[0] >= 0, "COVERAGE_BODY_BYTES")
            if self.quoted:
                if byte == 34:
                    self.quoted, self.after_quote = False, True
            elif self.after_quote and byte == 34:
                self.quoted, self.after_quote = True, False
            elif byte in (10, 13):
                if self.header:
                    self._end_header()
                self.row = self.field = 0
                self.column = 0
                self.header = self.after_quote = False
                self.start = True
                self.after_cr = byte == 13
            elif byte == 44:
                self.field = 0
                self.column += 1
                self.start, self.after_quote = True, False
            else:
                self.quoted = self.start and byte == 34
                self.start = self.after_quote = False


def _capture(path, destination, *, size, digest, budget):
    """The student-controls exact-size, nonblocking private-capture pattern.

    Add raw record/token limits during that same copy, plus a private inode
    identity retained until parsing finishes. No second coordinate parse.
    """
    _require(
        type(size) is int and 0 < size <= literal.MAX_MEMBER_BYTES,
        "COVERAGE_MEMBER_BYTES",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "COVERAGE_SOURCE_FILE_KIND")
        _require(before.st_size == size, "COVERAGE_SOURCE_SIZE")
        count, hashed, guard = 0, hashlib.sha256(), _CsvBounds(budget)
        with (
            os.fdopen(descriptor, "rb", buffering=0, closefd=False) as src,
            destination.open("xb") as dst,
        ):
            while chunk := src.read(min(_CHUNK, size - count + 1)):
                count += len(chunk)
                _require(count <= size, "COVERAGE_SOURCE_SIZE")
                guard.feed(chunk)
                hashed.update(chunk)
                dst.write(chunk)
            dst.flush()
            captured = _identity(os.fstat(dst.fileno()))
        _require(
            count == size and _identity(before) == _identity(os.fstat(descriptor)),
            "COVERAGE_SOURCE_CHANGED",
        )
        _require(hashed.hexdigest() == digest, "COVERAGE_SOURCE_DIGEST")
        return captured
    finally:
        os.close(descriptor)


def _parent_binding(source):
    _require(
        type(source) is money.AuthenticatedCurrentMoneySource,
        "COVERAGE_AUTHENTICATED_PARENT",
    )
    source.validate()
    return {
        "authority_type": "AuthenticatedCurrentMoneySource",
        "source_identity_sha256": _sha(source.source.identity),
        "frame_sha256": money._frame_signature(source.frame),
        "scope_sha256": _sha(source.scope.identity),
    }


def _partitions(person):
    published_to_parent, parent_to_published = {}, {}
    for year, native, parent in person[
        ["source_year", "source_household_id", "person_household_id"]
    ].itertuples(index=False, name=None):
        published = (year, native)
        _require(parent > 0, "COVERAGE_HOUSEHOLD_COORDINATE")
        _require(
            published_to_parent.setdefault(published, parent) == parent,
            "COVERAGE_HOUSEHOLD_SPLIT",
        )
        _require(
            parent_to_published.setdefault(parent, published) == published,
            "COVERAGE_HOUSEHOLD_MERGE",
        )
    return {
        "relation": "bidirectional_partition_bijection",
        "published_key": ["source_year", "source_household_id"],
        "parent_key": "person_household_id",
        "households": len(parent_to_published),
        "rows": len(person),
    }


def _utf8_size(token):
    _require(type(token) is str and len(token) <= _TOKEN_MAX, "COVERAGE_TOKEN_BYTES")
    size = sum(
        1 if ord(c) < 128 else 2 if ord(c) < 2048 else 3 if ord(c) < 65536 else 4
        for c in token
    )
    _require(size <= _TOKEN_MAX, "COVERAGE_TOKEN_BYTES")
    return size


def _body(table):
    _require(0 < len(table) <= _MAX_PERSONS, "COVERAGE_ROWS")
    result = bytearray()
    for row in table[
        [
            "person_id",
            "person_household_id",
            "source_year",
            "source_household_id",
            "A_LINENO",
            "A_AGE",
            "PERIDNUM",
            "PRPERTYP",
            "PRPERTYP_state",
        ]
    ].itertuples(index=False, name=None):
        sizes = [_utf8_size(token) for token in row[6:]]
        size = _NUMBERS.size + 3 * _LENGTH.size + sum(sizes)
        _require(size <= _ROW_MAX, "COVERAGE_ROW_BYTES")
        _require(len(result) + _LENGTH.size + size <= _BODY_MAX, "COVERAGE_BODY_BYTES")
        result.extend(_LENGTH.pack(size))
        result.extend(_NUMBERS.pack(*row[:6]))
        for token, length in zip(row[6:], sizes, strict=True):
            result.extend(_LENGTH.pack(length))
            result.extend(token.encode("utf-8"))
    return bytes(result)


def _decode_rows(body, rows):
    _require(type(body) is bytes and len(body) <= _BODY_MAX, "COVERAGE_BODY_BYTES")
    _require(type(rows) is int and 0 < rows <= _MAX_PERSONS, "COVERAGE_ROWS")
    view, offset = memoryview(body), 0
    for _ in range(rows):
        _require(offset + 4 <= len(view), "COVERAGE_ROW_BYTES")
        size = _LENGTH.unpack_from(view, offset)[0]
        offset += 4
        end = offset + size
        _require(
            _NUMBERS.size + 12 <= size <= _ROW_MAX and end <= len(view),
            "COVERAGE_ROW_BYTES",
        )
        values = list(_NUMBERS.unpack_from(view, offset))
        offset += _NUMBERS.size
        for _ in _STRING_COLUMNS:
            _require(offset + 4 <= end, "COVERAGE_TOKEN_BYTES")
            length = _LENGTH.unpack_from(view, offset)[0]
            offset += 4
            _require(
                length <= _TOKEN_MAX and offset + length <= end, "COVERAGE_TOKEN_BYTES"
            )
            values.append(bytes(view[offset : offset + length]).decode("utf-8"))
            offset += length
        _require(offset == end, "COVERAGE_ROW_BYTES")
        yield values
    _require(offset == len(view), "COVERAGE_BODY_BYTES")


def _bounded_header(header):
    # Count ASCII JSON bytes without first allocating escaped strings or a
    # serialized header. This vocabulary is owned metadata, never row tokens.
    remaining = _HEADER_MAX

    def charge(size):
        nonlocal remaining
        remaining -= size
        _require(remaining >= 0, "COVERAGE_HEADER_BYTES")

    def visit(value, depth=0):
        _require(depth <= 16, "COVERAGE_HEADER_DEPTH")
        if isinstance(value, str):
            charge(2)
            for c in value:
                charge(
                    2
                    if c in '\\"\b\f\n\r\t'
                    else 6
                    if ord(c) < 32 or 127 <= ord(c) <= 65535
                    else 12
                    if ord(c) > 65535
                    else 1
                )
        elif isinstance(value, (list, tuple, dict)):
            charge(2 + max(0, len(value) - 1))
            if isinstance(value, dict):
                for key, item in value.items():
                    _require(type(key) is str, "COVERAGE_HEADER_TYPE")
                    visit(key, depth + 1)
                    charge(1)
                    visit(item, depth + 1)
            else:
                for item in value:
                    visit(item, depth + 1)
        else:
            _require(
                value is None
                or type(value) is bool
                or (type(value) is int and -(2**63) <= value < 2**63),
                "COVERAGE_HEADER_TYPE",
            )
            charge(len(_json(value)))

    visit(header)
    return _json(header)


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class AuthenticatedAsecCoverage:
    """Issued immutable bytes, with fresh receipt and table views on demand."""

    _header: bytes
    _body: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "COVERAGE_CONSTRUCTOR_UNAVAILABLE")
        _ISSUED[self] = (_sha(self._header), _sha(self._body))

    def validate(self):
        try:
            _require(
                type(self._header) is bytes and 0 < len(self._header) <= _HEADER_MAX,
                "COVERAGE_HEADER_BYTES",
            )
            _require(
                type(self._body) is bytes and len(self._body) <= _BODY_MAX,
                "COVERAGE_BODY_BYTES",
            )
            _require(
                _ISSUED.get(self) == (_sha(self._header), _sha(self._body)),
                "COVERAGE_ISSUED_CONTENT_CHANGED",
            )
            header = json.loads(self._header)
            _require(
                _json(header["implementation"]) == _json(_implementation()),
                "COVERAGE_IMPLEMENTATION_CHANGED",
            )
            _require(0 < header["rows"] <= _MAX_PERSONS, "COVERAGE_ROWS")
        except AsecCoverageAuthenticationError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise AsecCoverageAuthenticationError("COVERAGE_CONTENT_REFUSAL") from None

    @property
    def receipt(self):
        self.validate()
        return json.loads(self._header)

    @property
    def content_sha256(self):
        self.validate()
        return _sha(self._header + self._body)

    def to_bytes(self):
        self.validate()
        payload = MAGIC + _LENGTH.pack(len(self._header)) + self._header + self._body
        return payload + hashlib.sha256(payload).digest()

    def table(self):
        self.validate()
        table = pd.DataFrame(
            _decode_rows(self._body, self.receipt["rows"]), columns=COLUMNS
        )
        for name in _STRING_COLUMNS:
            table[name] = pd.array(table[name], dtype="string")
        return table


def verify_asec_coverage_parent(coverage, source):
    """Verify the closed artifact against the exact named live receiving parent."""
    try:
        _require(
            type(coverage) is AuthenticatedAsecCoverage,
            "COVERAGE_AUTHENTICATED_ARTIFACT",
        )
        coverage.validate()
        _require(
            coverage.receipt["parent"] == _parent_binding(source),
            "COVERAGE_PARENT_CHANGED",
        )
    except AsecCoverageAuthenticationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise AsecCoverageAuthenticationError("COVERAGE_PARENT_REFUSAL") from None


def _reconstruct(source, member_paths: Mapping[int, str | Path]):
    before = _implementation()
    parent = _parent_binding(source)
    _require(
        isinstance(member_paths, Mapping)
        and set(member_paths) == {2022, 2023, 2024}
        and all(type(year) is int for year in member_paths)
        and all(isinstance(p, (str, Path)) for p in member_paths.values()),
        "COVERAGE_MEMBER_PATHS",
    )
    _require(
        tuple(p[0] for p in _MEMBER_PINS) == (2022, 2023, 2024)
        and len({p[1] for p in _MEMBER_PINS}) == 3,
        "COVERAGE_PIN_COHORTS",
    )
    person = source.frame.person
    _require(0 < len(person) <= _MAX_PERSONS, "COVERAGE_ROWS")
    _require(
        set(_INT_COLUMNS + ("PERIDNUM",)) <= set(person), "COVERAGE_PARENT_COLUMNS"
    )
    _require(
        all(str(person[name].dtype) == "int64" for name in _INT_COLUMNS),
        "COVERAGE_PARENT_DTYPE",
    )
    # Build only from the validated parent's owned scope/native columns; never
    # from model age or a caller's DataFrame/hash/receipt.
    person = person[
        [
            "person_id",
            "person_household_id",
            "source_year",
            "source_household_id",
            "A_LINENO",
            "A_AGE",
            "PERIDNUM",
        ]
    ].copy(deep=True)
    _require(
        tuple(person.person_id) == source.scope.person_ids
        and tuple(person.person_household_id) == source.scope.person_household_ids
        and tuple(person.source_year) == source.scope.person_years,
        "COVERAGE_PARENT_COORDINATES",
    )
    _require(_parent_binding(source) == parent, "COVERAGE_PARENT_CHANGED")
    roster = person[list(literal.ROSTER_COLUMNS)].copy().reset_index(drop=True)
    _require(
        tuple(roster.PERIDNUM) == source.scope.person_native_keys,
        "COVERAGE_PARENT_NATIVE_KEYS",
    )
    membership = _partitions(person)
    with tempfile.TemporaryDirectory(prefix="microcosm-asec-coverage-") as directory:
        paths, identities, sources = {}, {}, []
        budget = [_BODY_MAX]
        pins = {pin[0]: pin for pin in _MEMBER_PINS}
        for year in (2022, 2023, 2024):
            _, member, archive, digest, rows, size = pins[year]
            _require(
                type(rows) is int
                and 0 < rows <= _MAX_PERSONS
                and int((roster.source_year == year).sum()) == rows,
                "COVERAGE_COHORT_ROWS",
            )
            path = Path(directory) / member
            identities[year] = _capture(
                member_paths[year], path, size=size, digest=digest, budget=budget
            )
            paths[year] = path
            sources.append(
                {
                    "source_year": year,
                    "survey_year": year + 1,
                    "member": member,
                    "archive_sha256": archive,
                    "member_sha256": digest,
                    "member_bytes": size,
                    "rows": rows,
                }
            )
        projection, receipt = literal.read_asec_person_coverage_source(
            paths, person_roster=roster
        )
        for year, path in paths.items():
            _require(
                _identity(path.stat(follow_symlinks=False)) == identities[year],
                "COVERAGE_PRIVATE_SOURCE_CHANGED",
            )
    for name in _INT_COLUMNS[:2]:
        projection[name] = person[name].to_numpy(copy=True)
    body = _body(projection)
    _require(_parent_binding(source) == parent, "COVERAGE_PARENT_CHANGED")
    _require(
        _json(before) == _json(_implementation()), "COVERAGE_IMPLEMENTATION_CHANGED"
    )
    header = {
        "artifact_kind": ARTIFACT_KIND,
        "encoding": "length_prefixed_utf8_fixed_int64_v1",
        "columns": COLUMNS,
        "rows": len(projection),
        "body_sha256": _sha(body),
        "parent": parent,
        "sources": sources,
        "household_partitions": membership,
        "source_authenticated": True,
        "named_parent_binding_authenticated": True,
        "archive_bytes_read": False,
        "source_scope": "all_three_complete_original_person_cohorts",
        "age_relation": {
            "relation": "original_csv_A_AGE_equals_parent_A_AGE",
            "comparison": "strict_unsigned_decimal_int64_identity",
            "evidence": "authenticated_private_csv_read_in_this_reconstruction",
            "compared_rows": len(projection),
            "conflicts": 0,
            "model_age_used": False,
        },
        "coverage_status": "literal_source_fields_only",
        "period_harmonized": False,
        "cross_survey_coverage_equivalence_established": False,
        "domain_authority": False,
        "original_design_weight_semantics_established": False,
        "release_eligible": False,
        "literal_reader_receipt": receipt,
        "implementation": before,
    }
    return AuthenticatedAsecCoverage(_bounded_header(header), body, _token=_TOKEN)


def _compare_candidate(path, payload):
    # No candidate header, token, checksum or self-declared source is decoded.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "COVERAGE_CANDIDATE_FILE_KIND")
        _require(before.st_size == len(payload), "COVERAGE_CANDIDATE_BYTES")
        offset = 0
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as handle:
            while chunk := handle.read(min(_CHUNK, len(payload) - offset + 1)):
                _require(
                    chunk == memoryview(payload)[offset : offset + len(chunk)],
                    "COVERAGE_CANONICAL_BYTES",
                )
                offset += len(chunk)
        _require(
            offset == len(payload)
            and _identity(before) == _identity(os.fstat(descriptor)),
            "COVERAGE_CANDIDATE_CHANGED",
        )
    finally:
        os.close(descriptor)


def authenticate_asec_coverage(source, *, member_paths, candidate_path=None):
    """Independently reconstruct before comparing optional candidate bytes."""
    try:
        expected = _reconstruct(source, member_paths)
        if candidate_path is not None:
            _compare_candidate(candidate_path, expected.to_bytes())
        verify_asec_coverage_parent(expected, source)
        return expected
    except AsecCoverageAuthenticationError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        csv.Error,
    ):
        raise AsecCoverageAuthenticationError("COVERAGE_SOURCE_REFUSAL") from None


# The imported literal parser is code authority, not a caller-selectable
# decoder. Refuse monkeypatched parsing/status/field code even before issuance.
_LITERAL_AUTHORITY = _runtime_code(literal)
