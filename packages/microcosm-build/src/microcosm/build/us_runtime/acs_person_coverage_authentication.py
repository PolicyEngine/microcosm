"""Closed ACS coverage source issuance; native consistency grants no identity.

The existing prepared ACS container is publicly constructible. Until a closed
prepared/native authority exists, even a consistent Frame receives only an
UngrantedACSNativeBinding. Nothing here attaches columns or prepares a population.
"""

from __future__ import annotations

import _csv
import csv
import hashlib
import io
import json
import platform
import re
import stat
import sys
import sysconfig
import zipfile
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import BuiltinFunctionType

import numpy as np
import pandas as pd

from microcosm.frame import Frame

from . import acs_housing_universe_source as custody
from . import acs_person_coverage_columns as literal
from .acs_pums import AcsPumsSource
from .source_csv_builtin import csv_reader_bound

PROTOCOL = "microcosm.acs-person-coverage-authentication.v1"
MAX_HEADER_BYTES = 1024**2
MAX_BODY_BYTES = 64 * 1024**2
MAX_RECORD_BYTES = 400_000
MAX_CSV_HEADER_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 64 * 1024
_MAGIC = b"ACS-COVERAGE/1\n"
_TOKEN = object()
_CSV_READER = _csv.reader
_COLUMNS = (*literal.READ_COLUMNS, "MIL_state", "ESR_state")
_IMPLEMENTATION_FILES = (
    "acs_person_coverage_authentication.py",
    "acs_person_coverage_columns.py",
    "acs_housing_universe_source.py",
    "acs_sources.py",
    "acs_pums.py",
    "source_csv_builtin.py",
    "acs_2024_1yr_sources.json",
)


class ACSCoverageAuthenticationError(ValueError):
    """Static refusal code, without source paths, keys, tokens or cause chains."""


def _require(condition, code):
    if not condition:
        raise ACSCoverageAuthenticationError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value, cap=MAX_HEADER_BYTES):
    # ASCII JSON escapes preserve CR/LF/HT and distinguish null from "". Values
    # originate in byte-bounded records or the fixed, bounded header inventory.
    # Count exact escaped bytes before an encoder can allocate a whole token.
    count = 0

    def charge(size):
        nonlocal count
        count += size
        _require(count <= cap, "CANONICAL_SIZE")

    def visit(item):
        if isinstance(item, str):
            charge(2)
            for char in item:
                code = ord(char)
                charge(
                    2
                    if char in '\\"\b\f\n\r\t'
                    else 1
                    if 32 <= code <= 126
                    else 6
                    if code <= 0xFFFF
                    else 12
                )
        elif item is None:
            charge(4)
        elif type(item) is bool:
            charge(4 if item else 5)
        elif type(item) is int:
            charge(len(str(item)))
        elif isinstance(item, (list, tuple)):
            charge(2 + max(0, len(item) - 1))
            for part in item:
                visit(part)
        elif type(item) is dict:
            charge(2 + max(0, len(item) - 1) + len(item))
            for key, part in item.items():
                _require(type(key) is str, "CANONICAL_KEY")
                visit(key)
                visit(part)
        else:
            _require(False, "CANONICAL_TYPE")

    visit(value)
    parts, count = [], 0
    for part in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).iterencode(value):
        count += len(part)
        _require(count <= cap, "CANONICAL_SIZE")
        parts.append(part.encode("ascii"))
    return b"".join(parts)


def _producer():
    # Provider bytes alone cannot attest the callable used by either parsing
    # pass. Preserve the real C binding across issuance/verification, including
    # refusal when both public aliases were replaced before this owner imported.
    _require(
        type(_CSV_READER) is BuiltinFunctionType
        and _CSV_READER.__module__ == "_csv"
        and _CSV_READER.__name__ == "reader"
        and _CSV_READER.__self__ is _csv
        and csv.reader is _csv.reader is _CSV_READER
        and literal.csv is custody.csv is csv
        and csv_reader_bound(csv),
        "CSV_READER_CHANGED",
    )
    parser_binary = getattr(_csv, "__file__", None)
    if parser_binary is None:
        # This runtime builds _csv into libpython. Other builds ship an
        # extension or link the parser into the executable itself.
        parser_binary = (
            Path(sysconfig.get_config_var("LIBDIR"))
            / sysconfig.get_config_var("LDLIBRARY")
            if sysconfig.get_config_var("Py_ENABLE_SHARED")
            else Path(sys.executable)
        )
    return {
        "protocol": PROTOCOL,
        "files": {
            name: _sha(Path(__file__).with_name(name).read_bytes())
            for name in _IMPLEMENTATION_FILES
        },
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "csv_binary_sha256": _sha(Path(parser_binary).read_bytes()),
        "stdlib": {
            name: _sha(Path(module.__file__).read_bytes())
            for name, module in (
                ("csv", csv),
                ("zipfile", zipfile),
                ("json", json),
            )
        },
        "parser": {
            "reader": "_csv.reader:verified_builtin_identity",
            "encoding": "utf-8-sig",
            "newline": "",
            "csv_field_size_limit": csv.field_size_limit(),
            "literal_record_chars": literal.MAX_CSV_RECORD_CHARS,
            "source_rows": literal.MAX_ROWS,
            "selected_rows": literal.MAX_SELECTED_ROWS,
        },
        "ceilings": {
            "header": MAX_HEADER_BYTES,
            "body": MAX_BODY_BYTES,
            "record": MAX_RECORD_BYTES,
            "csv_header": MAX_CSV_HEADER_BYTES,
            "token": MAX_TOKEN_BYTES,
            "member": custody._MEMBER_MAX,
            "expanded_archive": custody._EXPANDED_MAX,
            "member_count": custody._MEMBER_COUNT_MAX,
        },
    }


def _records(stream):
    """Fence raw logical records/tokens before UTF-8 decoding or CSV allocation.

    Quote parity only locates boundaries; the unchanged strict literal parser
    subsequently owns CSV validity. CR, LF and CRLF are preserved, including
    inside quotes. The conservative token ceiling includes raw CSV quoting.
    """
    record = bytearray()
    quoted, pending_cr, first, token_bytes = False, False, True, 0
    cap = MAX_CSV_HEADER_BYTES
    while block := stream.read(4096):
        for byte in block:
            if pending_cr:
                if byte == 10:
                    _require(len(record) < cap, "CSV_RECORD_BYTES")
                    record.append(byte)
                yield bytes(record)
                record.clear()
                first, pending_cr, token_bytes = False, False, 0
                if byte == 10:
                    continue
            cap = MAX_CSV_HEADER_BYTES if first else MAX_RECORD_BYTES
            _require(len(record) < cap, "CSV_RECORD_BYTES")
            if byte == 44 and not quoted:
                token_bytes = 0
            else:
                token_bytes += 1
                _require(token_bytes <= MAX_TOKEN_BYTES, "CSV_TOKEN_BYTES")
            record.append(byte)
            if byte == 34:
                quoted = not quoted
            if not quoted and byte in (10, 13):
                if byte == 13:
                    pending_cr = True
                else:
                    yield bytes(record)
                    record.clear()
                    first, token_bytes = False, 0
    if record:
        yield bytes(record)


def _decode_record(raw, *, first):
    # BOM is only consumed at member start, exactly as in the literal reader.
    with io.TextIOWrapper(
        io.BytesIO(raw), encoding="utf-8-sig" if first else "utf-8", newline=""
    ) as text:
        reader = literal._literal_csv_records(text)
        result = next(reader, None)
        _require(result is not None and next(reader, None) is None, "CSV_RECORD")
        return result


def _members(archive, role):
    # These are the closed housing owner's complete central-directory checks,
    # kept local because _archive also invokes its physical-record parser.
    # Editing/extracting that owner would change historical custody producers.
    prefix = "psam_hus" if role == "household" else "psam_pus"
    members = archive.infolist()
    _require(0 < len(members) <= custody._MEMBER_COUNT_MAX, "ZIP_MEMBER_COUNT")
    names = [m.filename for m in members]
    _require(len({n.casefold() for n in names}) == len(names), "ZIP_DUPLICATE_MEMBER")
    expanded = 0
    for member in members:
        name, mode = member.filename, member.external_attr >> 16
        _require(
            name not in ("", ".", "..")
            and not any(c in name for c in ("/", "\\", "\x00")),
            "ZIP_MEMBER_PATH",
        )
        _require(
            not member.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG),
            "ZIP_MEMBER_TYPE",
        )
        _require(
            not member.flag_bits & 1
            and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
            "ZIP_MEMBER_ENCODING",
        )
        _require(0 <= member.file_size <= custody._MEMBER_MAX, "ZIP_MEMBER_SIZE")
        expanded += member.file_size
        _require(expanded <= custody._EXPANDED_MAX, "ZIP_EXPANDED_SIZE")
        _require(
            not name.casefold().startswith(prefix) or name.casefold().endswith(".csv"),
            "ZIP_PREFIX_NONCSV",
        )
    _require(any(n.casefold().startswith(prefix) for n in names), "ZIP_ROLE_MISSING")
    return sorted(members, key=lambda m: m.filename), prefix


def _inventory(path, role, serialnos):
    inventory, selected, rows, selected_budget = [], {}, 0, 0
    required = ("SERIALNO", "NP") if role == "household" else literal.READ_COLUMNS
    with zipfile.ZipFile(path) as archive:
        members, prefix = _members(archive, role)
        for member in members:
            applicable = member.filename.casefold().startswith(prefix)
            digest, count, row_count, header, header_sha = (
                hashlib.sha256(),
                0,
                0,
                None,
                None,
            )
            with archive.open(member) as stream:
                if not applicable:
                    while block := stream.read(min(4096, member.file_size - count + 1)):
                        count += len(block)
                        _require(count <= member.file_size, "ZIP_MEMBER_GREW")
                        digest.update(block)
                else:
                    for raw in _records(stream):
                        count += len(raw)
                        _require(count <= member.file_size, "ZIP_MEMBER_GREW")
                        digest.update(raw)
                        values = _decode_record(raw, first=header is None)
                        if header is None:
                            header, header_sha = values, _sha(raw)
                            _require(
                                all(header)
                                and len(set(header)) == len(header)
                                and set(required) <= set(header),
                                "CSV_HEADER",
                            )
                            positions = [header.index(c) for c in required]
                            continue
                        _require(len(values) == len(header), "CSV_ROW_WIDTH")
                        rows += 1
                        row_count += 1
                        _require(rows <= literal.MAX_ROWS, "SOURCE_ROWS")
                        cells = [values[p] for p in positions]
                        if cells[0] not in serialnos:
                            continue
                        # Upper bound on escaped row + statuses + lineage before
                        # the low-level reader allocates its selected DataFrame.
                        selected_budget += 6 * len(raw) + 1024
                        _require(
                            selected_budget <= MAX_BODY_BYTES, "SELECTED_BODY_BUDGET"
                        )
                        if role == "household":
                            key = cells[0]
                            _require(re.fullmatch(r"[0-9]{1,2}", cells[1]), "SOURCE_NP")
                            value = int(cells[1])
                        else:
                            _require(
                                re.fullmatch(r"[0-9]{1,2}", cells[1]), "SOURCE_SPORDER"
                            )
                            key = (cells[0], int(cells[1]))
                            _require(1 <= key[1] <= 20, "SOURCE_SPORDER")
                            value = [member.filename, row_count]
                        _require(key not in selected, "SOURCE_DUPLICATE_KEY")
                        _require(
                            len(selected) < literal.MAX_SELECTED_ROWS, "SELECTED_ROWS"
                        )
                        selected[key] = value
            _require(count == member.file_size, "ZIP_MEMBER_SIZE_MISMATCH")
            _require(not applicable or header is not None, "CSV_HEADER_MISSING")
            inventory.append(
                {
                    "name": member.filename,
                    "bytes": count,
                    "sha256": digest.hexdigest(),
                    "compressed_bytes": member.compress_size,
                    "crc32": member.CRC,
                    "applicable": applicable,
                    "header_sha256": header_sha,
                    "rows": row_count,
                }
            )
    return inventory, selected


def _native_roster(frame):
    _require(type(frame) is Frame, "NATIVE_FRAME_REQUIRED")
    p, h = frame.person, frame.table("household")
    _require(0 < len(h) <= len(p) <= literal.MAX_SELECTED_ROWS, "NATIVE_ROWS")
    _require(p.columns.is_unique and h.columns.is_unique, "NATIVE_COLUMNS")
    _require(
        {
            "person_id",
            "person_household_id",
            "SPORDER",
            "A_LINENO",
            "source_year",
            "source_household_id",
            "source_person_id",
            "source_row_id",
            "AGEP",
            "A_AGE",
            "age",
        }
        <= set(p)
        and {"household_id", "SERIALNO", "NP"} <= set(h),
        "NATIVE_COLUMNS",
    )
    for table, names in (
        (
            p,
            (
                "person_id",
                "person_household_id",
                "SPORDER",
                "A_LINENO",
                "source_year",
                "source_household_id",
                "source_row_id",
            ),
        ),
        (h, ("household_id", "NP")),
    ):
        for name in names:
            _require(
                pd.api.types.is_integer_dtype(table[name].dtype)
                and not table[name].isna().any(),
                "NATIVE_INTEGER",
            )
    _require(
        not p.person_id.duplicated().any()
        and not h.household_id.duplicated().any()
        and not p.source_row_id.duplicated().any()
        and bool((p.source_row_id >= 0).all()),
        "NATIVE_IDS",
    )
    _require(
        not h.SERIALNO.isna().any() and not h.SERIALNO.duplicated().any(),
        "NATIVE_SERIALNO",
    )
    _require(set(p.person_household_id) == set(h.household_id), "NATIVE_MEMBERSHIP")
    household = h.set_index("household_id")
    serials = p.person_household_id.map(household.SERIALNO)
    _require(bool((p.source_year == 2024).all()), "NATIVE_VINTAGE")
    _require(
        np.array_equal(p.SPORDER, p.A_LINENO)
        and np.array_equal(p.person_household_id, p.source_household_id)
        and list(p.source_person_id) == [str(v) for v in p.SPORDER],
        "NATIVE_ALIASES",
    )
    keys = literal._keys(
        pd.DataFrame({"SERIALNO": serials, "SPORDER": p.SPORDER})
    ).reset_index(drop=True)
    counts = p.groupby("person_household_id", sort=False).size()
    _require(np.array_equal(counts.loc[h.household_id], h.NP), "NATIVE_HOUSEHOLD_COUNT")
    roster = (
        [int(pid), int(hid), serial, int(order), int(row)]
        for pid, hid, serial, order, row in zip(
            p.person_id,
            p.person_household_id,
            keys.SERIALNO,
            keys.SPORDER,
            p.source_row_id,
            strict=True,
        )
    )
    digest = hashlib.sha256()
    for row in roster:
        digest.update(_json(row, MAX_RECORD_BYTES - 1) + b"\n")
    return keys, digest.hexdigest()


def _age_relation(frame, table):
    matched, unresolved, mismatch = 0, 0, 0
    for raw, ag, aa, age in zip(
        table.AGEP, frame.person.AGEP, frame.person.A_AGE, frame.person.age, strict=True
    ):
        if re.fullmatch(r"[0-9]{1,2}", raw) is None:
            unresolved += 1
        elif all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
            and np.isfinite(v)
            and v == int(raw)
            for v in (ag, aa, age)
        ):
            matched += 1
        else:
            mismatch += 1
    return {
        "matching_rows": matched,
        "unresolved_literal_rows": unresolved,
        "mismatching_rows": mismatch,
        "relation": "numeric_identity"
        if not unresolved and not mismatch
        else "unproven",
        "preparation_provenance_authenticated": False,
    }


@dataclass(frozen=True, slots=True)
class UngrantedACSNativeBinding:
    """Owned consistency evidence, explicitly lacking prepared population proof."""

    receipt_json: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "BINDING_CONSTRUCTOR")

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


@dataclass(frozen=True, slots=True)
class AuthenticatedACSPersonCoverage:
    """Immutable source envelope; candidate bytes cannot construct this type."""

    payload: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "SOURCE_CONSTRUCTOR")

    def _parts(self):
        start = len(_MAGIC) + 4
        size = int.from_bytes(self.payload[len(_MAGIC) : start], "big")
        return self.payload[start : start + size], self.payload[start + size :]

    @property
    def receipt(self):
        return json.loads(self._parts()[0])

    @property
    def table(self):
        rows = [json.loads(row) for row in self._parts()[1].splitlines()]
        result = pd.DataFrame([r[: len(_COLUMNS)] for r in rows], columns=_COLUMNS)
        for column in _COLUMNS:
            result[column] = result[column].astype(
                "int64" if column == "SPORDER" else "string"
            )
        return result

    @property
    def native_binding(self):
        return UngrantedACSNativeBinding(
            _json(
                {
                    **self.receipt["native_consistency"],
                    "source_payload_sha256": _sha(self.payload),
                    "population_binding_authenticated": False,
                    "missing_authority": "closed_prepared_acs_native_population",
                },
                MAX_HEADER_BYTES,
            ),
            _token=_TOKEN,
        )


def verify_acs_coverage_native_consistency(coverage, frame):
    """Check the exact live parent; never upgrade consistency to population proof."""
    try:
        _require(type(coverage) is AuthenticatedACSPersonCoverage, "SOURCE_TYPE")
        _require(coverage.receipt["producer"] == _producer(), "PRODUCER_CHANGED")
        _keys, roster = _native_roster(frame)
        evidence = coverage.native_binding.receipt
        _require(
            roster == evidence["native_roster_sha256"]
            and custody.frame_content_sha256(frame) == evidence["frame_sha256"],
            "NATIVE_FRAME_CHANGED",
        )
        _require(
            _age_relation(frame, coverage.table) == evidence["original_age"],
            "NATIVE_AGE_CHANGED",
        )
        return coverage.native_binding
    except ACSCoverageAuthenticationError:
        raise
    except Exception:
        raise ACSCoverageAuthenticationError("NATIVE_CONSISTENCY_REFUSED") from None


def load_authenticated_acs_person_coverage(
    source_dir, *, snapshot_root, frame, candidate_path=None
):
    """Reconstruct from closed default archive pins, then compare optional bytes.

    ``frame`` supplies a checked selection, never source or population authority.
    No manifest/pin/preparation-receipt argument and no publishing API exists.
    Successful and failed private captures retain the housing owner's policy.
    """
    try:
        _require(type(frame) is Frame, "NATIVE_FRAME_REQUIRED")
        before = custody.frame_content_sha256(frame)
        keys, roster_sha = _native_roster(frame)
        producer = _producer()
        with custody._capture(source_dir, snapshot_root) as (private, paths, pins):
            _require(
                {r for r, _n, _d, _s in pins} == {"household", "person"}
                and len(pins) == 2,
                "SOURCE_ROLES",
            )
            members = {}
            members["household"], households = _inventory(
                paths["household"], "household", set(keys.SERIALNO)
            )
            members["person"], lineage = _inventory(
                paths["person"], "person", set(keys.SERIALNO)
            )
            _require(
                households == keys.groupby("SERIALNO").size().to_dict(),
                "SOURCE_HOUSEHOLD_ROSTER",
            )
            _require(
                set(lineage) == set(zip(keys.SERIALNO, keys.SPORDER, strict=True)),
                "SOURCE_PERSON_ROSTER",
            )
            table, original_receipt = literal.read_acs_person_coverage_columns(
                AcsPumsSource(paths["household"], paths["person"], vintage=2024),
                person_keys=keys,
                chunksize=min(1000, len(keys)),
            )
            body, size = [], 0
            for row in table.itertuples(index=False, name=None):
                raw = (
                    _json([*row, *lineage[(row[0], row[1])]], MAX_RECORD_BYTES - 1)
                    + b"\n"
                )
                size += len(raw)
                _require(size <= MAX_BODY_BYTES, "BODY_SIZE")
                body.append(raw)
            body = b"".join(body)
            header = {
                "protocol": PROTOCOL,
                "encoding": "ascii-escaped-json-header-and-ndjson-v1",
                "producer": producer,
                "source_authenticated": True,
                "vintage": 2024,
                "population_binding_authenticated": False,
                "coverage_status": "literal_source_fields_only",
                "cross_survey_coverage_equivalence_established": False,
                "domain_assignment_authenticated": False,
                "period_harmonized": False,
                "release_eligible": False,
                "archives": [
                    {"role": r, "filename": n, "sha256": d, "bytes": s}
                    for r, n, d, s in pins
                ],
                "members": members,
                "field_contract": literal.coverage_field_contract(),
                "original_literal_receipt": original_receipt,
                "columns": [*_COLUMNS, "source_member", "source_row_ordinal"],
                "body_bytes": len(body),
                "body_sha256": _sha(body),
                "rows": len(table),
                "native_consistency": {
                    "frame_sha256": before,
                    "native_roster_sha256": roster_sha,
                    "complete_households_checked": True,
                    "source_aliases_checked": True,
                    "original_age": _age_relation(frame, table),
                },
            }
            raw_header = _json(header, MAX_HEADER_BYTES)
            payload = _MAGIC + len(raw_header).to_bytes(4, "big") + raw_header + body
            # No candidate header, digest, DataFrame or claimed producer is parsed.
            if candidate_path is not None:
                target = private / "candidate.coverage"
                actual = custody._copy(
                    Path(candidate_path), target, len(payload), exact_size=len(payload)
                )
                _require(
                    actual == _sha(payload)
                    and custody._persisted_sha(target, len(payload)) == actual,
                    "CANDIDATE_MISMATCH",
                )
            for role, _name, digest, size in pins:
                _require(
                    custody._persisted_sha(paths[role], size) == digest,
                    "SNAPSHOT_CHANGED",
                )
            _require(_producer() == producer, "PRODUCER_CHANGED")
            _require(_native_roster(frame)[1] == roster_sha, "NATIVE_FRAME_CHANGED")
            _require(
                custody.frame_content_sha256(frame) == before, "NATIVE_FRAME_CHANGED"
            )
        # _capture's authority-stability check must complete before issuance.
        return AuthenticatedACSPersonCoverage(payload, _token=_TOKEN)
    except ACSCoverageAuthenticationError:
        raise
    except Exception:
        raise ACSCoverageAuthenticationError("SOURCE_RECONSTRUCTION_REFUSED") from None
