"""Closed ACS source observations, with a same-snapshot native Frame bridge.

Source projections retain vacancies, both GQ classes and unknown occupied tenure.
Their graph transport is separate; source authentication requires reconstruction
from the two fixed Census archives. No source or model download occurs here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager, suppress
from dataclasses import InitVar, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token
from microcosm.graph.store import ContentStore

from .acs_housing_universe import classify_acs_housing_universe
from .acs_inputs import map_acs_native_inputs
from .acs_pums import AcsPumsSource, _validate_person_counts, build_acs_pums_unit_frame
from .acs_sources import load_acs_source_manifest
from .graph_implementation import implementation_hash
from .operator_boundary import assert_operator_free_source_frame
from .source_csv_builtin import capture_csv_reader

ACS_HU_STAGE = "acs_housing_universe_2024"
ACS_HU_CODEC = "us-acs-housing-universe-2024-v1"
ACS_HU_SOURCE_MAX_BYTES = 8 * 1024**3
ACS_HU_RECEIPT_MAX_BYTES = 1024**2
_CHUNK = 1024**2
_MEMBER_MAX = 8 * 1024**3
_EXPANDED_MAX = 16 * 1024**3
_MEMBER_COUNT_MAX = 64
_RECORD_MAX = 1024**2
_FIELD_MAX = 64 * 1024
_STATES = frozenset(
    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 "
    "26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 "
    "49 50 51 53 54 55 56".split()
)
_H_FIELDS = ("SERIALNO", "TYPEHUGQ", "NP", "TEN", "WGTP", "PUMA")
_P_FIELDS = ("SERIALNO", "SPORDER", "PWGTP")
_LINEAGE = ("source_member", "source_row_ordinal")
# No source/resource I/O merely to register the callable beside legacy codecs.
# Invented tests replace this private slot; public callers have no pin argument.
_ARCHIVE_PINS = None
_TOKEN = object()


class ACSHousingSourceError(ValueError):
    """Sanitized source refusal; never print source keys or cells."""


def _require(condition, code):
    if not condition:
        raise ACSHousingSourceError(code)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value) -> bytes:
    return canonical_json(value)


def _parse_json(payload: bytes, cap: int) -> dict:
    _require(type(payload) is bytes and len(payload) <= cap, "JSON_SIZE")
    try:
        value = json.loads(payload)
        _require(type(value) is dict and _json(value) == payload, "CANONICAL_JSON")
    except RecursionError:
        raise ACSHousingSourceError("JSON_DEPTH") from None
    return value


def _definition() -> tuple[dict, str]:
    raw = Path(__file__).with_name("acs_2024_housing_universe.json").read_bytes()
    return json.loads(raw), _sha(raw)


def _implementation() -> str:
    _require(capture_csv_reader(csv) is not None, "SOURCE_CSV_READER_CHANGED")
    return implementation_hash(ACS_HU_STAGE)


def _pins():
    if _ARCHIVE_PINS is not None:
        return _ARCHIVE_PINS
    return tuple(
        (a.role, a.filename, a.sha256, a.size_bytes)
        for a in load_acs_source_manifest().artifacts
    )


def _identity(value: os.stat_result):
    return tuple(
        getattr(value, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _path_components(path: Path):
    path = path.absolute()
    for parent in reversed((path, *path.parents)):
        value = parent.lstat()
        _require(not stat.S_ISLNK(value.st_mode), "SYMLINK_PATH")
    return Path(os.path.abspath(path))


def _directory(path: Path):
    path = _path_components(path)
    _require(stat.S_ISDIR(path.lstat().st_mode), "DIRECTORY_REQUIRED")
    return path


def _copy(source: Path, destination: Path, cap: int, *, exact_size=None):
    source = _path_components(source)
    fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode), "REGULAR_FILE_REQUIRED")
        _require(before.st_size <= cap, "FILE_TOO_LARGE")
        if exact_size is not None:
            _require(before.st_size == exact_size, "SOURCE_SIZE")
        digest, count = hashlib.sha256(), 0
        with destination.open("xb") as output:
            while chunk := stream.read(min(_CHUNK, cap - count + 1)):
                count += len(chunk)
                _require(count <= cap, "FILE_GREW")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _require(
            count == before.st_size
            and _identity(before) == _identity(os.fstat(stream.fileno()))
            and _identity(before) == _identity(source.lstat()),
            "FILE_CHANGED",
        )
        _path_components(source)
    destination.chmod(0o400)
    return digest.hexdigest()


def _persisted_sha(path, size):
    digest, count = hashlib.sha256(), 0
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(
            stat.S_ISREG(before.st_mode) and before.st_size == size, "SNAPSHOT_SIZE"
        )
        while chunk := stream.read(min(_CHUNK, size - count + 1)):
            count += len(chunk)
            _require(count <= size, "SNAPSHOT_GREW")
            digest.update(chunk)
        _require(
            count == size and _identity(before) == _identity(os.fstat(stream.fileno())),
            "SNAPSHOT_CHANGED",
        )
    return digest.hexdigest()


def _write(path: Path, data: bytes, cap: int):
    _require(len(data) <= cap, "OUTPUT_SIZE")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    _require(_persisted_sha(path, len(data)) == _sha(data), "PERSISTED_OUTPUT")


@contextmanager
def _capture(source_dir, snapshot_root):
    root = _directory(Path(snapshot_root))
    source = _directory(Path(source_dir))
    _require(not root.is_relative_to(source), "SNAPSHOT_SOURCE_OVERLAP")
    pins = _pins()
    expected = {p[1] for p in pins}
    _require(expected == {"csv_hus.zip", "csv_pus.zip"}, "SOURCE_ROSTER")
    _require({p.name for p in source.iterdir()} == expected, "SOURCE_ROSTER")
    # Full private projection and selected publication/candidate capture can
    # coexist. Existing captures already reduce free space and are not deducted
    # a second time. This bound is operational, not a source-size prediction.
    required = (
        sum(p[3] for p in pins)
        + 2 * ACS_HU_SOURCE_MAX_BYTES
        + 2 * ACS_HU_RECEIPT_MAX_BYTES
        + 1024**3
    )
    _require(shutil.disk_usage(root).free >= required, "INSUFFICIENT_DISK")
    private = Path(tempfile.mkdtemp(prefix="acs-hu-capture-", dir=root))
    paths = {}
    try:
        for role, name, digest, size in pins:
            target = private / name
            actual = _copy(source / name, target, size, exact_size=size)
            _require(actual == digest, "SOURCE_SHA256")
            _require(_persisted_sha(target, size) == digest, "PERSISTED_SOURCE_SHA256")
            paths[role] = target
        yield private, paths, pins
        _require(pins == _pins(), "SOURCE_AUTHORITY_CHANGED")
    except Exception as error:
        # Captures belong exclusively to this invocation. Keep failed evidence;
        # no rename can replace an earlier artifact or caller input.
        code = (
            str(error)
            if isinstance(error, ACSHousingSourceError)
            else "SOURCE_PREPARATION_REFUSED"
        )
        # Recording is best effort and strictly subordinate: if the record
        # cannot be written, the refusal it was recording still propagates.
        with suppress(Exception):
            failure = _json({"status": "failed", "reason": code})
            if not (private / "failure.json").exists():
                _write(private / "failure.json", failure, ACS_HU_RECEIPT_MAX_BYTES)
        raise


def _csv_record(raw: bytes) -> list[str]:
    csv_reader = capture_csv_reader(csv)
    _require(csv_reader is not None, "SOURCE_CSV_READER_CHANGED")
    _require(0 < len(raw) <= _RECORD_MAX, "CSV_RECORD_SIZE")
    _require(not raw.startswith(b"\xef\xbb\xbf"), "CSV_BOM")
    if raw.endswith(b"\n"):
        raw = raw[:-1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
    _require(
        bool(raw) and b"\x00" not in raw and b"\r" not in raw and b"\n" not in raw,
        "CSV_PHYSICAL_RECORD",
    )
    values = next(
        csv_reader(
            [raw.decode("utf-8", errors="strict")],
            strict=True,
            quoting=csv.QUOTE_MINIMAL,
        )
    )
    _require(
        all(len(v.encode("utf-8")) <= _FIELD_MAX for v in values), "CSV_FIELD_SIZE"
    )
    return values


def _archive(path: Path, role: str):
    """Validate the complete central directory before resolving any member name."""
    prefix = "psam_hus" if role == "household" else "psam_pus"
    required = _H_FIELDS if role == "household" else _P_FIELDS
    inventory, rows, role_header, fields, total_bytes = [], [], None, None, 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        _require(0 < len(members) <= _MEMBER_COUNT_MAX, "ZIP_MEMBER_COUNT")
        names = [m.filename for m in members]
        _require(
            len({n.casefold() for n in names}) == len(names), "ZIP_DUPLICATE_MEMBER"
        )
        for member in members:
            name = member.filename
            mode = member.external_attr >> 16
            _require(
                name not in ("", ".", "..")
                and "/" not in name
                and "\\" not in name
                and "\x00" not in name,
                "ZIP_MEMBER_PATH",
            )
            _require(
                not member.is_dir() and (stat.S_IFMT(mode) in (0, stat.S_IFREG)),
                "ZIP_MEMBER_TYPE",
            )
            _require(
                not member.flag_bits & 1
                and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "ZIP_MEMBER_ENCODING",
            )
            _require(0 <= member.file_size <= _MEMBER_MAX, "ZIP_MEMBER_SIZE")
            total_bytes += member.file_size
            _require(total_bytes <= _EXPANDED_MAX, "ZIP_EXPANDED_SIZE")
            _require(
                not name.casefold().startswith(prefix)
                or name.casefold().endswith(".csv"),
                "ZIP_PREFIX_NONCSV",
            )
        selected_bytes = 0
        for member in sorted(members, key=lambda m: m.filename):
            applicable = member.filename.casefold().startswith(prefix)
            digest, count, header_raw, header, record_count = (
                hashlib.sha256(),
                0,
                b"",
                None,
                0,
            )
            with archive.open(member) as stream:
                while True:
                    raw = (
                        stream.readline(
                            min(_RECORD_MAX + 1, member.file_size - count + 1)
                        )
                        if applicable
                        else stream.read(min(_CHUNK, member.file_size - count + 1))
                    )
                    if not raw:
                        break
                    count += len(raw)
                    _require(count <= member.file_size, "ZIP_MEMBER_GREW")
                    digest.update(raw)
                    if not applicable:
                        continue
                    tokens = _csv_record(raw)
                    if header is None:
                        header, header_raw = tokens, raw
                        _require(
                            all(header) and len(set(header)) == len(header),
                            "CSV_HEADER",
                        )
                        _require(set(required) <= set(header), "CSV_REQUIRED_FIELDS")
                        if role == "household":
                            _require(
                                bool({"ST", "STATE"} & set(header)),
                                "CSV_STATE_REQUIRED",
                            )
                        if role_header is None:
                            role_header = header
                            fields = (
                                (
                                    *required,
                                    *(n for n in ("ST", "STATE") if n in header),
                                )
                                if role == "household"
                                else required
                            )
                        _require(header == role_header, "CSV_ROLE_HEADER_MISMATCH")
                        positions = [header.index(f) for f in fields]
                        continue
                    _require(len(tokens) == len(header), "CSV_ROW_WIDTH")
                    record_count += 1
                    row = [
                        *(tokens[p] for p in positions),
                        member.filename,
                        record_count,
                    ]
                    selected_bytes += len(_json(row))
                    _require(
                        selected_bytes <= ACS_HU_SOURCE_MAX_BYTES,
                        "PROJECTION_TOO_LARGE",
                    )
                    rows.append(row)
            _require(count == member.file_size, "ZIP_MEMBER_SIZE_MISMATCH")
            _require(not applicable or header is not None, "CSV_HEADER_MISSING")
            inventory.append(
                {
                    "name": member.filename,
                    "compressed_bytes": member.compress_size,
                    "bytes": count,
                    "crc32": member.CRC,
                    "sha256": digest.hexdigest(),
                    "is_data": applicable,
                    "header_hex": header_raw.hex(),
                    "header_sha256": _sha(header_raw),
                    "header": header,
                    "rows": record_count,
                }
            )
    _require(role_header is not None, "ZIP_ROLE_MISSING")
    return list((*fields, *_LINEAGE)), rows, inventory


def _integer(token, low, high, label):
    _require(
        type(token) is str
        and re.fullmatch(r"[0-9]+", token, flags=re.ASCII) is not None
        and len(token) <= 5,
        label,
    )
    value = int(token)
    _require(low <= value <= high, label)
    return value


def _tables(projection):
    _require(
        set(projection)
        == {"format", "household_columns", "person_columns", "households", "persons"}
        and projection["format"] == "acs-housing-lexical-projection/1",
        "PROJECTION_SCHEMA",
    )
    hcols, pcols = projection["household_columns"], projection["person_columns"]
    _require(
        hcols
        in [
            list((*_H_FIELDS, *state, *_LINEAGE))
            for state in (("ST",), ("STATE",), ("ST", "STATE"))
        ]
        and pcols == list((*_P_FIELDS, *_LINEAGE)),
        "PROJECTION_COLUMNS",
    )
    for cols, name in ((hcols, "households"), (pcols, "persons")):
        _require(type(projection[name]) is list, "PROJECTION_ROWS")
        for row in projection[name]:
            _require(
                type(row) is list
                and len(row) == len(cols)
                and all(type(v) is str for v in row[:-1])
                and type(row[-1]) is int
                and row[-1] > 0,
                "PROJECTION_ROW",
            )
    household = pd.DataFrame(projection["households"], columns=hcols)
    person = pd.DataFrame(projection["persons"], columns=pcols)
    _require(not household.empty, "EMPTY_HOUSEHOLD_SOURCE")
    for table in (household, person):
        _require(
            all(
                re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", value, flags=re.ASCII)
                and int(value[-7:]) > 0
                for value in table.SERIALNO
            ),
            "SERIALNO",
        )
    _require(not household.SERIALNO.duplicated().any(), "DUPLICATE_HOUSEHOLD")
    _require(set(person.SERIALNO) <= set(household.SERIALNO), "ORPHAN_PERSON")
    typed = pd.DataFrame(index=household.index)
    for name, low, high in (("TYPEHUGQ", 1, 3), ("NP", 0, 20), ("WGTP", 0, 9999)):
        typed[name] = np.asarray(
            [_integer(v, low, high, name) for v in household[name]], dtype="int64"
        )
    valid = household.TEN.to_numpy() != ""
    typed["TEN"] = np.asarray(
        [
            _integer(v, 1, 4, "TEN") if present else 0
            for v, present in zip(household.TEN, valid, strict=True)
        ],
        dtype="int64",
    )
    lines = np.asarray(
        [
            _integer(v, 1, 20, "SPORDER")
            if len(v) <= 2
            else _integer("", 1, 20, "SPORDER")
            for v in person.SPORDER
        ],
        dtype="int64",
    )
    pw = np.asarray(
        [_integer(v, 1, 9999, "PWGTP") for v in person.PWGTP], dtype="int64"
    )
    _require(
        not pd.MultiIndex.from_arrays([person.SERIALNO, lines]).duplicated().any(),
        "DUPLICATE_PERSON",
    )
    state = "ST" if "ST" in household else "STATE"
    _require(set(household[state]) <= _STATES, "STATE_SCOPE")
    if "ST" in household and "STATE" in household:
        _require(household.ST.equals(household.STATE), "STATE_CONFLICT")
    for v in household.PUMA:
        _require(len(v) == 5, "PUMA")
        _integer(v, 100, 81003, "PUMA")
    gq = typed.TYPEHUGQ.to_numpy() != 1
    vacant = ~gq & (typed.NP.to_numpy() == 0)
    _require(
        np.array_equal(household.SERIALNO.str.startswith("2024GQ").to_numpy(), gq),
        "SERIALNO_TYPE",
    )
    _require(bool((typed.NP.to_numpy()[gq] == 1).all()), "GQ_NP")
    _require(
        bool((typed.WGTP.to_numpy()[gq] == 0).all())
        and bool((typed.WGTP.to_numpy()[~gq] > 0).all()),
        "WGTP_SCOPE",
    )
    _require(not bool(valid[gq | vacant].any()), "TEN_NIU_SCOPE")
    full_h = typed.assign(SERIALNO=household.SERIALNO)
    _validate_person_counts(full_h, person)
    codes, _header = classify_acs_housing_universe(
        typed.loc[:, ["TYPEHUGQ", "NP", "TEN"]], tenure_valid=valid.astype(bool)
    )
    return household, person, typed, lines, pw, codes


def _select(projection, serialnos):
    household, person, typed, lines, _pw, _codes = _tables(projection)
    if serialnos is None:
        chosen = tuple(sorted(household.SERIALNO))
    else:
        _require(
            type(serialnos) is tuple
            and bool(serialnos)
            and all(type(v) is str for v in serialnos)
            and len(set(serialnos)) == len(serialnos),
            "SELECTION_KEYS",
        )
        _require(set(serialnos) <= set(household.SERIALNO), "SELECTION_UNKNOWN")
        chosen = tuple(sorted(serialnos))
    horder = household.sort_values("SERIALNO", kind="stable").index
    porder = (
        person.assign(_line=lines)
        .sort_values(["SERIALNO", "_line"], kind="stable")
        .index
    )
    chosen_set = set(chosen)
    selected = {
        **projection,
        "households": [
            projection["households"][i]
            for i in horder
            if household.SERIALNO[i] in chosen_set
        ],
        "persons": [
            projection["persons"][i] for i in porder if person.SERIALNO[i] in chosen_set
        ],
    }
    full_counts = {
        "households": len(household),
        "persons": len(person),
        "vacant": int(((typed.TYPEHUGQ == 1) & (typed.NP == 0)).sum()),
        "institutional_gq": int((typed.TYPEHUGQ == 2).sum()),
        "noninstitutional_gq": int((typed.TYPEHUGQ == 3).sum()),
    }
    return selected, chosen, full_counts


@dataclass(frozen=True)
class AuthenticatedACSHousingSource:
    """Owned immutable source bytes; DataFrame access returns independent copies."""

    projection_json: bytes
    receipt_json: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "AUTHENTICATED_SOURCE_CONSTRUCTOR")

    @property
    def households(self):
        return _tables(json.loads(self.projection_json))[0]

    @property
    def persons(self):
        return _tables(json.loads(self.projection_json))[1]

    @property
    def codes(self):
        return _tables(json.loads(self.projection_json))[5]


def _reconstruct(private, paths, pins, serialnos, implementation):
    hc, h, hi = _archive(paths["household"], "household")
    pc, p, pi = _archive(paths["person"], "person")
    full = {
        "format": "acs-housing-lexical-projection/1",
        "household_columns": hc,
        "person_columns": pc,
        "households": h,
        "persons": p,
    }
    full_bytes = _json(full)
    _write(private / "full-projection.json", full_bytes, ACS_HU_SOURCE_MAX_BYTES)
    selected, chosen, counts = _select(full, serialnos)
    projection = _json(selected)
    _require(len(projection) <= ACS_HU_SOURCE_MAX_BYTES, "PROJECTION_TOO_LARGE")
    definition, definition_sha = _definition()
    receipt = {
        "format": "microcosm.acs_housing_universe_source.v1",
        "release_eligible": False,
        "vintage": 2024,
        "definition_sha256": definition_sha,
        "source_definitions": [
            definition["vacancy_definition"],
            definition["gq_definition"],
        ],
        "implementation_sha256": implementation,
        "parser_profile": {
            "engine": "stdlib_csv_reader",
            "encoding": "utf-8",
            "quoting": "QUOTE_MINIMAL",
            "strict": True,
            "csv.field_size_limit": csv.field_size_limit(),
            "field_max_bytes": _FIELD_MAX,
            "record_max_bytes": _RECORD_MAX,
        },
        "archives": [
            {"role": r, "filename": n, "sha256": d, "bytes": s} for r, n, d, s in pins
        ],
        "members": {"household": hi, "person": pi},
        "full_projection_sha256": _sha(full_bytes),
        "full_counts": counts,
        "selection": "all" if serialnos is None else "exact_serialnos",
        "selected_serialnos_sha256": _sha(_json(chosen)),
        "selected_counts": {
            "households": len(selected["households"]),
            "persons": len(selected["persons"]),
        },
        "projection_sha256": _sha(projection),
        "projection_bytes": len(projection),
        "source_weight_kind": "design",
        "calibrated_descendants_approved": False,
    }
    receipt_bytes = _json(receipt)
    _require(len(receipt_bytes) <= ACS_HU_RECEIPT_MAX_BYTES, "RECEIPT_TOO_LARGE")
    return AuthenticatedACSHousingSource(projection, receipt_bytes, _token=_TOKEN)


def _run_source(source_dir, snapshot_root, serialnos, output_dir, readback):
    try:
        original = _directory(Path(source_dir))
        destination = Path(os.path.abspath(output_dir))
        _require(not destination.is_relative_to(original), "OUTPUT_SOURCE_OVERLAP")
        if readback:
            _directory(destination)
        else:
            _directory(destination.parent)
            _require(not destination.exists(), "OUTPUT_EXISTS")
        implementation = _implementation()
        with _capture(source_dir, snapshot_root) as (private, paths, pins):
            result = _reconstruct(private, paths, pins, serialnos, implementation)
            _require(_implementation() == implementation, "IMPLEMENTATION_CHANGED")
            if readback:
                _directory(destination)
                _require(
                    {p.name for p in destination.iterdir()}
                    == {"projection.json", "receipt.json"},
                    "OUTPUT_ROSTER",
                )
                for name, expected, cap in (
                    (
                        "projection.json",
                        result.projection_json,
                        ACS_HU_SOURCE_MAX_BYTES,
                    ),
                    ("receipt.json", result.receipt_json, ACS_HU_RECEIPT_MAX_BYTES),
                ):
                    target = private / ("candidate-" + name)
                    # SOURCE_SIZE and FILE_TOO_LARGE are the pinned-archive
                    # codes. A candidate of the wrong length is a different
                    # fact, and an operator has to be able to tell them apart,
                    # so only this call's size refusals are renamed.
                    try:
                        digest = _copy(
                            destination / name, target, cap, exact_size=len(expected)
                        )
                    except ACSHousingSourceError as error:
                        if str(error) not in ("SOURCE_SIZE", "FILE_TOO_LARGE"):
                            raise
                        raise ACSHousingSourceError("RECONSTRUCTION_SIZE") from None
                    _require(
                        digest == _sha(expected)
                        and _persisted_sha(target, len(expected)) == digest,
                        "RECONSTRUCTION_MISMATCH",
                    )
            else:
                _directory(destination.parent)
                destination.mkdir(mode=0o700, exist_ok=False)
                _write(
                    destination / "projection.json",
                    result.projection_json,
                    ACS_HU_SOURCE_MAX_BYTES,
                )
                _write(
                    destination / "receipt.json",
                    result.receipt_json,
                    ACS_HU_RECEIPT_MAX_BYTES,
                )
            return result
    except ACSHousingSourceError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
        UnicodeError,
        csv.Error,
    ):
        raise ACSHousingSourceError("SOURCE_PREPARATION_REFUSED") from None


def produce_acs_housing_source(
    source_dir, *, snapshot_root, output_dir, serialnos=None
):
    """Authenticate full parents, retain whole selected keys, publish exclusively."""
    return _run_source(source_dir, snapshot_root, serialnos, output_dir, False)


def load_acs_housing_source(source_dir, output_dir, *, snapshot_root, serialnos=None):
    """Fresh full-parent reconstruction; candidate files never supply authority."""
    return _run_source(source_dir, snapshot_root, serialnos, output_dir, True)


def verify_acs_frame_projection(frame: Frame, projection: dict):
    """Check a Frame against source observations, without granting source authority."""
    household, person, typed, lines, pw, _codes = _tables(projection)
    h = frame.table("household")
    p = frame.person
    expected_h = household.loc[typed.NP > 0].reset_index(drop=True)
    _require(
        list(h.SERIALNO.astype(str)) == list(expected_h.SERIALNO),
        "FRAME_HOUSEHOLD_ORIGIN",
    )
    indexed = household.set_index("SERIALNO")
    observed = indexed.loc[h.SERIALNO.astype(str)]
    # The unchanged Frame builder retains TYPEHUGQ/NP/TEN but not raw WGTP.
    # WGTP stays lexical in the source artifact and binds the typed design
    # weights below, including the distinct GQ PWGTP substitution.
    for field in ("TYPEHUGQ", "NP"):
        _require(
            np.array_equal(h[field].to_numpy(), observed[field].map(int).to_numpy()),
            "FRAME_RAW_CONTROLS",
        )
    valid = observed.TEN.to_numpy() != ""
    _require(np.array_equal(h.TEN.notna().to_numpy(), valid), "FRAME_TEN_VALIDITY")
    _require(
        np.array_equal(
            h.TEN.to_numpy()[valid], observed.TEN[valid].map(int).to_numpy()
        ),
        "FRAME_TEN_VALUES",
    )
    state = "ST" if "ST" in observed else "STATE"
    _require(
        list(h.ST.astype(str)) == list(observed[state])
        and list(h.PUMA.astype(str)) == list(observed.PUMA),
        "FRAME_GEOGRAPHY",
    )
    ids = dict(zip(h.SERIALNO.astype(str), h.household_id, strict=True))
    _require(len(p) == len(person), "FRAME_PERSON_COUNT")
    expected_ids = np.asarray([ids[s] for s in person.SERIALNO], dtype="int64")
    _require(
        np.array_equal(p.person_household_id.to_numpy(), expected_ids)
        and np.array_equal(p.source_household_id.to_numpy(), expected_ids)
        and np.array_equal(p.SPORDER.to_numpy(), lines)
        and list(p.source_person_id.astype(str)) == [str(v) for v in lines],
        "FRAME_PERSON_ORIGIN",
    )
    _require(
        np.array_equal(p.PWGTP.to_numpy(), pw)
        and bool((p.source_year.to_numpy() == 2024).all()),
        "FRAME_PERSON_SOURCE",
    )
    _require(
        frame.weighted_entities == ("household",)
        and frame.weights_for("household").kind.value == "design",
        "FRAME_WEIGHT_KIND",
    )
    by_serial = dict(zip(person.SERIALNO, pw, strict=True))
    expected_weights = np.asarray(
        [
            by_serial[serial] if int(weight) == 0 else int(weight)
            for serial, weight in zip(observed.index, observed.WGTP, strict=True)
        ],
        dtype="float64",
    )
    _require(
        frame.weights_for("household").values.tobytes() == expected_weights.tobytes(),
        "FRAME_DESIGN_WEIGHT",
    )


def frame_content_sha256(frame: Frame):
    """Typed storage identity of every declared cell, axis, stratum and weight."""
    digest = hashlib.sha256(b"microcosm.acs-frame-storage/1\0")

    def series(values):
        digest.update(_json({"dtype": str(values.dtype), "name": values.name}))
        mask = values.isna().to_numpy(dtype=bool)
        digest.update(mask.tobytes())
        array = values.to_numpy(copy=False)
        if isinstance(values.dtype, np.dtype) and not values.dtype.hasobject:
            digest.update(np.ascontiguousarray(array).tobytes())
        else:
            for value, missing in zip(array, mask, strict=True):
                item = (
                    None
                    if missing
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                raw = _json(item)
                digest.update(len(raw).to_bytes(8, "little"))
                digest.update(raw)

    _require(frame.schema == US_SCHEMA, "FRAME_SCHEMA")
    digest.update(_json(dict(frame.metadata)))
    _require(not frame.mass_log, "FRAME_MASS_LOG")
    for entity in frame.entities:
        table = frame.table(entity)
        digest.update(_json([entity, list(table.columns)]))
        series(pd.Series(table.index.to_numpy(), name=table.index.name))
        for column in table:
            series(table[column])
    series(frame.strata)
    series(pd.Series(frame.strata.index.to_numpy(), name=frame.strata.index.name))
    for entity in frame.weighted_entities:
        digest.update(_json([entity, frame.weights_for(entity).kind.value]))
        digest.update(frame.weights_for(entity).values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedACSHousingPopulation:
    frame: Frame
    source: AuthenticatedACSHousingSource
    receipt_json: bytes

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def prepare_acs_housing_population(source_dir, *, snapshot_root, serialnos=None):
    """Validate the full lexical source; construct only exact selected native rows."""
    try:
        serialnos = AcsPumsSource.snapshot_serialnos(serialnos)
        implementation = _implementation()
        with _capture(source_dir, snapshot_root) as (private, paths, pins):
            source = _reconstruct(private, paths, pins, serialnos, implementation)
            projection = json.loads(source.projection_json)
            _require(bool(projection["persons"]), "EMPTY_GRAPH_POPULATION")
            raw, _builder_receipt = build_acs_pums_unit_frame(
                AcsPumsSource(
                    paths["household"],
                    paths["person"],
                    vintage=2024,
                    max_households=None,
                ),
                serialnos=serialnos,
            )
            mapped = map_acs_native_inputs(raw)
            frame = mapped.frame
            assert_operator_free_source_frame(
                frame,
                label="ACS HU authenticated source",
                native_inputs=mapped.native_inputs,
            )
            verify_acs_frame_projection(frame, projection)
            before = frame_content_sha256(frame)
            transitions = []
            original_dtypes = {
                (e, c): repr(frame.table(e)[c].dtype)
                for e in frame.entities
                for c in frame.table(e)
            }
            canonicalize_frame_string_dtypes(
                frame, boundary="ACS HU graph storage", in_place=True
            )
            for entity in frame.entities:
                table = frame.table(entity)
                for column in table:
                    original = table[column]
                    if isinstance(
                        original.dtype, pd.StringDtype
                    ) and original.dtype != dtype_for_token("string"):
                        promoted = original.astype(dtype_for_token("string"))
                        _require(
                            original.isna().equals(promoted.isna())
                            and np.array_equal(
                                original[original.notna()].to_numpy(),
                                promoted[promoted.notna()].to_numpy(),
                            ),
                            "GRAPH_STRING_PROMOTION",
                        )
                        table[column] = promoted
                    if original_dtypes[(entity, column)] != repr(table[column].dtype):
                        transitions.append(
                            {
                                "entity": entity,
                                "column": column,
                                "from": original_dtypes[(entity, column)],
                                "to": repr(table[column].dtype),
                            }
                        )
            verify_acs_frame_projection(frame, projection)
            _require(_implementation() == implementation, "IMPLEMENTATION_CHANGED")
            receipt = {
                "format": "microcosm.acs_housing_preparation.v2",
                "release_eligible": False,
                "source_receipt_sha256": _sha(source.receipt_json),
                "projection_sha256": _sha(source.projection_json),
                "implementation_sha256": implementation,
                "same_snapshot_frame_and_observations": True,
                "full_source_frame_before_selection": serialnos is None,
                "full_source_lexical_projection": True,
                "native_selection_before_person_accumulation": serialnos is not None,
                "selection_kind": "all"
                if serialnos is None
                else "engineering_exact_keys",
                "requested_serialnos": serialnos,
                "full_source_inclusion_probability": None,
                "pre_promotion_frame_sha256": before,
                "frame_sha256": frame_content_sha256(frame),
                "dtype_transitions": transitions,
                "entity_rows": {e: frame.n(e) for e in frame.entities},
                "weight_kind": "design",
                "HU_columns": "artifact_only",
            }
            return PreparedACSHousingPopulation(frame, source, _json(receipt))
    except ACSHousingSourceError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
        UnicodeError,
        csv.Error,
    ):
        raise ACSHousingSourceError("FRAME_PREPARATION_REFUSED") from None


def load_graph_acs_housing_universe(path, *, store=None):
    """Frame codec with an explicit store and a dedicated private capture child."""
    _require(type(store) is ContentStore, "EXPLICIT_CONTENT_STORE_REQUIRED")
    root = _directory(store.root) / "acs-hu-source-captures-v1"
    root.mkdir(mode=0o700, exist_ok=True)
    return prepare_acs_housing_population(path, snapshot_root=root).frame
