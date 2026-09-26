"""Authenticate literal original ASEC household weights without pooling them.

Only the closed, reviewed original-household member registry grants source
authority. This owner preserves whole requested cohorts, including records whose
weights cannot support a design anchor. It neither binds a receiving population
nor creates Frame weights, selects survey coverage, or authorizes a genuine run.
"""

from __future__ import annotations

import _csv
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
import sysconfig
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass, field
from fractions import Fraction
from importlib import resources
from pathlib import Path

from microcosm.build.cd_benchmark.origin import (
    AsecHouseholdOrigin,
    SourceMember,
    origin_key,
)

from . import asec_student_controls as _capture_owner
from . import native_household_origin as _origin
from . import source_csv_builtin as _csv_builtin

_TOKEN = object()
_SCHEMA = "microcosm.us.asec-original-household-weights.v1"
_MAX_MEMBER_BYTES = 64 * 1024**2
_MAX_ROWS = 1_000_000
_MAX_RECORD_CHARS = 65_536
_MAX_FIELD_CHARS = 128
_MAX_PAYLOAD_BYTES = 512 * 1024**2
_CAPSULE_ISSUANCE: dict[int, tuple[weakref.ReferenceType, str]] = {}
_COLUMNS = ("H_SEQ", "H_HHTYPE", "HSUP_WGT")
_DICTIONARIES = (
    (2023, 1, "66bd6e3fe516233ab63b75c60573451222b3b3d3235d61cfed96f64608de2117"),
    (2024, 1, "761c67ea53f5c3264329b3e9ddbdd802826ba4a859122b8a8a2f29ab85b3a840"),
    (2025, 8, "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f"),
)
_FIELD_CONTRACT = {
    "H_SEQ": {
        "concept": "Household sequence number",
        "role": "native_household_identity",
        "printed_range": "00001:99999",
        "literal_preserved": True,
    },
    "H_HHTYPE": {
        "role": "weight_universe_predicate",
        "predicate": "literal H_HHTYPE == 1",
        "other_codes_not_classified_here": True,
        "literal_preserved": True,
    },
    "HSUP_WGT": {
        "concept": "ASEC Supplement Final Weight",
        "universe_as_printed": "H_HHTYPE = 1",
        "printed_length": 8,
        "printed_range": "00000000:999999999",
        "implied_decimal_places": 2,
        "example_as_printed": "255212=2552.12",
        "width_range_inconsistency": True,
        "lexical_policy": "bounded ASCII unsigned integer; no CSV-width inference",
        "literal_preserved": True,
    },
}


class HouseholdWeightSourceError(ValueError):
    """Value-free source refusal; never exposes a path, native ID or weight."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HouseholdWeightSourceError(reason)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _issue_capsule(source: object, payload: bytes) -> str:
    """Seal this identity outside writable capsule state without retaining it."""
    _require(
        type(payload) is bytes and len(payload) <= _MAX_PAYLOAD_BYTES, "SOURCE_BYTES"
    )
    digest = _sha(payload)
    identity = id(source)

    def discard(reference):
        entry = _CAPSULE_ISSUANCE.get(identity)
        if entry is not None and entry[0] is reference:
            del _CAPSULE_ISSUANCE[identity]

    # WeakKeyDictionary would merge distinct, value-equal frozen dataclasses.
    _CAPSULE_ISSUANCE[identity] = (weakref.ref(source, discard), digest)
    return digest


def _checked_capsule_payload(source: object) -> bytes:
    """Return exactly the checked bytes; copies and deserialization are unissued."""
    entry = _CAPSULE_ISSUANCE.get(id(source))
    _require(entry is not None and entry[0]() is source, "SOURCE_MUTATION")
    payload = getattr(source, "payload", None)
    _require(
        type(payload) is bytes
        and len(payload) <= _MAX_PAYLOAD_BYTES
        and _sha(payload) == entry[1],
        "SOURCE_MUTATION",
    )
    return payload


def _member_path_snapshot(
    member_paths: Mapping[int, str | Path],
    pins: tuple[_origin.AsecNativeMemberPin, ...],
) -> dict[int, Path]:
    """Bound the closed cohort snapshot before allocation and each value lookup."""
    _require(isinstance(member_paths, Mapping), "MEMBER_PATHS")
    try:
        count = len(member_paths)
        _require(0 < count <= len(pins), "MEMBER_PATHS")
        allowed = {pin.income_year for pin in pins}
        paths = {}
        for year in member_paths:
            _require(
                len(paths) < count
                and type(year) is int
                and year in allowed
                and year not in paths,
                "MEMBER_PATHS",
            )
            path = member_paths[year]
            _require(isinstance(path, (str, Path)), "MEMBER_PATHS")
            paths[year] = Path(path)
        _require(len(paths) == count == len(member_paths), "MEMBER_PATHS")
    except (TypeError, ValueError, RuntimeError, KeyError, OverflowError):
        raise HouseholdWeightSourceError("MEMBER_PATHS") from None
    return paths


def _encode(value: object) -> bytes:
    """Canonical JSON with a bound during encoding, not after a huge allocation."""
    encoder = json.JSONEncoder(
        ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    result = bytearray()
    for fragment in encoder.iterencode(value):
        encoded = fragment.encode("ascii")
        _require(len(result) + len(encoded) <= _MAX_PAYLOAD_BYTES, "ENCODING_SIZE")
        result.extend(encoded)
    return bytes(result)


def _registry() -> tuple[_origin.AsecNativeMemberPin, ...]:
    pins = tuple(_origin._ASEC_MEMBER_PINS)
    _require(0 < len(pins) <= 3, "MEMBER_REGISTRY")
    try:
        for pin in pins:
            _origin.AsecNativeMemberPin(**asdict(pin))
            _require(
                pin.income_year in (2022, 2023, 2024)
                and pin.size_bytes <= _MAX_MEMBER_BYTES
                and pin.rows <= _MAX_ROWS,
                "MEMBER_REGISTRY",
            )
        _require(
            len({p.income_year for p in pins}) == len(pins)
            and len({p.canonical_member_id for p in pins}) == len(pins),
            "MEMBER_REGISTRY",
        )
    except (TypeError, ValueError, AttributeError):
        raise HouseholdWeightSourceError("MEMBER_REGISTRY") from None
    return tuple(sorted(pins, key=lambda pin: pin.income_year))


def _implementation() -> dict:
    """Bind the live owner, registry, private capture, codec and field policy."""
    _require(_csv_builtin.csv_reader_bound(csv), "SOURCE_CSV_READER_CHANGED")
    parser_binary = getattr(_csv, "__file__", None)
    if parser_binary is None:
        parser_binary = (
            Path(sysconfig.get_config_var("LIBDIR"))
            / sysconfig.get_config_var("LDLIBRARY")
            if sysconfig.get_config_var("Py_ENABLE_SHARED")
            else Path(sys.executable)
        )
    return {
        "schema": _SCHEMA,
        "modules": {
            name: _sha(resources.files(__package__).joinpath(name).read_bytes())
            for name in (
                "asec_original_household_weights.py",
                "source_csv_builtin.py",
                "native_household_origin.py",
                "asec_student_controls.py",
                "education_assistance_source.py",
            )
        },
        "origin_codec_sha256": _sha(
            resources.files("microcosm.build.cd_benchmark")
            .joinpath("origin.py")
            .read_bytes()
        ),
        "python": sys.version,
        "csv_python_sha256": _sha(Path(csv.__file__).read_bytes()),
        "csv_native_origin": _csv.__spec__.origin,
        "csv_native_provider": Path(parser_binary).name,
        "csv_native_sha256": _sha(Path(parser_binary).read_bytes()),
        "registry": [asdict(pin) for pin in _registry()],
        "fields": _FIELD_CONTRACT,
        "dictionaries": _DICTIONARIES,
        "limits": [
            _MAX_MEMBER_BYTES,
            _MAX_ROWS,
            _MAX_RECORD_CHARS,
            _MAX_FIELD_CHARS,
            _MAX_PAYLOAD_BYTES,
        ],
    }


class _DigestReader(io.RawIOBase):
    """Bound and hash the actual captured bytes consumed by UTF-8 decoding."""

    def __init__(self, raw: io.FileIO, size: int):
        self.raw = raw
        self.size = size
        self.count = 0
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        limit = min(len(buffer), self.size - self.count + 1)
        count = self.raw.readinto(memoryview(buffer)[:limit])
        self.count += count
        _require(self.count <= self.size, "CAPTURE_SIZE")
        self.digest.update(memoryview(buffer)[:count])
        return count


class _RecordLines:
    """Limit the complete logical CSV record, including embedded newlines."""

    def __init__(self, text: io.TextIOWrapper):
        self.text = text
        self.characters = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.text.readline(_MAX_RECORD_CHARS - self.characters + 1)
        if not line:
            raise StopIteration
        self.characters += len(line)
        _require(self.characters <= _MAX_RECORD_CHARS, "MEMBER_RECORD_SIZE")
        return line


def _record_fields(tokens: Sequence[str]) -> dict:
    native_token, household_type, weight_token = tokens
    _require(all(len(t) <= _MAX_FIELD_CHARS for t in tokens), "MEMBER_FIELD_SIZE")
    _require(
        re.fullmatch(r"[0-9]+", native_token) is not None
        and 1 <= int(native_token) <= 99999,
        "MEMBER_NATIVE_KEY",
    )
    if household_type == "1":
        universe = "in"
    elif re.fullmatch(r"(?:0|[2-9])", household_type):
        # Establish only that the printed weight predicate is false, not the
        # meaning/validity of any other household classification.
        universe = "outside"
    else:
        universe = "unresolved"
    units = None
    if weight_token == "":
        state = "missing"
    elif re.fullmatch(r"[0-9]+", weight_token) is None:
        state = "malformed"
    else:
        units = int(weight_token)
        state = "integer" if units <= 999999999 else "out_of_range"
    return {
        "H_SEQ": native_token,
        "native_household_id": int(native_token),
        "H_HHTYPE": household_type,
        "HSUP_WGT": weight_token,
        "universe_status": universe,
        "weight_token_status": state,
        "weight_integer_units": units,
        "weight_denominator": 100 if units is not None else None,
        "weight_status": (
            "valid_in_universe"
            if universe == "in" and state == "integer"
            else "unresolved_weight"
            if universe == "in"
            else "outside_universe"
            if universe == "outside"
            else "unresolved_universe"
        ),
    }


def _regular_opener(path, flags):
    fd = os.open(path, flags | os.O_NONBLOCK)
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), "CAPTURE_KIND")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_capture(capture: Path, pin: _origin.AsecNativeMemberPin) -> list[dict]:
    csv_reader = _csv_builtin.capture_csv_reader(csv)
    _require(csv_reader is not None, "SOURCE_CSV_READER_CHANGED")
    member = SourceMember(pin.canonical_member_id, pin.member_sha256)
    result, seen = [], set()
    with open(capture, "rb", buffering=0, opener=_regular_opener) as raw:
        before = os.fstat(raw.fileno())
        _require(before.st_size == pin.size_bytes, "CAPTURE_SIZE")
        digest_reader = _DigestReader(raw, pin.size_bytes)
        with io.TextIOWrapper(
            io.BufferedReader(digest_reader), encoding="utf-8-sig", newline=""
        ) as text:
            lines = _RecordLines(text)
            reader = csv_reader(lines, strict=True)
            header = next(reader, [])
            _require(
                bool(header)
                and all(header)
                and len(set(header)) == len(header)
                and set(_COLUMNS) <= set(header),
                "MEMBER_HEADER",
            )
            positions = [header.index(column) for column in _COLUMNS]
            while True:
                lines.characters = 0
                try:
                    row = next(reader)
                except StopIteration:
                    break
                _require(len(row) == len(header), "MEMBER_ROW_WIDTH")
                _require(len(result) < pin.rows, "MEMBER_ROWS")
                record = _record_fields([row[i] for i in positions])
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
                digest_reader.count == pin.size_bytes
                and digest_reader.digest.hexdigest() == pin.member_sha256,
                "CAPTURE_CHANGED",
            )
            after = os.fstat(raw.fileno())
            _require(
                all(
                    getattr(before, attribute) == getattr(after, attribute)
                    for attribute in (
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
class AuthenticatedAsecHouseholdWeights:
    """Reader-issued immutable source evidence, not a population-weight vector."""

    payload: bytes
    _token: InitVar[object] = None
    _issued_sha256: str = field(init=False, repr=False)

    def __post_init__(self, _token: object) -> None:
        _require(_token is _TOKEN, "SOURCE_CONSTRUCTOR")
        object.__setattr__(self, "_issued_sha256", _issue_capsule(self, self.payload))

    @property
    def document(self) -> dict:
        """Fresh defensive decoding, checked against the live source producer."""
        return verify_asec_household_weights_source(self)

    def original_weight_fractions(
        self, native_roster: Sequence[tuple[int, int]]
    ) -> tuple[Fraction, ...]:
        """Exact weights for known in-universe keys; no Frame binding or stamping."""
        doc = self.document
        _require(
            isinstance(native_roster, (list, tuple))
            and 0 < len(native_roster) <= _MAX_ROWS,
            "ANCHOR_ROSTER",
        )
        keys = []
        for key in native_roster:
            _require(
                isinstance(key, (list, tuple))
                and len(key) == 2
                and all(type(value) is int for value in key),
                "ANCHOR_ROSTER",
            )
            keys.append(tuple(key))
        _require(len(set(keys)) == len(keys), "ANCHOR_ROSTER")
        records = {
            (row["income_year"], row["native_household_id"]): row
            for row in doc["records"]
        }
        values = []
        for key in keys:
            _require(key in records, "ANCHOR_UNKNOWN")
            row = records[key]
            _require(row["weight_status"] == "valid_in_universe", "ANCHOR_UNRESOLVED")
            values.append(Fraction(row["weight_integer_units"], 100))
        return tuple(values)


def verify_asec_household_weights_source(
    source: AuthenticatedAsecHouseholdWeights,
) -> dict:
    """Validate reader-issued evidence; arbitrary JSON/digests cannot issue it."""
    _require(type(source) is AuthenticatedAsecHouseholdWeights, "SOURCE_TYPE")
    payload = _checked_capsule_payload(source)
    doc = json.loads(payload)
    _require(_encode(doc["producer"]) == _encode(_implementation()), "PRODUCER_CHANGED")
    return doc


def load_authenticated_asec_household_weights(
    member_paths: Mapping[int, str | Path], *, candidate: bytes | None = None
) -> AuthenticatedAsecHouseholdWeights:
    """Reconstruct complete requested cohorts from closed original member pins.

    Mapping keys are income/source years, not survey years. Any nonempty subset
    of the closed cohorts is explicit; every requested original member is read
    completely. Optional candidate bytes must equal the independent canonical
    reconstruction. There is no public pin override or candidate-only admission.
    """
    pins = _registry()
    paths = _member_path_snapshot(member_paths, pins)
    _require(
        candidate is None
        or (type(candidate) is bytes and len(candidate) <= _MAX_PAYLOAD_BYTES),
        "CANDIDATE_SIZE",
    )
    producer = _implementation()
    _require(producer["registry"] == [asdict(pin) for pin in pins], "PRODUCER_CHANGED")
    implementation = _encode(producer)
    selected = [pin for pin in pins if pin.income_year in paths]
    records = []
    with tempfile.TemporaryDirectory(prefix="asec-original-household-weights-") as tmp:
        for pin in selected:
            capture = Path(tmp) / f"{pin.income_year}.csv"
            try:
                digest = _capture_owner._snapshot(
                    paths[pin.income_year], capture, size=pin.size_bytes
                )
            except (OSError, ValueError, TypeError):
                raise HouseholdWeightSourceError("MEMBER_CAPTURE") from None
            _require(digest == pin.member_sha256, "MEMBER_SHA256")
            try:
                records.extend(_read_capture(capture, pin))
            except HouseholdWeightSourceError:
                raise
            except (OSError, UnicodeError, csv.Error, ValueError, OverflowError):
                raise HouseholdWeightSourceError("MEMBER_ENCODING") from None
    _require(_encode(_implementation()) == implementation, "PRODUCER_CHANGED")
    payload = _encode(
        {
            "schema": _SCHEMA,
            "producer": json.loads(implementation),
            "members": [asdict(pin) for pin in selected],
            "fields": _FIELD_CONTRACT,
            "dictionary_authorities": [
                {
                    "survey_year": year,
                    "pdf_page_1based": page,
                    "printed_page": "6A-1",
                    "sha256": digest,
                    "url": "https://www2.census.gov/programs-surveys/cps/datasets/"
                    f"{year}/march/asec{year}_ddl_pub_full.pdf",
                }
                for year, page, digest in _DICTIONARIES
            ],
            "native_roster_sha256": _sha(
                _encode([[r["income_year"], r["native_household_id"]] for r in records])
            ),
            "projection_sha256": _sha(_encode(records)),
            "records": records,
            "source_authenticated": True,
            "population_binding_authenticated": False,
            "release_eligible": False,
            "period_and_coverage_policy": "not_selected_by_this_source_owner",
        }
    )
    _require(candidate is None or candidate == payload, "CANDIDATE_MISMATCH")
    return AuthenticatedAsecHouseholdWeights(payload, _token=_TOKEN)
