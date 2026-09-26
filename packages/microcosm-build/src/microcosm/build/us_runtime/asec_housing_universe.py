"""Source-specific occupied-HU evidence; never a population or release verdict."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path

import numpy as np
import pandas as pd

from .asec_housing_status import DICTIONARIES, ZERO_ORIGIN_EVIDENCE, ZERO_POLICY

RAW_COLUMNS = (
    "household_id",
    "income_year",
    "H_SEQ",
    "H_HHTYPE",
    "H_LIVQRT",
    "HRHTYPE",
    "H_TENURE",
)
CONTEXT_COLUMNS = ("H_HHTYPE", "H_LIVQRT", "HRHTYPE", "H_TENURE")
DERIVED_COLUMNS = (
    "interview_scope",
    "physical_unit",
    "household_kind",
    "tenure_subtype",
    "occupied_hu",
    "hu_tenure_class",
    "unresolved_reasons",
)
COLUMNS = RAW_COLUMNS + DERIVED_COLUMNS
MAGIC = b"MCAHUNIV\x01"
HEADER_MAX_BYTES = 65536
MAX_HOUSEHOLDS = 400_000
PAYLOAD_MAX_BYTES = (
    len(MAGIC)
    + 4
    + HEADER_MAX_BYTES
    + MAX_HOUSEHOLDS * (8 * len(RAW_COLUMNS) + len(DERIVED_COLUMNS))
    + 32
)
_UNIVERSE_TOKEN = object()


class HousingUniverseRefusalError(ValueError):
    """Sanitized reason for unavailable or inconsistent source evidence."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HousingUniverseRefusalError(reason)


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse(value: bytes) -> dict:
    _require(type(value) is bytes and 0 < len(value) <= HEADER_MAX_BYTES, "HEADER_SIZE")
    try:
        result = json.loads(value)
        _require(type(result) is dict and _json(result) == value, "HEADER_CANONICAL")
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise HousingUniverseRefusalError("HEADER_FORMAT") from None


def _codebook() -> dict:
    return {
        "interview_scope": {
            "unavailable": 0,
            "interview": 1,
            "type_a_noninterview": 2,
            "type_bc_noninterview": 3,
        },
        "physical_unit": {"unavailable": 0, "housing_unit": 1, "other_unit": 2},
        "household_kind": {
            "source_zero_or_not_established": 0,
            "primary_household": 1,
            "group_quarters": 2,
        },
        "tenure_subtype": {
            "niu_or_zero_unresolved": 0,
            "owned_or_being_bought": 1,
            "cash_renter": 2,
            "no_cash_rent": 3,
        },
        "occupied_hu": {
            "not_established": 0,
            "occupied_housing_unit": 1,
            "outside_housing_unit": 2,
        },
        "hu_tenure_class": {
            "outside_or_unestablished_hu": 0,
            "owner": 1,
            "renter": 2,
            "occupied_hu_unresolved_tenure": 3,
        },
        "unresolved_reason_bits": {
            "outside_interview_scope": 1,
            "interview_zero": 2,
            "quarters_zero": 4,
            "household_kind_zero": 8,
            "hu_quarters_gq_kind": 16,
            "context_conflict": 32,
            "hu_tenure_zero": 64,
            "other_quarters_primary_kind": 128,
        },
    }


def _definition_identity() -> str:
    return _sha(
        _json(
            {
                "module": _sha(
                    resources.files(__package__)
                    .joinpath("asec_housing_universe.py")
                    .read_bytes()
                ),
                "codebook": _codebook(),
                "dictionaries": DICTIONARIES,
                "zero_origin_evidence": ZERO_ORIGIN_EVIDENCE,
                "dependencies": {
                    name: metadata.version(name) for name in ("numpy", "pandas")
                },
            }
        )
    )


def _derive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(raw["household_id"])
    _require(0 < n <= MAX_HOUSEHOLDS, "ROW_COUNT")
    _require(
        set(raw) == set(RAW_COLUMNS)
        and all(v.shape == (n,) and v.dtype == np.dtype("int64") for v in raw.values()),
        "RAW_INT64",
    )
    for name, maximum in (
        ("H_HHTYPE", 3),
        ("H_LIVQRT", 12),
        ("HRHTYPE", 10),
        ("H_TENURE", 3),
    ):
        _require(
            bool(((raw[name] >= 0) & (raw[name] <= maximum)).all()), "CONTEXT_DOMAIN"
        )
    _require(bool(np.isin(raw["income_year"], (2022, 2023, 2024)).all()), "INCOME_YEAR")
    _require(bool((raw["H_SEQ"] > 0).all()), "NATIVE_KEY")
    _require(len(np.unique(raw["household_id"])) == n, "DUPLICATE_HOUSEHOLD_ID")
    _require(
        pd.MultiIndex.from_arrays([raw["income_year"], raw["H_SEQ"]]).is_unique,
        "DUPLICATE_NATIVE_KEY",
    )
    interview = raw["H_HHTYPE"] == 1
    noninterview = raw["H_HHTYPE"] >= 2
    quarters, kind, tenure = raw["H_LIVQRT"], raw["HRHTYPE"], raw["H_TENURE"]
    hu_quarters = (quarters >= 1) & (quarters <= 7)
    other_quarters = quarters >= 8
    primary = (kind >= 1) & (kind <= 8)
    gq = kind >= 9
    context_conflict = (interview & (kind == 0)) | (
        noninterview & ((kind > 0) | (tenure > 0))
    )
    hu = interview & hu_quarters & primary & ~context_conflict
    outside = interview & other_quarters & gq & ~context_conflict
    reasons = noninterview.astype("uint8")
    for mask, bit in (
        (raw["H_HHTYPE"] == 0, 2),
        (interview & (quarters == 0), 4),
        (interview & (kind == 0), 8),
        (interview & hu_quarters & gq, 16),
        (context_conflict, 32),
        (hu & (tenure == 0), 64),
        (interview & other_quarters & primary, 128),
    ):
        reasons |= mask.astype("uint8") * bit
    return {
        "interview_scope": raw["H_HHTYPE"].astype("uint8"),
        "physical_unit": np.where(
            hu_quarters, 1, np.where(other_quarters, 2, 0)
        ).astype("uint8"),
        "household_kind": np.where(primary, 1, np.where(gq, 2, 0)).astype("uint8"),
        "tenure_subtype": tenure.astype("uint8"),
        "occupied_hu": np.where(hu, 1, np.where(outside, 2, 0)).astype("uint8"),
        "hu_tenure_class": np.where(
            hu, np.where(tenure == 1, 1, np.where(tenure >= 2, 2, 3)), 0
        ).astype("uint8"),
        "unresolved_reasons": reasons,
    }


@dataclass(frozen=True)
class _HousingUniverse:
    header: bytes
    buffers: tuple[bytes, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _UNIVERSE_TOKEN, "UNIVERSE_CONSTRUCTOR_UNAVAILABLE")

    @property
    def header_data(self) -> dict:
        return _parse(self.header)

    def array(self, name: str) -> np.ndarray:
        if name not in COLUMNS:
            raise KeyError(name)
        return np.frombuffer(
            self.buffers[COLUMNS.index(name)],
            dtype="<i8" if name in RAW_COLUMNS else "u1",
        )

    def validate(self) -> None:
        """Recompute every derived code from immutable raw evidence."""
        _require(
            type(self) in (SyntheticHousingUniverse, AuthenticatedHousingUniverse),
            "UNIVERSE_TYPE",
        )
        data = self.header_data
        try:
            _require(
                data["definition_sha256"] == _definition_identity(),
                "DEFINITION_CHANGED",
            )
            n = data["household_rows"]
            _require(type(n) is int and 0 < n <= MAX_HOUSEHOLDS, "ROW_COUNT")
            _require(
                type(self.buffers) is tuple and len(self.buffers) == len(COLUMNS),
                "BUFFER_ROSTER",
            )
            for name, buf in zip(COLUMNS, self.buffers, strict=True):
                _require(
                    type(buf) is bytes
                    and len(buf) == n * (8 if name in RAW_COLUMNS else 1),
                    "BUFFER_SIZE",
                )
            authenticated = type(self) is AuthenticatedHousingUniverse
            _require(
                data["source_authentication"]
                == (
                    "checkpoint_bytes_verified"
                    if authenticated
                    else "synthetic_unverified"
                ),
                "UNIVERSE_AUTHENTICATION",
            )
            derived = _derive({name: self.array(name) for name in RAW_COLUMNS})
            _require(
                all(
                    derived[name].tobytes() == self.buffers[COLUMNS.index(name)]
                    for name in DERIVED_COLUMNS
                ),
                "DERIVED_EVIDENCE",
            )
            if authenticated:
                from .asec_housing_universe_source import _implementation

                _require(
                    data["implementation"] == _implementation(),
                    "IMPLEMENTATION_CHANGED",
                )
        except (KeyError, TypeError, OverflowError):
            raise HousingUniverseRefusalError("UNIVERSE_CONTRACT") from None

    @property
    def content_sha256(self) -> str:
        self.validate()
        digest = hashlib.sha256(b"microcosm/asec-housing-universe-content/1\0")
        for part in _parts(self):
            digest.update(part)
        return digest.hexdigest()


class SyntheticHousingUniverse(_HousingUniverse):
    """Invented observations, with no source authority."""


class AuthenticatedHousingUniverse(_HousingUniverse):
    """Issued only from an authenticated current-money source."""


def _issue(raw: pd.DataFrame, *, source=None, implementation=None) -> _HousingUniverse:
    _require(
        type(raw) is pd.DataFrame
        and raw.columns.is_unique
        and set(raw.columns) == set(RAW_COLUMNS),
        "RAW_COLUMNS",
    )
    _require(
        all(raw[name].dtype == np.dtype("int64") for name in RAW_COLUMNS), "RAW_INT64"
    )
    _require(0 < len(raw) <= MAX_HOUSEHOLDS, "ROW_COUNT")
    buffers = tuple(
        raw[name].to_numpy(dtype="<i8", copy=True).tobytes() for name in RAW_COLUMNS
    )
    derived = _derive(
        {
            name: np.frombuffer(buf, dtype="<i8")
            for name, buf in zip(RAW_COLUMNS, buffers, strict=True)
        }
    )
    header = {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_housing_universe.v1",
        "source_authentication": "synthetic_unverified"
        if source is None
        else "checkpoint_bytes_verified",
        "release_eligible": False,
        "source_period_kind": "interview_household_universe",
        "zero_origin_policy": ZERO_POLICY,
        "zero_origin_evidence": ZERO_ORIGIN_EVIDENCE,
        "household_rows": len(raw),
        "columns": COLUMNS,
        "codebook": _codebook(),
        "dictionaries": DICTIONARIES,
        "definition_sha256": _definition_identity(),
        "source": source,
        "implementation": implementation,
        "limitations": "No vacancy assertion, ACS source authority, stacked denominator, income bridge or launch certification.",
    }
    cls = SyntheticHousingUniverse if source is None else AuthenticatedHousingUniverse
    result = cls(
        _json(header),
        buffers + tuple(derived[name].tobytes() for name in DERIVED_COLUMNS),
        _token=_UNIVERSE_TOKEN,
    )
    result.validate()
    return result


def classify_synthetic_housing_universe(
    observations: pd.DataFrame,
) -> SyntheticHousingUniverse:
    """Classify invented rows without granting source authority."""
    return _issue(observations)


def _parts(value):
    yield MAGIC
    yield struct.pack("<I", len(value.header))
    yield value.header
    yield from value.buffers


def encode_housing_universe(value: _HousingUniverse) -> bytes:
    """Canonical primitive bytes with a bounded header and transport checksum."""
    _require(
        type(value) in (SyntheticHousingUniverse, AuthenticatedHousingUniverse),
        "UNIVERSE_TYPE",
    )
    value.validate()
    payload = b"".join(_parts(value))
    return payload + hashlib.sha256(payload).digest()


def decode_housing_universe(
    payload: bytes, *, expected: _HousingUniverse
) -> _HousingUniverse:
    """Only independently issued complete expected evidence confers replay authority."""
    _require(
        type(expected) in (SyntheticHousingUniverse, AuthenticatedHousingUniverse),
        "EXPECTED_UNIVERSE",
    )
    expected.validate()
    _require(
        type(payload) is bytes
        and len(MAGIC) + 4 + 32 < len(payload) <= PAYLOAD_MAX_BYTES,
        "PAYLOAD_SIZE",
    )
    # Compare before parsing any candidate header or buffer; no alternate encoding.
    _require(payload == encode_housing_universe(expected), "EXPECTED_CONTENT")
    return type(expected)(expected.header, expected.buffers, _token=_UNIVERSE_TOKEN)


def write_housing_universe(value: _HousingUniverse, path: str | Path) -> str:
    """Create an immutable artifact; never overwrite an existing path."""
    payload = encode_housing_universe(value)
    try:
        with Path(path).open("xb") as handle:
            handle.write(payload)
    except OSError:
        raise HousingUniverseRefusalError("ARTIFACT_WRITE") from None
    return value.content_sha256


def read_housing_universe(
    path: str | Path, *, expected: _HousingUniverse
) -> _HousingUniverse:
    """Bound regular-file reads; no FIFO, device or candidate header parsing."""
    _require(
        type(expected) in (SyntheticHousingUniverse, AuthenticatedHousingUniverse),
        "EXPECTED_UNIVERSE",
    )
    size = len(encode_housing_universe(expected))
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_size == size, "ARTIFACT_FILE")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            payload = handle.read(size + 1)
        return decode_housing_universe(payload, expected=expected)
    except (OSError, TypeError):
        raise HousingUniverseRefusalError("ARTIFACT_READ") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
