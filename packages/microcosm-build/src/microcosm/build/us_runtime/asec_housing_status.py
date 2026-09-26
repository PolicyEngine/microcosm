"""Immutable interview-time ASEC housing observations and publisher routing.

Synthetic construction is deliberately distinct from source-issued authority.
No annual participation, program eligibility, rent or subsidy dollars are derived.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "household_id",
    "income_year",
    "survey_year",
    "H_SEQ",
    "H_TENURE",
    "HRHTYPE",
    "H_HHTYPE",
    "HPUBLIC",
    "HLORENT",
    "I_HPUBLI",
    "I_HLOREN",
)
CONTEXT_COLUMNS = ("H_TENURE", "HRHTYPE", "H_HHTYPE")
OBSERVED_COLUMNS = ("HPUBLIC", "HLORENT", "I_HPUBLI", "I_HLOREN")
DERIVED_COLUMNS = (
    "status",
    "route",
    "receipt_valid",
    "group_quarters",
    "conflicts",
    "unknown_reasons",
    "public_quality",
    "lower_quality",
    "HPUBLIC_valid",
    "HLORENT_valid",
    "HPUBLIC_zero_origin",
    "HLORENT_zero_origin",
    "I_HPUBLI_zero_origin",
    "I_HLOREN_zero_origin",
)
COLUMNS = RAW_COLUMNS + DERIVED_COLUMNS
MAGIC = b"MCAHSTAT\x01"
HEADER_MAX_BYTES = 65536
MAX_HOUSEHOLDS = 400_000
ROW_BYTES = 8 * len(RAW_COLUMNS) + len(DERIVED_COLUMNS)
PAYLOAD_MAX_BYTES = len(MAGIC) + 4 + HEADER_MAX_BYTES + ROW_BYTES * MAX_HOUSEHOLDS + 32
ZERO_POLICY = "frozen_fillna_zero_origin_unresolved"
ZERO_ORIGIN_EVIDENCE = {
    "inspected_repository": "PolicyEngine/policyengine-us-data",
    "inspected_commit": "000c90c1289994ae6d12d141578c83c315aa58b8",
    "inspected_path": "policyengine_us_data/datasets/cps/census_cps.py",
    "inspected_sha256": "ce3dac429aad665df25c503fdcb850a4e31d89268dd73acec8343c70c7cfb647",
    "historical_execution_authenticated": False,
    "original_csv_compared": False,
    "inference": "Int64 storage is consistent with fillna being a no-op only if the inspected default-CSV producer and its assumed runtime actually produced the pinned HDFs; no historical execution proof or CSV comparison is claimed.",
}
_STATUS_TOKEN = object()
DICTIONARIES = (
    (
        2023,
        "66bd6e3fe516233ab63b75c60573451222b3b3d3235d61cfed96f64608de2117",
        (1, 2, 3, 8, 10),
    ),
    (
        2024,
        "761c67ea53f5c3264329b3e9ddbdd802826ba4a859122b8a8a2f29ab85b3a840",
        (1, 2, 3, 8, 10),
    ),
    (
        2025,
        "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f",
        (8, 9, 10, 15, 17),
    ),
)


class HousingStatusRefusalError(ValueError):
    """Value-free refusal reason for unsupported source or artifact evidence."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HousingStatusRefusalError(reason)


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _parse(value: bytes) -> dict:
    _require(type(value) is bytes and 0 < len(value) <= HEADER_MAX_BYTES, "HEADER_SIZE")
    try:
        parsed = json.loads(value)
        _require(type(parsed) is dict and _json(parsed) == value, "HEADER_CANONICAL")
        return parsed
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise HousingStatusRefusalError("HEADER_FORMAT") from None


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _codebook() -> dict:
    return {
        "status": {
            "unknown": 0,
            "receipt": 1,
            "nonreceipt": 2,
            "not_in_universe": 3,
            "conflict": 4,
        },
        "route": {
            "unresolved": 0,
            "public_housing": 1,
            "lower_rent": 2,
            "both_negative": 3,
            "not_in_universe": 4,
        },
        "group_quarters": {"not_gq": 0, "gq": 1, "unavailable": 2},
        "conflict_bits": {
            "excluded_scope": 1,
            "parent_route": 2,
            "public_allocation": 4,
            "lower_allocation": 8,
            "context": 16,
        },
        "unknown_reason_bits": {"parent_answer": 1, "lower_answer": 2, "scope": 4},
        "quality": {
            "not_applicable": 0,
            "no_allocation_code_origin_unresolved": 1,
            "allocated": 2,
            "conflict": 3,
        },
        "validity": "1 iff substantive response; receipt_valid additionally requires consistent known receipt/nonreceipt",
        "zero_origin": {"not_zero": 0, "frozen_zero_origin_unresolved": 1},
    }


def _definition_identity() -> str:
    return _sha(
        _json(
            {
                "module": _sha(
                    resources.files(__package__)
                    .joinpath("asec_housing_status.py")
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
    """Vectorized dictionary-domain checks and explicit interpretation axes."""
    n = len(raw["household_id"])
    _require(0 < n <= MAX_HOUSEHOLDS, "ROW_COUNT")
    _require(
        all(v.shape == (n,) and v.dtype == np.dtype("int64") for v in raw.values()),
        "RAW_INT64",
    )
    for name, maximum in (
        ("HPUBLIC", 2),
        ("HLORENT", 2),
        ("I_HPUBLI", 1),
        ("I_HLOREN", 1),
    ):
        _require(bool(((raw[name] >= 0) & (raw[name] <= maximum)).all()), "CODE_DOMAIN")
    for name, maximum in (("H_TENURE", 3), ("HRHTYPE", 10), ("H_HHTYPE", 3)):
        # Zero context remains unresolved under the frozen fill provenance.
        _require(
            bool(((raw[name] >= 0) & (raw[name] <= maximum)).all()), "CONTEXT_DOMAIN"
        )
    _require(bool(np.isin(raw["income_year"], [2022, 2023, 2024]).all()), "INCOME_YEAR")
    _require(bool((raw["survey_year"] == raw["income_year"] + 1).all()), "SURVEY_YEAR")
    _require(bool((raw["H_SEQ"] > 0).all()), "NATIVE_KEY")
    _require(len(np.unique(raw["household_id"])) == n, "DUPLICATE_HOUSEHOLD_ID")
    keys = pd.MultiIndex.from_arrays([raw["income_year"], raw["H_SEQ"]])
    _require(keys.is_unique, "DUPLICATE_NATIVE_KEY")
    public, lower = raw["HPUBLIC"], raw["HLORENT"]
    interview = raw["H_HHTYPE"] == 1
    noninterview = raw["H_HHTYPE"] >= 2
    known = interview & np.isin(raw["H_TENURE"], [2, 3])
    excluded = noninterview | (interview & (raw["H_TENURE"] == 1))
    scope_unknown = ~(known | excluded)
    context_conflict = (interview & (raw["HRHTYPE"] == 0)) | (
        noninterview & ((raw["HRHTYPE"] > 0) | (raw["H_TENURE"] > 0))
    )
    parent_conflict = (lower > 0) & (public != 2)
    public_alloc_conflict = (raw["I_HPUBLI"] == 1) & ((public == 0) | excluded)
    lower_alloc_conflict = (raw["I_HLOREN"] == 1) & (
        (lower == 0) | (public != 2) | excluded
    )
    conflicts = (
        ((excluded & ((public > 0) | (lower > 0))).astype("uint8"))
        | (parent_conflict.astype("uint8") * 2)
        | (public_alloc_conflict.astype("uint8") * 4)
        | (lower_alloc_conflict.astype("uint8") * 8)
        | (context_conflict.astype("uint8") * 16)
    )
    status = np.zeros(n, dtype="uint8")
    route = np.zeros(n, dtype="uint8")
    for mask, state, path in (
        (excluded, 3, 4),
        (known & (public == 1) & (lower == 0), 1, 1),
        (known & (public == 2) & (lower == 1), 1, 2),
        (known & (public == 2) & (lower == 2), 2, 3),
    ):
        status[mask], route[mask] = state, path
    status[conflicts > 0] = 4
    route[conflicts > 0] = 0
    unknown = (known & (public == 0)).astype("uint8")
    unknown |= (known & (public == 2) & (lower == 0)).astype("uint8") * 2
    unknown |= scope_unknown.astype("uint8") * 4
    quality = []
    for answer, flag, invalid in (
        (public, raw["I_HPUBLI"], public_alloc_conflict),
        (lower, raw["I_HLOREN"], lower_alloc_conflict),
    ):
        q = np.where(answer > 0, np.where(flag == 1, 2, 1), 0).astype("uint8")
        q[invalid] = 3
        quality.append(q)
    return {
        "status": status,
        "route": route,
        "receipt_valid": np.isin(status, [1, 2]).astype("uint8"),
        "group_quarters": np.where(
            raw["HRHTYPE"] == 0, 2, np.isin(raw["HRHTYPE"], [9, 10])
        ).astype("uint8"),
        "conflicts": conflicts,
        "unknown_reasons": unknown,
        "public_quality": quality[0],
        "lower_quality": quality[1],
        "HPUBLIC_valid": (public > 0).astype("uint8"),
        "HLORENT_valid": (lower > 0).astype("uint8"),
        **{
            name + "_zero_origin": (raw[name] == 0).astype("uint8")
            for name in OBSERVED_COLUMNS
        },
    }


@dataclass(frozen=True)
class _HousingStatus:
    header: bytes
    buffers: tuple[bytes, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _STATUS_TOKEN, "STATUS_CONSTRUCTOR_UNAVAILABLE")

    @property
    def header_data(self) -> dict:
        """Independent provenance copy; modifying it cannot change the artifact."""
        return _parse(self.header)

    def array(self, name: str) -> np.ndarray:
        """Return an immutable byte-backed view in declared household order."""
        if name not in COLUMNS:
            raise KeyError(name)
        return np.frombuffer(
            self.buffers[COLUMNS.index(name)],
            dtype="<i8" if name in RAW_COLUMNS else "u1",
        )

    def validate(self) -> None:
        """Recheck shape, actual classifier closure, and all derived evidence."""
        _require(
            type(self) in (SyntheticHousingStatus, AuthenticatedHousingStatus),
            "STATUS_TYPE",
        )
        data = self.header_data
        _require(
            data["definition_sha256"] == _definition_identity(), "DEFINITION_CHANGED"
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
        _require(
            data["source_authentication"]
            == (
                "checkpoint_bytes_verified"
                if type(self) is AuthenticatedHousingStatus
                else "synthetic_unverified"
            ),
            "STATUS_AUTHENTICATION",
        )
        raw = {name: self.array(name) for name in RAW_COLUMNS}
        derived = _derive(raw)
        _require(
            all(
                derived[name].tobytes() == self.buffers[COLUMNS.index(name)]
                for name in DERIVED_COLUMNS
            ),
            "DERIVED_EVIDENCE",
        )
        if type(self) is AuthenticatedHousingStatus:
            from .asec_housing_status_source import _implementation

            _require(
                data["implementation"] == _implementation(), "IMPLEMENTATION_CHANGED"
            )

    @property
    def content_sha256(self) -> str:
        """Complete header and raw/derived buffer identity, without a large join."""
        self.validate()
        digest = hashlib.sha256(b"microcosm/asec-housing-status-content/1\0")
        for part in _parts(self):
            digest.update(part)
        return digest.hexdigest()


class SyntheticHousingStatus(_HousingStatus):
    """Invented observations; never accepted by the authenticated attachment."""


class AuthenticatedHousingStatus(_HousingStatus):
    """Only the fixed-cohort source loader may issue this status evidence."""


def _issue(
    raw: pd.DataFrame, *, source: dict | None = None, implementation: dict | None = None
) -> _HousingStatus:
    _require(
        type(raw) is pd.DataFrame
        and raw.columns.is_unique
        and set(raw.columns) == set(RAW_COLUMNS),
        "RAW_COLUMNS",
    )
    _require(
        all(raw[name].dtype == np.dtype("int64") for name in RAW_COLUMNS), "RAW_INT64"
    )
    # Copy into immutable bytes before either validating values or deriving output.
    buffers = tuple(
        raw[name].to_numpy(dtype="<i8", copy=True).tobytes() for name in RAW_COLUMNS
    )
    arrays = {
        name: np.frombuffer(buf, dtype="<i8")
        for name, buf in zip(RAW_COLUMNS, buffers, strict=True)
    }
    derived = _derive(arrays)
    header = {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_housing_status_observations.v1",
        "source_authentication": "synthetic_unverified"
        if source is None
        else "checkpoint_bytes_verified",
        "release_eligible": False,
        "source_period_kind": "interview_status",
        "zero_origin_policy": ZERO_POLICY,
        "household_rows": len(raw),
        "columns": COLUMNS,
        "codebook": _codebook(),
        "dictionaries": DICTIONARIES,
        "definition_sha256": _definition_identity(),
        "source": source,
        "implementation": implementation,
        "context_columns": CONTEXT_COLUMNS,
        "non_housing_unit_scope": "not_derived",
        "zero_origin_evidence": ZERO_ORIGIN_EVIDENCE,
    }
    cls = SyntheticHousingStatus if source is None else AuthenticatedHousingStatus
    result = cls(
        _json(header),
        buffers + tuple(derived[name].tobytes() for name in DERIVED_COLUMNS),
        _token=_STATUS_TOKEN,
    )
    result.validate()
    return result


def classify_synthetic_housing_status(
    observations: pd.DataFrame,
) -> SyntheticHousingStatus:
    """Classify explicitly invented rows; this API never mints source authority."""
    return _issue(observations)


def _parts(status):
    yield MAGIC
    yield struct.pack("<I", len(status.header))
    yield status.header
    yield from status.buffers


def encode_housing_status(status: _HousingStatus) -> bytes:
    """Serialize primitive buffers with a bounded header and transport checksum."""
    _require(
        type(status) in (SyntheticHousingStatus, AuthenticatedHousingStatus),
        "STATUS_TYPE",
    )
    status.validate()
    parts = tuple(_parts(status))
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return b"".join((*parts, digest.digest()))


def decode_housing_status(
    payload: bytes, *, expected: _HousingStatus
) -> _HousingStatus:
    """Require independently issued COMPLETE expected content before replay authority."""
    _require(
        type(expected) in (SyntheticHousingStatus, AuthenticatedHousingStatus),
        "EXPECTED_STATUS",
    )
    expected.validate()
    _require(
        type(payload) is bytes
        and len(MAGIC) + 4 + 32 < len(payload) <= PAYLOAD_MAX_BYTES,
        "PAYLOAD_SIZE",
    )
    view = memoryview(payload)
    _require(payload.startswith(MAGIC), "PAYLOAD_VERSION")
    size = struct.unpack_from("<I", payload, len(MAGIC))[0]
    start = len(MAGIC) + 4
    _require(0 < size <= HEADER_MAX_BYTES, "HEADER_SIZE")
    _require(
        len(payload) == start + size + sum(map(len, expected.buffers)) + 32,
        "PAYLOAD_LENGTH",
    )
    _require(hashlib.sha256(view[:-32]).digest() == payload[-32:], "PAYLOAD_CHECKSUM")
    _require(payload[start : start + size] == expected.header, "EXPECTED_CONTENT")
    cursor = start + size
    buffers = []
    for buf in expected.buffers:
        actual = payload[cursor : cursor + len(buf)]
        _require(actual == buf, "EXPECTED_CONTENT")
        buffers.append(actual)
        cursor += len(buf)
    return type(expected)(expected.header, tuple(buffers), _token=_STATUS_TOKEN)


def write_housing_status(status: _HousingStatus, path: str | Path) -> str:
    """Write an artifact and return its complete-content identity."""
    Path(path).write_bytes(encode_housing_status(status))
    return status.content_sha256


def read_housing_status(
    path: str | Path, *, expected: _HousingStatus
) -> _HousingStatus:
    """Bound disk reads and require independently source-issued expected content."""
    with Path(path).open("rb") as handle:
        payload = handle.read(PAYLOAD_MAX_BYTES + 1)
    return decode_housing_status(payload, expected=expected)
