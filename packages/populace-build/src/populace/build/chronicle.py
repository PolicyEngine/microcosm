"""Local-only Chronicle spool rows compatible with the #628 build ledger.

This module is the direct-write fallback for build tools whose branch base
does not yet contain :mod:`populace.build.ledger`. It implements the exact
17-field row schema and hash contract from populace#628, but deliberately has
no remote client: a successful call validates one terminal build-attempt row
and durably writes ``<row_digest>.json`` beneath the caller's spool directory.

The caller owns chain coordination and must provide ``prev_row_digest`` for
the ledger head it extends. The row digest is

``sha256(canonical JSON of all non-chain fields || prev_row_digest)``

where the predecessor is lowercase ASCII, or the empty string for genesis.
Local validation and durable-spool failures are fatal. Writes use the house
pattern: file fsync, atomic rename, then containing-directory fsync.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

__all__ = [
    "BUILD_DISPOSITIONS",
    "CHRONICLE_ROW_FIELDS",
    "CHRONICLE_RUNGS",
    "ChronicleRow",
    "ChronicleWriteResult",
    "canonical_json_bytes",
    "compute_row_digest",
    "load_chronicle_row",
    "record_build_attempt",
]


BUILD_DISPOSITIONS = frozenset(
    {
        "iterating",
        "billed",
        "published",
        "certified",
        "failed",
        "superseded",
        "discarded",
    }
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BUILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
CHRONICLE_RUNGS = frozenset({"f001", "f010", "f100"})
CHRONICLE_ROW_FIELDS = frozenset(
    {
        "build_id",
        "ts",
        "pipeline",
        "rung",
        "seed",
        "code_pin",
        "input_pins_digest",
        "identity_digest",
        "phases_reached",
        "gate_verdicts",
        "wall_seconds",
        "cost_usd",
        "artifact_location",
        "disposition",
        "prediction_id",
        "prev_row_digest",
        "row_digest",
    }
)
_HASH_EXCLUDED_FIELDS = frozenset({"prev_row_digest", "row_digest"})


@dataclass(frozen=True)
class ChronicleRow:
    """A validated terminal build-attempt row with a verified chain digest."""

    build_id: str
    ts: str
    pipeline: str
    rung: str
    seed: int | None
    code_pin: str
    input_pins_digest: str
    identity_digest: str
    phases_reached: tuple[str, ...]
    gate_verdicts: dict[str, Any]
    wall_seconds: int | float | None
    cost_usd: int | float | None
    artifact_location: str | None
    disposition: str
    prediction_id: str | None
    prev_row_digest: str | None
    row_digest: str

    @classmethod
    def create(
        cls,
        *,
        build_id: str,
        ts: str | datetime,
        pipeline: str,
        rung: str,
        seed: int | None,
        code_pin: str,
        input_pins_digest: str,
        identity_digest: str,
        phases_reached: Sequence[str],
        gate_verdicts: Mapping[str, Any],
        wall_seconds: int | float | None,
        cost_usd: int | float | None,
        artifact_location: str | None,
        disposition: str,
        prediction_id: str | None,
        prev_row_digest: str | None,
        row_digest: str | None = None,
    ) -> ChronicleRow:
        """Validate, normalize, and hash a complete attempt receipt."""

        normalized_build_id = _nonempty_text(build_id, "build_id")
        if not _BUILD_ID_PATTERN.fullmatch(normalized_build_id):
            raise ValueError(
                "build_id must use only letters, digits, '.', '_', ':', or '-'."
            )
        normalized_pipeline = _nonempty_text(pipeline, "pipeline")
        normalized_rung = _validate_rung(rung)
        normalized_seed = _validate_seed(seed)
        normalized_code_pin = _nonempty_text(code_pin, "code_pin")
        normalized_input_digest = _validate_digest(
            input_pins_digest,
            "input_pins_digest",
            nullable=False,
        )
        normalized_identity_digest = _validate_digest(
            identity_digest,
            "identity_digest",
            nullable=False,
        )
        normalized_phases = _validate_phases(phases_reached)
        normalized_gates = _validate_gate_verdicts(gate_verdicts)
        normalized_wall = _validate_nonnegative_number(wall_seconds, "wall_seconds")
        normalized_cost = _validate_nonnegative_number(cost_usd, "cost_usd")
        normalized_artifact = _optional_text(artifact_location, "artifact_location")
        if disposition not in BUILD_DISPOSITIONS:
            raise ValueError(
                f"disposition must be one of {sorted(BUILD_DISPOSITIONS)}, "
                f"got {disposition!r}."
            )
        if disposition in {"published", "certified"} and normalized_artifact is None:
            raise ValueError(
                f"artifact_location is required for disposition {disposition!r}."
            )
        normalized_prediction = _optional_text(prediction_id, "prediction_id")
        normalized_prev = _validate_digest(
            prev_row_digest,
            "prev_row_digest",
            nullable=True,
        )

        values: dict[str, Any] = {
            "build_id": normalized_build_id,
            "ts": _normalize_timestamp(ts, "ts"),
            "pipeline": normalized_pipeline,
            "rung": normalized_rung,
            "seed": normalized_seed,
            "code_pin": normalized_code_pin,
            "input_pins_digest": normalized_input_digest,
            "identity_digest": normalized_identity_digest,
            "phases_reached": list(normalized_phases),
            "gate_verdicts": normalized_gates,
            "wall_seconds": normalized_wall,
            "cost_usd": normalized_cost,
            "artifact_location": normalized_artifact,
            "disposition": disposition,
            "prediction_id": normalized_prediction,
            "prev_row_digest": normalized_prev,
        }
        calculated = compute_row_digest(values)
        if row_digest is not None:
            supplied = _validate_digest(row_digest, "row_digest", nullable=False)
            if supplied != calculated:
                raise ValueError(
                    f"row_digest for build {normalized_build_id!r} does not match "
                    f"the canonical row: {supplied} != {calculated}."
                )

        return cls(
            build_id=normalized_build_id,
            ts=values["ts"],
            pipeline=normalized_pipeline,
            rung=normalized_rung,
            seed=normalized_seed,
            code_pin=normalized_code_pin,
            input_pins_digest=normalized_input_digest,
            identity_digest=normalized_identity_digest,
            phases_reached=normalized_phases,
            gate_verdicts=normalized_gates,
            wall_seconds=normalized_wall,
            cost_usd=normalized_cost,
            artifact_location=normalized_artifact,
            disposition=disposition,
            prediction_id=normalized_prediction,
            prev_row_digest=normalized_prev,
            row_digest=calculated,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ChronicleRow:
        """Load a row object, rejecting missing, extra, or tampered fields."""

        if not isinstance(value, Mapping):
            raise ValueError(
                f"Chronicle row must be an object, got {type(value).__name__}."
            )
        keys = frozenset(value)
        if keys != CHRONICLE_ROW_FIELDS:
            missing = sorted(CHRONICLE_ROW_FIELDS - keys)
            extra = sorted(keys - CHRONICLE_ROW_FIELDS)
            raise ValueError(
                f"Chronicle row schema mismatch; missing={missing}, extra={extra}."
            )
        return cls.create(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        """Return the normalized JSON row in #628 database-column order."""

        return {
            "build_id": self.build_id,
            "ts": self.ts,
            "pipeline": self.pipeline,
            "rung": self.rung,
            "seed": self.seed,
            "code_pin": self.code_pin,
            "input_pins_digest": self.input_pins_digest,
            "identity_digest": self.identity_digest,
            "phases_reached": list(self.phases_reached),
            "gate_verdicts": deepcopy(self.gate_verdicts),
            "wall_seconds": self.wall_seconds,
            "cost_usd": self.cost_usd,
            "artifact_location": self.artifact_location,
            "disposition": self.disposition,
            "prediction_id": self.prediction_id,
            "prev_row_digest": self.prev_row_digest,
            "row_digest": self.row_digest,
        }

    def to_json_line(self) -> str:
        """Serialize one portable compact JSON row."""

        return (
            json.dumps(
                self.to_mapping(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True)
class ChronicleWriteResult:
    """Receipt for one durable local Chronicle spool write."""

    row: ChronicleRow
    spool_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON using #628's deterministic Python/PostgreSQL form.

    Object keys are lexicographically sorted, whitespace is omitted, Unicode
    remains UTF-8, and every number is rendered in plain base-10 form without
    insignificant trailing zeroes. Booleans remain distinct from integers.
    """

    try:
        return _canonical_json_text(value).encode("utf-8")
    except (UnicodeEncodeError, InvalidOperation) as exc:
        raise ValueError(f"Value is not canonical JSON: {exc}.") from exc


def compute_row_digest(value: Mapping[str, Any]) -> str:
    """Compute SHA-256(canonical non-chain fields || predecessor)."""

    missing = (CHRONICLE_ROW_FIELDS - {"row_digest"}) - frozenset(value)
    if missing:
        raise ValueError(
            f"Cannot hash Chronicle row; missing fields: {sorted(missing)}."
        )
    predecessor = _validate_digest(
        value.get("prev_row_digest"),
        "prev_row_digest",
        nullable=True,
    )
    payload = {
        key: value[key] for key in sorted(value) if key not in _HASH_EXCLUDED_FIELDS
    }
    material = canonical_json_bytes(payload) + (predecessor or "").encode("ascii")
    return hashlib.sha256(material).hexdigest()


def record_build_attempt(
    *,
    build_id: str,
    ts: str | datetime,
    pipeline: str,
    rung: str,
    seed: int | None,
    code_pin: str,
    input_pins_digest: str,
    identity_digest: str,
    phases_reached: Sequence[str],
    gate_verdicts: Mapping[str, Any],
    wall_seconds: int | float | None,
    cost_usd: int | float | None,
    artifact_location: str | None,
    disposition: str,
    prediction_id: str | None,
    prev_row_digest: str | None,
    row_digest: str | None = None,
    spool_dir: str | Path = "ledger-spool",
) -> ChronicleWriteResult:
    """Validate and durably spool one terminal attempt without network I/O."""

    row = ChronicleRow.create(
        build_id=build_id,
        ts=ts,
        pipeline=pipeline,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        input_pins_digest=input_pins_digest,
        identity_digest=identity_digest,
        phases_reached=phases_reached,
        gate_verdicts=gate_verdicts,
        wall_seconds=wall_seconds,
        cost_usd=cost_usd,
        artifact_location=artifact_location,
        disposition=disposition,
        prediction_id=prediction_id,
        prev_row_digest=prev_row_digest,
        row_digest=row_digest,
    )
    spool_path = Path(spool_dir) / f"{row.row_digest}.json"
    _atomic_write_row(spool_path, row)
    return ChronicleWriteResult(row=row, spool_path=spool_path)


def load_chronicle_row(path: str | Path) -> ChronicleRow:
    """Load one completed spool row and authenticate its digest filename."""

    spool_path = Path(path)
    try:
        value = json.loads(spool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Chronicle spool row {spool_path}: {exc}.") from exc
    row = ChronicleRow.from_mapping(value)
    if spool_path.suffix != ".json" or spool_path.stem != row.row_digest:
        raise ValueError(
            "Chronicle spool filename does not match row_digest for "
            f"{row.build_id}: {spool_path.name}."
        )
    return row


def _canonical_json_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON cannot contain non-finite numbers.")
        return _canonical_decimal(Decimal(str(value)))
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(
                "Canonical JSON cannot contain NUL characters; PostgreSQL "
                "text and jsonb cannot store them."
            )
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON object keys must be strings.")
        return (
            "{"
            + ",".join(
                f"{_canonical_json_text(key)}:{_canonical_json_text(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json_text(item) for item in value) + "]"
    raise ValueError(
        f"Canonical JSON does not support values of type {type(value).__name__}."
    )


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("Canonical JSON cannot contain non-finite numbers.")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _normalize_timestamp(value: str | datetime, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{field} must be an ISO-8601 timestamp: {value!r}."
            ) from exc
    else:
        raise ValueError(
            f"{field} must be an ISO-8601 string or datetime, got "
            f"{type(value).__name__}."
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset.")
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validate_rung(value: str) -> str:
    if value not in CHRONICLE_RUNGS:
        raise ValueError(
            f"rung must be one of {sorted(CHRONICLE_RUNGS)}, got {value!r}."
        )
    return value


def _validate_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        raise ValueError(
            f"seed must be a non-negative signed 64-bit integer or null, got {value!r}."
        )
    return value


def _validate_phases(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("phases_reached must be a non-empty array of phase names.")
    phases = tuple(_nonempty_text(item, "phases_reached item") for item in value)
    if not phases:
        raise ValueError("phases_reached must contain at least one phase.")
    if len(set(phases)) != len(phases):
        raise ValueError("phases_reached must not contain duplicate phase names.")
    return phases


def _validate_gate_verdicts(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("gate_verdicts must be a non-empty JSON object.")
    normalized = _normalize_json_value(value)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise AssertionError("gate_verdicts normalization lost its object shape")
    for gate, receipt in normalized.items():
        _nonempty_text(gate, "gate_verdicts gate name")
        if not isinstance(receipt, Mapping):
            raise ValueError(f"gate_verdicts[{gate!r}] must be an object.")
        _nonempty_text(receipt.get("verdict"), f"gate_verdicts[{gate!r}].verdict")
        _nonempty_text(receipt.get("receipt"), f"gate_verdicts[{gate!r}].receipt")
    canonical_json_bytes(normalized)
    return normalized


def _normalize_json_value(value: Any) -> Any:
    """Return JSON containers with tuples and mappings normalized for retries."""

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON object keys must be strings.")
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(
        f"Canonical JSON does not support values of type {type(value).__name__}."
    )


def _validate_nonnegative_number(
    value: int | float | None,
    field: str,
) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative JSON number or null.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative, got {value!r}.")
    return value


def _validate_digest(
    value: Any,
    field: str,
    *,
    nullable: bool,
) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        suffix = " or null" if nullable else ""
        raise ValueError(f"{field} must be a lowercase SHA-256 digest{suffix}.")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL characters.")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace.")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, field)


def _atomic_write_row(path: Path, row: ChronicleRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_chronicle_row(path)
        if existing != row:
            raise ValueError(
                f"Chronicle spool digest collision or divergent retry at {path}."
            )
        # Complete a possibly interrupted replacement whose file became
        # visible before the parent-directory fsync succeeded.
        _fsync_file_and_parent(path)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(row.to_json_line())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file_and_parent(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_parent_directory(path.parent)


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
