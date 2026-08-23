"""Canonical ``plan.lock.json`` emission and reproducibility assertions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes
from .compiler_ir import CompiledSpecIR, compile_spec
from .model import ResolvedSpec
from .schemas import SchemaRegistry, load_schema_registry

PLAN_LOCK_SCHEMA_ID = "locks.schema.json#/$defs/plan_lock"


class PlanLockError(ValueError):
    """A plan lock is malformed, noncanonical, or differs from its compiler."""


def _compiled(value: CompiledSpecIR | ResolvedSpec) -> CompiledSpecIR:
    if isinstance(value, CompiledSpecIR):
        return value
    if isinstance(value, ResolvedSpec):
        return compile_spec(value)
    raise TypeError("plan lock functions require CompiledSpecIR or ResolvedSpec")


def plan_lock_payload(
    value: CompiledSpecIR | ResolvedSpec,
    *,
    schema_registry: SchemaRegistry | None = None,
) -> dict[str, object]:
    """Return and validate the complete emitted plan-lock projection."""

    payload = _compiled(value).to_wire()
    (schema_registry or load_schema_registry()).validate(
        payload, PLAN_LOCK_SCHEMA_ID
    )
    return payload


def plan_lock_bytes(
    value: CompiledSpecIR | ResolvedSpec,
    *,
    schema_registry: SchemaRegistry | None = None,
) -> bytes:
    return canonical_json_bytes(
        plan_lock_payload(value, schema_registry=schema_registry)
    )


def emit_plan_lock(
    value: CompiledSpecIR | ResolvedSpec,
    path: str | Path,
    *,
    schema_registry: SchemaRegistry | None = None,
) -> Path:
    """Write the sole canonical byte representation of ``plan.lock.json``."""

    destination = Path(path)
    destination.write_bytes(
        plan_lock_bytes(value, schema_registry=schema_registry)
    )
    return destination


def _strict_json_object(raw: bytes, *, source: str) -> Mapping[str, object]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PlanLockError(f"{source}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PlanLockError(f"{source}: non-finite JSON number {value!r}")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except PlanLockError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanLockError(f"{source}: invalid UTF-8 JSON: {error}") from error
    if not isinstance(parsed, Mapping):
        raise PlanLockError(f"{source}: object root required")
    return parsed


def _pointer(path: str, token: object) -> str:
    encoded = str(token).replace("~", "~0").replace("/", "~1")
    return f"{path}/{encoded}" if path else f"/{encoded}"


def _diff(
    expected: object,
    observed: object,
    *,
    path: str = "",
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        rows: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            location = _pointer(path, key)
            if key not in observed:
                rows.append(f"{location}: missing")
            elif key not in expected:
                rows.append(f"{location}: unexpected")
            else:
                rows.extend(_diff(expected[key], observed[key], path=location))
        return rows
    if (
        isinstance(expected, Sequence)
        and not isinstance(expected, str | bytes)
        and isinstance(observed, Sequence)
        and not isinstance(observed, str | bytes)
    ):
        rows = []
        common = min(len(expected), len(observed))
        for index in range(common):
            rows.extend(
                _diff(
                    expected[index],
                    observed[index],
                    path=_pointer(path, index),
                )
            )
        for index in range(common, len(expected)):
            rows.append(f"{_pointer(path, index)}: missing")
        for index in range(common, len(observed)):
            rows.append(f"{_pointer(path, index)}: unexpected")
        return rows
    if expected != observed or type(expected) is not type(observed):
        return [f"{path or '/'}: expected={expected!r}, observed={observed!r}"]
    return []


def assert_plan_lock_payload_current(
    value: CompiledSpecIR | ResolvedSpec,
    observed: Mapping[str, object],
    *,
    schema_registry: SchemaRegistry | None = None,
) -> None:
    """Refuse a parsed lock that is not exactly reproducible from its spec."""

    registry = schema_registry or load_schema_registry()
    registry.validate(observed, PLAN_LOCK_SCHEMA_ID)
    expected = plan_lock_payload(value, schema_registry=registry)
    differences = _diff(expected, observed)
    if differences:
        detail = "\n".join(f"  {row}" for row in differences)
        raise PlanLockError(
            f"plan lock differs from compiler output ({len(differences)} field(s))\n"
            f"{detail}"
        )


def assert_plan_lock_current(
    value: CompiledSpecIR | ResolvedSpec,
    path: str | Path,
    *,
    schema_registry: SchemaRegistry | None = None,
) -> Path:
    """Validate exact content and canonical bytes of an emitted plan lock."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise PlanLockError(f"unable to read plan lock {source}: {error}") from error
    observed = _strict_json_object(raw, source=str(source))
    registry = schema_registry or load_schema_registry()
    assert_plan_lock_payload_current(
        value,
        observed,
        schema_registry=registry,
    )
    canonical = canonical_json_bytes(observed)
    if raw != canonical:
        raise PlanLockError(
            f"{source}: plan lock JSON is not in the canonical emitted byte form"
        )
    return source


__all__ = [
    "PLAN_LOCK_SCHEMA_ID",
    "PlanLockError",
    "assert_plan_lock_current",
    "assert_plan_lock_payload_current",
    "emit_plan_lock",
    "plan_lock_bytes",
    "plan_lock_payload",
]
