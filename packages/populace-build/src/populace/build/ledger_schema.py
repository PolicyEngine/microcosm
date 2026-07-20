"""Deterministic, stdlib-only validation of Ledger consumer fact rows.

Populace pins the PolicyEngine Ledger consumer-fact contract by vendoring the
published JSON Schema (``schemas/consumer_fact.v1.schema.json``) and validating
every consumed row against it before a fact can resolve a calibration target
(finding #8: nominal row validation). Structurally incomplete or mistyped rows
fail at load with a precise field path rather than silently compiling into a
wrong target.

Like the rest of Populace's Ledger consumption this module is stdlib-only and
does not import the Ledger implementation package. It implements exactly the
JSON Schema (draft 2020-12) constructs the vendored schema uses --- ``type``,
``const``, ``enum``, ``pattern``, ``required``, ``properties``,
``additionalProperties`` (boolean or subschema), ``items``, and ``$ref`` into
``$defs`` --- and raises on any construct it does not implement, so a future
schema revision that introduces an unhandled keyword fails loudly at import
instead of silently passing rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib.resources import files
from typing import Any

__all__ = [
    "CONSUMER_FACT_SCHEMA_SHA256",
    "CONSUMER_FACT_SCHEMA_VERSION",
    "validate_consumer_fact_row",
]

CONSUMER_FACT_SCHEMA_VERSION = "ledger.consumer_fact.v1"

_SCHEMA_BYTES = (
    files("populace.build.schemas")
    .joinpath("consumer_fact.v1.schema.json")
    .read_bytes()
)
CONSUMER_FACT_SCHEMA_SHA256 = hashlib.sha256(_SCHEMA_BYTES).hexdigest()
_SCHEMA: dict[str, Any] = json.loads(_SCHEMA_BYTES)
_DEFS: dict[str, Any] = _SCHEMA.get("$defs", {})

# JSON Schema keywords this validator implements. Any other keyword appearing in
# the vendored schema is a construct we do not enforce; we raise on it so the
# pin cannot silently weaken when the schema evolves.
_SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "const",
        "enum",
        "pattern",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "$ref",
    }
)
# Annotation / structural keywords that carry no row-level validation semantics.
_ANNOTATION_KEYWORDS = frozenset(
    {"$schema", "$id", "$defs", "title", "description"}
)

_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    # JSON has one number type; booleans are a distinct JSON type and must not
    # satisfy integer/number even though Python bool subclasses int.
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float))
    and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


class _UnsupportedSchemaError(RuntimeError):
    """The vendored schema uses a construct this validator does not implement."""


def _assert_supported(schema: Any) -> None:
    """Walk the schema once at import so unhandled keywords fail loudly."""
    if not isinstance(schema, dict):
        raise _UnsupportedSchemaError(
            f"Consumer-fact schema node must be an object, got {type(schema).__name__}."
        )
    for keyword in schema:
        if keyword in _SUPPORTED_KEYWORDS or keyword in _ANNOTATION_KEYWORDS:
            continue
        raise _UnsupportedSchemaError(
            f"Consumer-fact schema uses unimplemented keyword {keyword!r}; the "
            "vendored stdlib validator must be extended to enforce it before "
            "this schema revision can be pinned."
        )
    type_value = schema.get("type")
    if type_value is not None:
        types = type_value if isinstance(type_value, list) else [type_value]
        for name in types:
            if name not in _TYPE_CHECKS:
                raise _UnsupportedSchemaError(
                    f"Consumer-fact schema uses unimplemented type {name!r}."
                )
    for subschema in schema.get("properties", {}).values():
        _assert_supported(subschema)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _assert_supported(additional)
    elif additional is not None and not isinstance(additional, bool):
        raise _UnsupportedSchemaError(
            "Consumer-fact schema additionalProperties must be a boolean or a "
            f"subschema, got {type(additional).__name__}."
        )
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_supported(items)
    elif items is not None:
        raise _UnsupportedSchemaError(
            "Consumer-fact schema items must be a single subschema, got "
            f"{type(items).__name__}."
        )
    ref = schema.get("$ref")
    if ref is not None and not (isinstance(ref, str) and ref.startswith("#/$defs/")):
        raise _UnsupportedSchemaError(
            f"Consumer-fact schema $ref must target #/$defs/, got {ref!r}."
        )


for _definition in _DEFS.values():
    _assert_supported(_definition)
_assert_supported(_SCHEMA)


def _resolve_ref(ref: str) -> dict[str, Any]:
    name = ref[len("#/$defs/") :]
    target = _DEFS.get(name)
    if not isinstance(target, dict):
        raise _UnsupportedSchemaError(
            f"Consumer-fact schema $ref {ref!r} does not resolve to a definition."
        )
    return target


def _type_names(schema: dict[str, Any]) -> str:
    type_value = schema["type"]
    types = type_value if isinstance(type_value, list) else [type_value]
    return " | ".join(str(name) for name in types)


def _matches_type(value: Any, schema: dict[str, Any]) -> bool:
    type_value = schema["type"]
    types = type_value if isinstance(type_value, list) else [type_value]
    return any(_TYPE_CHECKS[name](value) for name in types)


def _validate(value: Any, schema: dict[str, Any], path: str, *, prefix: str) -> None:
    ref = schema.get("$ref")
    if ref is not None:
        # The vendored schema only uses $ref as a sole-key subschema; a sibling
        # validation keyword would be a combination we do not enforce.
        siblings = set(schema) - {"$ref"} - _ANNOTATION_KEYWORDS
        if siblings:
            raise _UnsupportedSchemaError(
                f"Consumer-fact schema $ref combined with unimplemented keywords "
                f"{sorted(siblings)!r}."
            )
        _validate(value, _resolve_ref(ref), path, prefix=prefix)
        return

    where = f"{prefix} field {path!r}" if path else prefix

    # A non-finite float (NaN / Infinity / -Infinity) satisfies a bare
    # ``number`` type check but is not a valid contract value (finding #7):
    # ``json.loads`` accepts the JS constants by default, so reject them on
    # every numeric regardless of where they appear in the row.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: number {value!r} is not finite.")

    if "type" in schema and not _matches_type(value, schema):
        raise ValueError(
            f"{where}: expected type {_type_names(schema)}, got "
            f"{_json_type_name(value)}."
        )
    if "const" in schema and value != schema["const"]:
        raise ValueError(
            f"{where}: expected constant {schema['const']!r}, got {value!r}."
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(
            f"{where}: {value!r} is not one of {schema['enum']!r}."
        )
    if "pattern" in schema and isinstance(value, str):
        if re.search(schema["pattern"], value) is None:
            raise ValueError(
                f"{where}: {value!r} does not match pattern {schema['pattern']!r}."
            )

    if isinstance(value, dict) and (
        "properties" in schema or "required" in schema or "additionalProperties" in schema
    ):
        _validate_object(value, schema, path, prefix=prefix)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            _validate(item, schema["items"], item_path, prefix=prefix)


def _validate_object(
    value: dict[str, Any], schema: dict[str, Any], path: str, *, prefix: str
) -> None:
    properties = schema.get("properties", {})
    for required in schema.get("required", ()):
        if required not in value:
            child = f"{path}.{required}" if path else required
            raise ValueError(
                f"{prefix} field {child!r} is required but missing."
            )
    additional = schema.get("additionalProperties", True)
    for key, child_value in value.items():
        child = f"{path}.{key}" if path else key
        if key in properties:
            _validate(child_value, properties[key], child, prefix=prefix)
        elif additional is False:
            raise ValueError(
                f"{prefix} field {child!r} is not an allowed property."
            )
        elif isinstance(additional, dict):
            _validate(child_value, additional, child, prefix=prefix)


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def validate_consumer_fact_row(
    row: Any,
    *,
    line_number: int,
    source: str,
) -> None:
    """Validate one consumer fact row against the pinned contract schema.

    Raises ``ValueError`` with a precise field path when the row is not a valid
    ``ledger.consumer_fact.v1`` row.
    """
    prefix = f"{source} line {line_number}"
    if not isinstance(row, dict):
        raise ValueError(
            f"{prefix}: expected a JSON object, got {_json_type_name(row)}."
        )
    _validate(row, _SCHEMA, "", prefix=prefix)
