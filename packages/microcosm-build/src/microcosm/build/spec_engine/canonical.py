"""Schema-directed normalization, surface projection, and identity bytes."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping

from jsonschema import Draft202012Validator

from .model import FrozenMap, FrozenValue, Surface, freeze_json, thaw_json

CANONICALIZER_ID = "microcosm-json-v1"
CANONICALIZER_VERSION = 1
SPEC_DOMAIN = "microcosm.spec-engine.resolved-spec.v1"
DOCUMENTATION_DOMAIN = "microcosm.spec-engine.documentation.v1"

_ABSENT = object()


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON value with the F0 canonical JSON grammar."""

    if isinstance(value, (FrozenMap, tuple)):
        value = thaw_json(value)  # type: ignore[arg-type]
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalize_scalar(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not canonical JSON")
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def _schema_documents(registry: object) -> Mapping[str, Mapping[str, object]]:
    for attribute in ("schemas", "documents", "schema_documents"):
        candidate = getattr(registry, attribute, None)
        if isinstance(candidate, Mapping):
            return candidate
    getter = getattr(registry, "all_schemas", None)
    if getter is not None:
        candidate = getter()
        if isinstance(candidate, Mapping):
            return candidate
    raise TypeError("SchemaRegistry does not expose its schema documents")


def _json_pointer(document: object, pointer: str) -> Mapping[str, object]:
    current = document
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError(f"unsupported schema fragment #{pointer}")
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or token not in current:
                raise ValueError(f"unknown schema fragment #{pointer}")
            current = current[token]
    if not isinstance(current, Mapping):
        raise ValueError(f"schema fragment #{pointer} is not an object")
    return current


def _resolve_ref(
    ref: str,
    *,
    base_schema_id: str,
    registry: object,
) -> tuple[Mapping[str, object], str]:
    if "#" in ref:
        document_id, pointer = ref.split("#", 1)
    else:
        document_id, pointer = ref, ""
    selected_id = document_id or base_schema_id
    documents = _schema_documents(registry)
    try:
        document = documents[selected_id]
    except KeyError as error:
        raise ValueError(f"unknown schema reference {ref!r}") from error
    return _json_pointer(document, pointer), selected_id


def _branch_valid(
    value: object,
    branch: Mapping[str, object],
    *,
    base_schema_id: str,
    registry: object,
) -> bool:
    # Branch-local references need an explicit base document when evaluated
    # outside the root validator.  Resolving them here avoids treating two
    # failed ``$ref`` evaluations as two matches in the conservative fallback.
    while "$ref" in branch:
        branch, base_schema_id = _resolve_ref(
            str(branch["$ref"]),
            base_schema_id=base_schema_id,
            registry=registry,
        )
    checker = getattr(registry, "is_valid", None)
    if checker is not None:
        try:
            return bool(checker(value, branch, base_schema_id=base_schema_id))
        except TypeError:
            pass
    ref_registry = getattr(registry, "registry", None)
    try:
        return Draft202012Validator(branch, registry=ref_registry).is_valid(value)
    except Exception:
        # The full resource was already validated.  This conservative fallback
        # is used only with alternate SchemaRegistry implementations.
        if "const" in branch:
            return value == branch["const"]
        required = branch.get("required", ())
        return isinstance(value, Mapping) and all(key in value for key in required)


def _effective_schema(
    value: object,
    schema: Mapping[str, object],
    *,
    base_schema_id: str,
    registry: object,
) -> tuple[Mapping[str, object], str]:
    seen: set[tuple[str, str]] = set()
    while "$ref" in schema:
        ref = str(schema["$ref"])
        marker = (base_schema_id, ref)
        if marker in seen:
            raise ValueError(f"cyclic schema reference {ref!r}")
        seen.add(marker)
        schema, base_schema_id = _resolve_ref(
            ref, base_schema_id=base_schema_id, registry=registry
        )
    if "oneOf" in schema:
        branches = schema["oneOf"]
        if not isinstance(branches, list):
            raise TypeError("oneOf must be an array")
        matches = [
            branch
            for branch in branches
            if isinstance(branch, Mapping)
            and _branch_valid(
                value,
                branch,
                base_schema_id=base_schema_id,
                registry=registry,
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"oneOf matched {len(matches)} branches")
        branch = dict(matches[0])
        inherited = {key: child for key, child in schema.items() if key != "oneOf"}
        inherited.update(branch)
        schema = inherited
        return _effective_schema(
            value, schema, base_schema_id=base_schema_id, registry=registry
        )
    return schema, base_schema_id


def normalize_and_project(
    value: object,
    *,
    schema_id: str,
    registry: object,
) -> tuple[FrozenValue, FrozenMap]:
    """Normalize a validated value and split it into physical surfaces.

    ``x-spec-surface`` is inherited.  Every authored schema has a normative
    root annotation, with explicit overrides for documentation and operational
    bindings.  ``x-canonical-order: set`` is the sole mechanism that permits
    array reordering.
    """

    documents = _schema_documents(registry)
    try:
        schema = documents[schema_id]
    except KeyError as error:
        raise ValueError(f"unknown schema id {schema_id!r}") from error
    normalized, projected = _walk(
        value,
        schema,
        base_schema_id=schema_id,
        registry=registry,
        inherited_surface=Surface.NORMATIVE,
    )
    projections = {
        surface.value: fragment
        for surface, fragment in projected.items()
        if fragment is not _ABSENT
    }
    for surface in Surface:
        projections.setdefault(surface.value, {})
    return freeze_json(normalized), freeze_json(projections)  # type: ignore[arg-type]


def _walk(
    value: object,
    schema: Mapping[str, object],
    *,
    base_schema_id: str,
    registry: object,
    inherited_surface: Surface,
) -> tuple[object, dict[Surface, object]]:
    schema, base_schema_id = _effective_schema(
        value, schema, base_schema_id=base_schema_id, registry=registry
    )
    surface = Surface(str(schema.get("x-spec-surface", inherited_surface.value)))

    if isinstance(value, Mapping):
        normalized_object: dict[str, object] = {}
        surface_objects: dict[Surface, dict[str, object]] = {
            candidate: {} for candidate in Surface
        }
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            child_schema = properties.get(key, additional)
            if child_schema is True:
                child_schema = {}
            if child_schema is False or not isinstance(child_schema, Mapping):
                child_schema = {}
            normalized_child, projected_child = _walk(
                value[key],
                child_schema,
                base_schema_id=base_schema_id,
                registry=registry,
                inherited_surface=surface,
            )
            normalized_object[key] = normalized_child
            for candidate, fragment in projected_child.items():
                if fragment is not _ABSENT:
                    surface_objects[candidate][key] = fragment
        return normalized_object, {
            candidate: child if child else _ABSENT
            for candidate, child in surface_objects.items()
        }

    if isinstance(value, list | tuple):
        item_schema = schema.get("items", {})
        if not isinstance(item_schema, Mapping):
            item_schema = {}
        rows: list[tuple[object, dict[Surface, object]]] = [
            _walk(
                child,
                item_schema,
                base_schema_id=base_schema_id,
                registry=registry,
                inherited_surface=surface,
            )
            for child in value
        ]
        if schema.get("x-canonical-order") == "set":
            rows.sort(key=lambda row: canonical_json_bytes(row[0]))
            encoded = [canonical_json_bytes(row[0]) for row in rows]
            if len(encoded) != len(set(encoded)):
                raise ValueError("schema-declared set contains duplicate members")
        normalized_array = [row[0] for row in rows]
        projected_arrays: dict[Surface, object] = {}
        for candidate in Surface:
            fragments = [row[1].get(candidate, _ABSENT) for row in rows]
            if all(fragment is _ABSENT for fragment in fragments):
                projected_arrays[candidate] = _ABSENT
                continue
            projected_arrays[candidate] = [
                {} if fragment is _ABSENT else fragment for fragment in fragments
            ]
        return normalized_array, projected_arrays

    normalized_scalar = _normalize_scalar(value)
    return normalized_scalar, {
        candidate: normalized_scalar if candidate is surface else _ABSENT
        for candidate in Surface
    }


def spec_envelope(
    *,
    country: str,
    schema_version: int,
    normative_files: Mapping[str, object],
    resolved_bindings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    envelope: dict[str, object] = {
        "domain": SPEC_DOMAIN,
        "canonicalizer": {
            "id": CANONICALIZER_ID,
            "version": CANONICALIZER_VERSION,
        },
        "schema": {"id": "country_spec", "version": schema_version},
        "country": country,
        "files": dict(normative_files),
    }
    if resolved_bindings is not None:
        envelope["resolved_bindings"] = dict(resolved_bindings)
    return envelope


def documentation_envelope(
    *,
    country: str,
    schema_version: int,
    documentation_files: Mapping[str, object],
) -> dict[str, object]:
    return {
        "domain": DOCUMENTATION_DOMAIN,
        "canonicalizer": {
            "id": CANONICALIZER_ID,
            "version": CANONICALIZER_VERSION,
        },
        "schema": {"id": "country_spec", "version": schema_version},
        "country": country,
        "files": dict(documentation_files),
    }
