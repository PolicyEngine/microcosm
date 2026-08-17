"""Closed-world JSON Schema registry for country specification bundles.

The registry deliberately has no filesystem or network retrieval fallback.
Every reference must resolve inside the fifteen schemas shipped with the
compiler.  Keeping that boundary here makes both authored resources and
emitted lock fragments use the same draft-2020-12 implementation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource

from .errors import SpecValidationError

try:  # ``SpecSchemaError`` is the preferred, more specific catalog error.
    from .errors import SpecSchemaError
except ImportError:  # pragma: no cover - compatibility with an early seam.
    SpecSchemaError = SpecValidationError  # type: ignore[misc, assignment]


DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"

SCHEMA_FILENAMES = (
    "battery.schema.json",
    "bundle.schema.json",
    "calibration.schema.json",
    "catalogs.schema.json",
    "defs.schema.json",
    "geography.schema.json",
    "imputation.schema.json",
    "locks.schema.json",
    "publication.schema.json",
    "resource_manifest.schema.json",
    "selection.schema.json",
    "sources.schema.json",
    "spine.schema.json",
    "take_up.schema.json",
    "vintages.schema.json",
)
SCHEMA_IDS = frozenset(SCHEMA_FILENAMES)

# Resource kinds with a single schema authority.  ``schema`` is handled by
# ``SCHEMA_IDS_BY_KIND`` because each of the fifteen schema files is a valid
# resource of that kind.  ``legacy_json`` is an explicit generation-0
# compatibility sentinel and is never passed to the draft-2020 validator.
SCHEMA_ID_BY_KIND: Mapping[str, str] = MappingProxyType(
    {
        "bundle": "bundle.schema.json",
        "sources": "sources.schema.json",
        "spine": "spine.schema.json",
        "geography": "geography.schema.json",
        "imputation": "imputation.schema.json",
        "take_up": "take_up.schema.json",
        "battery": "battery.schema.json",
        "calibration": "calibration.schema.json",
        "selection": "selection.schema.json",
        "publication": "publication.schema.json",
        "vintages": "vintages.schema.json",
        "catalogs": "catalogs.schema.json",
        "legacy_json": "legacy_json",
    }
)
SCHEMA_IDS_BY_KIND: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        **{
            kind: frozenset((schema_id,))
            for kind, schema_id in SCHEMA_ID_BY_KIND.items()
        },
        "schema": SCHEMA_IDS,
    }
)


def assert_schema_id_allowed(kind: str, schema_id: str) -> None:
    """Refuse a manifest row whose kind and schema authority disagree."""

    allowed = SCHEMA_IDS_BY_KIND.get(kind)
    if allowed is None:
        known = ", ".join(sorted(SCHEMA_IDS_BY_KIND))
        raise SpecValidationError(
            f"unknown resource kind {kind!r}; expected one of: {known}"
        )
    if schema_id not in allowed:
        expected = ", ".join(sorted(allowed))
        raise SpecValidationError(
            f"resource kind {kind!r} cannot use schema_id {schema_id!r}; "
            f"expected one of: {expected}"
        )


def _refuse_retrieval(uri: str) -> Resource[Any]:
    """The registry is hermetic: unknown references are always an error."""

    raise NoSuchResource(ref=uri)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecSchemaError(f"duplicate key {key!r} in packaged schema")
        result[key] = value
    return result


def _pointer(parts: Iterable[object]) -> str:
    encoded = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(encoded) if encoded else "/"


def _error_sort_key(error: Any) -> tuple[str, str, str]:
    return (
        _pointer(error.absolute_path),
        _pointer(error.absolute_schema_path),
        error.message,
    )


def _format_validation_errors(schema_id: str, errors: Sequence[Any]) -> str:
    rows = [f"  {_pointer(error.absolute_path)}: {error.message}" for error in errors]
    return f"{schema_id}: {len(rows)} schema validation error(s)\n" + "\n".join(rows)


def _iter_refs(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        ref = value.get("$ref")
        if isinstance(ref, str):
            yield ref
        for child in value.values():
            yield from _iter_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_refs(child)


def _replace_value(original: Any, replacement: Any) -> Any:
    if isinstance(original, MutableMapping) and isinstance(replacement, Mapping):
        original.clear()
        original.update(replacement)
        return original
    if isinstance(original, list) and isinstance(replacement, list):
        original[:] = replacement
        return original
    return replacement


@dataclass(frozen=True, slots=True)
class SchemaRegistry:
    """The immutable schema catalog and its closed reference registry."""

    schemas: Mapping[str, Mapping[str, Any]]
    registry: Registry[Any]
    source: str

    @classmethod
    def from_package_data(
        cls,
        schema_root: Traversable | Path | None = None,
    ) -> SchemaRegistry:
        """Load and verify exactly the approved fifteen-schema catalog.

        An explicit ``schema_root`` is useful for golden and mutation tests.
        Normal source-tree imports fall back to ``<repo>/specs/schema`` when
        package data has not yet been materialized; installed wheels must
        contain ``microcosm/build/spec_engine/schema``.
        """

        root = schema_root if schema_root is not None else _schema_data_root()
        documents = _read_schema_documents(root)
        return cls._from_documents(documents, source=str(root))

    @classmethod
    def _from_documents(
        cls,
        documents: Mapping[str, Mapping[str, Any]],
        *,
        source: str,
    ) -> SchemaRegistry:
        actual_files = frozenset(documents)
        if actual_files != SCHEMA_IDS:
            missing = sorted(SCHEMA_IDS - actual_files)
            unexpected = sorted(actual_files - SCHEMA_IDS)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing!r}")
            if unexpected:
                details.append(f"unexpected={unexpected!r}")
            raise SpecSchemaError(
                "schema catalog must contain exactly the approved 15 JSON "
                f"schemas ({'; '.join(details)})"
            )

        ids: dict[str, str] = {}
        checked: dict[str, Mapping[str, Any]] = {}
        for filename in SCHEMA_FILENAMES:
            schema = documents[filename]
            declared_draft = schema.get("$schema")
            if declared_draft != DRAFT_2020_12:
                raise SpecSchemaError(
                    f"{filename}: $schema must be exactly {DRAFT_2020_12!r}; "
                    f"got {declared_draft!r}"
                )
            schema_id = schema.get("$id")
            if not isinstance(schema_id, str) or not schema_id:
                raise SpecSchemaError(f"{filename}: non-empty string $id required")
            previous = ids.get(schema_id)
            if previous is not None:
                raise SpecSchemaError(
                    f"duplicate schema $id {schema_id!r}: {previous}, {filename}"
                )
            ids[schema_id] = filename
            if schema_id != filename:
                raise SpecSchemaError(
                    f"{filename}: $id must equal its packaged filename; "
                    f"got {schema_id!r}"
                )
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                raise SpecSchemaError(
                    f"{filename}: invalid draft-2020-12 schema: {error.message}"
                ) from error
            checked[filename] = schema

        closed_registry: Registry[Any] = Registry(retrieve=_refuse_retrieval)
        for schema_id in SCHEMA_FILENAMES:
            resource = Resource.from_contents(checked[schema_id])
            closed_registry = closed_registry.with_resource(schema_id, resource)

        # ``check_schema`` verifies the keywords but deliberately does not
        # retrieve every reference.  Resolve each packaged $ref now so a
        # broken or external reference fails before any bundle is accepted.
        for schema_id, schema in checked.items():
            resolver = closed_registry.resolver(schema_id)
            for ref in _iter_refs(schema):
                try:
                    resolver.lookup(ref)
                except Exception as error:
                    raise SpecSchemaError(
                        f"{schema_id}: unresolved closed-world $ref {ref!r}: {error}"
                    ) from error

        return cls(
            schemas=MappingProxyType(checked),
            registry=closed_registry,
            source=source,
        )

    def schema(self, schema_id: str) -> Mapping[str, Any]:
        """Resolve a whole schema or JSON Pointer fragment by identifier."""

        try:
            resolved = self.registry.resolver().lookup(schema_id)
        except Exception as error:
            raise SpecSchemaError(
                f"unknown or unresolved closed-world schema {schema_id!r}: {error}"
            ) from error
        if not isinstance(resolved.contents, Mapping):
            raise SpecSchemaError(
                f"{schema_id}: resolved schema fragment must be an object"
            )
        return resolved.contents

    def validate(self, instance: Any, schema_id: str) -> None:
        """Validate an instance and report all failures in stable path order."""

        self.schema(schema_id)
        validator = Draft202012Validator(
            {"$schema": DRAFT_2020_12, "$ref": schema_id},
            registry=self.registry,
        )
        try:
            errors = sorted(validator.iter_errors(instance), key=_error_sort_key)
        except Exception as error:
            raise SpecSchemaError(
                f"{schema_id}: schema reference resolution failed: {error}"
            ) from error
        if errors:
            raise SpecValidationError(_format_validation_errors(schema_id, errors))

    def inject_defaults(self, instance: Any, schema_id: str) -> Any:
        """Return a deep copy with unambiguous JSON Schema defaults applied."""

        self.schema(schema_id)
        try:
            resolved = self.registry.resolver().lookup(schema_id)
            value = deepcopy(instance)
            return self._inject(value, resolved.contents, resolved.resolver)
        except (SpecSchemaError, SpecValidationError):
            raise
        except Exception as error:
            raise SpecSchemaError(
                f"{schema_id}: default injection failed: {error}"
            ) from error

    def validate_and_inject_defaults(self, instance: Any, schema_id: str) -> Any:
        """Apply defaults without mutating input, then validate the result."""

        resolved = self.inject_defaults(instance, schema_id)
        self.validate(resolved, schema_id)
        return resolved

    def _branch_is_valid(self, instance: Any, schema: Any, resolver: Any) -> bool:
        validator = Draft202012Validator(
            schema,
            registry=self.registry,
            # jsonschema keeps the current referencing base through this
            # private constructor hook; without it local fragment refs inside
            # a selected oneOf branch would incorrectly resolve at ``""``.
            _resolver=resolver,
        )
        return validator.is_valid(instance)

    def _declared_default(
        self,
        schema: Any,
        resolver: Any,
        seen: frozenset[tuple[int, str]] = frozenset(),
    ) -> tuple[bool, Any]:
        if not isinstance(schema, Mapping):
            return False, None

        values: list[Any] = []
        if "default" in schema:
            values.append(schema["default"])

        ref = schema.get("$ref")
        if isinstance(ref, str):
            marker = (id(schema), ref)
            if marker not in seen:
                resolved = resolver.lookup(ref)
                found, value = self._declared_default(
                    resolved.contents,
                    resolved.resolver,
                    seen | {marker},
                )
                if found:
                    values.append(value)

        for child in schema.get("allOf", ()):
            found, value = self._declared_default(child, resolver, seen)
            if found:
                values.append(value)

        if not values:
            return False, None
        first = values[0]
        if any(value != first for value in values[1:]):
            raise SpecSchemaError("conflicting defaults in composed schema")
        return True, deepcopy(first)

    def _inject(self, instance: Any, schema: Any, resolver: Any) -> Any:
        if isinstance(schema, bool):
            return instance
        if not isinstance(schema, Mapping):
            raise SpecSchemaError("schema node must be an object or boolean")

        ref = schema.get("$ref")
        if isinstance(ref, str):
            resolved = resolver.lookup(ref)
            instance = self._inject(
                instance,
                resolved.contents,
                resolved.resolver,
            )

        for child in schema.get("allOf", ()):
            instance = self._inject(instance, child, resolver)

        condition = schema.get("if")
        if isinstance(condition, (Mapping, bool)):
            branch_name = (
                "then"
                if self._branch_is_valid(instance, condition, resolver)
                else "else"
            )
            if branch_name in schema:
                instance = self._inject(instance, schema[branch_name], resolver)

        # Defaults from failed alternatives must never leak.  Evaluate each
        # alternative on its own copy, including that alternative's defaults,
        # and adopt a result only when exactly one branch validates.
        for keyword in ("oneOf", "anyOf"):
            alternatives = schema.get(keyword)
            if not isinstance(alternatives, list):
                continue
            matches: list[Any] = []
            for alternative in alternatives:
                trial = self._inject(deepcopy(instance), alternative, resolver)
                if self._branch_is_valid(trial, alternative, resolver):
                    matches.append(trial)
            if len(matches) == 1:
                instance = _replace_value(instance, matches[0])

        if isinstance(instance, MutableMapping):
            properties = schema.get("properties", {})
            if isinstance(properties, Mapping):
                for name, child_schema in properties.items():
                    if name not in instance:
                        found, default = self._declared_default(
                            child_schema,
                            resolver,
                        )
                        if found:
                            instance[name] = default
                    if name in instance:
                        instance[name] = self._inject(
                            instance[name],
                            child_schema,
                            resolver,
                        )

        if isinstance(instance, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, (Mapping, bool)):
                for index, value in enumerate(instance):
                    instance[index] = self._inject(value, item_schema, resolver)

            prefix_items = schema.get("prefixItems")
            if isinstance(prefix_items, list):
                for index, child_schema in enumerate(prefix_items[: len(instance)]):
                    instance[index] = self._inject(
                        instance[index],
                        child_schema,
                        resolver,
                    )

        return instance


def _schema_data_root() -> Traversable | Path:
    packaged = resources.files("microcosm.build.spec_engine").joinpath("schema")
    if packaged.is_dir():
        return packaged

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "specs" / "schema"
        if candidate.is_dir():
            return candidate
    raise SpecSchemaError(
        "approved schema package data is absent and no source-tree "
        "specs/schema fallback exists"
    )


def _read_schema_documents(root: Traversable | Path) -> dict[str, Mapping[str, Any]]:
    try:
        json_entries = {
            entry.name: entry
            for entry in root.iterdir()
            if entry.name.endswith(".json") and not entry.is_dir()
        }
    except (FileNotFoundError, NotADirectoryError) as error:
        raise SpecSchemaError(f"schema directory is unavailable: {root}") from error

    documents: dict[str, Mapping[str, Any]] = {}
    for filename in sorted(json_entries):
        entry = json_entries[filename]
        try:
            document = json.loads(
                entry.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value}")
                ),
            )
        except SpecSchemaError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise SpecSchemaError(
                f"{filename}: invalid schema JSON: {error}"
            ) from error
        if not isinstance(document, Mapping):
            raise SpecSchemaError(f"{filename}: schema document must be an object")
        documents[filename] = document
    return documents


@lru_cache(maxsize=1)
def load_schema_registry() -> SchemaRegistry:
    """Return the process-wide verified package schema registry."""

    return SchemaRegistry.from_package_data()


__all__ = [
    "DRAFT_2020_12",
    "SCHEMA_FILENAMES",
    "SCHEMA_IDS",
    "SCHEMA_ID_BY_KIND",
    "SCHEMA_IDS_BY_KIND",
    "SchemaRegistry",
    "assert_schema_id_allowed",
    "load_schema_registry",
]
