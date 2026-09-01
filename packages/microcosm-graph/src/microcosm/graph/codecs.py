"""Source codecs: the sole boundary from source bytes to :class:`Frame`.

Two codecs ship with the graph runtime:

``frame-store``
    Loads a content-verified Frame from the path of a ``ContentStore`` frame
    object directory.

``csv-tables``
    Loads one CSV per entity using ``schema.json``.  The schema may give a
    ``tables`` mapping (otherwise ``<entity>.csv`` is used), global or
    per-entity dtype mappings, a ``strata_column``, and either (a) a weight
    declaration plus ``weights_table`` CSV or (b) ``weights.json`` entries.
    A JSON weight entry is ``{"kind": "design", "values": [...]}`` or
    ``{"kind": "design", "column": "household_weight"}``; entries are
    keyed by entity, or may carry their own ``entity`` field.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema, Frame, LinkSpec, WeightKind, Weights

from .store import ContentStore, StoreUnavailable

__all__ = [
    "SOURCE_CODECS",
    "SourceCodec",
    "SourceCodecRegistry",
    "load_csv_tables",
    "load_frame_store",
    "load_source",
]

type SourceCodec = Callable[..., Frame]


class SourceCodecRegistry:
    """Named source-to-Frame loaders."""

    def __init__(self) -> None:
        self._loaders: dict[str, SourceCodec] = {}

    def register(self, name: str, loader: SourceCodec) -> SourceCodec:
        """Register and return ``loader`` under a non-empty codec name."""

        if not isinstance(name, str) or not name:
            raise ValueError("Source codec names must be non-empty strings.")
        if not callable(loader):
            raise TypeError("Source codec loaders must be callable.")
        incumbent = self._loaders.get(name)
        if incumbent is not None and incumbent is not loader:
            raise ValueError(f"Source codec {name!r} is already registered.")
        self._loaders[name] = loader
        return loader

    def get(self, name: str) -> SourceCodec:
        """Resolve ``name`` or raise fatal :class:`StoreUnavailable`."""

        try:
            return self._loaders[name]
        except KeyError as error:
            raise StoreUnavailable(
                f"Source codec {name!r} is not installed."
            ) from error

    def load(
        self,
        name: str,
        path: Path,
        *,
        store: ContentStore | None = None,
    ) -> Frame:
        """Decode ``path`` with ``name`` and require a Frame result."""

        loader = self.get(name)
        try:
            frame = loader(Path(path), store=store)
        except StoreUnavailable:
            raise
        except ImportError as error:
            raise StoreUnavailable(
                f"Source codec {name!r} needs an unavailable dependency."
            ) from error
        if not isinstance(frame, Frame):
            raise TypeError(
                f"Source codec {name!r} returned {type(frame).__name__}, not Frame."
            )
        return frame

    def names(self) -> tuple[str, ...]:
        """Registered names in canonical order."""

        return tuple(sorted(self._loaders))

    def as_mapping(self) -> Mapping[str, SourceCodec]:
        """A read-only snapshot of registered loaders."""

        return MappingProxyType(dict(self._loaders))


def load_frame_store(path: Path, *, store: ContentStore | None = None) -> Frame:
    """Load a verified frame object from its content-store directory."""

    del store  # the object path is self-describing and may belong to another store
    return ContentStore.load_frame_path(path)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} at {path} is not readable JSON.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} at {path} must be a JSON object.")
    return value


def _entity_schema(document: Mapping[str, Any]) -> EntitySchema:
    raw = document.get("schema", document)
    if not isinstance(raw, Mapping):
        raise ValueError("schema.json 'schema' must be an object.")
    groups = raw.get("group_entities")
    if not isinstance(groups, list) or any(
        not isinstance(item, str) for item in groups
    ):
        raise ValueError("schema.json group_entities must be an array of strings.")
    raw_links = raw.get("links", [])
    if not isinstance(raw_links, list):
        raise ValueError("schema.json links must be an array.")
    try:
        links = tuple(
            LinkSpec(
                name=item["name"],
                left_entity=item["left_entity"],
                right_entity=item["right_entity"],
            )
            for item in raw_links
            if isinstance(item, Mapping)
        )
        if len(links) != len(raw_links):
            raise ValueError("schema.json links must be objects.")
        return EntitySchema(
            person_entity=raw.get("person_entity", "person"),
            group_entities=tuple(groups),
            links=links,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"schema.json declares an invalid entity schema: {error}"
        ) from error


def _table_files(
    source: Path,
    document: Mapping[str, Any],
    schema: EntitySchema,
) -> dict[str, Path]:
    expected = (*schema.entities, *(link.name for link in schema.links))
    raw_tables = document.get("tables")
    if raw_tables is None:
        return {name: source / f"{name}.csv" for name in expected}
    if not isinstance(raw_tables, Mapping):
        raise ValueError("schema.json tables must be an entity-to-file object.")
    if set(raw_tables) != set(expected):
        raise ValueError(
            "schema.json tables must name exactly the declared entities and links."
        )
    files: dict[str, Path] = {}
    for name in expected:
        filename = raw_tables[name]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"Unsafe CSV table filename {filename!r} for {name!r}.")
        files[name] = source / filename
    return files


def _dtype_map(
    document: Mapping[str, Any], entity: str, columns: pd.Index
) -> dict[str, str]:
    raw_dtypes = document.get("dtypes", {})
    if not isinstance(raw_dtypes, Mapping):
        raise ValueError("schema.json dtypes must be an object.")
    nested = raw_dtypes.get(entity)
    if isinstance(nested, Mapping):
        selected = nested
    else:
        selected = raw_dtypes
    result: dict[str, str] = {}
    for column in columns:
        if column not in selected:
            continue
        dtype = selected[column]
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Invalid dtype for {entity}.{column}.")
        result[str(column)] = dtype
    return result


def _read_tables(
    source: Path,
    document: Mapping[str, Any],
    schema: EntitySchema,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, path in _table_files(source, document, schema).items():
        table = pd.read_csv(path)
        dtypes = _dtype_map(document, name, table.columns)
        if dtypes:
            try:
                table = table.astype(dtypes)
            except ImportError as error:
                raise StoreUnavailable(
                    f"CSV table {name!r} needs an unavailable dtype dependency."
                ) from error
        tables[name] = table
    return tables


def _weight_specs(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if any(not isinstance(item, dict) for item in value):
            raise ValueError("Weight declarations must be JSON objects.")
        return list(value)
    if not isinstance(value, dict):
        raise ValueError("Weight declarations must be an object or array.")
    if "entity" in value:
        return [value]
    specs: list[dict[str, Any]] = []
    for entity, raw_spec in value.items():
        if isinstance(raw_spec, str):
            specs.append({"entity": entity, "kind": raw_spec})
        elif isinstance(raw_spec, dict):
            specs.append({"entity": entity, **raw_spec})
        else:
            raise ValueError(f"Weight declaration for {entity!r} is malformed.")
    return specs


def _weight_values_from_table(
    spec: Mapping[str, Any],
    weight_table: pd.DataFrame,
    entity_table: pd.DataFrame,
    schema: EntitySchema,
) -> np.ndarray:
    entity = spec.get("entity")
    column = spec.get("column")
    if not isinstance(entity, str) or not isinstance(column, str):
        raise ValueError("CSV weight declarations require entity and column.")
    if column not in weight_table:
        raise ValueError(f"Weight table has no column {column!r}.")
    id_column = schema.entity_id_column(entity)
    if id_column in weight_table:
        expected_ids = entity_table[id_column].reset_index(drop=True)
        actual_ids = (
            weight_table[id_column].reset_index(drop=True).astype(expected_ids.dtype)
        )
        if not actual_ids.equals(expected_ids):
            raise ValueError(f"Weight table ids do not align to entity {entity!r}.")
    return weight_table[column].to_numpy(dtype=np.float64)


def _weights_from_json(
    source: Path,
    document: Mapping[str, Any],
    tables: dict[str, pd.DataFrame],
    schema: EntitySchema,
) -> dict[str, Weights]:
    weights_path = source / "weights.json"
    weights_document = _json_object(weights_path, label="weights.json")
    raw_specs = weights_document.get("weights", weights_document)
    # schema.json may carry kinds/entities while weights.json is a table-shaped
    # object of value arrays. Prefer the schema declaration in that case.
    declared = document.get("weights")
    specs = _weight_specs(declared if declared is not None else raw_specs)
    weights: dict[str, Weights] = {}
    for spec in specs:
        entity = spec.get("entity")
        kind_value = spec.get("kind")
        if not isinstance(entity, str) or entity not in tables:
            raise ValueError(f"Weight declaration has unknown entity {entity!r}.")
        if not isinstance(kind_value, str):
            raise ValueError(f"Weight declaration for {entity!r} has no kind.")
        try:
            kind = WeightKind(kind_value)
        except ValueError as error:
            raise ValueError(f"Unknown weight kind {kind_value!r}.") from error
        raw_entry = raw_specs.get(entity) if isinstance(raw_specs, dict) else None
        values = spec.get("values")
        column = spec.get("column")
        if values is None and isinstance(raw_entry, dict):
            values = raw_entry.get("values")
            column = raw_entry.get("column", column)
        if values is None and isinstance(column, str) and column in weights_document:
            values = weights_document[column]
        if values is None and isinstance(column, str) and column in tables[entity]:
            values = tables[entity].pop(column).to_numpy()
        if values is None:
            raise ValueError(f"No values supplied for weights of {entity!r}.")
        weights[entity] = Weights(np.asarray(values, dtype=np.float64), kind)
    return weights


def _load_weights(
    source: Path,
    document: Mapping[str, Any],
    tables: dict[str, pd.DataFrame],
    schema: EntitySchema,
) -> dict[str, Weights]:
    raw_table = document.get("weights_table")
    if raw_table is None:
        return _weights_from_json(source, document, tables, schema)
    if (
        not isinstance(raw_table, str)
        or not raw_table
        or Path(raw_table).name != raw_table
    ):
        raise ValueError("schema.json weights_table must be a safe filename.")
    specs = _weight_specs(document.get("weights"))
    weight_table = pd.read_csv(source / raw_table)
    dtypes = _dtype_map(document, "weights", weight_table.columns)
    if dtypes:
        weight_table = weight_table.astype(dtypes)
    weights: dict[str, Weights] = {}
    for spec in specs:
        entity = spec.get("entity")
        kind_value = spec.get("kind")
        if not isinstance(entity, str) or entity not in tables:
            raise ValueError(f"Weight declaration has unknown entity {entity!r}.")
        if not isinstance(kind_value, str):
            raise ValueError(f"Weight declaration for {entity!r} has no kind.")
        try:
            kind = WeightKind(kind_value)
        except ValueError as error:
            raise ValueError(f"Unknown weight kind {kind_value!r}.") from error
        values = _weight_values_from_table(spec, weight_table, tables[entity], schema)
        weights[entity] = Weights(values, kind)
    return weights


def load_csv_tables(path: Path, *, store: ContentStore | None = None) -> Frame:
    """Load a directory of entity CSVs plus schema and typed weights."""

    del store
    source = Path(path)
    document = _json_object(source / "schema.json", label="schema.json")
    schema = _entity_schema(document)
    tables = _read_tables(source, document, schema)
    strata_column = document.get("strata_column")
    if strata_column is None:
        strata = None
    else:
        if (
            not isinstance(strata_column, str)
            or strata_column not in tables[schema.person_entity]
        ):
            raise ValueError("schema.json strata_column is not on the person table.")
        strata = tables[schema.person_entity].pop(strata_column)
    weights = _load_weights(source, document, tables, schema)
    return Frame(tables, schema, weights, strata)


SOURCE_CODECS = SourceCodecRegistry()
SOURCE_CODECS.register("frame-store", load_frame_store)
SOURCE_CODECS.register("csv-tables", load_csv_tables)


def load_source(
    codec: str,
    path: Path,
    *,
    store: ContentStore | None = None,
    registry: SourceCodecRegistry = SOURCE_CODECS,
) -> Frame:
    """Decode one source through the selected registry."""

    return registry.load(codec, path, store=store)
