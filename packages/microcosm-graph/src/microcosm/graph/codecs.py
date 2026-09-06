"""Source codecs: the boundary from source bytes to a decoded input.

A codec is registered in exactly one of two explicit modes, and a name
belongs to at most one mode:

*Frame mode* (:data:`SourceCodec`, :meth:`SourceCodecRegistry.register`,
:meth:`SourceCodecRegistry.load`) decodes a source into a population
:class:`Frame`. A ``CREATE`` node's kernel calls it to build a population,
and so does any other node whose source really is one — a held-out reference
frame read for scoring, say.

*Raw-byte mode* (:data:`SourceBytesCodec`,
:meth:`SourceCodecRegistry.register_bytes`,
:meth:`SourceCodecRegistry.load_bytes`) decodes a source into immutable
``bytes``. A lookup table (an NPZ of ratios, a CSV crosswalk) is not a
population, and an import kernel that turns one into a typed
:class:`~microcosm.graph.decl.ArtifactOutput` must be able to read its real
bytes without a Frame codec registered as a pretence. Neither mode can be
loaded through the other: the mismatch is a :class:`TypeError` naming the
mode the codec actually has, so no caller receives bytes where it declared a
Frame.

Three codecs ship with the graph runtime:

``frame-store`` (Frame)
    Loads a content-verified Frame from the path of a ``ContentStore`` frame
    object directory.

``csv-tables`` (Frame)
    Loads one CSV per entity using ``schema.json``.  The schema may give a
    ``tables`` mapping (otherwise ``<entity>.csv`` is used), global or
    per-entity dtype mappings, a ``strata_column``, and either (a) a weight
    declaration plus ``weights_table`` CSV or (b) ``weights.json`` entries.
    A JSON weight entry is ``{"kind": "design", "values": [...]}`` or
    ``{"kind": "design", "column": "household_weight"}``; entries are
    keyed by entity, or may carry their own ``entity`` field.

``raw-bytes-v1`` (raw bytes)
    Reads one regular file, bounded at :data:`RAW_BYTES_MAX_BYTES`. It
    interprets nothing: the consuming kernel owns the payload's format and
    validates it. Identity, the pre-run content key, and the post-run
    mutation check stay where they already are, in the executor.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema, Frame, LinkSpec, WeightKind, Weights

from .store import ContentStore, StoreUnavailable

__all__ = [
    "RAW_BYTES_MAX_BYTES",
    "SOURCE_CODECS",
    "SourceBytesCodec",
    "SourceCodec",
    "SourceCodecRegistry",
    "load_csv_tables",
    "load_frame_store",
    "load_raw_bytes",
    "load_source",
    "load_source_bytes",
]

type SourceCodec = Callable[..., Frame]
type SourceBytesCodec = Callable[..., bytes]

RAW_BYTES_MAX_BYTES = 64 * 1024 * 1024
"""The most ``raw-bytes-v1`` will read from one source file (64 MiB)."""


class SourceCodecRegistry:
    """Named source loaders in two explicit modes: Frame and raw bytes.

    A name is registered in one mode only. :meth:`get` answers the
    availability question the executor asks before any kernel runs, in
    either mode; :meth:`load` and :meth:`load_bytes` are what hold a codec
    to the mode it was registered in.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, SourceCodec] = {}
        self._byte_loaders: dict[str, SourceBytesCodec] = {}

    @staticmethod
    def _check_declaration(name: str, loader: object) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Source codec names must be non-empty strings.")
        if not callable(loader):
            raise TypeError("Source codec loaders must be callable.")

    def register(self, name: str, loader: SourceCodec) -> SourceCodec:
        """Register and return a Frame ``loader`` under a non-empty codec name."""

        self._check_declaration(name, loader)
        if name in self._byte_loaders:
            raise ValueError(
                f"Source codec {name!r} is already registered as a raw-bytes codec."
            )
        incumbent = self._loaders.get(name)
        if incumbent is not None and incumbent is not loader:
            raise ValueError(f"Source codec {name!r} is already registered.")
        self._loaders[name] = loader
        return loader

    def register_bytes(self, name: str, loader: SourceBytesCodec) -> SourceBytesCodec:
        """Register and return a raw-byte ``loader`` under a non-empty codec name.

        A raw-byte codec returns the source's bytes and claims nothing about
        their meaning; it never stands in for a population.
        """

        self._check_declaration(name, loader)
        if name in self._loaders:
            raise ValueError(
                f"Source codec {name!r} is already registered as a Frame codec."
            )
        incumbent = self._byte_loaders.get(name)
        if incumbent is not None and incumbent is not loader:
            raise ValueError(f"Source codec {name!r} is already registered.")
        self._byte_loaders[name] = loader
        return loader

    def get(self, name: str) -> SourceCodec | SourceBytesCodec:
        """Resolve ``name`` in either mode, or raise fatal :class:`StoreUnavailable`.

        This is availability, not decoding: a registered raw-byte codec is
        installed, and resolving it here never turns it into a Frame codec.
        """

        for loaders in (self._loaders, self._byte_loaders):
            try:
                return loaders[name]
            except KeyError:
                continue
        raise StoreUnavailable(f"Source codec {name!r} is not installed.")

    def _frame_loader(self, name: str) -> SourceCodec:
        if name in self._byte_loaders:
            raise TypeError(
                f"Source codec {name!r} is a raw-bytes codec; read it with "
                "load_bytes. Bytes are never presented as a Frame."
            )
        try:
            return self._loaders[name]
        except KeyError as error:
            raise StoreUnavailable(
                f"Source codec {name!r} is not installed."
            ) from error

    def _bytes_loader(self, name: str) -> SourceBytesCodec:
        if name in self._loaders:
            raise TypeError(
                f"Source codec {name!r} is a Frame codec; read it with load. "
                "A Frame is never presented as raw bytes."
            )
        try:
            return self._byte_loaders[name]
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
        """Decode ``path`` with the Frame codec ``name`` and require a Frame."""

        loader = self._frame_loader(name)
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

    def load_bytes(
        self,
        name: str,
        path: Path,
        *,
        store: ContentStore | None = None,
    ) -> bytes:
        """Read ``path`` with the raw-byte codec ``name`` and require bytes.

        This decodes an external source path, unlike
        :meth:`ContentStore.load_bytes`, which reads a stored object back by
        its content key.

        Unavailability keeps the classification :meth:`load` gives it: an
        uninstalled codec or a missing dependency is fatal
        :class:`StoreUnavailable`, never a decoded value.
        """

        loader = self._bytes_loader(name)
        try:
            payload = loader(Path(path), store=store)
        except StoreUnavailable:
            raise
        except ImportError as error:
            raise StoreUnavailable(
                f"Source codec {name!r} needs an unavailable dependency."
            ) from error
        if not isinstance(payload, bytes):
            raise TypeError(
                f"Source codec {name!r} returned {type(payload).__name__}, not bytes."
            )
        return payload

    def names(self) -> tuple[str, ...]:
        """The registered Frame codec names, in canonical order.

        Each mode is enumerated by its own pair — ``names``/``as_mapping``
        here, :meth:`bytes_names`/:meth:`as_bytes_mapping` there — so a
        snapshot taken through either pair stays resolvable through it. The
        name space is still shared: a name in neither tuple is not therefore
        free, because registering it in one mode reserves it in both.
        """

        return tuple(sorted(self._loaders))

    def bytes_names(self) -> tuple[str, ...]:
        """The registered raw-byte codec names, in canonical order."""

        return tuple(sorted(self._byte_loaders))

    def as_mapping(self) -> Mapping[str, SourceCodec]:
        """A read-only snapshot of the registered Frame loaders."""

        return MappingProxyType(dict(self._loaders))

    def as_bytes_mapping(self) -> Mapping[str, SourceBytesCodec]:
        """A read-only snapshot of the registered raw-byte loaders."""

        return MappingProxyType(dict(self._byte_loaders))


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


def _nonblocking_opener(path: str, flags: int) -> int:
    """Open without blocking, so a FIFO in a source's place cannot hang a run."""

    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def load_raw_bytes(path: Path, *, store: ContentStore | None = None) -> bytes:
    """Read one regular file's bytes, bounded at :data:`RAW_BYTES_MAX_BYTES`.

    The codec claims nothing about the payload: a lookup NPZ, a crosswalk
    CSV, and a corrupt file are all just bytes here, and the kernel that
    imports them validates their format. What this function does own is the
    refusal to read something that is not one bounded regular file.

    Symlinks are followed, as the executor's ``resolve(strict=True)`` and
    ``source_content_key`` already do. A directory, a FIFO, a device, or a
    socket is refused: the mode is read from the descriptor this function
    itself opened, and the open is non-blocking, so the codec cannot be made
    to wait on a pipe or stream a device. The read is bounded rather than
    trusted to ``st_size``, because a file may grow after it is measured.
    Detecting that a source moved is the executor's post-run content check;
    this bound only keeps the codec from reading an unbounded amount first.
    """

    del store  # raw bytes are self-describing; no content store is consulted
    source = Path(path)
    try:
        handle = open(source, "rb", opener=_nonblocking_opener)
    except IsADirectoryError as error:
        raise ValueError(
            f"Raw source at {source} is a directory; raw-bytes-v1 reads one "
            "regular file."
        ) from error
    except OSError as error:
        raise ValueError(f"Raw source at {source} is not readable: {error}") from error
    with handle:
        # A directory never reaches here: opening one raises IsADirectoryError
        # above.  Everything else that is not a regular file is refused from the
        # descriptor's own mode, not from a second look at the path.
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(
                f"Raw source at {source} is not a regular file; raw-bytes-v1 "
                "reads one regular file."
            )
        try:
            payload = handle.read(RAW_BYTES_MAX_BYTES + 1)
        except OSError as error:
            raise ValueError(
                f"Raw source at {source} is not readable: {error}"
            ) from error
    if len(payload) > RAW_BYTES_MAX_BYTES:
        raise ValueError(
            f"Raw source at {source} is larger than the "
            f"{RAW_BYTES_MAX_BYTES}-byte raw-bytes-v1 limit."
        )
    return payload


SOURCE_CODECS = SourceCodecRegistry()
SOURCE_CODECS.register("frame-store", load_frame_store)
SOURCE_CODECS.register("csv-tables", load_csv_tables)
SOURCE_CODECS.register_bytes("raw-bytes-v1", load_raw_bytes)


def load_source(
    codec: str,
    path: Path,
    *,
    store: ContentStore | None = None,
    registry: SourceCodecRegistry = SOURCE_CODECS,
) -> Frame:
    """Decode one source into a Frame through the selected registry."""

    return registry.load(codec, path, store=store)


def load_source_bytes(
    codec: str,
    path: Path,
    *,
    store: ContentStore | None = None,
    registry: SourceCodecRegistry = SOURCE_CODECS,
) -> bytes:
    """Read one source's bytes through the selected registry."""

    return registry.load_bytes(codec, path, store=store)
