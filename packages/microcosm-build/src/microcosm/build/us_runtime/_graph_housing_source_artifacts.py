"""Strict local housing source-projection, output and merge-evidence contracts.

This module selects no population and reads no source data. It encodes exactly
the five registered housing leaves for three ID-associated tables, plus the
projection and merge evidence that bind them to their producing nodes. Column
buffers carry only little-endian int64/float64, canonical single-byte booleans
and UTF-8 offset payloads; there is no pickle transport and no arbitrary object
dtype support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.fit._graph_legacy_qrf import decode_json, encode_json, sha
from microcosm.fit.qrf import _index_identity, _IndexIdentity
from microcosm.graph import ArtifactType, Numeric, platform_fingerprint
from microcosm.graph.population import assert_dtype, dtype_for_token

from ._graph_acs_rent_artifacts import (
    assert_string_runtime,
    encoding_implementation_hash,
    is_hash,
    string_runtime_identity,
)
from .graph_context import _row_identity
from .operator_boundary import PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES

__all__ = [
    "FRESH_SOURCE_MODE",
    "HOUSING_GATE_SCOPE",
    "HOUSING_MERGE_EVIDENCE_TYPE",
    "HOUSING_PENDING_CONTEXT_TYPE",
    "HOUSING_PHASE",
    "HOUSING_SOURCE_OUTPUTS_TYPE",
    "HOUSING_SOURCE_PROJECTION_TYPE",
    "SOURCE_COLUMNS",
    "artifact",
    "assert_string_runtime",
    "binding",
    "decode_json",
    "decode_merge_evidence",
    "decode_projection",
    "decode_source_outputs",
    "encode_json",
    "encode_merge_evidence",
    "encode_projection",
    "encode_source_outputs",
    "encoding_implementation_hash",
    "housing_output_family",
    "is_hash",
    "merge_columns",
    "sha",
    "source_tables",
    "strict_ids",
    "string_runtime_identity",
]

HOUSING_SOURCE_PROJECTION_TYPE = ArtifactType(
    "microcosm.us.housing_source_projection", 1
)
HOUSING_SOURCE_OUTPUTS_TYPE = ArtifactType("microcosm.us.housing_source_outputs", 1)
HOUSING_PENDING_CONTEXT_TYPE = ArtifactType("microcosm.us.housing_pending_context", 1)
HOUSING_MERGE_EVIDENCE_TYPE = ArtifactType("microcosm.us.housing_merge_evidence", 1)

HOUSING_PHASE = "prepare_multispine_source_inputs_for_clone"
SOURCE_OPERATOR_PHASE = "pre_clone"
SOURCE_OPERATOR_NAME = "with_us_housing_inputs"
SOURCE_OPERATOR_FAMILY = "housing_inputs"
FRESH_SOURCE_MODE = "fresh_source"
HOUSING_GATE_SCOPE = "source_housing_completion_integrity"

HEADER_MAX_BYTES = 256 * 1024
PROJECTION_MAGIC = b"microcosm.us.housing_source_projection/1\n"
OUTPUTS_MAGIC = b"microcosm.us.housing_source_outputs/1\n"

_CANONICAL_STRING = dtype_for_token("string")
_FLOAT = np.dtype("<f8")
_INT = np.dtype("<i8")

#: The five leaves this recipe owns, in the order their buffers are written.
#: ``source`` is the representation on the projected source population, and
#: ``merge`` is the representation the registered full-pool merge produces.
SOURCE_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("person", "pre_subsidy_rent", "float64", "float64"),
    ("household", "tenure_type", "string", "string"),
    ("spm_unit", "receives_housing_assistance", "bool", "boolean"),
    ("spm_unit", "spm_unit_tenure_type", "string", "string"),
    ("spm_unit", "takes_up_housing_assistance_if_eligible", "bool", "boolean"),
)
SOURCE_ENTITIES: tuple[str, ...] = ("person", "household", "spm_unit")
_ENTITY_FIELDS = {"rows", "id_dtype", "ordered_ids_sha256"}
_COLUMN_FIELDS = {
    "entity",
    "column",
    "dtype",
    "encoding",
    "nulls",
    "value_bytes",
    "values_sha256",
}
_STRING_COLUMN_FIELDS = _COLUMN_FIELDS | {
    "string_storage",
    "string_na_marker",
    "utf8_bytes",
}
_OUTPUT_FIELDS = {
    "schema_version",
    "mode",
    "phase",
    "operator",
    "family",
    "entities",
    "columns",
    "projection_sha256",
    "projection_producer_key",
    "frame_context_sha256",
    "frame_context_producer_key",
    "validation_sha256",
    "validation_producer_key",
}
_PROJECTION_FIELDS = {
    "schema_version",
    "mode",
    "phase",
    "operator",
    "family",
    "selection",
    "entities",
    "person_index",
    "weight_sources",
    "output_family",
    "parent_context_sha256",
    "parent_context_producer_key",
}
_MERGE_EVIDENCE_FIELDS = {
    "schema_version",
    "mode",
    "phase",
    "scope",
    "projection_sha256",
    "projection_producer_key",
    "source_outputs_sha256",
    "source_outputs_producer_key",
    "pending_context_sha256",
    "merged_rows",
    "selected_rows",
    "pool_rows",
    "merged_cells",
    "native_cells",
    "source_preparation",
}
_CELL_FIELDS = {"entity", "column", "dtype", "rows", "digest"}


def housing_output_family() -> dict[str, tuple[str, ...]]:
    """Return the live registered housing output roster, not a private copy."""
    declared = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[SOURCE_OPERATOR_FAMILY]
    family = {entity: tuple(sorted(columns)) for entity, columns in declared.items()}
    expected: dict[str, list[str]] = {entity: [] for entity in SOURCE_ENTITIES}
    for entity, column, _source, _merge in SOURCE_COLUMNS:
        expected[entity].append(column)
    if family != {entity: tuple(sorted(names)) for entity, names in expected.items()}:
        raise ValueError(
            "The registered housing output family differs from this recipe's five "
            f"declared leaves: {family}."
        )
    return family


def source_tables() -> dict[str, tuple[tuple[str, str], ...]]:
    """Return the source-population representation, grouped by entity."""
    grouped: dict[str, list[tuple[str, str]]] = {
        entity: [] for entity in SOURCE_ENTITIES
    }
    for entity, column, source, _merge in SOURCE_COLUMNS:
        grouped[entity].append((column, source))
    housing_output_family()
    return {entity: tuple(columns) for entity, columns in grouped.items()}


def merge_columns() -> tuple[tuple[str, str, str], ...]:
    """Return ``(entity, column, dtype)`` for the merged full-pool leaves."""
    housing_output_family()
    return tuple(
        (entity, column, merge) for entity, column, _source, merge in SOURCE_COLUMNS
    )


def artifact(context, name, expected_type):
    """Return the single declared typed edge named ``name``, or refuse."""
    value = context.artifacts.get(name)
    edges = [edge for edge in context.node.artifact_inputs if edge.name == name]
    if (
        value is None
        or len(edges) != 1
        or edges[0].type != expected_type
        or value.type != expected_type
    ):
        raise ValueError(f"US housing source requires the exact typed {name} edge.")
    binding(value)
    return value


def binding(value) -> dict[str, str]:
    if not is_hash(value.producer_key):
        raise ValueError("US housing source artifact requires its producer key.")
    if (
        value.numerics.numeric is not Numeric.PLATFORM_BITWISE
        or value.numerics.platform != platform_fingerprint()
    ):
        raise ValueError(
            "US housing source artifact requires the current platform's bitwise scope."
        )
    if not isinstance(value.payload, bytes):
        raise ValueError("US housing source artifacts require immutable bytes.")
    return {"sha256": sha(value.payload), "producer_key": value.producer_key}


def strict_ids(series: pd.Series, *, label: str, unique: bool = True) -> np.ndarray:
    if series.dtype != _INT:
        raise ValueError(f"US housing source {label} requires ordinary int64 identity.")
    values = series.to_numpy(copy=False)
    if (
        values.dtype != _INT
        or len(values) == 0
        or (values < 0).any()
        or (unique and not series.is_unique)
    ):
        raise ValueError(
            f"US housing source {label} identity is empty, duplicated or invalid."
        )
    return values


def _entity_identity(ids: np.ndarray, entity: str) -> dict[str, object]:
    return _row_identity(
        pd.DataFrame({f"{entity}_id": pd.Series(ids, dtype="int64")}), entity
    )


def _validate_identity(value: object, *, rows: int | None = None) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _ENTITY_FIELDS
        or type(value["rows"]) is not int
        or value["rows"] < 0
        or value["id_dtype"] != "int64"
        or not is_hash(value["ordered_ids_sha256"])
        or (rows is not None and value["rows"] != rows)
    ):
        raise ValueError("US housing source row identity is malformed.")


def _envelope(magic: bytes, header: bytes, body: bytes = b"") -> bytes:
    if len(header) > HEADER_MAX_BYTES:
        raise ValueError("US housing source header exceeds its bound.")
    return magic + len(header).to_bytes(4, "big") + header + body


def _split(payload: bytes, magic: bytes) -> tuple[dict[str, object], memoryview]:
    if not isinstance(payload, bytes) or not payload.startswith(magic):
        raise ValueError("Invalid US housing source envelope.")
    start = len(magic)
    if len(payload) < start + 4:
        raise ValueError("Truncated US housing source header length.")
    length = int.from_bytes(payload[start : start + 4], "big")
    start += 4
    if not 0 < length <= HEADER_MAX_BYTES or len(payload) < start + length:
        raise ValueError("US housing source header is truncated or exceeds its bound.")
    return decode_json(payload[start : start + length]), memoryview(payload)[
        start + length :
    ]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def encode_projection(
    *,
    parent_context,
    selection: Mapping[str, str],
    entities: Mapping[str, Mapping[str, np.ndarray]],
    person_index: pd.Index,
    weight_sources: Mapping[str, str],
    mode: str,
) -> bytes:
    """Bind the parent context, the selection rule and both identity sets."""
    if mode != FRESH_SOURCE_MODE:
        raise ValueError(f"Unsupported US housing source mode {mode!r}.")
    parent = binding(parent_context)
    document = {
        "schema_version": 1,
        "mode": mode,
        "phase": HOUSING_PHASE,
        "operator": SOURCE_OPERATOR_NAME,
        "family": SOURCE_OPERATOR_FAMILY,
        "selection": dict(selection),
        "entities": {
            entity: {
                "full": _entity_identity(values["full"], entity),
                "selected": _entity_identity(values["selected"], entity),
            }
            for entity, values in sorted(entities.items())
        },
        "person_index": _index_identity(person_index).to_dict(),
        "weight_sources": dict(weight_sources),
        "output_family": {
            entity: list(columns)
            for entity, columns in sorted(housing_output_family().items())
        },
        "parent_context_sha256": parent["sha256"],
        "parent_context_producer_key": parent["producer_key"],
    }
    return _envelope(PROJECTION_MAGIC, encode_json(document))


def decode_projection(payload: bytes) -> Mapping[str, object]:
    document, body = _split(payload, PROJECTION_MAGIC)
    if len(body):
        raise ValueError("US housing source projection carries no body.")
    if (
        set(document) != _PROJECTION_FIELDS
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["mode"] != FRESH_SOURCE_MODE
        or document["phase"] != HOUSING_PHASE
        or document["operator"] != SOURCE_OPERATOR_NAME
        or document["family"] != SOURCE_OPERATOR_FAMILY
        or not is_hash(document["parent_context_sha256"])
        or not is_hash(document["parent_context_producer_key"])
    ):
        raise ValueError("Unsupported US housing source projection schema.")
    selection = document["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "column",
        "phase",
        "clone_index_column",
    }:
        raise ValueError("US housing source projection selection is malformed.")
    entities = document["entities"]
    if not isinstance(entities, dict) or not entities:
        raise ValueError("US housing source projection describes no entity.")
    for value in entities.values():
        if not isinstance(value, dict) or set(value) != {"full", "selected"}:
            raise ValueError("US housing source projection identity is malformed.")
        _validate_identity(value["full"])
        _validate_identity(value["selected"])
        if value["selected"]["rows"] > value["full"]["rows"]:
            raise ValueError(
                "US housing source selection cannot exceed its parent population."
            )
    _IndexIdentity.from_dict(document["person_index"])
    weight_sources = document["weight_sources"]
    if (
        not isinstance(weight_sources, dict)
        or not weight_sources
        or any(not isinstance(kind, str) for kind in weight_sources.values())
    ):
        raise ValueError("US housing source projection weight sources are malformed.")
    if {
        entity: tuple(columns) for entity, columns in document["output_family"].items()
    } != housing_output_family():
        raise ValueError(
            "US housing source projection output family differs from the registry."
        )
    return MappingProxyType(document)


# ---------------------------------------------------------------------------
# Source outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HousingSourceOutputs:
    header: Mapping[str, object]
    tables: Mapping[str, pd.DataFrame]


def _float_buffer(series: pd.Series, label: str) -> bytes:
    if series.dtype != _FLOAT:
        raise ValueError(f"US housing source {label} requires exact float64 values.")
    values = series.to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"US housing source {label} contains nonfinite values.")
    return np.ascontiguousarray(values).tobytes()


def _bool_buffer(series: pd.Series, label: str) -> bytes:
    if series.dtype != np.dtype(np.bool_):
        raise ValueError(f"US housing source {label} requires physical boolean values.")
    if series.isna().any():
        raise ValueError(f"US housing source {label} cannot carry null booleans.")
    return np.ascontiguousarray(series.to_numpy(dtype=np.bool_)).tobytes()


def _string_buffer(series: pd.Series, label: str) -> tuple[bytes, int, int]:
    dtype = series.dtype
    if (
        not isinstance(dtype, pd.StringDtype)
        or dtype != _CANONICAL_STRING
        or dtype.storage != _CANONICAL_STRING.storage
    ):
        raise ValueError(
            f"US housing source {label} requires the canonical string representation."
        )
    nulls = series.isna().to_numpy(dtype=np.bool_, copy=True)
    payload = bytearray()
    offsets = [0]
    for value, is_null in zip(
        series.to_numpy(dtype=object, copy=False), nulls, strict=True
    ):
        if not is_null:
            payload.extend(str(value).encode("utf-8"))
        offsets.append(len(payload))
    body = (
        np.ascontiguousarray(nulls).tobytes()
        + np.asarray(offsets, dtype=_INT).tobytes()
        + bytes(payload)
    )
    return body, int(nulls.sum()), len(payload)


def encode_source_outputs(
    tables: Mapping[str, pd.DataFrame],
    *,
    projection,
    frame_context,
    validation,
    mode: str,
) -> bytes:
    """Encode the five source leaves with their exact ordered entity IDs."""
    if mode != FRESH_SOURCE_MODE:
        raise ValueError(f"Unsupported US housing source mode {mode!r}.")
    if set(tables) != set(SOURCE_ENTITIES):
        raise ValueError("US housing source outputs require its three actual tables.")
    projection_binding = binding(projection)
    context_binding = binding(frame_context)
    validation_binding = binding(validation)
    identities: dict[str, dict[str, object]] = {}
    id_body = bytearray()
    for entity in SOURCE_ENTITIES:
        table = tables[entity]
        ids = strict_ids(table[f"{entity}_id"], label=entity)
        identities[entity] = _entity_identity(ids, entity)
        id_body.extend(np.ascontiguousarray(ids).tobytes())
    columns: list[dict[str, object]] = []
    value_body = bytearray()
    for entity, column, source_dtype, _merge_dtype in SOURCE_COLUMNS:
        table = tables[entity]
        if column not in table:
            raise ValueError(f"US housing source outputs omit {entity}.{column!r}.")
        series = table[column]
        label = f"{entity}.{column}"
        if len(series) != identities[entity]["rows"]:
            raise ValueError(f"US housing source {label} length differs from its IDs.")
        record: dict[str, object] = {
            "entity": entity,
            "column": column,
            "dtype": source_dtype,
        }
        if source_dtype == "float64":
            buffer = _float_buffer(series, label)
            record.update({"encoding": "raw-f8-v1", "nulls": 0})
        elif source_dtype == "bool":
            buffer = _bool_buffer(series, label)
            record.update({"encoding": "raw-bool-v1", "nulls": 0})
        else:
            buffer, nulls, utf8_bytes = _string_buffer(series, label)
            record.update(
                {
                    "encoding": "utf8-offsets-v1",
                    "nulls": nulls,
                    "utf8_bytes": utf8_bytes,
                    "string_storage": _CANONICAL_STRING.storage,
                    "string_na_marker": "pd_na",
                }
            )
        record["value_bytes"] = len(buffer)
        record["values_sha256"] = sha(bytes(buffer))
        columns.append(record)
        value_body.extend(buffer)
    header = encode_json(
        {
            "schema_version": 1,
            "mode": mode,
            "phase": HOUSING_PHASE,
            "operator": SOURCE_OPERATOR_NAME,
            "family": SOURCE_OPERATOR_FAMILY,
            "entities": identities,
            "columns": columns,
            "projection_sha256": projection_binding["sha256"],
            "projection_producer_key": projection_binding["producer_key"],
            "frame_context_sha256": context_binding["sha256"],
            "frame_context_producer_key": context_binding["producer_key"],
            "validation_sha256": validation_binding["sha256"],
            "validation_producer_key": validation_binding["producer_key"],
        }
    )
    return _envelope(OUTPUTS_MAGIC, header, bytes(id_body) + bytes(value_body))


def _read_string(buffer: memoryview, rows: int, record: Mapping[str, object]):
    mask_bytes = rows
    offset_bytes = (rows + 1) * 8
    utf8_bytes = record["utf8_bytes"]
    if len(buffer) != mask_bytes + offset_bytes + utf8_bytes:
        raise ValueError("US housing source string buffer length differs.")
    raw_mask = np.frombuffer(buffer[:mask_bytes], dtype=np.uint8)
    if (raw_mask > 1).any():
        raise ValueError("US housing source string mask has noncanonical null bits.")
    mask = raw_mask.view(np.bool_)
    if int(mask.sum()) != record["nulls"]:
        raise ValueError("US housing source string null count differs from its header.")
    offsets = np.frombuffer(buffer[mask_bytes : mask_bytes + offset_bytes], dtype=_INT)
    payload = bytes(buffer[mask_bytes + offset_bytes :])
    if offsets[0] != 0 or offsets[-1] != len(payload) or (np.diff(offsets) < 0).any():
        raise ValueError("US housing source string offsets are invalid.")
    values: list[object] = []
    for position in range(rows):
        start, stop = int(offsets[position]), int(offsets[position + 1])
        if mask[position]:
            if start != stop:
                raise ValueError(
                    "US housing source string hides bytes beneath a null value."
                )
            values.append(pd.NA)
            continue
        try:
            values.append(payload[start:stop].decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError(
                "US housing source string payload is not valid UTF-8."
            ) from error
    return pd.Series(values, dtype=_CANONICAL_STRING)


def decode_source_outputs(payload: bytes) -> HousingSourceOutputs:
    header, body = _split(payload, OUTPUTS_MAGIC)
    if (
        set(header) != _OUTPUT_FIELDS
        or type(header["schema_version"]) is not int
        or header["schema_version"] != 1
        or header["mode"] != FRESH_SOURCE_MODE
        or header["phase"] != HOUSING_PHASE
        or header["operator"] != SOURCE_OPERATOR_NAME
        or header["family"] != SOURCE_OPERATOR_FAMILY
        or any(
            not is_hash(header[key])
            for key in (
                "projection_sha256",
                "projection_producer_key",
                "frame_context_sha256",
                "frame_context_producer_key",
                "validation_sha256",
                "validation_producer_key",
            )
        )
    ):
        raise ValueError("Unsupported US housing source outputs schema.")
    entities = header["entities"]
    if not isinstance(entities, dict) or set(entities) != set(SOURCE_ENTITIES):
        raise ValueError("US housing source outputs describe the wrong entities.")
    for value in entities.values():
        _validate_identity(value)
    records = header["columns"]
    if not isinstance(records, list) or len(records) != len(SOURCE_COLUMNS):
        raise ValueError("US housing source outputs must carry its five leaves.")
    expected_length = sum(entities[entity]["rows"] * 8 for entity in SOURCE_ENTITIES)
    for record, (entity, column, source_dtype, _merge) in zip(
        records, SOURCE_COLUMNS, strict=True
    ):
        fields = _STRING_COLUMN_FIELDS if source_dtype == "string" else _COLUMN_FIELDS
        if (
            not isinstance(record, dict)
            or set(record) != fields
            or record["entity"] != entity
            or record["column"] != column
            or record["dtype"] != source_dtype
            or type(record["nulls"]) is not int
            or record["nulls"] < 0
            or type(record["value_bytes"]) is not int
            or record["value_bytes"] < 0
            or not is_hash(record["values_sha256"])
        ):
            raise ValueError(
                "US housing source output column record is malformed or reordered."
            )
        if source_dtype == "string" and (
            record["encoding"] != "utf8-offsets-v1"
            or record["string_storage"] != _CANONICAL_STRING.storage
            or record["string_na_marker"] != "pd_na"
            or type(record["utf8_bytes"]) is not int
            or record["utf8_bytes"] < 0
        ):
            raise ValueError("US housing source string record is malformed.")
        if source_dtype != "string" and record["encoding"] not in {
            "raw-f8-v1",
            "raw-bool-v1",
        }:
            raise ValueError("US housing source scalar record is malformed.")
        expected_length += record["value_bytes"]
    if len(body) != expected_length:
        raise ValueError("US housing source outputs body length differs from header.")

    tables: dict[str, pd.DataFrame] = {}
    cursor = 0
    for entity in SOURCE_ENTITIES:
        rows = entities[entity]["rows"]
        ids = np.frombuffer(body[cursor : cursor + rows * 8], dtype=_INT)
        cursor += rows * 8
        series = pd.Series(ids, dtype="int64")
        if not series.is_unique or (ids < 0).any():
            raise ValueError(
                f"US housing source outputs {entity} IDs must be unique and "
                "nonnegative."
            )
        actual = _entity_identity(ids, entity)
        if encode_json(actual) != encode_json(entities[entity]):
            raise ValueError(
                f"US housing source outputs {entity} IDs differ from their identity."
            )
        tables[entity] = pd.DataFrame({f"{entity}_id": series})
    for record, (entity, column, source_dtype, _merge) in zip(
        records, SOURCE_COLUMNS, strict=True
    ):
        rows = entities[entity]["rows"]
        buffer = body[cursor : cursor + record["value_bytes"]]
        cursor += record["value_bytes"]
        if sha(bytes(buffer)) != record["values_sha256"]:
            raise ValueError(
                f"US housing source {entity}.{column} bytes differ from their digest."
            )
        if source_dtype == "float64":
            if len(buffer) != rows * 8:
                raise ValueError("US housing source float buffer length differs.")
            values = np.frombuffer(buffer, dtype=_FLOAT)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"US housing source {entity}.{column} decoded nonfinite values."
                )
            series = pd.Series(values.copy(), dtype="float64")
        elif source_dtype == "bool":
            if len(buffer) != rows:
                raise ValueError("US housing source boolean buffer length differs.")
            raw = np.frombuffer(buffer, dtype=np.uint8)
            if (raw > 1).any():
                raise ValueError(
                    f"US housing source {entity}.{column} has noncanonical bool bytes."
                )
            series = pd.Series(raw.view(np.bool_).copy(), dtype="bool")
        else:
            series = _read_string(buffer, rows, record)
        assert_dtype(series, source_dtype, label=f"US housing source {entity}.{column}")
        tables[entity][column] = series
    if cursor != len(body):
        raise ValueError("US housing source outputs body has trailing bytes.")
    return HousingSourceOutputs(
        MappingProxyType(header),
        MappingProxyType({entity: table for entity, table in tables.items()}),
    )


# ---------------------------------------------------------------------------
# Merge evidence
# ---------------------------------------------------------------------------


def cell_digest(series: pd.Series, *, entity: str, column: str) -> dict[str, object]:
    """Digest one column's exact values and null pattern for gate replay."""
    nulls = series.isna().to_numpy(dtype=np.bool_, copy=True)
    if isinstance(series.dtype, pd.StringDtype):
        parts = _string_buffer(series.astype(_CANONICAL_STRING), f"{entity}.{column}")[
            0
        ]
    elif isinstance(series.dtype, pd.BooleanDtype):
        filled = series.fillna(False).to_numpy(dtype=np.bool_)
        parts = (
            np.ascontiguousarray(filled).tobytes()
            + np.ascontiguousarray(nulls).tobytes()
        )
    elif series.dtype == _FLOAT:
        values = series.to_numpy(dtype=np.float64, copy=True)
        values[nulls] = 0.0
        parts = (
            np.ascontiguousarray(values).tobytes()
            + np.ascontiguousarray(nulls).tobytes()
        )
    else:
        raise ValueError(
            f"US housing merge cannot digest {entity}.{column} dtype {series.dtype!s}."
        )
    return {
        "entity": entity,
        "column": column,
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "digest": sha(parts),
    }


def _validate_cells(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("US housing merge evidence cells are malformed.")
    for record in value:
        if (
            not isinstance(record, dict)
            or set(record) != _CELL_FIELDS
            or type(record["rows"]) is not int
            or record["rows"] < 0
            or not is_hash(record["digest"])
        ):
            raise ValueError("US housing merge evidence cell record is malformed.")


def encode_merge_evidence(
    *,
    projection,
    source_outputs,
    pending_context: bytes,
    merged_rows: Mapping[str, int],
    selected_rows: Mapping[str, int],
    pool_rows: Mapping[str, int],
    merged_cells: Sequence[Mapping[str, object]],
    native_cells: Sequence[Mapping[str, object]],
    source_preparation: Mapping[str, object],
    mode: str,
) -> bytes:
    projection_binding = binding(projection)
    outputs_binding = binding(source_outputs)
    if not isinstance(pending_context, bytes):
        raise ValueError("US housing merge evidence requires the pending bytes.")
    return encode_json(
        {
            "schema_version": 1,
            "mode": mode,
            "phase": HOUSING_PHASE,
            "scope": HOUSING_GATE_SCOPE,
            "projection_sha256": projection_binding["sha256"],
            "projection_producer_key": projection_binding["producer_key"],
            "source_outputs_sha256": outputs_binding["sha256"],
            "source_outputs_producer_key": outputs_binding["producer_key"],
            "pending_context_sha256": sha(pending_context),
            "merged_rows": dict(merged_rows),
            "selected_rows": dict(selected_rows),
            "pool_rows": dict(pool_rows),
            "merged_cells": [dict(cell) for cell in merged_cells],
            "native_cells": [dict(cell) for cell in native_cells],
            "source_preparation": source_preparation,
        }
    )


def decode_merge_evidence(payload: bytes) -> Mapping[str, object]:
    if not isinstance(payload, bytes) or len(payload) > HEADER_MAX_BYTES:
        raise ValueError("US housing merge evidence exceeds its bound.")
    document = decode_json(payload)
    if (
        set(document) != _MERGE_EVIDENCE_FIELDS
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["mode"] != FRESH_SOURCE_MODE
        or document["phase"] != HOUSING_PHASE
        or document["scope"] != HOUSING_GATE_SCOPE
        or any(
            not is_hash(document[key])
            for key in (
                "projection_sha256",
                "projection_producer_key",
                "source_outputs_sha256",
                "source_outputs_producer_key",
                "pending_context_sha256",
            )
        )
    ):
        raise ValueError("Unsupported US housing merge evidence schema.")
    for key in ("merged_rows", "selected_rows", "pool_rows"):
        value = document[key]
        if not isinstance(value, dict) or any(
            type(count) is not int or count < 0 for count in value.values()
        ):
            raise ValueError(f"US housing merge evidence {key} is malformed.")
    _validate_cells(document["merged_cells"])
    _validate_cells(document["native_cells"])
    receipt = document["source_preparation"]
    if (
        not isinstance(receipt, dict)
        or receipt.get("phase") != SOURCE_OPERATOR_PHASE
        or receipt.get("operator_order") != [SOURCE_OPERATOR_NAME]
    ):
        raise ValueError(
            "US housing merge evidence carries a different source operation."
        )
    return MappingProxyType(document)
