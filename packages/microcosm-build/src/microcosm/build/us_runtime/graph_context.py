"""Identity-bound metadata for real US operators in the population graph.

Population cells, weights and strata remain executor-owned. This typed artifact
carries the metadata and prior mass records that existing US operators require,
so they cannot silently reload context from an unrelated mutable checkpoint.
Reconstruction uses only the node's declared table views; it grants no access to
undeclared columns or a live population object.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, MassChangeRecord
from microcosm.graph import ArtifactType, KernelContext
from microcosm.graph.canonical import canonical_json

US_FRAME_CONTEXT_TYPE = ArtifactType("microcosm.us.frame_context", 1)

_FIELDS = {"schema_version", "entities", "metadata", "mass_log", "weight_sources"}
_ENTITY_FIELDS = {"columns", "rows", "id_dtype", "ordered_ids_sha256"}
_MASS_FIELDS = {"entity", "old_total", "new_total", "declared_factor", "reason"}

# These five receipts are read as authority by the existing US operators.
# Operational observations belong in run receipts, outside this keyed edge.
# An additional normative receipt requires an explicit contract revision here.
US_NORMATIVE_METADATA_KEYS = frozenset(
    {
        "us_spine_assembly_manifest",
        "us_stacked_spine_manifest",
        "us_puf_clone_attachment_manifest",
        "us_late_producer_transition_authority",
        "acs_pums_earnings_universe_application",
    }
)
_RUN_FIELDS = frozenset(
    {
        "created_at",
        "updated_at",
        "timestamp",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "wall_time_seconds",
        "duration_seconds",
        "peak_rss",
        "hostname",
        "host",
        "pid",
        "run_id",
        "path",
        "checkpoint_dir",
        "output_dir",
        "timings",
        "cache_hit",
        "cache_hits",
    }
)


def _normative_metadata(metadata: Mapping[str, object]) -> object:
    unknown = set(metadata) - US_NORMATIVE_METADATA_KEYS
    if unknown:
        raise ValueError(
            f"US graph context has undeclared metadata: {sorted(unknown)}."
        )

    def validate(value: object) -> None:
        if isinstance(value, Mapping):
            if set(value) & _RUN_FIELDS:
                raise ValueError("US graph context cannot contain run-level metadata.")
            for item in value.values():
                validate(item)
        elif isinstance(value, list | tuple):
            for item in value:
                validate(item)

    validate(metadata)
    return _json_data(metadata)


def _json_data(value: object) -> object:
    """Normalize NumPy metadata without coercing unknown objects to strings."""
    if isinstance(value, np.generic):
        return _json_data(value.item())
    if isinstance(value, np.ndarray):
        return _json_data(value.tolist())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("US graph context metadata requires string mapping keys.")
        return {key: _json_data(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_data(item) for item in value]
    # canonical_json validates scalar types and refuses nonfinite numbers.
    return value


def _row_identity(table: pd.DataFrame, entity: str) -> dict[str, object]:
    column = US_SCHEMA.entity_id_column(entity)
    if column not in table:
        raise ValueError(f"US graph context {entity} identity column is missing.")
    ids = table[column]
    if not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any():
        raise ValueError(f"US graph context {entity} identity must be integral.")
    if len(ids) and (int(ids.min()) < 0 or int(ids.max()) > np.iinfo(np.int64).max):
        raise ValueError(f"US graph context {entity} identity exceeds int64 bounds.")
    values = ids.to_numpy(dtype="<i8")
    digest = hashlib.sha256(b"microcosm.us.graph-context.ordered-ids.v1\0")
    digest.update(values.tobytes(order="C"))
    return {
        "rows": len(ids),
        "id_dtype": str(ids.dtype),
        "ordered_ids_sha256": digest.hexdigest(),
    }


def encode_us_frame_context(frame: Frame) -> bytes:
    """Encode current US metadata and row identities, without population cells."""
    if frame.schema != US_SCHEMA:
        raise ValueError("US graph context requires the US entity schema.")
    entities = {}
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        if entity in US_SCHEMA.group_entities and (
            not isinstance(table.index, pd.RangeIndex)
            or table.index.start != 0
            or table.index.stop != len(table)
            or table.index.step != 1
            or table.index.name is not None
        ):
            # Version 1 carries group IDs, not pandas index descriptors. An
            # identity FILTER uses Frame.select, which resets group indices.
            # Refuse an unsupported representation before emitting authority
            # that downstream person-only nodes cannot independently inspect.
            raise ValueError(
                f"US graph context {entity} requires a default unnamed RangeIndex."
            )
        if any(not isinstance(column, str) for column in table.columns):
            raise ValueError("US graph context column names must be strings.")
        entities[entity] = {
            "columns": list(table.columns),
            **_row_identity(table, entity),
        }
    document = {
        "schema_version": US_FRAME_CONTEXT_TYPE.schema_version,
        "entities": entities,
        "metadata": _normative_metadata(frame.metadata),
        "mass_log": [_json_data(asdict(record)) for record in frame.mass_log],
        "weight_sources": {
            entity: frame.weights_for(entity).kind.value
            for entity in frame.weighted_entities
        },
    }
    return canonical_json(document)


def _decode(payload: bytes) -> dict[str, object]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate US graph context field {key!r}.")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=unique_pairs)
        canonical_json(document)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            "US graph context must be finite, unambiguous JSON."
        ) from error
    if (
        not isinstance(document, dict)
        or set(document) != _FIELDS
        or type(document["schema_version"]) is not int
        or document["schema_version"] != US_FRAME_CONTEXT_TYPE.schema_version
    ):
        raise ValueError("US graph context has an unsupported schema.")
    entities = document["entities"]
    if not isinstance(entities, dict) or set(entities) != set(US_SCHEMA.entities):
        raise ValueError("US graph context must describe exactly the US entities.")
    for entity, value in entities.items():
        if not isinstance(value, dict) or set(value) != _ENTITY_FIELDS:
            raise ValueError(f"US graph context {entity} identity is malformed.")
        columns = value["columns"]
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
            or US_SCHEMA.entity_id_column(entity) not in columns
            or type(value["rows"]) is not int
            or value["rows"] < 0
        ):
            raise ValueError(f"US graph context {entity} schema is malformed.")
    if not isinstance(document["metadata"], dict):
        raise ValueError("US graph context metadata must be an object.")
    _normative_metadata(document["metadata"])
    weight_sources = document["weight_sources"]
    if (
        not isinstance(weight_sources, dict)
        or not weight_sources
        or set(weight_sources) - set(US_SCHEMA.entities)
        or any(not isinstance(kind, str) for kind in weight_sources.values())
    ):
        raise ValueError("US graph context weight sources are malformed.")
    if not isinstance(document["mass_log"], list):
        raise ValueError("US graph context mass_log must be an array.")
    return document


def _mass_records(raw: list[object]) -> tuple[MassChangeRecord, ...]:
    records = []
    for value in raw:
        if not isinstance(value, dict) or set(value) != _MASS_FIELDS:
            raise ValueError("US graph context mass record is malformed.")
        if value["entity"] not in US_SCHEMA.entities or not isinstance(
            value["reason"], str
        ):
            raise ValueError("US graph context mass record has invalid authority.")
        for field in ("old_total", "new_total", "declared_factor"):
            number = value[field]
            if field == "declared_factor" and number is None:
                continue
            if type(number) not in (int, float) or number < 0:
                raise ValueError("US graph context mass record is invalid.")
        records.append(MassChangeRecord(**value))
    return tuple(records)


def us_frame_from_context(
    context: KernelContext, *, artifact_alias: str = "frame_context"
) -> Frame:
    """Restore an isolated US operator view from declared cells and typed context.

    The complete row identity of each entity must match the artifact. Columns
    omitted by the node stay omitted. Callers needing a narrower row population
    must declare a structural graph stage and publish its own context first.
    """
    artifact = context.artifacts.get(artifact_alias)
    if artifact is None or artifact.type != US_FRAME_CONTEXT_TYPE:
        raise ValueError("US graph context artifact has a missing or incorrect type.")
    document = _decode(artifact.payload)
    if set(context.tables) != set(US_SCHEMA.entities):
        raise ValueError("US graph context requires declared views for every entity.")
    tables = {}
    for entity in US_SCHEMA.entities:
        table = context.tables[entity]
        declared = document["entities"][entity]
        actual = _row_identity(table, entity)
        if any(declared[key] != value for key, value in actual.items()):
            raise ValueError(f"US graph context {entity} identity does not match.")
        if set(table.columns) - set(declared["columns"]):
            raise ValueError(f"US graph context {entity} includes unknown columns.")
        ordered = [column for column in declared["columns"] if column in table]
        tables[entity] = table.loc[:, ordered].copy(deep=True)
    weights = {}
    for entity, kind in document["weight_sources"].items():
        value = context.weights.get(entity)
        if value is None or value.kind.value != kind:
            raise ValueError(
                f"US graph context has no matching {entity} weight source."
            )
        weights[entity] = value
    return Frame(
        tables,
        US_SCHEMA,
        weights,
        context.strata.copy(deep=True),
        mass_log=_mass_records(document["mass_log"]),
        metadata=document["metadata"],
    )
