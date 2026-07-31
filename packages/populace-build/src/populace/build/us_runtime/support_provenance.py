"""US support-source provenance and operator-role helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd

__all__ = [
    "BASE_ASEC_SUPPORT_CHANNEL",
    "PERSON_SUPPORT_CHANNEL_COLUMN",
    "PUF_TAX_DETAIL_CLONE_INDEX",
    "PUF_TAX_DETAIL_SUPPORT_CHANNEL",
    "SPINE_ASSEMBLY_MANIFEST_KEY",
    "has_support_role_metadata",
    "puf_tax_detail_clone_mask",
    "spine_assembly_manifest",
    "spine_assembly_receipt",
    "spine_provenance_counts",
    "spine_source_id_column",
    "support_channel_column",
    "support_clone_index_column",
    "support_role_series",
    "support_source_id_column",
    "validate_assembly_provenance",
    "without_support_role_metadata",
]

BASE_ASEC_SUPPORT_CHANNEL = "asec"
PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
PUF_TAX_DETAIL_CLONE_INDEX = 1
SPINE_ASSEMBLY_MANIFEST_KEY = "us_spine_assembly_manifest"
_SPINE_ASSEMBLY_MANIFEST_VERSION = 1


class _ProvenanceSchema(Protocol):
    person_entity: str
    group_entities: Sequence[str]

    def entity_id_column(self, entity: str) -> str: ...

    def membership_column(self, entity: str) -> str: ...


class _ProvenanceFrame(Protocol):
    entities: Sequence[str]
    metadata: Mapping[str, Any]
    schema: _ProvenanceSchema

    def table(self, entity: str) -> pd.DataFrame: ...


def spine_assembly_manifest(
    tables: Mapping[str, pd.DataFrame],
    *,
    channels: Sequence[str],
) -> dict[str, object]:
    """Build deeply frozen channel/count receipt data for one assembly output."""

    declared_channels = tuple(channels)
    if len(declared_channels) < 2 or len(set(declared_channels)) != len(
        declared_channels
    ):
        raise ValueError(
            "Spine assembly manifest requires at least two unique channels."
        )
    row_counts: dict[str, dict[str, int]] = {}
    for entity, table in tables.items():
        channel_column = support_channel_column(entity)
        if channel_column not in table:
            raise ValueError(
                f"Spine assembly manifest cannot be built; missing {channel_column!r}."
            )
        row_counts[entity] = {
            channel: int(table[channel_column].eq(channel).sum())
            for channel in declared_channels
        }
    return {
        SPINE_ASSEMBLY_MANIFEST_KEY: {
            "version": _SPINE_ASSEMBLY_MANIFEST_VERSION,
            "channels": declared_channels,
            "native_row_counts": row_counts,
        }
    }


def validate_assembly_provenance(
    frame: _ProvenanceFrame,
    *,
    boundary: str,
    require_manifest: bool = True,
) -> Mapping[str, Any] | None:
    """Validate live support provenance against the frozen assembly manifest.

    Counts cover clone-index-zero rows, so clone operators may add role copies
    without rewriting the source assembly receipt. Every live channel must be
    declared, native counts must remain exact, and each person's channel must
    agree with every group row referenced by that person.
    """

    metadata = getattr(frame, "metadata", {})
    manifest = metadata.get(SPINE_ASSEMBLY_MANIFEST_KEY)
    if manifest is None:
        if require_manifest:
            raise ValueError(
                f"{boundary}: receipt-validated support provenance has no assembly "
                f"manifest {SPINE_ASSEMBLY_MANIFEST_KEY!r}."
            )
        return None
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{boundary}: assembly manifest is malformed.")
    if manifest.get("version") != _SPINE_ASSEMBLY_MANIFEST_VERSION:
        raise ValueError(
            f"{boundary}: assembly manifest has unsupported version "
            f"{manifest.get('version')!r}."
        )
    raw_channels = manifest.get("channels")
    raw_counts = manifest.get("native_row_counts")
    if (
        not isinstance(raw_channels, Sequence)
        or isinstance(raw_channels, (str, bytes))
        or not isinstance(raw_counts, Mapping)
    ):
        raise ValueError(f"{boundary}: assembly manifest is malformed.")
    channels = tuple(raw_channels)
    if (
        len(channels) < 2
        or len(set(channels)) != len(channels)
        or any(not isinstance(channel, str) or not channel for channel in channels)
    ):
        raise ValueError(
            f"{boundary}: assembly manifest declares invalid channels "
            f"{list(channels)!r}."
        )
    declared = set(channels)

    for entity in frame.entities:
        table = frame.table(entity)
        channel_column = support_channel_column(entity)
        clone_index_column = support_clone_index_column(entity)
        missing = [
            column
            for column in (channel_column, clone_index_column)
            if column not in table
        ]
        if missing:
            raise ValueError(
                f"{boundary}: assembly manifest provenance is incomplete for "
                f"{entity!r}; missing {missing}."
            )
        channel_values = table[channel_column]
        invalid_channels = channel_values.isna() | ~channel_values.map(
            lambda value: isinstance(value, str) and bool(value)
        )
        if invalid_channels.any():
            raise ValueError(
                f"{boundary}: assembly manifest provenance column "
                f"{channel_column!r} has invalid value(s)."
            )
        observed = set(channel_values.astype(str))
        unknown = sorted(observed - declared)
        if unknown:
            raise ValueError(
                f"{boundary}: assembly manifest declares channels "
                f"{list(channels)!r}, but {channel_column!r} contains unknown "
                f"channel(s) {unknown}."
            )

        numeric_clone_indices = pd.to_numeric(
            table[clone_index_column],
            errors="coerce",
        )
        clone_indices = numeric_clone_indices.to_numpy(dtype=np.float64)
        if (
            numeric_clone_indices.isna().any()
            or (clone_indices < 0.0).any()
            or not np.equal(clone_indices, np.floor(clone_indices)).all()
        ):
            raise ValueError(
                f"{boundary}: assembly manifest provenance column "
                f"{clone_index_column!r} must contain nonnegative integers."
            )
        expected_by_channel = raw_counts.get(entity)
        if (
            not isinstance(expected_by_channel, Mapping)
            or set(expected_by_channel) != declared
        ):
            raise ValueError(
                f"{boundary}: assembly manifest row counts for {entity!r} "
                "do not exactly cover its declared channels."
            )
        native = clone_indices == 0.0
        actual_counts = {
            channel: int(
                np.count_nonzero(
                    native & channel_values.astype(str).eq(channel).to_numpy()
                )
            )
            for channel in channels
        }
        try:
            expected_counts = {
                channel: int(expected_by_channel[channel]) for channel in channels
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{boundary}: assembly manifest row counts for {entity!r} "
                "must be integers."
            ) from exc
        if any(value < 0 for value in expected_counts.values()):
            raise ValueError(
                f"{boundary}: assembly manifest row counts for {entity!r} "
                "must be nonnegative."
            )
        if actual_counts != expected_counts:
            raise ValueError(
                f"{boundary}: live {entity!r} native row counts drifted from "
                f"the assembly manifest; expected {expected_counts}, observed "
                f"{actual_counts}."
            )

    person_entity = frame.schema.person_entity
    person = frame.table(person_entity)
    person_channel_column = support_channel_column(person_entity)
    for group in frame.schema.group_entities:
        membership_column = frame.schema.membership_column(group)
        group_id_column = frame.schema.entity_id_column(group)
        group_channel_column = support_channel_column(group)
        group_channels = frame.table(group).set_index(group_id_column)[
            group_channel_column
        ]
        expected = person[membership_column].map(group_channels)
        mismatch = expected.isna() | expected.astype(str).ne(
            person[person_channel_column].astype(str)
        )
        if mismatch.any():
            raise ValueError(
                f"{boundary}: cross-grain provenance disagrees with the "
                f"assembly manifest on {int(mismatch.sum())} person/{group} "
                "link(s); each person's support channel must match its "
                f"{group} channel."
            )
    return manifest


def spine_assembly_receipt(
    frame: _ProvenanceFrame,
    *,
    boundary: str,
) -> dict[str, object]:
    """Return a mutable, JSON-ready copy of a validated assembly receipt.

    :class:`~populace.frame.Frame` freezes metadata recursively. Build
    manifests need ordinary dictionaries and lists, but must only publish a
    receipt after validating it against the live provenance columns.
    """

    manifest = validate_assembly_provenance(frame, boundary=boundary)
    if manifest is None:  # pragma: no cover - require_manifest defaults true
        raise AssertionError("Validated assembly receipt unexpectedly absent.")
    return _json_ready_mapping(manifest)


def spine_provenance_counts(
    frame: _ProvenanceFrame,
    *,
    boundary: str,
) -> dict[str, dict[str, object]]:
    """Count every source channel and clone index without exposing routing.

    This reporting helper deliberately lives with the provenance owner.
    Population operators remain unable to branch on source identity; manifests
    can still publish a complete per-entity receipt after the shared validator
    proves the live columns agree with assembly.
    """

    manifest = validate_assembly_provenance(frame, boundary=boundary)
    if manifest is None:  # pragma: no cover - require_manifest defaults true
        raise AssertionError("Validated assembly receipt unexpectedly absent.")
    channels = tuple(str(channel) for channel in manifest["channels"])
    counts: dict[str, dict[str, object]] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        source = table[support_channel_column(entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(entity)],
            errors="raise",
        ).astype("int64")
        observed_clone_indices = sorted(int(value) for value in clone_index.unique())
        counts[entity] = {
            "rows": int(len(table)),
            "by_source_channel": {
                channel: int(source.eq(channel).sum()) for channel in channels
            },
            "by_clone_index": {
                str(index): int(clone_index.eq(index).sum())
                for index in observed_clone_indices
            },
            "by_source_channel_and_clone_index": {
                channel: {
                    str(index): int((source.eq(channel) & clone_index.eq(index)).sum())
                    for index in observed_clone_indices
                }
                for channel in channels
            },
        }
    return counts


def has_support_role_metadata(
    table: pd.DataFrame,
    *,
    entity: str,
) -> bool:
    """Return whether clone-role or legacy support-role metadata is present."""

    return (
        support_clone_index_column(entity) in table
        or support_channel_column(entity) in table
    )


def spine_source_id_column(entity: str) -> str:
    """Return the entity-prefixed raw spine-record ID metadata column."""

    _require_entity_name(entity)
    return f"{entity}_spine_source_id"


def support_channel_column(entity: str) -> str:
    """Return the entity-prefixed source-support metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_channel"


def support_clone_index_column(entity: str) -> str:
    """Return the entity-prefixed clone-index metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_clone_index"


def support_source_id_column(entity: str) -> str:
    """Return the entity-prefixed assembly-unique pre-clone ID column."""

    _require_entity_name(entity)
    return f"{entity}_source_id"


def without_support_role_metadata(
    table: pd.DataFrame,
    *,
    entity: str,
) -> pd.DataFrame:
    """Copy a table without source-channel or clone-role metadata."""

    return table.drop(
        columns=[
            support_channel_column(entity),
            support_clone_index_column(entity),
        ],
        errors="ignore",
    ).copy(deep=True)


def support_role_series(
    table: pd.DataFrame,
    *,
    entity: str,
) -> pd.Series:
    """Return legacy-compatible operator roles derived from clone provenance.

    Native records have the ASEC-compatible role and every donor-detail clone
    has the PUF-compatible role. Clone provenance takes precedence. Legacy
    fixtures without clone indices may use the two historical role labels in
    their support-channel column.
    """

    clone_index_column = support_clone_index_column(entity)
    if clone_index_column not in table:
        channel_column = support_channel_column(entity)
        if channel_column not in table:
            raise ValueError(
                "PUF support metadata is missing both "
                f"{clone_index_column!r} and {channel_column!r}."
            )
        channels = table[channel_column]
        if channels.isna().any():
            raise ValueError(
                f"Legacy support metadata column {channel_column!r} requires "
                "complete support provenance."
            )
        valid = channels.isin(
            (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        )
        if not valid.all():
            invalid = sorted(set(channels.loc[~valid].astype(str)))
            raise ValueError(
                f"Legacy support metadata column {channel_column!r} must contain "
                "the exact ASEC/PUF roles; unsupported value(s): "
                f"{invalid}."
            )
        return pd.Series(
            channels.to_numpy(dtype=object, copy=True),
            index=table.index,
            name=f"{entity}_support_role",
        )
    numeric = pd.to_numeric(table[clone_index_column], errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"PUF support metadata column {clone_index_column!r} must be integral."
        )
    clone_indices = numeric.to_numpy(dtype=np.float64)
    if (clone_indices < 0).any() or not np.equal(
        clone_indices, np.floor(clone_indices)
    ).all():
        raise ValueError(
            f"PUF support metadata column {clone_index_column!r} must contain "
            "nonnegative integers."
        )
    channel_column = support_channel_column(entity)
    if channel_column not in table:
        raise ValueError(
            f"PUF support metadata requires complete support provenance; "
            f"missing {channel_column!r}."
        )
    channels = table[channel_column]
    invalid_channels = channels.isna() | ~channels.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if invalid_channels.any():
        raise ValueError(
            f"PUF support metadata column {channel_column!r} requires complete "
            "support provenance."
        )

    # Before multispine assembly, this column carried the operator role. Keep
    # validating that historical contract so malformed current-lineage frames
    # still fail closed. Assembled frames carry a raw spine ID and use the
    # channel for receipt-declared source identity, so arbitrary declared names
    # are valid and never influence the returned operator role.
    if spine_source_id_column(entity) not in table:
        valid = channels.isin(
            (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        )
        if not valid.all():
            invalid = sorted(set(channels.loc[~valid].astype(str)))
            raise ValueError(
                f"Legacy support metadata column {channel_column!r} has an "
                "unsupported support channel; expected exact ASEC/PUF roles, "
                f"got {invalid}."
            )
        inconsistent = channels.eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy() != (
            clone_indices == 0
        )
        if inconsistent.any():
            raise ValueError(
                f"Legacy support metadata columns {channel_column!r} and "
                f"{clone_index_column!r} are inconsistent."
            )
    return pd.Series(
        np.where(
            clone_indices == 0,
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ),
        index=table.index,
        name=f"{entity}_support_role",
    )


def puf_tax_detail_clone_mask(
    table: pd.DataFrame,
    *,
    entity: str,
) -> np.ndarray:
    """Select primary PUF-detail clones without reading their source spine."""

    roles = support_role_series(table, entity=entity)
    clone_index_column = support_clone_index_column(entity)
    if clone_index_column not in table:
        return roles.eq(PUF_TAX_DETAIL_SUPPORT_CHANNEL).to_numpy()
    clone_indices = pd.to_numeric(
        table[clone_index_column],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    return clone_indices == PUF_TAX_DETAIL_CLONE_INDEX


def _require_entity_name(entity: str) -> None:
    if not isinstance(entity, str) or not entity:
        raise ValueError("entity must be a non-empty string.")


def _json_ready_mapping(value: Mapping[str, Any]) -> dict[str, object]:
    """Deep-copy frozen receipt values into JSON-compatible containers."""

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(nested) for key, nested in item.items()}
        if isinstance(item, tuple):
            return [thaw(nested) for nested in item]
        if isinstance(item, list):
            return [thaw(nested) for nested in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return {str(key): thaw(item) for key, item in value.items()}
