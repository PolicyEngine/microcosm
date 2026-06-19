"""US support expansion for PUF tax-detail imputations.

The PUF tax-detail donor needs a distinct support channel: the ASEC/CPS
records remain as the baseline channel, and a cloned channel receives
PUF-sourced tax detail without overwriting the baseline rows. The expansion is
mass-conserving: with two support channels, each channel receives half of the
incoming weights so the frame's aggregate population does not double.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from populace.frame import US_SCHEMA, Frame, Weights
from populace.frame.schema import EntitySchema

__all__ = [
    "BASE_ASEC_SUPPORT_CHANNEL",
    "PUF_TAX_DETAIL_SUPPORT_CHANNEL",
    "US_PUF_SUPPORT_STAGE_NAME",
    "clone_us_frame_for_puf_support",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
]

BASE_ASEC_SUPPORT_CHANNEL = "asec"
PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
US_PUF_SUPPORT_STAGE_NAME = "puf_support_channel"

_DEFAULT_SUPPORT_CHANNELS = (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
)


def clone_us_frame_for_puf_support(
    frame: Frame,
    *,
    channels: Sequence[str] = _DEFAULT_SUPPORT_CHANNELS,
) -> Frame:
    """Clone a US frame into support channels for PUF detail imputation.

    Args:
        frame: A US-schema frame after unit assignment and CPS-carried
            derivations.
        channels: Ordered support-channel names. The first channel keeps the
            original IDs; later channels receive remapped IDs. Weights are
            split evenly across channels.

    Returns:
        A new frame with every entity table cloned once per support channel,
        channel/source metadata added with entity-prefixed column names, all
        structural IDs remapped consistently, and typed weights mass-conserved.

    Raises:
        ValueError: If the frame is not US-schema, channel names are invalid,
            metadata columns already exist, or an ID remapping would collide.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF support expansion currently requires the US schema.")
    support_channels = _validate_channels(channels)
    _reject_metadata_collisions(frame, support_channels)
    id_multiplier = _id_multiplier_for_frame(frame)

    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        tables[entity] = _clone_entity_table(
            frame.table(entity),
            entity=entity,
            schema=frame.schema,
            channels=support_channels,
            id_multiplier=id_multiplier,
        )
    for link_name in frame.links:
        tables[link_name] = _clone_link_table(
            frame.link(link_name),
            link_name=link_name,
            schema=frame.schema,
            channels=support_channels,
            id_multiplier=id_multiplier,
        )

    weights = {
        entity: Weights(
            values=np.tile(
                frame.weights_for(entity).values / len(support_channels),
                len(support_channels),
            ),
            kind=frame.weights_for(entity).kind,
        )
        for entity in frame.weighted_entities
    }
    strata = pd.concat(
        [frame.strata.copy() for _channel in support_channels],
        ignore_index=True,
    )
    return Frame(
        tables,
        frame.schema,
        weights,
        strata,
        mass_log=frame.mass_log,
    )


def support_channel_column(entity: str) -> str:
    """Return the entity-prefixed support-channel metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_channel"


def support_clone_index_column(entity: str) -> str:
    """Return the entity-prefixed clone-index metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_clone_index"


def support_source_id_column(entity: str) -> str:
    """Return the entity-prefixed original-ID provenance column."""

    _require_entity_name(entity)
    return f"{entity}_source_id"


def _clone_entity_table(
    table: pd.DataFrame,
    *,
    entity: str,
    schema: EntitySchema,
    channels: tuple[str, ...],
    id_multiplier: int,
) -> pd.DataFrame:
    id_columns = _entity_id_columns(schema, entity)
    source_id = support_source_id_column(entity)
    channel_column = support_channel_column(entity)
    clone_index_column = support_clone_index_column(entity)
    primary_id = schema.entity_id_column(entity)

    clones: list[pd.DataFrame] = []
    for clone_index, channel in enumerate(channels):
        clone = table.copy(deep=True)
        clone[source_id] = table[primary_id].to_numpy()
        clone[channel_column] = channel
        clone[clone_index_column] = clone_index
        for column in id_columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        clones.append(clone)
    result = pd.concat(clones, ignore_index=True)
    if result[primary_id].duplicated().any():
        duplicates = result.loc[result[primary_id].duplicated(), primary_id].unique()
        raise ValueError(
            f"remapped {primary_id!r} values are not unique; id multiplier "
            "is too small. Duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return result


def _clone_link_table(
    table: pd.DataFrame,
    *,
    link_name: str,
    schema: EntitySchema,
    channels: tuple[str, ...],
    id_multiplier: int,
) -> pd.DataFrame:
    links = {link.name: link for link in schema.links}
    link = links[link_name]
    id_columns = (
        schema.entity_id_column(link.left_entity),
        schema.entity_id_column(link.right_entity),
    )
    missing = [column for column in id_columns if column not in table]
    if missing:
        raise ValueError(
            f"link table {link_name!r} is missing ID column(s): {missing}."
        )

    clones: list[pd.DataFrame] = []
    for clone_index, _channel in enumerate(channels):
        clone = table.copy(deep=True)
        for column in id_columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        clones.append(clone)
    return pd.concat(clones, ignore_index=True)


def _entity_id_columns(schema: EntitySchema, entity: str) -> tuple[str, ...]:
    if entity == schema.person_entity:
        return (
            schema.person_id_column,
            *(schema.membership_column(group) for group in schema.group_entities),
        )
    return (schema.entity_id_column(entity),)


def _validate_channels(channels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(channels, str):
        raise ValueError("support channels must be a sequence of names, not a string.")
    values = tuple(channels)
    if len(values) < 2:
        raise ValueError("PUF support expansion requires at least two channels.")
    bad = [value for value in values if not isinstance(value, str) or not value]
    if bad:
        raise ValueError("support channels must be non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"support channels must be unique, got {values!r}.")
    return values


def _reject_metadata_collisions(
    frame: Frame,
    channels: tuple[str, ...],
) -> None:
    expected = {
        column
        for entity in frame.entities
        for column in (
            support_source_id_column(entity),
            support_channel_column(entity),
            support_clone_index_column(entity),
        )
    }
    existing = {
        column for entity in frame.entities for column in frame.table(entity).columns
    }
    collisions = sorted(expected & existing)
    if collisions:
        raise ValueError(
            "PUF support expansion metadata column(s) already exist: "
            f"{collisions}. The stage should run exactly once."
        )
    if channels[0] != BASE_ASEC_SUPPORT_CHANNEL:
        raise ValueError(
            f"support channels must start with {BASE_ASEC_SUPPORT_CHANNEL!r} "
            "so the baseline ASEC channel keeps the original IDs."
        )
    if PUF_TAX_DETAIL_SUPPORT_CHANNEL not in channels:
        raise ValueError(
            f"support channels must include {PUF_TAX_DETAIL_SUPPORT_CHANNEL!r}."
        )


def _id_multiplier_for_frame(frame: Frame) -> int:
    values = [
        frame.table(entity)[frame.schema.entity_id_column(entity)]
        for entity in frame.entities
    ]
    return _id_multiplier_for_values(*values)


def _id_multiplier_for_values(*values: Sequence[Any]) -> int:
    if not values:
        raise ValueError("at least one ID value sequence is required.")
    max_id = 0
    for sequence in values:
        numeric = pd.to_numeric(pd.Series(sequence), errors="raise").astype("int64")
        if len(numeric):
            max_id = max(max_id, int(numeric.abs().max()))
    return 10 ** max(1, len(str(max_id)))


def _remap_ids(
    ids: Sequence[Any],
    *,
    clone_index: int,
    id_multiplier: int,
) -> np.ndarray:
    values = pd.to_numeric(pd.Series(ids), errors="raise").astype("int64").to_numpy()
    if clone_index == 0:
        return values.copy()
    return values + clone_index * id_multiplier


def _require_entity_name(entity: str) -> None:
    if not isinstance(entity, str) or not entity:
        raise ValueError("entity must be a non-empty string.")
