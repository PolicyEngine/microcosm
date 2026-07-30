"""Pre-operator assembly of peer US household-support spines.

The assembly stage combines source frames and records their source channel
before any clone, imputation, derivation, take-up, simulation, or calibration
operator runs.  PUF tax detail is deliberately excluded: it is a clone
operator applied after this seam, not a peer household spine.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from populace.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.frame import (
    US_SCHEMA,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)

__all__ = ["assemble_spines"]

_CHANNEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SUPPORT_CLONE_INDEX = 0
_SHARE_RTOL = 1e-12


def assemble_spines(
    spines: Mapping[str, Frame],
    *,
    household_mass_shares: Mapping[str, float],
    mass_anchor_channel: str = BASE_ASEC_SUPPORT_CHANNEL,
) -> Frame:
    """Combine peer household spines into one pre-operator frame.

    ``household_mass_shares`` allocates the anchor frame's incoming household
    mass across the peer sources.  The anchor is emitted first and remaining
    channels are emitted in lexical order, so mapping insertion order cannot
    change record order.  Each entity receives four provenance fields:
    ``*_support_channel`` (the immutable source-spine channel),
    ``*_spine_source_id`` (the raw ID before collision remapping),
    ``*_source_id`` (the assembly-unique ID before cloning), and
    ``*_support_clone_index`` (zero before clone operators).

    Source frames must already use the same US schema and column dtypes for
    every column they share.  Source-specific columns are carried with missing
    values on other sources.  Measured values in the inputs are copied without
    modification; only structural IDs can be remapped to avoid collisions.

    Args:
        spines: Two or more peer source frames keyed by stable source channel.
        household_mass_shares: Positive shares, one per source, summing to one.
        mass_anchor_channel: Source whose incoming household mass is conserved.

    Returns:
        A combined, household-weighted frame with importance weights and
        source-spine provenance on every entity.

    Raises:
        TypeError: If the mappings or their frames have the wrong types.
        ValueError: If source channels, shares, schemas, weights, links,
            provenance, columns, or ID spaces violate the assembly contract.
    """

    ordered_channels = _validated_channels(spines, mass_anchor_channel)
    shares = _validated_shares(household_mass_shares, ordered_channels)
    frames = {channel: spines[channel] for channel in ordered_channels}
    for channel, frame in frames.items():
        _validate_source_frame(frame, channel=channel)
    _validate_shared_column_dtypes(frames)

    offsets = _id_offsets(frames, ordered_channels)
    column_orders = _column_orders(frames, ordered_channels)
    column_dtypes = _column_dtypes(frames, ordered_channels)
    prepared = {
        channel: _prepared_tables(
            frames[channel],
            channel=channel,
            offsets=offsets[channel],
            column_orders=column_orders,
            column_dtypes=column_dtypes,
        )
        for channel in ordered_channels
    }
    tables, group_orders = _combined_tables(
        prepared,
        ordered_channels=ordered_channels,
        column_orders=column_orders,
    )

    anchor_mass = frames[mass_anchor_channel].weights_for("household").total
    weights, mass_log = _assembled_household_weights(
        frames,
        ordered_channels=ordered_channels,
        shares=shares,
        anchor_mass=anchor_mass,
    )
    household_order = group_orders.get("household")
    if household_order is not None:
        weights = Weights(weights.values[household_order], weights.kind)
    weights = _with_exact_total(weights, anchor_mass)

    strata = pd.concat(
        [frames[channel].strata for channel in ordered_channels],
        ignore_index=True,
    )
    result = Frame(
        tables,
        US_SCHEMA,
        {"household": weights},
        strata,
        mass_log=mass_log,
    )
    if not np.isclose(
        result.weights_for("household").total,
        anchor_mass,
        rtol=_SHARE_RTOL,
        atol=0.0,
    ):
        raise RuntimeError("Spine assembly failed to conserve anchor household mass.")
    return result


def _validated_channels(
    spines: Mapping[str, Frame],
    mass_anchor_channel: str,
) -> tuple[str, ...]:
    if not isinstance(spines, Mapping):
        raise TypeError(f"spines must be a mapping, got {type(spines).__name__}.")
    if len(spines) < 2:
        raise ValueError("assemble_spines requires at least two peer source frames.")
    channels = tuple(spines)
    invalid = [
        channel
        for channel in channels
        if not isinstance(channel, str) or _CHANNEL_PATTERN.fullmatch(channel) is None
    ]
    if invalid:
        raise ValueError(
            "Spine channels must be stable lower-snake-case identifiers; "
            f"invalid channel(s): {invalid}."
        )
    if PUF_TAX_DETAIL_SUPPORT_CHANNEL in channels:
        raise ValueError(
            f"{PUF_TAX_DETAIL_SUPPORT_CHANNEL!r} is a clone operator channel, "
            "not a peer household spine."
        )
    if (
        not isinstance(mass_anchor_channel, str)
        or _CHANNEL_PATTERN.fullmatch(mass_anchor_channel) is None
    ):
        raise ValueError(
            "mass_anchor_channel must be a stable lower-snake-case identifier."
        )
    if mass_anchor_channel not in spines:
        raise ValueError(
            f"mass_anchor_channel {mass_anchor_channel!r} is absent from spines."
        )
    return (
        mass_anchor_channel,
        *sorted(channel for channel in channels if channel != mass_anchor_channel),
    )


def _validated_shares(
    household_mass_shares: Mapping[str, float],
    ordered_channels: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(household_mass_shares, Mapping):
        raise TypeError(
            "household_mass_shares must be a mapping, got "
            f"{type(household_mass_shares).__name__}."
        )
    expected = set(ordered_channels)
    actual = set(household_mass_shares)
    if actual != expected:
        raise ValueError(
            "household_mass_shares keys must exactly match spine channels "
            f"(missing: {sorted(expected - actual, key=repr)}; extra: "
            f"{sorted(actual - expected, key=repr)})."
        )
    shares: dict[str, float] = {}
    for channel in ordered_channels:
        raw_share: Any = household_mass_shares[channel]
        if isinstance(raw_share, bool):
            raise ValueError(f"Share for {channel!r} must be positive and finite.")
        try:
            share = float(raw_share)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Share for {channel!r} must be positive and finite."
            ) from exc
        if not np.isfinite(share) or share <= 0.0:
            raise ValueError(f"Share for {channel!r} must be positive and finite.")
        shares[channel] = share
    total = float(sum(shares.values()))
    if not np.isclose(total, 1.0, rtol=_SHARE_RTOL, atol=_SHARE_RTOL):
        raise ValueError(
            f"household_mass_shares must sum to one; received total {total!r}."
        )
    return shares


def _validate_source_frame(frame: Frame, *, channel: str) -> None:
    if not isinstance(frame, Frame):
        raise TypeError(
            f"Spine {channel!r} must be a Frame, got {type(frame).__name__}."
        )
    if frame.schema != US_SCHEMA:
        raise ValueError(f"Spine {channel!r} must use the US entity schema.")
    if frame.weighted_entities != ("household",):
        raise ValueError(
            f"Spine {channel!r} must carry household weights only; got "
            f"{list(frame.weighted_entities)}."
        )
    if frame.links:
        raise ValueError(
            f"Spine {channel!r} carries link tables {list(frame.links)}; "
            "pre-operator assembly does not yet carry links."
        )
    if frame.weights_for("household").total <= 0.0:
        raise ValueError(f"Spine {channel!r} household weight mass must be positive.")
    conflicts: list[str] = []
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        metadata = _support_metadata_columns(entity)
        present = [column for column in metadata if column in table]
        if present:
            conflicts.append(f"{entity}: {present}")
    if conflicts:
        raise ValueError(
            f"Spine {channel!r} already carries support provenance; assembly "
            f"must be the provenance owner ({'; '.join(conflicts)})."
        )


def _validate_shared_column_dtypes(frames: Mapping[str, Frame]) -> None:
    for entity in US_SCHEMA.entities:
        owners: dict[str, list[tuple[str, Any]]] = {}
        for channel, frame in frames.items():
            for column, dtype in frame.table(entity).dtypes.items():
                owners.setdefault(column, []).append((channel, dtype))
        mismatches = {
            column: values
            for column, values in owners.items()
            if not all(dtype == values[0][1] for _, dtype in values[1:])
        }
        if mismatches:
            details = ", ".join(
                f"{column}={[(channel, str(dtype)) for channel, dtype in values]}"
                for column, values in sorted(mismatches.items())
            )
            raise ValueError(
                f"Shared {entity!r} columns must have identical dtypes before "
                f"spine assembly; {details}."
            )


def _support_metadata_columns(entity: str) -> tuple[str, str, str, str]:
    return (
        spine_source_id_column(entity),
        support_source_id_column(entity),
        support_channel_column(entity),
        support_clone_index_column(entity),
    )


def _column_orders(
    frames: Mapping[str, Frame],
    ordered_channels: tuple[str, ...],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for entity in US_SCHEMA.entities:
        columns: list[str] = []
        for channel in ordered_channels:
            for column in frames[channel].table(entity).columns:
                if column not in columns:
                    columns.append(column)
        columns.extend(_support_metadata_columns(entity))
        result[entity] = columns
    return result


def _column_dtypes(
    frames: Mapping[str, Frame],
    ordered_channels: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entity in US_SCHEMA.entities:
        dtypes: dict[str, Any] = {}
        for channel in ordered_channels:
            table = frames[channel].table(entity)
            for column in table:
                dtypes.setdefault(column, table[column].dtype)
        spine_source_id, source_id, support_channel, clone_index = (
            _support_metadata_columns(entity)
        )
        dtypes[spine_source_id] = (
            frames[ordered_channels[0]]
            .table(entity)[US_SCHEMA.entity_id_column(entity)]
            .dtype
        )
        dtypes[source_id] = (
            frames[ordered_channels[0]]
            .table(entity)[US_SCHEMA.entity_id_column(entity)]
            .dtype
        )
        dtypes[support_channel] = np.dtype(object)
        dtypes[clone_index] = np.dtype(np.int64)
        result[entity] = dtypes
    return result


def _id_offsets(
    frames: Mapping[str, Frame],
    ordered_channels: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    offsets = {channel: {} for channel in ordered_channels}
    accumulated: dict[str, np.ndarray] = {}
    for position, channel in enumerate(ordered_channels):
        frame = frames[channel]
        for entity in US_SCHEMA.entities:
            id_column = US_SCHEMA.entity_id_column(entity)
            ids = frame.table(entity)[id_column].to_numpy()
            if not np.issubdtype(ids.dtype, np.integer):
                raise ValueError(
                    f"Spine {channel!r} {id_column!r} must be integer-typed "
                    "for collision-safe assembly."
                )
            if position == 0:
                accumulated[entity] = np.array(ids, copy=True)
                continue
            used = accumulated[entity]
            offset = 0
            if np.intersect1d(used, ids, assume_unique=True).size:
                offset = int(used.max()) + 1 - int(ids.min())
                info = np.iinfo(ids.dtype)
                if int(ids.max()) + offset > int(info.max):
                    raise ValueError(
                        f"Spine {channel!r} {entity!r} IDs cannot be remapped "
                        f"without overflowing {ids.dtype}."
                    )
                offsets[channel][entity] = offset
            remapped = ids if offset == 0 else ids + offset
            accumulated[entity] = np.concatenate([used, remapped])
    return offsets


def _prepared_tables(
    frame: Frame,
    *,
    channel: str,
    offsets: Mapping[str, int],
    column_orders: Mapping[str, list[str]],
    column_dtypes: Mapping[str, Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        source = frame.table(entity)
        table = source.copy()
        id_column = US_SCHEMA.entity_id_column(entity)
        table[spine_source_id_column(entity)] = source[id_column].to_numpy()
        table[support_channel_column(entity)] = channel
        table[support_clone_index_column(entity)] = _SUPPORT_CLONE_INDEX
        if entity == US_SCHEMA.person_entity:
            person_offset = offsets.get(entity)
            if person_offset is not None:
                table[id_column] = source[id_column].to_numpy() + person_offset
            for group in US_SCHEMA.group_entities:
                group_offset = offsets.get(group)
                if group_offset is not None:
                    membership = US_SCHEMA.membership_column(group)
                    table[membership] = source[membership].to_numpy() + group_offset
        else:
            offset = offsets.get(entity)
            if offset is not None:
                table[id_column] = source[id_column].to_numpy() + offset
        table[support_source_id_column(entity)] = table[id_column].to_numpy()
        result[entity] = _align_table(
            table,
            column_orders[entity],
            column_dtypes[entity],
        )
    return result


def _align_table(
    table: pd.DataFrame,
    columns: list[str],
    dtypes: Mapping[str, Any],
) -> pd.DataFrame:
    aligned = table.copy()
    for column in columns:
        if column not in aligned:
            aligned[column] = _missing_series(
                len(aligned),
                dtype=dtypes[column],
                index=aligned.index,
            )
    return aligned.loc[:, columns]


def _missing_series(
    length: int,
    *,
    dtype: Any,
    index: pd.Index,
) -> pd.Series:
    if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        return pd.Series(np.full(length, np.nan, dtype=dtype), index=index)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.Series(pd.NaT, index=index, dtype=dtype)
    if pd.api.types.is_timedelta64_dtype(dtype):
        return pd.Series(pd.NaT, index=index, dtype=dtype)
    values = np.empty(length, dtype=object)
    values[:] = None
    return pd.Series(values, index=index, dtype=object)


def _combined_tables(
    prepared: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    ordered_channels: tuple[str, ...],
    column_orders: Mapping[str, list[str]],
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    tables: dict[str, pd.DataFrame] = {}
    group_orders: dict[str, np.ndarray] = {}
    for entity in US_SCHEMA.entities:
        combined = pd.concat(
            [prepared[channel][entity] for channel in ordered_channels],
            ignore_index=True,
            sort=False,
        ).loc[:, column_orders[entity]]
        if entity in US_SCHEMA.group_entities:
            id_column = US_SCHEMA.id_column(entity)
            order = np.argsort(combined[id_column].to_numpy(), kind="stable")
            if not np.array_equal(order, np.arange(len(combined))):
                combined = combined.iloc[order].reset_index(drop=True)
                group_orders[entity] = order
        tables[entity] = combined
    return tables, group_orders


def _assembled_household_weights(
    frames: Mapping[str, Frame],
    *,
    ordered_channels: tuple[str, ...],
    shares: Mapping[str, float],
    anchor_mass: float,
) -> tuple[Weights, tuple[MassChangeRecord, ...]]:
    values: list[np.ndarray] = []
    mass_log: list[MassChangeRecord] = []
    allocated = 0.0
    for index, channel in enumerate(ordered_channels):
        existing = frames[channel].weights_for("household")
        target = (
            anchor_mass - allocated
            if index == len(ordered_channels) - 1
            else anchor_mass * shares[channel]
        )
        scaled = _values_to_total(existing.values, target)
        values.append(scaled)
        allocated += float(scaled.sum())
        factor = target / existing.total
        mass_log.extend(frames[channel].mass_log)
        mass_log.append(
            MassChangeRecord(
                entity="household",
                old_total=existing.total,
                new_total=float(scaled.sum()),
                declared_factor=factor,
                reason=(
                    f"allocated {channel!r} source mass in pre-operator spine assembly"
                ),
            )
        )
    return (
        Weights(np.concatenate(values), WeightKind.IMPORTANCE),
        tuple(mass_log),
    )


def _values_to_total(values: np.ndarray, target: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64) * (
        target / float(np.asarray(values, dtype=np.float64).sum())
    )
    correction_index = int(np.argmax(result))
    result[correction_index] += target - float(result.sum())
    return result


def _with_exact_total(weights: Weights, target: float) -> Weights:
    if weights.total == target:
        return weights
    values = np.array(weights.values, copy=True)
    correction_index = int(np.argmax(values))
    values[correction_index] += target - float(values.sum())
    return Weights(values, weights.kind)
