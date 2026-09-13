"""Pre-operator assembly of peer US household-support spines.

The assembly stage combines source frames and records their source channel
before any clone, imputation, derivation, take-up, simulation, or calibration
operator runs.  PUF tax detail is deliberately excluded: it is a clone
operator applied after this seam, not a peer household spine.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.operator_column_contracts import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_assembly_manifest,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.frame import (
    US_SCHEMA,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)

__all__ = [
    "SpinePreparation",
    "SpineHarmonization",
    "assemble_spines",
    "prepare_spines",
    "stack_survey_spines",
    "harmonize_spine_weights",
]

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
    ``*_support_channel`` (the receipt-validated source-spine channel),
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

    prepared = _stack_spine_tables(
        spines,
        household_mass_shares=household_mass_shares,
        mass_anchor_channel=mass_anchor_channel,
    )
    harmonized = _harmonize_spine_values(
        prepared.tables["household"], prepared.values, prepared.context
    )
    # These are newly composed per-source records, not a carried Frame log.
    # Table provenance remains owned by the independent stacking result.
    assembled_mass_log = harmonized.mass_log
    result = Frame(
        prepared.tables,
        US_SCHEMA,
        {"household": harmonized.weights},
        prepared.strata,
        mass_log=assembled_mass_log,
        metadata=prepared.metadata,
    )
    validate_assembly_provenance(result, boundary="spine assembly output")
    anchor_mass = prepared.context["incoming_masses"][mass_anchor_channel]
    if not np.isclose(
        result.weights_for("household").total, anchor_mass, rtol=_SHARE_RTOL, atol=0.0
    ):
        raise RuntimeError("Spine assembly failed to conserve anchor household mass.")
    return result


@dataclass(frozen=True)
class SpinePreparation:
    """Actual DESIGN-weight tables plus an immutable assembly context.

    The context schema determines which subsequent weight operation can use it.
    This numerical container does not authenticate source issuance or membership.
    """

    frame: Frame
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError("SpinePreparation.frame must be a Frame.")
        if self.frame.weights_for("household").kind is not WeightKind.DESIGN:
            raise ValueError("Spine preparation requires actual DESIGN weights.")
        if not isinstance(self.context, Mapping):
            raise ValueError("Spine preparation context must be a mapping.")
        object.__setattr__(self, "context", _freeze_context(self.context))


@dataclass(frozen=True)
class SpineHarmonization:
    """Computed importance weights and exact ordered legacy operator history."""

    weights: Weights
    mass_log: tuple[MassChangeRecord, ...]


@dataclass(frozen=True)
class _StackedTables:
    tables: Mapping[str, pd.DataFrame]
    strata: pd.Series
    values: np.ndarray
    context: Mapping[str, object]
    metadata: Mapping[str, object]
    mass_log: tuple[MassChangeRecord, ...]


def _freeze_context(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_context(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_context(item) for item in value)
    return value


def _numeric_digest(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _channel_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("Prepared household channels must be strings.")
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _stack_spine_tables(
    spines: Mapping[str, Frame],
    *,
    household_mass_shares: Mapping[str, float],
    mass_anchor_channel: str,
    require_design: bool = False,
) -> _StackedTables:
    ordered_channels = _validated_channels(spines, mass_anchor_channel)
    shares = _validated_shares(household_mass_shares, ordered_channels)
    frames = {channel: spines[channel] for channel in ordered_channels}
    prepared = _stack_source_tables(frames, ordered_channels, require_design)
    context = dict(prepared.context)
    context.update(
        schema="microcosm.us.spine-preparation.v1",
        household_mass_shares=shares,
        mass_anchor_channel=mass_anchor_channel,
        incoming_masses={
            channel: float(frames[channel].weights_for("household").total)
            for channel in ordered_channels
        },
    )
    return _StackedTables(
        prepared.tables,
        prepared.strata,
        prepared.values,
        _freeze_context(context),
        prepared.metadata,
        prepared.mass_log,
    )


def _stack_source_tables(
    spines: Mapping[str, Frame],
    ordered_channels: tuple[str, ...],
    require_design: bool,
) -> _StackedTables:
    """Preserve source values while remapping only structural collisions."""
    frames = {channel: spines[channel] for channel in ordered_channels}
    for channel, frame in frames.items():
        _validate_source_frame(frame, channel=channel)
        if (
            require_design
            and frame.weights_for("household").kind is not WeightKind.DESIGN
        ):
            raise ValueError(
                f"Spine {channel!r} preparation requires actual DESIGN weights."
            )
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

    values = np.concatenate(
        [
            frames[channel].weights_for("household").values
            for channel in ordered_channels
        ]
    )
    order = group_orders.get("household")
    if order is not None:
        values = values[order]
    household = tables["household"]
    source_ids_digest = hashlib.sha256()
    for channel in ordered_channels:
        source_ids = np.ascontiguousarray(
            prepared[channel]["household"]["household_id"].to_numpy(), dtype="<i8"
        )
        source_ids_digest.update(memoryview(source_ids).cast("B"))
    context = {
        "schema": "microcosm.us.survey-stack.v1",
        "ordered_channels": ordered_channels,
        "household_counts": {
            channel: frames[channel].n("household") for channel in ordered_channels
        },
        "source_weight_kinds": {
            channel: frames[channel].weights_for("household").kind.value
            for channel in ordered_channels
        },
        "household_order": None if order is None else order.tolist(),
        "source_household_ids_sha256": source_ids_digest.hexdigest(),
        "household_ids_sha256": _numeric_digest(
            household["household_id"].to_numpy(), "<i8"
        ),
        "household_channels_sha256": _channel_digest(
            household[support_channel_column("household")].to_numpy()
        ),
        "household_weights_sha256": _numeric_digest(values, "<f8"),
        "source_mass_logs": {
            channel: [asdict(record) for record in frames[channel].mass_log]
            for channel in ordered_channels
        },
    }
    strata = pd.concat(
        [frames[channel].strata for channel in ordered_channels], ignore_index=True
    )
    return _StackedTables(
        tables,
        strata,
        values,
        _freeze_context(context),
        spine_assembly_manifest(tables, channels=ordered_channels),
        tuple(
            record
            for channel in ordered_channels
            for record in frames[channel].mass_log
        ),
    )


def prepare_spines(
    spines: Mapping[str, Frame],
    *,
    household_mass_shares: Mapping[str, float],
    mass_anchor_channel: str = BASE_ASEC_SUPPORT_CHANNEL,
) -> SpinePreparation:
    """Stack actual DESIGN sources without performing importance allocation.

    Legacy ``assemble_spines`` continues to accept its existing mixed-kind
    inputs. This new graph-facing seam never relabels such inputs as DESIGN.
    """
    prepared = _stack_spine_tables(
        spines,
        household_mass_shares=household_mass_shares,
        mass_anchor_channel=mass_anchor_channel,
        require_design=True,
    )
    return _design_preparation(prepared)


def stack_survey_spines(spines: Mapping[str, Frame]) -> SpinePreparation:
    """Stack original DESIGN survey weights before domain allocation.

    Channels are ordered lexically, independent of mapping insertion order.
    Every household weight, source age and measured value is carried unchanged;
    only structural IDs are remapped and source provenance is added. No survey
    total is an anchor, no shares are assigned, and no mass is normalized.

    This is the numerical stacking seam. A graph source owner must separately
    authenticate the original issuances, complete membership and publisher
    anchors. Sampling and population-domain allocation are subsequent declared
    operations. The returned v1 survey-stack context is deliberately refused
    by the legacy anchor-total harmonizer.
    """
    channels = _validated_source_channels(spines)
    prepared = _stack_source_tables(spines, channels, require_design=True)
    return _design_preparation(prepared)


def _design_preparation(prepared: _StackedTables) -> SpinePreparation:
    frame = Frame(
        prepared.tables,
        US_SCHEMA,
        {"household": Weights(prepared.values, WeightKind.DESIGN)},
        prepared.strata,
        mass_log=prepared.mass_log,
        metadata=prepared.metadata,
    )
    validate_assembly_provenance(frame, boundary="spine preparation output")
    return SpinePreparation(frame, prepared.context)


def harmonize_spine_weights(
    *,
    household: pd.DataFrame,
    weights: Weights,
    context: Mapping[str, object],
) -> SpineHarmonization:
    """Compute importance allocation from declared DESIGN views and context.

    Input digests bind exact normalized values and ordered IDs/channels.
    The explicit permutation restores source-order sums and correction rows.
    The result includes legacy per-source logs, not an executor graph ledger.
    """
    if not isinstance(weights, Weights) or weights.kind is not WeightKind.DESIGN:
        raise ValueError("Spine harmonization requires actual DESIGN weights.")
    if (
        not isinstance(context, Mapping)
        or context.get("schema") != "microcosm.us.spine-preparation.v1"
    ):
        raise ValueError("Unsupported spine preparation context.")
    kinds = context.get("source_weight_kinds")
    ordered = context.get("ordered_channels")
    if (
        not isinstance(kinds, Mapping)
        or not isinstance(ordered, (tuple, list))
        or set(kinds) != set(ordered)
        or any(kind != WeightKind.DESIGN.value for kind in kinds.values())
    ):
        raise ValueError("Spine preparation context does not declare DESIGN sources.")
    return _harmonize_spine_values(household, weights.values, context)


def _validated_spine_inputs(
    household: pd.DataFrame, values: np.ndarray, context: Mapping[str, object]
) -> tuple:
    if (
        not isinstance(context, Mapping)
        or context.get("schema") != "microcosm.us.spine-preparation.v1"
    ):
        raise ValueError("Unsupported spine preparation context.")
    if not isinstance(household, pd.DataFrame) or not {
        "household_id",
        support_channel_column("household"),
    }.issubset(household):
        raise ValueError("Harmonization requires household IDs and source channels.")
    ids = household["household_id"].to_numpy()
    channels = household[support_channel_column("household")].to_numpy()
    if not np.issubdtype(ids.dtype, np.integer) or len(values) != len(ids):
        raise ValueError("Prepared household IDs/weights are malformed.")
    bindings = {
        "household_ids_sha256": _numeric_digest(ids, "<i8"),
        "household_channels_sha256": _channel_digest(channels),
        "household_weights_sha256": _numeric_digest(values, "<f8"),
    }
    if any(context.get(key) != value for key, value in bindings.items()):
        raise ValueError("Prepared household input binding changed.")
    ordered = tuple(context["ordered_channels"])
    if (
        len(ordered) < 2
        or len(set(ordered)) != len(ordered)
        or context["mass_anchor_channel"] != ordered[0]
    ):
        raise ValueError("Prepared source channel order/anchor is invalid.")
    shares = _validated_shares(context["household_mass_shares"], ordered)
    counts = context["household_counts"]
    if (
        set(counts) != set(ordered)
        or any(
            isinstance(counts[ch], bool)
            or not isinstance(counts[ch], int)
            or counts[ch] <= 0
            for ch in ordered
        )
        or sum(counts.values()) != len(values)
    ):
        raise ValueError("Prepared source household counts are invalid.")
    order = context["household_order"]
    if order is not None:
        if (
            not isinstance(order, (list, tuple))
            or len(order) != len(values)
            or any(isinstance(i, bool) or not isinstance(i, int) for i in order)
        ):
            raise ValueError("Prepared household permutation is invalid.")
        order = np.asarray(order, dtype=np.int64)
        if not np.array_equal(np.sort(order), np.arange(len(values))):
            raise ValueError("Prepared household permutation is invalid.")
        inverse = np.argsort(order, kind="stable")
        source_values, source_channels, source_ids = (
            values[inverse],
            channels[inverse],
            ids[inverse],
        )
    else:
        source_values, source_channels, source_ids = values, channels, ids
    if _numeric_digest(source_ids, "<i8") != context.get("source_household_ids_sha256"):
        raise ValueError("Prepared permutation changed the original source ID order.")
    anchor_mass = float(context["incoming_masses"][ordered[0]])
    return ordered, shares, counts, source_values, source_channels, order, anchor_mass


def _harmonize_spine_values(
    household: pd.DataFrame, values: np.ndarray, context: Mapping[str, object]
) -> SpineHarmonization:
    ordered, shares, counts, source_values, source_channels, order, anchor_mass = (
        _validated_spine_inputs(household, values, context)
    )
    result, logs, allocated, start = [], [], 0.0, 0
    for index, channel in enumerate(ordered):
        count = counts[channel]
        existing_values = source_values[start : start + count]
        if not np.all(source_channels[start : start + count] == channel):
            raise ValueError(
                "Prepared permutation does not restore source channel order."
            )
        start += count
        existing_mass = float(existing_values.sum())
        if existing_mass != float(context["incoming_masses"][channel]):
            raise ValueError("Prepared source-order mass changed.")
        target = (
            anchor_mass - allocated
            if index == len(ordered) - 1
            else anchor_mass * shares[channel]
        )
        scaled = _values_to_total(existing_values, target)
        result.append(scaled)
        allocated += float(scaled.sum())
        logs.extend(
            MassChangeRecord(**dict(record))
            for record in context["source_mass_logs"][channel]
        )
        logs.append(
            MassChangeRecord(
                entity="household",
                old_total=existing_mass,
                new_total=float(scaled.sum()),
                declared_factor=target / existing_mass,
                reason=f"allocated {channel!r} source mass in pre-operator spine assembly",
            )
        )
    combined = np.concatenate(result)
    if order is not None:
        combined = combined[order]
    return SpineHarmonization(
        _with_exact_total(Weights(combined, WeightKind.IMPORTANCE), anchor_mass),
        tuple(logs),
    )


def _validated_channels(
    spines: Mapping[str, Frame],
    mass_anchor_channel: str,
) -> tuple[str, ...]:
    channels = _validated_source_channels(spines)
    if (
        not isinstance(mass_anchor_channel, str)
        or _CHANNEL_PATTERN.fullmatch(mass_anchor_channel) is None
    ):
        raise ValueError(
            "mass_anchor_channel must be a stable lower-snake-case identifier."
        )
    if mass_anchor_channel not in channels:
        raise ValueError(
            f"mass_anchor_channel {mass_anchor_channel!r} is absent from spines."
        )
    return (
        mass_anchor_channel,
        *(channel for channel in channels if channel != mass_anchor_channel),
    )


def _validated_source_channels(spines: Mapping[str, Frame]) -> tuple[str, ...]:
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
    return tuple(sorted(channels))


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
        id_column = US_SCHEMA.entity_id_column(entity)
        ids = table[id_column].to_numpy()
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                f"Spine {channel!r} {id_column!r} must contain integral source IDs."
            )
        negative_ids = np.unique(ids[ids < 0])
        if negative_ids.size:
            raise ValueError(
                f"Spine {channel!r} {id_column!r} contains negative source IDs "
                f"{negative_ids[:5].tolist()}; source IDs must be nonnegative."
            )
        oversized_ids = np.unique(ids[ids > PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID])
        if oversized_ids.size:
            # The clone stage's decimal remap (id + clone_index *
            # 10**digits(max_id)) must stay inside int64 for every clone
            # index; IDs above the shared bound compose into an overflow
            # there, so assembly rejects them at the door.
            raise ValueError(
                f"Spine {channel!r} {id_column!r} contains source IDs above "
                f"the clone-safe bound {PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID} "
                f"({oversized_ids[:5].tolist()}); larger IDs overflow the "
                "clone stage's int64 decimal remap."
            )
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
            # repr, not str: pandas string dtypes with different storages all
            # str() as 'str', which renders the mismatch invisible.
            details = ", ".join(
                f"{column}={[(channel, repr(dtype)) for channel, dtype in values]}"
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
            if int(remapped.max()) > PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID:
                raise ValueError(
                    f"Spine {channel!r} {entity!r} collision remapping exceeds "
                    f"the clone-safe bound {PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID}."
                )
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
