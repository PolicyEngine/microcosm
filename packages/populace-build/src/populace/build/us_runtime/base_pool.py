"""Optional ACS augmentation of the US ASEC-by-PUF base pool."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from populace.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from populace.build.us_runtime.puf_support import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.frame import CONSERVE_MASS, US_SCHEMA, Frame, MassChange, Weights

__all__ = [
    "ACS_2024_1YR_SPINE",
    "ASEC_PUF_SPINE",
    "spine_column",
    "with_optional_acs_spine",
]

ASEC_PUF_SPINE = "asec_puf"


def spine_column(entity: str) -> str:
    """Return the entity-prefixed base-spine metadata column."""

    if not isinstance(entity, str) or not entity:
        raise ValueError("entity must be a non-empty string.")
    return f"{entity}_spine"


def with_optional_acs_spine(
    base: Frame,
    acs: Frame | None = None,
    *,
    acs_share: float = 0.5,
) -> Frame:
    """Append an ACS frame while conserving the base's household mass.

    The no-ACS path is an identity operation: it returns ``base`` itself before
    validating, copying, tagging, or changing weights. When ACS is present, the
    two frames receive entity-prefixed spine tags, are aligned with missing
    values for source-specific columns, and are joined through
    :meth:`populace.frame.Frame.concat`.

    ``acs_share`` allocates that share of the incoming base household mass to
    ACS. The remaining share stays on the ASEC-by-PUF spine. Each allocation is
    recorded as a deliberate :class:`~populace.frame.MassChange`.
    """

    if acs is None:
        return base

    share = _validated_acs_share(acs_share)
    _require_us_base_frame(base, label="base")
    _require_us_base_frame(acs, label="ACS")

    base_support_metadata = _has_complete_support_metadata(
        _entity_table_refs(base),
        label="Base",
    )
    acs_support_metadata = False
    if base_support_metadata:
        acs_support_metadata = _has_complete_support_metadata(
            _entity_table_refs(acs),
            label="ACS",
        )
        if acs_support_metadata:
            _validate_acs_support_metadata(_entity_table_refs(acs))
    column_orders = _aligned_column_orders(
        base,
        acs,
        add_acs_support_metadata=base_support_metadata,
    )
    original_mass = base.weights_for("household").total
    base_target = original_mass * (1.0 - share)
    acs_target = original_mass - base_target

    allocated_base = _with_household_mass(
        base,
        target=base_target,
        reason="allocated ASEC-by-PUF mass for the optional ACS multispine pool",
    )
    base_tables = _prepared_tables(
        allocated_base,
        spine=ASEC_PUF_SPINE,
        column_orders=column_orders,
    )
    pooled_base = _rebuild(allocated_base, base_tables)
    del allocated_base, base_tables

    allocated_acs = _with_household_mass(
        acs,
        target=acs_target,
        reason="allocated ACS mass for the optional ACS multispine pool",
    )
    acs_tables = _prepared_tables(
        allocated_acs,
        spine=ACS_2024_1YR_SPINE,
        column_orders=column_orders,
        populate_support_metadata=(base_support_metadata and not acs_support_metadata),
    )
    pooled_acs = _rebuild(
        allocated_acs,
        acs_tables,
        strata=pd.Series(
            ACS_2024_1YR_SPINE,
            index=acs.table("person").index,
            dtype=object,
        ),
    )
    del allocated_acs, acs_tables
    return _with_conserved_household_mass(
        pooled_base.concat(pooled_acs),
        target=original_mass,
    )


def _validated_acs_share(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("acs_share must be a number strictly between 0 and 1.")
    try:
        share = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "acs_share must be a number strictly between 0 and 1."
        ) from exc
    if not np.isfinite(share) or not 0.0 < share < 1.0:
        raise ValueError("acs_share must be a finite number strictly between 0 and 1.")
    return share


def _require_us_base_frame(frame: Frame, *, label: str) -> None:
    if not isinstance(frame, Frame):
        raise TypeError(f"{label} must be a Frame, got {type(frame).__name__}.")
    if frame.schema != US_SCHEMA:
        raise ValueError(f"{label} must use the US entity schema.")
    if frame.weighted_entities != ("household",):
        raise ValueError(
            f"{label} must carry household weights only; got weighted entities "
            f"{list(frame.weighted_entities)}."
        )


def _entity_table_refs(frame: Frame) -> dict[str, pd.DataFrame]:
    return {entity: frame.table(entity) for entity in frame.entities}


def _prepared_tables(
    frame: Frame,
    *,
    spine: str,
    column_orders: dict[str, list[str]],
    populate_support_metadata: bool = False,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        source = frame.table(entity)
        column = spine_column(entity)
        if column in source and not (source[column] == spine).all():
            raise ValueError(
                f"{entity!r} already carries conflicting spine metadata in {column!r}."
            )
        table = source.reindex(columns=column_orders[entity])
        table[column] = spine
        if populate_support_metadata:
            source_id, channel, clone_index = _support_metadata_columns(entity)
            table[source_id] = table[US_SCHEMA.entity_id_column(entity)].to_numpy(
                copy=True
            )
            table[channel] = ACS_2024_1YR_SPINE
            table[clone_index] = 0
        tables[entity] = table
    return tables


def _aligned_column_orders(
    base: Frame,
    acs: Frame,
    *,
    add_acs_support_metadata: bool,
) -> dict[str, list[str]]:
    orders: dict[str, list[str]] = {}
    for entity in US_SCHEMA.entities:
        base_columns = list(base.table(entity).columns)
        acs_columns = list(acs.table(entity).columns)
        tag = spine_column(entity)
        if tag not in base_columns:
            base_columns.append(tag)
        if tag not in acs_columns:
            acs_columns.append(tag)
        if add_acs_support_metadata:
            for column in _support_metadata_columns(entity):
                if column not in acs_columns:
                    acs_columns.append(column)
        orders[entity] = [
            *base_columns,
            *(column for column in acs_columns if column not in base_columns),
        ]
    return orders


def _has_complete_support_metadata(
    tables: dict[str, pd.DataFrame],
    *,
    label: str,
) -> bool:
    presence: list[bool] = []
    incomplete: list[str] = []
    for entity, table in tables.items():
        expected = _support_metadata_columns(entity)
        present = [column for column in expected if column in table]
        presence.append(bool(present))
        if present and len(present) != len(expected):
            incomplete.append(entity)
    if incomplete or (any(presence) and not all(presence)):
        affected = incomplete or [
            entity
            for entity, present in zip(tables, presence, strict=True)
            if not present
        ]
        raise ValueError(
            f"{label} support metadata must be complete on every entity; "
            f"incomplete entity table(s): {affected}."
        )
    return all(presence)


def _support_metadata_columns(entity: str) -> tuple[str, str, str]:
    return (
        support_source_id_column(entity),
        support_channel_column(entity),
        support_clone_index_column(entity),
    )


def _validate_acs_support_metadata(tables: dict[str, pd.DataFrame]) -> None:
    conflicts: list[str] = []
    for entity, table in tables.items():
        source_id, channel, clone_index = _support_metadata_columns(entity)
        own_ids = table[US_SCHEMA.entity_id_column(entity)].reset_index(drop=True)
        source_ids = table[source_id].reset_index(drop=True)
        source_ids_match = np.array_equal(source_ids.to_numpy(), own_ids.to_numpy())
        channels_match = table[channel].eq(ACS_2024_1YR_SPINE).all()
        clone_indexes_match = table[clone_index].eq(0).all()
        if not (source_ids_match and channels_match and clone_indexes_match):
            conflicts.append(entity)
    if conflicts:
        raise ValueError(
            "ACS support metadata conflicts with native ACS IDs/channel; "
            f"entity table(s): {conflicts}."
        )


def _rebuild(
    frame: Frame,
    tables: dict[str, pd.DataFrame],
    *,
    strata: pd.Series | None = None,
) -> Frame:
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata if strata is None else strata,
        mass_log=frame.mass_log,
    )


def _with_household_mass(frame: Frame, *, target: float, reason: str) -> Frame:
    existing = frame.weights_for("household")
    factor = target / existing.total
    return frame.with_weights(
        "household",
        Weights(existing.values * factor, existing.kind),
        mass=MassChange(factor=factor, reason=reason),
    )


def _with_conserved_household_mass(frame: Frame, *, target: float) -> Frame:
    existing = frame.weights_for("household")
    if existing.total == target:
        return frame
    result = frame.with_weights(
        "household",
        Weights(_values_nearest_total(existing.values, target), existing.kind),
        mass=CONSERVE_MASS,
    )
    if not np.isclose(
        result.weights_for("household").total,
        target,
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError("Optional ACS pool assembly failed to conserve mass.")
    return result


def _values_nearest_total(values: np.ndarray, target: float) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    correction_index = int(np.argmax(result))
    result[correction_index] += target - float(result.sum())
    return result
