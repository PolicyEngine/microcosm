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
from populace.build.us_runtime.puma_ladder import (
    UsPumaLadder,
    assign_us_puma_ladder,
)
from populace.frame import (
    US_SCHEMA,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)

__all__ = [
    "ACS_2024_1YR_SPINE",
    "ASEC_PUF_SPINE",
    "DEFAULT_ACS_POOL_PEAK_LIMIT_BYTES",
    "estimate_optional_acs_pool_peak_bytes",
    "spine_column",
    "with_optional_acs_spine",
]

ASEC_PUF_SPINE = "asec_puf"

# The launch worker has a 30 GB RSS budget. The estimate includes the two
# incoming frames, the assembled tables, the Frame constructor's defensive
# copies, and a fixed allowance for pandas/Python scratch allocations.
DEFAULT_ACS_POOL_PEAK_LIMIT_BYTES = 30_000_000_000
_PEAK_ESTIMATE_FIXED_OVERHEAD_BYTES = 256 * 1024**2
_ID_OVERLAP_CHUNK_ROWS = 65_536
_PUMA_LADDER_ANCHOR_COLUMN = "__populace_puma_ladder_anchor"


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
    max_peak_bytes: int | None = DEFAULT_ACS_POOL_PEAK_LIMIT_BYTES,
    puma_ladder: UsPumaLadder | None = None,
    geography_seed: int = 0,
    expected_congressional_district_vintage: str | None = None,
) -> Frame:
    """Append an ACS frame while conserving the base's household mass.

    The no-ACS path is an identity operation: it returns ``base`` itself before
    validating, copying, tagging, estimating memory, or changing weights. When
    ACS is present, the two frames receive entity-prefixed spine tags, are
    aligned with missing values for source-specific columns, and are assembled
    with the same ID-remapping semantics as :meth:`populace.frame.Frame.concat`.

    ``acs_share`` allocates that share of the incoming base household mass to
    ACS. The remaining share stays on the ASEC-by-PUF spine. Each allocation is
    recorded as a deliberate :class:`~populace.frame.MassChangeRecord`.

    When ``puma_ladder`` is present, ACS records retain their known 2020 PUMA
    while ASEC-by-PUF records draw a PUMA within their state. Congressional
    district and county are then assigned from the ladder for both spines.
    Assignment happens on the newly assembled household table before the one
    final :class:`~populace.frame.Frame` construction, avoiding a second copy
    of every entity table.

    ``max_peak_bytes`` is a fail-fast bound on estimated frame-resident peak
    memory. Pass ``None`` to disable the preflight. The estimate includes both
    inputs and the defensive copies made by :class:`~populace.frame.Frame`; it
    intentionally does not claim to account for unrelated objects already in
    the process.
    """

    if acs is None:
        return base

    share = _validated_acs_share(acs_share)
    limit = _validated_peak_limit(max_peak_bytes)
    _require_us_base_frame(base, label="base")
    _require_us_base_frame(acs, label="ACS")

    base_tables = _entity_table_refs(base)
    acs_tables = _entity_table_refs(acs)
    _validate_spine_metadata(base_tables, spine=ASEC_PUF_SPINE)
    _validate_spine_metadata(acs_tables, spine=ACS_2024_1YR_SPINE)
    base_support_metadata = _has_complete_support_metadata(
        base_tables,
        label="Base",
    )
    acs_support_metadata = False
    if base_support_metadata:
        acs_support_metadata = _has_complete_support_metadata(
            acs_tables,
            label="ACS",
        )
        if acs_support_metadata:
            _validate_acs_support_metadata(acs_tables)
    populate_acs_support_metadata = base_support_metadata and not acs_support_metadata
    column_orders = _aligned_column_orders(
        base,
        acs,
        add_acs_support_metadata=base_support_metadata,
    )
    id_offsets = _id_remap_offsets(base_tables, acs_tables)
    _validate_concat_strata(base, id_offsets)

    estimated_peak = _estimate_peak_bytes(
        base,
        acs,
        base_tables=base_tables,
        acs_tables=acs_tables,
        column_orders=column_orders,
        populate_acs_support_metadata=populate_acs_support_metadata,
        id_offsets=id_offsets,
    )
    if limit is not None and estimated_peak > limit:
        raise MemoryError(
            "Optional ACS pool assembly is estimated to require "
            f"{_format_bytes(estimated_peak)} of frame-resident peak memory, "
            f"above max_peak_bytes={_format_bytes(limit)}. Increase the limit "
            "only on a worker with sufficient memory, or reduce the input pool."
        )

    original_mass = base.weights_for("household").total
    base_target = original_mass * (1.0 - share)
    acs_target = original_mass - base_target
    household_weights, mass_log = _pooled_household_weights(
        base,
        acs,
        base_target=base_target,
        acs_target=acs_target,
        original_mass=original_mass,
    )
    tables, household_order = _combined_tables(
        base_tables,
        acs_tables,
        column_orders=column_orders,
        id_offsets=id_offsets,
        populate_acs_support_metadata=populate_acs_support_metadata,
    )
    if puma_ladder is not None:
        _assign_pooled_puma_ladder(
            tables,
            puma_ladder,
            seed=geography_seed,
            expected_congressional_district_vintage=(
                expected_congressional_district_vintage
            ),
        )
    if household_order is not None:
        reordered = household_weights.values[household_order]
        household_weights = Weights(reordered, household_weights.kind)

    household_weights = _with_exact_total(household_weights, original_mass)
    strata = pd.concat(
        [
            base.strata,
            pd.Series(
                ACS_2024_1YR_SPINE,
                index=acs.table("person").index,
                dtype=object,
                name="stratum",
            ),
        ],
        ignore_index=True,
    )
    result = Frame(
        tables,
        base.schema,
        {"household": household_weights},
        strata,
        mass_log=mass_log,
    )
    if not np.isclose(
        result.weights_for("household").total,
        original_mass,
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError("Optional ACS pool assembly failed to conserve mass.")
    return result


def estimate_optional_acs_pool_peak_bytes(base: Frame, acs: Frame) -> int:
    """Estimate frame-resident peak bytes for optional ACS pool assembly.

    This is a conservative data-dependent estimate rather than an RSS
    measurement. It counts both live input frames, two aligned output copies
    (the assembled pandas tables and the kernel's defensive copies), any group
    sort scratch known to be necessary, and a fixed pandas/Python allowance.
    """

    _require_us_base_frame(base, label="base")
    _require_us_base_frame(acs, label="ACS")
    base_tables = _entity_table_refs(base)
    acs_tables = _entity_table_refs(acs)
    _validate_spine_metadata(base_tables, spine=ASEC_PUF_SPINE)
    _validate_spine_metadata(acs_tables, spine=ACS_2024_1YR_SPINE)
    base_support_metadata = _has_complete_support_metadata(
        base_tables,
        label="Base",
    )
    acs_support_metadata = False
    if base_support_metadata:
        acs_support_metadata = _has_complete_support_metadata(
            acs_tables,
            label="ACS",
        )
        if acs_support_metadata:
            _validate_acs_support_metadata(acs_tables)
    column_orders = _aligned_column_orders(
        base,
        acs,
        add_acs_support_metadata=base_support_metadata,
    )
    id_offsets = _id_remap_offsets(base_tables, acs_tables)
    _validate_concat_strata(base, id_offsets)
    return _estimate_peak_bytes(
        base,
        acs,
        base_tables=base_tables,
        acs_tables=acs_tables,
        column_orders=column_orders,
        populate_acs_support_metadata=(
            base_support_metadata and not acs_support_metadata
        ),
        id_offsets=id_offsets,
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


def _validated_peak_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("max_peak_bytes must be a positive integer or None.")
    limit = int(value)
    if limit <= 0:
        raise ValueError("max_peak_bytes must be a positive integer or None.")
    return limit


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


def _combined_tables(
    base_tables: dict[str, pd.DataFrame],
    acs_tables: dict[str, pd.DataFrame],
    *,
    column_orders: dict[str, list[str]],
    id_offsets: dict[str, int],
    populate_acs_support_metadata: bool,
) -> tuple[dict[str, pd.DataFrame], np.ndarray | None]:
    """Assemble aligned entity tables without constructing source Frames."""

    tables: dict[str, pd.DataFrame] = {}
    household_order: np.ndarray | None = None
    for entity in US_SCHEMA.entities:
        base_view = _prepared_source_view(
            base_tables[entity],
            entity=entity,
            spine=ASEC_PUF_SPINE,
            id_offsets={},
            populate_support_metadata=False,
        )
        acs_view = _prepared_source_view(
            acs_tables[entity],
            entity=entity,
            spine=ACS_2024_1YR_SPINE,
            id_offsets=id_offsets,
            populate_support_metadata=populate_acs_support_metadata,
        )
        combined = pd.concat(
            [base_view, acs_view],
            ignore_index=True,
            sort=False,
        )
        expected_columns = column_orders[entity]
        if list(combined.columns) != expected_columns:
            combined = combined.loc[:, expected_columns]
        if entity in US_SCHEMA.group_entities:
            id_column = US_SCHEMA.id_column(entity)
            if not combined[id_column].is_monotonic_increasing:
                order = np.argsort(
                    combined[id_column].to_numpy(),
                    kind="stable",
                )
                combined = combined.iloc[order].reset_index(drop=True)
                if entity == "household":
                    household_order = order
        # Frame defensively copies every incoming table. Pandas' deep copy can
        # transiently allocate each unconsolidated block and a second merged
        # block, tripling the wide person table at the process peak. These are
        # newly owned assembly tables, so consolidating their blocks in place
        # first leaves Frame's public copy/validation contract intact while
        # bounding that transient to one merged block.
        consolidate = getattr(combined, "_consolidate_inplace", None)
        if callable(consolidate):
            consolidate()
        tables[entity] = combined
    return tables, household_order


def _assign_pooled_puma_ladder(
    tables: dict[str, pd.DataFrame],
    ladder: UsPumaLadder,
    *,
    seed: int,
    expected_congressional_district_vintage: str | None,
) -> None:
    """Assign the ladder while treating only ACS PUMAs as observed anchors."""

    household = tables["household"]
    tag = spine_column("household")
    if tag not in household:
        raise ValueError(
            f"Pooled household table lacks required spine tag {tag!r}."
        )
    if "puma" not in household:
        raise ValueError(
            "ACS household table must carry canonical seven-digit puma geoids "
            "before PUMA-ladder assignment."
        )
    if _PUMA_LADDER_ANCHOR_COLUMN in household:
        raise ValueError(
            "Pooled household table conflicts with the internal PUMA-ladder "
            f"anchor column {_PUMA_LADDER_ANCHOR_COLUMN!r}."
        )

    acs_rows = household[tag].eq(ACS_2024_1YR_SPINE)
    known = pd.to_numeric(household.loc[acs_rows, "puma"], errors="coerce")
    invalid = known.isna() | ~np.isfinite(known) | (known <= 0)
    if invalid.any():
        examples = household.loc[acs_rows, "puma"].loc[invalid].head().tolist()
        raise ValueError(
            "Every ACS household must carry its observed seven-digit PUMA "
            f"geoid before ladder assignment; invalid value(s): {examples}."
        )

    # ASEC rows must use the ladder's own state -> PUMA draw even if a future
    # donor happens to carry a similarly named column. Only the ACS spine's
    # source PUMA is an observed anchor for this multispine build.
    household[_PUMA_LADDER_ANCHOR_COLUMN] = household["puma"].where(acs_rows)
    assigned = assign_us_puma_ladder(
        household,
        ladder,
        seed=seed,
        expected_congressional_district_vintage=(
            expected_congressional_district_vintage
        ),
        puma_column=_PUMA_LADDER_ANCHOR_COLUMN,
    )
    assigned.drop(columns=[_PUMA_LADDER_ANCHOR_COLUMN], inplace=True)
    tables["household"] = assigned


def _prepared_source_view(
    source: pd.DataFrame,
    *,
    entity: str,
    spine: str,
    id_offsets: dict[str, int],
    populate_support_metadata: bool,
) -> pd.DataFrame:
    """Return a shallow source view with only new/replaced columns allocated."""

    column = spine_column(entity)
    table = source.copy(deep=False)
    # Always replace a compatible existing tag: nullable StringDtype columns
    # can pass validation with pd.NA because Series.all() skips missing values.
    # Every pooled record must receive a concrete spine value.
    table[column] = spine
    if populate_support_metadata:
        source_id, channel, clone_index = _support_metadata_columns(entity)
        table[source_id] = source[US_SCHEMA.entity_id_column(entity)].to_numpy()
        table[channel] = ACS_2024_1YR_SPINE
        table[clone_index] = 0

    if entity == US_SCHEMA.person_entity:
        person_offset = id_offsets.get(entity)
        if person_offset is not None:
            id_column = US_SCHEMA.person_id_column
            table[id_column] = source[id_column].to_numpy() + person_offset
        for group in US_SCHEMA.group_entities:
            offset = id_offsets.get(group)
            if offset is not None:
                membership = US_SCHEMA.membership_column(group)
                table[membership] = source[membership].to_numpy() + offset
    else:
        offset = id_offsets.get(entity)
        if offset is not None:
            id_column = US_SCHEMA.id_column(entity)
            table[id_column] = source[id_column].to_numpy() + offset
    return table


def _validate_spine_metadata(
    tables: dict[str, pd.DataFrame],
    *,
    spine: str,
) -> None:
    for entity, table in tables.items():
        column = spine_column(entity)
        if column in table and not table[column].eq(spine).all():
            raise ValueError(
                f"{entity!r} already carries conflicting spine metadata in {column!r}."
            )


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


def _id_remap_offsets(
    base_tables: dict[str, pd.DataFrame],
    acs_tables: dict[str, pd.DataFrame],
) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        mine = base_tables[entity][id_column].to_numpy()
        theirs = acs_tables[entity][id_column].to_numpy()
        if not _arrays_overlap(mine, theirs):
            continue
        if not (
            np.issubdtype(mine.dtype, np.integer)
            and np.issubdtype(theirs.dtype, np.integer)
        ):
            raise ValueError(
                f"Cannot concat: id spaces for entity {entity!r} overlap and "
                f"are not integer-typed ({mine.dtype} vs {theirs.dtype}); "
                "remap ids to disjoint spaces before concatenating."
            )
        offsets[entity] = int(mine.max()) + 1 - int(theirs.min())
    return offsets


def _arrays_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    """Return exact ID overlap while bounding temporary arrays to one chunk."""

    if left.size == 0 or right.size == 0:
        return False
    if left.max() < right.min() or right.max() < left.min():
        return False

    left_sorted = pd.Index(left).is_monotonic_increasing
    right_sorted = pd.Index(right).is_monotonic_increasing
    if left_sorted and right_sorted:
        probes, target = (left, right) if left.size <= right.size else (right, left)
    elif left_sorted:
        probes, target = right, left
    elif right_sorted:
        probes, target = left, right
    elif left.size <= right.size:
        probes, target = right, np.sort(left)
    else:
        probes, target = left, np.sort(right)

    for start in range(0, probes.size, _ID_OVERLAP_CHUNK_ROWS):
        chunk = probes[start : start + _ID_OVERLAP_CHUNK_ROWS]
        positions = np.searchsorted(target, chunk)
        in_bounds = positions < target.size
        if (
            in_bounds.any()
            and np.equal(
                target[positions[in_bounds]],
                chunk[in_bounds],
            ).any()
        ):
            return True
    return False


def _validate_concat_strata(base: Frame, id_offsets: dict[str, int]) -> None:
    if ACS_2024_1YR_SPINE not in set(base.strata.unique()) or not id_offsets:
        return
    raise ValueError(
        "Cannot concat: bundles share strata "
        f"{[ACS_2024_1YR_SPINE]} and overlapping id spaces for entities "
        f"{list(id_offsets)}; concatenated strata must differ or id spaces "
        "must be disjoint."
    )


def _pooled_household_weights(
    base: Frame,
    acs: Frame,
    *,
    base_target: float,
    acs_target: float,
    original_mass: float,
) -> tuple[Weights, tuple[MassChangeRecord, ...]]:
    base_existing = base.weights_for("household")
    acs_existing = acs.weights_for("household")
    base_factor = base_target / base_existing.total
    acs_factor = acs_target / acs_existing.total
    base_values = base_existing.values * base_factor
    acs_values = acs_existing.values * acs_factor
    # Rescaling and mixing distinct source frames creates importance weights.
    # A calibrated donor does not make the new, explicitly uncalibrated union
    # calibrated; the downstream solve owns that transition.
    pooled = Weights(
        np.concatenate([base_values, acs_values]),
        WeightKind.IMPORTANCE,
    )
    mass_log = (
        *base.mass_log,
        MassChangeRecord(
            entity="household",
            old_total=base_existing.total,
            new_total=float(base_values.sum()),
            declared_factor=base_factor,
            reason="allocated ASEC-by-PUF mass for the optional ACS multispine pool",
        ),
        *acs.mass_log,
        MassChangeRecord(
            entity="household",
            old_total=acs_existing.total,
            new_total=float(acs_values.sum()),
            declared_factor=acs_factor,
            reason="allocated ACS mass for the optional ACS multispine pool",
        ),
    )
    if not np.isclose(pooled.total, original_mass, rtol=1e-9, atol=0.0):
        raise RuntimeError("Optional ACS weight allocation changed household mass.")
    return pooled, mass_log


def _with_exact_total(weights: Weights, target: float) -> Weights:
    if weights.total == target:
        return weights
    values = _values_nearest_total(weights.values, target)
    return Weights(values, weights.kind)


def _values_nearest_total(values: np.ndarray, target: float) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    correction_index = int(np.argmax(result))
    result[correction_index] += target - float(result.sum())
    return result


def _estimate_peak_bytes(
    base: Frame,
    acs: Frame,
    *,
    base_tables: dict[str, pd.DataFrame],
    acs_tables: dict[str, pd.DataFrame],
    column_orders: dict[str, list[str]],
    populate_acs_support_metadata: bool,
    id_offsets: dict[str, int],
) -> int:
    input_bytes = _frame_storage_bytes(base) + _frame_storage_bytes(acs)
    output_table_bytes: dict[str, int] = {}
    for entity in US_SCHEMA.entities:
        output_table_bytes[entity] = _estimated_aligned_table_bytes(
            base_tables[entity],
            acs_tables[entity],
            entity=entity,
            columns=column_orders[entity],
            populate_acs_support_metadata=populate_acs_support_metadata,
        )

    output_bytes = sum(output_table_bytes.values())
    output_bytes += _estimated_strata_bytes(base, acs)
    output_bytes += 8 * (base.n("household") + acs.n("household"))
    sort_scratch = max(
        (
            output_table_bytes[entity]
            for entity in US_SCHEMA.group_entities
            if _group_concat_needs_sort(
                base_tables[entity],
                acs_tables[entity],
                entity=entity,
                id_offsets=id_offsets,
            )
        ),
        default=0,
    )
    return int(
        input_bytes
        + 2 * output_bytes
        + sort_scratch
        + _PEAK_ESTIMATE_FIXED_OVERHEAD_BYTES
    )


def _frame_storage_bytes(frame: Frame) -> int:
    tables = sum(
        int(frame.table(entity).memory_usage(index=True, deep=True).sum())
        for entity in frame.entities
    )
    weights = sum(
        frame.weights_for(entity).values.nbytes for entity in frame.weighted_entities
    )
    strata = int(frame.strata.memory_usage(index=True, deep=True))
    return tables + weights + strata


def _estimated_aligned_table_bytes(
    base: pd.DataFrame,
    acs: pd.DataFrame,
    *,
    entity: str,
    columns: list[str],
    populate_acs_support_metadata: bool,
) -> int:
    n_base = len(base)
    n_acs = len(acs)
    tag = spine_column(entity)
    support_columns = _support_metadata_columns(entity)
    total = 8 * (n_base + n_acs)  # Conservative materialized-index allowance.
    for column in columns:
        if column == tag:
            base_series = _constant_object_series(ASEC_PUF_SPINE)
            acs_series = _constant_object_series(ACS_2024_1YR_SPINE)
        else:
            base_series = base[column] if column in base else None
            acs_series = acs[column] if column in acs else None
            if (
                acs_series is None
                and populate_acs_support_metadata
                and column in support_columns
            ):
                source_id, channel, clone_index = support_columns
                if column == source_id:
                    acs_series = acs[US_SCHEMA.entity_id_column(entity)]
                elif column == channel:
                    acs_series = _constant_object_series(ACS_2024_1YR_SPINE)
                elif column == clone_index:
                    acs_series = pd.Series([0], dtype=np.int64)
        base_width = _column_width(base_series) if base_series is not None else None
        acs_width = _column_width(acs_series) if acs_series is not None else None
        present_widths = [
            width for width in (base_width, acs_width) if width is not None
        ]
        width = max(
            *present_widths,
            _concat_sample_width(base_series, acs_series),
            8,
        )
        total += width * (n_base + n_acs)
    return int(total)


def _column_width(series: pd.Series) -> int:
    if len(series) == 0:
        return max(8, int(getattr(series.dtype, "itemsize", 8)))
    return max(
        1,
        int(np.ceil(series.memory_usage(index=False, deep=True) / len(series))),
    )


def _constant_object_series(value: str) -> pd.Series:
    return pd.Series([value], dtype=object)


def _concat_sample_width(
    base: pd.Series | None,
    acs: pd.Series | None,
) -> int:
    """Estimate pandas' promoted storage width from one value per source."""

    parts = [
        series.iloc[:1] if series is not None else pd.Series([np.nan])
        for series in (base, acs)
    ]
    combined = pd.concat(parts, ignore_index=True)
    if isinstance(combined.dtype, pd.CategoricalDtype):
        # The sample retains the entire category dictionary. Full-column
        # widths above already amortize that fixed dictionary correctly.
        return 1
    return int(np.ceil(combined.memory_usage(index=False, deep=True) / len(combined)))


def _estimated_strata_bytes(base: Frame, acs: Frame) -> int:
    base_width = _column_width(base.strata)
    acs_width = _column_width(_constant_object_series(ACS_2024_1YR_SPINE))
    return max(base_width, acs_width) * (base.n("person") + acs.n("person"))


def _group_concat_needs_sort(
    base: pd.DataFrame,
    acs: pd.DataFrame,
    *,
    entity: str,
    id_offsets: dict[str, int],
) -> bool:
    if entity in id_offsets:
        return False
    id_column = US_SCHEMA.id_column(entity)
    return bool(base[id_column].iloc[-1] > acs[id_column].iloc[0])


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"
