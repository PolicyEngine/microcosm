"""Optional ACS augmentation of the US ASEC-by-PUF base pool."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.block_location import (
    LOCATION_CANDIDATE_BLOCKS_COLUMN,
    LOCATION_CLONE_INDEX_COLUMN,
    LOCATION_SOURCE_GEOGRAPHY_COLUMN,
    SOURCE_GEOGRAPHY_PUMA,
    SOURCE_GEOGRAPHY_STATE,
    US_LOCATION_RULE_BLOCK_V1,
    US_LOCATION_RULE_CHOICES,
    US_LOCATION_RULE_LEGACY,
    UsLocationLadder,
    derive_us_block_geography,
    draw_us_block_locations,
    location_geography_columns,
    us_block_location_manifest,
)
from microcosm.build.us_runtime.puf_support import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.puma_ladder import (
    UsPumaLadder,
    assign_us_puma_ladder,
)
from microcosm.frame import (
    US_SCHEMA,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)

__all__ = [
    "ACS_2024_1YR_SPINE",
    "ACS_POOL_BLOCK_LOCATION_METADATA_KEY",
    "ACS_POOL_LOCATION_CLONES_REFUSAL",
    "ASEC_PUF_SPINE",
    "DEFAULT_ACS_POOL_PEAK_LIMIT_BYTES",
    "DONOR_ASSIGNED_GEOGRAPHY_COLUMNS",
    "DONOR_BLOCK_PRESERVED",
    "DONOR_BLOCK_STATE_DRAWN",
    "donor_assigned_geography_complete",
    "estimate_optional_acs_pool_peak_bytes",
    "preflight_pooled_ladder_geography",
    "spine_column",
    "validate_pool_location_options",
    "with_optional_acs_spine",
]

ASEC_PUF_SPINE = "asec_puf"

#: The block-ladder columns whose presence marks a donor with certified
#: assigned geography (the base-O build line). Such a donor keeps its
#: congressional district and county through pooling, with its PUMA derived
#: exactly from its 2020 tract instead of drawn within its state.
DONOR_ASSIGNED_GEOGRAPHY_COLUMNS = (
    "tract_geoid",
    "congressional_district_geoid",
    "county_fips",
)

# The launch worker has a 30 GB RSS budget. The estimate includes the two
# incoming frames, the assembled tables, the Frame constructor's defensive
# copies, and a fixed allowance for pandas/Python scratch allocations.
DEFAULT_ACS_POOL_PEAK_LIMIT_BYTES = int(
    os.environ.get("POPULACE_ACS_POOL_PEAK_LIMIT_BYTES", 30_000_000_000)
)
_PEAK_ESTIMATE_FIXED_OVERHEAD_BYTES = 256 * 1024**2
_ID_OVERLAP_CHUNK_ROWS = 65_536
_PUMA_LADDER_ANCHOR_COLUMN = "__microcosm_puma_ladder_anchor"

#: Frame-metadata key of the pooled ``block_v1`` location record (the
#: :func:`~microcosm.build.us_runtime.block_location.us_block_location_manifest`
#: record plus this line's donor-block accounting).
ACS_POOL_BLOCK_LOCATION_METADATA_KEY = "acs_pool_block_location"
#: Donor (ASEC-by-PUF) row outcomes under ``block_v1``: a donor that already
#: carries a ladder block keeps it (no redraw); a donor without one draws a
#: block within its state.
DONOR_BLOCK_PRESERVED = "preserved_donor_block"
DONOR_BLOCK_STATE_DRAWN = "state_drawn_missing_donor_block"
#: Why this line refuses ``location_clones > 1`` today; admitting K > 1 end
#: to end is the follow-up to microcosm#696.
ACS_POOL_LOCATION_CLONES_REFUSAL = (
    "location_clones > 1 is not yet admitted on the ACS local-release line. "
    "(1) block_location.clone_frame_for_location adds an "
    "'{entity}_location_clone_index' column to every entity table; those "
    "names are model-named, are not policyengine-us variables and are not in "
    "microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS, so the "
    "package stage's stored-inputs gate "
    "(tools/build_us_acs_local_release.py:_require_stored_inputs -> "
    "stored_inputs.require_h5_stored_inputs) refuses the release. "
    "(2) base_pool._assign_pooled_block_location preserves each ASEC-by-PUF "
    "donor's already-assigned block rather than redrawing it, so K copies of "
    "a preserved donor would be K identical locations, not K draws; no rule "
    "for redrawing preserved donor blocks per clone exists yet. "
    "Use location_clones=1."
)


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
    location_rule: str = US_LOCATION_RULE_LEGACY,
    block_ladder: UsLocationLadder | None = None,
    location_seed: int = 0,
    location_clones: int = 1,
) -> Frame:
    """Append an ACS frame while conserving the base's household mass.

    The no-ACS path is an identity operation: it returns ``base`` itself before
    validating, copying, tagging, estimating memory, or changing weights. When
    ACS is present, the two frames receive entity-prefixed spine tags, are
    aligned with missing values for source-specific columns, and are assembled
    with the same ID-remapping semantics as :meth:`microcosm.frame.Frame.concat`.

    ``acs_share`` allocates that share of the incoming base household mass to
    ACS. The remaining share stays on the ASEC-by-PUF spine. Each allocation is
    recorded as a deliberate :class:`~microcosm.frame.MassChangeRecord`.

    When ``puma_ladder`` is present, ACS records retain their known 2020 PUMA.
    ASEC-by-PUF records without assigned geography draw a PUMA within their
    state and then draw a congressional district and county from the ladder;
    records carrying the certified block-ladder triple (tract, congressional
    district, county) keep their assigned district and county, with PUMA
    derived exactly from the 2020 tract (see
    :func:`donor_assigned_geography_complete`). Assignment happens on the
    newly assembled household table before the one final
    :class:`~microcosm.frame.Frame` construction, avoiding a second copy of
    every entity table.

    ``max_peak_bytes`` is a fail-fast bound on estimated frame-resident peak
    memory. Pass ``None`` to disable the preflight. The estimate includes both
    inputs and the defensive copies made by :class:`~microcosm.frame.Frame`; it
    intentionally does not claim to account for unrelated objects already in
    the process.

    ``location_rule`` selects the household location rule. ``"legacy"`` (the
    default) is the PUMA-ladder behaviour above, byte for byte, and refuses
    ``block_ladder``, a nonzero ``location_seed`` and ``location_clones != 1``.
    ``"block_v1"`` (microcosm#696) replaces the PUMA-ladder draw with one 2020
    census block per household from ``block_ladder`` (which must carry the
    per-block ``puma``): ACS rows draw a block within their observed PUMA;
    ASEC-by-PUF donor rows that already carry a ladder block keep it (no
    redraw); donor rows without one draw a block within their state. Every
    geography column then derives from the block (see
    :func:`_assign_pooled_block_location`), and the location record is stored
    in the result's metadata under :data:`ACS_POOL_BLOCK_LOCATION_METADATA_KEY`.
    ``puma_ladder`` and ``geography_seed`` play no part in a ``block_v1``
    assignment, so a ``puma_ladder`` or a nonzero ``geography_seed`` is
    refused rather than ignored.
    """

    if acs is None:
        return base

    validate_pool_location_options(
        location_rule,
        block_ladder=block_ladder,
        location_seed=location_seed,
        location_clones=location_clones,
        geography_seed=geography_seed,
    )
    if location_rule == US_LOCATION_RULE_BLOCK_V1 and puma_ladder is not None:
        raise ValueError(
            "puma_ladder assigns geography only under the legacy location "
            "rule; a block_v1 pool derives every geography from block_ladder. "
            "Refusing puma_ladder rather than ignoring it."
        )
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
    location_record: dict[str, Any] | None = None
    if location_rule == US_LOCATION_RULE_BLOCK_V1:
        assert block_ladder is not None  # validate_pool_location_options
        location_record = _assign_pooled_block_location(
            tables,
            block_ladder,
            seed=location_seed,
            expected_congressional_district_vintage=(
                expected_congressional_district_vintage
            ),
        )
    elif puma_ladder is not None:
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
    if location_record is None:
        result = Frame(
            tables,
            base.schema,
            {"household": household_weights},
            strata,
            mass_log=mass_log,
        )
    else:
        result = Frame(
            tables,
            base.schema,
            {"household": household_weights},
            strata,
            mass_log=mass_log,
            metadata={ACS_POOL_BLOCK_LOCATION_METADATA_KEY: location_record},
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


def donor_assigned_geography_complete(
    household: pd.DataFrame,
    donor_mask: pd.Series | None = None,
) -> bool:
    """True when every (masked) row carries the assigned-geography triple.

    The triple is :data:`DONOR_ASSIGNED_GEOGRAPHY_COLUMNS` — the columns a
    donor built through the block-ladder geography stage carries. All-present
    and all-non-empty means the donor's certified assignment can be preserved
    through pooling; all-absent means the legacy state-conditional PUMA draw
    applies. Anything in between is a defective donor and fails loudly rather
    than silently mixing preserved and drawn geography.
    """

    present = [
        column
        for column in DONOR_ASSIGNED_GEOGRAPHY_COLUMNS
        if column in household.columns
    ]
    if not present:
        return False
    if len(present) != len(DONOR_ASSIGNED_GEOGRAPHY_COLUMNS):
        missing = sorted(set(DONOR_ASSIGNED_GEOGRAPHY_COLUMNS) - set(present))
        raise ValueError(
            "Donor household table carries a partial assigned-geography "
            f"surface: {sorted(present)} present but {missing} absent. A "
            "donor either provides the full block-ladder triple or none of it."
        )
    view = household if donor_mask is None else household.loc[donor_mask]
    incomplete: dict[str, int] = {}
    for column in DONOR_ASSIGNED_GEOGRAPHY_COLUMNS:
        values = view[column]
        empty = values.isna() | values.astype(str).str.strip().eq("")
        if empty.any():
            incomplete[column] = int(empty.sum())
    if incomplete:
        raise ValueError(
            "Donor household rows carry a partially assigned geography "
            f"surface (empty value counts: {incomplete}). Preserving a mix "
            "of assigned and unassigned donor geography would silently blend "
            "certified and drawn assignments; fix the donor instead."
        )
    return True


def _tract_to_puma(ladder: UsPumaLadder) -> pd.Series:
    """The ladder's (PUMA, tract) overlap read as an exact tract -> PUMA map.

    2020 tracts nest in exactly one 2020 PUMA, so each tract appears in the
    overlap table under a single PUMA; a duplicate with a conflicting PUMA is
    a ladder defect, not an overlap.
    """

    tracts = np.asarray(ladder.tract_overlap_tract, dtype=np.int64)
    pumas = np.asarray(ladder.tract_overlap_puma, dtype=np.int64)
    mapping = pd.Series(pumas, index=tracts)
    duplicated = mapping.index.duplicated(keep=False)
    if duplicated.any():
        conflicting = mapping[duplicated].groupby(level=0).nunique()
        conflicting = conflicting[conflicting > 1]
        if len(conflicting):
            raise ValueError(
                "US PUMA ladder tract overlap assigns tract(s) to multiple "
                f"PUMAs: {conflicting.head().index.tolist()}; 2020 tracts "
                "nest in exactly one 2020 PUMA."
            )
        mapping = mapping[~mapping.index.duplicated(keep="first")]
    return mapping


def _derived_donor_puma(
    household: pd.DataFrame,
    donor_rows: pd.Series,
    ladder: UsPumaLadder,
) -> pd.Series:
    """Derive donor PUMAs from assigned tracts via the ladder overlap."""

    tracts = pd.to_numeric(household.loc[donor_rows, "tract_geoid"], errors="coerce")
    invalid = (
        tracts.isna()
        | ~np.isfinite(tracts)
        | (tracts <= 0)
        | (np.mod(tracts.fillna(0.5), 1) != 0)
    )
    if invalid.any():
        examples = household.loc[donor_rows, "tract_geoid"].loc[invalid].head().tolist()
        raise ValueError(
            "Donor households must carry parseable integral 11-digit tract "
            f"geoids to derive their PUMA; invalid value(s): {examples}."
        )
    mapping = _tract_to_puma(ladder)
    derived = tracts.astype(np.int64).map(mapping)
    unmapped = derived.isna()
    if unmapped.any():
        examples = sorted(set(tracts.astype(np.int64)[unmapped].head().tolist()))
        raise ValueError(
            "Donor tract geoid(s) absent from the US PUMA ladder tract "
            f"overlap: {examples}. The donor's geography vintage must match "
            "the ladder's 2020 tract layer."
        )
    derived = derived.astype(np.int64)
    states = pd.to_numeric(household.loc[donor_rows, "state_fips"]).astype(np.int64)
    inconsistent = (derived // 100_000) != states
    if inconsistent.any():
        examples = sorted(
            set(
                zip(
                    states[inconsistent].head(),
                    derived[inconsistent].head(),
                    strict=True,
                )
            )
        )
        raise ValueError(
            f"Derived donor PUMA state prefixes disagree with state_fips: {examples}."
        )
    return derived


def _preserved_donor_geography(
    household: pd.DataFrame,
    donor_rows: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Validate and normalize the donor's assigned CD and county columns."""

    states = pd.to_numeric(household.loc[donor_rows, "state_fips"]).astype(np.int64)
    cd = pd.to_numeric(
        household.loc[donor_rows, "congressional_district_geoid"],
        errors="coerce",
    )
    cd_invalid = cd.isna() | (np.mod(cd.fillna(0.5), 1) != 0)
    if cd_invalid.any():
        examples = (
            household.loc[donor_rows, "congressional_district_geoid"]
            .loc[cd_invalid]
            .head()
            .tolist()
        )
        raise ValueError(
            "Donor congressional_district_geoid values must be integral "
            f"state*100+district geoids; invalid value(s): {examples}."
        )
    cd = cd.astype(np.int64)
    if ((cd // 100) != states).any():
        raise ValueError(
            "Donor congressional_district_geoid state prefixes disagree "
            "with state_fips."
        )
    county = pd.to_numeric(household.loc[donor_rows, "county_fips"], errors="coerce")
    county_invalid = county.isna() | (np.mod(county.fillna(0.5), 1) != 0)
    if county_invalid.any():
        examples = (
            household.loc[donor_rows, "county_fips"].loc[county_invalid].head().tolist()
        )
        raise ValueError(
            "Donor county_fips values must be integral 5-digit state+county "
            f"codes; invalid value(s): {examples}."
        )
    county = county.astype(np.int64)
    if ((county // 1_000) != states).any():
        raise ValueError("Donor county_fips state prefixes disagree with state_fips.")
    # 2020 tracts nest in counties: the assigned county must equal the
    # tract's own county prefix, or the preserved pair is incoherent.
    tracts = pd.to_numeric(
        household.loc[donor_rows, "tract_geoid"], errors="coerce"
    ).astype(np.int64)
    mismatched = (tracts // 1_000_000) != county
    if mismatched.any():
        examples = sorted(
            set(
                zip(
                    tracts[mismatched].head().tolist(),
                    county[mismatched].head().tolist(),
                    strict=True,
                )
            )
        )
        raise ValueError(
            "Donor county_fips disagrees with the assigned tract's county "
            f"prefix: {examples}."
        )
    county_strings = pd.Series(
        [f"{value:05d}" for value in county.tolist()],
        index=county.index,
        dtype=object,
    )
    return cd, county_strings


def preflight_pooled_ladder_geography(
    base_household: pd.DataFrame,
    acs_household: pd.DataFrame,
    ladder: UsPumaLadder | None,
    *,
    location_rule: str = US_LOCATION_RULE_LEGACY,
    block_ladder: UsLocationLadder | None = None,
) -> str:
    """Validate everything pooled ladder assignment needs, before transfer.

    The QRF transfer is the expensive stage; geography incompatibilities
    (an unmapped donor tract, incoherent preserved values, an ACS PUMA the
    ladder does not know, a donor state without ladder PUMAs) must fail
    here, not after the fits have run. Returns the donor geography mode
    (``"preserved_assigned"`` or ``"ladder_drawn"``); raises ``ValueError``
    on any incompatibility, matching the assignment-time checks.

    Under ``location_rule="block_v1"`` the checks are the block assignment's
    instead (:func:`_assign_pooled_block_location`), against
    ``block_ladder``; ``ladder`` is not consulted. The returned mode is then
    :data:`DONOR_BLOCK_PRESERVED` (every donor carries a ladder block),
    :data:`DONOR_BLOCK_STATE_DRAWN` (none does) or
    ``"preserved_donor_block_with_state_draws"`` (some do).
    """

    if location_rule != US_LOCATION_RULE_LEGACY:
        if location_rule != US_LOCATION_RULE_BLOCK_V1:
            raise ValueError(
                f"location_rule must be one of {US_LOCATION_RULE_CHOICES}, got "
                f"{location_rule!r}."
            )
        if block_ladder is None:
            raise ValueError("location_rule='block_v1' requires a block_ladder.")
        return _preflight_pooled_block_location(
            base_household, acs_household, block_ladder
        )
    if block_ladder is not None:
        raise ValueError(
            "block_ladder is only used by location_rule='block_v1'; the legacy "
            "rule assigns geography from the PUMA ladder."
        )
    if ladder is None:
        raise ValueError("The legacy location rule requires a PUMA ladder.")

    mode = "ladder_drawn"
    if len(base_household):
        donor_rows = pd.Series(True, index=base_household.index)
        if donor_assigned_geography_complete(base_household):
            _derived_donor_puma(base_household, donor_rows, ladder)
            _preserved_donor_geography(base_household, donor_rows)
            mode = "preserved_assigned"
        else:
            states = set(pd.to_numeric(base_household["state_fips"]).astype(np.int64))
            ladder_states = set((ladder.puma // 100_000).astype(int).tolist())
            missing = sorted(states - ladder_states)
            if missing:
                raise ValueError(
                    "Donor state_fips without any ladder PUMA (state-"
                    f"conditional draw impossible): {missing}."
                )
    if len(acs_household):
        if "puma" not in acs_household.columns:
            raise ValueError(
                "ACS household table must carry canonical seven-digit puma "
                "geoids before PUMA-ladder assignment."
            )
        parsed = pd.to_numeric(acs_household["puma"], errors="coerce")
        invalid = parsed.isna() | ~np.isfinite(parsed) | (parsed <= 0)
        if invalid.any():
            examples = acs_household["puma"].loc[invalid].head().tolist()
            raise ValueError(
                "Every ACS household must carry its observed seven-digit "
                f"PUMA geoid; invalid value(s): {examples}."
            )
        known = np.isin(parsed.astype(np.int64).to_numpy(), ladder.puma)
        if not known.all():
            examples = sorted(set(parsed.astype(np.int64)[~known].head().tolist()))
            raise ValueError(f"ACS PUMA geoid(s) absent from the ladder: {examples}.")
    return mode


def _assign_pooled_puma_ladder(
    tables: dict[str, pd.DataFrame],
    ladder: UsPumaLadder,
    *,
    seed: int,
    expected_congressional_district_vintage: str | None,
) -> None:
    """Assign the ladder, preserving a donor's certified assigned geography.

    ACS PUMAs are always observed anchors. Donor (ASEC-by-PUF) rows come in
    two build lines: a donor without assigned geography draws a PUMA within
    its state (the buildl behavior), while a donor carrying the block-ladder
    triple (tract, congressional district, county — the base-O behavior)
    keeps its certified congressional district and county, with its PUMA
    derived exactly from its 2020 tract. Assigned and drawn donor geography
    never mix; partial assignment fails loudly.
    """

    household = tables["household"]
    tag = spine_column("household")
    if tag not in household:
        raise ValueError(f"Pooled household table lacks required spine tag {tag!r}.")
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

    donor_rows = ~acs_rows
    preserve = bool(donor_rows.any()) and donor_assigned_geography_complete(
        household, donor_rows
    )

    # Anchor semantics: the ACS spine's source PUMA is always observed. A
    # donor with certified assigned geography anchors on its tract-derived
    # PUMA; a legacy donor leaves its anchor empty and draws within state.
    anchor = household["puma"].where(acs_rows).astype(object)
    preserved_cd: pd.Series | None = None
    preserved_county: pd.Series | None = None
    if preserve:
        derived_puma = _derived_donor_puma(household, donor_rows, ladder)
        preserved_cd, preserved_county = _preserved_donor_geography(
            household, donor_rows
        )
        anchor.loc[donor_rows] = pd.Series(
            [f"{value:07d}" for value in derived_puma.tolist()],
            index=derived_puma.index,
            dtype=object,
        )
    household[_PUMA_LADDER_ANCHOR_COLUMN] = anchor
    assigned = assign_us_puma_ladder(
        household,
        ladder,
        seed=seed,
        expected_congressional_district_vintage=(
            expected_congressional_district_vintage
        ),
        puma_column=_PUMA_LADDER_ANCHOR_COLUMN,
    )
    if preserve:
        assert preserved_cd is not None and preserved_county is not None
        assigned.loc[donor_rows, "congressional_district_geoid"] = preserved_cd
        assigned.loc[donor_rows, "county_fips"] = preserved_county
    assigned.drop(columns=[_PUMA_LADDER_ANCHOR_COLUMN], inplace=True)
    tables["household"] = assigned


def validate_pool_location_options(
    location_rule: str,
    *,
    block_ladder: UsLocationLadder | None,
    location_seed: int,
    location_clones: int,
    geography_seed: int,
) -> None:
    """Refuse location options the selected rule would silently ignore."""

    if location_rule not in US_LOCATION_RULE_CHOICES:
        raise ValueError(
            f"location_rule must be one of {US_LOCATION_RULE_CHOICES}, got "
            f"{location_rule!r}."
        )
    if isinstance(location_seed, bool) or not isinstance(
        location_seed, (int, np.integer)
    ):
        raise ValueError(f"location_seed must be an integer, got {location_seed!r}.")
    if isinstance(location_clones, bool) or not isinstance(
        location_clones, (int, np.integer)
    ):
        raise ValueError(
            f"location_clones must be an integer, got {location_clones!r}."
        )
    if location_rule == US_LOCATION_RULE_LEGACY:
        if block_ladder is not None:
            raise ValueError(
                "block_ladder is only used by location_rule='block_v1'; the "
                "legacy rule assigns geography from the PUMA ladder."
            )
        if int(location_seed) != 0 or int(location_clones) != 1:
            raise ValueError(
                "location_seed and location_clones apply only to "
                "location_rule='block_v1'; the legacy rule takes its seed from "
                f"geography_seed (got location_seed={location_seed!r}, "
                f"location_clones={location_clones!r})."
            )
        return
    if block_ladder is None:
        raise ValueError("location_rule='block_v1' requires a block_ladder.")
    if block_ladder.puma is None:
        raise ValueError(
            "location_rule='block_v1' on the ACS pool needs a block ladder that "
            "carries the per-block 'puma' array: ACS rows draw within their "
            "observed PUMA."
        )
    if int(location_seed) < 0:
        raise ValueError(f"location_seed must be non-negative, got {location_seed!r}.")
    if int(location_clones) < 1:
        raise ValueError(f"location_clones must be >= 1, got {location_clones!r}.")
    if int(location_clones) > 1:
        raise ValueError(ACS_POOL_LOCATION_CLONES_REFUSAL)
    if int(geography_seed) != 0:
        raise ValueError(
            "geography_seed seeds only the legacy PUMA-ladder draw; under "
            "location_rule='block_v1' the draw is seeded by location_seed. "
            f"Refusing geography_seed={geography_seed!r} rather than ignoring it."
        )


def _acs_block_pumas(puma: pd.Series, ladder: UsLocationLadder) -> np.ndarray:
    """Each ACS household's observed 2020 PUMA, validated against the ladder."""

    parsed = pd.to_numeric(puma, errors="coerce")
    invalid = parsed.isna() | ~np.isfinite(parsed) | (parsed <= 0)
    if invalid.any():
        examples = puma.loc[invalid].head().tolist()
        raise ValueError(
            "Every ACS household must carry its observed seven-digit PUMA "
            f"geoid before block location; invalid value(s): {examples}."
        )
    values = parsed.astype(np.int64).to_numpy()
    assert ladder.puma is not None  # validate_pool_location_options
    known = np.isin(values, ladder.puma)
    if not known.all():
        examples = sorted(set(values[~known][:5].tolist()))
        raise ValueError(
            f"ACS PUMA geoid(s) absent from the block ladder's PUMA layer: {examples}."
        )
    return values


def _donor_block_index(
    donor: pd.DataFrame, ladder: UsLocationLadder
) -> tuple[np.ndarray, np.ndarray]:
    """Donor rows' already-assigned blocks as ladder indices.

    Returns ``(has_block, block_index)``: a donor row whose ``block_geoid`` is
    missing or blank has no block (``-1``) and will draw within its state. A
    present block must be a 15-digit code the ladder carries, in the row's
    ``state_fips``; anything else fails loudly, because preserving an
    unknown or mis-stated block would publish geography the ladder cannot
    derive.
    """

    n = len(donor)
    if "block_geoid" not in donor.columns or n == 0:
        return np.zeros(n, dtype=bool), np.full(n, -1, dtype=np.int64)
    raw = donor["block_geoid"]
    if pd.api.types.is_numeric_dtype(raw):
        # An integer-typed block geoid loses the leading zero of states 01-09.
        text = pd.Series(
            ["" if pd.isna(value) else f"{int(value):015d}" for value in raw.tolist()],
            index=raw.index,
            dtype=object,
        )
    else:
        text = raw.astype(object).where(raw.notna(), "").astype(str).str.strip()
    has_block = (text != "").to_numpy()
    index = np.full(n, -1, dtype=np.int64)
    if not has_block.any():
        return has_block, index
    present = text.to_numpy()[has_block].astype(np.str_)
    malformed = ~((np.char.str_len(present) == 15) & np.char.isdigit(present))
    if malformed.any():
        examples = present[malformed][:5].tolist()
        raise ValueError(
            "Donor block_geoid values must be 15-digit 2020 block codes; "
            f"invalid value(s): {examples}."
        )
    blocks = present.astype(np.int64)
    position = np.clip(np.searchsorted(ladder.block_geoid, blocks), 0, len(ladder) - 1)
    unknown = ladder.block_geoid[position] != blocks
    if unknown.any():
        examples = sorted(set(blocks[unknown][:5].tolist()))
        raise ValueError(
            "Donor block_geoid(s) absent from the block ladder: "
            f"{[f'{value:015d}' for value in examples]}. The donor's block "
            "vintage must match the ladder's 2020 tabulation blocks."
        )
    states = pd.to_numeric(donor["state_fips"], errors="coerce").to_numpy()[has_block]
    mismatch = (blocks // 10**13) != states
    if mismatch.any():
        examples = sorted(
            {
                (f"{block:015d}", int(state))
                for block, state in zip(
                    blocks[mismatch][:5].tolist(),
                    states[mismatch][:5].tolist(),
                    strict=True,
                )
            }
        )
        raise ValueError(
            f"Donor block_geoid state prefixes disagree with state_fips: {examples}."
        )
    index[has_block] = position
    return has_block, index


def _require_ladder_states(state_fips: pd.Series, ladder: UsLocationLadder) -> None:
    states = set(pd.to_numeric(state_fips).astype(np.int64).tolist())
    ladder_states = set((ladder.block_geoid // 10**13).astype(int).tolist())
    missing = sorted(states - ladder_states)
    if missing:
        raise ValueError(
            "Donor state_fips without any ladder block (state-conditional "
            f"block draw impossible): {missing}."
        )


def _block_donor_mode(has_block: np.ndarray) -> str:
    if has_block.all():
        return DONOR_BLOCK_PRESERVED
    if not has_block.any():
        return DONOR_BLOCK_STATE_DRAWN
    return f"{DONOR_BLOCK_PRESERVED}_with_state_draws"


def _preflight_pooled_block_location(
    base_household: pd.DataFrame,
    acs_household: pd.DataFrame,
    ladder: UsLocationLadder,
) -> str:
    """The ``block_v1`` half of :func:`preflight_pooled_ladder_geography`."""

    if ladder.puma is None:
        raise ValueError(
            "location_rule='block_v1' on the ACS pool needs a block ladder that "
            "carries the per-block 'puma' array."
        )
    has_block = np.zeros(0, dtype=bool)
    if len(base_household):
        has_block, _ = _donor_block_index(base_household, ladder)
        if (~has_block).any():
            _require_ladder_states(base_household.loc[~has_block, "state_fips"], ladder)
    if len(acs_household):
        if "puma" not in acs_household.columns:
            raise ValueError(
                "ACS household table must carry canonical seven-digit puma "
                "geoids before block location."
            )
        _acs_block_pumas(acs_household["puma"], ladder)
    return _block_donor_mode(has_block)


def _assign_pooled_block_location(
    tables: dict[str, pd.DataFrame],
    ladder: UsLocationLadder,
    *,
    seed: int,
    expected_congressional_district_vintage: str | None,
) -> dict[str, Any]:
    """Locate every pooled household on one 2020 block (``block_v1``).

    - ACS rows (``household_spine == "acs_2024_1yr"``) draw one block within
      their observed 2020 PUMA, proportional to block population
      (:func:`~microcosm.build.us_runtime.block_location.draw_us_block_locations`,
      keyed by the pooled ``household_id``).
    - ASEC-by-PUF donor rows that already carry a ``block_geoid`` (the base
      H5 line assigned it, under either rule) keep that block: it is verified
      to exist in the ladder and to lie in the row's state, and is never
      redrawn.
    - Donor rows without a block draw one within their state; the count is
      recorded under :data:`DONOR_BLOCK_STATE_DRAWN`.

    Every geography column (:func:`location_geography_columns`: block,
    tract, county, place, SLDU/SLDL, CBSA, PUMA, primary CD and any attached
    plan) is then written from the ladder's lookup of the block, for every
    row, so no household carries a geography its block contradicts.
    ``state_fips`` is never rewritten. Returns the manifest record.
    """

    household = tables["household"]
    tag = spine_column("household")
    if tag not in household:
        raise ValueError(f"Pooled household table lacks required spine tag {tag!r}.")
    if "puma" not in household:
        raise ValueError(
            "ACS household table must carry canonical seven-digit puma geoids "
            "before block location."
        )
    if expected_congressional_district_vintage is not None:
        plan = ladder.primary_congressional_district_plan
        if plan != expected_congressional_district_vintage:
            raise ValueError(
                f"US block ladder congressional-district vintage {plan!r} does "
                "not match the vintage the caller expects "
                f"({expected_congressional_district_vintage!r})."
            )
    n = len(household)
    acs_rows = household[tag].eq(ACS_2024_1YR_SPINE).to_numpy()
    donor_rows = ~acs_rows
    household_key = pd.to_numeric(household["household_id"]).to_numpy(np.int64)
    state = pd.to_numeric(household["state_fips"]).to_numpy(np.int64)

    block_index = np.full(n, -1, dtype=np.int64)
    source_geography = np.full(n, DONOR_BLOCK_PRESERVED, dtype=object)
    donor_has_block, donor_index = _donor_block_index(household.loc[donor_rows], ladder)
    preserved = np.zeros(n, dtype=bool)
    preserved[np.flatnonzero(donor_rows)[donor_has_block]] = True
    block_index[preserved] = donor_index[donor_has_block]

    drawn = ~preserved
    puma = np.zeros(n, dtype=np.int64)
    if acs_rows.any():
        puma[acs_rows] = _acs_block_pumas(household.loc[acs_rows, "puma"], ladder)
    if (drawn & donor_rows).any():
        _require_ladder_states(household.loc[drawn & donor_rows, "state_fips"], ladder)
    if drawn.any():
        draw = draw_us_block_locations(
            ladder,
            household_key=household_key[drawn],
            state_fips=state[drawn],
            seed=seed,
            clones=1,
            puma=puma[drawn],
        )
        block_index[drawn] = draw.block_index
        source_geography[drawn] = draw.source_geography
    if (block_index < 0).any():
        raise AssertionError("internal error: a pooled household received no block.")
    if ((ladder.block_geoid[block_index] // 10**13) != state).any():
        raise AssertionError("internal error: a pooled block left its state.")

    derived = derive_us_block_geography(ladder, block_index)
    columns = location_geography_columns(ladder)
    changed = _rederived_donor_changes(household, derived, preserved, columns)
    for column in columns:
        household[column] = derived[column]
    if acs_rows.any():
        drawn_puma = pd.to_numeric(household.loc[acs_rows, "puma"]).to_numpy(np.int64)
        if not np.array_equal(drawn_puma, puma[acs_rows]):
            raise AssertionError("internal error: an ACS block left its PUMA.")
    tables["household"] = household

    record = us_block_location_manifest(
        pd.DataFrame(
            {
                LOCATION_CLONE_INDEX_COLUMN: np.zeros(n, dtype=np.int64),
                LOCATION_SOURCE_GEOGRAPHY_COLUMN: source_geography,
                LOCATION_CANDIDATE_BLOCKS_COLUMN: np.zeros(n, dtype=np.int64),
            }
        ),
        ladder,
        seed=seed,
        clones=1,
        candidate_set_rule={
            ACS_2024_1YR_SPINE: (
                f"{SOURCE_GEOGRAPHY_PUMA}: the blocks of the record's observed "
                "2020 PUMA (ACS PUMS ST+PUMA)"
            ),
            ASEC_PUF_SPINE: (
                f"{DONOR_BLOCK_PRESERVED}: the donor base's already-assigned "
                "block, kept (never redrawn) and re-derived through this "
                f"ladder; a donor row without a block: {SOURCE_GEOGRAPHY_STATE}"
            ),
        },
    )
    n_state_drawn = int((drawn & donor_rows).sum())
    record.update(
        {
            "household_key": "pooled household_id (after ACS id remapping)",
            "spine_households": {
                ACS_2024_1YR_SPINE: int(acs_rows.sum()),
                ASEC_PUF_SPINE: int(donor_rows.sum()),
            },
            "donor_blocks": {
                "mode": _block_donor_mode(donor_has_block),
                DONOR_BLOCK_PRESERVED: int(preserved.sum()),
                DONOR_BLOCK_STATE_DRAWN: n_state_drawn,
                "state_drawn_reason": (
                    "the donor household carries no block_geoid, so its finest "
                    "known geography is its state"
                    if n_state_drawn
                    else None
                ),
                "rederived_columns_changed": changed,
            },
        }
    )
    return record


def _rederived_donor_changes(
    household: pd.DataFrame,
    derived: dict[str, np.ndarray],
    preserved: np.ndarray,
    columns: tuple[str, ...],
) -> dict[str, int]:
    """Preserved donor rows whose stored, non-empty column differs from the block.

    Preserving a donor's block re-derives every other geography from this
    ladder. A nonzero count means the donor was located against a ladder that
    disagrees with this one on that column; it is recorded, not hidden. A
    stored value that is missing or blank is filled, not changed, and is not
    counted.
    """

    changes: dict[str, int] = {}
    if not preserved.any():
        return changes
    for column in columns:
        if column not in household.columns:
            continue
        want = derived[column][preserved]
        stored = pd.Series(household[column].to_numpy()[preserved], dtype=object)
        text = stored.where(stored.notna(), "").astype(str).str.strip()
        present = (text != "").to_numpy()
        if want.dtype.kind in "iu":
            got = pd.to_numeric(stored, errors="coerce").to_numpy(dtype=np.float64)
            differs = got != want.astype(np.float64)
        else:
            differs = text.to_numpy() != want.astype(str)
        count = int((differs & present).sum())
        if count:
            changes[column] = count
    return changes


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
