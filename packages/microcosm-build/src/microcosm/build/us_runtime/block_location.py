"""US household location, version 1: one 2020 census block per household.

The rule (Max, 27 September 2026; microcosm#696): for each household, draw one
2020 census tabulation block at random, proportional to block population,
within the finest geography its source provides; then look every other
geography up from that block. One rule and one code path for every US line.
Fidelity comes from calibrating weights afterwards, not from the draw.

A household's finest source geography, and so its candidate blocks, is:

- ``puma``: an ACS PUMS record's published 2020 PUMA → the blocks of that PUMA;
- ``county``: a CPS ASEC record that identifies its county (``GTCO`` nonzero)
  → the blocks of that county;
- ``state_unidentified_counties``: a CPS ASEC record with ``GTCO = 0`` → the
  blocks of its state outside the counties its ASEC vintage identifies.
  County identification is a countywide disclosure property (Max,
  2026-08-16), so such a household is known not to live in an identified
  county. Optionally narrowed to its CBSA
  (``state_unidentified_counties_cbsa``) when the source's CBSA delineation
  agrees with the ladder's (:func:`cbsa_vintage_agreement`);
- ``state``: every other record (PUF-lineage records without a CPS source,
  or any source that carries nothing finer).

Every other geography derives from the block through the ladder. Tract and
county are structural prefixes of the block geoid. Place, state legislative
districts, CBSA, PUMA and the district of every attached congressional
plan are block-level lookups. So one household's geographies can never
contradict each other, and a new district map is a re-lookup, not a
re-draw.

Draws are keyed, not sequential. A household clone's uniform is
``stable_identity_uniforms("{key}:{clone}", seed, salt=rule id)``, mapped
to a block by the exact inverse CDF of block population within its
candidate set (blocks in geoid order). A draw depends only on the seed, the
household key, the clone index and the candidate set. It does not depend
on row order, on other households, or on the order strata are visited,
which the legacy ``rng.choice`` paths cannot promise. Clone ``k`` of a
``K``-clone draw is the same block whatever ``K`` is.

With ``clones = K > 1`` every household is drawn ``K`` times, each clone
gets ``weight / K``, and weighted household and person mass is unchanged.
Calibration can then move weight between a household's clones, which in
effect chooses its location. With ``K = 1`` a household can only be
rescaled where it landed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.us_runtime.congressional_district_geography import (
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
)
from microcosm.build.us_runtime.geography_ladder import (
    UsBlockLadder,
    load_us_block_ladder,
)
from microcosm.frame import Frame, Weights

#: The rule id every build records when it locates households this way; also
#: the salt of the keyed uniforms.
US_BLOCK_LOCATION_RULE_ID = "us_block_location.population_draw.v1"

#: ``--location-rule`` values. ``legacy`` keeps each line's historical rule
#: (and its bytes); ``block_v1`` is this module.
US_LOCATION_RULE_LEGACY = "legacy"
US_LOCATION_RULE_BLOCK_V1 = "block_v1"
US_LOCATION_RULE_CHOICES = (US_LOCATION_RULE_LEGACY, US_LOCATION_RULE_BLOCK_V1)

#: Source-geography kinds, finest first.
SOURCE_GEOGRAPHY_PUMA = "puma"
SOURCE_GEOGRAPHY_COUNTY = "county"
SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA = "state_unidentified_counties_cbsa"
SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES = "state_unidentified_counties"
SOURCE_GEOGRAPHY_STATE = "state"
SOURCE_GEOGRAPHY_KINDS = (
    SOURCE_GEOGRAPHY_PUMA,
    SOURCE_GEOGRAPHY_COUNTY,
    SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA,
    SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES,
    SOURCE_GEOGRAPHY_STATE,
)

#: Location-table bookkeeping columns.
LOCATION_CLONE_INDEX_COLUMN = "location_clone_index"
LOCATION_CLONE_ID_COLUMN = "location_clone_id"
LOCATION_WEIGHT_COLUMN = "location_weight"
LOCATION_SOURCE_GEOGRAPHY_COLUMN = "location_source_geography"
LOCATION_CANDIDATE_BLOCKS_COLUMN = "location_candidate_blocks"
#: Geography columns every draw writes (``puma`` only from a ladder that
#: carries it). Each attached non-primary CD plan adds
#: ``congressional_district_geoid__<plan>``.
US_BLOCK_LOCATION_GEOGRAPHY_COLUMNS = (
    "block_geoid",
    "tract_geoid",
    "county_fips",
    "place_fips",
    "sldu",
    "sldl",
    "cbsa_code",
    "puma",
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
)
#: Household-column prefix of every non-primary CD plan attached to a ladder.
US_CD_PLAN_COLUMN_PREFIX = "congressional_district_geoid__"
#: Ladder array key of the per-block 2020 PUMA. It is additive to block-ladder
#: schema 1: the schema-1 loader ignores arrays it does not require, so a
#: legacy build reads a PUMA-carrying ladder unchanged.
US_BLOCK_LADDER_PUMA_ARRAY = "puma"
#: Frame metadata key of the location receipt.
US_BLOCK_LOCATION_METADATA_KEY = "us_block_location"
#: H5 root attributes every line writes on a ``block_v1`` artifact (never on a
#: legacy one): the rule, its seed and clone count as text, and the block
#: ladder's digest and layer vintages.
POPULACE_LOCATION_RULE_ATTR = "populace_location_rule"
POPULACE_LOCATION_SEED_ATTR = "populace_location_seed"
POPULACE_LOCATION_CLONES_ATTR = "populace_location_clones"
POPULACE_BLOCK_LADDER_SHA256_ATTR = "populace_block_ladder_sha256"
POPULACE_BLOCK_LADDER_VINTAGES_ATTR = "populace_block_ladder_vintages"
#: Packaged official CPS ASEC identified-county lists (Census technical
#: documentation, Appendix F, List 4), keyed by ASEC survey year.
CPS_ASEC_IDENTIFIED_COUNTIES_RESOURCE = "cps_asec_identified_counties.csv"

_BLOCK_STATE_DIVISOR = 10**13
_BLOCK_COUNTY_DIVISOR = 10**10
_PUMA_STATE_DIVISOR = 10**5
_CBSA_KEY_MULTIPLIER = 100_000
_EXCLUDED_KEY = -1


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UsLocationLadder:
    """The block ladder plus what the location draw derives beyond it.

    Attributes:
        blocks: The validated schema-1 block ladder: block geoid (sorted),
            population, primary-plan CD, SLDU/SLDL, place, CBSA.
        puma: 2020 PUMA per block (``state_fips * 10**5 + PUMA5CE``), or
            ``None`` for an artifact without the PUMA array. A PUMA-
            constrained draw refuses such a ladder.
        congressional_district_plans: CD geoid array per plan id, aligned to
            the blocks. Always carries the primary plan (the ladder's
            ``congressional_district`` layer); more are attached from a
            block → CD-plan registry by :func:`attach_congressional_district_plans`.
        congressional_district_plan_sources: Vintage, source and any known
            deviations per plan id.
        sha256: SHA-256 of the artifact file, when loaded from one.
    """

    blocks: UsBlockLadder
    puma: np.ndarray | None = None
    congressional_district_plans: Mapping[str, np.ndarray] = field(default_factory=dict)
    congressional_district_plan_sources: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    sha256: str | None = None

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def block_geoid(self) -> np.ndarray:
        return self.blocks.block_geoid

    @property
    def population(self) -> np.ndarray:
        return self.blocks.population

    @property
    def cbsa_code(self) -> np.ndarray:
        return self.blocks.cbsa_code

    @property
    def primary_congressional_district_plan(self) -> str:
        """The plan id the ``congressional_district_geoid`` column carries."""
        return self.blocks.layer_vintages["congressional_district"]

    @property
    def layer_vintages(self) -> dict[str, str]:
        vintages = dict(self.blocks.layer_vintages)
        if self.puma is not None:
            vintages["puma"] = str(self.blocks.metadata["layers"]["puma"]["vintage"])
        return vintages


def us_location_ladder(
    blocks: UsBlockLadder,
    *,
    puma: np.ndarray | None = None,
    sha256: str | None = None,
) -> UsLocationLadder:
    """Wrap a validated block ladder for location draws.

    Block geoids must be sorted ascending (the artifact builder writes them
    so), because block lookups are binary searches. An optional per-block
    ``puma`` is validated: one positive 2020 PUMA per block, in the block's
    state, constant within each tract, with a recorded ``puma`` layer vintage
    and source.
    """

    if (np.diff(blocks.block_geoid) <= 0).any():
        raise ValueError("block ladder block_geoid values must be sorted ascending.")
    validated_puma = None
    if puma is not None:
        layers = blocks.metadata.get("layers", {})
        puma_layer = layers.get("puma") if isinstance(layers, Mapping) else None
        if not isinstance(puma_layer, Mapping) or not all(
            str(puma_layer.get(key) or "") for key in ("vintage", "source")
        ):
            raise ValueError(
                "a block ladder carrying 'puma' must record a 'puma' layer with "
                "a non-empty vintage and source (vintage_policy: error)."
            )
        validated_puma = _validated_puma_array(puma, blocks.block_geoid)
    primary = blocks.layer_vintages["congressional_district"]
    return UsLocationLadder(
        blocks=blocks,
        puma=validated_puma,
        congressional_district_plans={primary: blocks.congressional_district_geoid},
        congressional_district_plan_sources={
            primary: {
                **dict(blocks.metadata["layers"]["congressional_district"]),
                "primary": True,
            }
        },
        sha256=sha256,
    )


def load_us_location_ladder(path: str | Path) -> UsLocationLadder:
    """Load a block ladder and its optional per-block PUMA for location draws.

    The core arrays go through :func:`load_us_block_ladder` unchanged; the
    additive ``puma`` array is validated by :func:`us_location_ladder`.
    """

    source = Path(path)
    blocks = load_us_block_ladder(source)
    with np.load(source, allow_pickle=False) as payload:
        puma = (
            np.asarray(payload[US_BLOCK_LADDER_PUMA_ARRAY])
            if US_BLOCK_LADDER_PUMA_ARRAY in payload.files
            else None
        )
    return us_location_ladder(blocks, puma=puma, sha256=file_sha256(source))


def attach_congressional_district_plans(
    ladder: UsLocationLadder,
    *,
    block_geoid: np.ndarray,
    plans: Mapping[str, np.ndarray],
    plan_sources: Mapping[str, Mapping[str, Any]],
) -> UsLocationLadder:
    """Return ``ladder`` with more congressional-district plans attached.

    ``block_geoid`` and each ``plans[plan]`` array come from a block → CD-plan
    registry: one district per 2020 block per plan, in the
    ``state_fips * 100 + district`` convention. The registry may cover more
    blocks than the ladder but must cover every ladder block, and each
    district must lie in its block's state. A plan the ladder already carries
    must agree with it block for block: the same map read from two artifacts
    is a differential check, never a silent override. ``plan_sources`` must
    give each plan a non-empty ``vintage`` and ``source``; any other keys
    (source hashes, known deviations) are carried into the manifest verbatim.
    """

    registry_blocks = _int_values(block_geoid, label="registry block_geoid")
    if len(registry_blocks) == 0:
        raise ValueError("CD-plan registry has no blocks.")
    if len(np.unique(registry_blocks)) != len(registry_blocks):
        raise ValueError("CD-plan registry block_geoid values must be unique.")
    order = np.argsort(registry_blocks, kind="stable")
    sorted_blocks = registry_blocks[order]
    position = np.clip(
        np.searchsorted(sorted_blocks, ladder.block_geoid), 0, len(sorted_blocks) - 1
    )
    uncovered = sorted_blocks[position] != ladder.block_geoid
    if uncovered.any():
        examples = [f"{value:015d}" for value in ladder.block_geoid[uncovered][:5]]
        raise ValueError(
            f"CD-plan registry omits {int(uncovered.sum())} ladder block(s); "
            f"examples: {examples}."
        )
    aligned = order[position]
    merged = dict(ladder.congressional_district_plans)
    sources = {
        plan: dict(spec)
        for plan, spec in ladder.congressional_district_plan_sources.items()
    }
    for plan, values in plans.items():
        spec = plan_sources.get(plan)
        if not isinstance(spec, Mapping) or not all(
            str(spec.get(key) or "") for key in ("vintage", "source")
        ):
            raise ValueError(
                f"CD plan {plan!r} must record a non-empty vintage and source."
            )
        raw = np.asarray(values)
        if len(raw) != len(registry_blocks):
            raise ValueError(
                f"CD plan {plan!r} has {len(raw)} values for "
                f"{len(registry_blocks)} registry blocks."
            )
        districts = _validated_cd_array(
            raw[aligned], ladder.block_geoid, label=f"CD plan {plan!r}"
        )
        if plan in merged:
            differs = merged[plan] != districts
            if differs.any():
                raise ValueError(
                    f"CD plan {plan!r} disagrees with the ladder's own copy on "
                    f"{int(differs.sum())} block(s)."
                )
            sources[plan]["registry"] = dict(spec)
            continue
        merged[str(plan)] = districts
        sources[str(plan)] = {**dict(spec), "primary": False}
    return replace(
        ladder,
        congressional_district_plans=merged,
        congressional_district_plan_sources=sources,
    )


def plan_column(plan: str, ladder: UsLocationLadder) -> str:
    """Household column an attached CD plan is written to."""

    if plan not in ladder.congressional_district_plans:
        raise KeyError(f"CD plan {plan!r} is not attached to the ladder.")
    if plan == ladder.primary_congressional_district_plan:
        return CONGRESSIONAL_DISTRICT_GEOID_COLUMN
    return f"{US_CD_PLAN_COLUMN_PREFIX}{plan}"


def location_geography_columns(ladder: UsLocationLadder) -> tuple[str, ...]:
    """Every geography column a draw against ``ladder`` writes, in order."""

    base = tuple(
        column
        for column in US_BLOCK_LOCATION_GEOGRAPHY_COLUMNS
        if column != "puma" or ladder.puma is not None
    )
    extra = tuple(
        plan_column(plan, ladder)
        for plan in sorted(ladder.congressional_district_plans)
        if plan != ladder.primary_congressional_district_plan
    )
    return (*base, *extra)


# --------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------


def location_uniforms(
    seed: int, household_key: np.ndarray, clone_index: np.ndarray
) -> np.ndarray:
    """Keyed uniforms in ``[0, 1]``, one per ``(household key, clone)`` pair.

    ``blake2b("{seed}:{rule id}:{key}:{clone}")`` through the repository's
    :func:`~microcosm.build.stochastic_assignment.stable_identity_uniforms`:
    a pure function of its arguments, so a household's draw never depends on
    where it sits in the table or on which other households are present.
    """

    keys = np.asarray(household_key, dtype=np.int64)
    clones = np.asarray(clone_index, dtype=np.int64)
    if keys.shape != clones.shape:
        raise ValueError("household_key and clone_index must align.")
    if int(seed) < 0:
        raise ValueError(f"location seed must be non-negative, got {seed!r}.")
    return stable_identity_uniforms(
        [
            f"{key}:{clone}"
            for key, clone in zip(keys.tolist(), clones.tolist(), strict=True)
        ],
        seed=int(seed),
        salt=US_BLOCK_LOCATION_RULE_ID,
    )


class _BlockStrata:
    """Blocks grouped by an ``int64`` stratum key, each stratum contiguous.

    Blocks keyed ``-1`` belong to no stratum a household may draw from. Within
    a stratum blocks are in geoid order, and a uniform ``u`` selects the block
    whose cumulative-population interval ``[C_{i-1}, C_i)`` holds
    ``C_start + u * P`` (``P`` the stratum population): the exact inverse
    CDF, so ``P(block i) = population_i / P``.
    """

    def __init__(
        self, block_keys: np.ndarray, block_geoid: np.ndarray, population: np.ndarray
    ) -> None:
        self.order = np.lexsort((block_geoid, block_keys))
        self.keys = block_keys[self.order]
        self.population = population[self.order]
        self.cumulative = np.cumsum(self.population)

    def bounds(self, household_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.searchsorted(self.keys, household_keys, side="left")
        hi = np.searchsorted(self.keys, household_keys, side="right")
        return lo, hi

    def draw(self, household_keys: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds(household_keys)
        if (hi <= lo).any() or (household_keys == _EXCLUDED_KEY).any():
            raise AssertionError("internal error: draw on an empty stratum.")
        start = self.cumulative[lo] - self.population[lo]
        total = self.cumulative[hi - 1] - start
        target = start + uniforms * total
        position = np.searchsorted(self.cumulative, target, side="right")
        return self.order[np.clip(position, lo, hi - 1)]


@dataclass(frozen=True)
class UsBlockLocationDraw:
    """One block per ``(household row, clone)``.

    Attributes:
        row: Input row of each output row (``np.repeat`` of the rows, so a
            household's clones are adjacent, clone 0 first).
        clone_index: Clone ``0..K-1`` of each output row.
        block_index: Index into the ladder arrays of each drawn block.
        source_geography: Source-geography kind of each output row.
        candidate_blocks: Size of each output row's candidate set.
        seed: The seed the uniforms were keyed with.
        clones: ``K``.
    """

    row: np.ndarray
    clone_index: np.ndarray
    block_index: np.ndarray
    source_geography: np.ndarray
    candidate_blocks: np.ndarray
    seed: int
    clones: int


def draw_us_block_locations(
    ladder: UsLocationLadder,
    *,
    household_key: np.ndarray,
    state_fips: np.ndarray,
    seed: int,
    clones: int = 1,
    puma: np.ndarray | None = None,
    county_fips: np.ndarray | None = None,
    identification_group: np.ndarray | None = None,
    identified_counties: Mapping[int, Collection[int]] | None = None,
    cbsa_code: np.ndarray | None = None,
    unidentified_county_share: Mapping[int, Mapping[int, float]] | None = None,
) -> UsBlockLocationDraw:
    """Draw one block per household (per clone) within its source geography.

    Args:
        ladder: The location ladder. PUMA-constrained households need one
            that carries a per-block ``puma``.
        household_key: Unique non-negative integer key per household (its
            stable id). Draws are keyed by it, so keys must be unique.
        state_fips: State FIPS per household; every candidate block lies in it.
        seed: Location seed, recorded in the build manifest.
        clones: ``K >= 1`` independent draws per household.
        puma: 2020 PUMA per household (``state * 10**5 + PUMA5CE``); ``<= 0``
            where the source publishes none.
        county_fips: 5-digit county FIPS per household as an integer; ``<= 0``
            where the source identifies none.
        identification_group: For households with neither PUMA nor county,
            the group (CPS ASEC vintage) whose identified-county list their
            county was suppressed against; ``< 0`` means none, so the draw
            is statewide.
        identified_counties: Identified (excluded) county FIPS per group;
            required when any household has an identification group.
        cbsa_code: Optional CBSA per household, used only by
            ``state_unidentified_counties`` households with a positive code.
            Pass it only when :func:`cbsa_vintage_agreement` holds.
        unidentified_county_share: Optional per-group ``{county: share}``
            multiplying block population in that group's unidentified-county
            pool (default 1; a share of 0 removes the county). This is the
            opt-in correction for partially coded counties
            (:func:`cps_unidentified_county_shares`); the ruled default is
            plain block population.

    Raises:
        ValueError: On inconsistent inputs, or when any household's candidate
            set is empty. The draw never falls back silently to a coarser
            geography.
    """

    clones = int(clones)
    if clones < 1:
        raise ValueError(f"clones must be >= 1, got {clones}.")
    keys = _int_values(household_key, label="household_key")
    n = len(keys)
    if n == 0:
        raise ValueError("draw_us_block_locations needs at least one household.")
    if (keys < 0).any():
        raise ValueError("household_key values must be non-negative.")
    if len(np.unique(keys)) != n:
        raise ValueError(
            "household_key values must be unique: draws are keyed by them."
        )
    if int(keys.max()) > (np.iinfo(np.int64).max - clones) // clones:
        raise ValueError("household_key too large to derive clone ids.")
    state = _aligned(state_fips, n, label="state_fips")
    if ((state < 1) | (state > 99)).any():
        raise ValueError("state_fips must be two-digit state codes.")
    puma_values = _optional_aligned(puma, n, label="puma")
    county_values = _optional_aligned(county_fips, n, label="county_fips")
    group_values = _optional_aligned(
        identification_group, n, label="identification_group", missing=-1
    )
    cbsa_values = _optional_aligned(cbsa_code, n, label="cbsa_code")

    has_puma = puma_values > 0
    has_county = county_values > 0
    if (has_puma & has_county).any():
        raise ValueError(
            "a household carries at most one of puma and county_fips; "
            f"{int((has_puma & has_county).sum())} carry both."
        )
    has_group = ~has_puma & ~has_county & (group_values >= 0)
    if has_group.any() and identified_counties is None:
        raise ValueError(
            "identification_group is set but identified_counties is missing."
        )
    if has_puma.any() and ladder.puma is None:
        raise ValueError(
            "PUMA-constrained households need a block ladder that carries a "
            "per-block 'puma' array; this ladder has none."
        )
    _require_same_state(state, puma_values // _PUMA_STATE_DIVISOR, has_puma, "puma")
    _require_same_state(state, county_values // 1000, has_county, "county_fips")

    block_state = ladder.block_geoid // _BLOCK_STATE_DIVISOR
    block_county = ladder.block_geoid // _BLOCK_COUNTY_DIVISOR
    kind = np.full(n, SOURCE_GEOGRAPHY_STATE, dtype=object)
    household_stratum = state.copy()
    population = np.asarray(ladder.population, dtype=np.float64)
    views: list[tuple[np.ndarray, str, np.ndarray, np.ndarray]] = []

    if has_puma.any():
        assert ladder.puma is not None
        kind[has_puma] = SOURCE_GEOGRAPHY_PUMA
        household_stratum[has_puma] = puma_values[has_puma]
        views.append((has_puma, SOURCE_GEOGRAPHY_PUMA, ladder.puma, population))
    if has_county.any():
        kind[has_county] = SOURCE_GEOGRAPHY_COUNTY
        household_stratum[has_county] = county_values[has_county]
        views.append((has_county, SOURCE_GEOGRAPHY_COUNTY, block_county, population))
    if has_group.any():
        assert identified_counties is not None
        for group in sorted({int(value) for value in group_values[has_group]}):
            if group not in identified_counties:
                raise ValueError(
                    "identified_counties has no entry for identification group "
                    f"{group}."
                )
            identified = np.asarray(
                sorted(int(value) for value in identified_counties[group]),
                dtype=np.int64,
            )
            excluded = np.isin(block_county, identified)
            group_population = _county_scaled_population(
                ladder.population,
                block_county,
                (unidentified_county_share or {}).get(group, {}),
            )
            excluded = excluded | (group_population <= 0)
            in_group = has_group & (group_values == group)
            with_cbsa = in_group & (cbsa_values > 0)
            without_cbsa = in_group & ~with_cbsa
            if without_cbsa.any():
                kind[without_cbsa] = SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES
                views.append(
                    (
                        without_cbsa,
                        f"{SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES}[{group}]",
                        np.where(excluded, _EXCLUDED_KEY, block_state),
                        group_population,
                    )
                )
            if with_cbsa.any():
                kind[with_cbsa] = SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA
                household_stratum[with_cbsa] = (
                    state[with_cbsa] * _CBSA_KEY_MULTIPLIER + cbsa_values[with_cbsa]
                )
                cbsa = ladder.cbsa_code.astype(np.int64)
                views.append(
                    (
                        with_cbsa,
                        f"{SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA}[{group}]",
                        np.where(
                            excluded | (cbsa <= 0),
                            _EXCLUDED_KEY,
                            block_state * _CBSA_KEY_MULTIPLIER + cbsa,
                        ),
                        group_population,
                    )
                )
    plain_state = kind == SOURCE_GEOGRAPHY_STATE
    if plain_state.any():
        views.append((plain_state, SOURCE_GEOGRAPHY_STATE, block_state, population))

    row = np.repeat(np.arange(n, dtype=np.int64), clones)
    clone_index = np.tile(np.arange(clones, dtype=np.int64), n)
    uniforms = location_uniforms(seed, keys[row], clone_index)
    block_index = np.full(n * clones, -1, dtype=np.int64)
    candidates = np.zeros(n * clones, dtype=np.int64)
    for mask, label, block_keys, view_population in views:
        strata = _BlockStrata(
            np.asarray(block_keys, dtype=np.int64), ladder.block_geoid, view_population
        )
        rows = np.flatnonzero(mask)
        lo, hi = strata.bounds(household_stratum[rows])
        empty = hi <= lo
        if empty.any():
            examples = [
                f"household {int(keys[r])} (state {int(state[r]):02d}, "
                f"stratum {int(household_stratum[r])})"
                for r in rows[empty][:5].tolist()
            ]
            raise ValueError(
                f"{int(empty.sum())} household(s) have no candidate block for "
                f"source geography {label}: {examples}. The household geography "
                "and the block ladder must share a vintage."
            )
        out = np.flatnonzero(mask[row])
        block_index[out] = strata.draw(household_stratum[row[out]], uniforms[out])
        candidates[out] = np.repeat((hi - lo).astype(np.int64), clones)
    if (block_index < 0).any():
        raise AssertionError("internal error: some rows received no block.")
    if (block_state[block_index] != state[row]).any():
        raise AssertionError("internal error: a drawn block left its state.")
    return UsBlockLocationDraw(
        row=row,
        clone_index=clone_index,
        block_index=block_index,
        source_geography=kind[row],
        candidate_blocks=candidates,
        seed=int(seed),
        clones=clones,
    )


def derive_us_block_geography(
    ladder: UsLocationLadder, block_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Every geography the block determines, as household-column arrays.

    Tract and county are prefixes of the block geoid; the rest are the
    ladder's values at the block. String formats match the legacy ladders:
    ``""`` for no place, CBSA or SLD, zero-padded text codes, 7-character
    PUMA, integer ``state * 100 + district`` CDs. So downstream readers and
    gates see the same shapes.
    """

    index = np.asarray(block_index, dtype=np.int64)
    blocks = ladder.blocks
    block_str = np.array(
        [f"{value:015d}" for value in blocks.block_geoid[index].tolist()]
    )
    columns: dict[str, np.ndarray] = {
        "block_geoid": block_str,
        "tract_geoid": np.array([value[:11] for value in block_str.tolist()]),
        "county_fips": np.array([value[:5] for value in block_str.tolist()]),
        "place_fips": _fips_strings(blocks.place_fips[index], width=5),
        "sldu": blocks.sldu[index].astype(object),
        "sldl": blocks.sldl[index].astype(object),
        "cbsa_code": _fips_strings(blocks.cbsa_code[index], width=5),
    }
    if ladder.puma is not None:
        columns["puma"] = np.array(
            [f"{value:07d}" for value in ladder.puma[index].tolist()], dtype=object
        )
    for plan in sorted(ladder.congressional_district_plans):
        columns[plan_column(plan, ladder)] = np.asarray(
            ladder.congressional_district_plans[plan][index], dtype=np.int64
        )
    return columns


def assign_us_block_location(
    household: pd.DataFrame,
    ladder: UsLocationLadder,
    *,
    seed: int,
    clones: int = 1,
    weights: np.ndarray | None = None,
    key_column: str = "household_id",
    state_fips_column: str = "state_fips",
    puma_column: str | None = None,
    county_column: str | None = None,
    identification_group_column: str | None = None,
    identified_counties: Mapping[int, Collection[int]] | None = None,
    cbsa_column: str | None = None,
    unidentified_county_share: Mapping[int, Mapping[int, float]] | None = None,
) -> pd.DataFrame:
    """The location table: one row per household, per clone when ``K > 1``.

    This is the one assignment function every US line calls. Columns:
    ``key_column``; :data:`LOCATION_CLONE_INDEX_COLUMN`; the stable
    :data:`LOCATION_CLONE_ID_COLUMN` (``key * K + clone``);
    :data:`LOCATION_WEIGHT_COLUMN` (``weight / K``, when ``weights`` is
    given); :data:`LOCATION_SOURCE_GEOGRAPHY_COLUMN`;
    :data:`LOCATION_CANDIDATE_BLOCKS_COLUMN`; ``state_fips``; and every
    :func:`location_geography_columns` column. Rows follow the input order,
    with each household's clones adjacent.
    """

    for column in (key_column, state_fips_column):
        if column not in household.columns:
            raise ValueError(f"household table must contain {column!r}.")
    keys = _int_values(household[key_column], label=key_column)
    draw = draw_us_block_locations(
        ladder,
        household_key=keys,
        state_fips=_int_values(household[state_fips_column], label=state_fips_column),
        seed=seed,
        clones=clones,
        puma=_optional_column(household, puma_column),
        county_fips=_optional_column(household, county_column),
        identification_group=_optional_column(
            household, identification_group_column, missing=-1
        ),
        identified_counties=identified_counties,
        cbsa_code=_optional_column(household, cbsa_column),
        unidentified_county_share=unidentified_county_share,
    )
    return location_table(
        draw, ladder, household_key=keys, weights=weights, key_column=key_column
    )


def location_table(
    draw: UsBlockLocationDraw,
    ladder: UsLocationLadder,
    *,
    household_key: np.ndarray,
    weights: np.ndarray | None = None,
    key_column: str = "household_id",
) -> pd.DataFrame:
    """Materialize a draw as the location table (:func:`assign_us_block_location`)."""

    keys = np.asarray(household_key, dtype=np.int64)[draw.row]
    table: dict[str, Any] = {
        key_column: keys,
        LOCATION_CLONE_INDEX_COLUMN: draw.clone_index,
        LOCATION_CLONE_ID_COLUMN: keys * draw.clones + draw.clone_index,
    }
    if weights is not None:
        values = np.asarray(weights, dtype=np.float64)
        if len(values) * draw.clones != len(draw.row):
            raise ValueError("weights must align with household rows.")
        table[LOCATION_WEIGHT_COLUMN] = values[draw.row] / draw.clones
    table[LOCATION_SOURCE_GEOGRAPHY_COLUMN] = draw.source_geography
    table[LOCATION_CANDIDATE_BLOCKS_COLUMN] = draw.candidate_blocks
    table["state_fips"] = ladder.block_geoid[draw.block_index] // _BLOCK_STATE_DIVISOR
    table.update(derive_us_block_geography(ladder, draw.block_index))
    return pd.DataFrame(table)


# --------------------------------------------------------------------------
# CPS ASEC county identification
# --------------------------------------------------------------------------


def identified_counties_by_group(
    state_fips: np.ndarray,
    county_code: np.ndarray,
    group: np.ndarray,
) -> dict[int, frozenset[int]]:
    """Counties each group's records code: ``{group: {state * 1000 + county}}``.

    For CPS ASEC, ``county_code`` is ``GTCO`` (the 3-digit county within the
    state, ``0`` when suppressed) and ``group`` the source vintage. Each
    vintage keeps its own set, because a county's identification status can
    change between years.
    """

    state = _int_values(state_fips, label="state_fips")
    county = _int_values(county_code, label="county_code")
    groups = _int_values(group, label="group")
    if not (len(state) == len(county) == len(groups)):
        raise ValueError("state_fips, county_code and group must align.")
    if ((county < 0) | (county > 999)).any():
        raise ValueError("county_code must be a 3-digit county code (0 = none).")
    identified = county > 0
    result: dict[int, frozenset[int]] = {}
    for value in sorted({int(g) for g in groups.tolist()}):
        mask = identified & (groups == value)
        result[value] = frozenset(
            int(code) for code in np.unique(state[mask] * 1000 + county[mask])
        )
    return result


@dataclass(frozen=True)
class CpsIdentifiedCountyList:
    """Census's official identified-county list for one CPS ASEC year.

    Attributes:
        asec_year: ASEC survey year (the ``cpsmar{yy}`` documentation year).
        counties: 5-digit county FIPS as integers.
        source: URL of the technical documentation it was read from.
        source_sha256: SHA-256 of that PDF.
        entire_county_guarantee: Whether that year's List 4 preamble says
            "Counties are only included on this list if the entire county is
            identified" (ASEC 2023-2025 do; the 2026 documentation does not).
    """

    asec_year: int
    counties: frozenset[int]
    source: str
    source_sha256: str
    entire_county_guarantee: bool = True


def default_cps_asec_identified_counties_path() -> Path:
    """Path of the packaged official identified-county CSV."""

    return Path(
        str(
            files("microcosm.build.us_runtime.data").joinpath(
                CPS_ASEC_IDENTIFIED_COUNTIES_RESOURCE
            )
        )
    )


def load_cps_asec_identified_counties(
    path: str | Path | None = None,
) -> dict[int, CpsIdentifiedCountyList]:
    """Load the official CPS ASEC identified-county lists, keyed by ASEC year.

    The default is the packaged CSV built by
    ``tools/build_us_cps_identified_counties.py`` from List 4 ("FIPS County
    Codes") of each year's ASEC technical documentation. ASEC 2023-2025 say
    "Counties are only included on this list if the entire county is
    identified"; the ``entire_county_guarantee`` column records, per year,
    whether the preamble says so (a CSV without the column means it does).
    """

    source = (
        Path(path) if path is not None else default_cps_asec_identified_counties_path()
    )
    provenance = source.with_name(source.name + ".provenance.json")
    if provenance.is_file():
        expected = json.loads(provenance.read_text()).get("csv_sha256")
        actual = file_sha256(source)
        if expected != actual:
            raise ValueError(
                f"{source.name} has SHA-256 {actual}, but its provenance records "
                f"{expected}; rebuild both with "
                "tools/build_us_cps_identified_counties.py."
            )
    elif path is None:
        raise FileNotFoundError(f"packaged {source.name} lacks {provenance.name}.")
    rows: dict[int, dict[str, Any]] = {}
    with source.open(newline="") as stream:
        for record in csv.DictReader(stream):
            year = int(record["asec_year"])
            entry = rows.setdefault(
                year,
                {
                    "counties": set(),
                    "source": record["source_url"],
                    "source_sha256": record["source_sha256"],
                    "guarantee": record.get("entire_county_guarantee", "true"),
                },
            )
            if (
                entry["source"] != record["source_url"]
                or entry["source_sha256"] != record["source_sha256"]
                or entry["guarantee"] != record.get("entire_county_guarantee", "true")
            ):
                raise ValueError(f"ASEC {year} rows disagree on their source.")
            if entry["guarantee"] not in ("true", "false"):
                raise ValueError(
                    f"ASEC {year} entire_county_guarantee must be true or false."
                )
            county = int(record["state_fips"]) * 1000 + int(record["county_code"])
            if county in entry["counties"]:
                raise ValueError(f"ASEC {year} lists county {county:05d} twice.")
            entry["counties"].add(county)
    if not rows:
        raise ValueError(f"no identified-county rows in {source}.")
    return {
        year: CpsIdentifiedCountyList(
            asec_year=year,
            counties=frozenset(entry["counties"]),
            source=str(entry["source"]),
            source_sha256=str(entry["source_sha256"]),
            entire_county_guarantee=entry["guarantee"] == "true",
        )
        for year, entry in sorted(rows.items())
    }


def cps_excluded_county_sets(
    data_identified: Mapping[int, Collection[int]],
    official: Mapping[int, CpsIdentifiedCountyList] | None,
    *,
    asec_year_of_group: Mapping[int, int],
) -> tuple[dict[int, frozenset[int]], dict[str, Any]]:
    """Counties a ``GTCO = 0`` record is known not to live in, per group.

    A county is excluded only with two confirmations: Census lists it (List 4
    of that ASEC year), *and* the file actually codes it. Either source alone
    misleads:

    - the public files code some counties only in part (on ASEC 2023-2025,
      Tarrant TX, King WA and Snohomish WA carry codes for well under half
      their population), so a data-derived set would wrongly rule out the
      uncoded remainder;
    - List 4 names counties the files never code (e.g. Lubbock TX, New London
      CT), whose households therefore carry ``GTCO = 0``.

    Both confirmations prove a county only while List 4 promises that a
    listed county is identified in full. The 2026 documentation drops that
    sentence, and its file breaks the premise: every Delaware county is listed
    and coded for about its whole population, yet 49 Delaware households carry
    ``GTCO = 0``. For a year whose list lacks the guarantee, nothing is
    excluded: a ``GTCO = 0`` record draws from its whole state.

    Without an official list for a group's year, the data-derived set is the
    documented fallback (microcosm#696). Returns the per-group excluded sets
    and a diagnostics record for the manifest.
    """

    excluded: dict[int, frozenset[int]] = {}
    per_group: dict[str, Any] = {}
    for group in sorted(data_identified):
        coded = frozenset(int(value) for value in data_identified[group])
        asec_year = asec_year_of_group.get(group)
        listed = (
            None if official is None or asec_year is None else official.get(asec_year)
        )
        if listed is None:
            excluded[group] = coded
            per_group[str(group)] = {
                "asec_year": asec_year,
                "basis": "data_derived_fallback",
                "coded_counties": len(coded),
                "excluded_counties": len(coded),
            }
            continue
        both = coded & listed.counties
        if listed.entire_county_guarantee:
            basis = "official_list_and_coded"
            excluded[group] = frozenset(both)
        else:
            basis = "no_whole_county_guarantee_state_draw"
            excluded[group] = frozenset()
        per_group[str(group)] = {
            "asec_year": asec_year,
            "basis": basis,
            "official_source": listed.source,
            "official_source_sha256": listed.source_sha256,
            "entire_county_guarantee": listed.entire_county_guarantee,
            "official_counties": len(listed.counties),
            "coded_counties": len(coded),
            "listed_and_coded_counties": len(both),
            "excluded_counties": len(excluded[group]),
            "coded_not_listed": [
                f"{value:05d}" for value in sorted(coded - listed.counties)
            ],
            "listed_not_coded": [
                f"{value:05d}" for value in sorted(listed.counties - coded)
            ],
        }
    union = frozenset().union(*excluded.values()) if excluded else frozenset()
    varying = sorted(
        county
        for county in union
        if len({county in members for members in excluded.values()}) > 1
    )
    return excluded, {
        "rule": (
            "GTCO=0 records draw from their state's blocks outside counties "
            "that are both on Census's identified-county list for the ASEC "
            "year and coded in that year's file; a year whose list drops the "
            "whole-county guarantee excludes nothing (whole-state draw)"
        ),
        "groups": per_group,
        "counties_whose_exclusion_varies_across_groups": [
            f"{value:05d}" for value in varying
        ],
    }


def cps_county_coverage_ratios(
    ladder: UsLocationLadder,
    *,
    county_fips: np.ndarray,
    person_weight: np.ndarray,
    group: np.ndarray,
) -> dict[int, dict[int, float]]:
    """Coded share of each county's population, per group, median-normalized.

    For every county a group's records code, the weighted persons coded to it
    divided by its 2020 census population, rescaled so the group's median
    county is 1 (which absorbs the survey's weight scale). About 1 means the
    file codes the whole county; well below 1 means it codes only part.
    ``person_weight`` is the household weight times household size.
    """

    county = _int_values(county_fips, label="county_fips")
    weight = np.asarray(person_weight, dtype=np.float64)
    groups = _int_values(group, label="group")
    if not (len(county) == len(weight) == len(groups)):
        raise ValueError("county_fips, person_weight and group must align.")
    census = (
        pd.Series(np.asarray(ladder.population, dtype=np.float64))
        .groupby(ladder.block_geoid // _BLOCK_COUNTY_DIVISOR)
        .sum()
    )
    result: dict[int, dict[int, float]] = {}
    for value in sorted({int(g) for g in groups.tolist()}):
        mask = (groups == value) & (county > 0)
        coded = pd.Series(weight[mask]).groupby(county[mask]).sum()
        if coded.empty:
            result[value] = {}
            continue
        missing = sorted(set(coded.index) - set(census.index))
        if missing:
            raise ValueError(f"coded counties absent from the ladder: {missing[:5]}.")
        ratio = coded / census.reindex(coded.index)
        ratio = ratio / float(np.median(ratio.to_numpy()))
        result[value] = {int(k): float(v) for k, v in ratio.items()}
    return result


def cps_unidentified_county_shares(
    coverage: Mapping[int, Mapping[int, float]],
    excluded: Mapping[int, Collection[int]],
) -> dict[int, dict[int, float]]:
    """Opt-in: weight each coded-but-not-excluded county by its uncoded remainder.

    A county a file codes only in part appears among its own coded records
    and again in the ``GTCO = 0`` pool. Weighted by full population, it
    collects about ``(1 + coverage)`` of its population in expectation.
    Scaling its blocks in the pool by ``max(0, 1 - coverage)`` makes the
    expectation its population. Excluded counties already have share 0.
    """

    return {
        int(group): {
            int(county): max(0.0, 1.0 - float(ratio))
            for county, ratio in sorted(ratios.items())
            if int(county) not in set(excluded.get(group, ()))
        }
        for group, ratios in sorted(coverage.items())
    }


def cbsa_vintage_agreement(
    ladder: UsLocationLadder,
    county_fips: np.ndarray,
    cbsa_code: np.ndarray,
) -> dict[str, Any]:
    """Check a source's county → CBSA pairs against the ladder's delineation.

    Narrowing a draw to the source's CBSA is sound only when the source codes
    CBSAs on the ladder's delineation. Every distinct pair of (identified
    county, CBSA code) in the source must match the ladder's county → CBSA
    lookup. Returns ``{"agrees", "pairs", "disagreeing_pairs", "examples"}``.
    """

    county = _int_values(county_fips, label="county_fips")
    cbsa = _int_values(cbsa_code, label="cbsa_code")
    if len(county) != len(cbsa):
        raise ValueError("county_fips and cbsa_code must align.")
    block_county = ladder.block_geoid // _BLOCK_COUNTY_DIVISOR
    first_county, first_index = np.unique(block_county, return_index=True)
    ladder_cbsa = dict(
        zip(
            first_county.tolist(),
            ladder.cbsa_code[first_index].astype(np.int64).tolist(),
            strict=True,
        )
    )
    mask = county > 0
    pairs = sorted(
        {(int(c), int(b)) for c, b in zip(county[mask], cbsa[mask], strict=True)}
    )
    disagreeing = [
        {"county_fips": f"{c:05d}", "source_cbsa": b, "ladder_cbsa": ladder_cbsa.get(c)}
        for c, b in pairs
        if ladder_cbsa.get(c) != b
    ]
    return {
        "agrees": bool(pairs) and not disagreeing,
        "pairs": len(pairs),
        "disagreeing_pairs": len(disagreeing),
        "examples": disagreeing[:20],
    }


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def clone_frame_for_location(
    frame: Frame,
    clones: int,
    *,
    person_reference_columns: Collection[str] = (),
) -> Frame:
    """Replicate every household ``K`` times with ids ``id * K + k``.

    Every entity table is repeated row by row (so a unit's clones are
    adjacent and ascending ids stay ascending). Every entity id column and
    person membership column maps ``id → id * K + k``, as do the named
    person-reference columns (``0`` stays the no-person sentinel). Each
    entity gains ``{entity}_location_clone_index``. Weighted entities split
    each weight into ``K`` equal parts, so every total is conserved. Link
    tables, strata, the mass log and metadata carry over. ``K = 1`` returns
    ``frame`` itself.
    """

    clones = int(clones)
    if clones < 1:
        raise ValueError(f"clones must be >= 1, got {clones}.")
    if clones == 1:
        return frame
    schema = frame.schema
    id_columns = {entity: schema.entity_id_column(entity) for entity in frame.entities}
    membership = [schema.membership_column(group) for group in schema.group_entities]
    largest = max(
        int(np.abs(frame.table(entity)[column].to_numpy(dtype=np.int64)).max())
        for entity, column in id_columns.items()
    )
    if largest > (np.iinfo(np.int64).max - clones) // clones:
        raise ValueError("entity ids are too large to clone without overflow.")

    def remap(
        values: pd.Series, clone: np.ndarray, *, keep_zero: bool = False
    ) -> np.ndarray:
        raw = values.to_numpy(dtype=np.int64)
        mapped = raw * clones + clone
        return np.where(raw == 0, 0, mapped) if keep_zero else mapped

    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        original = frame.table(entity)
        clone_column = f"{entity}_location_clone_index"
        if clone_column in original.columns:
            raise ValueError(f"{entity} is already location-cloned ({clone_column}).")
        repeated = original.loc[original.index.repeat(clones)].reset_index(drop=True)
        clone = np.tile(np.arange(clones, dtype=np.int64), len(original))
        repeated[id_columns[entity]] = remap(repeated[id_columns[entity]], clone)
        if entity == schema.person_entity:
            for column in membership:
                repeated[column] = remap(repeated[column], clone)
            for column in person_reference_columns:
                if column in repeated.columns:
                    repeated[column] = remap(repeated[column], clone, keep_zero=True)
        repeated[clone_column] = clone
        tables[entity] = repeated
    for link_name in frame.links:
        link = frame.link(link_name)
        repeated = link.loc[link.index.repeat(clones)].reset_index(drop=True)
        clone = np.tile(np.arange(clones, dtype=np.int64), len(link))
        for column in id_columns.values():
            if column in repeated.columns:
                repeated[column] = remap(repeated[column], clone)
        tables[link_name] = repeated
    weights = {}
    for entity in frame.weighted_entities:
        stored = frame.weights_for(entity)
        split = Weights(
            values=np.repeat(stored.values / clones, clones), kind=stored.kind
        )
        stored.assert_mass_conserved(split)
        weights[entity] = split
    strata = pd.Series(
        np.repeat(frame.strata.to_numpy(), clones), name=frame.strata.name
    )
    return Frame(
        tables,
        schema,
        weights,
        strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def with_household_us_block_location(
    frame: Frame,
    ladder: UsLocationLadder,
    *,
    seed: int,
    clones: int = 1,
    puma: np.ndarray | None = None,
    county_fips: np.ndarray | None = None,
    identification_group: np.ndarray | None = None,
    identified_counties: Mapping[int, Collection[int]] | None = None,
    cbsa_code: np.ndarray | None = None,
    unidentified_county_share: Mapping[int, Mapping[int, float]] | None = None,
    person_reference_columns: Collection[str] = (),
    receipt: Mapping[str, Any] | None = None,
) -> tuple[Frame, pd.DataFrame]:
    """Locate every household of ``frame``; return the new frame and the table.

    Source-geography arrays align with the household table's rows. With
    ``K > 1`` the frame is cloned first (:func:`clone_frame_for_location`):
    clone ``k`` of household ``h`` becomes household ``h * K + k`` and receives
    the block drawn for ``(h, k)``. The household table gains every
    :func:`location_geography_columns` column, overwriting same-named inputs
    such as an ACS ``puma`` (which the draw reproduces). ``state_fips`` is
    never rewritten; the draw asserts each block lies in it. The frame's
    metadata gains a :data:`US_BLOCK_LOCATION_METADATA_KEY` receipt.
    """

    household = frame.table("household")
    if "state_fips" not in household.columns:
        raise ValueError("household table must contain 'state_fips'.")
    keys = _int_values(household["household_id"], label="household_id")
    draw = draw_us_block_locations(
        ladder,
        household_key=keys,
        state_fips=_int_values(household["state_fips"], label="state_fips"),
        seed=seed,
        clones=clones,
        puma=puma,
        county_fips=county_fips,
        identification_group=identification_group,
        identified_counties=identified_counties,
        cbsa_code=cbsa_code,
        unidentified_county_share=unidentified_county_share,
    )
    table = location_table(
        draw, ladder, household_key=keys, weights=frame.weights_for("household").values
    )
    located = clone_frame_for_location(
        frame, draw.clones, person_reference_columns=person_reference_columns
    )
    tables = {entity: located.table(entity).copy() for entity in located.entities}
    target = tables["household"]
    expected_ids = (
        table["household_id"] if draw.clones == 1 else table[LOCATION_CLONE_ID_COLUMN]
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(
        target["household_id"].to_numpy(dtype=np.int64), expected_ids
    ):
        raise AssertionError("internal error: cloned household order drifted.")
    for column in location_geography_columns(ladder):
        target[column] = table[column].to_numpy()
    for link_name in located.links:
        tables[link_name] = located.link(link_name).copy()
    location_receipt = {
        "rule": US_BLOCK_LOCATION_RULE_ID,
        "seed": int(seed),
        "clones": int(draw.clones),
        "ladder_sha256": ladder.sha256,
        **dict(receipt or {}),
    }
    result = Frame(
        tables,
        located.schema,
        {entity: located.weights_for(entity) for entity in located.weighted_entities},
        located.strata,
        mass_log=located.mass_log,
        metadata={**located.metadata, US_BLOCK_LOCATION_METADATA_KEY: location_receipt},
    )
    return result, table


# --------------------------------------------------------------------------
# Manifest and gate
# --------------------------------------------------------------------------


def us_block_location_manifest(
    table: pd.DataFrame,
    ladder: UsLocationLadder,
    *,
    seed: int,
    clones: int,
    candidate_set_rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The build-manifest record of a ``block_v1`` location step."""

    kinds = table[LOCATION_SOURCE_GEOGRAPHY_COLUMN].astype(str).to_numpy()
    first_clone = table[LOCATION_CLONE_INDEX_COLUMN].to_numpy() == 0
    record: dict[str, Any] = {
        "rule": US_BLOCK_LOCATION_RULE_ID,
        "location_rule": US_LOCATION_RULE_BLOCK_V1,
        "seed": int(seed),
        "clones": int(clones),
        "uniforms": (
            "stable_identity_uniforms('{household_id}:{clone}', seed, "
            f"salt={US_BLOCK_LOCATION_RULE_ID!r}) (blake2b-64)"
        ),
        "sampling_basis": "2020 block population (P.L. 94-171 POP100)",
        "block_ladder": {
            "sha256": ladder.sha256,
            "blocks": int(len(ladder)),
            "layer_vintages": ladder.layer_vintages,
            "carries_puma": ladder.puma is not None,
            "primary_congressional_district_plan": (
                ladder.primary_congressional_district_plan
            ),
            "congressional_district_plans": {
                plan: dict(spec)
                for plan, spec in sorted(
                    ladder.congressional_district_plan_sources.items()
                )
            },
        },
        "households": int(first_clone.sum()),
        "rows": int(len(table)),
        "source_geography_households": {
            kind: int(((kinds == kind) & first_clone).sum())
            for kind in SOURCE_GEOGRAPHY_KINDS
        },
        "candidate_set_rule": dict(candidate_set_rule or {}),
        "geography_columns": list(location_geography_columns(ladder)),
    }
    if LOCATION_WEIGHT_COLUMN in table.columns:
        record["location_weight_total"] = float(table[LOCATION_WEIGHT_COLUMN].sum())
    return record


def us_block_location_gate(
    household: pd.DataFrame,
    ladder: UsLocationLadder,
    *,
    state_fips_column: str = "state_fips",
) -> GateResult:
    """Every written geography must equal the ladder's lookup of its block.

    Fails on a malformed or unknown block, a ``state_fips`` that disagrees
    with the block's state, or any derived column that differs from what the
    block determines: the consistency invariant, stated as a release gate.
    """

    columns = location_geography_columns(ladder)
    missing = [
        column
        for column in (*columns, state_fips_column)
        if column not in household.columns
    ]
    if missing:
        return _gate([f"household table is missing geography column(s): {missing}"], {})
    text = household["block_geoid"].astype(str).to_numpy().astype(np.str_)
    malformed = ~((np.char.str_len(text) == 15) & np.char.isdigit(text))
    if malformed.any():
        return _gate(
            [f"block_geoid: {int(malformed.sum())} row(s) are not 15-digit codes"], {}
        )
    blocks = text.astype(np.int64)
    index = np.clip(np.searchsorted(ladder.block_geoid, blocks), 0, len(ladder) - 1)
    unknown = ladder.block_geoid[index] != blocks
    if unknown.any():
        return _gate(
            [
                f"block_geoid: {int(unknown.sum())} row(s) name a block the "
                "ladder does not carry"
            ],
            {},
        )
    failures: list[str] = []
    expected = derive_us_block_geography(ladder, index)
    for column in columns:
        want = expected[column]
        actual = household[column]
        if want.dtype.kind in "iu":
            got = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=np.float64)
            differs = got != want.astype(np.float64)
        else:
            differs = actual.astype(str).to_numpy() != want.astype(str)
        if differs.any():
            failures.append(
                f"{column}: {int(differs.sum())} row(s) disagree with the block's "
                "ladder lookup"
            )
    state = pd.to_numeric(household[state_fips_column], errors="coerce").to_numpy()
    mismatch = state != (blocks // _BLOCK_STATE_DIVISOR)
    if mismatch.any():
        failures.append(
            f"{state_fips_column}: {int(mismatch.sum())} row(s) disagree with the "
            "block's state"
        )
    return _gate(failures, {"rows": int(len(household)), "columns": list(columns)})


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, streamed."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of ``value`` as sorted, compact JSON."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _gate(failures: list[str], details: Mapping[str, Any]) -> GateResult:
    return GateResult(
        name="us_block_location",
        passed=not failures,
        failures=tuple(failures),
        details=dict(details),
    )


def _validated_cd_array(
    values: np.ndarray, block: np.ndarray, *, label: str
) -> np.ndarray:
    cd = _int_values(values, label=label)
    if ((cd < 100) | (cd > 9999)).any():
        bad = cd[(cd < 100) | (cd > 9999)][:5].tolist()
        raise ValueError(f"{label} must be state_fips*100+district; invalid: {bad}.")
    mismatched = (block // _BLOCK_STATE_DIVISOR) != (cd // 100)
    if mismatched.any():
        examples = [
            f"block {b:015d} -> cd {c}"
            for b, c in zip(
                block[mismatched][:5].tolist(),
                cd[mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(
            f"{label} districts must lie in their block's state: {examples}."
        )
    return cd


def _validated_puma_array(values: np.ndarray, block: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"puma must be an integer array, got dtype {raw.dtype}.")
    puma = raw.astype(np.int64)
    if len(puma) != len(block):
        raise ValueError(f"puma has {len(puma)} values for {len(block)} blocks.")
    if ((puma <= 0) | (puma % _PUMA_STATE_DIVISOR <= 0)).any():
        raise ValueError("puma must be state_fips*10**5+PUMA5CE for every block.")
    mismatched = (block // _BLOCK_STATE_DIVISOR) != (puma // _PUMA_STATE_DIVISOR)
    if mismatched.any():
        examples = [
            f"block {b:015d} -> puma {p:07d}"
            for b, p in zip(
                block[mismatched][:5].tolist(),
                puma[mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(f"puma must lie in its block's state: {examples}.")
    tract = block // 10**4
    same_tract = tract[1:] == tract[:-1]
    if (same_tract & (puma[1:] != puma[:-1])).any():
        raise ValueError("puma must be constant within a 2020 tract (tracts nest).")
    return puma


def _county_scaled_population(
    population: np.ndarray, block_county: np.ndarray, share: Mapping[int, float]
) -> np.ndarray:
    scaled = np.asarray(population, dtype=np.float64).copy()
    for county, value in share.items():
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"unidentified-county share for {int(county):05d} must be in "
                f"[0, 1], got {value!r}."
            )
        scaled[block_county == int(county)] *= value
    return scaled


def _require_same_state(
    state: np.ndarray, source_state: np.ndarray, mask: np.ndarray, label: str
) -> None:
    mismatched = mask & (source_state != state)
    if mismatched.any():
        examples = [
            f"state {int(s):02d} -> {label} state {int(t):02d}"
            for s, t in zip(
                state[mismatched][:5].tolist(),
                source_state[mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(f"household {label} disagrees with state_fips: {examples}.")


def _int_values(values: Any, *, label: str) -> np.ndarray:
    series = pd.Series(values).reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].head(5).tolist()
        raise ValueError(f"{label} contains missing or non-numeric value(s): {bad}.")
    floats = numeric.to_numpy(dtype=np.float64)
    if (floats != np.floor(floats)).any():
        raise ValueError(f"{label} must contain integers.")
    return numeric.to_numpy().astype(np.int64)


def _aligned(values: Any, n: int, *, label: str) -> np.ndarray:
    array = _int_values(values, label=label)
    if len(array) != n:
        raise ValueError(f"{label} has {len(array)} values for {n} households.")
    return array


def _optional_aligned(
    values: Any, n: int, *, label: str, missing: int = 0
) -> np.ndarray:
    if values is None:
        return np.full(n, missing, dtype=np.int64)
    numeric = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    return _aligned(numeric.fillna(missing), n, label=label)


def _optional_column(
    household: pd.DataFrame, column: str | None, *, missing: int = 0
) -> np.ndarray | None:
    if column is None:
        return None
    if column not in household.columns:
        raise ValueError(f"household table has no {column!r} column.")
    return (
        pd.to_numeric(household[column], errors="coerce")
        .fillna(missing)
        .to_numpy(dtype=np.float64)
    )


def _fips_strings(values: np.ndarray, *, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    return np.array(
        [("" if value == 0 else f"{value:0{width}d}") for value in array.tolist()],
        dtype=object,
    )


__all__ = [
    "CPS_ASEC_IDENTIFIED_COUNTIES_RESOURCE",
    "POPULACE_BLOCK_LADDER_SHA256_ATTR",
    "POPULACE_BLOCK_LADDER_VINTAGES_ATTR",
    "POPULACE_LOCATION_CLONES_ATTR",
    "POPULACE_LOCATION_RULE_ATTR",
    "POPULACE_LOCATION_SEED_ATTR",
    "LOCATION_CANDIDATE_BLOCKS_COLUMN",
    "LOCATION_CLONE_ID_COLUMN",
    "LOCATION_CLONE_INDEX_COLUMN",
    "LOCATION_SOURCE_GEOGRAPHY_COLUMN",
    "LOCATION_WEIGHT_COLUMN",
    "SOURCE_GEOGRAPHY_COUNTY",
    "SOURCE_GEOGRAPHY_KINDS",
    "SOURCE_GEOGRAPHY_PUMA",
    "SOURCE_GEOGRAPHY_STATE",
    "SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES",
    "SOURCE_GEOGRAPHY_STATE_UNIDENTIFIED_COUNTIES_CBSA",
    "US_BLOCK_LADDER_PUMA_ARRAY",
    "US_BLOCK_LOCATION_GEOGRAPHY_COLUMNS",
    "US_BLOCK_LOCATION_METADATA_KEY",
    "US_BLOCK_LOCATION_RULE_ID",
    "US_CD_PLAN_COLUMN_PREFIX",
    "US_LOCATION_RULE_BLOCK_V1",
    "US_LOCATION_RULE_CHOICES",
    "US_LOCATION_RULE_LEGACY",
    "CpsIdentifiedCountyList",
    "UsBlockLocationDraw",
    "UsLocationLadder",
    "assign_us_block_location",
    "attach_congressional_district_plans",
    "canonical_json_sha256",
    "cbsa_vintage_agreement",
    "clone_frame_for_location",
    "cps_county_coverage_ratios",
    "cps_excluded_county_sets",
    "cps_unidentified_county_shares",
    "default_cps_asec_identified_counties_path",
    "derive_us_block_geography",
    "draw_us_block_locations",
    "file_sha256",
    "identified_counties_by_group",
    "load_cps_asec_identified_counties",
    "load_us_location_ladder",
    "location_geography_columns",
    "location_table",
    "location_uniforms",
    "plan_column",
    "us_block_location_gate",
    "us_block_location_manifest",
    "us_location_ladder",
    "with_household_us_block_location",
]
