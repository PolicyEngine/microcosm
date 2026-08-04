"""US PUMA-anchored geography-ladder assignment.

The US ACS artifact's geography spine is anchored at the 2020 Public Use
Microdata Area — the finest geography the ACS PUMS publishes. A household that
already knows its PUMA (an ACS-spine record) keeps it; a household that knows
only its state (an ASEC-spine record) is assigned a PUMA within that state,
sampled proportional to 2020 PUMA population. Every coarser layer of the
ladder then derives from the PUMA by a population-weighted draw over the
PUMA's overlap with that layer:

- **congressional district (119th)** — sampled within the PUMA proportional to
  the ``(PUMA, CD)`` block-population overlap (a 2020 PUMA can span several
  districts and a district several PUMAs; they do not nest).
- **county** — sampled within the PUMA proportional to the ``(PUMA, county)``
  block-population overlap.
- **tract** (behind ``assign_tract``) — sampled within the PUMA proportional to
  the ``(PUMA, tract)`` block-population overlap; when tract is assigned the
  county derives structurally from it (``county = tract // 10**6``) so the two
  never disagree. Congressional district, county and state are the launch
  requirement; tract is the stretch rung.

One national dataset, filter by geography, at any grain (populace #275; no
per-area files, the standing rule): the dense ACS-spine artifact is filtered to
state / congressional district / county for local analysis because every record
carries them.

Vintage discipline follows the country-spec geography-spine schema
(``vintage_policy: "error"``): the artifact records one vintage per derived
layer, the loader refuses an artifact that omits any of them, and the
assignment refuses a ladder whose congressional-district vintage differs from
the vintage the caller expects — a silent partial join on mismatched geography
vintages is the failure these checks exist to forbid (the #205 lesson).

Column names follow the policyengine-us household input surface
(``county_fips``, ``tract_geoid``, ``congressional_district_geoid``); the
2020-PUMA geoid rides as a ``puma`` data column (policyengine-us has no PUMA
input, exactly as the UK ladder carries ``*_code`` columns policyengine-uk has
no input for). The exported artifact carries engine inputs, never formula
outputs, so ``in_nyc``/``nyc_income_tax`` recompute from the county rung
instead of being persisted (the #34 regression).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.us_runtime.congressional_district_geography import (
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
)
from populace.build.us_runtime.geography_ladder import US_NYC_COUNTY_FIPS
from populace.build.us_runtime.puma_ladder_sources import (
    COUNTY_FROM_TRACT_DIVISOR,
    PUMA_GEOID_STATE_DIVISOR,
)
from populace.frame import Frame

#: Derived layers the ladder artifact must carry a vintage for.
US_PUMA_LADDER_DERIVED_LAYERS = (
    "congressional_district",
    "county",
    "tract",
)

#: Household spine columns the launch assignment writes (state / CD / county
#: filtering surface), in write order. ``puma`` is the anchor; the other two
#: are policyengine-us household inputs.
US_PUMA_LADDER_COLUMNS = (
    "puma",
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
    "county_fips",
)

#: The finer tract rung, written only when ``assign_tract=True``.
US_PUMA_LADDER_TRACT_COLUMN = "tract_geoid"

US_PUMA_LADDER_SCHEMA_VERSION = 1
US_PUMA_LADDER_KIND = "us_puma_ladder"

PUMA_LADDER_ARTIFACT_SHA256_ATTR = "populace_puma_ladder_artifact_sha256"
PUMA_LADDER_VINTAGES_ATTR = "populace_puma_ladder_vintages"

_ANCHOR_KEYS = ("puma", "puma_population")
_OVERLAP_KEYS = {
    "congressional_district": (
        "cd_overlap_puma",
        "cd_overlap_cd",
        "cd_overlap_population",
    ),
    "county": (
        "county_overlap_puma",
        "county_overlap_county",
        "county_overlap_population",
    ),
    "tract": (
        "tract_overlap_puma",
        "tract_overlap_tract",
        "tract_overlap_population",
    ),
}
_REQUIRED_ARRAY_KEYS = (
    *_ANCHOR_KEYS,
    *(key for keys in _OVERLAP_KEYS.values() for key in keys),
)


@dataclass(frozen=True)
class UsPumaLadder:
    """The national PUMA ladder: one anchor row per populated 2020 PUMA plus
    three ``(PUMA, layer_value, population)`` overlap tables sorted by
    ``(puma, layer_value)``.

    Attributes:
        puma: 2020 PUMA geoids (``state_fips * 10**5 + PUMA5CE``) as ``int64``,
            unique and sorted — the anchor rung and the ASEC state -> PUMA
            sampling frame.
        puma_population: 2020 census population per PUMA (the state -> PUMA
            sampling weight), strictly positive.
        cd_overlap_puma / cd_overlap_cd / cd_overlap_population: the
            ``(PUMA, congressional district)`` population overlap. CD geoids use
            the populace ``state_fips * 100 + district`` convention; at-large
            and delegate districts are district ``00``.
        county_overlap_puma / county_overlap_county / county_overlap_population:
            the ``(PUMA, county)`` population overlap; county fips are the
            5-digit ``state+county`` integer.
        tract_overlap_puma / tract_overlap_tract / tract_overlap_population:
            the ``(PUMA, tract)`` population overlap; tract geoids are the
            11-digit ``state+county+tract`` integer.
        metadata: the artifact's embedded metadata, including one vintage per
            derived layer.
    """

    puma: np.ndarray
    puma_population: np.ndarray
    cd_overlap_puma: np.ndarray
    cd_overlap_cd: np.ndarray
    cd_overlap_population: np.ndarray
    county_overlap_puma: np.ndarray
    county_overlap_county: np.ndarray
    county_overlap_population: np.ndarray
    tract_overlap_puma: np.ndarray
    tract_overlap_tract: np.ndarray
    tract_overlap_population: np.ndarray
    metadata: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.puma)

    @property
    def layer_vintages(self) -> dict[str, str]:
        """Vintage per derived layer, plus the PUMA anchor's own vintage."""
        layers = self.metadata["layers"]
        vintages = {
            layer: str(layers[layer]["vintage"])
            for layer in US_PUMA_LADDER_DERIVED_LAYERS
        }
        vintages["puma"] = str(self.metadata["puma_vintage"])
        return vintages

    def _overlap(self, layer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        puma_key, value_key, population_key = _OVERLAP_KEYS[layer]
        return (
            getattr(self, puma_key),
            getattr(self, value_key),
            getattr(self, population_key),
        )


def load_us_puma_ladder(path: str | Path) -> UsPumaLadder:
    """Load and validate a US PUMA-ladder artifact (NPZ).

    A single national file (never per-area files) built by
    ``tools/build_us_puma_ladder_artifact.py`` from primary Census sources.
    Every validation failure raises — a ladder that loads is a ladder every
    assignment invariant holds for, including exact per-PUMA population
    conservation of each overlap table.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"US PUMA ladder artifact not found: {source}")
    with np.load(source, allow_pickle=False) as payload:
        missing = [key for key in _REQUIRED_ARRAY_KEYS if key not in payload.files]
        if missing:
            raise ValueError(
                f"US PUMA ladder artifact is missing required key(s): {missing}."
            )
        if "metadata_json" not in payload.files:
            raise ValueError(
                "US PUMA ladder artifact is missing required key 'metadata_json'."
            )
        arrays = {key: np.asarray(payload[key]) for key in _REQUIRED_ARRAY_KEYS}
        metadata = _metadata_from_scalar(payload["metadata_json"])

    _validate_ladder_metadata(metadata)

    puma = _int64_array(arrays["puma"], label="puma")
    if len(puma) == 0:
        raise ValueError("US PUMA ladder artifact has zero PUMAs.")
    puma_state = puma // PUMA_GEOID_STATE_DIVISOR
    if ((puma_state < 1) | (puma_state > 56)).any():
        bad = puma[(puma_state < 1) | (puma_state > 56)][:5].tolist()
        raise ValueError(
            "puma geoids must be state_fips*10**5+PUMA5CE with state in "
            f"[1, 56]; invalid: {bad}."
        )
    if len(np.unique(puma)) != len(puma):
        raise ValueError("puma geoids must be unique.")
    if (np.diff(puma) <= 0).any():
        raise ValueError("puma geoids must be sorted ascending.")

    population = np.asarray(arrays["puma_population"], dtype=np.float64)
    if len(population) != len(puma):
        raise ValueError("puma_population must align with puma.")
    if not np.isfinite(population).all() or (population <= 0).any():
        raise ValueError("puma_population must be positive and finite for every PUMA.")

    puma_set = set(puma.tolist())
    cd_puma = _validated_overlap_puma(arrays["cd_overlap_puma"], puma_set, layer="cd")
    cd_value = _int64_array(arrays["cd_overlap_cd"], label="cd_overlap_cd")
    cd_pop = _overlap_population(arrays["cd_overlap_population"], cd_puma, layer="cd")
    if ((cd_value < 100) | (cd_value > 9999)).any():
        bad = cd_value[(cd_value < 100) | (cd_value > 9999)][:5].tolist()
        raise ValueError(
            f"cd_overlap_cd must be state_fips*100+district; invalid: {bad}."
        )
    _assert_layer_state_matches(
        cd_puma, cd_value // 100, layer="congressional_district"
    )
    _assert_conservation(
        puma, population, cd_puma, cd_pop, layer="congressional_district"
    )

    county_puma = _validated_overlap_puma(
        arrays["county_overlap_puma"], puma_set, layer="county"
    )
    county_value = _int64_array(
        arrays["county_overlap_county"], label="county_overlap_county"
    )
    county_pop = _overlap_population(
        arrays["county_overlap_population"], county_puma, layer="county"
    )
    if ((county_value < 1001) | (county_value > 56999)).any():
        bad = county_value[(county_value < 1001) | (county_value > 56999)][:5].tolist()
        raise ValueError(f"county_overlap_county must be 5-digit fips; invalid: {bad}.")
    _assert_layer_state_matches(county_puma, county_value // 1000, layer="county")
    _assert_conservation(puma, population, county_puma, county_pop, layer="county")

    tract_puma = _validated_overlap_puma(
        arrays["tract_overlap_puma"], puma_set, layer="tract"
    )
    tract_value = _int64_array(
        arrays["tract_overlap_tract"], label="tract_overlap_tract"
    )
    tract_pop = _overlap_population(
        arrays["tract_overlap_population"], tract_puma, layer="tract"
    )
    if ((tract_value < 10**9) | (tract_value >= 10**11)).any():
        bad = tract_value[(tract_value < 10**9) | (tract_value >= 10**11)][:5].tolist()
        raise ValueError(
            f"tract_overlap_tract must be 11-digit geoids; invalid: {bad}."
        )
    _assert_layer_state_matches(
        tract_puma, tract_value // COUNTY_FROM_TRACT_DIVISOR // 1000, layer="tract"
    )
    if len(np.unique(tract_value)) != len(tract_value):
        raise ValueError("tract_overlap_tract geoids must be unique (tracts nest).")
    _assert_conservation(puma, population, tract_puma, tract_pop, layer="tract")

    return UsPumaLadder(
        puma=puma,
        puma_population=population,
        cd_overlap_puma=cd_puma,
        cd_overlap_cd=cd_value,
        cd_overlap_population=cd_pop,
        county_overlap_puma=county_puma,
        county_overlap_county=county_value.astype(np.int32),
        county_overlap_population=county_pop,
        tract_overlap_puma=tract_puma,
        tract_overlap_tract=tract_value,
        tract_overlap_population=tract_pop,
        metadata=metadata,
    )


def assign_us_puma_ladder(
    household: pd.DataFrame,
    ladder: UsPumaLadder,
    *,
    seed: int = 0,
    assign_tract: bool = False,
    expected_congressional_district_vintage: str | None = None,
    state_fips_column: str = "state_fips",
    puma_column: str = "puma",
) -> pd.DataFrame:
    """Assign each household its PUMA anchor and the derived ladder columns.

    Records that already carry a valid ``puma`` (the ACS spine) keep it; records
    that carry none (the ASEC spine) draw a PUMA within their ``state_fips``
    proportional to 2020 PUMA population. Every record then draws a congressional
    district and a county within its PUMA proportional to the block-population
    overlap, and — when ``assign_tract`` — a tract from which the county derives
    structurally. Draws are consumed from one seeded generator in a fixed sorted
    order (states, then PUMAs), so the result is reproducible from ``seed``.

    A household ``puma`` absent from the ladder, or a household ``state_fips``
    the ladder has no PUMAs for, is an error — never a silent partial join.
    """

    if state_fips_column not in household.columns:
        raise ValueError(
            f"household table must contain {state_fips_column!r} before PUMA-"
            "ladder assignment."
        )
    if expected_congressional_district_vintage is not None:
        ladder_cd_vintage = ladder.layer_vintages["congressional_district"]
        if ladder_cd_vintage != expected_congressional_district_vintage:
            raise ValueError(
                "US PUMA ladder congressional-district vintage "
                f"{ladder_cd_vintage!r} does not match the vintage the caller "
                f"expects ({expected_congressional_district_vintage!r})."
            )

    state_strings = _state_fips_strings(household[state_fips_column])
    state_ints = state_strings.astype(np.int64)
    n = len(household)
    rng = np.random.default_rng(seed)

    row_puma = np.full(n, -1, dtype=np.int64)
    if puma_column in household.columns:
        known_mask, known_puma = _parse_input_puma(
            household[puma_column], ladder, state_ints
        )
        row_puma[known_mask] = known_puma[known_mask]
    unknown_mask = row_puma < 0
    if unknown_mask.any():
        _draw_puma_within_state(row_puma, unknown_mask, state_ints, ladder, rng)

    cd_puma, cd_value, cd_pop = ladder._overlap("congressional_district")
    cd_values = _draw_layer_values(
        row_puma, cd_puma, cd_value, cd_pop, rng, layer="congressional_district"
    )

    tract_values: np.ndarray | None = None
    if assign_tract:
        tract_puma, tract_value, tract_pop = ladder._overlap("tract")
        tract_values = _draw_layer_values(
            row_puma, tract_puma, tract_value, tract_pop, rng, layer="tract"
        )
        county_values = tract_values // COUNTY_FROM_TRACT_DIVISOR
    else:
        county_puma, county_value, county_pop = ladder._overlap("county")
        county_values = _draw_layer_values(
            row_puma, county_puma, county_value, county_pop, rng, layer="county"
        )

    _assert_row_state_consistency(
        state_ints,
        puma_state=row_puma // PUMA_GEOID_STATE_DIVISOR,
        cd_state=cd_values // 100,
        county_state=county_values // 1000,
    )

    assigned = household.copy()
    assigned["puma"] = np.array(
        [f"{value:07d}" for value in row_puma.tolist()], dtype=object
    )
    assigned[CONGRESSIONAL_DISTRICT_GEOID_COLUMN] = cd_values.astype(np.int64)
    assigned["county_fips"] = np.array(
        [f"{value:05d}" for value in county_values.tolist()], dtype=object
    )
    if tract_values is not None:
        assigned[US_PUMA_LADDER_TRACT_COLUMN] = np.array(
            [f"{value:011d}" for value in tract_values.tolist()], dtype=object
        )
    return assigned


def with_household_us_puma_ladder(
    frame: Frame,
    ladder: UsPumaLadder,
    *,
    seed: int = 0,
    assign_tract: bool = False,
    expected_congressional_district_vintage: str | None = None,
) -> Frame:
    """Return ``frame`` with the household PUMA ladder assigned."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"] = assign_us_puma_ladder(
        tables["household"],
        ladder,
        seed=seed,
        assign_tract=assign_tract,
        expected_congressional_district_vintage=(
            expected_congressional_district_vintage
        ),
    )
    for link_name in frame.links:
        tables[link_name] = frame.link(link_name).copy()
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(
        tables,
        frame.schema,
        weights,
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_puma_ladder_assignment_summary(
    household: pd.DataFrame,
    ladder: UsPumaLadder,
    *,
    weight_values: np.ndarray | None = None,
    assign_tract: bool = False,
) -> dict[str, Any]:
    """Summarize ladder assignment coverage for provenance logs."""

    columns = list(US_PUMA_LADDER_COLUMNS)
    if assign_tract:
        columns.append(US_PUMA_LADDER_TRACT_COLUMN)
    applied = all(column in household.columns for column in columns)
    weights = (
        np.asarray(weight_values, dtype=np.float64)
        if weight_values is not None
        else np.ones(len(household), dtype=np.float64)
    )
    if len(weights) != len(household):
        raise ValueError(
            f"weight_values length {len(weights)} does not match household "
            f"rows {len(household)}."
        )
    summary: dict[str, Any] = {
        "applied": applied,
        "household_rows": int(len(household)),
        "ladder_pumas": int(len(ladder)),
        "layer_vintages": ladder.layer_vintages,
        "sampling_basis": str(ladder.metadata["sampling_basis"]),
    }
    if not applied:
        return summary
    total = float(weights.sum())
    for column in columns:
        values = household[column].astype(str).to_numpy()
        nonempty = values != ""
        summary[f"assigned_{column}_values"] = int(
            pd.Series(values[nonempty]).nunique()
        )
        summary[f"{column}_nonempty_weighted_share"] = (
            float(weights[nonempty].sum() / total) if total > 0 else 0.0
        )
    county = household["county_fips"].astype(str).to_numpy()
    nyc = np.isin(county, US_NYC_COUNTY_FIPS)
    summary["nyc_weighted_household_share"] = (
        float(weights[nyc].sum() / total) if total > 0 else 0.0
    )
    return summary


def us_puma_ladder_gate(
    household: pd.DataFrame,
    weight_values: np.ndarray,
    *,
    assign_tract: bool = False,
    state_fips_column: str = "state_fips",
    nyc_share_bounds: tuple[float, float] = (0.005, 0.06),
    nyc_share_of_new_york_bounds: tuple[float, float] = (0.20, 0.65),
) -> GateResult:
    """Gate the exported PUMA ladder: structure, state consistency, and NYC mass.

    The NYC checks are the permanent form of the populace #34 regression:
    recomputing ``in_nyc`` from the PUMA-derived county rung must never silently
    collapse NYC to zero. NYC holds roughly 2.6% of national household weight and
    roughly 42% of New York State's, so the default bounds fail on collapse (0%)
    and on gross misassignment, not on calibration noise.
    """

    failures: list[str] = []
    details: dict[str, object] = {}
    weights = np.asarray(weight_values, dtype=np.float64)
    if len(weights) != len(household):
        raise ValueError(
            f"weight_values length {len(weights)} does not match household "
            f"rows {len(household)}."
        )

    expected_columns = [*US_PUMA_LADDER_COLUMNS, state_fips_column]
    if assign_tract:
        expected_columns.append(US_PUMA_LADDER_TRACT_COLUMN)
    missing_columns = [
        column for column in expected_columns if column not in household.columns
    ]
    if missing_columns:
        failures.append(
            f"household table is missing geography column(s): {missing_columns}"
        )
        return GateResult(
            name="us_puma_ladder",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    state = _state_fips_strings(household[state_fips_column])
    puma = household["puma"].astype(str).to_numpy()
    county = household["county_fips"].astype(str).to_numpy()
    for label, values, width in (("puma", puma, 7), ("county_fips", county, 5)):
        bad = ~(np.char.str_len(values.astype(np.str_)) == width) | ~np.char.isdigit(
            values.astype(np.str_)
        )
        if bad.any():
            failures.append(
                f"{label}: {int(bad.sum())}/{len(values)} row(s) are not "
                f"{width}-digit codes; examples {sorted(set(values[bad]))[:5]}"
            )

    prefix_checks = [
        ("puma", np.array([value[:2] for value in puma.tolist()]), state),
        ("county_fips", np.array([value[:2] for value in county.tolist()]), state),
    ]
    cd_state = _household_cd_state(household[CONGRESSIONAL_DISTRICT_GEOID_COLUMN])
    prefix_checks.append((CONGRESSIONAL_DISTRICT_GEOID_COLUMN, cd_state, state))
    if assign_tract:
        tract = household[US_PUMA_LADDER_TRACT_COLUMN].astype(str).to_numpy()
        bad_tract = ~(np.char.str_len(tract.astype(np.str_)) == 11) | ~np.char.isdigit(
            tract.astype(np.str_)
        )
        if bad_tract.any():
            failures.append(
                f"tract_geoid: {int(bad_tract.sum())}/{len(tract)} row(s) are "
                f"not 11-digit codes; examples {sorted(set(tract[bad_tract]))[:5]}"
            )
        prefix_checks.append(
            ("tract_geoid", np.array([value[:5] for value in tract.tolist()]), county)
        )
    for label, values, expected in prefix_checks:
        mismatched = values != expected
        if mismatched.any():
            failures.append(
                f"{label}: {int(mismatched.sum())}/{len(values)} row(s) "
                "disagree with the expected state/county prefix"
            )

    total = float(weights.sum())
    if total <= 0:
        failures.append("household weights sum to zero; shares are undefined")
        return GateResult(
            name="us_puma_ladder",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    nyc_mask = np.isin(county, US_NYC_COUNTY_FIPS)
    nyc_share = float(weights[nyc_mask].sum() / total)
    details["nyc_weighted_household_share"] = nyc_share
    low, high = nyc_share_bounds
    if not low <= nyc_share <= high:
        failures.append(
            f"NYC weighted household share {nyc_share:.4f} outside "
            f"[{low}, {high}] (the #34 in_nyc collapse regression)"
        )
    new_york = state == "36"
    new_york_total = float(weights[new_york].sum())
    if new_york_total > 0:
        nyc_of_ny = float(weights[nyc_mask & new_york].sum() / new_york_total)
        details["nyc_weighted_share_of_new_york"] = nyc_of_ny
        low, high = nyc_share_of_new_york_bounds
        if not low <= nyc_of_ny <= high:
            failures.append(
                f"NYC weighted share of New York State {nyc_of_ny:.4f} "
                f"outside [{low}, {high}]"
            )
    else:
        failures.append("New York State carries zero household weight")

    return GateResult(
        name="us_puma_ladder",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def _parse_input_puma(
    values: Any, ladder: UsPumaLadder, state_ints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split household ``puma`` into a known-mask and integer geoids.

    Null / non-positive entries are ASEC records to be drawn later. A positive
    entry must be a PUMA the ladder covers, in the household's own state — an
    absent PUMA is a vintage mismatch, never a silent partial join.
    """

    series = pd.Series(values).reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    known = np.isfinite(numeric) & (numeric > 0)
    non_integer = known & (numeric != np.floor(numeric))
    if non_integer.any():
        bad = series.to_numpy()[non_integer][:5].tolist()
        raise ValueError(f"puma contains non-integer geoid value(s): {bad}.")
    puma_int = np.full(len(series), -1, dtype=np.int64)
    puma_int[known] = numeric[known].astype(np.int64)

    ladder_set = set(ladder.puma.tolist())
    missing = sorted(set(puma_int[known].tolist()) - ladder_set)
    if missing:
        raise ValueError(
            "US PUMA ladder has no anchor row for household puma geoid(s): "
            f"{missing[:10]} ({len(missing)} total). The household PUMAs and the "
            "ladder artifact must share a PUMA vintage."
        )
    puma_state = puma_int[known] // PUMA_GEOID_STATE_DIVISOR
    want_state = state_ints[known]
    mismatched = puma_state != want_state
    if mismatched.any():
        examples = [
            f"state {want:02d} -> puma {value:07d}"
            for want, value in zip(
                want_state[mismatched][:5].tolist(),
                puma_int[known][mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(f"household puma state disagrees with state_fips: {examples}.")
    return known, puma_int


def _draw_puma_within_state(
    row_puma: np.ndarray,
    unknown_mask: np.ndarray,
    state_ints: np.ndarray,
    ladder: UsPumaLadder,
    rng: np.random.Generator,
) -> None:
    """Draw a PUMA within state (proportional to PUMA population) in place."""

    ladder_state = ladder.puma // PUMA_GEOID_STATE_DIVISOR
    unknown_positions = np.nonzero(unknown_mask)[0]
    unknown_states = state_ints[unknown_mask]
    for state in np.unique(unknown_states):
        rows = unknown_positions[unknown_states == state]
        in_state = ladder_state == state
        state_pumas = ladder.puma[in_state]
        if len(state_pumas) == 0:
            raise ValueError(
                f"US PUMA ladder has no PUMAs for state_fips {int(state):02d}; "
                "the ASEC household state and the ladder must share coverage."
            )
        weights = ladder.puma_population[in_state].astype(np.float64)
        row_puma[rows] = rng.choice(
            state_pumas, size=len(rows), replace=True, p=weights / weights.sum()
        )


def _draw_layer_values(
    row_puma: np.ndarray,
    overlap_puma: np.ndarray,
    overlap_value: np.ndarray,
    overlap_population: np.ndarray,
    rng: np.random.Generator,
    *,
    layer: str,
) -> np.ndarray:
    """Draw one layer value per row proportional to its PUMA's overlap.

    ``overlap_*`` are sorted by PUMA, so each PUMA's slice is contiguous;
    households are grouped by PUMA in sorted order for reproducibility.
    """

    n = len(row_puma)
    out = np.full(n, -1, dtype=np.int64)
    positions = pd.Series(np.arange(n, dtype=np.int64))
    for puma, group_positions in positions.groupby(pd.Series(row_puma), sort=True):
        lo = int(np.searchsorted(overlap_puma, puma, side="left"))
        hi = int(np.searchsorted(overlap_puma, puma, side="right"))
        if hi <= lo:
            raise ValueError(
                f"US PUMA ladder has no {layer} overlap for PUMA {int(puma)}."
            )
        values = overlap_value[lo:hi]
        weights = overlap_population[lo:hi].astype(np.float64)
        rows = group_positions.to_numpy()
        out[rows] = rng.choice(
            values, size=len(rows), replace=True, p=weights / weights.sum()
        )
    if (out < 0).any():
        raise ValueError(f"internal error: some rows received no {layer} draw.")
    return out


def _assert_row_state_consistency(
    state_ints: np.ndarray,
    *,
    puma_state: np.ndarray,
    cd_state: np.ndarray,
    county_state: np.ndarray,
) -> None:
    for label, values in (
        ("puma", puma_state),
        (CONGRESSIONAL_DISTRICT_GEOID_COLUMN, cd_state),
        ("county_fips", county_state),
    ):
        mismatched = values != state_ints
        if mismatched.any():
            examples = [
                f"state {want:02d} -> {label} state {got:02d}"
                for want, got in zip(
                    state_ints[mismatched][:5].tolist(),
                    values[mismatched][:5].tolist(),
                    strict=True,
                )
            ]
            raise ValueError(
                f"assigned {label} disagrees with household state_fips "
                f"(ladder is inconsistent): {examples}."
            )


def _metadata_from_scalar(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string.")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):
        raise ValueError("metadata_json must be a scalar JSON string.")
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    return metadata


def _validate_ladder_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("schema_version") != US_PUMA_LADDER_SCHEMA_VERSION:
        raise ValueError(
            "US PUMA ladder metadata schema_version must be "
            f"{US_PUMA_LADDER_SCHEMA_VERSION}, got {metadata.get('schema_version')!r}."
        )
    if metadata.get("kind") != US_PUMA_LADDER_KIND:
        raise ValueError(
            f"US PUMA ladder metadata kind must be {US_PUMA_LADDER_KIND!r}, "
            f"got {metadata.get('kind')!r}."
        )
    if not str(metadata.get("puma_vintage") or ""):
        raise ValueError("US PUMA ladder metadata must record puma_vintage.")
    if not str(metadata.get("sampling_basis") or ""):
        raise ValueError("US PUMA ladder metadata must record sampling_basis.")
    layers = metadata.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("US PUMA ladder metadata must carry a layers mapping.")
    missing = [layer for layer in US_PUMA_LADDER_DERIVED_LAYERS if layer not in layers]
    if missing:
        raise ValueError(
            f"US PUMA ladder metadata layers are missing: {missing}. Every "
            "derived layer records its own vintage (vintage_policy: error)."
        )
    for layer in US_PUMA_LADDER_DERIVED_LAYERS:
        spec = layers[layer]
        if not isinstance(spec, Mapping) or not str(spec.get("vintage") or ""):
            raise ValueError(
                f"US PUMA ladder layer {layer!r} must record a non-empty vintage."
            )
        if not str(spec.get("source") or ""):
            raise ValueError(
                f"US PUMA ladder layer {layer!r} must record a non-empty "
                "source citation."
            )


def _validated_overlap_puma(
    values: np.ndarray, puma_set: set[int], *, layer: str
) -> np.ndarray:
    array = _int64_array(values, label=f"{layer}_overlap_puma")
    if len(array) == 0:
        raise ValueError(f"{layer} overlap table is empty.")
    if (np.diff(array) < 0).any():
        raise ValueError(f"{layer}_overlap_puma must be sorted ascending.")
    unknown = sorted(set(array.tolist()) - puma_set)
    if unknown:
        raise ValueError(
            f"{layer} overlap references PUMA(s) absent from the anchor: {unknown[:5]}."
        )
    return array


def _overlap_population(
    values: np.ndarray, aligned: np.ndarray, *, layer: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != len(aligned):
        raise ValueError(
            f"{layer}_overlap_population must align with {layer}_overlap_puma."
        )
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"{layer} overlap population must be positive and finite.")
    return array


def _assert_layer_state_matches(
    overlap_puma: np.ndarray, value_state: np.ndarray, *, layer: str
) -> None:
    puma_state = overlap_puma // PUMA_GEOID_STATE_DIVISOR
    mismatched = puma_state != value_state
    if mismatched.any():
        examples = [
            f"puma {p} -> {layer} state {s}"
            for p, s in zip(
                overlap_puma[mismatched][:5].tolist(),
                value_state[mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(
            f"{layer} overlap state prefix must match its PUMA's state; "
            f"mismatched: {examples}."
        )


def _assert_conservation(
    puma: np.ndarray,
    population: np.ndarray,
    overlap_puma: np.ndarray,
    overlap_population: np.ndarray,
    *,
    layer: str,
) -> None:
    anchor = dict(zip(puma.tolist(), population.tolist(), strict=True))
    totals: dict[int, float] = {int(key): 0.0 for key in puma.tolist()}
    for key, value in zip(
        overlap_puma.tolist(), overlap_population.tolist(), strict=True
    ):
        totals[key] += value
    for key, total in totals.items():
        if abs(total - anchor[key]) > 1e-6:
            raise ValueError(
                f"{layer} overlap for PUMA {key} sums to {total} but the anchor "
                f"population is {anchor[key]} (must conserve exactly)."
            )


def _int64_array(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in ("i", "u"):
        raise ValueError(f"{label} must be an integer array, got dtype {array.dtype}.")
    return array.astype(np.int64, copy=False)


def _state_fips_strings(values: Any) -> np.ndarray:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].head(5).tolist()
        raise ValueError(f"state_fips contains non-numeric value(s): {bad}.")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    if (numeric_values != np.floor(numeric_values)).any():
        raise ValueError("state_fips must contain integer values.")
    integers = numeric_values.astype(np.int64)
    if ((integers < 1) | (integers > 99)).any():
        raise ValueError(
            "state_fips must contain two-digit state FIPS codes; invalid: "
            f"{integers[(integers < 1) | (integers > 99)][:5].tolist()}."
        )
    return np.array([f"{value:02d}" for value in integers.tolist()])


def _household_cd_state(values: Any) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any():
        bad = pd.Series(values)[numeric.isna()].head(5).tolist()
        raise ValueError(
            f"{CONGRESSIONAL_DISTRICT_GEOID_COLUMN} contains non-numeric "
            f"value(s): {bad}."
        )
    integers = numeric.to_numpy(dtype=np.int64)
    return np.array([f"{value // 100:02d}" for value in integers.tolist()])


__all__ = [
    "PUMA_LADDER_ARTIFACT_SHA256_ATTR",
    "PUMA_LADDER_VINTAGES_ATTR",
    "US_PUMA_LADDER_COLUMNS",
    "US_PUMA_LADDER_DERIVED_LAYERS",
    "US_PUMA_LADDER_KIND",
    "US_PUMA_LADDER_SCHEMA_VERSION",
    "US_PUMA_LADDER_TRACT_COLUMN",
    "UsPumaLadder",
    "assign_us_puma_ladder",
    "load_us_puma_ladder",
    "us_puma_ladder_assignment_summary",
    "us_puma_ladder_gate",
    "with_household_us_puma_ladder",
]
