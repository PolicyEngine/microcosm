"""Verbatim copies of the US ladder population code before microcosm#1728 moved it.

``tools/us_acs_local_cd_surface.py`` and ``tools/build_us_acs_local_release.py``
held this code at ``868f414bc``. The state and district rows and specs now live
in the shared ``microcosm.build.ladder_population_targets``, with the US levels
in ``us/local_target_construction.json``. The equivalence tests hold the shared
code to these copies on state and district cells. Only one name changed:
``cd_surface.population_target_name`` is ``population_target_name`` here. The
county branches still name the county module; the tests do not reach them.
"""

# ruff: noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

GEOGRAPHY_STATE = "state"

GEOGRAPHY_CD = "congressional_district"

PERIOD = 2024

POPULATION_TARGET_CODE_DIGITS: Mapping[str, int] = {"state": 2, "cd": 4, "county": 5}

def population_target_name(geography: str, code: int) -> str:
    """A ladder population target's name: ``pop_state_06``, ``pop_cd_0601``."""

    digits = POPULATION_TARGET_CODE_DIGITS[geography]
    return f"pop_{geography}_{int(code):0{digits}d}"

def population_rows(
    households: pd.DataFrame,
    household_size: np.ndarray,
    ladder_populations: Mapping[str, Mapping[int, float]],
    geographies: Sequence[str],
    *,
    county_column: str = "county_fips",
) -> tuple[list[dict], sparse.csr_array, list[str]]:
    """Ladder population marginals as sparse rows (household size x 1[geo]).

    Returns (target records, CSR rows, dropped cell names). A ladder cell
    with no supporting household is dropped and named, as before.
    """

    if "county" in geographies:
        import us_acs_local_county_surface as county_surface

        return county_surface.population_rows(
            households, household_size, ladder_populations, geographies,
            county_column=county_column,
        )
    column_map = {
        "state": ("state_fips", "state", GEOGRAPHY_STATE),
        "cd": ("congressional_district_geoid", "cd", GEOGRAPHY_CD),
    }
    size32 = np.asarray(household_size, dtype=np.float32)
    records: list[dict] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    dropped: list[str] = []
    for geography in geographies:
        column, key, level = column_map[geography]
        codes = pd.to_numeric(households[column]).to_numpy()
        for value, population in sorted(ladder_populations[key].items()):
            name = population_target_name(geography, value)
            present = np.flatnonzero(codes == value)
            if not len(present):
                dropped.append(name)
                continue
            present = present[size32[present] != 0]
            row = len(records)
            rows.append(np.full(len(present), row, dtype=np.int32))
            cols.append(present.astype(np.int32))
            data.append(size32[present])
            record = {
                "name": name,
                "value": float(population),
                "source": "us_puma_ladder_2020",
                "family": "census_population_ladder",
                "geography_level": level,
                "state_fips": f"{int(value) // (100 if geography == 'cd' else 1):02d}",
                "congressional_district_geoid": (
                    f"{int(value):04d}" if geography == "cd" else None
                ),
            }
            records.append(record)
    matrix = sparse.csr_array(
        (
            np.concatenate(data) if data else np.empty(0, np.float32),
            (
                np.concatenate(rows) if rows else np.empty(0, np.int32),
                np.concatenate(cols) if cols else np.empty(0, np.int32),
            ),
        ),
        shape=(len(records), len(households)),
    )
    matrix.sort_indices()
    return records, matrix, dropped

def population_target_specs(names, values):
    """Declare ladder population targets with the shared diagnostics hierarchy.

    Each geography carries the label the release's other targets give the
    same id: a state its Census name, the name the ledger targets take from
    their facts ("Maine"), and a district the Census Bureau's name, which
    the IRS SOI district rows carry too. A registry refuses one US geography
    with two labels (microcosm#1335).
    """

    from microcosm.build.us_runtime.congressional_district_vintage import (
        CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
        congressional_district_label,
    )
    from microcosm.calibrate import TargetSpec
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME
    from microcosm.calibrate.hierarchy import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    provider = HierarchyNode("census_population", "Census population")
    category = HierarchyCategory(
        "census_population.resident_population",
        "Resident population",
        provider.id,
    )
    specs = []
    for name, value in zip(names, values, strict=True):
        if name.startswith("pop_county_"):
            specs.append(county_surface.population_spec(name, value, period=PERIOD))
            continue
        if name.startswith("pop_state_"):
            fips = name.removeprefix("pop_state_")
            label = US_STATE_FIPS_TO_NAME.get(fips)
            if label is None:
                raise ValueError(
                    f"Population target {name!r} names no US state FIPS code."
                )
            geography = HierarchyGeography(f"0400000US{fips}", label, "state")
        elif name.startswith("pop_cd_"):
            geoid = name.removeprefix("pop_cd_")
            # The ladder's districts are the 119th Congress plan.
            geography_id = f"{CURRENT_CONGRESSIONAL_DISTRICT_PREFIX}{geoid}"
            label = congressional_district_label(geography_id)
            geography = HierarchyGeography(
                geography_id, label, "congressional_district"
            )
        else:  # pragma: no cover - names originate in population_measure_arrays
            raise ValueError(f"Unknown population target name {name!r}.")
        specs.append(
            TargetSpec(
                name=name,
                entity="household",
                measure=name,
                value=value,
                period=PERIOD,
                source="US Census Bureau 2020 PUMA population ladder",
                family="census_population",
                metadata={"geography_level": geography.level},
                hierarchy=CalibrationHierarchy(
                    provider=provider,
                    category=category,
                    geography=geography,
                    dimensions=(),
                    target=HierarchyNode(name, f"{label} resident population"),
                ),
            )
        )
    return tuple(specs)

def ladder_population_target_specs(ladder_populations, geographies):
    """Every population target the ladder declares, before a household is read.

    :func:`population_targets` later drops a ladder cell no staged household
    supports, so its targets are a subset of these with the same labels.
    """

    cells = [
        (population_target_name(geography, code), float(population))
        for geography in geographies
        for code, population in sorted(ladder_populations[geography].items())
    ]
    return population_target_specs(
        [name for name, _population in cells],
        [population for _name, population in cells],
    )
