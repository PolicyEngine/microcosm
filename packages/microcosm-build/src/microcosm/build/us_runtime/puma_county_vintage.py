"""Add 2023 county geography after the PUMA ladder's original draws."""

from __future__ import annotations

import numpy as np

from microcosm.build.us_runtime.block_ladder_sources import CT_STATE_FIPS
from microcosm.build.us_runtime.county_vintage import county_fips_2023_from_tracts
from microcosm.build.us_runtime.puma_ladder_sources import COUNTY_FROM_TRACT_DIVISOR


def county_fips_2023_for_puma_assignment(
    row_puma: np.ndarray,
    county_values: np.ndarray,
    *,
    tract_values: np.ndarray | None,
    tract_overlap: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve the county vintage without changing an assigned legacy county.

    An assigned tract determines its planning region directly. Without a
    stored tract, Connecticut rows draw one conditionally on their PUMA and
    already assigned 2020 county, using the existing tract population overlap.
    Call this after the ladder's original draws so those inputs keep their
    existing seed behavior. Other states consume no additional random draws.
    """

    if tract_values is not None:
        return county_fips_2023_from_tracts(tract_values)

    result = np.asarray([f"{value:05d}" for value in county_values], dtype=object)
    ct_mask = county_values // 1000 == int(CT_STATE_FIPS)
    if not ct_mask.any():
        return result

    tract_puma, tract_geoid, tract_population = tract_overlap
    tract_county = tract_geoid // COUNTY_FROM_TRACT_DIVISOR
    ct_pairs = np.unique(
        np.column_stack((row_puma[ct_mask], county_values[ct_mask])), axis=0
    )
    for puma, county in ct_pairs:
        rows = np.flatnonzero(ct_mask & (row_puma == puma) & (county_values == county))
        support = (tract_puma == puma) & (tract_county == county)
        if not support.any():
            raise ValueError(
                "No tract overlap for Connecticut PUMA "
                f"{puma:07d} and assigned 2020 county {county:05d}."
            )
        populations = tract_population[support]
        sampled = rng.choice(
            tract_geoid[support], size=len(rows), p=populations / populations.sum()
        )
        result[rows] = county_fips_2023_from_tracts(sampled)
    return result
