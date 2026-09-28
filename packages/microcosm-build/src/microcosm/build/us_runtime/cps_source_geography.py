"""CPS ASEC household source geography for the block location draw.

A CPS ASEC record's finest published geography is its county when the public
file identifies it (``GTCO`` nonzero), else its state, with the known
restriction that it does not live in any county the file identifies in full.
This module reads ``GESTFIPS``/``GTCO``/``GTCBSA`` from the processed ASEC H5s
(``census_cps_{income_year}.h5``, household table keyed by ``H_SEQ``), joins
them onto frame households by ``(source_year, source_household_id)``, and
builds the inputs :func:`~microcosm.build.us_runtime.block_location.draw_us_block_locations`
takes, plus the manifest record of the candidate-set rule.

``census_cps_{Y}.h5`` holds the ASEC fielded in ``Y + 1`` (income year ``Y``):
the microcosm #720 person-column join matches ``census_cps_2024.h5`` to
``pppub25.csv`` exactly. So a record's identification group is its
``source_year`` and its official list is ASEC ``source_year + 1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.block_location import (
    CpsIdentifiedCountyList,
    UsLocationLadder,
    cbsa_vintage_agreement,
    cps_county_coverage_ratios,
    cps_excluded_county_sets,
    cps_unidentified_county_shares,
    default_cps_asec_identified_counties_path,
    file_sha256,
    identified_counties_by_group,
    load_cps_asec_identified_counties,
)
from microcosm.frame import Frame

#: ``census_cps_{Y}.h5`` is the ASEC fielded in ``Y + 1``.
CPS_ASEC_SURVEY_YEAR_OFFSET = 1
CPS_HOUSEHOLD_GEOGRAPHY_COLUMNS = ("H_SEQ", "GESTFIPS", "GTCO", "GTCBSA")
_WEIGHT_COLUMNS = ("HSUP_WGT", "H_NUMPER")


def load_cps_household_geography(
    sources: Mapping[int, str | Path],
) -> pd.DataFrame:
    """Read every source year's household geography from its ASEC H5.

    Returns one row per source household with ``source_year``,
    ``source_household_id`` (``H_SEQ``), ``state_fips``, ``gtco``, ``gtcbsa``
    and ``person_weight`` (``HSUP_WGT * H_NUMPER``; its scale cancels in the
    median-normalized coverage ratios).
    """

    frames: list[pd.DataFrame] = []
    for year, path in sorted(sources.items()):
        household = pd.read_hdf(Path(path), "household")
        missing = [
            column
            for column in (*CPS_HOUSEHOLD_GEOGRAPHY_COLUMNS, *_WEIGHT_COLUMNS)
            if column not in household.columns
        ]
        if missing:
            raise ValueError(f"ASEC H5 {path} household table lacks {missing}.")
        if household["H_SEQ"].duplicated().any():
            raise ValueError(f"ASEC H5 {path} repeats H_SEQ.")
        frames.append(
            pd.DataFrame(
                {
                    "source_year": int(year),
                    "source_household_id": household["H_SEQ"].to_numpy(np.int64),
                    "state_fips": household["GESTFIPS"].to_numpy(np.int64),
                    "gtco": household["GTCO"].to_numpy(np.int64),
                    "gtcbsa": household["GTCBSA"].to_numpy(np.int64),
                    "person_weight": (
                        household["HSUP_WGT"].to_numpy(np.float64)
                        * household["H_NUMPER"].to_numpy(np.float64)
                    ),
                }
            )
        )
    if not frames:
        raise ValueError("no ASEC sources given.")
    return pd.concat(frames, ignore_index=True)


def household_source_keys(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """``(source_year, source_household_id)`` for each household row.

    Read from the person table: every member of a household must carry the
    same pair. Support copies (PUF detail, capital-gains tail) share their
    native row's pair, so they inherit its source geography.
    """

    person = frame.table("person")
    for column in ("source_year", "source_household_id", "person_household_id"):
        if column not in person.columns:
            raise ValueError(f"person table lacks {column!r}; no CPS source key.")
    keys = person[["person_household_id", "source_year", "source_household_id"]]
    distinct = keys.drop_duplicates()
    if distinct["person_household_id"].duplicated().any():
        raise ValueError("a household's members disagree on their CPS source key.")
    by_household = distinct.set_index("person_household_id")
    household_ids = frame.table("household")["household_id"].to_numpy(np.int64)
    aligned = by_household.reindex(household_ids)
    if aligned.isna().any().any():
        raise ValueError("some households have no member carrying a CPS source key.")
    return (
        aligned["source_year"].to_numpy(np.int64),
        aligned["source_household_id"].to_numpy(np.int64),
    )


@dataclass(frozen=True)
class CpsSourceGeography:
    """Draw inputs for CPS-lineage households, aligned to household rows."""

    county_fips: np.ndarray
    identification_group: np.ndarray
    identified_counties: dict[int, frozenset[int]]
    cbsa_code: np.ndarray | None
    unidentified_county_share: dict[int, dict[int, float]] | None
    record: dict[str, Any]


def cps_source_geography(
    ladder: UsLocationLadder,
    cps: pd.DataFrame,
    *,
    source_year: np.ndarray,
    source_household_id: np.ndarray,
    state_fips: np.ndarray,
    official: Mapping[int, CpsIdentifiedCountyList] | None = None,
    cbsa_narrowing: bool = True,
    partial_county_remainder: bool = False,
) -> CpsSourceGeography:
    """Source geography of CPS-lineage households and its manifest record.

    Every household joins its own ASEC row on ``(source_year,
    source_household_id)``; its ``state_fips`` must equal ``GESTFIPS``.
    County identification (coded sets, coverage ratios, CBSA agreement) is
    measured on the whole file of each year, not only the rows the frame
    kept. ``official`` defaults to the packaged Census lists.

    ``cbsa_narrowing`` narrows ``GTCO = 0`` households to their CBSA only if
    every coded (county, CBSA) pair in the files matches the ladder; the
    check's result is recorded either way. ``partial_county_remainder``
    (off by default; the ruled draw is plain block population) scales
    partially coded counties in the ``GTCO = 0`` pool by their uncoded
    remainder.
    """

    lookup = cps.set_index(["source_year", "source_household_id"])
    if not lookup.index.is_unique:
        raise ValueError("CPS household geography repeats a source key.")
    index = pd.MultiIndex.from_arrays(
        [
            np.asarray(source_year, dtype=np.int64),
            np.asarray(source_household_id, dtype=np.int64),
        ]
    )
    joined = lookup.reindex(index)
    if joined["gtco"].isna().any():
        missing = int(joined["gtco"].isna().sum())
        raise ValueError(f"{missing} household(s) have no ASEC source row.")
    state = np.asarray(state_fips, dtype=np.int64)
    if (joined["state_fips"].to_numpy(np.int64) != state).any():
        raise ValueError("household state_fips disagrees with its ASEC GESTFIPS.")
    gtco = joined["gtco"].to_numpy(np.int64)
    county = np.where(gtco > 0, state * 1000 + gtco, 0)
    group = np.asarray(source_year, dtype=np.int64)

    official_lists = (
        load_cps_asec_identified_counties() if official is None else official
    )
    coded = identified_counties_by_group(
        cps["state_fips"], cps["gtco"], cps["source_year"]
    )
    asec_year_of_group = {
        int(year): int(year) + CPS_ASEC_SURVEY_YEAR_OFFSET for year in coded
    }
    identified_rows = cps["gtco"].to_numpy(np.int64) > 0
    coded_county = np.where(
        identified_rows,
        cps["state_fips"].to_numpy(np.int64) * 1000 + cps["gtco"].to_numpy(np.int64),
        0,
    )
    coverage = cps_county_coverage_ratios(
        ladder,
        county_fips=coded_county,
        person_weight=cps["person_weight"].to_numpy(np.float64),
        group=cps["source_year"].to_numpy(np.int64),
    )
    excluded, record = cps_excluded_county_sets(
        coded,
        official_lists,
        asec_year_of_group=asec_year_of_group,
    )
    agreement = cbsa_vintage_agreement(
        ladder, coded_county, cps["gtcbsa"].to_numpy(np.int64)
    )
    use_cbsa = bool(cbsa_narrowing and agreement["agrees"])
    cbsa = (
        np.where(gtco == 0, joined["gtcbsa"].to_numpy(np.int64), 0)
        if use_cbsa
        else None
    )
    shares = (
        cps_unidentified_county_shares(coverage, excluded)
        if partial_county_remainder
        else None
    )
    partial = {
        str(year): {
            f"{county_fips:05d}": round(ratio, 4)
            for county_fips, ratio in sorted(ratios.items())
            if county_fips not in excluded.get(year, frozenset())
        }
        for year, ratios in coverage.items()
    }
    record.update(
        {
            "official_list_csv_sha256": (
                file_sha256(default_cps_asec_identified_counties_path())
                if official is None
                else None
            ),
            "cps_asec_survey_year_offset": CPS_ASEC_SURVEY_YEAR_OFFSET,
            "cbsa_narrowing": {
                "requested": bool(cbsa_narrowing),
                "applied": use_cbsa,
                "vintage_agreement": agreement,
            },
            "partial_county_remainder": {
                "applied": bool(partial_county_remainder),
                "coverage_ratio_definition": (
                    "coded weighted persons / 2020 census population, "
                    "median-normalized per source year"
                ),
                "coded_not_excluded_coverage": partial,
            },
        }
    )
    return CpsSourceGeography(
        county_fips=county,
        identification_group=np.where(gtco > 0, -1, group),
        identified_counties=excluded,
        cbsa_code=cbsa,
        unidentified_county_share=shares,
        record=record,
    )


__all__ = [
    "CPS_ASEC_SURVEY_YEAR_OFFSET",
    "CPS_HOUSEHOLD_GEOGRAPHY_COLUMNS",
    "CpsSourceGeography",
    "cps_source_geography",
    "household_source_keys",
    "load_cps_household_geography",
]
