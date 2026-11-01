"""Census communal-establishment residents by area and age band (#1123, #1278).

The local age targets are mid-year estimates, which count residents of
communal establishments (student halls, care homes), while the FRS samples
private households. The census residence-type tables give each area's
communal share by age band: ONS RM120 for England and Wales and NRS UV101a for
Scotland carry both residence types by age at constituency and
local-authority grain, and NISRA CT0105 counts Northern Ireland's communal
residents by district. The target-surface receipt sizes the age-universe
change from these shares, and the incumbent evaluation reports the age rows on
a private-household basis beside the total-resident one (microcosm#1095).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: The census residence-type tables, by source concept, with the dimensions
#: that carry the age, the residence type and the sex.
COMMUNAL_RESIDENCE_TABLES: Mapping[str, tuple[str, str, str]] = {
    "ons.census2021_residents": (
        "C2021_AGE_24_NAME",
        "C2021_RESTYPE_3_NAME",
        "C_SEX_NAME",
    ),
    "nrs.census2022_residents": ("Age", "Residence Type Indicator", "Sex"),
}
#: Northern Ireland: NISRA CT0105 counts communal residents by district, age
#: band, sex and position (no household count), and the only district
#: population by age in the feed is NISRA's mid-2025 estimate. Its shares
#: spread each CT0105 band evenly over its single years and divide by that
#: estimate, so they are approximate.
NI_COMMUNAL_CONCEPT = "nisra.census2021_communal_residents"
NI_POPULATION_CONCEPT = "ons.mid_year_population_estimate"
NI_OLDEST_SINGLE_YEAR = 90
#: The local age cells' ten-year bands, then 80 and over (the region tier's
#: 80-89 row; every census residence-type table stops at 85 and over).
COMMUNAL_AGE_BANDS: tuple[tuple[int, int | None], ...] = (
    *((lower, lower + 10) for lower in range(0, 80, 10)),
    (80, None),
)
#: The grains the census tables carry for Great Britain.
COMMUNAL_LEVELS = ("constituency", "local_authority")


def band_key(band: tuple[int, int | None]) -> str:
    lower, upper = band
    return f"{lower}_plus" if upper is None else f"{lower}_{upper}"


BAND_KEYS = tuple(band_key(band) for band in COMMUNAL_AGE_BANDS)


def band_of(ages: tuple[int, int]) -> str | None:
    """The ten-year band a census age category falls wholly inside, if any."""

    low, high = ages
    for band in COMMUNAL_AGE_BANDS:
        lower, upper = band
        if low >= lower and (upper is None or high < upper):
            return band_key(band)
    return None


def age_range(label: str) -> tuple[int, int] | None:
    numbers = [int(value) for value in re.findall(r"\d+", str(label))]
    text = str(label).lower()
    if "under 1" in text:
        return (0, 0)
    if "and under" in text and len(numbers) == 1:
        return (0, numbers[0])
    if len(numbers) == 2:
        return (numbers[0], numbers[1])
    if len(numbers) == 1 and ("over" in text or "+" in text):
        return (numbers[0], 200)
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    return None


def is_communal_fact_line(line: str) -> bool:
    """A cheap test on a raw feed line before it is parsed."""

    census = "census2021_residents" in line or "census2022_residents" in line
    return (
        census
        or NI_COMMUNAL_CONCEPT in line
        or (NI_POPULATION_CONCEPT in line and "lgd_2014" in line)
    )


@dataclass
class CommunalCells:
    """The residents the census tables count, by area and residence type."""

    #: ``(level, area)`` -> ``{"<residence>_<band or all_ages or 18_24>": count}``
    #: for Great Britain, with residence ``communal`` or ``household``.
    totals: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    #: Northern Ireland district -> single year of age -> communal residents.
    ni_communal: dict[str, dict[int, float]] = field(default_factory=dict)
    #: Northern Ireland district -> single year of age -> mid-2025 residents.
    ni_population: dict[str, dict[int, float]] = field(default_factory=dict)


def communal_cells(facts: Iterable[Mapping[str, Any]]) -> CommunalCells:
    """Sum the census residence-type facts into ``CommunalCells``."""

    cells = CommunalCells()
    for fact in facts:
        concept = str((fact.get("observed_measure") or {}).get("source_concept"))
        geography = fact.get("geography") or {}
        dimensions = fact.get("dimensions") or {}
        value = float(fact.get("value") or 0.0)
        if concept == NI_COMMUNAL_CONCEPT:
            if str(geography.get("level")) != "local_authority":
                continue
            ages = age_range(str(dimensions.get("Age", "")))
            if ages is None:
                continue
            low, high = ages[0], min(ages[1], NI_OLDEST_SINGLE_YEAR)
            per_year = value / (high - low + 1)
            cell = cells.ni_communal.setdefault(str(geography.get("id")), {})
            for age in range(low, high + 1):
                cell[age] = cell.get(age, 0.0) + per_year
            continue
        if concept == NI_POPULATION_CONCEPT:
            if str(geography.get("vintage")) != "lgd_2014":
                continue
            if (fact.get("period") or {}).get("value") != 2025:
                continue
            ages = age_range(str(dimensions.get("age", "")))
            if ages is None:
                continue
            cell = cells.ni_population.setdefault(str(geography.get("id")), {})
            age = min(ages[0], NI_OLDEST_SINGLE_YEAR)
            cell[age] = cell.get(age, 0.0) + value
            continue
        keys = COMMUNAL_RESIDENCE_TABLES.get(concept)
        if keys is None:
            continue
        age_label, residence, sex = (str(dimensions.get(key, "")) for key in keys)
        if not sex.lower().startswith("all"):
            continue
        level = str(geography.get("level"))
        if level not in COMMUNAL_LEVELS:
            continue
        cell = cells.totals.setdefault((level, str(geography.get("id"))), {})
        residence_key = (
            "communal"
            if "communal" in residence.lower()
            else "household"
            if "household" in residence.lower()
            else "all"
        )
        if age_label.lower() == "total":
            cell[f"{residence_key}_all_ages"] = (
                cell.get(f"{residence_key}_all_ages", 0.0) + value
            )
            continue
        ages = age_range(age_label)
        if ages is None:
            continue
        if 18 <= ages[0] and ages[1] <= 24:
            cell[f"{residence_key}_18_24"] = (
                cell.get(f"{residence_key}_18_24", 0.0) + value
            )
        band = band_of(ages)
        if band is not None:
            cell[f"{residence_key}_{band}"] = (
                cell.get(f"{residence_key}_{band}", 0.0) + value
            )
    return cells


def band_shares(cells: CommunalCells, level: str) -> dict[str, dict[str, float]]:
    """Each band's communal share of residents, by Great Britain area at ``level``."""

    shares: dict[str, dict[str, float]] = {key: {} for key in BAND_KEYS}
    for (cell_level, area), cell in sorted(cells.totals.items()):
        if cell_level != level:
            continue
        for key in BAND_KEYS:
            total = cell.get(f"communal_{key}", 0.0) + cell.get(f"household_{key}", 0.0)
            if total:
                shares[key][area] = cell.get(f"communal_{key}", 0.0) / total
    return shares


def ni_band_counts(cells: CommunalCells) -> dict[str, dict[str, tuple[float, float]]]:
    """Each Northern Ireland district's communal residents and residents by band."""

    counts: dict[str, dict[str, tuple[float, float]]] = {}
    for area, communal_by_age in sorted(cells.ni_communal.items()):
        population = cells.ni_population.get(area)
        if not population:
            continue
        for band in COMMUNAL_AGE_BANDS:
            lower, upper = band
            last = NI_OLDEST_SINGLE_YEAR if upper is None else upper - 1
            communal = sum(
                communal_by_age.get(age, 0.0) for age in range(lower, last + 1)
            )
            total = sum(population.get(age, 0.0) for age in range(lower, last + 1))
            counts.setdefault(area, {})[band_key(band)] = (communal, total)
    return counts


def ni_band_shares(cells: CommunalCells) -> dict[str, dict[str, float]]:
    """Each band's approximate communal share, by Northern Ireland district."""

    shares: dict[str, dict[str, float]] = {key: {} for key in BAND_KEYS}
    for area, bands in ni_band_counts(cells).items():
        for key, (communal, total) in bands.items():
            if total:
                shares[key][area] = communal / total
    return shares


def private_household_shares(
    cells: CommunalCells,
) -> dict[tuple[str, str], dict[str, float]]:
    """``(level, area)`` -> band -> communal share, Northern Ireland approximate."""

    shares: dict[tuple[str, str], dict[str, float]] = {}
    for level in COMMUNAL_LEVELS:
        for key, by_area in band_shares(cells, level).items():
            for area, share in by_area.items():
                shares.setdefault((level, area), {})[key] = share
    for key, by_area in ni_band_shares(cells).items():
        for area, share in by_area.items():
            shares.setdefault(("local_authority", area), {})[key] = share
    return shares
