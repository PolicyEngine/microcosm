"""2023 county-equivalents on the US ladder's 2020 block geography.

Connecticut's nine planning regions replace its eight historical counties.
The existing Census BAF town assignment and county-to-town crosswalk show
that every town-defined 2020 CT tract lies in exactly one planning region.
The packaged tract crosswalk preserves that exact join without shipping a
second block ladder. Undefined coastal-water blocks cannot host households
and are refused. All other states retain their 2020 county FIPS.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from functools import cache
from importlib.resources import files
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from microcosm.build.us_runtime.block_ladder_sources import (
    CT_STATE_FIPS,
    parse_ct_planning_region_crosswalk,
)
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL

COUNTY_FIPS_2023_COLUMN = "county_fips_2023"
CT_TRACT_COUNTY_CROSSWALK_RESOURCE = "ct_2020_tract_to_2023_county.csv"
CT_TOWN_COUNTY_CROSSWALK_RESOURCE = "ct_cou_to_cousub_crosswalk.txt"
COUNTY_VINTAGE_PROVENANCE_RESOURCE = (
    CT_TRACT_COUNTY_CROSSWALK_RESOURCE + ".provenance.json"
)


def load_county_vintage_provenance() -> dict[str, Any]:
    """Read the packaged source URLs, digests and CT coverage evidence."""

    return json.loads(
        files("microcosm.build.us_runtime.data")
        .joinpath(COUNTY_VINTAGE_PROVENANCE_RESOURCE)
        .read_text(encoding="utf-8")
    )


@cache
def _ct_crosswalks() -> tuple[Mapping[str, str], Mapping[str, str], frozenset[str]]:
    resource_dir = files("microcosm.build.us_runtime.data")
    provenance = load_county_vintage_provenance()
    tract_bytes = resource_dir.joinpath(CT_TRACT_COUNTY_CROSSWALK_RESOURCE).read_bytes()
    town_bytes = resource_dir.joinpath(CT_TOWN_COUNTY_CROSSWALK_RESOURCE).read_bytes()
    for name, payload, expected in (
        (
            CT_TRACT_COUNTY_CROSSWALK_RESOURCE,
            tract_bytes,
            provenance["output_sha256"],
        ),
        (
            CT_TOWN_COUNTY_CROSSWALK_RESOURCE,
            town_bytes,
            provenance["source_files"]["ct_planning_region_crosswalk"]["sha256"],
        ),
    ):
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"County-vintage resource {name} has a SHA-256 mismatch.")
    reader = csv.DictReader(io.StringIO(tract_bytes.decode("utf-8")))
    if reader.fieldnames != ["tract_geoid", COUNTY_FIPS_2023_COLUMN, "population_2020"]:
        raise ValueError("County-vintage tract crosswalk has an invalid header.")
    tracts: dict[str, str] = {}
    for row in reader:
        tract = row["tract_geoid"]
        county = row[COUNTY_FIPS_2023_COLUMN]
        if (
            len(tract) != 11
            or not tract.isascii()
            or not tract.isdigit()
            or not tract.startswith(CT_STATE_FIPS)
            or len(county) != 5
            or not county.isascii()
            or not county.isdigit()
            or not county.startswith(CT_STATE_FIPS)
            or tract in tracts
        ):
            raise ValueError(f"Invalid county-vintage crosswalk row: {row}.")
        tracts[tract] = county
    towns = parse_ct_planning_region_crosswalk(
        town_bytes.decode("utf-8-sig").splitlines(keepends=True)
    )
    coverage = provenance["coverage"]
    if len(tracts) != coverage["tracts"] or len(towns) != coverage["towns"]:
        raise ValueError("County-vintage crosswalk coverage disagrees with provenance.")
    if set(tracts.values()) != set(towns.values()):
        raise ValueError("County-vintage tract and town region rosters disagree.")
    return (
        MappingProxyType(tracts),
        MappingProxyType(towns),
        frozenset(provenance["excluded_zero_population_blocks"]),
    )


def load_ct_planning_region_by_tract() -> Mapping[str, str]:
    """Return the immutable, hash-checked 2020 CT tract → 2023 county map."""

    return _ct_crosswalks()[0]


def load_ct_planning_region_by_town() -> Mapping[str, str]:
    """Return the existing Census crosswalk's immutable CT town → county map."""

    return _ct_crosswalks()[1]


def county_fips_2023_from_blocks(values: Iterable[object]) -> np.ndarray:
    """Map 2020 block GEOIDs to five-digit 2023 county-equivalent strings.

    Inputs are a one-dimensional iterable of 15-digit strings or integer
    GEOIDs. Unknown CT tracts and town-undefined water blocks fail closed;
    every populated 2020 Connecticut block is covered.
    """

    return _county_fips_2023(values, width=15)


def county_fips_2023_from_tracts(values: Iterable[object]) -> np.ndarray:
    """Map 2020 tract GEOIDs to five-digit 2023 county-equivalent strings.

    Every town-defined Connecticut tract maps unambiguously in the source
    BAF. This includes all populated CT tracts. Inputs are 11-digit strings
    or integer GEOIDs, with leading zeros restored for integer inputs.
    """

    return _county_fips_2023(values, width=11)


def county_fips_2023_assignment_failures(
    values: Iterable[object],
    legacy_counties: Iterable[object],
    *,
    blocks: Iterable[object] | None = None,
    tracts: Iterable[object] | None = None,
) -> tuple[str, ...]:
    """Check stored calibration counties, returning release-gate failure messages.

    A county-only PUMA assignment can check region membership and non-CT
    identity. Block- or tract-anchored assignments additionally check each
    stored county against the exact source crosswalk.
    """

    failures: list[str] = []
    arrays: list[np.ndarray] = []
    for label, items in (
        (COUNTY_FIPS_2023_COLUMN, values),
        ("county_fips", legacy_counties),
    ):
        if isinstance(items, (str, bytes)):
            return (f"{label}: expected a one-dimensional county vector.",)
        try:
            array = np.asarray(list(items), dtype=object)
        except TypeError:
            return (f"{label}: expected a one-dimensional county vector.",)
        if array.ndim != 1:
            return (f"{label}: expected a one-dimensional county vector.",)
        arrays.append(array)
    counties, legacy = arrays
    if len(counties) != len(legacy):
        return (
            f"{COUNTY_FIPS_2023_COLUMN}: {len(counties)} values for "
            f"{len(legacy)} household counties.",
        )
    valid = np.asarray(
        [
            isinstance(value, str)
            and len(value) == 5
            and value.isascii()
            and value.isdigit()
            for value in counties
        ],
        dtype=bool,
    )
    if not valid.all():
        failures.append(
            f"{COUNTY_FIPS_2023_COLUMN}: {int((~valid).sum())}/{len(counties)} "
            "row(s) are not five-digit text codes."
        )
    ct_mask = np.asarray(
        [
            isinstance(value, str) and value.startswith(CT_STATE_FIPS)
            for value in legacy
        ],
        dtype=bool,
    )
    if ct_mask.any():
        try:
            regions = frozenset(load_ct_planning_region_by_town().values())
        except (OSError, ValueError) as exc:
            failures.append(
                f"{COUNTY_FIPS_2023_COLUMN}: cannot load planning-region source: {exc}"
            )
        else:
            invalid_ct = (
                valid
                & ct_mask
                & np.asarray(
                    [
                        value not in regions if isinstance(value, str) else False
                        for value in counties
                    ],
                    dtype=bool,
                )
            )
            if invalid_ct.any():
                failures.append(
                    f"{COUNTY_FIPS_2023_COLUMN}: {int(invalid_ct.sum())} Connecticut "
                    "row(s) do not name a 2023 planning region."
                )
    non_ct_mismatch = np.asarray(
        [
            valid[index]
            and not ct_mask[index]
            and (not isinstance(legacy[index], str) or value != legacy[index])
            for index, value in enumerate(counties)
        ],
        dtype=bool,
    )
    if non_ct_mismatch.any():
        failures.append(
            f"{COUNTY_FIPS_2023_COLUMN}: {int(non_ct_mismatch.sum())} non-CT "
            "row(s) disagree with county_fips."
        )
    for label, source, mapper in (
        ("block_geoid", blocks, county_fips_2023_from_blocks),
        ("tract_geoid", tracts, county_fips_2023_from_tracts),
    ):
        if source is None:
            continue
        try:
            expected = mapper(source)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(f"{COUNTY_FIPS_2023_COLUMN}: cannot resolve {label}: {exc}")
            continue
        if len(expected) != len(counties):
            failures.append(
                f"{COUNTY_FIPS_2023_COLUMN}: {label} has {len(expected)} values "
                f"for {len(counties)} household counties."
            )
            continue
        mismatch = np.asarray(
            [
                valid[index] and value != expected[index]
                for index, value in enumerate(counties)
            ],
            dtype=bool,
        )
        if mismatch.any():
            failures.append(
                f"{COUNTY_FIPS_2023_COLUMN}: {int(mismatch.sum())} row(s) "
                f"disagree with {label}'s 2023 county-equivalent."
            )
    return tuple(failures)


def _county_fips_2023(values: Iterable[object], *, width: int) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError("County-vintage GEOIDs must be a one-dimensional iterable.")
    try:
        array = np.asarray(list(values), dtype=object)
    except TypeError as exc:
        raise ValueError(
            "County-vintage GEOIDs must be a one-dimensional iterable."
        ) from exc
    if array.ndim != 1:
        raise ValueError("County-vintage GEOIDs must be a one-dimensional iterable.")
    result = np.empty(len(array), dtype=object)
    ct_crosswalk: Mapping[str, str] | None = None
    excluded: frozenset[str] = frozenset()
    for index, value in enumerate(array):
        if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            geoid = f"{int(value):0{width}d}"
        elif isinstance(value, str):
            geoid = value
        else:
            raise ValueError(f"Expected a {width}-digit GEOID, got {value!r}.")
        if (
            len(geoid) != width
            or not geoid.isascii()
            or not geoid.isdigit()
            or geoid[:2] not in US_STATE_FIPS_TO_POSTAL
        ):
            raise ValueError(f"Expected a {width}-digit US GEOID, got {value!r}.")
        if not geoid.startswith(CT_STATE_FIPS):
            result[index] = geoid[:5]
            continue
        if ct_crosswalk is None:
            ct_crosswalk, _, excluded = _ct_crosswalks()
        if width == 15 and geoid in excluded:
            raise ValueError(f"Connecticut block {geoid} has no defined town.")
        county = ct_crosswalk.get(geoid[:11])
        if county is None:
            raise ValueError(
                f"Connecticut tract {geoid[:11]} is absent from the "
                "2023 county-equivalent crosswalk."
            )
        result[index] = county
    return result
