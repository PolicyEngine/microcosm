"""Join the primary ONS/Nomis sources into the UK OA-ladder NPZ payload.

The row-wise crosswalk builder (``geography_sources.py``) already composes the
2021 Census OA hierarchy, OA population, OA -> PCON24 constituency best-fit, OA
-> LAD23, and LAD -> region into an England-&-Wales OA frame. This module takes
that base frame and the three ladder-only layers — OA household counts (the
stage-one constituency-draw weight), the OA -> ward best-fit, and the LAD ->
ITL lookup — and assembles the array payload the ladder artifact stores.
Everything here is pure and unit-testable; download orchestration lives in
``tools/build_uk_oa_ladder_artifact.py``.

Sources and their vintages (each carried, sha-pinned, in the artifact
metadata; see the build tool for URLs and per-run sha256):

- OA21 -> LSOA21 -> MSOA21 -> LAD22 structural hierarchy, and OA21 -> LAD (April
  2023), OA21 -> PCON24 best-fit, and OA population (Nomis Census 2021 TS001):
  ONS Open Geography Portal / Nomis, via ``build_england_wales_crosswalk``.
- OA21 household counts: Nomis Census 2021 TS041 (number of households) at
  output-area grain — the stage-one weight, summed to the constituency.
- OA21 -> ward: ONS "Output Area (2021) to Ward to LAD to ... to Region to
  Country (2022) Best Fit Lookup in EW" (Open Geography Portal).
- LAD (April 2023) -> ITL3/ITL2/ITL1 (January 2021): ONS "Local Authority
  District (April 2023) to LAU1 to ITL3 to ITL2 to ITL1 Lookup in the UK". The
  LAD vintage matches the base frame's LAD (April 2023), so the join is exact,
  not a cross-vintage best-fit (the #205 rule).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

#: Columns the fully joined per-OA ladder frame must carry before assembly.
LADDER_OA_COLUMNS = (
    "oa_code",
    "population",
    "households",
    "constituency_code",
    "region_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "itl3_code",
)


def join_uk_oa_ladder_layers(
    base_crosswalk: pd.DataFrame,
    *,
    oa_households: pd.DataFrame,
    oa_ward: pd.DataFrame,
    lad_itl: pd.DataFrame,
) -> pd.DataFrame:
    """Join the ladder-only layers onto a base England-&-Wales OA crosswalk.

    ``base_crosswalk`` is the output of ``build_england_wales_crosswalk`` (one
    row per OA with ``oa_code``, ``population``, ``constituency_code``,
    ``region_code``, ``lsoa_code``, ``msoa_code``, and ``la_code`` = LAD April
    2023). The join is inner-by-construction: every OA must gain a household
    count and a ward, and every OA's LAD must gain an ITL — an unmatched row is
    a source defect, never a silently dropped or null-joined household (#205).
    """

    base = base_crosswalk.copy()
    missing = sorted(
        {
            "oa_code",
            "population",
            "constituency_code",
            "region_code",
            "lsoa_code",
            "msoa_code",
            "la_code",
        }
        - set(base.columns)
    )
    if missing:
        raise ValueError(f"base crosswalk is missing column(s): {missing}.")
    base = base.rename(columns={"la_code": "local_authority_code"})

    households = _normalise_lookup(
        oa_households, key="oa_code", value="households", label="OA households"
    )
    wards = _normalise_lookup(
        oa_ward, key="oa_code", value="ward_code", label="OA ward"
    )
    itl = _normalise_lookup(
        lad_itl, key="local_authority_code", value="itl3_code", label="LAD ITL"
    )

    joined = base.merge(households, on="oa_code", how="left")
    _raise_on_unmatched(joined, "households", key="oa_code", label="OA household count")
    joined = joined.merge(wards, on="oa_code", how="left")
    _raise_on_unmatched(joined, "ward_code", key="oa_code", label="OA ward")
    joined = joined.merge(itl, on="local_authority_code", how="left")
    _raise_on_unmatched(
        joined, "itl3_code", key="local_authority_code", label="LAD ITL"
    )

    joined["households"] = pd.to_numeric(joined["households"], errors="raise")
    return joined.loc[:, list(LADDER_OA_COLUMNS)]


def assemble_uk_oa_ladder(
    oa_frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Assemble the joined per-OA frame into the ladder artifact's NPZ payload.

    Output areas with no usual residents (population <= 0) cannot seat a
    household draw and are dropped — they belong outside the artifact, matching
    the US block ladder's exclusion of unpopulated blocks. Every remaining OA
    carries a positive population, a non-negative household count, and a
    non-blank code for every derived layer.
    """

    missing = sorted(set(LADDER_OA_COLUMNS) - set(oa_frame.columns))
    if missing:
        raise ValueError(f"ladder OA frame is missing column(s): {missing}.")
    frame = oa_frame.loc[:, list(LADDER_OA_COLUMNS)].copy()

    for column in LADDER_OA_COLUMNS:
        if column in ("population", "households"):
            continue
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        blank = frame[column] == ""
        if blank.any():
            raise ValueError(
                f"ladder OA frame has {int(blank.sum())} blank {column} value(s)."
            )

    if frame["oa_code"].duplicated().any():
        duplicates = frame.loc[frame["oa_code"].duplicated(), "oa_code"].unique()
        raise ValueError(
            "ladder OA frame oa_code values must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )

    population = pd.to_numeric(frame["population"], errors="raise").to_numpy(
        dtype=np.float64
    )
    households = pd.to_numeric(frame["households"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(population).all():
        raise ValueError("ladder OA frame population contains non-finite values.")
    if not np.isfinite(households).all() or (households < 0).any():
        raise ValueError("ladder OA frame households must be finite and non-negative.")
    populated = population > 0
    if not populated.any():
        raise ValueError("ladder OA frame has no populated output areas.")
    frame = frame[populated].reset_index(drop=True)
    population = population[populated]
    households = households[populated]

    payload = {
        "oa_code": frame["oa_code"].to_numpy().astype("U"),
        "population": population.astype(np.float64),
        "households": households.astype(np.float64),
        "constituency_code": frame["constituency_code"].to_numpy().astype("U"),
        "region_code": frame["region_code"].to_numpy().astype("U"),
        "lsoa_code": frame["lsoa_code"].to_numpy().astype("U"),
        "msoa_code": frame["msoa_code"].to_numpy().astype("U"),
        "local_authority_code": frame["local_authority_code"].to_numpy().astype("U"),
        "ward_code": frame["ward_code"].to_numpy().astype("U"),
        "itl3_code": frame["itl3_code"].to_numpy().astype("U"),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    return payload


def concat_uk_ladder_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate per-country ladder OA frames into one full-UK frame.

    Every frame must carry the full ladder column set and be non-empty, and
    OA codes must be disjoint across frames — the concat is the seam where a
    silent cross-country code collision would corrupt every derived layer,
    so it fails closed instead.
    """

    if not frames:
        raise ValueError("at least one ladder frame is required.")
    for index, frame in enumerate(frames):
        if frame.empty:
            raise ValueError(f"ladder frame {index} is empty.")
        missing = sorted(set(LADDER_OA_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"ladder frame {index} is missing column(s): {missing}.")
    combined = pd.concat(
        [frame.loc[:, list(LADDER_OA_COLUMNS)] for frame in frames],
        ignore_index=True,
    )
    if combined["oa_code"].duplicated().any():
        duplicates = combined.loc[combined["oa_code"].duplicated(), "oa_code"].unique()
        raise ValueError(
            "ladder frames must have disjoint oa_code values; duplicate "
            f"value(s): {list(map(str, duplicates[:5]))}."
        )
    return combined


def _normalise_lookup(
    frame: pd.DataFrame,
    *,
    key: str,
    value: str,
    label: str,
) -> pd.DataFrame:
    if key not in frame.columns or value not in frame.columns:
        raise ValueError(f"{label} lookup must include {key!r} and {value!r}.")
    lookup = frame.loc[:, [key, value]].copy()
    lookup[key] = lookup[key].fillna("").astype(str).str.strip()
    blank_key = lookup[key] == ""
    if blank_key.any():
        raise ValueError(f"{label} lookup contains blank {key} value(s).")
    if lookup[key].duplicated().any():
        duplicates = lookup.loc[lookup[key].duplicated(), key].unique()
        raise ValueError(
            f"{label} lookup {key} values must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return lookup


def _raise_on_unmatched(
    frame: pd.DataFrame,
    column: str,
    *,
    key: str,
    label: str,
) -> None:
    unmatched = frame[frame[column].isna()][key]
    if not unmatched.empty:
        raise ValueError(
            f"{label} lookup is missing {len(unmatched)} {key} value(s); "
            f"examples: {list(map(str, unmatched.unique()[:5]))}."
        )


__all__ = [
    "LADDER_OA_COLUMNS",
    "assemble_uk_oa_ladder",
    "concat_uk_ladder_frames",
    "join_uk_oa_ladder_layers",
]
