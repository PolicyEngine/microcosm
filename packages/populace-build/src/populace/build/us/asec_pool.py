"""CPS ASEC source-year pooling for US support construction.

The production build needs to pool ASEC years before unit construction and
donor/source stages, not append partially populated rows after an H5 has
already been enhanced. This module is the reusable source-side primitive:
load raw Census CPS ASEC H5 files, make household ids globally unique across
years, scale each year to an explicit person-population share, and construct
the Populace US unit frame from the pooled person table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.frame import (
    MICROUNIT_REQUIRED_COLUMNS,
    US_SCHEMA,
    Frame,
    WeightKind,
    Weights,
    assign_us_unit_structure,
)

__all__ = [
    "AsecSource",
    "PooledAsecSource",
    "build_pooled_asec_unit_frame",
    "load_asec_h5_tables",
    "pool_asec_sources",
]

ASEC_WEIGHT_SCALE = 0.01
ASEC_PERSON_WEIGHT_COLUMN = "_populace_household_weight"


@dataclass(frozen=True)
class AsecSource:
    """One raw CPS ASEC H5 input to pool."""

    year: int
    path: Path
    share: float | None = None
    max_households: int | None = None


@dataclass(frozen=True)
class PooledAsecSource:
    """Pooled ASEC person frame plus per-person household weights."""

    person: pd.DataFrame
    person_household_weights: Weights
    metadata: Mapping[str, Any]


def load_asec_h5_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Load the raw Census CPS ASEC H5 tables needed for source pooling."""

    if not path.exists():
        raise FileNotFoundError(f"ASEC H5 not found: {path}")
    required = ("person", "household")
    tables: dict[str, pd.DataFrame] = {}
    with pd.HDFStore(path) as store:
        available = {key.strip("/") for key in store.keys()}
        missing = sorted(set(required) - available)
        if missing:
            raise ValueError(f"ASEC H5 {path} is missing table(s): {missing}.")
        for key in required:
            tables[key] = store[key].copy()
    return tables


def pool_asec_sources(
    sources: Iterable[AsecSource],
    *,
    weight_scale: float = ASEC_WEIGHT_SCALE,
) -> PooledAsecSource:
    """Pool raw ASEC source years before unit construction.

    Shares are resolved at the weighted-person-population level. If no source
    declares shares, every year receives equal share and total population is
    anchored to the first source's weighted person population.
    """

    source_list = tuple(sources)
    if not source_list:
        raise ValueError("At least one ASEC source is required.")
    shares = _resolved_shares(source_list)
    raw_inputs = [
        _prepare_year_input(source, weight_scale=weight_scale) for source in source_list
    ]
    target_population = raw_inputs[0]["raw_person_population"]

    pieces: list[pd.DataFrame] = []
    metadata_sources: list[dict[str, Any]] = []
    household_offset = 0
    for source, share, prepared in zip(source_list, shares, raw_inputs, strict=True):
        person = prepared["person"]
        household = prepared["household"]
        raw_person_population = float(prepared["raw_person_population"])
        scale = (
            0.0
            if raw_person_population == 0.0
            else (target_population * share / raw_person_population)
        )
        pooled, household_offset = _remap_source_year(
            source,
            person=person,
            household=household,
            household_offset=household_offset,
            weight_scale=weight_scale * scale,
        )
        weighted_population = float(pooled[ASEC_PERSON_WEIGHT_COLUMN].sum())
        pieces.append(pooled)
        metadata_sources.append(
            {
                "year": source.year,
                "path": str(source.path),
                "share": share,
                "raw_person_rows": int(len(person)),
                "raw_household_rows": int(len(household)),
                "raw_person_population": raw_person_population,
                "relationship_recode_source": prepared["relationship_recode_source"],
                "scale": scale,
                "weighted_person_population": weighted_population,
            }
        )

    pooled_person = pd.concat(pieces, ignore_index=True)
    metadata = {
        "target_person_population": float(target_population),
        "weighted_person_population": float(
            pooled_person[ASEC_PERSON_WEIGHT_COLUMN].sum()
        ),
        "sources": metadata_sources,
    }
    return PooledAsecSource(
        person=pooled_person,
        person_household_weights=Weights(
            pooled_person[ASEC_PERSON_WEIGHT_COLUMN].to_numpy(dtype=np.float64),
            WeightKind.DESIGN,
        ),
        metadata=metadata,
    )


def build_pooled_asec_unit_frame(
    sources: Iterable[AsecSource],
    *,
    target_year: int,
    tax_unit_mode: str = "policyengine",
) -> tuple[Frame, Mapping[str, Any]]:
    """Construct the US entity frame from pooled raw ASEC years."""

    pooled = pool_asec_sources(sources)
    strata = pd.Series(
        "asec_" + pooled.person["source_year"].astype(str),
        index=pooled.person.index,
        dtype="object",
    )
    frame = assign_us_unit_structure(
        pooled.person.drop(columns=[ASEC_PERSON_WEIGHT_COLUMN]),
        year=target_year,
        household_weights=pooled.person_household_weights,
        schema=US_SCHEMA,
        tax_unit_mode=tax_unit_mode,
        strata=strata,
    )
    return frame, pooled.metadata


def _resolved_shares(sources: tuple[AsecSource, ...]) -> tuple[float, ...]:
    raw = [source.share for source in sources]
    if all(value is None for value in raw):
        return tuple(1.0 / len(sources) for _source in sources)
    if any(value is None for value in raw):
        raise ValueError("Either all ASEC sources declare share or none do.")
    shares = tuple(float(value) for value in raw if value is not None)
    if any(value <= 0.0 for value in shares):
        raise ValueError("ASEC source shares must be positive.")
    total = sum(shares)
    if not np.isclose(total, 1.0):
        raise ValueError(f"ASEC source shares must sum to 1, got {total}.")
    return shares


def _prepare_year_input(source: AsecSource, *, weight_scale: float) -> dict[str, Any]:
    tables = load_asec_h5_tables(source.path)
    person = tables["person"].reset_index(drop=True)
    household = tables["household"].reset_index(drop=True)
    person, relationship_source = _with_relationship_recode(person)
    _require_columns(person, ("PH_SEQ", *MICROUNIT_REQUIRED_COLUMNS), label="person")
    _require_columns(household, ("H_SEQ", "HSUP_WGT"), label="household")

    if source.max_households is not None:
        keep = set(household["H_SEQ"].head(source.max_households).tolist())
        household = household[household["H_SEQ"].isin(keep)].copy()
        person = person[person["PH_SEQ"].isin(keep)].copy()

    raw_weights = _household_weights_by_raw_id(household, weight_scale=weight_scale)
    person_weights = person["PH_SEQ"].map(raw_weights).astype(float)
    if person_weights.isna().any():
        missing = person.loc[person_weights.isna(), "PH_SEQ"].head().tolist()
        raise ValueError(
            f"ASEC person rows reference household ids missing from household: {missing}."
        )
    return {
        "person": person.reset_index(drop=True),
        "household": household.reset_index(drop=True),
        "raw_person_population": float(person_weights.sum()),
        "relationship_recode_source": relationship_source,
    }


def _remap_source_year(
    source: AsecSource,
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_offset: int,
    weight_scale: float,
) -> tuple[pd.DataFrame, int]:
    raw_households = pd.Series(household["H_SEQ"]).drop_duplicates().sort_values()
    household_map = {
        raw_id: household_offset + index + 1
        for index, raw_id in enumerate(raw_households.tolist())
    }
    next_offset = household_offset + len(household_map)
    raw_weights = _household_weights_by_raw_id(household, weight_scale=weight_scale)

    result = person.copy(deep=True)
    result["source_year"] = int(source.year)
    result["source_household_id"] = result["PH_SEQ"].to_numpy()
    result["source_person_id"] = _source_person_id(result)
    result["source_row_id"] = np.arange(len(result), dtype=np.int64)
    result["PH_SEQ"] = result["PH_SEQ"].map(household_map).astype("int64")
    result["household_id"] = result["PH_SEQ"]
    result[ASEC_PERSON_WEIGHT_COLUMN] = (
        result["source_household_id"].map(raw_weights).astype(float)
    )
    return result, next_offset


def _household_weights_by_raw_id(
    household: pd.DataFrame, *, weight_scale: float
) -> pd.Series:
    weights = (
        household[["H_SEQ", "HSUP_WGT"]]
        .drop_duplicates("H_SEQ")
        .set_index("H_SEQ")["HSUP_WGT"]
    )
    return pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float) * float(
        weight_scale
    )


def _source_person_id(person: pd.DataFrame) -> pd.Series:
    if "PERIDNUM" in person:
        return person["PERIDNUM"].astype(str)
    return pd.Series(np.arange(len(person), dtype=np.int64), index=person.index)


def _with_relationship_recode(person: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "A_EXPRRP" in person:
        return person, "source:A_EXPRRP"
    _require_columns(
        person,
        ("PH_SEQ", "A_LINENO", "A_SPOUSE", "A_SEX", "PEPAR1", "PEPAR2"),
        label="person",
    )
    result = person.copy(deep=True)
    line = pd.to_numeric(result["A_LINENO"], errors="coerce").fillna(0).astype(int)
    spouse = pd.to_numeric(result["A_SPOUSE"], errors="coerce").fillna(0).astype(int)
    sex = pd.to_numeric(result["A_SEX"], errors="coerce").fillna(0).astype(int)
    parent1 = pd.to_numeric(result["PEPAR1"], errors="coerce").fillna(0).astype(int)
    parent2 = pd.to_numeric(result["PEPAR2"], errors="coerce").fillna(0).astype(int)
    household_size = result.groupby("PH_SEQ")["PH_SEQ"].transform("size")

    # CPSRelationshipCode values from microunit's public enum.
    recode = pd.Series(14, index=result.index, dtype="int64")
    recode.loc[household_size > 1] = 12
    recode.loc[(parent1 > 0) | (parent2 > 0)] = 5
    recode.loc[(spouse > 0) & (sex == 1)] = 3
    recode.loc[(spouse > 0) & (sex == 2)] = 4
    recode.loc[(line == 1) & (household_size > 1)] = 1
    recode.loc[(line == 1) & (household_size <= 1)] = 2
    result["A_EXPRRP"] = recode
    return result, "derived:line_spouse_parent"


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"ASEC {label} table is missing required column(s): {missing}."
        )
