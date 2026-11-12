"""ACS household-type indicators from the frame's observed person roster.

The ASEC source carries CPS ``A_EXPRRP`` and ``A_MARITL``; the ACS source
maps its native relationships onto those same codes before unit construction.
``age`` and ``is_female`` are the carried model inputs. No engine calculation
or tax-unit filing decision enters this household classification.

ACS group-quarters records are one-person placeholders, not households.
Their source ``TYPEHUGQ`` excludes them before the householder check; every
remaining household must have exactly one CPS reference person (code 1/2).
The ASEC frame has no household record-type field, so an ASEC row with no
householder refuses rather than being inferred to be group quarters.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .asec_census_person_columns import ASEC_CENSUS_PERSON_COLUMNS
from .relationship_inputs import _VALID_A_MARITL_CODES
from .spm_universe_source import (
    ACS_ARM,
    ACS_GROUP_QUARTERS_KINDS,
    ACS_HOUSEHOLD_KIND_COLUMN,
    ACS_HOUSEHOLD_KINDS,
    HOUSEHOLD_SPINE_COLUMN,
    HOUSEHOLD_SUPPORT_CHANNEL_COLUMN,
    SPM_UNIVERSE_CHANNEL_ARMS,
)

__all__ = ["HOUSEHOLD_TYPE_MATERIALIZER", "household_type_household_values"]

HOUSEHOLD_TYPE_MATERIALIZER = "household_type"


def household_type_household_values(
    frame: Any, metadata: Mapping[str, str]
) -> np.ndarray:
    """Return one 0/1 indicator per household in the frame's table order.

    ``household_type`` and ``own_children`` use Chronicle's ACS dimension
    values. In addition to the bound cells, the complements are available so
    callers can verify the household and child-presence partitions. A child
    is the householder's own child (CPS relationship 5), under 18 and never
    married (marital status 7). Missing columns and unreadable values refuse;
    optional state masks read the household's own ``state_fips``.
    """
    person = frame.table("person")
    household = frame.table("household")
    _require_columns(
        person,
        ("person_household_id", "A_EXPRRP", "A_MARITL", "age", "is_female"),
        "person",
    )
    _require_columns(household, ("household_id",), "household")
    household_ids = household["household_id"].to_numpy()
    if pd.Index(household_ids).has_duplicates:
        raise ValueError("Household-type targets require unique household ids.")
    positions = pd.Index(household_ids).get_indexer(person["person_household_id"])
    if (positions < 0).any():
        raise ValueError("A person's household is missing from the household table.")
    n_households = len(household)
    relationship_domain = next(
        column.domain
        for column in ASEC_CENSUS_PERSON_COLUMNS
        if column.name == "A_EXPRRP"
    )
    relationship = _codes(person, "A_EXPRRP", relationship_domain)
    marital = _codes(person, "A_MARITL", _VALID_A_MARITL_CODES)
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    if (~np.isfinite(age) | (age < 0)).any():
        raise ValueError("Household-type targets require a finite nonnegative age.")
    sex = person["is_female"]
    if sex.isna().any() or not sex.isin([False, True]).all():
        raise ValueError("Household-type targets require observed boolean is_female.")
    female = sex.to_numpy(dtype=bool)

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(positions[mask], minlength=n_households)

    members = count(np.ones(len(person), dtype=bool))
    in_household = _household_universe(household)
    if ((~in_household) & (members != 1)).any():
        raise ValueError("ACS group-quarters records must contain exactly one person.")
    head = np.isin(relationship, [1, 2])
    head_count = count(head)
    if (in_household & (head_count != 1)).any():
        bad = household_ids[in_household & (head_count != 1)][:5].tolist()
        raise ValueError(
            "Household-type targets require exactly one householder in each "
            f"household; invalid household ids: {bad}."
        )
    family = in_household & (count(relationship == 1) == 1)
    spouse = count(np.isin(relationship, [3, 4])) > 0
    female_head = count(head & female) == 1
    married = family & spouse
    other_family = family & ~spouse
    types = {
        "all": in_household,
        "family": family,
        "married_couple_family": married,
        "other_family": other_family,
        "male_householder_no_spouse": other_family & ~female_head,
        "female_householder_no_spouse": other_family & female_head,
        "nonfamily": in_household & ~family,
        "householder_living_alone": in_household & (members == 1),
        "householder_not_living_alone": in_household & ~family & (members > 1),
    }
    household_type = metadata.get("household_type", "")
    if household_type not in types:
        raise ValueError(f"Unknown ACS household type {household_type!r}.")
    own_child = (relationship == 5) & (age < 18) & (marital == 7)
    with_child = count(own_child) > 0
    own_children = metadata.get("own_children", "all")
    if own_children == "all":
        children = np.ones(n_households, dtype=bool)
    elif own_children == "with_own_children_under_18":
        children = with_child
    elif own_children == "no_own_children_under_18":
        children = ~with_child
    else:
        raise ValueError(f"Unknown ACS own-children filter {own_children!r}.")
    result = types[household_type] & children
    state_fips = metadata.get("state_fips")
    if state_fips:
        _require_columns(household, ("state_fips",), "household")
        state = pd.to_numeric(household["state_fips"], errors="coerce")
        if state.isna().any() or (state % 1 != 0).any():
            raise ValueError(
                "Household-type targets require observed integer state_fips."
            )
        result &= state.to_numpy() == int(state_fips)
    return result.astype(np.float64)


def _household_universe(household: pd.DataFrame) -> np.ndarray:
    """Exclude source-declared ACS group quarters, allowing ASEC's absent kind."""
    if ACS_HOUSEHOLD_KIND_COLUMN in household:
        raw = household[ACS_HOUSEHOLD_KIND_COLUMN]
        kind = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & ~kind.isin(ACS_HOUSEHOLD_KINDS)).any():
            raise ValueError("Household-type targets require TYPEHUGQ in {1, 2, 3}.")
    else:
        kind = pd.Series(np.nan, index=household.index)
    for column in (HOUSEHOLD_SPINE_COLUMN, HOUSEHOLD_SUPPORT_CHANNEL_COLUMN):
        if column in household:
            acs = household[column].map(SPM_UNIVERSE_CHANNEL_ARMS).eq(ACS_ARM)
            if (acs & ~kind.isin(ACS_HOUSEHOLD_KINDS)).any():
                raise ValueError(
                    "ACS household-type targets require observed TYPEHUGQ."
                )
    return ~kind.isin(ACS_GROUP_QUARTERS_KINDS).to_numpy()


def _require_columns(
    table: pd.DataFrame, columns: tuple[str, ...], entity: str
) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(
            f"Household-type targets require {entity} column(s): {missing}."
        )


def _codes(table: pd.DataFrame, column: str, allowed: Collection[int]) -> np.ndarray:
    values = pd.to_numeric(table[column], errors="coerce")
    if not values.isin(allowed).all():
        raise ValueError(f"Household-type targets require valid {column} codes.")
    return values.to_numpy(dtype=np.int64)
