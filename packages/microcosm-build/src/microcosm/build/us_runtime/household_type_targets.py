"""ACS household-type indicators from the frame's observed person roster.

The ASEC source carries CPS ``A_EXPRRP`` and ``A_MARITL``; the ACS source
maps its native relationships onto those same codes before unit construction.
``age`` and ``is_female`` are the carried model inputs. No engine calculation
or tax-unit filing decision enters this household classification.

ACS group-quarters records are one-person placeholders, not households.
Source ``TYPEHUGQ`` and ASEC ``HRHTYPE`` exclude group quarters before the
householder check; ASEC group quarters may contain multiple people and a
valid code-1/2 householder. Historical ASEC frames without HRHTYPE still use
the roster: their group quarters cannot be distinguished from housing units.
That roster fallback warns (:class:`HouseholdUniverseFallbackWarning`), and
:func:`household_universe_receipt` counts how many households and how much
weight it classified. A frame that carries HRHTYPE has no fallback: every
ASEC household, and every household without TYPEHUGQ, must carry a code, and
no household may carry both codes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .asec_census_person_columns import ASEC_CENSUS_PERSON_COLUMNS
from .asec_household_type import (
    ASEC_GROUP_QUARTERS_KINDS,
    ASEC_HOUSEHOLD_KIND_COLUMN,
    ASEC_HOUSEHOLD_KINDS,
)
from .relationship_inputs import _VALID_A_MARITL_CODES
from .spm_universe_source import (
    ACS_ARM,
    ACS_GROUP_QUARTERS_KINDS,
    ACS_HOUSEHOLD_KIND_COLUMN,
    ACS_HOUSEHOLD_KINDS,
    ASEC_ARM,
    HOUSEHOLD_SPINE_COLUMN,
    HOUSEHOLD_SUPPORT_CHANNEL_COLUMN,
    SPM_UNIVERSE_CHANNEL_ARMS,
)

__all__ = [
    "HOUSEHOLD_TYPE_MATERIALIZER",
    "HOUSEHOLD_UNIVERSE_RECEIPT_SCHEMA_VERSION",
    "HOUSEHOLD_UNIVERSE_SOURCES",
    "ROSTER_FALLBACK",
    "HouseholdUniverseFallbackWarning",
    "household_type_household_values",
    "household_universe_receipt",
]

HOUSEHOLD_TYPE_MATERIALIZER = "household_type"

#: The label of a household that carries neither TYPEHUGQ nor HRHTYPE.
ROSTER_FALLBACK = "roster_fallback"
#: Each household is classified by exactly one of these.
HOUSEHOLD_UNIVERSE_SOURCES = (
    ACS_HOUSEHOLD_KIND_COLUMN,
    ASEC_HOUSEHOLD_KIND_COLUMN,
    ROSTER_FALLBACK,
)
HOUSEHOLD_UNIVERSE_RECEIPT_SCHEMA_VERSION = 1
_HOUSEHOLD_UNIVERSE_RULE = (
    "TYPEHUGQ classifies ACS records and HRHTYPE classifies ASEC records; a "
    "household with neither uses its roster alone, so any group quarters "
    "among those households count as housing households."
)
_ROSTER_FALLBACK_WARNING = (
    "Household-type targets classified households that carry neither "
    "TYPEHUGQ nor HRHTYPE by their roster alone: ASEC group quarters among "
    "them count as housing households. household_universe_receipt counts "
    "them, and the US fiscal-refresh release records that receipt in its "
    "calibration diagnostics."
)


class HouseholdUniverseFallbackWarning(UserWarning):
    """Household-type targets classified some households by the roster alone."""


def household_type_household_values(
    frame: Any, metadata: Mapping[str, str]
) -> np.ndarray:
    """Return one 0/1 indicator per household in the frame's table order.

    ``household_type`` and ``own_children`` use Chronicle's ACS dimension
    values. In addition to the bound cells, the complements are available so
    callers can verify the household and child-presence partitions. A child
    is the householder's own child (CPS relationship 5), under 18 and never
    married (marital status 7). Missing columns and unreadable values refuse;
    optional state masks read the household's own ``state_fips``. When any
    household is classified by the roster fallback, it emits
    :class:`HouseholdUniverseFallbackWarning` through :func:`warnings.warn`.
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
    in_household, sources = _household_universe(household)
    if (sources == ROSTER_FALLBACK).any():
        warnings.warn(
            _ROSTER_FALLBACK_WARNING, HouseholdUniverseFallbackWarning, stacklevel=2
        )
    acs_gq = (
        pd.to_numeric(household[ACS_HOUSEHOLD_KIND_COLUMN], errors="coerce")
        .isin(ACS_GROUP_QUARTERS_KINDS)
        .to_numpy()
        if ACS_HOUSEHOLD_KIND_COLUMN in household
        else np.zeros(n_households, dtype=bool)
    )
    if (acs_gq & (members != 1)).any():
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


def household_universe_receipt(
    frame: Any,
    *,
    weights: Any,
    indicators: Mapping[str, Any],
) -> dict[str, Any]:
    """Record which field classified each household, and the fallback's weight.

    ``weights`` and every array in ``indicators`` (target name to the 0/1
    values :func:`household_type_household_values` returned) align with the
    household table's rows. The receipt counts households (unweighted rows)
    and their weight by :data:`HOUSEHOLD_UNIVERSE_SOURCES`, and gives each
    target's weighted value with the part that roster-fallback households
    contribute. Sums are exactly rounded (:func:`math.fsum`), so the receipt
    does not depend on summation order.
    """
    household = frame.table("household")
    _require_columns(household, ("household_id",), "household")
    _, sources = _household_universe(household)
    n_households = len(household)
    weight_values = _aligned_finite(weights, n_households, "weights")
    fallback = sources == ROSTER_FALLBACK
    targets = {}
    for name in sorted(indicators):
        values = _aligned_finite(indicators[name], n_households, f"indicator {name!r}")
        contribution = weight_values * values
        targets[str(name)] = {
            "weighted_value": math.fsum(contribution),
            "roster_fallback_weighted_value": math.fsum(contribution[fallback]),
        }
    return {
        "schema_version": HOUSEHOLD_UNIVERSE_RECEIPT_SCHEMA_VERSION,
        "rule": _HOUSEHOLD_UNIVERSE_RULE,
        "households": {
            source: int((sources == source).sum())
            for source in HOUSEHOLD_UNIVERSE_SOURCES
        },
        "weighted_households": {
            source: math.fsum(weight_values[sources == source])
            for source in HOUSEHOLD_UNIVERSE_SOURCES
        },
        "targets": targets,
    }


def _household_universe(household: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exclude native group quarters, and name the field behind each row.

    Returns the housing-household mask and, per row, the entry of
    :data:`HOUSEHOLD_UNIVERSE_SOURCES` that classified it. Without an HRHTYPE
    column, rows without TYPEHUGQ fall back to the roster. With one, missing
    codes refuse on the ASEC arm and on rows that carry no TYPEHUGQ: a frame
    built with the carry has no record the roster must classify.
    """
    if ACS_HOUSEHOLD_KIND_COLUMN in household:
        raw = household[ACS_HOUSEHOLD_KIND_COLUMN]
        kind = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & ~kind.isin(ACS_HOUSEHOLD_KINDS)).any():
            raise ValueError("Household-type targets require TYPEHUGQ in {1, 2, 3}.")
    else:
        kind = pd.Series(np.nan, index=household.index)
    acs_kind_observed = kind.notna().to_numpy()
    asec_arm = np.zeros(len(household), dtype=bool)
    arm_read = False
    for column in (HOUSEHOLD_SPINE_COLUMN, HOUSEHOLD_SUPPORT_CHANNEL_COLUMN):
        if column in household:
            arm = household[column].map(SPM_UNIVERSE_CHANNEL_ARMS)
            acs = arm.eq(ACS_ARM)
            if (acs & ~kind.isin(ACS_HOUSEHOLD_KINDS)).any():
                raise ValueError(
                    "ACS household-type targets require observed TYPEHUGQ."
                )
            if not arm_read:
                # The ASEC arm comes from the first tag present, spine before
                # support channel, as the SPM universe producer resolves it: a
                # clone can stamp "asec" on the support channel of an ACS row
                # its spine still names.
                asec_arm = arm.eq(ASEC_ARM).to_numpy()
                arm_read = True
    in_household = ~kind.isin(ACS_GROUP_QUARTERS_KINDS).to_numpy()
    sources = np.full(len(household), ROSTER_FALLBACK, dtype=object)
    sources[acs_kind_observed] = ACS_HOUSEHOLD_KIND_COLUMN
    if ASEC_HOUSEHOLD_KIND_COLUMN in household:
        raw = household[ASEC_HOUSEHOLD_KIND_COLUMN]
        asec_kind = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & ~asec_kind.isin(ASEC_HOUSEHOLD_KINDS)).any():
            raise ValueError("Household-type targets require HRHTYPE in 1--10.")
        asec_kind_observed = raw.notna().to_numpy()
        if (asec_kind_observed & acs_kind_observed).any():
            raise ValueError(
                "Household-type targets require each household to carry at "
                "most one of TYPEHUGQ and HRHTYPE."
            )
        unclassified = ~asec_kind_observed & (asec_arm | ~acs_kind_observed)
        if unclassified.any():
            ids = household["household_id"].to_numpy()[unclassified][:5].tolist()
            raise ValueError(
                "Household-type targets require observed HRHTYPE on every ASEC "
                "household and every household without TYPEHUGQ once the frame "
                f"carries HRHTYPE; {int(unclassified.sum())} household(s) lack "
                f"it, including household ids {ids}."
            )
        in_household &= ~asec_kind.isin(ASEC_GROUP_QUARTERS_KINDS).to_numpy()
        sources[asec_kind_observed] = ASEC_HOUSEHOLD_KIND_COLUMN
    return in_household, sources


def _aligned_finite(values: Any, n_households: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_households,) or not np.isfinite(array).all():
        raise ValueError(
            f"Household-universe receipt {label} must be {n_households} finite "
            "values in household-table order."
        )
    return array


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
