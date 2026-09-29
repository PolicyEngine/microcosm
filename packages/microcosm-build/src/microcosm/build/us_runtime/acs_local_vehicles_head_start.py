"""Native vehicle count and Head Start take-up for the ACS rows of the retained
ACS local lane (microcosm#1022).

Both reached the engine pass missing on every ACS row, and the reviewed-null
fill wrote the engine defaults: no ACS household owned a vehicle, and every
Head Start-eligible ACS child took Head Start up. The ACS local take-up stage
(:mod:`~microcosm.build.us_runtime.acs_local_take_up`) fills them with the
helpers here, without the engine, beside its other engine-free fills. Only
missing ACS cells are filled; donor-spine values and any stored ACS value are
kept. The helpers read no origin tag: the stage passes the spine masks.

- ``household_vehicles_owned`` (household) is native ACS ``VEH``, the cars,
  vans and trucks of one ton or less kept at home for use by household
  members, 0-6 with 6 meaning six or more. The donor's count
  (:mod:`~microcosm.build.us_runtime.sipp_vehicles`) is SIPP ``TVEH_NUM``,
  the cars, trucks or vans owned by the household. ``VEH`` counts vehicles
  available, so a leased or employer-provided vehicle kept at home counts and
  an owned vehicle kept elsewhere does not; otherwise the two count the same
  vehicles, and ``VEH`` is used as the owned count. ``VEH`` is a housing-unit
  item: group quarters (``TYPEHUGQ`` 2/3) have none and get 0, and a housing
  unit without a ``VEH`` code is refused, never read as no vehicle.
- ``household_vehicles_value`` is not filled. The ACS has no vehicle-value
  item, and in policyengine-us 2.2.1 its only SNAP consumer is Texas's
  broad-based categorical eligibility asset test; it stays at the reviewed
  engine default, 0 (reviewed limitation ``acs_household_vehicle_value_default``
  in the release tool).
- ``takes_up_head_start_if_eligible`` (person). The donor sets it with the
  measured SIPP model of :mod:`~microcosm.build.us_runtime.sipp_head_start`,
  a weighted QRF over ages 3-5 that is ``False`` at every other age; there is
  no manifest rate and no generic Bernoulli seeding for it. ACS persons aged
  3-5 draw a stable uniform keyed on ``acs_2024_1yr:SERIALNO:SPORDER`` and
  take up when it falls below the donor spine's weighted take-up share among
  persons aged 3-5 on the same frame: the donor model's own output, at its
  population rate, without the model's household and earnings conditioning.
  Everyone else is ``False``. No engine pre-pass is needed: the engine
  applies Head Start eligibility, as it does on the donor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.sipp_head_start import (
    _ELIGIBLE_MAX_AGE,
    _ELIGIBLE_MIN_AGE,
    _ELIGIBLE_TAKE_UP_SHARE_BAND,
    US_SIPP_HEAD_START_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.sipp_vehicles import (
    _OWNED_NONZERO_SHARE_BAND,
    US_SIPP_VEHICLE_OUTPUT_COLUMNS,
)

__all__ = [
    "ACS_LOCAL_VEHICLES_HEAD_START_ISSUE",
    "ACS_VEHICLES_AVAILABLE",
    "ACS_VEHICLES_TOP_CODE",
    "HEAD_START_AGES",
    "US_HEAD_START_TAKE_UP_COLUMN",
    "US_VEHICLES_OWNED_COLUMN",
    "US_VEHICLES_VALUE_COLUMN",
    "donor_head_start_share",
    "grade_head_start",
    "grade_vehicles_owned",
    "head_start_fill",
    "vehicles_owned_fill",
]

ACS_LOCAL_VEHICLES_HEAD_START_ISSUE = "microcosm#1022"
US_VEHICLES_OWNED_COLUMN, US_VEHICLES_VALUE_COLUMN = US_SIPP_VEHICLE_OUTPUT_COLUMNS
(US_HEAD_START_TAKE_UP_COLUMN,) = US_SIPP_HEAD_START_OUTPUT_COLUMNS
#: ACS vehicles available (housing-unit item): 0-6, 6 meaning six or more;
#: blank in group quarters and vacant units.
ACS_VEHICLES_AVAILABLE = "VEH"
ACS_VEHICLES_TOP_CODE = 6
#: The donor model's age domain; it is ``False`` at every other age.
HEAD_START_AGES = (_ELIGIBLE_MIN_AGE, _ELIGIBLE_MAX_AGE)

_OWNED = US_VEHICLES_OWNED_COLUMN
_HEAD_START = US_HEAD_START_TAKE_UP_COLUMN
_AGE = "age"
_HOUSEHOLD_KIND = "TYPEHUGQ"
_HOUSING_UNIT = 1
_HOUSEHOLD_KINDS = (1, 2, 3)
#: ACS group quarters (institutional, noninstitutional): no housing unit.
_GROUP_QUARTERS_KINDS = (2, 3)
_VEHICLE_CODES = np.arange(ACS_VEHICLES_TOP_CODE + 1, dtype=np.float64)
_HEAD_START_DRAW_KEY = f"{ACS_2024_1YR_SPINE}:SERIALNO:SPORDER"
_HEAD_START_DRAW_SALT = "acs_local_head_start_take_up"
#: The weighted ACS take-up share among ages 3-5 must stay within these
#: multiples of the donor spine's share.
_HEAD_START_SHARE_FACTORS = (0.5, 1.5)


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _boolean_cells(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flags, present, invalid)`` of a nullable boolean column."""
    present = values.notna().to_numpy(dtype=bool)
    valid = present & values.isin([0, 1]).to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[valid] = values[valid].astype(bool).to_numpy(dtype=bool)
    return flags, present, present & ~valid


def _vehicle_codes(household: pd.DataFrame) -> np.ndarray:
    """``VEH`` as numbers; NaN for a blank or a code outside 0-6."""
    codes = _numeric(household[ACS_VEHICLES_AVAILABLE])
    return np.where(np.isin(codes, _VEHICLE_CODES), codes, np.nan)


def _head_start_ages(age: np.ndarray) -> np.ndarray:
    """The donor model's domain; a blank age is outside it."""
    low, high = HEAD_START_AGES
    return (age >= low) & (age <= high)


def _keyed_uniforms(keys: Sequence[str], *, seed: int) -> np.ndarray:
    """Seeded blake2b uniforms in ``[0, 1)``, one per key."""
    denominator = float(2**64)
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{_HEAD_START_DRAW_SALT}:{key}".encode(), digest_size=8
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ),
        dtype=np.float64,
        count=len(keys),
    )


def vehicles_owned_fill(
    household: pd.DataFrame, acs_households: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Missing ACS vehicle counts, their native values, and the receipt entry.

    Args:
        household: The household table, with ``household_vehicles_owned``,
            ``VEH`` and ``TYPEHUGQ``.
        acs_households: Mask of the ACS-spine households.

    Returns:
        ``(missing, assigned, entry)``: the ACS cells to fill, the counts to
        write there (``VEH`` in a housing unit, 0 in group quarters), and a
        JSON-ready receipt entry.

    Raises:
        ValueError: If a missing ACS household has no ``TYPEHUGQ`` 1/2/3, or
            a missing ACS housing unit has no ``VEH`` code 0-6.
    """
    present = household[_OWNED].notna().to_numpy(dtype=bool)
    missing = acs_households & ~present
    kinds = _numeric(household[_HOUSEHOLD_KIND])
    unknown = missing & ~np.isin(kinds, _HOUSEHOLD_KINDS)
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} ACS household(s) have no {_HOUSEHOLD_KIND} 1/2/3; "
            "the vehicle count cannot tell a housing unit from group quarters."
        )
    group_quarters = np.isin(kinds, _GROUP_QUARTERS_KINDS)
    housing_units = kinds == _HOUSING_UNIT
    codes = _vehicle_codes(household)
    uncoded = missing & housing_units & np.isnan(codes)
    if uncoded.any():
        raise ValueError(
            f"{int(uncoded.sum())} ACS housing unit(s) have no {ACS_VEHICLES_AVAILABLE} "
            f"code 0-{ACS_VEHICLES_TOP_CODE}; a blank must not read as no vehicle."
        )
    assigned = np.where(group_quarters, 0.0, codes)
    filled_units = missing & housing_units
    raw = _numeric(household[ACS_VEHICLES_AVAILABLE])
    return (
        missing,
        assigned,
        {
            "source": (
                f"ACS {ACS_VEHICLES_AVAILABLE} (cars, vans and trucks of one ton or "
                "less kept at home for household use, leased and employer-provided "
                f"ones included), 0-{ACS_VEHICLES_TOP_CODE} with "
                f"{ACS_VEHICLES_TOP_CODE} meaning {ACS_VEHICLES_TOP_CODE} or more; 0 "
                f"in {_HOUSEHOLD_KIND} 2/3 group quarters"
            ),
            "donor_definition": (
                "SIPP TVEH_NUM, the cars, trucks or vans owned by the household "
                "(sipp_vehicles)"
            ),
            "filled_rows": int(missing.sum()),
            "preserved_rows": int((acs_households & present).sum()),
            "housing_unit_filled_rows": int(filled_units.sum()),
            "group_quarters_zero_rows": int((missing & group_quarters).sum()),
            # VEH is blank in group quarters; a code there is ignored.
            "group_quarters_coded_rows": int(
                (missing & group_quarters & ~np.isnan(raw)).sum()
            ),
            "top_coded_rows": int(
                (filled_units & (codes == ACS_VEHICLES_TOP_CODE)).sum()
            ),
            "code_counts": {
                str(int(code)): int((filled_units & (codes == code)).sum())
                for code in _VEHICLE_CODES
            },
        },
    )


def donor_head_start_share(
    values: pd.Series,
    age: np.ndarray,
    donor_persons: np.ndarray,
    weights: np.ndarray,
) -> float:
    """The donor spine's weighted Head Start take-up share among ages 3-5.

    Raises:
        ValueError: If the donor spine has no weighted person aged 3-5, a
            donor cell there is missing or not boolean, or the share leaves
            the ``sipp_head_start`` stage's own band (a donor release without
            that stage carries all ``True`` or all ``False``).
    """
    domain = donor_persons & _head_start_ages(age)
    if not domain.any() or float(weights[domain].sum()) <= 0.0:
        raise ValueError(
            "The donor spine has no weighted person aged "
            f"{HEAD_START_AGES[0]}-{HEAD_START_AGES[1]} to take the Head Start "
            "take-up share from."
        )
    flags, present, invalid = _boolean_cells(values)
    unreadable = int((domain & (~present | invalid)).sum())
    if unreadable:
        raise ValueError(
            f"{unreadable} donor-spine person(s) aged {HEAD_START_AGES[0]}-"
            f"{HEAD_START_AGES[1]} carry no boolean {_HEAD_START}; the donor "
            "release's sipp_head_start stage must have run."
        )
    share = _share(weights[domain], flags[domain])
    low, high = _ELIGIBLE_TAKE_UP_SHARE_BAND
    if not low <= share <= high:
        raise ValueError(
            f"The donor spine's weighted {_HEAD_START} share among ages "
            f"{HEAD_START_AGES[0]}-{HEAD_START_AGES[1]} is {share:.4f}, outside the "
            f"sipp_head_start stage's band [{low}, {high}]; the donor release's "
            "measured SIPP stage must have run."
        )
    return share


def head_start_fill(
    person: pd.DataFrame,
    acs_persons: np.ndarray,
    donor_persons: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    draw_keys: Callable[[np.ndarray], pd.Series],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Missing ACS Head Start take-up cells, their draws, and the receipt entry.

    Args:
        person: The person table, with ``takes_up_head_start_if_eligible``
            and ``age``.
        acs_persons: Mask of the ACS-spine persons.
        donor_persons: Mask of the donor-spine persons, whose share among ages
            3-5 is the draw rate.
        weights: Person weights aligned to ``person``.
        seed: The build seed.
        draw_keys: Returns the unique ``acs_2024_1yr:SERIALNO:SPORDER`` key
            of each person in a row mask, in row order.

    Returns:
        ``(missing, assigned, entry)``: the ACS cells to fill, the flags to
        write there, and a JSON-ready receipt entry.

    Raises:
        ValueError: If the donor share cannot be read (see
            :func:`donor_head_start_share`) or ``draw_keys`` refuses a key.
    """
    _, present, _ = _boolean_cells(person[_HEAD_START])
    missing = acs_persons & ~present
    age = _numeric(person[_AGE])
    rate = donor_head_start_share(person[_HEAD_START], age, donor_persons, weights)
    drawn = missing & _head_start_ages(age)
    assigned = np.zeros(len(person), dtype=bool)
    if drawn.any():
        keys = draw_keys(drawn)
        assigned[drawn] = _keyed_uniforms(keys.tolist(), seed=int(seed)) < rate
    low, high = _HEAD_START_SHARE_FACTORS
    return (
        missing,
        assigned,
        {
            "source": (
                f"a stable draw at ages {HEAD_START_AGES[0]}-{HEAD_START_AGES[1]} "
                "below the donor spine's weighted take-up share there (the "
                "sipp_head_start stage's output); False at every other age"
            ),
            "rate": rate,
            "rate_source": (
                f"donor-spine weighted {_HEAD_START} share among persons aged "
                f"{HEAD_START_AGES[0]}-{HEAD_START_AGES[1]} on this frame"
            ),
            "ages": list(HEAD_START_AGES),
            "draw_key": _HEAD_START_DRAW_KEY,
            "draw_salt": _HEAD_START_DRAW_SALT,
            "filled_rows": int(missing.sum()),
            "preserved_rows": int((acs_persons & present).sum()),
            "age_domain_filled_rows": int(drawn.sum()),
            "take_up_filled_rows": int(assigned.sum()),
            "share_band": [rate * low, rate * high],
        },
    )


def grade_vehicles_owned(
    household: pd.DataFrame,
    weights: np.ndarray,
    entry: dict[str, object],
    *,
    graded: bool,
) -> list[str]:
    """Grade one spine's vehicle counts; record the evidence in ``entry``.

    Every spine must carry complete, non-negative whole counts that vary.
    Only ``graded`` (ACS) rows are held to the fill: ``VEH`` in every housing
    unit, 0 in group quarters, and a weighted share of housing units with a
    vehicle inside the donor stage's band.
    """
    values = _numeric(household[_OWNED])
    present = ~np.isnan(values)
    invalid = present & ((values < 0) | (values != np.rint(values)))
    entry.update(
        missing_rows=int((~present).sum()),
        invalid_rows=int(invalid.sum()),
        unique_values=int(len(np.unique(values[present]))),
        weighted_share_with_vehicle=_share(weights, present & (values > 0)),
    )
    failures = []
    if not present.all():
        failures.append(
            f"{_OWNED} has missing rows; an engine-default fill means no household "
            "owns a vehicle."
        )
    if invalid.any():
        failures.append(
            f"{_OWNED} has {int(invalid.sum())} negative or fractional value(s)."
        )
    if entry["unique_values"] < 2:
        failures.append(
            f"{_OWNED} is constant; the engine default is the same landmine."
        )
    if not graded:
        return failures
    for column in (ACS_VEHICLES_AVAILABLE, _HOUSEHOLD_KIND):
        if column not in household:
            return [*failures, f"missing {column}; {_OWNED} cannot be checked."]
    kinds = _numeric(household[_HOUSEHOLD_KIND])
    housing_units = kinds == _HOUSING_UNIT
    group_quarters = np.isin(kinds, _GROUP_QUARTERS_KINDS)
    codes = _vehicle_codes(household)
    unknown = int((~np.isin(kinds, _HOUSEHOLD_KINDS)).sum())
    uncoded = int((housing_units & np.isnan(codes)).sum())
    differs = int((housing_units & ~np.isnan(codes) & (values != codes)).sum())
    group_quarters_owning = int((group_quarters & (values != 0)).sum())
    share = _share(weights[housing_units], (values > 0)[housing_units])
    low, high = _OWNED_NONZERO_SHARE_BAND
    entry.update(
        differs_from_native_veh=differs,
        housing_units_without_veh=uncoded,
        group_quarters_with_vehicles=group_quarters_owning,
        housing_unit_share_with_vehicle=share,
        share_band=[low, high],
    )
    if unknown:
        failures.append(f"{unknown} household(s) have no {_HOUSEHOLD_KIND} 1/2/3.")
    if uncoded:
        failures.append(
            f"{uncoded} housing unit(s) have no {ACS_VEHICLES_AVAILABLE} code "
            f"0-{ACS_VEHICLES_TOP_CODE}."
        )
    if differs:
        failures.append(
            f"{_OWNED} differs from ACS {ACS_VEHICLES_AVAILABLE} in {differs} "
            "housing unit(s)."
        )
    if group_quarters_owning:
        failures.append(
            f"{group_quarters_owning} group-quarters household(s) own vehicles."
        )
    if housing_units.any() and not low <= share <= high:
        failures.append(
            f"weighted share of housing units with a vehicle is {share:.3f}, "
            f"outside [{low}, {high}]."
        )
    return failures


def grade_head_start(
    person: pd.DataFrame,
    flags: np.ndarray,
    weights: np.ndarray,
    entry: dict[str, object],
    *,
    donor_share: float | None,
    graded: bool,
) -> list[str]:
    """Grade one spine's Head Start take-up; record the evidence in ``entry``.

    Only ``graded`` (ACS) rows are held to the fill: no one outside ages 3-5
    takes up, and the weighted share among ages 3-5 stays within half to one
    and a half times the donor spine's share. Completeness and variation are
    the caller's.
    """
    if _AGE not in person:
        return [f"missing {_AGE}; {_HEAD_START} cannot be scoped."]
    domain = _head_start_ages(_numeric(person[_AGE]))
    share = _share(weights[domain], flags[domain])
    outside = int((flags & ~domain).sum())
    entry.update(age_domain_take_up_share=share, take_up_outside_ages=outside)
    if not graded:
        return []
    failures = []
    ages = f"{HEAD_START_AGES[0]}-{HEAD_START_AGES[1]}"
    if outside:
        failures.append(f"{outside} person(s) outside ages {ages} carry {_HEAD_START}.")
    if donor_share is None:
        return [*failures, f"no donor share to grade {_HEAD_START} against."]
    low, high = (donor_share * factor for factor in _HEAD_START_SHARE_FACTORS)
    entry.update(donor_share=donor_share, share_band=[low, high])
    if not low <= share <= high:
        failures.append(
            f"{_HEAD_START} share among ages {ages} is {share:.3f}, outside "
            f"[{low:.3f}, {high:.3f}] around the donor spine's {donor_share:.3f}."
        )
    return failures
