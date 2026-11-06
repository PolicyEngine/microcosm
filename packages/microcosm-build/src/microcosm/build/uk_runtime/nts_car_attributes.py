"""policyengine-uk's vehicle excise duty inputs for households' own cars (microcosm#1095 B3, VED).

policyengine-uk charges vehicle excise duty on one car a person
(``car_vehicle_excise_duty``) from inputs no earlier stage wrote, so every
household paid none. The ``lcfs_consumption`` stage's declared
``assign_car_attributes_from_nts`` step writes five of them from the cars NTS
households keep (UKDS SN 5340, the cars of ``UK_NTS_CARS``):

- **Which cars.** A household's vehicles less the company cars its members
  hold (``company_car_list_price`` above zero: an employer pays their duty),
  one to each adult who holds no company car, oldest first. policyengine-uk
  holds one car a person, so a car beyond the household's adults is left out
  and counted.
- **Which fuel.** A household the fuel flag leaves without road fuel keeps
  zero-emission cars only. One that buys fuel holds a petrol, diesel or other
  car first, and each further car is zero-emission at the household's own
  per-car share (the fuel flag's), against uniforms keyed on the household
  and the car's place.
- **Which car.** Each car is a kept NTS car of its class (zero-emission or
  not) in the household's income fifth, drawn by a keyed uniform at its
  interview weight scaled to its fifth and age band's kept cars (fewer of
  the newest and oldest carry the registration record the CO2 bands come
  from): its propulsion, CO2 bands, engine size and age.
- **The inputs.** Fuel: petrol as recorded; a diesel meets RDE2 if registered
  from 2021, when every new diesel did; another propulsion is an alternative
  fuel and a zero-emission car electric. CO2: the midpoint of the car's band
  on the scale its duty regime reads, inside the engine's thresholds (the
  first-year bands from April 2017, Table A to M before). First registration:
  the declared reference date less the car's age, drawn uniformly within its
  NTS band, so each car keeps its age. Engine size as recorded, or the
  declared default where it is not; none for a zero-emission car.
  ``car_is_ved_exempt``: a member with DLA's higher-rate or PIP's enhanced
  mobility component exempts their own car or, holding none, the oldest
  car-holder's in their benefit unit. The list price keeps the engine's
  default, so no car pays the expensive-car supplement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.receipt_counts import rounded_count, support_band

ASSIGN_CAR_ATTRIBUTES_KIND = "assign_car_attributes_from_nts"
#: The policyengine-uk person inputs the step writes; ``car_list_price`` keeps
#: the engine's default.
UK_CAR_INPUT_COLUMNS = (
    "car_fuel_type",
    "car_co2_emissions",
    "car_engine_size",
    "car_first_registration_date",
    "car_is_ved_exempt",
)
#: The engine's defaults, which a person without a car keeps.
UK_CAR_INPUT_DEFAULTS: dict[str, Any] = {
    "car_fuel_type": "PETROL",
    "car_co2_emissions": 0.0,
    "car_engine_size": 0.0,
    "car_first_registration_date": "9999-01-01",
    "car_is_ved_exempt": False,
}
#: The NTS vehicle tab's columns the step reads beside the J2 and K ones.
UK_NTS_CAR_ATTRIBUTE_COLUMNS = ("vehco2tax2008_b01id", "enginecap", "vehage_b01id")
#: The midpoint of each ``VehCO2Bands_B01ID`` band (g/km), inside the engine's
#: first-year thresholds, which are right-inclusive: 0, 1-50, 51-75, 76-90,
#: 91-100, 101-110, 111-130, 131-150, 151-170, 171-190, 191-225, 226-255, over 255.
UK_FIRST_YEAR_CO2_MIDPOINTS = {
    1: 0.0,
    2: 25.0,
    3: 63.0,
    4: 83.0,
    5: 96.0,
    6: 106.0,
    7: 121.0,
    8: 141.0,
    9: 161.0,
    10: 181.0,
    11: 208.0,
    12: 241.0,
    13: 270.0,
}
#: The midpoint of each ``VehCO2Tax2008_B01ID`` band A to M (g/km), inside the
#: engine's graduated thresholds: up to 100, 101-110, 111-120, 121-130,
#: 131-140, 141-150, 151-165, 166-175, 176-185, 186-200, 201-225, 226-255, over 255.
UK_GRADUATED_CO2_MIDPOINTS = {
    1: 50.0,
    2: 106.0,
    3: 116.0,
    4: 126.0,
    5: 136.0,
    6: 146.0,
    7: 158.0,
    8: 171.0,
    9: 181.0,
    10: 193.0,
    11: 213.0,
    12: 241.0,
    13: 270.0,
}
#: ``VehAge_B01ID`` bands in years at the interview, the open top band closed
#: at the declared ``oldest_age_years``.
UK_NTS_AGE_BANDS_YEARS = {
    1: (0.0, 0.5),
    2: (0.5, 1.0),
    3: (1.0, 1.5),
    4: (1.5, 2.0),
    5: (2.0, 3.0),
    6: (3.0, 4.0),
    7: (4.0, 5.0),
    8: (5.0, 6.0),
    9: (6.0, 7.0),
    10: (7.0, 8.0),
    11: (8.0, 10.0),
    12: (10.0, 13.0),
    13: (13.0, 18.0),
    14: (18.0, None),
}
#: The duty regimes' first registration dates (policyengine-uk's own parameters).
UK_VED_MODERN_START = date(2017, 4, 1)
UK_VED_GRADUATED_START = date(2001, 3, 1)
#: Every new diesel has met RDE2 since January 2021 (as the company car stage).
UK_DIESEL_RDE2_FROM = date(2021, 1, 1)
UK_DRIVING_AGE = 17
#: The mobility components that exempt a vehicle from the duty.
UK_VED_EXEMPT_MOBILITY = {"dla_m_category": "HIGHER", "pip_m_category": "ENHANCED"}
ZERO_EMISSION = "zero_emission"
FUEL_BURNING = "fuel_burning"


@dataclass(frozen=True)
class CarAttributes:
    """The person inputs, one row a person in the person table's order, and the receipt."""

    person: pd.DataFrame
    receipt: Mapping[str, Any]


def car_attributes_operation(stage: Any) -> Mapping[str, Any] | None:
    """The stage's declared ``assign_car_attributes_from_nts`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == ASSIGN_CAR_ATTRIBUTES_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def _donor_pools(
    cars: pd.DataFrame,
) -> tuple[dict[tuple[int, str], pd.DataFrame], float]:
    """The kept, linked NTS cars by income fifth and class, and their linked share.

    A car's CO2 bands come from its registration record, which fewer of the
    newest and oldest cars carry, so each linked car stands for the kept cars
    of its fifth and age band: its interview weight is scaled by the band's
    kept cars over its linked ones.
    """

    def number(column: str) -> pd.Series:
        return pd.to_numeric(cars[column], errors="coerce")

    band = number("vehco2bands_b01id")
    graduated = number("vehco2tax2008_b01id")
    age = number("vehage_b01id")
    fuel = number("vehproptypets_b01id")
    fifth = number("hhincqisengts_b01id")
    weight = number("w3")
    kept = (
        (number("carown_b01id") == 1)
        & age.isin(UK_NTS_AGE_BANDS_YEARS)
        & fifth.isin((1, 2, 3, 4, 5))
    )
    usable = (
        kept
        & band.isin(UK_FIRST_YEAR_CO2_MIDPOINTS)
        & graduated.isin(UK_GRADUATED_CO2_MIDPOINTS)
        & fuel.isin((1, 2, 3))
    )
    kept_mass = weight[kept].groupby([fifth[kept], age[kept]]).sum()
    linked_mass = weight[usable].groupby([fifth[usable], age[usable]]).sum()
    unlinked = kept_mass.index.difference(linked_mass.index)
    if len(unlinked):
        q, a = unlinked[0]
        raise ValueError(
            f"The NTS kept cars of fifth {int(q)} and age band {int(a)} hold no "
            "car with CO2 bands."
        )
    cell = pd.MultiIndex.from_arrays([fifth[usable], age[usable]])
    donors = pd.DataFrame(
        {
            "fifth": fifth[usable].astype(int),
            "zero_emission": band[usable] == 1,
            "band": band[usable].astype(int),
            "graduated": graduated[usable].astype(int),
            "age": age[usable].astype(int),
            "fuel": fuel[usable].astype(int),
            "engine": number("enginecap")[usable],
            "weight": weight[usable]
            * (kept_mass / linked_mass).reindex(cell).to_numpy(),
        }
    ).reset_index(drop=True)
    pools = {}
    for (q, zero), group in donors.groupby(["fifth", "zero_emission"]):
        group = group.reset_index(drop=True)
        cumulative = group["weight"].cumsum().to_numpy(dtype=float)
        group["cumulative"] = cumulative / cumulative[-1]
        pools[(int(q), ZERO_EMISSION if zero else FUEL_BURNING)] = group
    for q in (1, 2, 3, 4, 5):
        for kind in (ZERO_EMISSION, FUEL_BURNING):
            if (q, kind) not in pools:
                raise ValueError(f"The NTS kept cars hold no {kind} car in fifth {q}.")
    return pools, float(linked_mass.sum() / kept_mass.sum())


def _weighted_pick(pool: pd.DataFrame, uniform: float) -> pd.Series:
    """The pool's car at ``uniform`` on its cumulative interview weight."""

    index = int(np.searchsorted(pool["cumulative"].to_numpy(), uniform, side="right"))
    return pool.iloc[min(index, len(pool) - 1)]


def _co2(registration: date, car: pd.Series, *, zero_emission: bool) -> float:
    if zero_emission:
        return 0.0
    if registration >= UK_VED_MODERN_START:
        return UK_FIRST_YEAR_CO2_MIDPOINTS[int(car["band"])]
    return UK_GRADUATED_CO2_MIDPOINTS[int(car["graduated"])]


def _fuel_type(registration: date, car: pd.Series, *, zero_emission: bool) -> str:
    if zero_emission:
        return "ELECTRIC"
    if int(car["fuel"]) == 1:
        return "PETROL"
    if int(car["fuel"]) == 2:
        return (
            "DIESEL_RDE2" if registration >= UK_DIESEL_RDE2_FROM else "DIESEL_NON_RDE2"
        )
    return "ALTERNATIVE_FUEL"


def assign_car_attributes(
    *,
    person: pd.DataFrame,
    household_ids: Sequence[object],
    vehicles: Sequence[float],
    weights: Sequence[float],
    has_fuel: Sequence[bool],
    fifths: Sequence[int],
    zero_emission_share: Sequence[float],
    cars: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> CarAttributes:
    """Give each household's own cars policyengine-uk's duty inputs (module docstring).

    ``household_ids``, ``vehicles``, ``weights``, ``has_fuel``, ``fifths`` and
    ``zero_emission_share`` are one entry per household; ``person`` carries
    ``person_id``, ``person_household_id``, ``person_benunit_id``, ``age``,
    ``company_car_list_price`` and the two mobility categories.
    """

    seed = int(parameters["seed"])
    salt = str(parameters["salt"])
    reference = date.fromisoformat(str(parameters["reference_date"]))
    oldest = float(parameters["oldest_age_years"])
    default_engine = float(parameters["default_engine_size_cc"])
    pools, linked_share = _donor_pools(cars)
    ids = np.asarray(household_ids)
    index = {household: position for position, household in enumerate(ids)}
    member = person["person_household_id"].map(index)
    if member.isna().any():
        raise ValueError("Every person must belong to a frame household.")
    company = (
        pd.to_numeric(person["company_car_list_price"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
        > 0
        if "company_car_list_price" in person
        else np.zeros(len(person), dtype=bool)
    )
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    adults = (age >= UK_DRIVING_AGE) & ~company
    count = np.rint(np.asarray(vehicles, dtype=float)).clip(min=0)
    company_cars = pd.Series(company.astype(float)).groupby(member.to_numpy()).sum()
    weight = np.asarray(weights, dtype=float)
    out = {
        column: [value] * len(person) for column, value in UK_CAR_INPUT_DEFAULTS.items()
    }
    holder = np.zeros(len(person), dtype=bool)
    tally = {"cars": 0.0, "assigned": 0.0, "dropped": 0.0, "company": 0.0}
    fuel_mass: dict[str, float] = {}
    regime_mass = {"modern": 0.0, "graduated": 0.0, "engine_size": 0.0}
    order = np.lexsort((person["person_id"].to_numpy(), -age))
    by_household = pd.Series(order).groupby(member.to_numpy()[order])
    for position, rows in by_household:
        position = int(position)
        rows = rows.to_numpy()
        own = max(count[position] - company_cars.get(position, 0.0), 0.0)
        w = weight[position]
        tally["company"] += w * min(company_cars.get(position, 0.0), count[position])
        tally["cars"] += w * own
        eligible = [row for row in rows if adults[row]]
        assigned = int(min(own, len(eligible)))
        tally["assigned"] += w * assigned
        tally["dropped"] += w * (own - assigned)
        household = ids[position]
        for place in range(assigned):
            key = f"{household}:{place}"
            if not has_fuel[position]:
                kind = ZERO_EMISSION
            elif place == 0:
                kind = FUEL_BURNING
            else:
                (u,) = stable_identity_uniforms([key], seed=seed, salt=f"{salt}:class")
                kind = (
                    ZERO_EMISSION if u < zero_emission_share[position] else FUEL_BURNING
                )
            (u_car,) = stable_identity_uniforms([key], seed=seed, salt=f"{salt}:car")
            car = _weighted_pick(pools[(int(fifths[position]), kind)], u_car)
            low, high = UK_NTS_AGE_BANDS_YEARS[int(car["age"])]
            high = oldest if high is None else high
            (u_age,) = stable_identity_uniforms([key], seed=seed, salt=f"{salt}:age")
            registration = reference - timedelta(
                days=round((low + u_age * (high - low)) * 365.25)
            )
            zero = kind == ZERO_EMISSION
            row = eligible[place]
            holder[row] = True
            fuel = _fuel_type(registration, car, zero_emission=zero)
            out["car_fuel_type"][row] = fuel
            out["car_co2_emissions"][row] = _co2(registration, car, zero_emission=zero)
            if not zero:
                engine = car["engine"]
                out["car_engine_size"][row] = (
                    float(engine) if engine > 0 else default_engine
                )
            out["car_first_registration_date"][row] = registration.isoformat()
            fuel_mass[fuel] = fuel_mass.get(fuel, 0.0) + w
            regime = (
                "modern"
                if registration >= UK_VED_MODERN_START
                else (
                    "graduated"
                    if registration >= UK_VED_GRADUATED_START
                    else "engine_size"
                )
            )
            regime_mass[regime] += w
    exempt = _exemptions(person, holder, order=order)
    out["car_is_ved_exempt"] = list(exempt)
    frame = pd.DataFrame(out, index=person.index)
    person_weight = weight[member.to_numpy(dtype=int)]
    assigned_mass = tally["assigned"]
    receipt = {
        "operation": ASSIGN_CAR_ATTRIBUTES_KIND,
        "reference_date": reference.isoformat(),
        "weighted_own_cars": tally["cars"],
        "assigned_share_of_own_cars": assigned_mass / tally["cars"]
        if tally["cars"]
        else None,
        "dropped_share_of_own_cars": tally["dropped"] / tally["cars"]
        if tally["cars"]
        else None,
        "weighted_company_cars_left_out": tally["company"],
        "fuel_type_shares": {
            fuel: mass / assigned_mass for fuel, mass in sorted(fuel_mass.items())
        }
        if assigned_mass
        else {},
        "duty_regime_shares": {
            regime: mass / assigned_mass for regime, mass in regime_mass.items()
        }
        if assigned_mass
        else {},
        "exempt_share_of_assigned_cars": (
            float(person_weight[exempt].sum() / assigned_mass)
            if assigned_mass
            else None
        ),
        "linked_share_of_kept_cars": linked_share,
        "donors": {
            f"{q}:{kind}": support_band(len(pool))
            for (q, kind), pool in sorted(pools.items())
        },
        "donor_cars": rounded_count(sum(len(pool) for pool in pools.values())),
    }
    return CarAttributes(frame, receipt)


def _exemptions(
    person: pd.DataFrame, holder: np.ndarray, *, order: np.ndarray
) -> np.ndarray:
    """One exempt car for each member with an exempting mobility component.

    Their own car where they hold one; otherwise the oldest car-holder's in
    their benefit unit who is not exempt already.
    """

    qualifies = np.zeros(len(person), dtype=bool)
    for column, value in UK_VED_EXEMPT_MOBILITY.items():
        if column in person:
            qualifies |= person[column].astype(str).to_numpy() == value
    exempt = holder & qualifies
    benunit = person["person_benunit_id"].to_numpy()
    rank = np.empty(len(person), dtype=int)
    rank[order] = np.arange(len(person))
    for row in np.flatnonzero(qualifies & ~holder):
        candidates = np.flatnonzero((benunit == benunit[row]) & holder & ~exempt)
        if len(candidates):
            exempt[candidates[np.argmin(rank[candidates])]] = True
    return exempt
