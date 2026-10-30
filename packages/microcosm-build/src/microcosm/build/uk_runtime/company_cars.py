"""Company cars per employee, for policyengine-uk's benefits in kind.

policyengine-uk 2.165.0 (PolicyEngine/policyengine-uk#2750) taxes the company car
and car fuel benefits (ITEPA 2003 ss. 120 to 152) from nine person inputs that
default to no car. This stage, the last in the UK roster, writes six of them, so
Budget 2026's company car tax rise can be scored
(PolicyEngine/uk-autumn-budget-dashboard-2026#316):
``company_car_list_price``, ``company_car_co2_emissions``,
``company_car_fuel_type``, ``company_car_electric_range``,
``company_car_capital_contributions`` and ``company_car_private_fuel_provided``.
The other three keep the engine's defaults (a car available all year, no
payments for private use, the s. 139 rates of 2020-21 and 2021-22), since no
survey records them.

Who has a car, its list price band, fuel and private fuel, and the employee's
capital contributions come from the FRS job record, through the root stage's
carriers (``frs_spine.frs_company_car_carriers``). SPI-channel rows carry their
FRS twin's carriers, as they carry every root column. The steps, every draw keyed
on identity, so reordering rows moves none:

1. **Holders.** An FRS-channel person with a company car keeps it. An SPI-channel
   person keeps their twin's car only where their drawn EPB (expenses payments
   and benefits in kind) is positive: EPB holds the car benefit on the tape.
2. **Fuel.** The FRS fuel codes map to the engine's fuel types (a hybrid is an
   alternative fuel, a diesel is taken to meet RDE2, as the declaration says);
   a car whose fuel the respondent did not know draws one at Table 2C's shares.
3. **List price.** Within the respondent's band, a draw from WAS round-8 company
   cars' list prices, grown to the build year by the engine's CPI index; the open
   top band and an unknown band draw from the WAS cars of the same fuel class.
   A band with fewer WAS donor households than the disclosure floor is drawn
   uniformly instead.
4. **CO2.** An electric car emits none. The others are split between the 1 to
   50 g/km band (hybrids first) and the bands above it so the weighted share in
   1 to 50 g/km matches Table 2A's among cars that emit CO2, and the bands above
   it are drawn at Table 2A's shares, uniformly within each. A car at 1 to 50
   g/km takes the declared electric range.
5. **SPI pay.** SPI-channel pay is the tape's PAY + EPB + TAXTERM, so EPB already
   holds the car and fuel benefits. The engine's benefits on the kept cars come
   out of ``employment_income``, capped at EPB: private fuel goes first where its
   benefit alone exceeds EPB, and a car whose benefit still exceeds what is left
   has its list price scaled down to fit (its contributions dropped), or is
   dropped below the declared minimum price. Pay plus the benefits then equals
   the tape's total on every such row (María's ruling of 30 October 2026).

Receipts carry weighted aggregates, rounded counts and support bands only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import (
    FRS_COMPANY_CAR_CARRIER_COLUMNS,
    read_pinned_tab,
)
from microcosm.build.uk_runtime.hmrc_company_car_statistics import (
    HMRCCompanyCarStatistics,
    check_company_car_statistics_declaration,
    co2_band_bounds,
    load_hmrc_company_car_statistics,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import rounded_count, support_band
from microcosm.build.uk_runtime.spi_support import (
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

UK_COMPANY_CARS_STAGE_NAME = "company_cars"
#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it).
UK_COMPANY_CARS_MASS_CONSERVATION_REASON = (
    "Company cars per employee from the FRS job record, WAS round-8 list prices "
    "and HMRC's company car tables on the source spine: household weights pass "
    "through unchanged and total household mass is conserved."
)
UK_WAS_PERSON_TAB_ROLE = "was_person_tab"

CLEAN_WAS_COMPANY_CAR_DONOR_KIND = "clean_was_company_car_donor"
ASSIGN_COMPANY_CAR_FUEL_TYPES_KIND = "assign_company_car_fuel_types"
DRAW_COMPANY_CAR_LIST_PRICES_KIND = "draw_company_car_list_prices"
DRAW_COMPANY_CAR_CO2_EMISSIONS_KIND = "draw_company_car_co2_emissions"
CARVE_SPI_COMPANY_CAR_BENEFITS_KIND = "carve_spi_company_car_benefits"
VERIFY_VENDORED_FACT_RESOURCE_KIND = "verify_vendored_fact_resource"
CLASSIFY_COMPANY_CAR_FACTS_KIND = "classify_company_car_facts_with_reviewed_fence"

#: The reviewed fence over every vendored company car fact (the manifest's
#: ``classify_company_car_facts_with_reviewed_fence`` operation repeats it). The
#: fact count is the vendored resource's, set when the feed carries it.
UK_COMPANY_CAR_FACT_FENCE: dict[str, Any] = {
    "calibration_permitted": False,
    "fact_fence_id": "company_car_facts_imputation_seeds_measured_only",
    "fenced_fact_composition": (
        "HMRC Benefits in kind statistics (July 2026): Table 1 car and car fuel "
        "benefit recipients, taxable value, income tax and NICs; Table 2A "
        "recipients by CO2 band; Table 2C recipients by fuel; Tables 3A and 3B "
        "recipients and taxable value by income range; Table 4 recipients by region"
    ),
    "classification_rationale": (
        "The stage draws fuel and CO2 to Tables 2A and 2C, so binding them in the "
        "weight solve would be circular. The national build reports the calibrated "
        "file against Tables 1, 3A and 4; none blocks a build."
    ),
    "adjudication": (
        "https://github.com/PolicyEngine/uk-autumn-budget-dashboard-2026/issues/316"
    ),
}

#: The root stage's carriers (frs_spine.frs_company_car_carriers).
CARRIER_COLUMNS = FRS_COMPANY_CAR_CARRIER_COLUMNS
(
    PRICE_BAND_CARRIER,
    FUEL_CODE_CARRIER,
    PRIVATE_FUEL_CARRIER,
    CONTRIBUTION_CARRIER,
) = FRS_COMPANY_CAR_CARRIER_COLUMNS

LIST_PRICE = "company_car_list_price"
CO2_EMISSIONS = "company_car_co2_emissions"
FUEL_TYPE = "company_car_fuel_type"
ELECTRIC_RANGE = "company_car_electric_range"
CAPITAL_CONTRIBUTIONS = "company_car_capital_contributions"
PRIVATE_FUEL = "company_car_private_fuel_provided"
UK_COMPANY_CARS_OUTPUT_COLUMNS = (
    LIST_PRICE,
    CO2_EMISSIONS,
    FUEL_TYPE,
    ELECTRIC_RANGE,
    CAPITAL_CONTRIBUTIONS,
    PRIVATE_FUEL,
)
UK_COMPANY_CARS_NONNEGATIVE_OUTPUT_COLUMNS = (
    LIST_PRICE,
    CO2_EMISSIONS,
    ELECTRIC_RANGE,
    CAPITAL_CONTRIBUTIONS,
)
EMPLOYMENT_INCOME = "employment_income"
UK_COMPANY_CARS_REWRITES = (EMPLOYMENT_INCOME,)
#: The engine variables the SPI carve-out reads.
COMPANY_CAR_BENEFIT = "company_car_benefit"
CAR_FUEL_BENEFIT = "car_fuel_benefit"

#: The FRS list price bands (CARVAL 1 to 9, GBP, the price when new): each is
#: (above, up to); the top band is open. CARVAL 10 is "don't know".
FRS_PRICE_BANDS: dict[int, tuple[float, float | None]] = {
    1: (0.0, 10_000.0),
    2: (10_000.0, 13_000.0),
    3: (13_000.0, 16_000.0),
    4: (16_000.0, 19_000.0),
    5: (19_000.0, 22_000.0),
    6: (22_000.0, 25_000.0),
    7: (25_000.0, 30_000.0),
    8: (30_000.0, 40_000.0),
    9: (40_000.0, None),
}
UNKNOWN_PRICE_BAND = 10
OPEN_PRICE_BAND = 9

#: The FRS fuel codes (FUELTYP), and 6, unknown.
FRS_FUEL_PETROL = 1
FRS_FUEL_DIESEL = 2
FRS_FUEL_HYBRID = 3
FRS_FUEL_ELECTRIC = 4
FRS_FUEL_OTHER = 5
FRS_FUEL_UNKNOWN = 6

#: policyengine-uk's ``CarFuelType`` names.
PETROL = "PETROL"
DIESEL_RDE2 = "DIESEL_RDE2"
DIESEL_NON_RDE2 = "DIESEL_NON_RDE2"
ALTERNATIVE_FUEL = "ALTERNATIVE_FUEL"
ELECTRIC = "ELECTRIC"
ENGINE_FUEL_TYPES = (PETROL, DIESEL_RDE2, DIESEL_NON_RDE2, ALTERNATIVE_FUEL, ELECTRIC)
#: The engine's default, written for people without a car.
DEFAULT_FUEL_TYPE = PETROL

#: No list price cell may rest on fewer WAS households than CD171's absolute
#: floor; the declared minimum is the modelling choice above it.
DONOR_HOUSEHOLD_FLOOR = 3

CPI_INDEX = "gov.economic_assumptions.indices.obr.consumer_price_index"
CAPITAL_CONTRIBUTION_CAP = (
    "gov.hmrc.income_tax.benefits_in_kind.company_car.capital_contribution_cap"
)


class CompanyCarsError(ValueError):
    """Raised when the inputs or the declaration do not carry what the stage needs."""


# --------------------------------------------------------------------------
# Declaration
# --------------------------------------------------------------------------


def _operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any]:
    for operation in stage.operations:
        if operation.kind == kind:
            return dict(operation.parameters)
    raise CompanyCarsError(f"{stage.stage} declares no {kind!r} operation.")


def _artifact(stage: SourceStageSpec, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise CompanyCarsError(f"{stage.stage} declares no {role!r} artifact.")


@dataclass(frozen=True)
class WASCompanyCarColumns:
    """The declared WAS round-8 person-tab names and codes the donor reads."""

    household_key: str
    household_weight: str
    has_company_vehicle: str
    car_or_van: tuple[str, ...]
    fuel: tuple[str, ...]
    list_price: tuple[str, ...]
    yes_code: int
    car_code: int
    electric_code: int

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> WASCompanyCarColumns:
        vehicles = parameters.get("vehicles")
        if not isinstance(vehicles, Sequence) or not vehicles:
            raise CompanyCarsError(
                f"{CLEAN_WAS_COMPANY_CAR_DONOR_KIND} needs a vehicles list."
            )
        try:
            return cls(
                household_key=str(parameters["household_key"]),
                household_weight=str(parameters["household_weight"]),
                has_company_vehicle=str(parameters["has_company_vehicle"]),
                car_or_van=tuple(str(vehicle["car_or_van"]) for vehicle in vehicles),
                fuel=tuple(str(vehicle["fuel"]) for vehicle in vehicles),
                list_price=tuple(str(vehicle["list_price"]) for vehicle in vehicles),
                yes_code=int(parameters["yes_code"]),
                car_code=int(parameters["car_code"]),
                electric_code=int(parameters["electric_code"]),
            )
        except (KeyError, TypeError) as error:
            raise CompanyCarsError(
                f"{CLEAN_WAS_COMPANY_CAR_DONOR_KIND} lacks {error.args[0]!r}."
            ) from None

    def raw_columns(self) -> tuple[str, ...]:
        return (
            self.household_key,
            self.household_weight,
            self.has_company_vehicle,
            *self.car_or_van,
            *self.fuel,
            *self.list_price,
        )


@dataclass(frozen=True)
class CompanyCarsSpec:
    """Every declared parameter of the stage, read once from the manifest."""

    columns: WASCompanyCarColumns
    donor_price_year: int
    diesel_fuel_type: str
    fuel_seed: int
    fuel_salt: str
    price_seed: int
    price_salt: str
    minimum_donor_households: int
    open_band_fallback_upper_gbp: float
    co2_seed: int
    co2_salt: str
    top_band_upper_g_km: int
    electric_range_miles: float
    minimum_scaled_list_price_gbp: float

    @classmethod
    def from_stage(cls, stage: SourceStageSpec) -> CompanyCarsSpec:
        donor = _operation(stage, CLEAN_WAS_COMPANY_CAR_DONOR_KIND)
        fuel = _operation(stage, ASSIGN_COMPANY_CAR_FUEL_TYPES_KIND)
        prices = _operation(stage, DRAW_COMPANY_CAR_LIST_PRICES_KIND)
        co2 = _operation(stage, DRAW_COMPANY_CAR_CO2_EMISSIONS_KIND)
        carve = _operation(stage, CARVE_SPI_COMPANY_CAR_BENEFITS_KIND)
        fence = dict(_operation(stage, CLASSIFY_COMPANY_CAR_FACTS_KIND))
        fence.pop("fenced_fact_count", None)
        if fence != UK_COMPANY_CAR_FACT_FENCE:
            raise CompanyCarsError(
                "the declared company car fact fence differs from the reviewed one: "
                "no company car fact is a calibration target."
            )
        for operation, outputs in (
            (fuel, [FUEL_TYPE]),
            (prices, [LIST_PRICE, CAPITAL_CONTRIBUTIONS]),
            (co2, [CO2_EMISSIONS, ELECTRIC_RANGE]),
            (carve, [EMPLOYMENT_INCOME, PRIVATE_FUEL]),
        ):
            if list(operation.get("outputs", ())) != outputs:
                raise CompanyCarsError(
                    f"the declaration writes {operation.get('outputs')!r}, the stage "
                    f"writes {outputs!r}."
                )
        diesel = str(fuel["diesel_fuel_type"])
        if diesel not in (DIESEL_RDE2, DIESEL_NON_RDE2):
            raise CompanyCarsError(
                f"diesel_fuel_type must be {DIESEL_RDE2} or {DIESEL_NON_RDE2}."
            )
        if fuel.get("unknown_fuel") != "table_2c_shares":
            raise CompanyCarsError("an unknown fuel draws at Table 2C's shares.")
        bands = {int(band): tuple(bounds) for band, bounds in prices["bands"].items()}
        expected_bands = {
            band: (low, high) for band, (low, high) in FRS_PRICE_BANDS.items()
        }
        if bands != expected_bands:
            raise CompanyCarsError(
                f"the declared list price bands {bands} are not the FRS's CARVAL "
                f"bands {expected_bands}."
            )
        if co2.get("margins") != "table_2a_nonzero_shares_hybrids_first":
            raise CompanyCarsError(
                "CO2 is drawn to Table 2A's shares among cars that emit it, "
                "hybrids first into 1 to 50 g/km."
            )
        if carve.get("epb_column") != SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN or list(
            carve.get("benefits", ())
        ) != [COMPANY_CAR_BENEFIT, CAR_FUEL_BENEFIT]:
            raise CompanyCarsError(
                "the SPI carve-out takes the engine's company car and car fuel "
                f"benefits out of pay, capped at {SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN}."
            )
        minimum_households = int(prices["minimum_donor_households"])
        if minimum_households < DONOR_HOUSEHOLD_FLOOR:
            raise CompanyCarsError(
                f"a donor cell needs at least {DONOR_HOUSEHOLD_FLOOR} households."
            )
        return cls(
            columns=WASCompanyCarColumns.from_parameters(donor),
            donor_price_year=int(donor["price_year"]),
            diesel_fuel_type=diesel,
            fuel_seed=int(fuel["seed"]),
            fuel_salt=str(fuel["salt"]),
            price_seed=int(prices["seed"]),
            price_salt=str(prices["salt"]),
            minimum_donor_households=minimum_households,
            open_band_fallback_upper_gbp=float(prices["open_band_fallback_upper_gbp"]),
            co2_seed=int(co2["seed"]),
            co2_salt=str(co2["salt"]),
            top_band_upper_g_km=int(co2["top_band_upper_g_km"]),
            electric_range_miles=float(co2["electric_range_miles"]),
            minimum_scaled_list_price_gbp=float(carve["minimum_scaled_list_price_gbp"]),
        )


# --------------------------------------------------------------------------
# Donor: WAS round-8 company cars' list prices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WASCompanyCarDonor:
    """One row per WAS company car with a list price, in the donor's money."""

    price: np.ndarray
    is_electric: np.ndarray
    weight: np.ndarray
    household: np.ndarray
    receipt: Mapping[str, Any]

    def grown(self, factor: float) -> WASCompanyCarDonor:
        if not factor > 0:
            raise CompanyCarsError("the list price growth factor must be positive.")
        return WASCompanyCarDonor(
            price=self.price * float(factor),
            is_electric=self.is_electric,
            weight=self.weight,
            household=self.household,
            receipt={**self.receipt, "growth_factor": float(factor)},
        )


def _lowered(table: pd.DataFrame) -> dict[str, str]:
    lowered: dict[str, str] = {}
    for column in table.columns:
        key = str(column).lower()
        if key in lowered:
            raise CompanyCarsError(f"the WAS person tab repeats column {key!r}.")
        lowered[key] = column
    return lowered


def _numeric(table: pd.DataFrame, lowered: Mapping[str, str], name: str) -> np.ndarray:
    actual = lowered.get(name.lower())
    if actual is None:
        raise CompanyCarsError(
            f"the WAS person tab lacks the declared column {name!r}."
        )
    return pd.to_numeric(table[actual], errors="coerce").to_numpy(dtype=float)


def clean_was_company_car_donor(
    person_raw: pd.DataFrame, *, columns: WASCompanyCarColumns
) -> WASCompanyCarDonor:
    """The WAS round-8 company cars (not vans) with a stated list price.

    A person may have up to three company vehicles; each car with a positive list
    price is one donor row, weighted by its household's cross-sectional weight.
    """

    lowered = _lowered(person_raw)
    has_vehicle = _numeric(person_raw, lowered, columns.has_company_vehicle)
    weight = _numeric(person_raw, lowered, columns.household_weight)
    household = person_raw[lowered[columns.household_key.lower()]].to_numpy()
    prices: list[np.ndarray] = []
    electric: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    households: list[np.ndarray] = []
    for kind, fuel, price in zip(
        columns.car_or_van, columns.fuel, columns.list_price, strict=True
    ):
        car_or_van = _numeric(person_raw, lowered, kind)
        list_price = _numeric(person_raw, lowered, price)
        fuel_code = _numeric(person_raw, lowered, fuel)
        keep = (
            (has_vehicle == columns.yes_code)
            & (car_or_van == columns.car_code)
            & np.isfinite(list_price)
            & (list_price > 0)
            & np.isfinite(weight)
            & (weight > 0)
        )
        prices.append(list_price[keep])
        electric.append(fuel_code[keep] == columns.electric_code)
        weights.append(weight[keep])
        households.append(household[keep])
    price_values = np.concatenate(prices)
    if price_values.size == 0:
        raise CompanyCarsError("the WAS person tab has no company car with a price.")
    is_electric = np.concatenate(electric)
    household_values = np.concatenate(households)
    receipt = {
        "cars": rounded_count(int(price_values.size)),
        "households": support_band(len(set(household_values.tolist()))),
        "electric_households": support_band(
            len(set(household_values[is_electric].tolist()))
        ),
    }
    return WASCompanyCarDonor(
        price=price_values,
        is_electric=is_electric,
        weight=np.concatenate(weights),
        household=household_values,
        receipt=receipt,
    )


# --------------------------------------------------------------------------
# The population
# --------------------------------------------------------------------------


def holder_table(frame: Frame) -> pd.DataFrame:
    """One row per person: identity, weight, channel, carriers, EPB and pay."""

    person = frame.table("person")
    household = frame.table("household")
    missing = [
        column
        for column in ("person_id", "person_household_id", EMPLOYMENT_INCOME)
        + CARRIER_COLUMNS
        if column not in person
    ]
    if missing:
        raise CompanyCarsError(
            f"the person table lacks {missing} (frs_spine writes the carriers)."
        )
    weights = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    household_ids = person["person_household_id"].to_numpy()
    channel_column = support_channel_column("person")
    spi = (
        person[channel_column].astype(str).eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
        if channel_column in person
        else np.zeros(len(person), dtype=bool)
    )
    if spi.any() and SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN not in person:
        raise CompanyCarsError(
            f"SPI-channel rows lack {SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN!r}."
        )
    epb = (
        pd.to_numeric(person[SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        if SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN in person
        else np.zeros(len(person))
    )

    def _number(column: str) -> np.ndarray:
        return (
            pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy(float)
        )

    return pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "household_id": household_ids,
            "weight": weights.reindex(household_ids).to_numpy(float),
            "spi": spi,
            "price_band": _number(PRICE_BAND_CARRIER).astype(int),
            "fuel_code": _number(FUEL_CODE_CARRIER).astype(int),
            "private_fuel": person[PRIVATE_FUEL_CARRIER].astype(bool).to_numpy(),
            "contribution": np.maximum(_number(CONTRIBUTION_CARRIER), 0.0),
            "epb": np.where(spi, epb, 0.0),
            "employment_income": _number(EMPLOYMENT_INCOME),
        },
        index=person.index,
    )


def _uniforms(ids: np.ndarray, seed: int, salt: str) -> np.ndarray:
    return np.asarray(
        stable_identity_uniforms(list(ids), seed=seed, salt=salt), dtype=float
    )


# --------------------------------------------------------------------------
# Step 2: fuel
# --------------------------------------------------------------------------


def assign_fuel_types(
    codes: np.ndarray,
    ids: np.ndarray,
    *,
    statistics: HMRCCompanyCarStatistics,
    spec: CompanyCarsSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """The engine fuel type of each car, and whether it was drawn.

    ``codes`` holds the FRS fuel codes of the holders only.
    """

    codes = np.asarray(codes, dtype=int)
    mapped = {
        FRS_FUEL_PETROL: PETROL,
        FRS_FUEL_DIESEL: spec.diesel_fuel_type,
        FRS_FUEL_HYBRID: ALTERNATIVE_FUEL,
        FRS_FUEL_ELECTRIC: ELECTRIC,
        FRS_FUEL_OTHER: ALTERNATIVE_FUEL,
    }
    fuel = np.array([mapped.get(int(code), "") for code in codes], dtype=object)
    drawn = fuel == ""
    if drawn.any():
        shares = statistics.fuel_type_shares()
        cumulative = np.cumsum([shares["electric"], shares["petrol"], shares["diesel"]])
        names = np.array([ELECTRIC, PETROL, spec.diesel_fuel_type], dtype=object)
        uniforms = _uniforms(ids[drawn], spec.fuel_seed, f"{spec.fuel_salt}:fuel")
        index = np.searchsorted(cumulative / cumulative[-1], uniforms, side="right")
        fuel[drawn] = names[np.minimum(index, len(names) - 1)]
    return fuel, drawn


# --------------------------------------------------------------------------
# Step 3: list price
# --------------------------------------------------------------------------


class _NoDonorError(Exception):
    """A list price cell with fewer donor households than the declared minimum."""


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    """The weighted empirical quantile function, linear between donor values."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative = cumulative / sorted_weights.sum()
    return np.interp(uniforms, cumulative, sorted_values)


def draw_list_prices(
    bands: np.ndarray,
    electric: np.ndarray,
    ids: np.ndarray,
    *,
    donor: WASCompanyCarDonor,
    spec: CompanyCarsSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Each holder's list price, in the build year's money.

    ``donor`` is already grown to the build year. Bands 1 to 8 draw from the donor
    cars inside the band; the open band and an unknown band from the donor cars of
    the holder's fuel class (above GBP 40,000 for the open band), falling back to
    every donor car, then to a uniform draw, when a cell has fewer donor households
    than the declared minimum.
    """

    bands = np.asarray(bands, dtype=int)
    electric = np.asarray(electric, dtype=bool)
    uniforms = _uniforms(ids, spec.price_seed, f"{spec.price_salt}:price")
    prices = np.zeros(len(bands), dtype=float)
    routes: dict[str, int] = {}

    def _households(mask: np.ndarray) -> int:
        return len(set(donor.household[mask].tolist()))

    def _draw(rows: np.ndarray, candidates: Sequence[np.ndarray], route: str) -> None:
        for level, mask in enumerate(candidates):
            if _households(mask) >= spec.minimum_donor_households:
                prices[rows] = _weighted_quantile(
                    donor.price[mask], donor.weight[mask], uniforms[rows]
                )
                key = f"{route}:donor_level_{level}"
                routes[key] = routes.get(key, 0) + int(rows.sum())
                return
        raise _NoDonorError(route)

    for band, (low, high) in FRS_PRICE_BANDS.items():
        rows = bands == band
        if not rows.any():
            continue
        if band == OPEN_PRICE_BAND:
            for is_electric in (True, False):
                class_rows = rows & (electric == is_electric)
                if not class_rows.any():
                    continue
                above = donor.price > low
                try:
                    _draw(
                        class_rows,
                        (above & (donor.is_electric == is_electric), above),
                        f"band_{band}",
                    )
                except _NoDonorError:
                    upper = spec.open_band_fallback_upper_gbp
                    prices[class_rows] = low + uniforms[class_rows] * (upper - low)
                    routes[f"band_{band}:uniform"] = routes.get(
                        f"band_{band}:uniform", 0
                    ) + int(class_rows.sum())
            continue
        inside = (donor.price > low) & (donor.price <= float(high))
        try:
            _draw(rows, (inside,), f"band_{band}")
        except _NoDonorError:
            prices[rows] = low + uniforms[rows] * (float(high) - low)
            routes[f"band_{band}:uniform"] = routes.get(
                f"band_{band}:uniform", 0
            ) + int(rows.sum())
    unknown = bands == UNKNOWN_PRICE_BAND
    for is_electric in (True, False):
        class_rows = unknown & (electric == is_electric)
        if not class_rows.any():
            continue
        everyone = np.ones(len(donor.price), dtype=bool)
        try:
            _draw(
                class_rows,
                (donor.is_electric == is_electric, everyone),
                f"band_{UNKNOWN_PRICE_BAND}",
            )
        except _NoDonorError:
            raise CompanyCarsError(
                f"fewer than {spec.minimum_donor_households} WAS donor households "
                "carry a company car list price."
            ) from None
    if (prices <= 0).any() or not np.isfinite(prices).all():
        raise CompanyCarsError("a holder drew no positive list price.")
    return prices, {
        "routes": {
            route: rounded_count(count) for route, count in sorted(routes.items())
        }
    }


# --------------------------------------------------------------------------
# Step 4: CO2
# --------------------------------------------------------------------------


def draw_co2_emissions(
    fuel: np.ndarray,
    hybrid: np.ndarray,
    weights: np.ndarray,
    ids: np.ndarray,
    *,
    statistics: HMRCCompanyCarStatistics,
    spec: CompanyCarsSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Each car's CO2 figure (g/km) and electric range (miles)."""

    fuel = np.asarray(fuel, dtype=object)
    hybrid = np.asarray(hybrid, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    emits = fuel != ELECTRIC
    shares = statistics.nonzero_co2_shares()
    low_share = shares["1_50"]
    hybrid_mass = float(weights[emits & hybrid].sum())
    other_mass = float(weights[emits & ~hybrid].sum())
    target = low_share * (hybrid_mass + other_mass)
    if hybrid_mass >= target:
        hybrid_rate = target / hybrid_mass if hybrid_mass > 0 else 0.0
        other_rate = 0.0
    else:
        hybrid_rate = 1.0
        other_rate = (target - hybrid_mass) / other_mass if other_mass > 0 else 0.0
    rate = np.where(hybrid, hybrid_rate, other_rate)
    low = emits & (_uniforms(ids, spec.co2_seed, f"{spec.co2_salt}:low") < rate)
    position = _uniforms(ids, spec.co2_seed, f"{spec.co2_salt}:g_km")
    co2 = np.zeros(len(fuel), dtype=float)
    co2[low] = 1.0 + np.floor(position[low] * 50.0)
    above_bands = [band for band in shares if band != "1_50"]
    above_shares = np.array([shares[band] for band in above_bands], dtype=float)
    cumulative = np.cumsum(above_shares) / above_shares.sum()
    above = emits & ~low
    band_index = np.minimum(
        np.searchsorted(
            cumulative,
            _uniforms(ids[above], spec.co2_seed, f"{spec.co2_salt}:band"),
            side="right",
        ),
        len(above_bands) - 1,
    )
    floors = np.empty(int(above.sum()), dtype=float)
    widths = np.empty(int(above.sum()), dtype=float)
    for index, band in enumerate(above_bands):
        lowest, highest = co2_band_bounds(band)
        top = spec.top_band_upper_g_km if highest is None else highest
        rows = band_index == index
        floors[rows] = lowest
        widths[rows] = top - lowest + 1
    co2[above] = floors + np.floor(position[above] * widths)
    electric_range = np.where(low, spec.electric_range_miles, 0.0)
    emitting_mass = float(weights[emits].sum())
    achieved = float(weights[low].sum()) / emitting_mass if emitting_mass > 0 else None
    drawn_shares = {}
    for index, band in enumerate(above_bands):
        mass = float(weights[above][band_index == index].sum())
        drawn_shares[band] = mass / emitting_mass if emitting_mass > 0 else None
    return (
        co2,
        electric_range,
        {
            "hybrid_rate_into_1_50": hybrid_rate,
            "other_rate_into_1_50": other_rate,
            "table_2a_share_1_50": low_share,
            "weighted_share_1_50": achieved,
            "max_abs_band_share_gap": max(
                abs((drawn_shares[band] or 0.0) - shares[band]) for band in above_bands
            ),
        },
    )


# --------------------------------------------------------------------------
# Step 5: the SPI carve-out
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SPICarveOut:
    """What the carve-out leaves on the SPI-channel holders' rows."""

    keep: np.ndarray
    private_fuel: np.ndarray
    list_price: np.ndarray
    contributions: np.ndarray
    benefit: np.ndarray
    receipt: Mapping[str, Any]


def carve_spi_benefits(
    *,
    epb: np.ndarray,
    pay: np.ndarray,
    weights: np.ndarray,
    list_price: np.ndarray,
    contributions: np.ndarray,
    private_fuel: np.ndarray,
    car_benefit: np.ndarray,
    fuel_benefit: np.ndarray,
    contribution_cap: float,
    spec: CompanyCarsSpec,
) -> SPICarveOut:
    """Fit each SPI-channel car's benefits inside its EPB and its pay.

    Every array holds the SPI-channel holders only. The engine's car benefit is
    the price less contributions up to the cap, times the appropriate percentage
    (the stage leaves availability and private-use payments at their defaults), so
    scaling the price above zero contributions scales the benefit exactly.
    """

    epb = np.asarray(epb, dtype=float)
    pay = np.asarray(pay, dtype=float)
    budget = np.minimum(np.maximum(epb, 0.0), np.maximum(pay, 0.0))
    car = np.asarray(car_benefit, dtype=float)
    fuel = np.where(private_fuel, np.asarray(fuel_benefit, dtype=float), 0.0)
    fuel_dropped = fuel > budget
    private = np.asarray(private_fuel, dtype=bool) & ~fuel_dropped
    fuel = np.where(fuel_dropped, 0.0, fuel)
    room = budget - fuel
    scaled = car > room
    price = np.asarray(list_price, dtype=float).copy()
    contribution = np.asarray(contributions, dtype=float).copy()
    interim = np.maximum(price - np.minimum(contribution, contribution_cap), 0.0)
    rate = np.divide(car, interim, out=np.zeros_like(car), where=interim > 0)
    new_price = np.divide(room, rate, out=np.zeros_like(room), where=rate > 0)
    dropped = scaled & (new_price < spec.minimum_scaled_list_price_gbp)
    kept_scaled = scaled & ~dropped
    price[kept_scaled] = new_price[kept_scaled]
    contribution[kept_scaled] = 0.0
    car = np.where(kept_scaled, room, car)
    keep = ~dropped
    private &= keep
    benefit = np.where(keep, car + np.where(private, fuel, 0.0), 0.0)
    receipt = {
        "holders": rounded_count(int(len(epb))),
        "private_fuel_dropped": rounded_count(int((fuel_dropped & keep).sum())),
        "list_price_scaled": rounded_count(int(kept_scaled.sum())),
        "dropped_below_minimum_price": rounded_count(int(dropped.sum())),
        "weighted_benefit_carved_gbp": float((np.asarray(weights) * benefit).sum()),
        "weighted_epb_gbp": float((np.asarray(weights) * epb).sum()),
    }
    return SPICarveOut(
        keep=keep,
        private_fuel=private,
        list_price=price,
        contributions=contribution,
        benefit=benefit,
        receipt=receipt,
    )


# --------------------------------------------------------------------------
# The whole stage on one frame
# --------------------------------------------------------------------------

#: Computes (company_car_benefit, car_fuel_benefit) per person on a frame that
#: carries the stage's inputs, at the frame's period.
BenefitCalculator = Callable[[Frame], tuple[np.ndarray, np.ndarray]]


def _frame_with(frame: Frame, person: pd.DataFrame, reason: str | None) -> Frame:
    mass_log = tuple(frame.mass_log)
    if reason is not None:
        mass_log = (*mass_log, uk_household_mass_conservation_receipt(frame, reason))
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=np.asarray(
            frame.weights_for("household").values, dtype=float
        ),
        mass_log=mass_log,
    )
    validate_uk_national_frame(result)
    return result


def impute_company_cars(
    frame: Frame,
    *,
    donor: WASCompanyCarDonor,
    statistics: HMRCCompanyCarStatistics,
    spec: CompanyCarsSpec,
    benefits: BenefitCalculator,
    contribution_cap: float,
) -> tuple[Frame, dict[str, Any]]:
    """Write the six inputs, carve the SPI benefits out of pay, and receipt it."""

    table = holder_table(frame)
    ids = table["person_id"].to_numpy()
    weights = table["weight"].to_numpy(float)
    spi = table["spi"].to_numpy(bool)
    reported = table["price_band"].to_numpy(int) > 0
    no_epb = reported & spi & (table["epb"].to_numpy(float) <= 0.0)
    holders = reported & ~no_epb

    fuel_codes = table["fuel_code"].to_numpy(int)
    fuel, drawn = assign_fuel_types(
        fuel_codes[holders], ids[holders], statistics=statistics, spec=spec
    )
    hybrid = fuel_codes[holders] == FRS_FUEL_HYBRID
    electric = fuel == ELECTRIC
    prices, price_receipt = draw_list_prices(
        table["price_band"].to_numpy(int)[holders],
        electric,
        ids[holders],
        donor=donor,
        spec=spec,
    )
    contributions = np.minimum(table["contribution"].to_numpy(float)[holders], prices)
    co2, electric_range, co2_receipt = draw_co2_emissions(
        fuel, hybrid, weights[holders], ids[holders], statistics=statistics, spec=spec
    )
    private = table["private_fuel"].to_numpy(bool)[holders] & ~electric

    person = frame.table("person").copy()
    columns = {
        LIST_PRICE: np.zeros(len(person)),
        CO2_EMISSIONS: np.zeros(len(person)),
        FUEL_TYPE: np.full(len(person), DEFAULT_FUEL_TYPE, dtype=object),
        ELECTRIC_RANGE: np.zeros(len(person)),
        CAPITAL_CONTRIBUTIONS: np.zeros(len(person)),
        PRIVATE_FUEL: np.zeros(len(person), dtype=bool),
    }
    columns[LIST_PRICE][holders] = prices
    columns[CO2_EMISSIONS][holders] = co2
    columns[FUEL_TYPE][holders] = fuel
    columns[ELECTRIC_RANGE][holders] = electric_range
    columns[CAPITAL_CONTRIBUTIONS][holders] = contributions
    columns[PRIVATE_FUEL][holders] = private
    for name, values in columns.items():
        person[name] = values

    pay = table["employment_income"].to_numpy(float)
    new_pay = pay.copy()
    carved = np.zeros(len(pay))
    spi_holders = holders & spi
    carve_receipt: dict[str, Any] = {"holders": rounded_count(0)}
    identity_error = 0.0
    if spi_holders.any():
        car_benefit, fuel_benefit = benefits(_frame_with(frame, person, None))
        rows = np.flatnonzero(spi_holders)
        carve = carve_spi_benefits(
            epb=table["epb"].to_numpy(float)[rows],
            pay=pay[rows],
            weights=weights[rows],
            list_price=columns[LIST_PRICE][rows],
            contributions=columns[CAPITAL_CONTRIBUTIONS][rows],
            private_fuel=columns[PRIVATE_FUEL][rows],
            car_benefit=np.asarray(car_benefit, dtype=float)[rows],
            fuel_benefit=np.asarray(fuel_benefit, dtype=float)[rows],
            contribution_cap=contribution_cap,
            spec=spec,
        )
        dropped = rows[~carve.keep]
        columns[LIST_PRICE][rows] = np.where(carve.keep, carve.list_price, 0.0)
        columns[CAPITAL_CONTRIBUTIONS][rows] = np.where(
            carve.keep, carve.contributions, 0.0
        )
        for name in (CO2_EMISSIONS, ELECTRIC_RANGE):
            columns[name][rows] = np.where(carve.keep, columns[name][rows], 0.0)
        columns[FUEL_TYPE][dropped] = DEFAULT_FUEL_TYPE
        columns[PRIVATE_FUEL][rows] = carve.private_fuel
        new_pay[rows] = pay[rows] - carve.benefit
        carved[rows] = carve.benefit
        identity_error = float(
            np.max(np.abs(new_pay[rows] + carve.benefit - pay[rows]), initial=0.0)
        )
        holders[dropped] = False
        carve_receipt = dict(carve.receipt)
        for name, values in columns.items():
            person[name] = values
        person[EMPLOYMENT_INCOME] = new_pay

    weighted = weights[holders]
    receipt = {
        "holders": {
            "frs_channel": rounded_count(int((holders & ~spi).sum())),
            "spi_channel": rounded_count(int((holders & spi).sum())),
            "spi_dropped_without_epb": rounded_count(int(no_epb.sum())),
            "weighted": float(weighted.sum()),
        },
        "fuel": {
            "drawn_unknown": rounded_count(int(drawn.sum())),
            "weighted_shares": {
                name: float(
                    weights[holders][columns[FUEL_TYPE][holders] == name].sum()
                    / weighted.sum()
                )
                if weighted.sum() > 0
                else None
                for name in ENGINE_FUEL_TYPES
            },
        },
        "list_price": {
            **price_receipt,
            "weighted_mean_gbp": float(
                (weighted * columns[LIST_PRICE][holders]).sum() / weighted.sum()
            )
            if weighted.sum() > 0
            else None,
        },
        "co2": co2_receipt,
        "spi_carve_out": {
            **carve_receipt,
            "max_abs_pay_plus_benefit_error_gbp": identity_error,
        },
        "checks": {
            "violations": _violations(
                columns,
                holders=holders,
                spi=spi,
                epb=table["epb"].to_numpy(float),
                carved=carved,
            ),
            "frs_holders_without_pay": rounded_count(
                int((holders & ~spi & (pay <= 0)).sum())
            ),
        },
    }
    return _frame_with(frame, person, UK_COMPANY_CARS_MASS_CONSERVATION_REASON), receipt


def _violations(
    columns: Mapping[str, np.ndarray],
    *,
    holders: np.ndarray,
    spi: np.ndarray,
    epb: np.ndarray,
    carved: np.ndarray,
) -> dict[str, int]:
    """Counts of rows breaking an invariant every population keeps; all zero."""

    price = columns[LIST_PRICE]
    co2 = columns[CO2_EMISSIONS]
    electric_range = columns[ELECTRIC_RANGE]
    contributions = columns[CAPITAL_CONTRIBUTIONS]
    private = columns[PRIVATE_FUEL]
    fuel = columns[FUEL_TYPE]
    electric = fuel == ELECTRIC
    numbers = (price, co2, electric_range, contributions)
    # NaN compares false, so every comparison below would pass it: count it first.
    finite = np.logical_and.reduce([np.isfinite(values) for values in numbers])
    without_car = ~holders & (
        np.logical_or.reduce([values != 0 for values in numbers])
        | private
        | (fuel != DEFAULT_FUEL_TYPE)
    )
    return {
        "non_finite_values": int((~finite).sum()),
        "negative_values": int(
            np.logical_or.reduce([values < 0 for values in numbers]).sum()
        ),
        "holder_without_list_price": int((holders & ~(price > 0)).sum()),
        "inputs_without_a_car": int(without_car.sum()),
        "unknown_fuel_type": int((holders & ~np.isin(fuel, ENGINE_FUEL_TYPES)).sum()),
        "emitting_car_without_co2": int((holders & ~electric & (co2 < 1)).sum()),
        "electric_car_with_co2": int((holders & electric & (co2 != 0)).sum()),
        "private_fuel_on_an_electric_car": int((private & electric).sum()),
        "contributions_above_list_price": int((contributions > price + 1e-6).sum()),
        "electric_range_outside_1_to_50_g_km": int(
            ((electric_range > 0) & ((co2 < 1) | (co2 > 50))).sum()
        ),
        "spi_benefit_above_epb": int((spi & (carved > epb + 1e-6)).sum()),
    }


def _live_parameter(path: str, instant: str) -> float:
    from microcosm.build.uk_runtime.hmrc_uprating import live_engine_parameter_value

    return live_engine_parameter_value(path, instant)


# --------------------------------------------------------------------------
# Stage transform
# --------------------------------------------------------------------------


@dataclass
class UKCompanyCarsStageTransform:
    """Whole-stage callable for the company cars imputation."""

    stage: SourceStageSpec
    engine: object
    was_person_tab_path: str | Path | None = None
    donor_person: pd.DataFrame | None = None
    #: Supplied facts (tests and the synthetic fixture); else the vendored resource.
    statistics: HMRCCompanyCarStatistics | None = None
    #: Supplied in engine-free runs; else read from the installed engine.
    price_growth: float | None = None
    contribution_cap: float | None = None
    benefits: BenefitCalculator | None = None
    last_receipt: Mapping[str, Any] | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        return ()

    def _donor_person(self, columns: WASCompanyCarColumns) -> pd.DataFrame:
        if self.donor_person is not None:
            return self.donor_person
        if self.was_person_tab_path is None:
            raise CompanyCarsError(
                f"{UK_COMPANY_CARS_STAGE_NAME} requires the caller-supplied WAS "
                "person tab."
            )
        return read_pinned_tab(
            Path(self.was_person_tab_path).expanduser().resolve(),
            _artifact(self.stage, UK_WAS_PERSON_TAB_ROLE),
            columns=columns.raw_columns(),
        )

    def _engine_benefits(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        materialized = self.engine.materialize(
            frame, (COMPANY_CAR_BENEFIT, CAR_FUEL_BENEFIT), uk_time_period(frame)
        )
        return (
            np.asarray(materialized[COMPANY_CAR_BENEFIT], dtype=float),
            np.asarray(materialized[CAR_FUEL_BENEFIT], dtype=float),
        )

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        spec = CompanyCarsSpec.from_stage(self.stage)
        period = int(uk_time_period(frame))
        check_company_car_statistics_declaration(
            _operation(self.stage, VERIFY_VENDORED_FACT_RESOURCE_KIND),
            stage=UK_COMPANY_CARS_STAGE_NAME,
            period=period,
        )
        statistics = self.statistics or load_hmrc_company_car_statistics()
        growth = self.price_growth
        if growth is None:
            growth = _live_parameter(CPI_INDEX, f"{period}-01-01") / _live_parameter(
                CPI_INDEX, f"{spec.donor_price_year}-01-01"
            )
        cap = self.contribution_cap
        if cap is None:
            cap = _live_parameter(CAPITAL_CONTRIBUTION_CAP, f"{period}-04-30")
        donor = clean_was_company_car_donor(
            self._donor_person(spec.columns), columns=spec.columns
        ).grown(growth)
        result, receipt = impute_company_cars(
            frame,
            donor=donor,
            statistics=statistics,
            spec=spec,
            benefits=self.benefits or self._engine_benefits,
            contribution_cap=float(cap),
        )
        self.last_receipt = {
            "stage": UK_COMPANY_CARS_STAGE_NAME,
            "facts": statistics.evidence(),
            "donor": dict(donor.receipt),
            "contribution_cap_gbp": float(cap),
            **receipt,
        }
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_COMPANY_CARS_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            return {}
        return {"evidence": dict(self.last_receipt)}


__all__ = [
    "UK_COMPANY_CARS_MASS_CONSERVATION_REASON",
    "UK_COMPANY_CARS_NONNEGATIVE_OUTPUT_COLUMNS",
    "UK_COMPANY_CARS_OUTPUT_COLUMNS",
    "UK_COMPANY_CARS_REWRITES",
    "UK_COMPANY_CARS_STAGE_NAME",
    "UK_COMPANY_CAR_FACT_FENCE",
    "CompanyCarsError",
    "CompanyCarsSpec",
    "SPICarveOut",
    "UKCompanyCarsStageTransform",
    "WASCompanyCarColumns",
    "WASCompanyCarDonor",
    "assign_fuel_types",
    "carve_spi_benefits",
    "clean_was_company_car_donor",
    "draw_co2_emissions",
    "draw_list_prices",
    "holder_table",
    "impute_company_cars",
]
