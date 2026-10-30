"""Synthetic company car facts, WAS donors and the stage declaration, for tests.

Every value here is invented. The HMRC margins only need to be consistent with
each other; the WAS tab carries the round-8 column names the stage reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime.company_cars import (
    FRS_PRICE_BANDS,
    UK_COMPANY_CAR_FACT_FENCE,
    UK_COMPANY_CARS_MASS_CONSERVATION_REASON,
)
from microcosm.build.uk_runtime.hmrc_company_car_statistics import (
    HMRC_CO2_BAND_IDS,
    HMRCCompanyCarStatistics,
)

WAS_VEHICLES = tuple(
    {
        "car_or_van": f"CARVAN{index}R8",
        "fuel": f"PETROL{index}R8",
        "list_price": f"LISTPR{index}R8",
    }
    for index in (1, 2, 3)
)

COMPANY_CARS_STAGE_OPERATIONS: tuple[dict[str, Any], ...] = (
    {
        "kind": "verify_vendored_fact_resource",
        "artifact_role": "benefits_in_kind_facts",
        "resource": "hmrc_benefits_in_kind_facts.json",
        "mapped_build_period": 2024,
    },
    {
        "kind": "clean_was_company_car_donor",
        "household_key": "CASER8",
        "household_weight": "R8xshhwgt",
        "has_company_vehicle": "COMPVEHR8",
        "vehicles": list(WAS_VEHICLES),
        "yes_code": 1,
        "car_code": 1,
        "electric_code": 4,
        "price_year": 2021,
    },
    {
        "kind": "assign_company_car_fuel_types",
        "outputs": ["company_car_fuel_type"],
        "diesel_fuel_type": "DIESEL_RDE2",
        "unknown_fuel": "table_2c_shares",
        "seed": 0,
        "salt": "company_car_fuel_type",
    },
    {
        "kind": "draw_company_car_list_prices",
        "outputs": ["company_car_list_price", "company_car_capital_contributions"],
        "bands": {str(band): list(bounds) for band, bounds in FRS_PRICE_BANDS.items()},
        "minimum_donor_households": 3,
        "open_band_fallback_upper_gbp": 80_000.0,
        "seed": 0,
        "salt": "company_car_list_price",
    },
    {
        "kind": "draw_company_car_co2_emissions",
        "outputs": ["company_car_co2_emissions", "company_car_electric_range"],
        "margins": "table_2a_nonzero_shares_hybrids_first",
        "top_band_upper_g_km": 179,
        "electric_range_miles": 50.0,
        "seed": 0,
        "salt": "company_car_co2_emissions",
    },
    {
        "kind": "carve_spi_company_car_benefits",
        "outputs": ["employment_income", "company_car_private_fuel_provided"],
        "epb_column": "hmrc_spi_employment_benefits",
        "benefits": ["company_car_benefit", "car_fuel_benefit"],
        "minimum_scaled_list_price_gbp": 5_000.0,
    },
    {
        "kind": "record_mass_conservation_receipt",
        "entity": "household",
        "reason": UK_COMPANY_CARS_MASS_CONSERVATION_REASON,
        "declared_factor": 1.0,
    },
    {
        "kind": "classify_company_car_facts_with_reviewed_fence",
        **UK_COMPANY_CAR_FACT_FENCE,
    },
)


def company_cars_stage_spec(
    operations: tuple[dict[str, Any], ...] = COMPANY_CARS_STAGE_OPERATIONS,
) -> SourceStageSpec:
    """The stage declaration the tests run against."""

    return SourceStageSpec(
        stage="company_cars",
        survey="FRS job record; Wealth and Assets Survey round 8; HMRC company cars",
        source="synthetic",
        grain="person",
        artifacts=(
            {"role": "benefits_in_kind_facts", "filename": "facts.json"},
            {"role": "was_person_tab", "filename": "was_person.tab"},
        ),
        operations=tuple(
            SourceOperationSpec(
                kind=operation["kind"],
                parameters={
                    key: value for key, value in operation.items() if key != "kind"
                },
            )
            for operation in operations
        ),
        outputs=(
            "company_car_list_price",
            "company_car_co2_emissions",
            "company_car_fuel_type",
            "company_car_electric_range",
            "company_car_capital_contributions",
            "company_car_private_fuel_provided",
            "employment_income",
        ),
        rewrites=("employment_income",),
    )


def synthetic_company_car_statistics() -> HMRCCompanyCarStatistics:
    """Invented 2024-25 margins: half the cars electric, a fifth at 1 to 50 g/km."""

    above = {band: 10.0 for band in HMRC_CO2_BAND_IDS if band not in ("zero", "1_50")}
    above["115_124"] = 40.0
    above["125_134"] = 30.0
    return HMRCCompanyCarStatistics(
        tax_year=2024,
        co2_band_recipients={"zero": 500.0, "1_50": 200.0, **above},
        fuel_type_recipients={"electric": 500.0, "petrol": 400.0, "diesel": 100.0},
        car_recipients=1000.0,
        car_taxable_value_gbp_m=3000.0,
        car_income_tax_gbp_m=1100.0,
        car_nics_gbp_m=400.0,
        fuel_recipients=40.0,
        fuel_taxable_value_gbp_m=150.0,
    )


def synthetic_was_company_car_tab(seed: int = 7, people: int = 400) -> pd.DataFrame:
    """A WAS round-8 person tab: a quarter of people with up to three vehicles."""

    rng = np.random.default_rng(seed)
    households = rng.integers(1, people // 2, people)
    frame = pd.DataFrame(
        {
            "CASER8": households,
            "R8xshhwgt": rng.uniform(500, 3000, people).round(1),
            "COMPVEHR8": np.where(rng.random(people) < 0.25, 1, 2),
        }
    )
    for vehicle in WAS_VEHICLES:
        has = rng.random(people) < 0.8
        frame[vehicle["car_or_van"]] = np.where(
            has, np.where(rng.random(people) < 0.85, 1, 2), -1
        )
        frame[vehicle["fuel"]] = np.where(has, rng.integers(1, 7, people), -1)
        frame[vehicle["list_price"]] = np.where(
            has & (rng.random(people) < 0.9),
            (rng.lognormal(10.4, 0.35, people) / 1000).round() * 1000,
            -9,
        )
    return frame
