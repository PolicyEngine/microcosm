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
    HMRC_COMPANY_CAR_RECORD_SETS,
    HMRCCompanyCarStatistics,
    load_hmrc_company_car_statistics,
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
        "record_sets": list(HMRC_COMPANY_CAR_RECORD_SETS),
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


_PREFIX = "hmrc.benefits_in_kind_july_2026."


def synthetic_company_car_statistics_payload() -> dict[str, Any]:
    """Every 2024-25 row the loader reads, as the vendored resource carries them.

    Invented, consistent figures in persons and pounds: 930,000 cars, half of
    them electric and two fifths of the rest at 1 to 50 g/km.
    """

    above = {
        band: 10_000.0
        for band in HMRC_CO2_BAND_IDS
        if band not in ("co2_0", "co2_1_to_50")
    }
    above["co2_115_to_124"] = 69_000.0
    above["co2_125_to_134"] = 50_000.0
    rows: dict[str, float] = {}

    def add(record_set: str, row: str, measure: str, value: float) -> None:
        rows[f"{_PREFIX}{record_set}.ty2024.{row}.{measure}"] = value

    for band, value in {"co2_0": 465_000.0, "co2_1_to_50": 186_000.0, **above}.items():
        add("table2a.recipients", band, "recipients", value)
    for fuel, value in (
        ("diesel_including_hybrids", 93_000.0),
        ("petrol_including_hybrids", 372_000.0),
        ("fully_electric", 465_000.0),
    ):
        add("table2c.recipients", fuel, "recipients", value)
    for row, figures in (
        ("car", (930_000.0, 3.0e9, 1.1e9, 4.1e8)),
        ("fuel", (40_000.0, 1.5e8, 6.0e7, 2.0e7)),
        ("all", (950_000.0, 3.15e9, 1.16e9, 4.3e8)),
    ):
        for measure, value in zip(
            (
                "recipients",
                "taxable_value",
                "income_tax_liability",
                "class_1a_nic_liability",
            ),
            figures,
            strict=True,
        ):
            add(f"table1.{measure}", row, measure, value)
    for income, recipients, value in (
        ("income_30000_to_39999", 80_000.0, 2.0e8),
        ("income_50000_to_74999", 320_000.0, 1.08e9),
        ("income_200000_plus", 50_000.0, 2.8e8),
    ):
        add("table3a.recipients", income, "recipients", recipients)
        add("table3a.taxable_value", income, "taxable_value", value)
    for region, recipients in (("E12000007", 150_000.0), ("S92000003", 60_000.0)):
        add("table4.recipients", region, "recipients", recipients)
    return {
        "source_fact_feed": {"source_commit": "synthetic"},
        "rows": [
            {"source_record_id": record_id, "value": value}
            for record_id, value in rows.items()
        ],
    }


def synthetic_company_car_statistics() -> HMRCCompanyCarStatistics:
    """The synthetic payload, typed by the loader the stage uses."""

    return load_hmrc_company_car_statistics(
        payload=synthetic_company_car_statistics_payload(), verify_feed_identity=False
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
