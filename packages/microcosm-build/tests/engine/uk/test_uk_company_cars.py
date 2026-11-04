"""Engine lockstep for the company cars stage (dashboard #316).

The stage writes six of policyengine-uk 2.165.0's company car inputs and takes
the engine's car and fuel benefits out of the SPI tape rows' pay. These tests fail if
the engine stops reading the inputs as the stage writes them (the fuel type as
an enum name included), if its car benefit stops being linear in the list price
above the capped contributions, on which the SPI scaling rests, or if Budget
2026's 2030-31 rise stops reaching the stage's holders alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import company_cars as stage
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_release_export_frame,
    write_uk_national_frame,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from test_support.microcosm_build.uk_company_cars import (
    company_cars_stage_spec,
    synthetic_company_car_statistics,
    synthetic_was_company_car_tab,
)

_VARIABLES = (
    "company_car_benefit",
    "car_fuel_benefit",
    "benefits_in_kind",
    "employment_income",
    "taxable_employment_income",
    "ni_class_1a",
)


def _frame():
    """Six employees, one per household: two FRS company car holders (an
    electric car, and a petrol car with private fuel), three holders on SPI tape
    rows (EPB with room for the benefits, EPB that is not, and no EPB), and no
    car."""

    person = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41, 51, 61],
            "person_benunit_id": [1, 2, 3, 4, 5, 6],
            "person_household_id": [1, 2, 3, 4, 5, 6],
            "age": [40.0] * 6,
            "employment_income": [
                60_000.0,
                45_000.0,
                150_000.0,
                70_000.0,
                30_000.0,
                50_000.0,
            ],
            "frs_company_car_price_band": [9, 7, 8, 8, 0, 8],
            "frs_company_car_fuel_code": [4, 1, 2, 1, 0, 1],
            "frs_company_car_private_fuel": [False, True, False, False, False, False],
            "frs_company_car_contribution": [0.0, 1_000.0, 0.0, 0.0, 0.0, 0.0],
            "person_support_channel": ["frs", "frs", "spi", "spi", "frs", "spi"],
            "person_is_spi_income_recipient": [False, False, True, True, False, True],
            # The tape's PAY and EPB on the tape rows (no TAXTERM); the FRS rows'
            # PAY is their own pay.
            "hmrc_spi_pay": [
                60_000.0,
                45_000.0,
                120_000.0,
                68_500.0,
                30_000.0,
                50_000.0,
            ],
            "hmrc_spi_employment_benefits": [0.0, 0.0, 30_000.0, 1_500.0, 0.0, 0.0],
            "employee_pension_contributions": [
                3_000.0,
                0.0,
                9_000.0,
                2_000.0,
                0.0,
                0.0,
            ],
            "employment_status": ["FT_EMPLOYED"] * 6,
            "self_employment_income": [0.0] * 6,
            "uc_is_in_gainful_self_employment": [False] * 6,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5, 6],
            "region": ["LONDON"] * 6,
            "council_tax": [0.0] * 6,
            "tenure_type": ["OWNED_OUTRIGHT"] * 6,
            "rent": [0.0] * 6,
            "household_source_id": [1, 2, 3, 4, 5, 6],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3, 4, 5, 6]}),
        household=household,
        household_weights=np.ones(6),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )


@pytest.fixture(scope="module")
def engine():
    return PolicyEngineUKEngine()


@pytest.fixture(scope="module")
def run(engine):
    transform = stage.UKCompanyCarsStageTransform(
        stage=company_cars_stage_spec(),
        engine=engine,
        donor_person=synthetic_was_company_car_tab(),
        statistics=synthetic_company_car_statistics(),
    )
    before = _frame()
    after = transform(before)
    return before, after, transform.last_receipt


def test_the_engine_reads_the_stage_s_inputs(engine, run) -> None:
    _, after, receipt = run
    values = engine.materialize(after, _VARIABLES, 2024)
    person = after.table("person")
    holder = person[stage.LIST_PRICE].to_numpy() > 0
    assert holder.tolist() == [True, True, True, True, False, True]
    assert person[stage.FUEL_TYPE].tolist()[:2] == [stage.ELECTRIC, stage.PETROL]
    # The electric car pays the zero-emission rate; the petrol car with private
    # fuel takes the fuel benefit too.
    assert (values["company_car_benefit"][holder] > 0).all()
    assert values["car_fuel_benefit"][1] > 0
    assert values["car_fuel_benefit"][0] == 0
    np.testing.assert_allclose(
        values["benefits_in_kind"],
        values["company_car_benefit"] + values["car_fuel_benefit"],
    )
    assert (values["ni_class_1a"][holder] > 0).all()
    assert values["ni_class_1a"][4] == 0
    assert receipt["contribution_cap_gbp"] == 5_000.0


def test_spi_pay_plus_the_engine_s_benefits_is_the_tape_s_total(engine, run) -> None:
    before, after, receipt = run
    values = engine.materialize(after, _VARIABLES, 2024)
    pay_before = before.table("person")["employment_income"].to_numpy()
    pay_after = after.table("person")["employment_income"].to_numpy()
    epb = before.table("person")["hmrc_spi_employment_benefits"].to_numpy()
    # FRS-channel pay is untouched.
    np.testing.assert_array_equal(pay_after[[0, 1, 4]], pay_before[[0, 1, 4]])
    # A tape row's pay gives up the engine's benefits. The engine computes in
    # single precision, so the identity holds to the penny, not to float64.
    np.testing.assert_allclose(
        pay_after[[2, 3, 5]] + values["benefits_in_kind"][[2, 3, 5]],
        pay_before[[2, 3, 5]],
        atol=0.01,
    )
    # The first tape row's benefits fit in its EPB. The second's go above its
    # EPB of GBP 1,500, and the rest moves from its PAY into its EPB; the third
    # drew no EPB, so all of its benefits move. Each keeps its PAY + EPB.
    spi_after = after.table("person")
    epb_after = spi_after["hmrc_spi_employment_benefits"].to_numpy()
    spi_pay_after = spi_after["hmrc_spi_pay"].to_numpy()
    assert values["benefits_in_kind"][2] <= epb[2] + 0.01
    assert epb_after[2] == pytest.approx(epb[2])
    for row in (3, 5):
        assert values["benefits_in_kind"][row] > epb[row]
        assert epb_after[row] == pytest.approx(
            values["benefits_in_kind"][row], abs=0.01
        )
    np.testing.assert_allclose(
        (spi_pay_after + epb_after)[[2, 3, 5]], [150_000.0, 70_000.0, 50_000.0]
    )
    # The stage priced its output again on the engine: within a penny.
    assert receipt["spi_carve_out"]["max_abs_pay_identity_error_gbp"] <= 0.01


def test_the_2030_rise_reaches_the_holders_alone(engine, run, tmp_path) -> None:
    from policyengine_uk import Microsimulation
    from policyengine_uk.scenarios.company_car_tax_rise_2030 import (
        COMPANY_CAR_TAX_RISE_2030,
    )

    _, after, _ = run
    path = write_uk_national_frame(
        uk_release_export_frame(after), tmp_path / "company_cars.h5"
    )
    baseline = Microsimulation(dataset=str(path))
    reformed = Microsimulation(dataset=str(path), reform=COMPANY_CAR_TAX_RISE_2030)
    for year, raised in ((2029, False), (2030, True)):
        tax = np.asarray(baseline.calculate("income_tax", year).values)
        tax_reformed = np.asarray(reformed.calculate("income_tax", year).values)
        holder = np.asarray(baseline.calculate("has_company_car", year).values)
        assert holder.tolist() == [True, True, True, True, False, True]
        assert tax_reformed[~holder] == pytest.approx(tax[~holder])
        if raised:
            assert (tax_reformed[holder] > tax[holder]).all()
        else:
            assert tax_reformed == pytest.approx(tax)
