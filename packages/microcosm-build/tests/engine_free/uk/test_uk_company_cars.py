"""The company cars stage's pieces on synthetic data (dashboard #316)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stage_evidence import snapshot_stage_evidence
from microcosm.build.uk_runtime import company_cars as stage
from microcosm.build.uk_runtime import salary_sacrifice
from microcosm.build.uk_runtime.frs_spine import (
    FRS_COMPANY_CAR_PRICE_BAND_BOUNDS,
    FRS_COMPANY_CAR_PRICE_BAND_UNKNOWN,
    uc_gainful_self_employment,
)
from microcosm.build.uk_runtime.hmrc_company_car_statistics import (
    HMRC_COMPANY_CAR_RECORD_SETS,
    HMRC_LOW_EMISSION_BAND,
    HMRCCompanyCarStatisticsError,
    check_company_car_statistics_declaration,
    co2_band_bounds,
    load_hmrc_company_car_statistics,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import (
    small_cell_share,
    support_band,
)
from microcosm.build.uk_runtime.spi_income import SPI_INCOME_RECIPIENT_COLUMN
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_company_cars import (
    COMPANY_CARS_STAGE_OPERATIONS,
    company_cars_stage_spec,
    synthetic_company_car_statistics,
    synthetic_company_car_statistics_payload,
    synthetic_was_company_car_tab,
)

CAP = 5_000.0


@pytest.fixture(scope="module")
def statistics():
    return synthetic_company_car_statistics()


@pytest.fixture(scope="module")
def spec():
    return stage.CompanyCarsSpec.from_stage(company_cars_stage_spec())


@pytest.fixture(scope="module")
def donor(spec):
    return stage.clean_was_company_car_donor(
        synthetic_was_company_car_tab(), columns=spec.columns
    ).grown(1.2)


def _replaced(kind: str, **changes):
    return tuple(
        {**operation, **changes} if operation["kind"] == kind else operation
        for operation in COMPANY_CARS_STAGE_OPERATIONS
    )


def test_the_declaration_is_read_once_and_checked(spec) -> None:
    assert spec.diesel_fuel_type == stage.DIESEL_RDE2
    assert spec.minimum_donor_households == 3
    assert spec.columns.list_price == ("LISTPR1R8", "LISTPR2R8", "LISTPR3R8")
    with pytest.raises(stage.CompanyCarsError, match="fence"):
        stage.CompanyCarsSpec.from_stage(
            company_cars_stage_spec(
                _replaced(
                    stage.CLASSIFY_COMPANY_CAR_FACTS_KIND, calibration_permitted=True
                )
            )
        )
    with pytest.raises(stage.CompanyCarsError, match="CARVAL"):
        stage.CompanyCarsSpec.from_stage(
            company_cars_stage_spec(
                _replaced(stage.DRAW_COMPANY_CAR_LIST_PRICES_KIND, bands={"1": [0, 9]})
            )
        )
    with pytest.raises(stage.CompanyCarsError, match="writes"):
        stage.CompanyCarsSpec.from_stage(
            company_cars_stage_spec(
                _replaced(stage.ASSIGN_COMPANY_CAR_FUEL_TYPES_KIND, outputs=[])
            )
        )
    with pytest.raises(stage.CompanyCarsError, match="households"):
        stage.CompanyCarsSpec.from_stage(
            company_cars_stage_spec(
                _replaced(
                    stage.DRAW_COMPANY_CAR_LIST_PRICES_KIND, minimum_donor_households=2
                )
            )
        )


def test_the_fact_fence_counts_the_vendored_rows(spec, statistics) -> None:
    assert spec.fenced_fact_count == len(
        synthetic_company_car_statistics_payload()["rows"]
    )
    assert statistics.row_count == spec.fenced_fact_count
    uncounted = _replaced(stage.CLASSIFY_COMPANY_CAR_FACTS_KIND, fenced_fact_count=None)
    with pytest.raises(stage.CompanyCarsError, match="count"):
        stage.CompanyCarsSpec.from_stage(company_cars_stage_spec(uncounted))
    # A resource with another row count than the fence's is refused at run time.
    other = dataclasses.replace(statistics, row_count=statistics.row_count + 1)
    transform = dataclasses.replace(_transform(statistics), statistics=other)
    with pytest.raises(stage.CompanyCarsError, match="fence covers"):
        transform(_stage_frame())


def test_the_donor_keeps_company_cars_with_a_price_only(spec) -> None:
    person = pd.DataFrame(
        {
            "CASER8": [1, 2, 3, 4],
            "R8xshhwgt": [100.0, 200.0, 300.0, 0.0],
            "COMPVEHR8": [1, 1, 2, 1],
            "CARVAN1R8": [1, 2, 1, 1],
            "PETROL1R8": [4, 1, 1, 1],
            "LISTPR1R8": [45_000, 30_000, 25_000, 20_000],
            "CARVAN2R8": [1, -1, -1, -1],
            "PETROL2R8": [1, -1, -1, -1],
            "LISTPR2R8": [-9, -9, -9, -9],
            "CARVAN3R8": [-1, -1, -1, -1],
            "PETROL3R8": [-1, -1, -1, -1],
            "LISTPR3R8": [-9, -9, -9, -9],
        }
    )
    donor = stage.clean_was_company_car_donor(person, columns=spec.columns)
    # The van, the person without a company vehicle, the car without a price and
    # the household without a weight are all left out.
    assert donor.price.tolist() == [45_000.0]
    assert donor.is_electric.tolist() == [True]
    assert donor.weight.tolist() == [100.0]
    assert donor.grown(1.5).price.tolist() == [67_500.0]
    with pytest.raises(stage.CompanyCarsError, match="lacks"):
        stage.clean_was_company_car_donor(
            person.drop(columns="LISTPR3R8"), columns=spec.columns
        )


def test_fuel_codes_map_and_an_unknown_fuel_draws_at_table_2c_shares(
    spec, statistics
) -> None:
    codes = np.array([1, 2, 3, 4, 5] + [6] * 20_000)
    ids = np.arange(len(codes)) + 1
    fuel, drawn = stage.assign_fuel_types(
        codes,
        ids,
        private_fuel=np.zeros(len(codes), dtype=bool),
        weights=np.ones(len(codes)),
        statistics=statistics,
        spec=spec,
    )
    assert fuel[:5].tolist() == [
        stage.PETROL,
        stage.DIESEL_RDE2,
        stage.ALTERNATIVE_FUEL,
        stage.ELECTRIC,
        stage.ALTERNATIVE_FUEL,
    ]
    assert not drawn[:5].any() and drawn[5:].all()
    shares = pd.Series(fuel[5:]).value_counts(normalize=True)
    assert shares[stage.ELECTRIC] == pytest.approx(0.5, abs=0.02)
    assert shares[stage.PETROL] == pytest.approx(0.4, abs=0.02)
    assert shares[stage.DIESEL_RDE2] == pytest.approx(0.1, abs=0.02)


def test_an_unknown_fuel_with_private_fuel_never_draws_electricity(
    spec, statistics
) -> None:
    """An electric car carries no fuel benefit, so a holder who reports private
    fuel draws petrol or diesel at their Table 2C shares (item 7 of the review)."""

    codes = np.array([6] * 20_000)
    ids = np.arange(len(codes)) + 1
    fueled = np.zeros(len(codes), dtype=bool)
    fueled[::2] = True
    fuel, drawn = stage.assign_fuel_types(
        codes,
        ids,
        private_fuel=fueled,
        weights=np.ones(len(codes)),
        statistics=statistics,
        spec=spec,
    )
    assert drawn.all()
    assert not (fuel[fueled] == stage.ELECTRIC).any()
    shares = pd.Series(fuel[fueled]).value_counts(normalize=True)
    assert shares[stage.PETROL] == pytest.approx(0.8, abs=0.02)


def test_unknown_fuels_meet_table_2c_together(spec, statistics) -> None:
    """The holders who report private fuel cannot draw electricity, so the
    others draw it at the share that keeps Table 2C's among all the cars whose
    fuel is drawn (item 7 of the second review)."""

    codes = np.array([6] * 50_000)
    ids = np.arange(len(codes)) + 1
    fueled = np.zeros(len(codes), dtype=bool)
    fueled[::5] = True
    fuel, _ = stage.assign_fuel_types(
        codes,
        ids,
        private_fuel=fueled,
        weights=np.ones(len(codes)),
        statistics=statistics,
        spec=spec,
    )
    assert not (fuel[fueled] == stage.ELECTRIC).any()
    shares = pd.Series(fuel).value_counts(normalize=True)
    assert shares[stage.ELECTRIC] == pytest.approx(0.5, abs=0.01)
    assert shares[stage.PETROL] == pytest.approx(0.4, abs=0.01)
    assert shares[stage.DIESEL_RDE2] == pytest.approx(0.1, abs=0.01)
    # Weighted, the private-fuel holders carry 60% of the mass: the others draw
    # electricity all, the most they can carry.
    fuel, _ = stage.assign_fuel_types(
        codes,
        ids,
        private_fuel=fueled,
        weights=np.where(fueled, 6.0, 1.0),
        statistics=statistics,
        spec=spec,
    )
    assert (fuel[~fueled] == stage.ELECTRIC).all()
    assert not (fuel[fueled] == stage.ELECTRIC).any()


def test_list_prices_stay_in_their_band(spec, donor) -> None:
    bands = np.repeat(np.arange(1, 11), 300)
    electric = np.tile([True, False], len(bands) // 2)
    ids = np.arange(len(bands)) + 1
    prices, receipt = stage.draw_list_prices(
        bands, electric, ids, donor=donor, spec=spec
    )
    for band, (low, high) in FRS_COMPANY_CAR_PRICE_BAND_BOUNDS.items():
        drawn = prices[bands == band]
        assert (drawn > low).all(), band
        if high is not None:
            assert (drawn <= high).all(), band
    assert (prices[bands == FRS_COMPANY_CAR_PRICE_BAND_UNKNOWN] > 0).all()
    assert receipt["routes"]


def test_a_band_with_too_few_donor_households_is_drawn_uniformly(spec) -> None:
    # Two band-2 cars in one household; twelve electric cars above GBP 40,000 in
    # twelve households.
    donor = stage.WASCompanyCarDonor(
        price=np.array([11_000.0, 12_000.0, *np.linspace(50_000, 72_000, 12)]),
        is_electric=np.array([False, False, *[True] * 12]),
        weight=np.ones(14),
        household=np.array([1, 1, *range(2, 14)]),
        receipt={},
    )
    bands = np.array([2] * 50 + [9] * 50)
    prices, receipt = stage.draw_list_prices(
        bands, np.ones(100, dtype=bool), np.arange(100) + 1, donor=donor, spec=spec
    )
    assert set(receipt["routes"]) >= {"band_2:uniform", "band_9:donor_level_0"}
    assert prices[:50].min() > 10_000 and prices[:50].max() <= 13_000
    assert prices[50:].min() >= 50_000 and prices[50:].max() <= 72_000
    # Too few above GBP 40,000 too: the open band is uniform up to the fallback.
    sparse = dataclasses.replace(donor, household=np.array([1] * 14))
    with pytest.raises(stage.CompanyCarsError, match="donor households"):
        stage.draw_list_prices(
            np.array([10]), np.array([True]), np.array([1]), donor=sparse, spec=spec
        )
    prices, receipt = stage.draw_list_prices(
        np.array([9] * 20),
        np.ones(20, dtype=bool),
        np.arange(20) + 1,
        donor=sparse,
        spec=spec,
    )
    assert "band_9:uniform" in receipt["routes"]
    assert prices.min() > 40_000 and prices.max() <= spec.open_band_fallback_upper_gbp


def test_band_1_s_uniform_draw_starts_at_the_minimum_list_price(spec) -> None:
    """Without band-1 donors the draw is uniform from the least price the
    carve-out keeps a car at, not from zero (item 9 of the review)."""

    donor = stage.WASCompanyCarDonor(
        price=np.array([8_000.0, 9_000.0]),
        is_electric=np.zeros(2, dtype=bool),
        weight=np.ones(2),
        household=np.array([1, 1]),
        receipt={},
    )
    prices, receipt = stage.draw_list_prices(
        np.ones(200, dtype=int),
        np.zeros(200, dtype=bool),
        np.arange(200) + 1,
        donor=donor,
        spec=spec,
    )
    assert "band_1:uniform" in receipt["routes"]
    assert prices.min() >= spec.minimum_scaled_list_price_gbp
    assert prices.max() <= FRS_COMPANY_CAR_PRICE_BAND_BOUNDS[1][1]


def test_co2_matches_table_2a_among_cars_that_emit_it(spec, statistics) -> None:
    rng = np.random.default_rng(3)
    n = 30_000
    fuel = rng.choice(
        [stage.ELECTRIC, stage.PETROL, stage.ALTERNATIVE_FUEL], n, p=[0.5, 0.4, 0.1]
    ).astype(object)
    hybrid = fuel == stage.ALTERNATIVE_FUEL
    weights = rng.uniform(1, 3, n)
    co2, electric_range, receipt = stage.draw_co2_emissions(
        fuel,
        hybrid,
        weights,
        np.arange(n) + 1,
        sources=np.arange(n),
        statistics=statistics,
        spec=spec,
    )
    assert (co2[fuel == stage.ELECTRIC] == 0).all()
    emits = fuel != stage.ELECTRIC
    assert (co2[emits] >= 1).all()
    assert (co2[emits] <= spec.top_band_upper_g_km).all()
    low = emits & (co2 <= 50)
    share = weights[low].sum() / weights[emits].sum()
    # The non-hybrids' draw meets the share in expectation.
    target = statistics.nonzero_co2_shares()[HMRC_LOW_EMISSION_BAND]
    assert receipt["table_2a_share_1_50"] == target
    assert share == pytest.approx(target, abs=0.01)
    # Hybrids go first: their weight is below the target, so every one is low.
    assert (co2[hybrid] <= 50).all()
    assert (electric_range[low] == spec.electric_range_miles).all()
    assert (electric_range[~low] == 0).all()
    assert receipt["max_abs_band_share_gap"] < 0.01
    for band in ("co2_95_to_104", "co2_170_plus"):
        lowest, _ = co2_band_bounds(band)
        assert (co2[emits & (co2 >= lowest)] >= lowest).all()


def test_the_carve_out_fits_each_benefit_inside_epb_and_pay(spec) -> None:
    price = np.array([40_000.0, 40_000.0, 40_000.0, 30_000.0])
    car = 0.25 * (price - np.minimum([0.0, 6_000.0, 0.0, 0.0], CAP))
    carve = stage.carve_spi_benefits(
        epb=np.array([20_000.0, 3_000.0, 500.0, 9_000.0]),
        pay=np.array([90_000.0, 60_000.0, 40_000.0, 50_000.0]),
        employee_contributions=np.zeros(4),
        list_price=price,
        contributions=np.array([0.0, 6_000.0, 0.0, 0.0]),
        private_fuel=np.array([True, False, False, True]),
        car_benefit=car,
        fuel_benefit=np.array([7_000.0, 0.0, 0.0, 7_000.0]),
        contribution_cap=CAP,
        spec=spec,
    )
    # 1: car 10,000 plus fuel 7,000 fit inside 20,000.
    assert carve.keep[0] and carve.benefit[0] == pytest.approx(17_000.0)
    # 2: car 8,750 above EPB 3,000: the price scales to 12,000 with no contribution.
    assert carve.keep[1] and carve.benefit[1] == pytest.approx(3_000.0)
    assert carve.list_price[1] == pytest.approx(12_000.0)
    assert carve.contributions[1] == 0.0
    # 3: EPB 500 would need a GBP 2,000 car, below the minimum price: dropped.
    assert not carve.keep[2] and carve.benefit[2] == 0.0
    # 4: car 7,500 and fuel 7,000 do not fit inside EPB 9,000 together, so the
    # fuel goes first and the car fits whole.
    assert carve.keep[3] and not carve.private_fuel[3]
    assert carve.benefit[3] == pytest.approx(7_500.0)
    assert carve.list_price[3] == pytest.approx(30_000.0)
    assert carve.scaled.tolist() == [False, True, False, False]
    assert (carve.benefit <= np.array([20_000.0, 3_000.0, 500.0, 9_000.0])).all()


def test_private_fuel_goes_first_when_car_and_fuel_do_not_fit_together(spec) -> None:
    """Dropping the fuel first never loses a car that fits without it (item 7 of
    the review, María's ruling of 31 October 2026)."""

    carve = stage.carve_spi_benefits(
        epb=np.array([6_000.0, 9_000.0]),
        pay=np.array([40_000.0, 40_000.0]),
        employee_contributions=np.zeros(2),
        list_price=np.array([40_000.0, 35_000.0]),
        contributions=np.zeros(2),
        private_fuel=np.array([True, True]),
        car_benefit=np.array([4_000.0, 0.29 * 35_000.0]),
        fuel_benefit=np.array([7_000.0, 0.29 * 27_800.0]),
        contribution_cap=CAP,
        spec=spec,
    )
    assert carve.keep.all() and not carve.private_fuel.any()
    assert carve.benefit[0] == pytest.approx(4_000.0)
    # The review's case: a 29% car on GBP 35,000 (10,150) with GBP 8,062 of fuel
    # in EPB 9,000. The fuel goes, and the car scales to GBP 31,034.
    assert carve.list_price[1] == pytest.approx(9_000.0 / 0.29)
    assert carve.benefit[1] == pytest.approx(9_000.0)


def test_the_cap_is_epb_within_the_pay(spec) -> None:
    carve = stage.carve_spi_benefits(
        epb=np.array([9_000.0, 9_000.0]),
        pay=np.array([40_000.0, 4_000.0]),
        employee_contributions=np.zeros(2),
        list_price=np.array([40_000.0, 40_000.0]),
        contributions=np.zeros(2),
        private_fuel=np.zeros(2, dtype=bool),
        car_benefit=np.array([10_000.0, 10_000.0]),
        fuel_benefit=np.zeros(2),
        contribution_cap=CAP,
        spec=spec,
    )
    assert carve.benefit.tolist() == pytest.approx([9_000.0, 4_000.0])
    assert carve.scaled.all()
    assert carve.list_price.tolist() == pytest.approx([36_000.0, 16_000.0])


def test_the_cap_leaves_the_pay_an_employee_contribution_needs(spec) -> None:
    """The SPI income stage capped a tape row's net-pay contribution at the pay
    it drew, and salary_sacrifice refuses one above the pay that remains, so the
    carve-out keeps that pay (item 2 of the second review, on its two probes)."""

    epb = np.array([3_000.0, 6_000.0, 9_000.0])
    pay = np.array([3_000.0, 20_000.0, 40_000.0])
    employee = np.array([150.0, 20_000.0, 2_000.0])
    carve = stage.carve_spi_benefits(
        epb=epb,
        pay=pay,
        employee_contributions=employee,
        list_price=np.array([40_000.0, 40_000.0, 40_000.0]),
        contributions=np.zeros(3),
        private_fuel=np.zeros(3, dtype=bool),
        car_benefit=np.array([10_000.0, 10_000.0, 10_000.0]),
        fuel_benefit=np.zeros(3),
        contribution_cap=CAP,
        spec=spec,
    )
    # PAY 0 and EPB 3,000 with a 150 contribution: the car fits 2,850.
    assert carve.keep[0] and carve.benefit[0] == pytest.approx(2_850.0)
    assert carve.list_price[0] == pytest.approx(2_850.0 / 0.25)
    # The contribution takes all the pay: no room, so the car is dropped.
    assert not carve.keep[1] and carve.benefit[1] == 0.0
    # Room to spare: EPB caps as before.
    assert carve.benefit[2] == pytest.approx(9_000.0)
    assert (pay - carve.benefit >= employee - 1e-9).all()
    assert carve.receipt["cap_below_epb_for_the_employee_contribution"] == (
        stage.rounded_count(2)
    )


class _UKEngine:
    country = "uk"


def _benefits(frame):
    person = frame.table("person")
    price = person[stage.LIST_PRICE].to_numpy(float)
    co2 = person[stage.CO2_EMISSIONS].to_numpy(float)
    contribution = np.minimum(person[stage.CAPITAL_CONTRIBUTIONS].to_numpy(float), CAP)
    rate = np.where(co2 == 0, 0.03, np.where(co2 <= 50, 0.10, 0.25))
    car = np.where(price > 0, np.maximum(price - contribution, 0.0) * rate, 0.0)
    fuel = np.where(
        person[stage.PRIVATE_FUEL].to_numpy(bool) & (co2 > 0), rate * 27_800.0, 0.0
    )
    return car, fuel


def _stage_frame(seed: int = 1, households: int = 600, order=None, sources=None):
    """``sources`` maps the household ids to their source households; each is its
    own by default."""

    rng = np.random.default_rng(seed)
    ids = np.arange(households) + 1000
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_source_id": ids if sources is None else sources(ids),
        }
    )
    rows = []
    for household_id in household["household_id"]:
        spi = bool(rng.random() < 0.4)
        for _ in range(int(rng.integers(1, 4))):
            has_car = rng.random() < 0.15
            # Most SPI-channel rows are drawn from the tape; the others (undrawn
            # copies and dependants) keep their twin's pay with no EPB.
            tape = spi and bool(rng.random() < 0.8)
            rows.append(
                {
                    "person_id": 10_000 + len(rows),
                    "person_household_id": household_id,
                    "person_benunit_id": household_id * 10,
                    "employment_income": float(rng.uniform(15_000, 120_000)),
                    "frs_company_car_price_band": int(rng.integers(1, 11))
                    if has_car
                    else 0,
                    "frs_company_car_fuel_code": int(rng.integers(1, 7))
                    if has_car
                    else 0,
                    "frs_company_car_private_fuel": bool(
                        has_car and rng.random() < 0.1
                    ),
                    "frs_company_car_contribution": float(
                        rng.choice([0.0, 2_000.0, 7_000.0])
                    )
                    if has_car
                    else 0.0,
                    "person_support_channel": "spi" if spi else "frs",
                    SPI_INCOME_RECIPIENT_COLUMN: tape,
                    "hmrc_spi_employment_benefits": float(
                        rng.choice([0.0, 800.0, 6_000.0, 20_000.0])
                    )
                    if tape
                    else 0.0,
                }
            )
    person = pd.DataFrame(rows)
    # The SPI income stage's pension and UC columns, on a stream of their own
    # so the draws above stay as they were. A tenth of the tape rows contribute
    # all their pay, which the SPI stage's cap allows.
    extra = np.random.default_rng(seed + 7)
    pay = person["employment_income"].to_numpy(float)
    tape_rows = person[SPI_INCOME_RECIPIENT_COLUMN].to_numpy(bool)
    u = extra.random(len(person))
    person["employee_pension_contributions"] = np.where(
        tape_rows & (u < 0.1), pay, np.where(u < 0.5, 1_500.0, 0.0)
    )
    self_employed = extra.random(len(person)) < 0.05
    person["employment_status"] = np.where(
        self_employed, "FT_SELF_EMPLOYED", "FT_EMPLOYED"
    )
    person["self_employment_income"] = np.where(
        extra.random(len(person)) < 0.3, extra.uniform(0, 100_000, len(person)), 0.0
    )
    person[stage.UC_GAINFUL_SELF_EMPLOYMENT] = uc_gainful_self_employment(
        self_employed, person["self_employment_income"].to_numpy(float), pay
    )
    if order is not None:
        person = person.iloc[order(len(person))].reset_index(drop=True)
    weights = rng.uniform(5, 30, households)
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": household["household_id"] * 10}),
        household=household,
        household_weights=weights,
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )


def _transform(statistics):
    return stage.UKCompanyCarsStageTransform(
        stage=company_cars_stage_spec(),
        engine=_UKEngine(),
        donor_person=synthetic_was_company_car_tab(),
        statistics=statistics,
        price_growth=1.2,
        contribution_cap=CAP,
        benefits=_benefits,
    )


def test_the_whole_stage_writes_its_columns_and_carves_spi_pay(statistics) -> None:
    frame = _stage_frame()
    before = frame.table("person").set_index("person_id")
    transform = _transform(statistics)
    result = transform(frame)
    after = result.table("person").set_index("person_id")
    for column in stage.UK_COMPANY_CARS_OUTPUT_COLUMNS:
        assert column in after
    holder = after[stage.LIST_PRICE] > 0
    tape = before[SPI_INCOME_RECIPIENT_COLUMN].astype(bool)
    reported = before["frs_company_car_price_band"] > 0
    # Off the tape (the FRS channel, and the SPI channel's undrawn copies and
    # dependants on their twin's pay) the holders are the reported ones, and
    # their pay is untouched.
    assert (holder[~tape] == reported[~tape]).all()
    assert (holder & ~tape & (after["person_support_channel"] == "spi")).any()
    pd.testing.assert_series_equal(
        after.loc[~tape, "employment_income"], before.loc[~tape, "employment_income"]
    )
    # A tape row's holder needs EPB, and its pay plus its benefits is the tape's.
    assert not (holder & tape & (before["hmrc_spi_employment_benefits"] <= 0)).any()
    car, fuel = _benefits(result)
    benefit = pd.Series(car + fuel, index=after.index)
    gap = (
        after.loc[tape, "employment_income"]
        + benefit[tape]
        - before.loc[tape, "employment_income"]
    )
    assert gap.abs().max() < 1e-6
    assert (
        benefit[tape] <= before.loc[tape, "hmrc_spi_employment_benefits"] + 1e-6
    ).all()
    # The carved pay keeps every net-pay contribution, including the tape rows
    # that contribute all of it, and a carved row's UC gainful self-employment
    # follows its carved pay; every other row keeps its flag.
    contributions = after["employee_pension_contributions"]
    assert (contributions[tape] <= after.loc[tape, "employment_income"] + 1e-6).all()
    carved = tape & (benefit > 0)
    assert carved.any()
    assert (holder & tape & (contributions == before["employment_income"])).sum() == 0
    gainful = after[stage.UC_GAINFUL_SELF_EMPLOYMENT]
    expected = uc_gainful_self_employment(
        after.loc[carved, "employment_status"].eq("FT_SELF_EMPLOYED").to_numpy(),
        after.loc[carved, "self_employment_income"].to_numpy(float),
        after.loc[carved, "employment_income"].to_numpy(float),
    )
    assert (gainful[carved].to_numpy() == expected).all()
    pd.testing.assert_series_equal(
        gainful[~carved], before.loc[~carved, stage.UC_GAINFUL_SELF_EMPLOYMENT]
    )
    # Non-holders keep the engine's defaults.
    nobody = after[~holder]
    assert (nobody[stage.CO2_EMISSIONS] == 0).all()
    assert (nobody[stage.FUEL_TYPE] == stage.DEFAULT_FUEL_TYPE).all()
    assert not nobody[stage.PRIVATE_FUEL].any()
    # Holders' invariants.
    holders = after[holder]
    assert (holders[stage.CAPITAL_CONTRIBUTIONS] <= holders[stage.LIST_PRICE]).all()
    electric = holders[stage.FUEL_TYPE] == stage.ELECTRIC
    assert (holders.loc[electric, stage.CO2_EMISSIONS] == 0).all()
    assert not holders.loc[electric, stage.PRIVATE_FUEL].any()
    assert result.mass_log[-1].reason == stage.UK_COMPANY_CARS_MASS_CONSERVATION_REASON
    receipt = transform.checkpoint_metadata()["evidence"]
    assert receipt["stage"] == "company_cars"
    assert receipt["spi_carve_out"]["max_abs_pay_identity_error_gbp"] <= 0.01
    assert set(receipt["checks"]["violations"].values()) == {0}
    # The base of the 2030-31 rise, by fuel class.
    totals = receipt["list_price"]["weighted_total_gbp"]
    electric = holders[stage.FUEL_TYPE] == stage.ELECTRIC
    assert set(totals) == {"electric", "not_electric"}
    assert totals["electric"] == pytest.approx(
        float(
            (
                holders.loc[electric, stage.LIST_PRICE]
                * _person_weights(result).loc[holders.index[electric]]
            ).sum()
        )
    )


def test_the_receipt_measures_what_the_epb_draw_and_the_carve_out_remove(
    statistics,
) -> None:
    """The holders the tape rows lose for want of EPB, the list price the
    carve-out scales away, and the CO2 shares of the cars the output holds
    (items 3 to 5 of the review)."""

    frame = _stage_frame()
    before = frame.table("person").set_index("person_id")
    transform = _transform(statistics)
    result = transform(frame)
    after = result.table("person").set_index("person_id")
    receipt = transform.checkpoint_metadata()["evidence"]
    weights = _person_weights(result)
    no_epb = (
        before[SPI_INCOME_RECIPIENT_COLUMN].astype(bool)
        & (before["frs_company_car_price_band"] > 0)
        & (before["hmrc_spi_employment_benefits"] <= 0)
    )
    dropped = receipt["holders"]["spi_dropped_without_epb"]
    assert dropped["weighted"] == pytest.approx(
        float(weights.loc[no_epb[no_epb].index].sum())
    )
    assert dropped["weighted_list_price_gbp"] > 0
    assert receipt["spi_carve_out"]["weighted_list_price_scaled_away_gbp"] >= 0
    held = after[stage.LIST_PRICE] > 0
    emits = held & (after[stage.FUEL_TYPE] != stage.ELECTRIC)
    low = emits & (after[stage.CO2_EMISSIONS] <= 50)
    assert receipt["co2"]["output"]["weighted_share_1_50"] == pytest.approx(
        float(weights.loc[low[low].index].sum() / weights.loc[emits[emits].index].sum())
    )


def _person_weights(frame) -> pd.Series:
    person = frame.table("person")
    weights = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=frame.table("household")["household_id"].to_numpy(),
    )
    return pd.Series(
        weights.reindex(person["person_household_id"]).to_numpy(),
        index=person["person_id"].to_numpy(),
    )


def test_an_engine_that_prices_a_car_otherwise_misses_the_identity(
    statistics,
) -> None:
    """The second pricing reads the engine, not the carve-out's own sum: an
    engine whose car benefit ignores the list price leaves a scaled car's
    benefit above what came out of pay."""

    def flat(frame):
        car, fuel = _benefits(frame)
        price = frame.table("person")[stage.LIST_PRICE].to_numpy(float)
        return np.where(price > 0, 9_000.0, 0.0), fuel

    transform = dataclasses.replace(_transform(statistics), benefits=flat)
    transform(_stage_frame())
    receipt = transform.checkpoint_metadata()["evidence"]
    assert receipt["spi_carve_out"]["max_abs_pay_identity_error_gbp"] > 1_000.0


def test_receipt_figures_on_too_few_source_households_are_withheld(
    statistics,
) -> None:
    """Every household here is a copy of one of two survey households, so each
    weighted figure rests on fewer than the floor and reads its marker."""

    # The floor's markers, from the helpers that set the floor.
    withheld = {small_cell_share(0.0, 1), small_cell_share(0.0, 2)}
    transform = _transform(statistics)
    transform(_stage_frame(sources=lambda ids: ids % 2))
    receipt = transform.checkpoint_metadata()["evidence"]
    assert all(isinstance(marker, str) for marker in withheld)
    assert receipt["holders"]["weighted"] in withheld
    assert receipt["holders"]["support"] in {support_band(1), support_band(2)}
    assert set(receipt["list_price"]["weighted_total_gbp"].values()) <= withheld
    shares = receipt["fuel"]["weighted_shares"]
    assert set(shares.values()) <= withheld | {0.0, None}
    carve = receipt["spi_carve_out"]
    assert carve["weighted_benefit_carved_gbp"] in withheld
    assert carve["weighted_epb_gbp"] in withheld
    for name in (
        "weighted_list_price_scaled_away_gbp",
        "weighted_list_price_dropped_gbp",
    ):
        assert carve[name] in withheld | {0.0}
    dropped = receipt["holders"]["spi_dropped_without_epb"]
    assert {dropped["weighted"], dropped["weighted_list_price_gbp"]} <= withheld
    for name in ("hybrid_rate_into_1_50", "weighted_share_1_50"):
        assert receipt["co2"]["drawn"][name] in withheld
    assert set(receipt["co2"]["output"].values()) <= withheld


def test_each_invariant_counts_the_rows_that_break_it() -> None:
    """Row 0 is a sound petrol car; each later row breaks one invariant."""

    holders = np.array([True, True, True, True, True, False, True, True, True])
    columns = {
        stage.LIST_PRICE: np.array([30e3, 30e3, 30e3, 30e3, 0.0, 0.0, 30e3, 30e3, 9e3]),
        stage.CO2_EMISSIONS: np.array([120.0, 0.0, 5.0, 0.0, 0, 0, 120, 120, 40]),
        stage.FUEL_TYPE: np.array(
            [
                stage.PETROL,
                stage.PETROL,
                stage.ELECTRIC,
                stage.ELECTRIC,
                stage.ELECTRIC,
                stage.PETROL,
                "LPG",
                stage.PETROL,
                stage.ALTERNATIVE_FUEL,
            ],
            dtype=object,
        ),
        stage.ELECTRIC_RANGE: np.array([0.0, 0, 0, 0, 0, 0, 0, 30.0, 50.0]),
        stage.CAPITAL_CONTRIBUTIONS: np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 10e3]),
        stage.PRIVATE_FUEL: np.array(
            [True, False, False, True, False, True, False, False, False]
        ),
    }
    violations = stage._violations(
        columns,
        holders=holders,
        tape=np.zeros(9, dtype=bool),
        epb=np.zeros(9),
        carved=np.zeros(9),
        pay=np.full(9, 30e3),
        employee_contributions=np.zeros(9),
    )
    assert violations == {
        "non_finite_values": 0,
        "negative_values": 0,
        "holder_without_list_price": 1,
        "inputs_without_a_car": 1,
        "unknown_fuel_type": 1,
        "emitting_car_without_co2": 1,
        "electric_car_with_co2": 1,
        "private_fuel_on_an_electric_car": 1,
        "contributions_above_list_price": 1,
        "electric_range_outside_1_to_50_g_km": 1,
        "spi_benefit_above_epb": 0,
        "employee_contribution_above_carved_pay": 0,
    }
    # A tape row whose carved pay falls below its net-pay contribution, beside
    # one that keeps it exactly; off the tape the stage leaves pay alone.
    tape = np.array([True, True, False, False, False, False, False, False, False])
    violations = stage._violations(
        columns,
        holders=holders,
        tape=tape,
        epb=np.full(9, 3_000.0),
        carved=np.where(tape, 3_000.0, 0.0),
        pay=np.array([0.0, 150.0, 0.0, 0, 0, 0, 0, 0, 0]),
        employee_contributions=np.array([150.0, 150.0, 900.0, 0, 0, 0, 0, 0, 0]),
    )
    assert violations["employee_contribution_above_carved_pay"] == 1


def test_a_carved_row_s_uc_gainful_self_employment_follows_its_carved_pay() -> None:
    """The SPI income stage set the flag on the pay the row drew; a side trade
    of GBP 5,000 beside pay carved from GBP 6,000 to GBP 1,000 becomes gainful
    (item 10 of the second review). Rows the carve-out leaves keep their flag."""

    person = pd.DataFrame(
        {
            "employment_status": ["FT_EMPLOYED", "FT_EMPLOYED", "FT_SELF_EMPLOYED"],
            "self_employment_income": [5_000.0, 5_000.0, 0.0],
            "employment_income": [1_000.0, 1_000.0, 20_000.0],
            stage.UC_GAINFUL_SELF_EMPLOYMENT: [False, False, True],
        }
    )
    flag = stage.refresh_uc_gainful_self_employment(
        person, np.array([True, False, True])
    )
    assert flag.tolist() == [True, False, True]
    unchanged = stage.refresh_uc_gainful_self_employment(person, np.zeros(3, bool))
    assert unchanged.tolist() == [False, False, True]


def test_the_stage_declares_no_fit_weight_records(statistics) -> None:
    """The stage fits no model. A declared empty list read as a fitting stage
    that recorded nothing, which failed the release-cut weights audit (item 1
    of the second review; microcosm#1428 does the same for ISA)."""

    transform = _transform(statistics)
    transform(_stage_frame())
    evidence = snapshot_stage_evidence("company_cars", transform)
    assert "fit_weight_records" not in evidence


def test_without_a_carve_out_the_output_co2_shares_repeat_the_draw_s(
    statistics,
) -> None:
    """The carve-out alone moves the output's CO2 shares off the draw's: on a
    frame without tape rows, co2.output repeats co2.drawn."""

    frame = _stage_frame()
    person = frame.table("person").copy()
    person["person_support_channel"] = "frs"
    person[SPI_INCOME_RECIPIENT_COLUMN] = False
    person["hmrc_spi_employment_benefits"] = 0.0
    frs_only = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        household_weights=np.asarray(frame.weights_for("household").values),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )
    transform = _transform(statistics)
    transform(frs_only)
    co2 = transform.checkpoint_metadata()["evidence"]["co2"]
    assert co2["output"]["weighted_share_1_50"] == pytest.approx(
        co2["drawn"]["weighted_share_1_50"]
    )


class _NoSacrifice:
    def predict_from_uniforms(self, predictors, *, quantiles, sign_uniforms):
        return pd.DataFrame(
            {salary_sacrifice.SALSAC_OUTPUT: 0.0}, index=predictors.index
        )


class _NoSacrificeForest:
    def __init__(self, *, n_estimators: int, seed: int) -> None:
        pass

    def fit(self, frame, predictors, targets, *, weights):
        return _NoSacrifice()


def test_salary_sacrifice_runs_on_the_stage_s_output(
    statistics, monkeypatch: pytest.MonkeyPatch
) -> None:
    """salary_sacrifice refuses a net-pay contribution above the pay that
    remains. The review's two tape rows: PAY 0 with EPB of GBP 3,000 and a
    GBP 150 contribution, and GBP 20,000 of pay all contributed, each with a car
    (item 2 of the second review). Without the contribution's room the stage
    would carve both below their contributions."""

    monkeypatch.setattr(salary_sacrifice, "QRF", _NoSacrificeForest)
    frs = 24
    ids = np.arange(frs + 2) + 1
    spi = ids > frs
    pay = np.concatenate([np.linspace(15_000, 90_000, frs), [3_000.0, 20_000.0]])
    car = np.where(spi, 9, 0)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_benunit_id": ids,
            "age": 40.0,
            "employment_income": pay,
            "frs_company_car_price_band": car,
            "frs_company_car_fuel_code": np.where(spi, 1, 0),
            "frs_company_car_private_fuel": False,
            "frs_company_car_contribution": 0.0,
            "person_support_channel": np.where(spi, "spi", "frs"),
            SPI_INCOME_RECIPIENT_COLUMN: spi,
            "hmrc_spi_employment_benefits": np.where(
                spi, np.where(ids == frs + 1, 3_000.0, 6_000.0), 0.0
            ),
            "employee_pension_contributions": np.where(
                spi, np.where(ids == frs + 1, 150.0, 20_000.0), 1_000.0
            ),
            "employment_status": "FT_EMPLOYED",
            "self_employment_income": 0.0,
            stage.UC_GAINFUL_SELF_EMPLOYMENT: False,
            "salary_sacrifice_asked": np.where(spi, 0, 1),
            salary_sacrifice.SALSAC_OUTPUT: 0.0,
            salary_sacrifice.BONUS_SACRIFICE_SOURCE: 0.0,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_source_id": ids,
            "household_support_channel": np.where(spi, "spi", "frs"),
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household,
        household_weights=np.full(len(ids), 10.0),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )
    carved = _transform(statistics)(frame)
    after = carved.table("person").set_index("person_id")
    # The first keeps a car carved down to its contribution; the second, whose
    # contribution leaves no pay to carve, loses its car and keeps its pay.
    assert after.loc[frs + 1, stage.LIST_PRICE] > 0
    assert after.loc[frs + 1, "employment_income"] == pytest.approx(150.0, abs=0.01)
    assert after.loc[frs + 2, stage.LIST_PRICE] == 0
    assert after.loc[frs + 2, "employment_income"] == 20_000.0
    anchor = {
        "engine": {"package": "policyengine-uk", "version": "toy"},
        "source": {},
        "obr_base_above_threshold": 0.0,
        "threshold": 2000,
        "data_year": 2024,
        "base_year": 2027,
        "earnings_growth": {"factor": 1.0},
        "weight_growth": {"factor": 1.0},
        "upper_earnings_limit": {"weekly": 50_000 / 52, "annual": 50_000.0},
    }
    result = salary_sacrifice.impute_salary_sacrifice(carved, bonus_anchor=anchor)
    out = result.frame.table("person")
    assert (
        out["employee_pension_contributions"] <= out["employment_income"] + 1e-6
    ).all()


def test_the_stage_s_draws_follow_identity_not_row_order(statistics) -> None:
    straight = _transform(statistics)(_stage_frame()).table("person")
    reversed_ = _transform(statistics)(
        _stage_frame(order=lambda n: np.arange(n)[::-1])
    ).table("person")
    columns = list(stage.UK_COMPANY_CARS_OUTPUT_COLUMNS)
    pd.testing.assert_frame_equal(
        straight.set_index("person_id")[columns].sort_index(),
        reversed_.set_index("person_id")[columns].sort_index(),
    )


def test_the_receipt_carries_no_unrounded_counts(statistics) -> None:
    transform = _transform(statistics)
    transform(_stage_frame())
    receipt = transform.checkpoint_metadata()["evidence"]
    holders = receipt["holders"]
    for value in (
        holders["frs_channel"],
        holders["spi_channel_tape"],
        holders["spi_channel_twin_pay"],
        holders["spi_dropped_without_epb"]["count"],
        receipt["fuel"]["drawn_with_private_fuel"],
    ):
        assert isinstance(value, str) or value == 0 or value % 10 == 0 or value < 0


def test_the_stage_refuses_a_frame_without_the_carriers(statistics) -> None:
    frame = _stage_frame()
    person = frame.table("person").drop(columns="frs_company_car_price_band")
    bare = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=np.asarray(frame.weights_for("household").values),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )
    with pytest.raises(stage.CompanyCarsError, match="carriers"):
        _transform(statistics)(bare)


def test_the_vendored_facts_are_hmrc_s_2024_25_tables_on_the_pinned_feed() -> None:
    statistics = load_hmrc_company_car_statistics()
    assert statistics.car_recipients == 920_000.0
    assert statistics.car_taxable_value_gbp == 3.07e9
    assert statistics.car_class_1a_nics_gbp == 4.2e8
    assert statistics.fuel_recipients == 40_000.0
    assert statistics.fuel_type_shares()["fully_electric"] == pytest.approx(465 / 915)
    assert len(statistics.evidence()["resource_sha256"]) == 64


def test_the_loader_types_chronicle_s_2024_25_rows(statistics) -> None:
    assert statistics.car_recipients == 930_000.0
    assert statistics.car_taxable_value_gbp == 3.0e9
    assert statistics.fuel_class_1a_nics_gbp == 2.0e7
    assert statistics.fuel_type_shares()["fully_electric"] == pytest.approx(0.5)
    assert statistics.nonzero_co2_shares()[HMRC_LOW_EMISSION_BAND] == pytest.approx(0.4)
    assert statistics.income_range_recipients["income_200000_plus"] == 50_000.0
    assert statistics.region_recipients == {
        "E12000007": 150_000.0,
        "S92000003": 60_000.0,
    }
    assert statistics.evidence()["source_commit"] == "synthetic"


def test_a_cell_the_feed_lacks_is_refused_never_read_as_zero() -> None:
    payload = synthetic_company_car_statistics_payload()
    payload["rows"] = [
        row
        for row in payload["rows"]
        if ".table2a." not in row["source_record_id"]
        or not row["source_record_id"].endswith("co2_165_to_169.recipients")
    ]
    with pytest.raises(HMRCCompanyCarStatisticsError, match="co2_165_to_169"):
        load_hmrc_company_car_statistics(payload=payload, verify_feed_identity=False)


def test_tables_that_disagree_beyond_their_rounding_are_refused() -> None:
    payload = synthetic_company_car_statistics_payload()
    for row in payload["rows"]:
        if row["source_record_id"].endswith(
            "table2c.recipients.ty2024.fully_electric.recipients"
        ):
            row["value"] = 400_000.0
    with pytest.raises(HMRCCompanyCarStatisticsError, match="Table 2C"):
        load_hmrc_company_car_statistics(payload=payload, verify_feed_identity=False)


def test_the_declaration_names_the_2024_25_record_sets() -> None:
    declared = {
        "artifact_role": "benefits_in_kind_facts",
        "resource": "hmrc_benefits_in_kind_facts.json",
        "record_sets": list(HMRC_COMPANY_CAR_RECORD_SETS),
        "mapped_build_period": 2024,
    }
    check_company_car_statistics_declaration(
        declared, stage="company_cars", period=2024
    )
    with pytest.raises(HMRCCompanyCarStatisticsError, match="record_sets"):
        check_company_car_statistics_declaration(
            {**declared, "record_sets": declared["record_sets"][:-1]},
            stage="company_cars",
            period=2024,
        )
    with pytest.raises(HMRCCompanyCarStatisticsError, match="period"):
        check_company_car_statistics_declaration(
            declared, stage="company_cars", period=2025
        )
