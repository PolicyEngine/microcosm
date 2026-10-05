"""Reserved SPI income band rows: stacking, propensity, resample and manifest pins."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_band_donors import (
    HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    PERSON_IS_SPI_INCOME_BAND_CARRIER,
    SPI_INCOME_BAND_DONOR_CLONE_INDEX,
    SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
    SPI_INCOME_BAND_DONOR_LOWER_BOUNDS,
    SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON,
    SPI_INCOME_BAND_DONOR_SEED,
    SPI_INCOME_BAND_DONOR_TAPE_SEED,
    SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT,
    SPI_INCOME_BAND_MINIMUM_DONORS,
    UKSPIIncomeBandDonorStageTransform,
    _assert_band_donor_stage_parameters,
    age_band_lower_bound,
    income_band_lower_bound,
    load_hmrc_itl_band_taxpayers,
    spi_income_band_donor_operation_parameters,
    spi_income_band_donor_plan,
    spi_income_band_donor_propensity,
    spi_income_band_donor_seats,
    spi_income_band_propensity,
    stack_spi_income_band_donors,
)
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPIDonorAgeModel,
    _resample_band_donor_leaves,
    prepare_spi_donor_table,
)
from microcosm.build.uk_runtime.spi_support import (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    build_uk_spi_support_channel,
    support_channel_column,
)
from microcosm.frame import WeightKind
from test_support.paths import paths_for

_IDENTITY_TOOL_PATH = (
    paths_for("microcosm-build").repository
    / "tools"
    / "verify_uk_identity_stability.py"
)

BANDS = SPI_INCOME_BAND_DONOR_LOWER_BOUNDS
TAXPAYERS = {
    200_000: 359_000.0,
    500_000: 61_000.0,
    1_000_000: 20_000.0,
    2_000_000: 10_000.0,
}
#: Band masses a 200-household synthetic frame can fund at full scale.
SMALL_TAXPAYERS = {
    200_000: 40.0,
    500_000: 12.0,
    1_000_000: 6.0,
    2_000_000: 3.0,
}


def _age_model(state_pension_age: int = 66) -> SPIDonorAgeModel:
    """A flat single-year population, so the engine-free lane needs no engine."""

    return SPIDonorAgeModel(
        populations={
            (sex, age): 100.0 for sex in ("MALE", "FEMALE") for age in range(0, 91)
        },
        state_pension_age=state_pension_age,
        source="fixture",
    )


def _propensity() -> pd.DataFrame:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    return spi_income_band_propensity(donor, lower_bounds=BANDS)


def _stack_small(frame, **overrides):
    arguments = {
        "propensity": _propensity(),
        "band_taxpayers": SMALL_TAXPAYERS,
        "minimum_donors": 3,
        "maximum_donor_weight": 10.0,
        "seed": SPI_INCOME_BAND_DONOR_SEED,
        "taxpayer_period": 2024,
    }
    arguments.update(overrides)
    return stack_spi_income_band_donors(frame, **arguments)


def _donor_source_ids(result) -> dict[int, float]:
    household = result.frame.table("household")
    donors = household.loc[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    return dict(
        zip(
            donors["household_source_id"].astype(int),
            donors[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN],
            strict=True,
        )
    )


def _raw_tape(rows: int = 240, *, seed: int = 0) -> pd.DataFrame:
    """A synthetic tape with both sexes, two regions and every reserved band.

    Non-composite rows sit in the 45-55 age band, the band the support
    frame's candidates occupy; every twelfth row of the top band is a
    composite (AGERANGE -1) at the tape's composite weight.
    """

    rng = np.random.default_rng(seed)
    records = []
    for index in range(rows):
        row = {column: 0.0 for column in SPI_DONOR_REQUIRED_COLUMNS}
        band = index % 6  # 0,1: below 200k; 2: 200k; 3: 500k; 4: 1m; 5: 2m+
        pay = (30_000.0, 80_000.0, 260_000.0, 620_000.0, 1_300_000.0, 2_600_000.0)[band]
        composite = band == 5 and index % 12 == 5
        row.update(
            {
                "SEX": float(1 + (index // 6) % 2),
                "FACT": 31.0 if composite else float(rng.integers(40, 80)),
                "GORCODE": float(7 if index % 3 else 8),  # London or South East
                "AGERANGE": -1.0 if composite else 4.0,
                "PAY": pay,
                "DIVIDENDS": pay * 0.05,
                "INCBBS": pay * 0.01,
            }
        )
        row["TEI"] = row["PAY"]
        row["TII"] = row["DIVIDENDS"] + row["INCBBS"]
        row["TI"] = row["TEI"] + row["TII"]
        records.append(row)
    return pd.DataFrame(records)


def _support_frame(
    n_households: int = 60,
    *,
    reverse: bool = False,
    dependant_age: int = 12,
    dependant_is_claimant: bool = False,
):
    ids = np.arange(1, n_households + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.r_[ids, ids + 1_000],
            "person_benunit_id": np.r_[ids, ids],
            "person_household_id": np.r_[ids, ids],
            "age": np.r_[
                np.full(n_households, 45), np.full(n_households, dependant_age)
            ],
            "is_uc_claimant": np.r_[
                np.ones(n_households, dtype=bool),
                np.full(n_households, dependant_is_claimant, dtype=bool),
            ],
            "gender": np.r_[
                np.where(ids % 2 == 0, "MALE", "FEMALE"),
                np.full(n_households, "FEMALE"),
            ],
            "employment_income": np.r_[
                np.linspace(10_000.0, 90_000.0, n_households), np.zeros(n_households)
            ],
        }
    )
    if reverse:
        person = person.iloc[::-1].reset_index(drop=True)
    benunit = pd.DataFrame({"benunit_id": ids})
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": np.linspace(100.0, 200.0, n_households),
            "region": np.where(ids % 3 == 0, "SOUTH_EAST", "LONDON"),
        }
    )
    support = build_uk_spi_support_channel(
        person,
        benunit,
        household,
        spi_household_count=10,
        seed=42,
        source_year=2024,
        zero_weight_declarations=(),
    )
    return uk_national_frame(
        person=support.person,
        benunit=support.benunit,
        household=support.household,
        time_period="2024",
        weight_kind=support.household_weight_kind,
        household_weights=support.household["household_weight"].to_numpy(dtype=float),
        mass_log=support.mass_log,
    )


def test_income_band_lower_bound_maps_edges_and_open_top() -> None:
    values = np.asarray([0.0, 199_999.0, 200_000.0, 999_999.0, 1_000_000.0, 5e6])
    assert income_band_lower_bound(values, lower_bounds=BANDS).tolist() == [
        0.0,
        0.0,
        200_000.0,
        500_000.0,
        1_000_000.0,
        2_000_000.0,
    ]


def test_propensity_table_is_fact_weighted_band_share_per_cell() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    table = spi_income_band_propensity(donor, lower_bounds=BANDS)
    assert set(table["band"]) == set(BANDS)
    assert (table["propensity"] > 0.0).all() and (table["propensity"] <= 1.0).all()
    cell = table.loc[(table["region"] == "LONDON") & (table["gender"] == "MALE")]
    grouped = cell.groupby("age_band")["propensity"].sum()
    assert (grouped <= 1.0 + 1e-12).all()
    # A tape with no record in a reserved band refuses.
    thin = donor.loc[donor["total_income"] < 2_000_000]
    with pytest.raises(ValueError, match="no record with total income from 2000000"):
        spi_income_band_propensity(thin, lower_bounds=BANDS)


def test_plan_keeps_every_donor_at_or_below_the_maximum_weight() -> None:
    # Given the 2024-25 Table 2.5 taxpayers, when the plan is read, then a
    # band seats the minimum unless that would put a donor above the cap.
    plan = spi_income_band_donor_plan(TAXPAYERS)
    assert {band: count for band, (count, _) in plan.items()} == {
        200_000: 1_436,
        500_000: 244,
        1_000_000: SPI_INCOME_BAND_MINIMUM_DONORS,
        2_000_000: SPI_INCOME_BAND_MINIMUM_DONORS,
    }
    for band, (count, weight) in plan.items():
        assert weight <= SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT
        assert weight * count == pytest.approx(TAXPAYERS[band], abs=1e-6)
    with pytest.raises(ValueError, match="positive published taxpayers"):
        spi_income_band_donor_plan({**TAXPAYERS, 2_000_000: 0.0})
    with pytest.raises(ValueError, match="minimum_donors"):
        spi_income_band_donor_plan(TAXPAYERS, minimum_donors=0)
    with pytest.raises(ValueError, match="maximum_donor_weight"):
        spi_income_band_donor_plan(TAXPAYERS, maximum_donor_weight=0.0)


def test_seats_scale_only_when_the_plan_outgrows_the_frame() -> None:
    plan = spi_income_band_donor_plan(TAXPAYERS)
    expected = sum(count for count, _ in plan.values())
    # Given a frame the plan fits in, the scale is one and the seats the plan's.
    scale, seats = spi_income_band_donor_seats(
        plan, eligible_households=40_000, maximum_donor_household_share=0.15
    )
    assert scale == 1.0
    assert seats == {band: count for band, (count, _) in plan.items()}
    # Given a small frame, every band is scaled alike and none is dropped.
    scale, seats = spi_income_band_donor_seats(
        plan, eligible_households=60, maximum_donor_household_share=0.15
    )
    assert scale == pytest.approx(0.15 * 60 / expected)
    assert seats == {200_000: 6, 500_000: 1, 1_000_000: 1, 2_000_000: 1}
    with pytest.raises(ValueError, match="no eligible base household"):
        spi_income_band_donor_seats(plan, eligible_households=0)


def test_stack_reserves_band_rows_and_conserves_every_region_mass() -> None:
    frame = _support_frame(200)
    result = _stack_small(frame)
    household = result.frame.table("household")
    person = result.frame.table("person")
    donors = household.loc[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    # 40 / 10 = 4 donors in the first band; the others seat the minimum.
    assert donors[
        SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN
    ].value_counts().to_dict() == {
        200_000.0: 4,
        500_000.0: 3,
        1_000_000.0: 3,
        2_000_000.0: 3,
    }
    assert donors[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN].all()
    assert (
        donors[support_channel_column("household")] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ).all()
    assert (
        donors["household_support_clone_index"] == SPI_INCOME_BAND_DONOR_CLONE_INDEX
    ).all()
    assert donors["household_source_id"].is_unique
    assert (
        household.loc[
            ~household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR],
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
        ]
        == 0.0
    ).all()
    carriers = person.loc[person[PERSON_IS_SPI_INCOME_BAND_CARRIER]]
    assert len(carriers) == len(donors)
    assert set(carriers["person_household_id"]) == set(donors["household_id"])
    assert (carriers["age"] >= 16).all()

    # Equal weights within a band, carrying the band's published taxpayers.
    weights = pd.Series(
        result.frame.weights_for("household").values, index=household.index
    )
    by_band = {int(row["lower_bound"]): row for row in result.band_rows}
    for band in BANDS:
        band_weights = weights[
            donors.index[donors[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN] == band]
        ]
        assert band_weights.nunique() == 1
        assert band_weights.iloc[0] <= 10.0
        assert band_weights.sum() == pytest.approx(SMALL_TAXPAYERS[band])
        assert by_band[band]["weighted_taxpayers"] == pytest.approx(
            SMALL_TAXPAYERS[band]
        )
        assert by_band[band]["donor_households"] == by_band[band]["carriers"]
        assert (
            by_band[band]["donor_households"]
            == by_band[band]["expected_donor_households"]
        )

    # The mass is reallocated, not created: the total is bit-equal, every
    # region keeps its mass, and the receipt declares factor one.
    incoming = frame.weights_for("household")
    assert float(weights.sum()) == float(incoming.total)
    before = (
        pd.Series(incoming.values)
        .groupby(frame.table("household")["region"].to_numpy())
        .sum()
    )
    after = weights.groupby(household["region"].to_numpy()).sum()
    assert after.to_dict() == pytest.approx(before.to_dict(), rel=1e-12)
    receipt = result.frame.mass_log[-1]
    assert receipt.declared_factor == 1.0
    assert receipt.old_total == receipt.new_total == float(incoming.total)
    assert receipt.reason == SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON
    evidence = result.evidence()
    assert evidence["seating_scale"] == 1.0 and evidence["mass_scale"] == 1.0
    assert evidence["reallocated_mass"] == pytest.approx(sum(SMALL_TAXPAYERS.values()))
    assert evidence["mass"]["old_total"] == evidence["mass"]["new_total"]

    # Every incumbent of a region is scaled by that region's one factor, and
    # the zero-weight support copies stay at zero.
    incumbents = len(frame.table("household"))
    ratio = pd.Series(weights.to_numpy()[:incumbents] / np.asarray(incoming.values))
    regions = pd.Series(frame.table("household")["region"].to_numpy())
    positive = np.asarray(incoming.values) > 0.0
    factors = {row["stratum"]: row["factor"] for row in evidence["funding"]}
    for region, factor in factors.items():
        assert 0.5 <= factor < 1.0
        rows = (regions == region).to_numpy() & positive
        assert ratio[rows].to_numpy() == pytest.approx(factor, rel=1e-9)
    assert (weights.to_numpy()[:incumbents][~positive] == 0.0).all()


def test_stack_scales_seats_then_mass_on_a_frame_too_small_for_the_plan() -> None:
    # Given the published masses on a 60-household frame, the plan's 1,920
    # donors do not fit: the seats shrink to the declared share of the frame.
    frame = _support_frame()
    result = stack_spi_income_band_donors(
        frame, propensity=_propensity(), band_taxpayers=TAXPAYERS, taxpayer_period=2024
    )
    evidence = result.evidence()
    assert evidence["expected_donor_count"] == 1_920
    assert evidence["donor_count"] == 9
    assert evidence["seating_scale"] == pytest.approx(0.15 * 60 / 1_920)
    assert evidence["mass_scale"] == 1.0
    assert float(result.frame.weights_for("household").total) == float(
        frame.weights_for("household").total
    )
    # When the seated mass would take more than the declared share of a
    # region, every band's weight is scaled alike and the total still holds.
    heavy = _stack_small(
        _support_frame(),
        band_taxpayers={band: value * 400.0 for band, value in SMALL_TAXPAYERS.items()},
        maximum_donor_weight=4_000.0,
    )
    evidence = heavy.evidence()
    assert 0.0 < evidence["mass_scale"] < 1.0
    assert min(row["factor"] for row in evidence["funding"]) == pytest.approx(0.5)
    for row in evidence["bands"]:
        assert row["donor_weight"] == pytest.approx(
            row["planned_donor_weight"] * evidence["mass_scale"]
        )
    assert evidence["mass"]["old_total"] == evidence["mass"]["new_total"]


def test_stack_is_keyed_on_identity_and_refuses_a_second_stack() -> None:
    first = _stack_small(_support_frame(200))
    # The same households are seated in the same bands whatever the row order.
    second = _stack_small(_support_frame(200, reverse=True))
    assert _donor_source_ids(first) == _donor_source_ids(second)
    # A different seed seats different households.
    other = _stack_small(_support_frame(200), seed=SPI_INCOME_BAND_DONOR_SEED + 1)
    assert set(_donor_source_ids(other)) != set(_donor_source_ids(first))
    with pytest.raises(ValueError, match="already stacked"):
        _stack_small(first.frame)


def test_dependants_aged_16_to_19_are_never_carriers() -> None:
    """A dependant keeps its twin's values, so it cannot carry a band draw.

    The second member of every household is 17 and the tape favours that age
    band; only claimants and partners are candidates (uk-data#504,
    microcosm#1095), so the 17-year-olds are seated only when the frame flags
    them as claimants or partners.
    """

    propensity = _propensity()
    teen_band = int(age_band_lower_bound(np.asarray([17.0]))[0])
    favoured = (
        propensity.assign(age_band=teen_band, propensity=1.0)
        .drop_duplicates(["region", "gender", "age_band", "band"])
        .reset_index(drop=True)
    )
    propensity = pd.concat(
        [propensity.loc[propensity["age_band"] != teen_band], favoured],
        ignore_index=True,
    )

    def carrier_ages(*, dependant_is_claimant: bool) -> pd.Series:
        frame = _support_frame(
            200, dependant_age=17, dependant_is_claimant=dependant_is_claimant
        )
        person = _stack_small(frame, propensity=propensity).frame.table("person")
        return person.loc[person[PERSON_IS_SPI_INCOME_BAND_CARRIER], "age"]

    assert carrier_ages(dependant_is_claimant=False).eq(45).all()
    assert carrier_ages(dependant_is_claimant=True).eq(17).any()


def test_stack_follows_weight_times_propensity_not_the_sample() -> None:
    # Given a tape whose band propensities are equal across the two regions,
    # seats follow household weight: the heavier half of the frame is seated
    # more often than its half share of the rows across many seeds.
    frame = _support_frame(200)
    weight = pd.Series(
        frame.weights_for("household").values,
        index=frame.table("household")["household_id"].to_numpy(),
    )
    heavy = weight > weight[weight > 0.0].median()
    heavy_seats = seats = 0
    propensity = _propensity()
    for seed in range(40):
        result = _stack_small(frame, propensity=propensity, seed=seed)
        ids = list(_donor_source_ids(result))
        heavy_seats += int(heavy.loc[ids].sum())
        seats += len(ids)
    assert heavy_seats / seats > 0.53


def test_stack_refuses_bad_inputs_and_a_missing_support_channel() -> None:
    propensity = _propensity()
    with pytest.raises(ValueError, match="positive published taxpayers"):
        _stack_small(
            _support_frame(), band_taxpayers={**SMALL_TAXPAYERS, 2_000_000: 0.0}
        )
    with pytest.raises(ValueError, match="maximum_reallocation_share"):
        _stack_small(_support_frame(), maximum_reallocation_share=1.0)
    with pytest.raises(ValueError, match="maximum_donor_household_share"):
        _stack_small(_support_frame(), maximum_donor_household_share=0.0)
    raw = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "age": [40],
                "gender": ["MALE"],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1], "region": ["LONDON"]}),
        household_weights=[1.0],
        time_period="2024",
    )
    with pytest.raises(ValueError, match="after spi_support_channel"):
        stack_spi_income_band_donors(
            raw,
            propensity=propensity,
            band_taxpayers=SMALL_TAXPAYERS,
            minimum_donors=1,
            seed=3,
            taxpayer_period=2024,
        )


def test_resample_gives_carriers_band_conditional_leaves_and_receipts() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "region": ["LONDON", "LONDON", "SOUTH_EAST", "LONDON", "LONDON"],
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN: [
                0.0,
                200_000.0,
                500_000.0,
                1_000_000.0,
                2_000_000.0,
            ],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41, 51, 52],
            "person_household_id": [1, 2, 3, 4, 5, 5],
            PERSON_IS_SPI_INCOME_BAND_CARRIER: [False, True, True, True, True, False],
        }
    )
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        person[column] = 1.0
    spi_people = pd.Series([False, True, True, True, True, True], index=person.index)
    factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)
    factors["dividend_income"] = 2.0
    updated, receipt = _resample_band_donor_leaves(
        person,
        donor,
        household=household,
        spi_people=spi_people,
        uprating_factors=factors,
        lower_bounds=BANDS,
        regional_pool_minimum=5,
        seed=44,
    )
    carriers = updated.loc[updated[PERSON_IS_SPI_INCOME_BAND_CARRIER]]
    # Every carrier's leaves came from a tape record in its band: the pay
    # column identifies the band in the synthetic tape.
    pay = carriers["hmrc_spi_pay"].to_numpy()
    assert pay.tolist() == [260_000.0, 620_000.0, 1_300_000.0, 2_600_000.0]
    assert carriers["dividend_income"].to_numpy() == pytest.approx(pay * 0.05 * 2.0)
    untouched = updated.loc[~updated[PERSON_IS_SPI_INCOME_BAND_CARRIER]]
    assert (untouched[list(SPI_INCOME_QRF_OUTPUT_COLUMNS)] == 1.0).all().all()
    assert receipt["carriers"] == 4 and receipt["weighting"] == "FACT"
    rows = {row["lower_bound"]: row for row in receipt["bands"]}
    assert rows[2_000_000]["pool_composite_records"] > 0
    assert rows[2_000_000]["realized_min_total_income"] >= 2_000_000
    assert rows[500_000]["realized_max_total_income"] < 1_000_000
    assert all(row["carriers"] == 1 for row in receipt["bands"])

    # A carrier matches its region only where that regional band pool holds
    # at least the minimum; the expectation is read off the tape itself.
    def regional_pool(lower: int, upper: float, region: str) -> int:
        band = donor.loc[
            (donor["total_income"] >= lower) & (donor["total_income"] < upper)
        ]
        return int((band["region"] == region).sum())

    assert rows[500_000]["regional_matched_carriers"] == int(
        regional_pool(500_000, 1_000_000, "SOUTH_EAST") >= 5
    )
    assert rows[200_000]["regional_matched_carriers"] == int(
        regional_pool(200_000, 500_000, "LONDON") >= 5
    )


def test_resample_cuts_the_pools_on_the_total_income_a_record_realises() -> None:
    # Given: pay uprated threefold, so the tape's 80k records (published total
    # 84.8k, below every reserved band) realise 240k-and-over on the frame
    # while the 260k records realise 780k-and-over.
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "LONDON"],
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN: [200_000.0, 500_000.0],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 21],
            "person_household_id": [1, 2],
            PERSON_IS_SPI_INCOME_BAND_CARRIER: [True, True],
        }
    )
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        person[column] = 1.0
    factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)
    factors["hmrc_spi_pay"] = 3.0
    updated, receipt = _resample_band_donor_leaves(
        person,
        donor,
        household=household,
        spi_people=pd.Series([True, True], index=person.index),
        uprating_factors=factors,
        lower_bounds=BANDS,
        regional_pool_minimum=5,
        seed=44,
    )
    # Then: the 200k carrier drew an 80k record (its uprated pay is 240k), the
    # 500k carrier a 260k record, and the pools are the uprated bands.
    pay = updated.loc[updated[PERSON_IS_SPI_INCOME_BAND_CARRIER], "hmrc_spi_pay"]
    assert pay.tolist() == [240_000.0, 780_000.0]
    rows = {row["lower_bound"]: row for row in receipt["bands"]}
    assert rows[200_000]["pool_records"] == int(
        (donor["hmrc_spi_pay"] == 80_000.0).sum()
    )
    assert rows[500_000]["pool_records"] == int(
        (donor["hmrc_spi_pay"] == 260_000.0).sum()
    )
    assert 200_000 <= rows[200_000]["realized_mean_total_income"] < 500_000
    assert 500_000 <= rows[500_000]["realized_mean_total_income"] < 1_000_000
    assert receipt["carriers_outside_band"] == 0
    assert all(row["carriers_outside_band"] == 0 for row in receipt["bands"])
    assert receipt["pool_basis"].startswith("uprated total income")


def test_resample_draw_is_keyed_on_the_carrier_not_on_the_row_order() -> None:
    # Given twelve carriers in one band, when the person rows are reversed or
    # a carrier is removed, then every remaining carrier draws the same record.
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    # The synthetic tape's records are identical within a band; a small
    # per-record amount tells them apart without moving any out of its band.
    donor["dividend_income"] = donor["dividend_income"] + np.arange(len(donor))
    ids = np.arange(1, 13)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": "LONDON",
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN: 200_000.0,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 10,
            "person_source_id": ids,
            "person_household_id": ids,
            PERSON_IS_SPI_INCOME_BAND_CARRIER: True,
        }
    )
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        person[column] = 0.0
    arguments = {
        "household": household,
        "uprating_factors": dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0),
        "lower_bounds": BANDS,
        "regional_pool_minimum": 5,
        "seed": 44,
    }

    def draw(frame: pd.DataFrame) -> pd.Series:
        updated, receipt = _resample_band_donor_leaves(
            frame,
            donor,
            spi_people=pd.Series(True, index=frame.index),
            **arguments,
        )
        assert receipt["draw_key"] == "person_source_id"
        assert receipt["salt"] == "spi_income_band_donor_leaf_draw"
        return updated.set_index("person_source_id")["dividend_income"]

    first = draw(person)
    assert first.nunique() > 1
    reversed_rows = draw(person.iloc[::-1].reset_index(drop=True))
    assert reversed_rows.sort_index().equals(first.sort_index())
    fewer = draw(person.iloc[3:].reset_index(drop=True))
    assert fewer.sort_index().equals(first.sort_index().iloc[3:])
    # The key is the lineage id, not the frame's person id.
    renumbered = draw(person.assign(person_id=person["person_id"] + 7))
    assert renumbered.sort_index().equals(first.sort_index())


def test_resample_refuses_missing_carriers_undeclared_band_or_non_recipient() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    household = pd.DataFrame(
        {
            "household_id": [1],
            "region": ["LONDON"],
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN: [300_000.0],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [1],
            "person_household_id": [1],
            PERSON_IS_SPI_INCOME_BAND_CARRIER: [True],
        }
    )
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        person[column] = 0.0
    with pytest.raises(ValueError, match="undeclared band"):
        _resample_band_donor_leaves(
            person,
            donor,
            household=household,
            spi_people=pd.Series([True]),
            uprating_factors=dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0),
            lower_bounds=BANDS,
            regional_pool_minimum=5,
            seed=1,
        )
    household[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN] = 200_000.0
    with pytest.raises(ValueError, match="must be SPI recipients"):
        _resample_band_donor_leaves(
            person,
            donor,
            household=household,
            spi_people=pd.Series([False]),
            uprating_factors=dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0),
            lower_bounds=BANDS,
            regional_pool_minimum=5,
            seed=1,
        )
    person[PERSON_IS_SPI_INCOME_BAND_CARRIER] = False
    with pytest.raises(ValueError, match="no reserved carriers"):
        _resample_band_donor_leaves(
            person,
            donor,
            household=household,
            spi_people=pd.Series([True]),
            uprating_factors=dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0),
            lower_bounds=BANDS,
            regional_pool_minimum=5,
            seed=1,
        )


def test_packaged_stage_binds_to_the_reviewed_constants_and_refuses_drift() -> None:
    spec = load_country_spec("uk")
    stage = spec.sources.stage_map()["spi_income_band_donors"]
    _assert_band_donor_stage_parameters(stage, seed=SPI_INCOME_BAND_DONOR_SEED)
    declared = dict(stage.operations[0].parameters)
    assert (
        declared
        == spi_income_band_donor_operation_parameters()[
            "stack_income_band_donor_households"
        ]
    )
    assert declared["declared_factor"] == 1.0
    assert declared["conservation"] == "exact_total"
    assert declared["reason"] == SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON
    for key, value in (
        ("minimum_donors_per_band", 119),
        ("maximum_donor_weight", 500.0),
        ("funding_strata", ["region", "tenure_type"]),
        ("declared_factor", 1.01),
    ):
        drifted = SourceStageSpec.from_mapping(
            {
                **stage.__dict__,
                "operations": [
                    {**declared, "kind": stage.operations[0].kind, key: value}
                ],
            }
        )
        with pytest.raises(ValueError, match=key):
            _assert_band_donor_stage_parameters(
                drifted, seed=SPI_INCOME_BAND_DONOR_SEED
            )
    with pytest.raises(ValueError, match=r"parameter\(s\) \['seed'\]"):
        _assert_band_donor_stage_parameters(stage, seed=SPI_INCOME_BAND_DONOR_SEED + 1)


def test_tape_is_prepared_as_the_income_stage_prepares_it() -> None:
    # The declared tape seed is the income stage's stage-1 seed, and that
    # stage declares the age draw the donor stage shares.
    stages = load_country_spec("uk").sources.stage_map()
    income = {op.kind: op for op in stages["hmrc_spi_income_spine"].operations}
    assert (
        income["fit_weighted_qrf_stage1"].parameters["seed"]
        == SPI_INCOME_BAND_DONOR_TAPE_SEED
    )
    assert "draw_spi_donor_ages_by_population" in income
    declared = stages["spi_income_band_donors"].operations[0].parameters
    assert declared["tape_preparation_seed"] == SPI_INCOME_BAND_DONOR_TAPE_SEED

    # The stage's propensity table is the band shares of the tape prepared
    # with that seed and the age model: a State Pension recipient the tape
    # publishes no age band for (a composite) sits at or over State Pension
    # age, never in a working-age band.
    raw = _raw_tape()
    composite = raw["AGERANGE"] == -1.0
    raw.loc[composite, "SRP"] = 9_000.0
    raw.loc[composite, ["TEI", "TI"]] += 9_000.0
    model = _age_model()
    table = spi_income_band_donor_propensity(raw, age_model=model)
    prepared = prepare_spi_donor_table(
        raw, seed=SPI_INCOME_BAND_DONOR_TAPE_SEED, age_model=model
    )
    pd.testing.assert_frame_equal(
        table, spi_income_band_propensity(prepared, lower_bounds=BANDS)
    )
    assert (prepared.loc[composite.to_numpy(), "age"] >= 66).all()
    # The synthetic tape's non-composite rows are all aged 45 to 54, so the
    # composites are the only mass in the 65-to-74 band.
    assert set(table["age_band"]) == {45, 65}


def test_vendored_table_2_5_counts_feed_the_build_year_weights() -> None:
    counts = load_hmrc_itl_band_taxpayers(2024)
    assert counts == {
        200_000: 359_000.0,
        500_000: 61_000.0,
        1_000_000: 20_000.0,
        2_000_000: 10_000.0,
    }
    with pytest.raises(ValueError, match="exactly one"):
        load_hmrc_itl_band_taxpayers(1999)


def test_stage_transform_runs_on_a_synthetic_tape_and_receipts() -> None:
    spec = load_country_spec("uk")
    stage = spec.sources.stage_map()["spi_income_band_donors"]
    transform = UKSPIIncomeBandDonorStageTransform(
        "put2223uk.tab",
        stage=stage,
        seed=SPI_INCOME_BAND_DONOR_SEED,
        sample_fraction=0.01,
        donor_table=_raw_tape(),
        band_taxpayers=TAXPAYERS,
        age_model=_age_model(),
    )
    frame = _support_frame()
    result = transform(frame)
    assert result.weights_for("household").kind is WeightKind.IMPORTANCE
    assert float(result.weights_for("household").total) == float(
        frame.weights_for("household").total
    )
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "spi_income_band_donors"
    assert evidence["minimum_donors_per_band"] == SPI_INCOME_BAND_MINIMUM_DONORS
    assert evidence["maximum_donor_weight"] == SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT
    # The survey-side sample fraction is provenance only: the seats follow
    # the frame, not the fraction.
    assert evidence["sample_fraction"] == 0.01
    assert evidence["donor_count"] == 9 and evidence["seating_scale"] < 1.0
    assert [row["lower_bound"] for row in evidence["bands"]] == list(BANDS)
    assert sum(evidence["carrier_region_counts"].values()) == 9
    assert evidence["draw"] == {
        "seed": SPI_INCOME_BAND_DONOR_SEED,
        "salt": "spi_income_band_donor_draw",
    }
    assert transform.output_columns() == (
        HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
        SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
        PERSON_IS_SPI_INCOME_BAND_CARRIER,
    )


def test_resample_receipts_a_frame_without_the_donor_stage() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    household = pd.DataFrame({"household_id": [1], "region": ["LONDON"]})
    person = pd.DataFrame({"person_id": [1], "person_household_id": [1]})
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        person[column] = 0.0
    updated, receipt = _resample_band_donor_leaves(
        person,
        donor,
        household=household,
        spi_people=pd.Series([True]),
        uprating_factors=dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0),
        lower_bounds=BANDS,
        regional_pool_minimum=5,
        seed=1,
    )
    assert receipt["carriers"] == 0 and "did not run" in receipt["skipped"]
    assert updated.equals(person)


def _identity_tool():
    spec = importlib.util.spec_from_file_location(
        "verify_uk_identity_stability", _IDENTITY_TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reweighted(frame, weights):
    return uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
        mass_log=frame.mass_log,
    )


def test_e8_receipt_reconstructs_the_seating_from_ids() -> None:
    # Given a stacked frame, when the E8 block reruns the stage on the
    # pre-donor frame it reconstructs from the artifact, then the seats, the
    # carriers and the band weights are the stored ones in any row order.
    tool = _identity_tool()
    propensity = _propensity()
    stacked = stack_spi_income_band_donors(
        _support_frame(200),
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        taxpayer_period=2024,
    ).frame
    problems: dict[str, object] = {}
    receipt = tool._e8_band_donors(
        stacked,
        problems,
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        permutation_seed=123,
    )
    assert problems == {}
    assert receipt["seating_recomputed"] is True
    assert receipt["donor_households"] == 27
    assert 0.0 < receipt["seating_scale"] < 1.0
    assert receipt["mass"]["declared_factor"] == 1.0
    assert [row["lower_bound"] for row in receipt["bands"]] == list(BANDS)
    assert set(receipt["funding"]) == {"LONDON", "SOUTH_EAST"}

    # Without the tape's propensities the layer is checked for structure and
    # mass only, and the receipt says so.
    problems = {}
    receipt = tool._e8_band_donors(
        stacked, problems, propensity=None, band_taxpayers=None, permutation_seed=123
    )
    assert problems == {} and receipt["seating_recomputed"] is False
    assert "pass --spi-tab" in receipt["seating_scope"]


def test_e8_receipt_reports_a_seating_the_ids_do_not_reproduce() -> None:
    tool = _identity_tool()
    propensity = _propensity()
    frame = _support_frame(200)
    stacked = stack_spi_income_band_donors(
        frame, propensity=propensity, band_taxpayers=TAXPAYERS, taxpayer_period=2024
    ).frame
    arguments = {
        "propensity": propensity,
        "band_taxpayers": TAXPAYERS,
        "permutation_seed": 123,
    }
    # A seating drawn under another seed is not the one the ids reproduce.
    other = stack_spi_income_band_donors(
        frame,
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        seed=SPI_INCOME_BAND_DONOR_SEED + 1,
        taxpayer_period=2024,
    ).frame
    problems: dict[str, object] = {}
    tool._e8_band_donors(other, problems, **arguments)
    assert "band_donor_seats_stored" in problems
    assert "band_donor_carriers_stored" in problems

    # Unequal weights within a band are a stored-layer defect.
    household = stacked.table("household")
    weights = np.asarray(stacked.weights_for("household").values, dtype=float).copy()
    first_donor = int(np.flatnonzero(household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR])[0])
    weights[first_donor] *= 1.5
    problems = {}
    tool._e8_band_donors(_reweighted(stacked, weights), problems, **arguments)
    assert problems["band_donor_layer"] == {"bands_with_unequal_donor_weights": 1}

    # A frame without the stage's mass record, or without the layer, refuses.
    problems = {}
    tool._e8_band_donors(
        uk_national_frame(
            person=stacked.table("person"),
            benunit=stacked.table("benunit"),
            household=household,
            time_period="2024",
            weight_kind=WeightKind.IMPORTANCE,
            household_weights=stacked.weights_for("household").values,
        ),
        problems,
        **arguments,
    )
    assert problems["band_donor_mass_record"] == "missing"
    with pytest.raises(ValueError, match="donor layer is absent"):
        tool._e8_band_donors(frame, {}, **arguments)
