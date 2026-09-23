"""Reserved SPI income band rows: stacking, propensity, resample and manifest pins."""

from __future__ import annotations

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
    SPI_INCOME_BAND_DONOR_SEED,
    SPI_INCOME_BAND_DONORS_PER_BAND,
    UKSPIIncomeBandDonorStageTransform,
    _assert_band_donor_stage_parameters,
    income_band_lower_bound,
    load_hmrc_itl_band_taxpayers,
    spi_income_band_propensity,
    stack_spi_income_band_donors,
)
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
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

BANDS = SPI_INCOME_BAND_DONOR_LOWER_BOUNDS
TAXPAYERS = {
    200_000: 359_000.0,
    500_000: 61_000.0,
    1_000_000: 20_000.0,
    2_000_000: 10_000.0,
}


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


def _support_frame(n_households: int = 60, *, reverse: bool = False):
    ids = np.arange(1, n_households + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.r_[ids, ids + 1_000],
            "person_benunit_id": np.r_[ids, ids],
            "person_household_id": np.r_[ids, ids],
            "age": np.r_[np.full(n_households, 45), np.full(n_households, 12)],
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


def test_stack_reserves_band_exact_positive_rows_with_one_carrier_each() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    propensity = spi_income_band_propensity(donor, lower_bounds=BANDS)
    frame = _support_frame()
    result = stack_spi_income_band_donors(
        frame,
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        donors_per_band=5,
        seed=SPI_INCOME_BAND_DONOR_SEED,
        taxpayer_period=2024,
    )
    household = result.frame.table("household")
    person = result.frame.table("person")
    donors = household.loc[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    assert len(donors) == 5 * len(BANDS)
    assert donors[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN].all()
    assert (
        donors[support_channel_column("household")] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ).all()
    assert (
        donors["household_support_clone_index"] == SPI_INCOME_BAND_DONOR_CLONE_INDEX
    ).all()
    assert sorted(donors[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN].unique()) == list(
        BANDS
    )
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
    weights = result.frame.weights_for("household").values[-len(donors) :]
    assert (weights > 0.0).all()
    by_band = {int(row["lower_bound"]): row for row in result.band_rows}
    for band in BANDS:
        assert by_band[band]["donor_households"] == 5
        assert by_band[band]["carriers"] == 5
        assert by_band[band]["donor_weight"] == pytest.approx(TAXPAYERS[band] / 5)
        assert by_band[band]["weighted_taxpayers"] == pytest.approx(TAXPAYERS[band])
    assert result.frame.mass_log[-1].new_total == pytest.approx(
        frame.weights_for("household").total + sum(TAXPAYERS.values())
    )
    assert result.frame.mass_log[-1].declared_factor is None
    # The base and ordinary synthetic rows keep their weights.
    assert np.allclose(
        result.frame.weights_for("household").values[: len(frame.table("household"))],
        frame.weights_for("household").values,
    )


def test_stack_is_permutation_stable_and_refuses_a_second_stack() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    propensity = spi_income_band_propensity(donor, lower_bounds=BANDS)
    first = stack_spi_income_band_donors(
        _support_frame(),
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        donors_per_band=3,
        seed=3,
        taxpayer_period=2024,
    )
    second = stack_spi_income_band_donors(
        _support_frame(reverse=True),
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        donors_per_band=3,
        seed=3,
        taxpayer_period=2024,
    )
    pick = lambda result: set(  # noqa: E731
        result.frame.table("household").loc[
            lambda t: t[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR], "household_id"
        ]
    )
    assert pick(first) == pick(second)
    with pytest.raises(ValueError, match="already stacked"):
        stack_spi_income_band_donors(
            first.frame,
            propensity=propensity,
            band_taxpayers=TAXPAYERS,
            donors_per_band=3,
            seed=3,
            taxpayer_period=2024,
        )


def test_stack_refuses_zero_band_weight_and_missing_support_channel() -> None:
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    propensity = spi_income_band_propensity(donor, lower_bounds=BANDS)
    with pytest.raises(ValueError, match="positive initial weight"):
        stack_spi_income_band_donors(
            _support_frame(),
            propensity=propensity,
            band_taxpayers={**TAXPAYERS, 2_000_000: 0.0},
            donors_per_band=3,
            seed=3,
            taxpayer_period=2024,
        )
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
            band_taxpayers=TAXPAYERS,
            donors_per_band=1,
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


def test_resample_refuses_missing_carriers_or_undeclared_band() -> None:
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
    assert _assert_band_donor_stage_parameters(
        stage, seed=SPI_INCOME_BAND_DONOR_SEED
    ) == (SPI_INCOME_BAND_DONORS_PER_BAND)
    drifted = SourceStageSpec.from_mapping(
        {
            **stage.__dict__,
            "operations": [
                {
                    **dict(stage.operations[0].parameters),
                    "kind": stage.operations[0].kind,
                    "donors_per_band": 119,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="donors_per_band"):
        _assert_band_donor_stage_parameters(drifted, seed=SPI_INCOME_BAND_DONOR_SEED)
    with pytest.raises(ValueError, match=r"parameter\(s\) \['seed'\]"):
        _assert_band_donor_stage_parameters(stage, seed=SPI_INCOME_BAND_DONOR_SEED + 1)


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
        sample_fraction=4 / SPI_INCOME_BAND_DONORS_PER_BAND,
        donor_table=_raw_tape(),
        band_taxpayers=TAXPAYERS,
    )
    result = transform(_support_frame())
    assert result.weights_for("household").kind is WeightKind.IMPORTANCE
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "spi_income_band_donors"
    assert evidence["donors_per_band"] == 4
    assert [row["lower_bound"] for row in evidence["bands"]] == list(BANDS)
    assert sum(evidence["carrier_region_counts"].values()) == 4 * len(BANDS)
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
