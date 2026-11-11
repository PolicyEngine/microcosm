"""Reserved SPI income band rows: stacking, propensity, resample and manifest pins."""

from __future__ import annotations

import importlib.util
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import copy_id_table, spi_band_donors
from microcosm.build.uk_runtime.cgt_residential_split import (
    CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
    HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
)
from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_COPIES_COLUMN,
    CGT_SUPPORT_COPY_INDEX_COLUMN,
    HOUSEHOLD_IS_CGT_SUPPORT_COPY,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.geographic_expand import expand_uk_geographic_clones
from microcosm.build.uk_runtime.high_value_home_support import (
    HIGH_VALUE_SUPPORT_COPIES_COLUMN,
    split_high_value_support_households,
)
from microcosm.build.uk_runtime.household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import UK_WITHHELD, rounded_count
from microcosm.build.uk_runtime.regional_home_values import (
    assign_regional_home_values,
)
from microcosm.build.uk_runtime.salary_sacrifice import (
    SALSAC_OUTPUT,
    SALSAC_PRE_CONVERSION_PAY_COLUMN,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    PERSON_IS_SPI_INCOME_BAND_CARRIER,
    SPI_INCOME_BAND_CAPPED_SEATING,
    SPI_INCOME_BAND_DONOR_CLONE_INDEX,
    SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
    SPI_INCOME_BAND_DONOR_LOWER_BOUNDS,
    SPI_INCOME_BAND_DONOR_MASS_CHANGE_REASON,
    SPI_INCOME_BAND_DONOR_SEED,
    SPI_INCOME_BAND_DONOR_TAPE_SEED,
    SPI_INCOME_BAND_FULL_SEATING,
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
    BASE_FRS_SUPPORT_CHANNEL,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_PROPERTY_FINANCE_COSTS_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    build_uk_spi_support_channel,
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame import WeightKind, engine_tables
from test_support.microcosm_build.uk_clone_first_frames import (
    IdTables,
    assert_copy_ids_follow_lineage,
    copy_households,
)
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
                # Annex A bridge codes: pay is every record's main source, no
                # trade pages, below State Pension age; composites carry none.
                "MAINSRCE": -1.0 if composite else 1.0,
                "SEINC_NUM": -1.0 if composite else 0.0,
                "SPA": -1.0 if composite else 0.0,
            }
        )
        row["TEI"] = row["PAY"]
        row["TII"] = row["DIVIDENDS"] + row["INCBBS"]
        row["TI"] = row["TEI"] + row["TII"]
        records.append(row)
    return pd.DataFrame(records)


def _raw_support_tables(
    n_households: int = 60,
    *,
    reverse: bool = False,
    dependant_age: int = 12,
    dependant_is_claimant: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The FRS base tables the support frames copy: a claimant and a dependant."""

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
    return person, benunit, household


def _support_frame(
    n_households: int = 60,
    *,
    reverse: bool = False,
    dependant_age: int = 12,
    dependant_is_claimant: bool = False,
    expected_spi_households: int = 10,
):
    return _spi_support_frame(
        *_raw_support_tables(
            n_households,
            reverse=reverse,
            dependant_age=dependant_age,
            dependant_is_claimant=dependant_is_claimant,
        ),
        expected_spi_households=expected_spi_households,
    )


def _clone_first_support_frame(
    n_households: int,
    n_clones: int,
    *,
    expected_spi_households: int = 10,
    tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    **options,
):
    """``_support_frame``'s households as geographic clones before stage 16.

    The spine's geographic expansion (``geographic_expand``) repeats every
    row as clone ``k`` at ``id + k x 10**9`` and an equal share of its
    household's weight, with the household's source columns, which the SPI
    support channel reads, and the channel then copies each clone's
    households, as the clone-first spine does (microcosm#931). ``tables``
    replaces the raw tables ``options`` build.
    """

    person, benunit, household = (
        _raw_support_tables(n_households, **options) if tables is None else tables
    )
    raw = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )
    cloned = engine_tables(expand_uk_geographic_clones(raw, clones=n_clones).frame)
    return _spi_support_frame(
        cloned["person"],
        cloned["benunit"],
        cloned["household"],
        expected_spi_households=expected_spi_households,
    )


def _spi_support_frame(person, benunit, household, *, expected_spi_households: int):
    # The SPI copies are a keyed Bernoulli share at 10 / n (microcosm#931), so
    # their count is binomial. At seed 42 both regions keep copies at the 60
    # and 200 households these tests build (11 copies, 2 in SOUTH_EAST, at
    # 60; 13, 5 in SOUTH_EAST, at 200), so every funding stratum has SPI mass.
    support = build_uk_spi_support_channel(
        person,
        benunit,
        household,
        expected_spi_households=expected_spi_households,
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
    assert result.seating_scale == 1.0
    assert evidence["seating_scale"] == "full" and evidence["mass_scale"] == 1.0
    assert evidence["clone_count"] == 1
    assert evidence["reallocated_mass"] == pytest.approx(sum(SMALL_TAXPAYERS.values()))
    assert evidence["mass"]["old_total"] == evidence["mass"]["new_total"]

    # Every incumbent of a region is scaled by that region's one factor, and
    # the zero-weight support copies stay at zero. The result keeps each
    # stratum's factor; the receipt gives the funding in two booleans and
    # withholds every funding row's masses and factor (microcosm#931).
    incumbents = len(frame.table("household"))
    ratio = pd.Series(weights.to_numpy()[:incumbents] / np.asarray(incoming.values))
    regions = pd.Series(frame.table("household")["region"].to_numpy())
    positive = np.asarray(incoming.values) > 0.0
    factors = {row["stratum"]: row["factor"] for row in result.funding_rows}
    for region, factor in factors.items():
        assert 0.5 <= factor < 1.0
        rows = (regions == region).to_numpy() & positive
        assert ratio[rows].to_numpy() == pytest.approx(factor, rel=1e-9)
    assert (weights.to_numpy()[:incumbents][~positive] == 0.0).all()
    assert evidence["funding_within_share"] is True
    assert evidence["funding_matches_bands"] is True
    for row in evidence["funding"]:
        assert row["stratum"] in factors
        for key in ("incumbent_mass", "donor_mass", "factor"):
            assert row[key] == UK_WITHHELD


def test_spi_copies_and_donors_take_their_base_roots_ids_plus_the_radix() -> None:
    # Every SPI-channel household copies a base household, which its
    # household_source_id names: the SPI copy at 1 and the income band donor at
    # 2 times the spi_support_channel radix, on every id column (microcosm#931).
    frame = _support_frame(200)
    result = _stack_small(frame)
    channel = support_channel_column("household")
    household = frame.table("household")
    base = household.loc[household[channel] == BASE_FRS_SUPPORT_CHANNEL]
    person = frame.table("person")
    before = {
        "household": base,
        "person": person.loc[person["person_household_id"].isin(base["household_id"])],
        "benunit": frame.table("benunit"),
    }

    checked = assert_copy_ids_follow_lineage(
        before,
        result.frame,
        field="spi_support_channel",
        key_columns=("household_source_id",),
        ordinal_column=support_clone_index_column("household"),
    )
    donors = result.frame.table("household")[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]
    after_channel = result.frame.table("household")[channel]
    assert checked == int((after_channel == SPI_SYNTHETIC_SUPPORT_CHANNEL).sum())
    assert int(donors.sum()) > 0
    assert (
        result.frame.table("household")
        .loc[donors, support_clone_index_column("household")]
        .eq(SPI_INCOME_BAND_DONOR_CLONE_INDEX)
        .all()
    )
    # The identity tool's old-spine gate reads both layers as on the table.
    gate = _identity_tool()._copy_id_rows_off_the_declared_table
    assert gate(frame.table("household")) == 0
    assert gate(result.frame.table("household")) == 0


def test_stack_scales_seats_then_mass_on_a_frame_too_small_for_the_plan() -> None:
    # Given the published masses on a 60-household frame, the plan's 1,920
    # donors do not fit: the seats shrink to the declared share of the frame.
    frame = _support_frame()
    result = stack_spi_income_band_donors(
        frame, propensity=_propensity(), band_taxpayers=TAXPAYERS, taxpayer_period=2024
    )
    evidence = result.evidence()
    assert evidence["expected_donor_count"] == 1_920
    # The capped receipt counts the donors in rounded source households.
    assert result.donor_count == result.donor_source_households == 9
    assert evidence["donor_count"] == rounded_count(9)
    assert result.seating_scale == pytest.approx(0.15 * 60 / 1_920)
    # The receipt says only that the cap bound: the exact scale gives back
    # the eligible household count.
    assert evidence["seating_scale"] == "capped"
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
    assert min(row["factor"] for row in heavy.funding_rows) == pytest.approx(0.5)
    assert evidence["funding_within_share"] is True
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
    household = frame.table("household")
    weight = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"].to_numpy(),
    )
    # The heavier half of the FRS base, the households seats are drawn from:
    # a median over every row would count the SPI copies, whose number is
    # binomial (microcosm#931), and leave the heavy set short of half.
    base = household[support_channel_column("household")].eq(BASE_FRS_SUPPORT_CHANNEL)
    heavy = weight > weight[base.to_numpy()].median()
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
    # No drawn record's own value, such as a band's lowest or highest, is in
    # the receipt, and a band whose carriers drew fewer than three records
    # withholds its mean (UKDS disclosure control).
    for row in receipt["bands"]:
        assert "realized_min_total_income" not in row
        assert "realized_max_total_income" not in row
        assert row["realized_mean_total_income"] == "<3"
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
    # while the 260k records realise 780k-and-over. Twenty carriers in each
    # band draw from at least ten of its records, so each band's mean is
    # recorded rather than withheld.
    donor = prepare_spi_donor_table(_raw_tape(), seed=1)
    per_band = 20
    ids = np.arange(1, 2 * per_band + 1)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": ["LONDON"] * len(ids),
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN: [200_000.0] * per_band
            + [500_000.0] * per_band,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 10 + 1,
            "person_household_id": ids,
            PERSON_IS_SPI_INCOME_BAND_CARRIER: [True] * len(ids),
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
        spi_people=pd.Series(True, index=person.index),
        uprating_factors=factors,
        lower_bounds=BANDS,
        regional_pool_minimum=5,
        seed=44,
    )
    # Then: every 200k carrier drew an 80k record (its uprated pay is 240k),
    # every 500k carrier a 260k record, and the pools are the uprated bands.
    pay = updated.loc[updated[PERSON_IS_SPI_INCOME_BAND_CARRIER], "hmrc_spi_pay"]
    assert pay.tolist() == [240_000.0] * per_band + [780_000.0] * per_band
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


def test_resample_pools_ignore_restricted_finance_costs() -> None:
    # Given: restricted finance costs of 200k on every record below 200k. At
    # the stage's factor of 1.0, counted as income they would lift the 30k
    # records (published total 31.8k) and the 80k records (84.8k) into the
    # 200k pool, which is what the pre-fix pools did; the 260k records carry
    # a modest cost too. TAX_CRED carries the 20% relief.
    raw = _raw_tape()
    low = raw["PAY"] < 200_000.0
    raw.loc[low, "LLIR_RESTRICT_AMT_TOT"] = 200_000.0
    raw.loc[low, "TAX_CRED"] = 40_000.0
    mid = raw["PAY"] == 260_000.0
    raw.loc[mid, "LLIR_RESTRICT_AMT_TOT"] = 10_000.0
    raw.loc[mid, "TAX_CRED"] = 2_000.0
    donor = prepare_spi_donor_table(raw, seed=1)
    household = pd.DataFrame(
        {
            "household_id": [1],
            "region": ["LONDON"],
            SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN: [200_000.0],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11],
            "person_household_id": [1],
            PERSON_IS_SPI_INCOME_BAND_CARRIER: [True],
        }
    )
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        person[column] = 1.0
    factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)
    updated, receipt = _resample_band_donor_leaves(
        person,
        donor,
        household=household,
        spi_people=pd.Series([True], index=person.index),
        uprating_factors=factors,
        lower_bounds=BANDS,
        regional_pool_minimum=5,
        seed=44,
    )
    # Then: every pool is the one the tape's income leaves give (the 200k pool
    # holds the 260k records alone and the 500k pool the 620k records alone),
    # so the costs cut no pool...
    rows = {row["lower_bound"]: row for row in receipt["bands"]}
    assert rows[200_000]["pool_records"] == int(
        (donor["hmrc_spi_pay"] == 260_000.0).sum()
    )
    assert rows[500_000]["pool_records"] == int(
        (donor["hmrc_spi_pay"] == 620_000.0).sum()
    )
    assert receipt["carriers_outside_band"] == 0
    assert "finance costs cut no pool" in receipt["pool_basis"]
    # ...and the drawn record's costs travel with its leaves.
    carrier = updated.loc[updated[PERSON_IS_SPI_INCOME_BAND_CARRIER]].iloc[0]
    assert carrier["hmrc_spi_pay"] == 260_000.0
    assert carrier[SPI_PROPERTY_FINANCE_COSTS_COLUMN] == 10_000.0


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
    assert evidence["seating_scale"] == "capped"
    assert evidence["donor_count"] == rounded_count(9)
    assert [row["lower_bound"] for row in evidence["bands"]] == list(BANDS)
    # The carriers' regions, each count rounded on distinct source households.
    household = result.table("household")
    carriers = result.table("person").loc[
        lambda people: people[PERSON_IS_SPI_INCOME_BAND_CARRIER]
    ]
    region_of = household.set_index("household_id")["region"]
    per_region = carriers["person_household_id"].map(region_of).value_counts()
    assert int(per_region.sum()) == 9
    assert evidence["carrier_region_counts"] == {
        str(region): rounded_count(int(count)) for region, count in per_region.items()
    }
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
    stage = stack_spi_income_band_donors(
        _support_frame(200),
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        taxpayer_period=2024,
    )
    stacked = stage.frame
    evidence = stage.evidence()
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
    assert receipt["seating_scale"] == evidence["seating_scale"] == "capped"
    assert receipt["mass"]["declared_factor"] == 1.0
    # The receipt gives the layer in the stage's capped forms (microcosm#931):
    # the donors and each band's in rounded distinct source households, and
    # the stage's own funding rows, masses and factors withheld, with the
    # funding rule's verdict.
    assert receipt["donor_households"] == evidence["donor_count"] == rounded_count(27)
    assert [row["lower_bound"] for row in receipt["bands"]] == list(BANDS)
    assert [row["donor_households"] for row in receipt["bands"]] == [
        row["donor_households"] for row in evidence["bands"]
    ]
    assert receipt["funding"] == evidence["funding"]
    assert [row["stratum"] for row in receipt["funding"]] == ["LONDON", "SOUTH_EAST"]
    assert receipt["funding_within_share"] is evidence["funding_within_share"] is True

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


# Seating per geographic clone (microcosm#931). The clone-first spine gives
# every household its clone; each clone seats its own donors, by the same walk
# and the same seats, and funds them from its own regions.

_CLONE = "household_clone_index"
_PUBLISHED = {"band_taxpayers": TAXPAYERS, "taxpayer_period": 2024}
#: The stage gate's declaration with the small frames' seating constants.
_SMALL_GATE_PARAMETERS = {
    "stage": "spi_income_band_donors",
    "check": "spi_income_band_donor_support",
    "minimum_donors_per_band": 3,
    "maximum_donor_weight": 10.0,
    "minimum_funding_factor": 0.5,
    "maximum_band_taxpayer_deviation": 0.5,
    "maximum_relative_mass_deviation": 1e-9,
    "band_lower_bounds": list(BANDS),
}
#: ``uk_frame_content_identity`` of the frames stage 17 reads and writes below,
#: computed with the stage at 50b6e629b, before it seated per geographic clone
#: (microcosm#931). On one clone the stage keeps every value, weight and mass
#: record, so the identities hold. Identities compare within the pinned
#: environment only (``content_identity``).
_ONE_CLONE_STAGE_17_INPUT_IDENTITIES = {
    200: "129e44480d4409444baf231a25477d50a8c2dbcb2e525af5f13a53a353908e50",
    60: "13ab2b40acb59e04cfd902f83dc0f0bdc69924adbc0c834315ac5610fc404ddd",
}
_ONE_CLONE_STAGE_17_IDENTITIES = {
    "full_seating": "d804531b98002638548b29c3c9e2dbb7a9363e19a079d502dab9323816ebae8f",
    "capped_seating": "2f8afbee4d45fea6c1787c4b183243bc49f260c9c490de06bddb238db3db7683",
    "scaled_mass": "5751c98409fa2e2d382ba35eb114ac1c10b2ccad59d0ac9de7c7db7840233a08",
}


def _clone_of(result) -> pd.Series:
    """Each household id's geographic clone, 0 on a frame without the index."""

    household = result.frame.table("household")
    clones = (
        household[_CLONE].to_numpy()
        if _CLONE in household.columns
        else np.zeros(len(household), dtype=np.int64)
    )
    return pd.Series(clones, index=household["household_id"].to_numpy())


def _seating(result) -> dict[int, dict[int, float]]:
    """Each clone's seats: the donor's true source household -> its band."""

    household = result.frame.table("household")
    donors = household.loc[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    clones = donors["household_id"].map(_clone_of(result)).to_numpy()
    return {
        int(clone): dict(
            zip(
                rows["source_household_id"].astype(int),
                rows[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN],
                strict=True,
            )
        )
        for clone, rows in donors.groupby(clones)
    }


def _carriers(result) -> dict[int, set[int]]:
    """Each clone's carriers, by the person id each copy was made from."""

    person = result.frame.table("person")
    carriers = person.loc[person[PERSON_IS_SPI_INCOME_BAND_CARRIER]]
    clones = carriers["person_household_id"].map(_clone_of(result)).to_numpy()
    return {
        int(clone): set(rows["person_source_id"].astype(int))
        for clone, rows in carriers.groupby(clones)
    }


def _weights_by_id(frame) -> pd.Series:
    return pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=frame.table("household")["household_id"].to_numpy(),
    )


def _mass_by_clone_and_region(frame) -> dict[tuple[int, str], float]:
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    return {
        (int(clone), str(region)): float(mass)
        for (clone, region), mass in pd.Series(weights)
        .groupby([household[_CLONE].to_numpy(), household["region"].to_numpy()])
        .sum()
        .items()
    }


def _with_tables(frame, **tables):
    """``frame`` with tables replaced; its weights and mass records kept."""

    weights = frame.weights_for("household")
    return uk_national_frame(
        person=tables.get("person", frame.table("person")),
        benunit=tables.get("benunit", frame.table("benunit")),
        household=tables.get("household", frame.table("household")),
        time_period="2024",
        weight_kind=weights.kind,
        household_weights=weights.values,
        mass_log=frame.mass_log,
    )


def _small_gate(evidence):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="spi_income_band_donors",
        check="spi_income_band_donor_support",
        parameters=_SMALL_GATE_PARAMETERS,
    )


def test_one_clone_frames_keep_their_content_identity_through_stage_17() -> None:
    support = {200: _support_frame(200), 60: _support_frame()}
    for households, frame in support.items():
        assert (
            uk_frame_content_identity(frame)
            == _ONE_CLONE_STAGE_17_INPUT_IDENTITIES[households]
        ), "the synthetic input moved: re-pin both tables, recording what moved it"
    results = {
        "full_seating": _stack_small(support[200]),
        "capped_seating": stack_spi_income_band_donors(
            support[60], propensity=_propensity(), **_PUBLISHED
        ),
        "scaled_mass": _stack_small(
            support[60],
            band_taxpayers={
                band: value * 400.0 for band, value in SMALL_TAXPAYERS.items()
            },
            maximum_donor_weight=4_000.0,
        ),
    }
    assert results["scaled_mass"].mass_scale < 1.0
    assert results["capped_seating"].seating_scale < 1.0
    assert {
        name: uk_frame_content_identity(result.frame)
        for name, result in results.items()
    } == _ONE_CLONE_STAGE_17_IDENTITIES


def test_a_one_clone_index_seats_weighs_and_funds_bit_for_bit_as_no_index() -> None:
    plain_frame = _support_frame(200)
    household = plain_frame.table("household")
    indexed_frame = _with_tables(
        plain_frame,
        household=household.assign(**{_CLONE: np.zeros(len(household), "int64")}),
    )
    plain, indexed = _stack_small(plain_frame), _stack_small(indexed_frame)

    assert _seating(indexed) == _seating(plain)
    assert _carriers(indexed) == _carriers(plain)
    np.testing.assert_array_equal(
        indexed.frame.weights_for("household").values,
        plain.frame.weights_for("household").values,
    )
    assert indexed.frame.mass_log == plain.frame.mass_log
    # The funding strata gain the clone in their labels and nothing else.
    assert [row["factor"] for row in indexed.funding_rows] == [
        row["factor"] for row in plain.funding_rows
    ]
    assert [row["stratum"] for row in indexed.funding_rows] == [
        f"{row['stratum']}|0" for row in plain.funding_rows
    ]
    assert plain.evidence()["funding_strata"] == ["region"]
    assert indexed.evidence()["funding_strata"] == ["region", _CLONE]
    assert indexed.evidence()["clone_count"] == plain.evidence()["clone_count"] == 1


@pytest.mark.parametrize("n_clones", [3, 7])
def test_clone_zero_seats_what_one_clone_seats_and_every_clone_takes_the_seats(
    n_clones: int,
) -> None:
    # Given the same households as one clone and as clones made before the
    # SPI support channel,
    one = _stack_small(_support_frame(200))
    many = _stack_small(_clone_first_support_frame(200, n_clones))
    seated, seated_once = _seating(many), _seating(one)[0]

    # then clone 0 seats exactly the one clone's households, bands and
    # carriers, every clone seats the same number in each band, and every
    # other clone draws households of its own.
    assert sorted(seated) == list(range(n_clones))
    assert seated[0] == seated_once
    assert _carriers(many)[0] == _carriers(one)[0]
    seats = pd.Series(seated_once).value_counts().to_dict()
    for clone in range(n_clones):
        assert pd.Series(seated[clone]).value_counts().to_dict() == seats
    for clone in range(1, n_clones):
        assert set(seated[clone]) != set(seated[0])

    # Clone 0's weights, its donors' included, are the one clone's over the
    # clone count while the common mass scale is the one clone's, as here,
    # where it is one at both. On these small frames they nest to rtol 1e-12,
    # not bit for bit: the exact-total correction moves one incumbent row by a
    # few ulps of the total, in whichever clone it lands, and the 1/K split
    # rounds when K is not a power of two, as 3 and 7 are not, so uncorrected
    # rows also miss by about one ulp.
    weights_once = _weights_by_id(one.frame)
    clone_zero = _clone_of(many).loc[lambda clones: clones.eq(0)].index
    assert set(clone_zero) == set(weights_once.index)
    np.testing.assert_allclose(
        _weights_by_id(many.frame).loc[clone_zero].to_numpy() * n_clones,
        weights_once.loc[clone_zero].to_numpy(),
        rtol=1e-12,
    )
    rows_once = {row["lower_bound"]: row for row in one.band_rows}
    for row in many.band_rows:
        once = rows_once[row["lower_bound"]]
        assert row["donor_weight"] * n_clones == pytest.approx(
            once["donor_weight"], rel=1e-12
        )
        assert row["weighted_taxpayers"] == pytest.approx(
            once["weighted_taxpayers"], rel=1e-12
        )
        for key in ("expected_donor_households", "donor_households", "carriers"):
            assert row[key] == n_clones * once[key]
    factors_once = {row["stratum"]: row["factor"] for row in one.funding_rows}
    factors = {row["stratum"]: row["factor"] for row in many.funding_rows}
    assert set(factors) == {
        f"{region}|{clone}" for region in factors_once for clone in range(n_clones)
    }
    for region, factor in factors_once.items():
        assert factors[f"{region}|0"] == pytest.approx(factor, rel=1e-12)
    evidence = many.evidence()
    assert evidence["clone_count"] == n_clones
    assert evidence["mass_scale"] == one.mass_scale == 1.0
    assert evidence["donor_count"] == n_clones * one.evidence()["donor_count"]
    assert evidence["expected_donor_count"] == n_clones * 13


def test_the_seats_count_source_households_so_the_cap_binds_alike_at_any_k() -> None:
    # Given a frame too small for the published plan, the household share's
    # cap binds. Counted in rows, three clones would treble the eligible
    # households and lift the cap; counted in source households, they do not.
    propensity = _propensity()
    one = stack_spi_income_band_donors(
        _support_frame(), propensity=propensity, **_PUBLISHED
    )
    many = stack_spi_income_band_donors(
        _clone_first_support_frame(60, 3), propensity=propensity, **_PUBLISHED
    )
    assert 0.0 < one.seating_scale < 1.0
    assert many.seating_scale == one.seating_scale
    assert many.evidence()["seating_scale"] == "capped"
    assert many.evidence()["eligible_households"] == rounded_count(60)
    assert one.evidence()["eligible_households"] == rounded_count(60)
    seats = pd.Series(_seating(one)[0]).value_counts().to_dict()
    for seated in _seating(many).values():
        assert pd.Series(seated).value_counts().to_dict() == seats


def test_each_clone_funds_its_donors_from_its_own_regions() -> None:
    frame = _clone_first_support_frame(200, 3)
    result = _stack_small(frame)
    assert _mass_by_clone_and_region(result.frame) == pytest.approx(
        _mass_by_clone_and_region(frame), rel=1e-12
    )
    # A band's donors weigh one weight in every clone.
    household = result.frame.table("household")
    weights = _weights_by_id(result.frame)
    donors = household.loc[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    for band, rows in donors.groupby(SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN):
        assert weights.loc[rows["household_id"]].nunique() == 1, band
    # Each clone and region gives up its own donors' mass.
    donor_mass = (
        weights.loc[donors["household_id"]]
        .groupby(
            [
                donors[_CLONE].to_numpy(),
                donors["region"].to_numpy(),
            ]
        )
        .sum()
    )
    for row in result.funding_rows:
        region, clone = row["stratum"].split("|")
        assert row["donor_mass"] == pytest.approx(
            float(donor_mass.get((int(clone), region), 0.0)), rel=1e-12
        )
        assert row["factor"] == pytest.approx(
            1.0 - row["donor_mass"] / row["incumbent_mass"], rel=1e-12
        )


def test_one_mass_scale_serves_every_clone_when_one_clone_region_binds() -> None:
    # Given clone 1 at a five-hundredth of its weight, its regions cannot fund
    # their donors at the band weights,
    frame = _clone_first_support_frame(200, 2)
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
    weights[household[_CLONE].eq(1).to_numpy()] *= 0.002
    light = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )
    result = _stack_small(light, sample_fraction=0.01)
    evidence = result.evidence()

    # then the binding stratum gives up exactly the declared share, and its
    # scale serves every band in every clone, so each band keeps one weight.
    factors = {row["stratum"]: row["factor"] for row in result.funding_rows}
    binding = min(factors, key=factors.get)
    assert 0.0 < evidence["mass_scale"] < 1.0
    assert binding.endswith("|1")
    assert factors[binding] == pytest.approx(0.5, rel=1e-12)
    assert all(factor > 0.5 for name, factor in factors.items() if name != binding)
    out = result.frame.table("household")
    donor_weights = _weights_by_id(result.frame).loc[
        out.loc[out[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR], "household_id"]
    ]
    bands = out.loc[
        out[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR],
        SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
    ].to_numpy()
    for row in evidence["bands"]:
        band_weights = donor_weights.to_numpy()[bands == row["lower_bound"]]
        assert np.unique(band_weights).size == 1
        assert band_weights[0] == row["donor_weight"]
        assert row["donor_weight"] == pytest.approx(
            row["planned_donor_weight"] * evidence["mass_scale"], rel=1e-12
        )
    assert _mass_by_clone_and_region(result.frame) == pytest.approx(
        _mass_by_clone_and_region(light), rel=1e-12
    )
    # The stage gate reconciles the scaled receipt band by band, and reads the
    # binding stratum's funding, at the declared share to the gate's relative
    # tolerance, from the receipt's booleans; its details give no factor.
    assert evidence["funding_within_share"] is True
    assert evidence["funding_matches_bands"] is True
    gate = _small_gate(evidence)
    assert gate.passed, gate.failures
    assert "smallest_funding_factor" not in gate.details
    assert gate.details["funding_within_share"] is True


def test_a_clone_without_candidates_is_refused_without_a_count() -> None:
    frame = _clone_first_support_frame(200, 2)
    person = frame.table("person").copy()
    clone_of = frame.table("household").set_index("household_id")[_CLONE]
    in_clone_one = person["person_household_id"].map(clone_of).eq(1).to_numpy()
    person.loc[in_clone_one, "is_uc_claimant"] = False

    # The refusal names the band and the clone's shortfall, never a count.
    with pytest.raises(ValueError) as caught:
        _stack_small(_with_tables(frame, person=person))
    assert str(caught.value) == (
        "SPI income band donors found fewer candidate households with positive "
        "weight and propensity for the band from 2000000 than the band seats in a "
        "geographic clone."
    )


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (
            lambda clones: clones.astype(float),
            "SPI income band donors seat each geographic clone on an integer "
            "'household_clone_index' column; it has dtype float64.",
        ),
        (
            lambda clones: pd.array(
                [None, *clones.to_numpy()[1:].tolist()], dtype="Int64"
            ),
            "A household row carries no 'household_clone_index'; SPI income band "
            "donors seat each geographic clone.",
        ),
        (
            lambda clones: np.r_[[-1, -1], clones.to_numpy()[2:]],
            "A household row carries a negative 'household_clone_index'; SPI "
            "income band donors seat each geographic clone.",
        ),
        (
            lambda clones: clones.replace({1: 2}).to_numpy(),
            "A 'household_clone_index' value from 0 to the largest is carried by "
            "no household; SPI income band donors seat clones 0 to K - 1.",
        ),
    ],
)
def test_a_malformed_clone_index_is_refused_without_a_count(
    index, message: str
) -> None:
    frame = _clone_first_support_frame(60, 2)
    household = frame.table("household").copy()
    household[_CLONE] = index(household[_CLONE])
    with pytest.raises(ValueError) as caught:
        _stack_small(_with_tables(frame, household=household))
    assert str(caught.value) == message


def test_the_clone_seating_is_keyed_on_identity_in_any_row_order() -> None:
    first = _stack_small(_clone_first_support_frame(200, 3))
    second = _stack_small(_clone_first_support_frame(200, 3, reverse=True))
    assert _seating(first) == _seating(second)
    assert _carriers(first) == _carriers(second)


def test_the_receipt_counts_source_households_at_the_disclosure_floor() -> None:
    # Given 200 households, 7 of them without an adult claimant,
    person, benunit, household = _raw_support_tables(200)
    person.loc[person["person_household_id"].le(7), "age"] = 12
    result = _stack_small(
        _spi_support_frame(person, benunit, household, expected_spi_households=10)
    )
    evidence = result.evidence()

    # then every count of survey households is rounded: the eligible ones in
    # all and per band, the donors per funding stratum and the carriers per
    # region, with no exact count beside them.
    assert evidence["eligible_households"] == rounded_count(193) == 190
    propensity = _propensity()
    claimants = person.loc[person["is_uc_claimant"] & person["age"].ge(16)]
    region_of = household.set_index("household_id")["region"]
    cells = {
        int(row.person_household_id): (
            region_of[row.person_household_id],
            str(row.gender),
            int(age_band_lower_bound(np.asarray([float(row.age)]))[0]),
        )
        for row in claimants.itertuples()
    }
    seated = _seating(result)[0]
    taken: set[int] = set()
    rows = {row["lower_bound"]: row for row in evidence["bands"]}
    for band in sorted(BANDS, reverse=True):
        positive = propensity.loc[
            (propensity["band"] == band) & (propensity["propensity"] > 0.0)
        ]
        open_cells = set(
            zip(
                positive["region"],
                positive["gender"],
                positive["age_band"],
                strict=True,
            )
        )
        eligible = {
            household_id
            for household_id, cell in cells.items()
            if cell in open_cells and household_id not in taken
        }
        assert rows[band]["eligible_households"] == rounded_count(len(eligible))
        taken |= {source for source, seat in seated.items() if seat == band}
    out = result.frame.table("household")
    donors = out.loc[out[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    by_region = donors["region"].value_counts()
    assert {row["stratum"]: row["donor_households"] for row in evidence["funding"]} == {
        str(region): rounded_count(int(by_region.get(region, 0)))
        for region in ("LONDON", "SOUTH_EAST")
    }
    assert evidence["carrier_region_counts"] == {
        str(region): rounded_count(int(count)) for region, count in by_region.items()
    }
    assert evidence["seating_scale"] == "full"
    assert all(isinstance(row["carrier_mean_age"], float) for row in evidence["bands"])
    assert set(evidence) == {
        "stage",
        "minimum_donors_per_band",
        "maximum_donor_weight",
        "maximum_donor_household_share",
        "maximum_reallocation_share",
        "clone_count",
        "funding_strata",
        "seating_scale",
        "mass_scale",
        "sample_fraction",
        "eligible_households",
        "expected_donor_count",
        "donor_count",
        "reallocated_mass",
        "funding_within_share",
        "funding_matches_bands",
        "mass",
        "draw",
        "taxpayer_count_resource",
        "taxpayer_count_period",
        "propensity",
        "propensity_cells",
        "bands",
        "funding",
        "carrier_region_counts",
    }
    assert {key for row in evidence["bands"] for key in row} == {
        "lower_bound",
        "expected_donor_households",
        "donor_households",
        "carriers",
        "eligible_households",
        "planned_donor_weight",
        "donor_weight",
        "weighted_taxpayers",
        "published_taxpayers",
        "carrier_mean_age",
        "carrier_regions",
    }
    # At every seating a funding row keeps its stratum and rounded donors and
    # withholds its masses and factor (microcosm#931).
    assert {key for row in evidence["funding"] for key in row} == {
        "stratum",
        "donor_households",
        "incumbent_mass",
        "donor_mass",
        "factor",
    }
    assert all(
        row[key] == UK_WITHHELD
        for row in evidence["funding"]
        for key in ("incumbent_mass", "donor_mass", "factor")
    )

    # A band whose carriers come from fewer than three households withholds
    # their mean age: the published plan on 60 households seats one in three
    # of the bands. The result's exact rows keep that rule.
    capped_result = stack_spi_income_band_donors(
        _support_frame(), propensity=propensity, **_PUBLISHED
    )
    for row in capped_result.band_rows:
        if row["carriers"] < 3:
            assert row["carrier_mean_age"] == UK_WITHHELD
        else:
            assert isinstance(row["carrier_mean_age"], float)
    assert sum(row["carriers"] < 3 for row in capped_result.band_rows) == 3

    # The capped receipt gives each band's donors and carriers in rounded
    # source households, withholds its weighted taxpayers and mean age, and
    # leaves its carrier regions out; its donor count is rounded too, and its
    # reallocated mass withheld (microcosm#931).
    capped = capped_result.evidence()
    assert capped["seating_scale"] == "capped"
    assert capped["donor_count"] == rounded_count(capped_result.donor_source_households)
    assert capped["reallocated_mass"] == UK_WITHHELD
    assert capped["funding_within_share"] is capped["funding_matches_bands"] is True
    for row, exact in zip(capped["bands"], capped_result.band_rows, strict=True):
        assert set(row) == {
            "lower_bound",
            "expected_donor_households",
            "donor_households",
            "carriers",
            "eligible_households",
            "planned_donor_weight",
            "donor_weight",
            "weighted_taxpayers",
            "published_taxpayers",
            "carrier_mean_age",
        }
        assert row["donor_households"] == rounded_count(
            exact["donor_source_households"]
        )
        assert row["carriers"] == rounded_count(exact["carrier_source_households"])
        assert row["weighted_taxpayers"] == row["carrier_mean_age"] == UK_WITHHELD
        for key in (
            "lower_bound",
            "expected_donor_households",
            "eligible_households",
            "planned_donor_weight",
            "donor_weight",
            "published_taxpayers",
        ):
            assert row[key] == exact[key]


def _one_top_band_candidate(person, household) -> pd.DataFrame:
    """Household 1's claimant aged 70, alone in the one cell the top band opens.

    Changes ``person`` in place and returns the propensity table.
    """

    elder = person["person_household_id"].eq(1) & person["is_uc_claimant"]
    person.loc[elder, "age"] = 70
    cell = (
        household.set_index("household_id").loc[1, "region"],
        str(person.loc[elder, "gender"].iloc[0]),
        int(age_band_lower_bound(np.asarray([70.0]))[0]),
    )
    propensity = _propensity()
    return pd.concat(
        [
            propensity.loc[propensity["band"] != 2_000_000],
            pd.DataFrame(
                [
                    {
                        "region": cell[0],
                        "gender": cell[1],
                        "age_band": cell[2],
                        "band": 2_000_000,
                        "band_weight": 1.0,
                        "cell_weight": 1.0,
                        "propensity": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


def test_a_band_short_of_candidates_on_one_clone_is_refused_without_a_count() -> None:
    # Given one claimant alone the tape lets carry the top band, on a spine
    # without the clone index, and three seats for the band,
    person, benunit, household = _raw_support_tables(200)
    propensity = _one_top_band_candidate(person, household)
    frame = _spi_support_frame(person, benunit, household, expected_spi_households=10)

    # then the stage refuses with one clone's message, which names the band
    # and gives no count.
    with pytest.raises(ValueError) as caught:
        _stack_small(frame, propensity=propensity)
    assert str(caught.value) == (
        "SPI income band donors found fewer candidate households with positive "
        "weight and propensity for the band from 2000000 than the band seats."
    )


def test_a_mean_age_resting_on_fewer_than_three_source_households_is_withheld() -> None:
    # Given one claimant alone the tape lets carry the top band, copied into
    # each of three clones,
    person, benunit, household = _raw_support_tables(200)
    propensity = _one_top_band_candidate(person, household)
    frame = _clone_first_support_frame(200, 3, tables=(person, benunit, household))
    result = _stack_small(frame, propensity=propensity, minimum_donors=1)

    # then each clone seats that household, and the band's three carriers
    # rest on one source household: their mean age is withheld.
    seated = _seating(result)
    assert sorted(seated) == [0, 1, 2]
    assert all(seats[1] == 2_000_000 for seats in seated.values())
    top = {row["lower_bound"]: row for row in result.band_rows}[2_000_000]
    assert top["carriers"] == 3
    assert top["carrier_mean_age"] == UK_WITHHELD
    first = {row["lower_bound"]: row for row in result.band_rows}[200_000]
    assert first["carriers"] == 12 and isinstance(first["carrier_mean_age"], float)


def test_a_band_seating_fewer_than_three_per_clone_withholds_its_mean_age() -> None:
    # Given the published plan on 60 households, which seats one household in
    # three of the bands, in each of three and of four clones,
    propensity = _propensity()
    for n_clones in (3, 4):
        result = stack_spi_income_band_donors(
            _clone_first_support_frame(60, n_clones),
            propensity=propensity,
            **_PUBLISHED,
        )
        out = result.frame.table("household").set_index("household_id")
        person = result.frame.table("person")
        carriers = out.loc[
            person.loc[person[PERSON_IS_SPI_INCOME_BAND_CARRIER], "person_household_id"]
        ]
        sources = carriers.groupby(SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN)[
            "source_household_id"
        ].nunique()

        # then a band seating fewer than three households in each clone
        # withholds its carriers' mean age, even where they come from three
        # source households or more; the band seating six keeps its mean.
        one_seat = [
            row for row in result.band_rows if row["donor_households"] < 3 * n_clones
        ]
        assert [row["donor_households"] for row in one_seat] == [n_clones] * 3
        assert all(row["carrier_mean_age"] == UK_WITHHELD for row in one_seat)
        assert any(sources[row["lower_bound"]] >= 3 for row in one_seat)
        (seated,) = [row for row in result.band_rows if row not in one_seat]
        assert seated["donor_households"] == 6 * n_clones
        assert isinstance(seated["carrier_mean_age"], float)


def test_the_receipt_counts_source_households_at_any_clone_count(monkeypatch) -> None:
    # Given the published plan on 60 households, as one clone and as three,
    propensity = _propensity()
    one = stack_spi_income_band_donors(
        _support_frame(), propensity=propensity, **_PUBLISHED
    )
    many = stack_spi_income_band_donors(
        _clone_first_support_frame(60, 3), propensity=propensity, **_PUBLISHED
    )
    evidence = many.evidence()

    # then the carriers per region are counted in source households over
    # every clone: SOUTH_EAST's carrier rows rest on fewer than three.
    out = many.frame.table("household").set_index("household_id")
    person = many.frame.table("person")
    carriers = out.loc[
        person.loc[person[PERSON_IS_SPI_INCOME_BAND_CARRIER], "person_household_id"]
    ]
    by_region = carriers.groupby("region")["source_household_id"].nunique()
    assert evidence["carrier_region_counts"] == {
        str(region): rounded_count(int(count)) for region, count in by_region.items()
    }
    assert int(carriers["region"].eq("SOUTH_EAST").sum()) >= 3
    assert evidence["carrier_region_counts"]["SOUTH_EAST"] == UK_WITHHELD
    # Each band's eligible households are clone 0's, which one clone counts
    # alike, so the per-band counts are the same at every clone count.
    assert [row["eligible_households"] for row in evidence["bands"]] == [
        row["eligible_households"] for row in one.evidence()["bands"]
    ]

    # Before rounding, each band's count is the source households clone 0's
    # own walk leaves eligible, recounted here from the cells the tape opens;
    # pooled over the clones, the lower bands would count more.
    monkeypatch.setattr(spi_band_donors, "rounded_count", int)
    exact = stack_spi_income_band_donors(
        _clone_first_support_frame(60, 3), propensity=propensity, **_PUBLISHED
    )
    raw_person, _, raw_household = _raw_support_tables(60)
    claimants = raw_person.loc[raw_person["is_uc_claimant"] & raw_person["age"].ge(16)]
    region_of = raw_household.set_index("household_id")["region"]
    cells = {
        int(row.person_household_id): (
            region_of[row.person_household_id],
            str(row.gender),
            int(age_band_lower_bound(np.asarray([float(row.age)]))[0]),
        )
        for row in claimants.itertuples()
    }
    seated = _seating(exact)
    taken: dict[int, set[int]] = {clone: set() for clone in seated}
    rows = {row["lower_bound"]: row for row in exact.band_rows}
    pooled_more = False
    for band in sorted(BANDS, reverse=True):
        positive = propensity.loc[
            (propensity["band"] == band) & (propensity["propensity"] > 0.0)
        ]
        open_cells = set(
            zip(
                positive["region"],
                positive["gender"],
                positive["age_band"],
                strict=True,
            )
        )
        eligible = {
            clone: {
                source
                for source, cell in cells.items()
                if cell in open_cells and source not in taken[clone]
            }
            for clone in seated
        }
        assert rows[band]["eligible_households"] == len(eligible[0])
        pooled_more |= len(set().union(*eligible.values())) > len(eligible[0])
        for clone, seats in seated.items():
            taken[clone] |= {source for source, seat in seats.items() if seat == band}
    assert pooled_more
    assert exact.eligible_households == 60


def test_the_stage_gate_reads_a_receipt_by_its_clone_count() -> None:
    one = _stack_small(_support_frame(200)).evidence()
    evidence = _stack_small(_clone_first_support_frame(200, 3)).evidence()
    for receipt in (one, evidence):
        gate = _small_gate(receipt)
        assert gate.passed, gate.failures
    assert _small_gate(evidence).details["clone_count"] == 3

    # Dropped, the clone count is refused rather than read as one clone.
    with pytest.raises(ValueError, match="clone_count must be a positive integer"):
        _small_gate(
            {key: value for key, value in evidence.items() if key != "clone_count"}
        )
    # Misread, three clones' donors split unevenly or outweigh the maximum.
    failures = " ".join(_small_gate({**evidence, "clone_count": 2}).failures)
    assert "do not split evenly across the geographic clones" in failures
    failures = " ".join(_small_gate({**evidence, "clone_count": 4}).failures)
    assert "exceeds the maximum divided by the geographic clone count" in failures
    # The gate spells the receipt's two seating forms itself.
    from microcosm.build.uk_runtime import stage_health

    assert stage_health._BAND_DONOR_SEATINGS == (
        SPI_INCOME_BAND_FULL_SEATING,
        SPI_INCOME_BAND_CAPPED_SEATING,
    )


def test_the_gate_spells_the_stages_columns_and_tolerances_as_the_stage_does() -> None:
    # The stage-health gate imports no stage module, so it spells the band
    # donor stage's columns, its copy-id field and the fields minted after it
    # itself; and the stage's funding booleans compare at the gate's own
    # tolerances (microcosm#931).
    from microcosm.build.uk_runtime import spi_support, stage_health

    assert (
        stage_health._BAND_DONOR_FLAG_COLUMN,
        stage_health._BAND_DONOR_BAND_COLUMN,
        stage_health._BAND_CARRIER_FLAG_COLUMN,
        stage_health._BAND_DONOR_CLONE_INDEX_COLUMN,
        stage_health._BAND_DONOR_COPY_ID_FIELD,
    ) == (
        HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
        SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
        PERSON_IS_SPI_INCOME_BAND_CARRIER,
        spi_support.SPI_GEOGRAPHIC_CLONE_INDEX_COLUMN,
        spi_support.SPI_SUPPORT_COPY_ID_FIELD,
    )
    assert stage_health._COPY_ID_FIELDS_AFTER_BAND_DONORS == _LATER_FIELDS
    assert (
        spi_band_donors.SPI_INCOME_BAND_FUNDING_RELATIVE_TOLERANCE
        == stage_health._FLOAT_RELATIVE_TOLERANCE
    )
    (entry,) = [
        entry
        for entry in load_country_spec("uk").gates.gates
        if entry.gate == "stage_health"
        and entry.parameters["check"] == "spi_income_band_donor_support"
    ]
    assert spi_band_donors.SPI_INCOME_BAND_FUNDING_MASS_TOLERANCE == max(
        entry.parameters["maximum_relative_mass_deviation"],
        stage_health._FLOAT_RELATIVE_TOLERANCE,
    )
    share = spi_band_donors.SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE
    assert 1.0 - share >= entry.parameters["minimum_funding_factor"]


def test_e8_receipt_replays_the_seating_of_every_clone() -> None:
    tool = _identity_tool()
    propensity = _propensity()
    arguments = {
        "propensity": propensity,
        "band_taxpayers": TAXPAYERS,
        "permutation_seed": 123,
    }
    stage = stack_spi_income_band_donors(
        _clone_first_support_frame(200, 3), propensity=propensity, **_PUBLISHED
    )
    stacked = stage.frame
    evidence = stage.evidence()
    problems: dict[str, object] = {}
    receipt = tool._e8_band_donors(stacked, problems, **arguments)
    assert problems == {}
    assert receipt["seating_recomputed"] is True
    assert receipt["seating_scale"] == evidence["seating_scale"] == "capped"
    # Every clone's donors, counted once per survey household and rounded, and
    # every clone's regions in the stage's funding rows.
    assert receipt["donor_households"] == evidence["donor_count"]
    assert receipt["funding"] == evidence["funding"]
    assert {row["stratum"] for row in receipt["funding"]} == {
        f"{region}|{clone}" for region in ("LONDON", "SOUTH_EAST") for clone in range(3)
    }

    # A donor reweighted in the last clone breaks its band's one weight.
    household = stacked.table("household")
    weights = np.asarray(stacked.weights_for("household").values, dtype=float).copy()
    last_donor = int(np.flatnonzero(household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR])[-1])
    assert household[_CLONE].iloc[last_donor] == 2
    weights[last_donor] *= 1.5
    problems = {}
    tool._e8_band_donors(_reweighted(stacked, weights), problems, **arguments)
    assert problems["band_donor_layer"] == {"bands_with_unequal_donor_weights": 1}


# The E8 receipt on the artifact as built (microcosm#931). Every layer stacked
# after stage 17 copies rows the stage wrote, donors among them, and mints its
# copies in a copy-id field above the one the donors take. A rerun on a frame
# that kept those rows seated them as households the stage saw, and the donors
# it appended sorted below their ids, which the frame refuses. The receipt
# reruns the stage on the frame the stage wrote, every later row folded onto
# the row it came from.

#: The copy-id fields minted after stage 17, in the order the spine mints them.
_LATER_FIELDS = (
    "high_value_support",
    "cgt_support_split",
    "cgt_incidence_clone",
    "cgt_residential_split",
)


def _stage_17_ids(household: pd.DataFrame) -> np.ndarray:
    """Each household's id with every later layer's ordinal cleared."""

    ids = household["household_id"].to_numpy(dtype=np.int64)
    for field in _LATER_FIELDS:
        ids = copy_id_table.parent(ids, field)
    return ids


def _minted(frame, *, field: str, index_column: str, ordinals):
    """``frame``'s tables with copies minted in ``field``, and their lineage.

    ``copy_households`` mints every id of a copy through the copy-id table,
    as the stages do, and records the copy's ordinal in ``index_column``;
    ``ordinals`` maps a household's position to its copies' ordinals. Returns
    the tables and, per household, its parent's weight, its ordinal and the
    size of its family (one for a household left whole).
    """

    copied = copy_households(
        IdTables(
            person=frame.table("person"),
            benunit=frame.table("benunit"),
            household=frame.table("household"),
        ),
        field=field,
        path_column=index_column,
        ordinals=ordinals,
        rule="table",
    )
    ids = copied.household["household_id"].to_numpy(dtype=np.int64)
    parents = pd.Series(copy_id_table.parent(ids, field))
    return (
        copied,
        _weights_by_id(frame).loc[parents.to_numpy()].to_numpy(),
        copy_id_table.ordinal(ids, field),
        parents.map(parents.value_counts()).to_numpy(),
    )


def _framed(previous, tables, weights):
    """``tables`` as a frame at ``weights``, with ``previous``'s mass records."""

    return uk_national_frame(
        person=tables.person,
        benunit=tables.benunit,
        household=tables.household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
        mass_log=previous.mass_log,
    )


def _through_the_later_layers(stacked) -> SimpleNamespace:
    """Stage 17's output carried through every layer stacked after it.

    The high-value home support (stage 26) and the regional home values (27)
    run as built, every household an owner valued on its true source, at a
    stratum that divides donors, SPI copies and base households alike. The
    CGT layers are replayed by hand with ``copy_households``: the support
    split (42) divides the households of one source in four into two or
    three rows at equal shares, the incidence twin (43) copies every
    household, the anchor (45) moves an eighth of the mass of two pairs in
    three, one way or the other, and the residential split (46) divides the
    twins of one source in five into two or four arms at product weights.
    Salary sacrifice (48) then converts the best-paid record's pay in place.
    Every layer keeps each household's mass. All values are invented.
    """

    household = stacked.table("household").copy()
    source = household["source_household_id"].to_numpy()
    household["tenure_type"] = pd.array(
        ["OWNED_OUTRIGHT"] * len(household), dtype="string"
    )
    household["main_residence_value"] = 100_000.0 * (1 + source % 10)
    household["property_wealth"] = household["main_residence_value"] + 10_000.0
    high_value = split_high_value_support_households(
        _with_tables(stacked, household=household), stratum_lower=0.9, copies=3
    )
    valued = assign_regional_home_values(high_value.frame).frame

    household = valued.table("household").copy()
    source = household["source_household_id"].to_numpy()
    rows = np.where(source % 4 == 1, 2 + (source % 8 == 1), 1)
    household[HOUSEHOLD_IS_CGT_SUPPORT_COPY] = False
    household[CGT_SUPPORT_COPY_INDEX_COLUMN] = 0
    household[CGT_SUPPORT_COPIES_COLUMN] = rows
    tables, weights, ordinal, size = _minted(
        _with_tables(valued, household=household),
        field="cgt_support_split",
        index_column=CGT_SUPPORT_COPY_INDEX_COLUMN,
        ordinals={
            int(position): tuple(range(1, int(rows[position])))
            for position in np.flatnonzero(rows > 1)
        },
    )
    tables.household[HOUSEHOLD_IS_CGT_SUPPORT_COPY] = ordinal != 0
    split = _framed(valued, tables, weights / size)

    household = split.table("household").copy()
    household[HOUSEHOLD_IS_CGT_CLONE] = 0
    tables, weights, ordinal, _ = _minted(
        _with_tables(split, household=household),
        field="cgt_incidence_clone",
        index_column=HOUSEHOLD_IS_CGT_CLONE,
        ordinals={position: (1,) for position in range(len(household))},
    )
    tables.household[HOUSEHOLD_IS_CGT_CLONE] = ordinal != 0
    kept = 0.5 + 0.125 * (tables.household["source_household_id"].to_numpy() % 3 - 1)
    twinned = _framed(split, tables, weights * np.where(ordinal == 0, kept, 1 - kept))

    household = twinned.table("household").copy()
    source = household["source_household_id"].to_numpy()
    gainers = np.where(
        household[HOUSEHOLD_IS_CGT_CLONE].to_numpy() & (source % 5 == 0),
        1 + (source % 10 == 0),
        0,
    )
    household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE] = False
    household[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN] = 0
    tables, weights, ordinal, size = _minted(
        _with_tables(twinned, household=household),
        field="cgt_residential_split",
        index_column=CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
        ordinals={
            int(position): tuple(range(1, 2 ** int(gainers[position])))
            for position in np.flatnonzero(gainers)
        },
    )
    tables.household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE] = ordinal != 0
    # One liable gainer at 1/4, or two at 1/4 and 1/2: arms at product weights.
    shares = {1: (1.0,), 2: (0.75, 0.25), 4: (0.375, 0.125, 0.375, 0.125)}
    armed = _framed(
        twinned,
        tables,
        weights
        * np.asarray([shares[n][j] for n, j in zip(size, ordinal, strict=True)]),
    )

    person = armed.table("person").reset_index(drop=True)
    pay = person["employment_income"].to_numpy(dtype=float)
    converted = int(np.argmax(pay))
    person["employee_pension_contributions"] = 0.0
    person[SALSAC_OUTPUT] = 0.0
    person[SALSAC_PRE_CONVERSION_PAY_COLUMN] = 0.0
    person.loc[converted, SALSAC_OUTPUT] = pay[converted] / 4
    person.loc[converted, "employment_income"] = pay[converted] * 3 / 4
    person.loc[converted, SALSAC_PRE_CONVERSION_PAY_COLUMN] = pay[converted]
    return SimpleNamespace(
        artifact=_with_tables(armed, person=person), high_value=high_value
    )


def _stage_17_and_artifact(n_clones: int, propensity: pd.DataFrame):
    support = (
        _support_frame(200)
        if n_clones == 1
        else _clone_first_support_frame(200, n_clones)
    )
    stage_17 = stack_spi_income_band_donors(
        support, propensity=propensity, **_PUBLISHED
    )
    return stage_17, _through_the_later_layers(stage_17.frame)


@pytest.mark.parametrize("n_clones", [1, 3])
def test_e8_receipt_reruns_stage_17_on_the_frame_it_wrote(n_clones: int) -> None:
    tool = _identity_tool()
    propensity = _propensity()
    stage_17, built = _stage_17_and_artifact(n_clones, propensity)

    # Given an artifact whose high-value copies divide donors, SPI copies and
    # base households alike, and whose CGT layers copy donors and high-value
    # copies too,
    divided = built.high_value.frame.table("household")
    copies = divided[HIGH_VALUE_SUPPORT_INDEX_COLUMN].to_numpy() != 0
    donor = divided[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].to_numpy()
    spi = divided["household_support_channel"].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)
    assert (copies & donor).any()
    assert (copies & spi.to_numpy() & ~donor).any()
    assert (copies & ~spi.to_numpy()).any()
    household = built.artifact.table("household")
    support_copies = household[CGT_SUPPORT_COPY_INDEX_COLUMN].to_numpy() != 0
    for copied in (HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR, HIGH_VALUE_SUPPORT_COPY_COLUMN):
        assert (support_copies & household[copied].to_numpy()).any()
    assert household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE].any()

    # its scope to stage 17 is the frame the stage wrote, row for row, at the
    # weights it left,
    scoped = tool._frame_as_stage_saw(built.artifact, "spi_income_band_donors")
    for entity in ("person", "benunit", "household"):
        assert sorted(scoped.table(entity)[f"{entity}_id"]) == sorted(
            stage_17.frame.table(entity)[f"{entity}_id"]
        )
    np.testing.assert_allclose(
        _weights_by_id(scoped).loc[_weights_by_id(stage_17.frame).index].to_numpy(),
        _weights_by_id(stage_17.frame).to_numpy(),
        rtol=1e-12,
    )

    # and the check scales each funding stratum back by the stage's own
    # factor,
    factors = {row["stratum"]: row["factor"] for row in stage_17.funding_rows}
    _, reconstructed = tool._band_donor_funding_factors(
        scoped.table("household"), scoped.weights_for("household").values
    )
    assert reconstructed.to_dict() == pytest.approx(factors, rel=1e-12)

    # and the stage rerun there reproduces the stored seats, carriers and band
    # weights, with the artifact's people in either order and the pre-donor
    # people permuted, reading each donor once. The receipt gives the layer in
    # the stage's capped forms, the exact weights and factors held, never
    # written.
    evidence = stage_17.evidence()
    for artifact in (built.artifact, tool._reverse_rows(built.artifact)):
        problems: dict[str, object] = {}
        receipt = tool._e8_band_donors(
            artifact,
            problems,
            propensity=propensity,
            band_taxpayers=TAXPAYERS,
            permutation_seed=123,
        )
        assert problems == {}
        assert receipt["seating_recomputed"] is True
        assert receipt["donor_households"] == rounded_count(
            stage_17.donor_source_households
        )
        assert receipt["funding"] == evidence["funding"]
        assert receipt["funding_within_share"] is True
        assert receipt["bands"] == [
            {
                "lower_bound": row["lower_bound"],
                "donor_households": rounded_count(row["donor_source_households"]),
            }
            for row in stage_17.band_rows
        ]


def test_e8_receipt_reads_each_donor_through_its_later_copies() -> None:
    tool = _identity_tool()
    propensity = _propensity()
    _, built = _stage_17_and_artifact(1, propensity)
    arguments = {
        "propensity": propensity,
        "band_taxpayers": TAXPAYERS,
        "permutation_seed": 123,
    }
    # The donor the later layers divided into the most rows: its stage-17
    # row and every row made from it.
    household = built.artifact.table("household")
    stems = _stage_17_ids(household)
    rows = pd.Series(stems).value_counts()
    donors = household.loc[
        household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR], "household_id"
    ]
    donor = min(rows.index[rows.index.isin(donors)], key=lambda s: (-rows[s], s))
    family = np.flatnonzero(stems == donor)
    assert len(family) > 2
    weights = _weights_by_id(built.artifact).to_numpy().copy()

    # Mass moved between the rows of a donor's family is no defect: the fold
    # gives back the weight stage 17 left.
    moved = weights.copy()
    moved[family[0]] += moved[family[-1]] / 2
    moved[family[-1]] /= 2
    problems: dict[str, object] = {}
    tool._e8_band_donors(_reweighted(built.artifact, moved), problems, **arguments)
    assert problems == {}

    # Reweighting the whole family is a defect: its band's donors no longer
    # share one weight.
    scaled = weights.copy()
    scaled[family] *= 1.5
    problems = {}
    tool._e8_band_donors(_reweighted(built.artifact, scaled), problems, **arguments)
    assert problems["band_donor_layer"] == {"bands_with_unequal_donor_weights": 1}


def test_e8_receipt_refuses_a_later_row_its_scope_leaves_in_place() -> None:
    # Given an artifact whose high-value copies have lost the lineage the
    # scope reads them by, those copies stay in scope; their ids still hold
    # an ordinal in a field minted after stage 17, so the receipt refuses by
    # count rather than seat them.
    tool = _identity_tool()
    propensity = _propensity()
    _, built = _stage_17_and_artifact(1, propensity)
    household = built.artifact.table("household").drop(
        columns=[
            HIGH_VALUE_SUPPORT_COPY_COLUMN,
            HIGH_VALUE_SUPPORT_INDEX_COLUMN,
            HIGH_VALUE_SUPPORT_COPIES_COLUMN,
        ]
    )
    unread = _with_tables(built.artifact, household=household)
    with pytest.raises(ValueError) as caught:
        tool._e8_band_donors(
            unread,
            {},
            propensity=propensity,
            band_taxpayers=TAXPAYERS,
            permutation_seed=123,
        )
    assert str(caught.value) == (
        f"{built.high_value.copies_created} household row(s) left in the scope "
        "to the SPI income band donor stage hold an ordinal in a copy-id field "
        "minted after it (microcosm#931); the band-donor rerun would seat them "
        "as households the stage saw."
    )


def test_the_e8_entry_hands_the_band_donor_check_the_artifact(monkeypatch) -> None:
    # The band-donor check scopes the artifact itself, which reverses the
    # salary-sacrifice conversion; handed the frame the entry has already
    # reversed, it would reverse the conversion twice and refuse. The other
    # E8 checks need the CGT resources and stand aside here.
    from microcosm.build.uk_runtime import cgt_imputation, student_loans

    tool = _identity_tool()
    propensity = _propensity()
    stage_17, built = _stage_17_and_artifact(1, propensity)
    person = built.artifact.table("person").copy()
    person["student_loan_plan"] = "NONE"
    artifact = _with_tables(built.artifact, person=person)
    for check in ("_e8_residential_split", "_e8_clone_pairs", "_e8_support_split"):
        monkeypatch.setattr(tool, check, lambda *args, **kwargs: {})
    monkeypatch.setattr(
        tool, "_band_donor_seating_inputs", lambda *args: (propensity, TAXPAYERS)
    )
    monkeypatch.setattr(cgt_imputation, "uk_cgt_policy_parameters", lambda _: None)
    monkeypatch.setattr(
        student_loans,
        "assign_student_loan_plans",
        lambda frame, **kwargs: SimpleNamespace(frame=frame),
    )

    receipt = tool.e8_identity_receipt(artifact, permutation_seed=123)
    assert receipt["pre_salary_sacrifice_restored"] is True
    assert receipt["stored_column_mismatches"] == {}
    assert receipt["band_donors"]["seating_recomputed"] is True
    assert receipt["band_donors"]["donor_households"] == rounded_count(
        stage_17.donor_source_households
    )


# E8's funding rule on frames the stage funds at the floor (microcosm#931).
# Every frame below is synthetic.


def _floor_funded_frame(n_households: int, salt: str):
    """``_support_frame``'s households, LONDON's light but for one heavy
    household and the other region's carrying most of the mass, so the mass
    scale binds in LONDON, which keeps exactly one less the reallocation
    share; the stage then makes the whole total exact on one incumbent row."""

    from microcosm.build.stochastic_assignment import stable_identity_uniforms

    frame = _support_frame(n_households)
    household = frame.table("household")
    uniforms = stable_identity_uniforms(
        household["household_id"].to_numpy(), seed=11, salt=salt
    )
    london = household["region"].astype(str).eq("LONDON").to_numpy()
    weights = np.where(london, 1.0 + uniforms, 9_000.0 + 500.0 * uniforms)
    heavy = np.flatnonzero(london)[0]
    weights[heavy] = 20_000.0 + 100.0 * uniforms[heavy]
    return uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )


def _floor_funded_stage():
    stage = stack_spi_income_band_donors(
        _floor_funded_frame(900, "j"),
        propensity=_propensity(),
        band_taxpayers=TAXPAYERS,
        taxpayer_period=2024,
    )
    floor = 1.0 - spi_band_donors.SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE
    assert stage.mass_scale < 1.0
    factors = {row["stratum"]: row["factor"] for row in stage.funding_rows}
    assert factors["LONDON"] == floor
    assert stage.funding_within_share is True
    return stage, floor


def _e8_funding(tool, frame) -> tuple[dict[str, object], dict[str, object]]:
    problems: dict[str, object] = {}
    receipt = tool._e8_band_donors(
        frame, problems, propensity=None, band_taxpayers=None, permutation_seed=123
    )
    return receipt, problems


def test_e8_holds_the_funding_of_a_stratum_the_stage_funds_at_the_floor() -> None:
    """The factor E8 rebuilds from the stored weights carries the stage's
    exact-total correction wherever it landed, so E8 holds the funding by
    the stage's rule at its own weight tolerance, not at the stage gate's
    few ulps: the stage's output is green."""

    tool = _identity_tool()
    stage, _ = _floor_funded_stage()
    receipt, problems = _e8_funding(tool, stage.frame)
    assert problems == {}
    assert receipt["funding_within_share"] is True


def test_e8_holds_a_floor_factor_moved_by_rounding_and_refuses_one_underfunded():
    """A floor stratum's incumbent moved down by a trillionth of the
    stratum's mass, which an exact-total correction can do to a small
    stratum, takes its rebuilt factor beyond the stage gate's ulps but not
    E8's tolerance, and is held; moved by a millionth, it is a stratum kept
    below one less the share, and E8 says so. The other region's incumbent
    takes the mass up, so the total and its mass record stand."""

    tool = _identity_tool()
    stage, floor = _floor_funded_stage()
    household = stage.frame.table("household")
    weights = np.asarray(stage.frame.weights_for("household").values, dtype=float)
    london = household["region"].astype(str).eq("LONDON").to_numpy()
    incumbent = ~household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].to_numpy(dtype=bool)
    in_london = np.flatnonzero(london & incumbent)
    heaviest = in_london[np.argmax(weights[in_london])]
    receiver = np.flatnonzero(~london & incumbent)[0]
    stratum_mass = float(weights[london].sum())
    for relative, held in ((1e-12, True), (1e-6, False)):
        moved = weights.copy()
        moved[heaviest] -= relative * stratum_mass
        moved[receiver] += relative * stratum_mass
        frame = _reweighted(stage.frame, moved)
        _, rebuilt = tool._band_donor_funding_factors(
            frame.table("household"), frame.weights_for("household").values
        )
        assert float(rebuilt["LONDON"]) < floor * (
            1.0 - spi_band_donors.SPI_INCOME_BAND_FUNDING_RELATIVE_TOLERANCE
        )
        receipt, problems = _e8_funding(tool, frame)
        assert receipt["funding_within_share"] is held
        assert ("band_donor_funding_share" in problems) is not held


def test_e8_refuses_a_stratum_funded_past_its_share() -> None:
    tool = _identity_tool()
    propensity = _propensity()
    stacked = stack_spi_income_band_donors(
        _support_frame(200),
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        taxpayer_period=2024,
    ).frame
    household = stacked.table("household")
    weights = np.asarray(stacked.weights_for("household").values, dtype=float).copy()
    incumbent = ~household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].to_numpy(dtype=bool)
    london = household["region"].astype(str).eq("LONDON").to_numpy()
    weights[incumbent & london] *= 0.05
    problems: dict[str, object] = {}
    receipt = tool._e8_band_donors(
        _reweighted(stacked, weights),
        problems,
        propensity=propensity,
        band_taxpayers=TAXPAYERS,
        permutation_seed=123,
    )
    assert receipt["funding_within_share"] is False
    assert problems["band_donor_funding_share"] == (
        "a funding stratum's incumbents kept less than one less the declared "
        "reallocation share of its mass"
    )


# The stage gate on a capped seating (microcosm#931). A capped receipt gives
# each band's donors and carriers only in rounded source households, so the
# gate recomputes the seats from the transferred phase's frame: the band
# donor rows that are originals in every copy-id field minted after the band
# donors. Every frame below is synthetic.

#: The stage gate's declaration at the published plan's constants.
_PUBLISHED_GATE_PARAMETERS = {
    **_SMALL_GATE_PARAMETERS,
    "minimum_donors_per_band": SPI_INCOME_BAND_MINIMUM_DONORS,
    "maximum_donor_weight": SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT,
}


def _capped_gate(evidence, frame):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="spi_income_band_donors",
        check="spi_income_band_donor_support",
        parameters=_PUBLISHED_GATE_PARAMETERS,
        frame=frame,
    )


def _no_data_digit(message: str) -> bool:
    """Whether ``message`` holds no digit beyond the declared band bounds."""

    bounds = "|".join(str(band) for band in sorted(BANDS, reverse=True))
    return re.search(r"\d", re.sub(rf"\b(?:{bounds})\b", "", message)) is None


def _capped_pair_frame(*, last_adult_claims: bool):
    """81 households, whose last adult is a claimant or not.

    The published plan's capped seating cuts its seats from the eligible
    households, so the frame whose last adult claims seats one more donor
    in its first band: 81 and 80 eligible households sit either side of a
    whole seat.
    """

    person, benunit, household = _raw_support_tables(81)
    if not last_adult_claims:
        last = person["person_household_id"].eq(81) & person["is_uc_claimant"]
        person.loc[last, "is_uc_claimant"] = False
    return _spi_support_frame(person, benunit, household, expected_spi_households=10)


def test_two_capped_frames_whose_seats_differ_by_one_give_one_receipt_and_verdict() -> (
    None
):
    # Given two frames of the same households and weights, one with one more
    # eligible claimant, whose capped seatings differ by one donor at a mass
    # scale of one,
    propensity = _propensity()
    fewer, more = (
        stack_spi_income_band_donors(
            _capped_pair_frame(last_adult_claims=claims),
            propensity=propensity,
            sample_fraction=0.01,
            **_PUBLISHED,
        )
        for claims in (False, True)
    )
    assert more.donor_count == fewer.donor_count + 1
    assert fewer.seating_scale < 1.0 and more.seating_scale < 1.0
    assert fewer.mass_scale == more.mass_scale == 1.0

    # then their receipts are byte-identical, and so are the gate's verdicts
    # on the frames each stage wrote, whose details carry no count.
    receipts = [
        json.dumps(result.evidence(), sort_keys=True) for result in (fewer, more)
    ]
    assert receipts[0] == receipts[1]
    # So are the results' reprs, which leave out the exact figures and the
    # frame.
    assert repr(fewer) == repr(more)
    for name in (
        "frame",
        "band_rows",
        "funding_rows",
        "seating_scale",
        "donor_count",
        "donor_source_households",
        "reallocated_mass",
    ):
        assert f"{name}=" not in repr(fewer).replace("expected_donor_count=", ""), name
    gates = [_capped_gate(result.evidence(), result.frame) for result in (fewer, more)]
    assert gates[0].passed, gates[0].failures
    assert gates[0] == gates[1]
    serialized = [
        json.dumps(
            [gate.name, gate.passed, list(gate.failures), dict(gate.details)],
            sort_keys=True,
        )
        for gate in gates
    ]
    assert serialized[0] == serialized[1]
    assert set(gates[0].details) == {
        "stage",
        "check",
        "bands_checked",
        "clone_count",
        "seating_scale",
        "mass_scale",
        "funding_within_share",
        "funding_matches_bands",
    }


def test_the_capped_gate_reads_each_seat_once_through_every_later_copy() -> None:
    # Given stage 17's capped seating on three clones, carried through every
    # layer stacked after it, which copies donors in copy-id fields above the
    # band donors' own,
    propensity = _propensity()
    stage_17, built = _stage_17_and_artifact(3, propensity)
    evidence = {**stage_17.evidence(), "sample_fraction": 0.01}
    assert evidence["seating_scale"] == "capped"
    household = built.artifact.table("household")
    donors = household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].to_numpy()
    copied_later = _stage_17_ids(household) != household["household_id"].to_numpy()
    assert (donors & copied_later).any()

    # whose donor rows, counted with their later copies, split unevenly
    # across the clones,
    naive = (
        household.loc[donors]
        .groupby([SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN, _CLONE])
        .size()
        .unstack(fill_value=0)
    )
    assert (naive.nunique(axis=1) > 1).any()

    # then the gate reads each band's seats as the stage wrote them, evenly
    # split and each holding its one carrier, on the artifact as on the
    # frame stage 17 wrote.
    for frame in (stage_17.frame, built.artifact):
        gate = _capped_gate(evidence, frame)
        assert gate.passed, gate.failures
        assert gate.details["clone_count"] == 3
        assert gate.details["seating_scale"] == "capped"


def test_the_capped_gate_fails_a_dropped_carrier_and_an_uneven_split() -> None:
    propensity = _propensity()
    stage_17, _ = _stage_17_and_artifact(3, propensity)
    evidence = {**stage_17.evidence(), "sample_fraction": 0.01}
    frame = stage_17.frame
    household = frame.table("household")
    person = frame.table("person")
    band_of = household.set_index("household_id")[
        SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN
    ]
    messages: list[str] = []

    # A seat whose carrier lost its flag holds no carrier.
    carriers = np.flatnonzero(person[PERSON_IS_SPI_INCOME_BAND_CARRIER].to_numpy())
    dropped = person.copy()
    dropped.loc[carriers[0], PERSON_IS_SPI_INCOME_BAND_CARRIER] = False
    band = int(band_of[person.loc[carriers[0], "person_household_id"]])
    gate = _capped_gate(evidence, _with_tables(frame, person=dropped))
    assert not gate.passed
    assert gate.failures == (
        f"spi_income_band_donors: band from {band}'s seats do not each hold one "
        "carrier.",
    )
    messages.extend(gate.failures)

    # A seat of clone 1 that is no longer a donor leaves its band uneven
    # across the clones.
    seats = household.loc[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    seat = seats.loc[seats[_CLONE].eq(1)].iloc[0]
    unseated = household.copy()
    unseated.loc[
        unseated["household_id"].eq(seat["household_id"]),
        HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    ] = False
    band = int(seat[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN])
    gate = _capped_gate(evidence, _with_tables(frame, household=unseated))
    assert not gate.passed
    assert gate.failures == (
        f"spi_income_band_donors: band from {band}'s donors do not split evenly "
        "across the geographic clones.",
    )
    messages.extend(gate.failures)

    # A receipt read against another clone count splits unevenly too, and a
    # capped receipt without the frame is not passed.
    gate = _capped_gate({**evidence, "clone_count": 2}, frame)
    assert not gate.passed
    messages.extend(gate.failures)
    gate = _capped_gate(evidence, None)
    assert gate.failures == (
        "spi_income_band_donors: a capped seating's seats are recomputed from the "
        "phase's frame, and the gate was given none.",
    )
    messages.extend(gate.failures)

    # No message gives a count, a weight or a mass, and no capped verdict's
    # details give the donors, their mass or their heaviest weight.
    assert messages and all(_no_data_digit(message) for message in messages)
    for verdict in (gate, _capped_gate(evidence, frame)):
        assert not {"donor_count", "reallocated_mass", "heaviest_donor_weight"} & set(
            verdict.details
        )
