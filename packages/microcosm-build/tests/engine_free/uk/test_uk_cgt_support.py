"""The CGT support split (microcosm#1045): selection, split, receipt, closed world."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_INVESTABLE_WEALTH_COLUMNS,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    uk_cgt_taxable_income_proxy,
)
from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_CLONE_SPLIT_FACTOR,
    CGT_SUPPORT_COPIES_COLUMN,
    CGT_SUPPORT_COPY_INDEX_COLUMN,
    CGT_SUPPORT_EXPECTED_MASS,
    CGT_SUPPORT_HEADROOM,
    CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS,
    CGT_SUPPORT_MASS_CHANGE_REASON,
    CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
    CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
    CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS,
    CGT_SUPPORT_SPLIT_OPERATION_KIND,
    CGT_SUPPORT_SPLIT_STAGE_NAME,
    HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    UKCGTSupportSplitStageTransform,
    _assert_cgt_support_split_stage_parameters,
    cgt_support_copy_counts,
    cgt_support_household_wealth,
    cgt_support_income_band,
    cgt_support_mass_by_income_band,
    cgt_support_split_operation_parameters,
    select_cgt_support_households,
    split_cgt_support_households,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_cgt_support import (
    BAND_INCOMES,
    PARAMETERS,
    drift,
    support_distribution,
    support_frame,
    support_stage,
    support_tables,
)

# --- the reference scenario -------------------------------------------------
#
# Seven one-person households (ids 1-7, multiplier 10). Band 0 (income
# 20,000): ids 1-4 and the zero-weight id 7; band 37,700 (income 60,000):
# ids 5-6. The synthetic joint publishes 60 top-band taxpayers in band 0
# (support mass 2 x 2 x 60 = 240, one suppressed cell) and 10 in band 37,700
# (support mass 40). Band 0 walks by wealth descending then id ascending: id
# 2 (500, 130) then id 3 (500, 121; the tie breaks on id) reach 251, so id 3
# is the crossing household and ids 4 (200) and 1 (100) stay. Band 37,700
# walks id 6 (20, 61) first, which crosses at once; id 5 (10, 90) stays and
# is the heaviest unselected incumbent, so the exact-total correction lands
# on it. Split: id 2 into 3 copies, id 3 into 3, id 6 into 2.

SCENARIO_WEIGHTS = [70.0, 130.0, 121.0, 30.0, 90.0, 61.0, 0.0]
SCENARIO_WEALTH = [100.0, 500.0, 500.0, 200.0, 10.0, 20.0, 1_000_000.0]
SCENARIO_INCOMES = [
    BAND_INCOMES[0],
    BAND_INCOMES[0],
    BAND_INCOMES[0],
    BAND_INCOMES[0],
    BAND_INCOMES[37_700],
    BAND_INCOMES[37_700],
    BAND_INCOMES[0],
]
SCENARIO_CHANNELS = ["frs", "spi", "frs", "frs", "spi", "frs", "frs"]
SCENARIO_TOP_CELLS = {
    0: {250_000: 60.0, 500_000: None},
    37_700: {250_000: 10.0},
}
SCENARIO_SUPPORT = {0: 240.0, 37_700: 40.0}
#: The reference scenario's support masses (2 x 2 x published) were tuned to a
#: headroom of 2.0; the packaged headroom is asserted separately.
SCENARIO_HEADROOM = 2.0


def _scenario_frame(**overrides):
    keywords = {
        "weights": SCENARIO_WEIGHTS,
        "wealth": SCENARIO_WEALTH,
        "incomes": SCENARIO_INCOMES,
        "channels": SCENARIO_CHANNELS,
    }
    keywords.update(overrides)
    return support_frame(**keywords)


def _scenario_split(**overrides):
    return split_cgt_support_households(
        _scenario_frame(**overrides),
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        headroom=SCENARIO_HEADROOM,
    )


def _household_weights(frame) -> pd.Series:
    return pd.Series(
        frame.weights_for("household").values,
        index=frame.table("household")["household_id"].to_numpy(),
    )


# --- support mass -------------------------------------------------------------


def test_support_mass_counts_top_bands_and_treats_suppressed_cells_as_zero():
    rows = cgt_support_mass_by_income_band(
        support_distribution(
            {
                0: {250_000: 60.0, 500_000: None, 5_000_000: None},
                50_000: {100_000: 999.0, 250_000: 3.0, 1_000_000: 4.0},
            }
        ),
        clone_split_factor=2,
        headroom=2.0,
        minimum_gain_band_lower=250_000,
    )

    assert [row["income_lower_bound"] for row in rows] == list(
        CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS
    )
    by_band = {row["income_lower_bound"]: row for row in rows}
    assert by_band[0]["published_top_band_taxpayers"] == 60.0
    assert by_band[0]["suppressed_cells"] == 2
    assert by_band[0]["support_mass"] == 240.0
    # The 100,000 band of gains sits below the minimum and is ignored.
    assert by_band[50_000]["published_top_band_taxpayers"] == 7.0
    assert by_band[50_000]["suppressed_cells"] == 0
    assert by_band[50_000]["support_mass"] == 28.0
    for income_lower in (37_700, 100_000, 125_140, 200_000):
        assert by_band[income_lower]["published_top_band_taxpayers"] == 0.0
        assert by_band[income_lower]["support_mass"] == 0.0
        assert by_band[income_lower]["suppressed_cells"] == 0


def test_support_mass_follows_its_parameters_and_refuses_bad_ones():
    distribution = support_distribution({0: {250_000: 10.0}})
    rows = cgt_support_mass_by_income_band(
        distribution, clone_split_factor=3, headroom=1.5, minimum_gain_band_lower=0
    )
    # The lowest band of gains now counts: 1,000 low-band people plus 10.
    assert rows[0]["published_top_band_taxpayers"] == 1_010.0
    assert rows[0]["support_mass"] == 3 * 1.5 * 1_010.0
    with pytest.raises(ValueError, match="split factor"):
        cgt_support_mass_by_income_band(distribution, clone_split_factor=0)
    with pytest.raises(ValueError, match="headroom"):
        cgt_support_mass_by_income_band(distribution, headroom=0.0)


def test_support_mass_pins_hold_on_the_vendored_joint():
    joint = load_hmrc_cgt_joint_distribution()
    rows = cgt_support_mass_by_income_band(
        joint,
        clone_split_factor=CGT_SUPPORT_CLONE_SPLIT_FACTOR,
        headroom=CGT_SUPPORT_HEADROOM,
        minimum_gain_band_lower=CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
    )

    assert [row["published_top_band_taxpayers"] for row in rows] == [
        17_000.0,
        5_000.0,
        11_000.0,
        3_000.0,
        6_000.0,
        16_000.0,
    ]
    assert [row["support_mass"] for row in rows] == [
        42_500.0,
        12_500.0,
        27_500.0,
        7_500.0,
        15_000.0,
        40_000.0,
    ]
    assert (
        sum(row["published_top_band_taxpayers"] for row in rows)
        == CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS
        == 58_000.0
    )
    assert sum(row["support_mass"] for row in rows) == CGT_SUPPORT_EXPECTED_MASS
    assert CGT_SUPPORT_EXPECTED_MASS == (
        CGT_SUPPORT_CLONE_SPLIT_FACTOR
        * CGT_SUPPORT_HEADROOM
        * CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS
    )
    suppressed = sum(
        cell.individuals is None
        for cell in joint.cells
        if cell.gain_lower_bound >= CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER
    )
    assert sum(row["suppressed_cells"] for row in rows) == suppressed > 0
    _assert_cgt_support_split_stage_parameters(support_stage())


# --- income band and wealth ---------------------------------------------------


def test_income_band_uses_the_oldest_adult_and_the_redraw_digitisation():
    person, _, household, _ = support_tables(
        weights=[1.0, 1.0, 1.0, 1.0],
        wealth=[0.0, 0.0, 0.0, 0.0],
        # Household 1: the 30-year-old earns 250,000 but the 50-year-old
        # carries. Household 2: tied 45-year-olds, the lower person id
        # carries. Household 3: a child's income never counts. Household 4:
        # exactly on the 37,700 edge (50,270 - 12,570).
        incomes=[
            (BAND_INCOMES[0], BAND_INCOMES[200_000]),
            (BAND_INCOMES[50_000], BAND_INCOMES[200_000]),
            (BAND_INCOMES[0], BAND_INCOMES[200_000]),
            50_270.0,
        ],
        ages=[(50, 30), (45, 45), (40, 12), (45,)],
    )

    bands = cgt_support_income_band(
        person, household_ids=household["household_id"], parameters=PARAMETERS
    )

    assert bands.tolist() == [0, 50_000, 0, 37_700]
    assert bands.dtype == np.dtype("int64")
    # Aligned to the ids handed in, whatever their order.
    reversed_bands = cgt_support_income_band(
        person, household_ids=[4, 3, 2, 1], parameters=PARAMETERS
    )
    assert reversed_bands.tolist() == [37_700, 0, 50_000, 0]
    # Byte-for-byte the redraw's classification of the carriers.
    carriers = person.loc[person.person_id.isin([1, 3, 5, 7])]
    expected = np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)[
        np.digitize(
            uk_cgt_taxable_income_proxy(carriers, PARAMETERS),
            HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[1:],
        )
    ]
    assert bands.tolist() == expected.tolist()


def test_income_band_refuses_a_household_without_an_adult_or_a_missing_component():
    person, _, household, _ = support_tables(
        weights=[1.0], wealth=[0.0], incomes=[(0.0,)], ages=[(12,)]
    )
    with pytest.raises(ValueError, match="adult carrier"):
        cgt_support_income_band(
            person, household_ids=household["household_id"], parameters=PARAMETERS
        )
    person, _, household, _ = support_tables(weights=[1.0], wealth=[0.0], incomes=[1.0])
    with pytest.raises(ValueError, match="components missing"):
        cgt_support_income_band(
            person.drop(columns=["dividend_income"]),
            household_ids=household["household_id"],
            parameters=PARAMETERS,
        )


def test_household_wealth_sums_the_investable_columns_and_refuses_gaps():
    _, _, household, _ = support_tables(
        weights=[1.0, 1.0], wealth=[10.0, 20.0], incomes=[1.0, 1.0]
    )
    household[UK_CGT_INVESTABLE_WEALTH_COLUMNS[1]] = [1.0, 2.0]
    household[UK_CGT_INVESTABLE_WEALTH_COLUMNS[3]] = [100.0, 200.0]

    assert cgt_support_household_wealth(household).tolist() == [111.0, 222.0]

    with pytest.raises(ValueError, match="lacks"):
        cgt_support_household_wealth(
            household.drop(columns=[UK_CGT_INVESTABLE_WEALTH_COLUMNS[2]])
        )
    for bad in (np.nan, np.inf, -np.inf):
        broken = household.copy()
        broken.loc[0, UK_CGT_INVESTABLE_WEALTH_COLUMNS[0]] = bad
        with pytest.raises(ValueError, match="finite"):
            cgt_support_household_wealth(broken)


# --- selection ----------------------------------------------------------------


def test_selection_walks_wealth_descending_then_id_ascending_whole_households():
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7])
    weights = np.asarray(SCENARIO_WEIGHTS)
    bands = np.asarray([0, 0, 0, 0, 37_700, 37_700, 0])
    wealth = np.asarray(SCENARIO_WEALTH)

    selected, rows = select_cgt_support_households(
        ids, weights, bands, wealth, SCENARIO_SUPPORT
    )

    # Band 0 at 240: id 2 (500, 130) then id 3 (500, 121; the tie breaks on
    # id) reach 251 and id 3 is the crossing household, taken whole; id 4
    # (200) and id 1 (100) stay. Band 37,700 at 40: id 6 (20, 61) crosses at
    # once; id 5 (10, 90) stays.
    assert ids[selected].tolist() == [2, 3, 6]
    band_0, band_37700 = rows
    assert band_0 == {
        "income_lower_bound": 0,
        "support_mass": 240.0,
        "pool_households": 4,
        "pool_mass": 351.0,
        "households_selected": 2,
        "selected_mass": 251.0,
        "wealth_threshold": 500.0,
        "heaviest_selected_weight": 130.0,
        "pool_exhausted": False,
    }
    assert band_37700 == {
        "income_lower_bound": 37_700,
        "support_mass": 40.0,
        "pool_households": 2,
        "pool_mass": 151.0,
        "households_selected": 1,
        "selected_mass": 61.0,
        "wealth_threshold": 20.0,
        "heaviest_selected_weight": 61.0,
        "pool_exhausted": False,
    }


def test_selection_takes_the_crossing_household_whole_and_stops_there():
    ids = np.arange(1, 5)
    weights = np.asarray([10.0, 10.0, 10.0, 10.0])
    bands = np.zeros(4, dtype=int)
    wealth = np.asarray([4.0, 3.0, 2.0, 1.0])

    for support, expected in ((0.0, []), (1.0, [1]), (10.0, [1]), (10.5, [1, 2])):
        selected, (row,) = select_cgt_support_households(
            ids, weights, bands, wealth, {0: support}
        )
        assert ids[selected].tolist() == expected
        assert row["households_selected"] == len(expected)
        assert row["pool_exhausted"] is False
    selected, (row,) = select_cgt_support_households(
        ids, weights, bands, wealth, {0: 40.0}
    )
    assert ids[selected].tolist() == [1, 2, 3, 4]
    assert row["pool_exhausted"] is False


def test_selection_records_pool_exhaustion_and_empty_bands_without_refusing():
    ids = np.asarray([1, 2, 3])
    weights = np.asarray([5.0, 7.0, 3.0])
    bands = np.asarray([0, 0, 0])
    wealth = np.asarray([1.0, 2.0, 3.0])

    selected, rows = select_cgt_support_households(
        ids, weights, bands, wealth, {0: 1_000.0, 50_000: 40.0, 200_000: 0.0}
    )

    assert selected.tolist() == [True, True, True]
    exhausted, empty, unneeded = rows
    assert exhausted["pool_exhausted"] is True
    assert exhausted["households_selected"] == 3
    assert exhausted["selected_mass"] == 15.0
    assert exhausted["wealth_threshold"] == 1.0
    assert empty == {
        "income_lower_bound": 50_000,
        "support_mass": 40.0,
        "pool_households": 0,
        "pool_mass": 0.0,
        "households_selected": 0,
        "selected_mass": 0.0,
        "wealth_threshold": None,
        "heaviest_selected_weight": 0.0,
        "pool_exhausted": True,
    }
    assert unneeded["pool_exhausted"] is False
    assert unneeded["households_selected"] == 0


def test_selection_excludes_zero_weight_rows_and_is_permutation_stable():
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7])
    weights = np.asarray(SCENARIO_WEIGHTS)
    bands = np.asarray([0, 0, 0, 0, 37_700, 37_700, 0])
    wealth = np.asarray(SCENARIO_WEALTH)
    selected, rows = select_cgt_support_households(
        ids, weights, bands, wealth, SCENARIO_SUPPORT
    )
    # Id 7 is the wealthiest household of band 0 and weighs nothing.
    assert not selected[6]
    assert rows[0]["pool_households"] == 4

    permutation = np.asarray([6, 4, 0, 5, 2, 1, 3])
    permuted, permuted_rows = select_cgt_support_households(
        ids[permutation],
        weights[permutation],
        bands[permutation],
        wealth[permutation],
        SCENARIO_SUPPORT,
    )

    assert permuted.tolist() == selected[permutation].tolist()
    assert permuted_rows == rows


def test_selection_refuses_inconsistent_inputs():
    with pytest.raises(ValueError, match="one entry per household"):
        select_cgt_support_households([1, 2], [1.0], [0, 0], [0.0, 0.0], {0: 1.0})
    with pytest.raises(ValueError, match="unique household ids"):
        select_cgt_support_households([1, 1], [1.0, 1.0], [0, 0], [0.0, 0.0], {0: 1.0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        select_cgt_support_households([1, 2], [1.0, -1.0], [0, 0], [0.0, 0.0], {0: 1.0})
    with pytest.raises(ValueError, match="wealth must be finite"):
        select_cgt_support_households(
            [1, 2], [1.0, 1.0], [0, 0], [0.0, np.nan], {0: 1.0}
        )
    with pytest.raises(ValueError, match="support mass"):
        select_cgt_support_households([1, 2], [1.0, 1.0], [0, 0], [0.0, 0.0], {0: -1.0})


# --- copy counts --------------------------------------------------------------


def test_copy_counts_are_integer_ceilings_of_at_least_one():
    counts = cgt_support_copy_counts(
        [0.0, 1.0, 60.0, 60.0001, 120.0, 121.0, 600.0], 60.0
    )

    assert counts.tolist() == [1, 1, 1, 2, 2, 3, 10]
    assert counts.dtype == np.dtype("int64")
    assert cgt_support_copy_counts([59.0]).tolist() == [
        math.ceil(59.0 / CGT_SUPPORT_MAXIMUM_COPY_WEIGHT)
    ]
    with pytest.raises(ValueError, match="maximum copy weight"):
        cgt_support_copy_counts([1.0], 0.0)
    with pytest.raises(ValueError, match="finite"):
        cgt_support_copy_counts([np.nan], 60.0)


# --- the split ----------------------------------------------------------------


def test_split_conserves_mass_bitwise_and_corrects_only_an_unselected_incumbent():
    frame = _scenario_frame()
    result = _scenario_split()
    old = frame.weights_for("household").total
    new = result.frame.weights_for("household")

    assert new.total == old
    assert new.kind is WeightKind.IMPORTANCE
    assert result.frame.mass_log[-1].reason == CGT_SUPPORT_MASS_CHANGE_REASON
    assert result.frame.mass_log[-1].declared_factor == 1.0
    assert result.frame.mass_log[-1].old_total == old
    assert result.frame.mass_log[-1].new_total == old
    assert len(result.frame.mass_log) == len(frame.mass_log) + 1
    weights = _household_weights(result.frame)
    # Roots and copies carry bitwise w / n: id 2 (130) in three, id 3 (121)
    # in three, id 6 (61) in two, ids offset by k x 10.
    for household_id, weight, copies in ((2, 130.0, 3), (3, 121.0, 3), (6, 61.0, 2)):
        family = [household_id + k * 10 for k in range(copies)]
        assert weights.loc[family].tolist() == [weight / copies] * copies
        assert (weights.loc[family] <= CGT_SUPPORT_MAXIMUM_COPY_WEIGHT).all()
    # The uncorrected vector drifts by rounding (that is why this scenario was
    # chosen), so the correction had to land somewhere: it landed on id 5,
    # the heaviest unselected incumbent, and nowhere else.
    assert weights.loc[5] != 90.0
    assert weights.loc[5] == pytest.approx(90.0, rel=1e-12)
    assert weights.loc[[1, 4, 7]].tolist() == [70.0, 30.0, 0.0]
    assert result.old_total == old
    assert result.new_total == old


def test_split_offsets_every_entity_id_by_k_times_the_multiplier():
    frame = _scenario_frame(
        ages=[(45,), (50, 30, 8), (45,), (45,), (45,), (60, 58), (45,)],
        benunits=[(0,), (0, 1, 1), (0,), (0,), (0,), (0, 1), (0,)],
    )
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    assert multiplier == 100

    result = split_cgt_support_households(
        frame,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        headroom=SCENARIO_HEADROOM,
    )
    out_person = result.frame.table("person")
    out_benunit = result.frame.table("benunit")
    out_household = result.frame.table("household")

    expected_households = sorted(
        household["household_id"].tolist()
        + [2 + multiplier, 3 + multiplier, 6 + multiplier]
        + [2 + 2 * multiplier, 3 + 2 * multiplier]
    )
    assert out_household["household_id"].tolist() == expected_households
    for household_id, copies in ((2, 3), (3, 3), (6, 2)):
        root_people = person.loc[person.person_household_id == household_id]
        root_benunits = benunit.loc[
            benunit.benunit_id.isin(root_people.person_benunit_id)
        ]
        for k in range(1, copies):
            offset = k * multiplier
            copy_people = out_person.loc[
                out_person.person_household_id == household_id + offset
            ]
            assert sorted(copy_people.person_id) == sorted(
                root_people.person_id + offset
            )
            assert sorted(copy_people.person_benunit_id) == sorted(
                root_people.person_benunit_id + offset
            )
            copy_benunits = out_benunit.loc[
                out_benunit.benunit_id.isin(root_benunits.benunit_id + offset)
            ]
            assert len(copy_benunits) == len(root_benunits)
    # Nothing else was minted: every new person and benunit belongs to a copy.
    assert len(out_person) == len(person) + 2 * 3 + 2 * 1 + 1 * 2
    assert len(out_benunit) == len(benunit) + 2 * 2 + 2 * 1 + 1 * 2
    for column in ("person_id", "person_household_id", "person_benunit_id"):
        assert out_person[column].dtype == np.dtype("int64")
    assert out_benunit["benunit_id"].dtype == np.dtype("int64")
    assert out_household["household_id"].dtype == np.dtype("int64")


def test_split_flags_only_copies_and_writes_the_family_copy_count():
    result = _scenario_split()
    household = result.frame.table("household").set_index("household_id")

    assert household[HOUSEHOLD_IS_CGT_SUPPORT_COPY].dtype == np.dtype(bool)
    assert household[CGT_SUPPORT_COPIES_COLUMN].dtype == np.dtype("int64")
    assert household.index[household[HOUSEHOLD_IS_CGT_SUPPORT_COPY]].tolist() == [
        12,
        13,
        16,
        22,
        23,
    ]
    assert household[CGT_SUPPORT_COPIES_COLUMN].to_dict() == {
        1: 1,
        2: 3,
        3: 3,
        4: 1,
        5: 1,
        6: 2,
        7: 1,
        12: 3,
        13: 3,
        16: 2,
        22: 3,
        23: 3,
    }
    assert result.households_before == 7
    assert result.households_after == 12
    assert result.copies_created == 5
    assert result.households_selected == 3
    assert result.selected_mass == 130.0 + 121.0 + 61.0
    assert result.zero_weight_excluded == 1
    assert result.frs_selected == 2
    assert result.spi_selected == 1


def test_split_copies_every_column_and_membership_unchanged():
    frame = _scenario_frame(
        ages=[(45,), (50, 30, 8), (45,), (45,), (45,), (60, 58), (45,)],
        benunits=[(0,), (0, 1, 1), (0,), (0,), (0,), (0, 1), (0,)],
    )
    result = split_cgt_support_households(
        frame,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        headroom=SCENARIO_HEADROOM,
    )
    person = result.frame.table("person")
    benunit = result.frame.table("benunit")
    household = result.frame.table("household")
    id_columns = {"person_id", "person_household_id", "person_benunit_id"}

    for household_id, copies in ((2, 3), (3, 3), (6, 2)):
        root = household.loc[household.household_id == household_id]
        root_people = person.loc[person.person_household_id == household_id]
        root_people = root_people.sort_values("person_id").reset_index(drop=True)
        root_benunits = benunit.loc[
            benunit.benunit_id.isin(root_people.person_benunit_id)
        ].reset_index(drop=True)
        for k in range(1, copies):
            offset = k * 100
            copy = household.loc[household.household_id == household_id + offset]
            # The explicit lineage: the root at 0, copy k at k.
            assert root[CGT_SUPPORT_COPY_INDEX_COLUMN].item() == 0
            assert copy[CGT_SUPPORT_COPY_INDEX_COLUMN].item() == k
            assert copy[CGT_SUPPORT_COPY_INDEX_COLUMN].dtype == "int64"
            pd.testing.assert_frame_equal(
                root.drop(
                    columns=[
                        "household_id",
                        HOUSEHOLD_IS_CGT_SUPPORT_COPY,
                        CGT_SUPPORT_COPY_INDEX_COLUMN,
                    ]
                ).reset_index(drop=True),
                copy.drop(
                    columns=[
                        "household_id",
                        HOUSEHOLD_IS_CGT_SUPPORT_COPY,
                        CGT_SUPPORT_COPY_INDEX_COLUMN,
                    ]
                ).reset_index(drop=True),
            )
            copy_people = (
                person.loc[person.person_household_id == household_id + offset]
                .sort_values("person_id")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                root_people.drop(columns=list(id_columns)),
                copy_people.drop(columns=list(id_columns)),
            )
            copy_benunits = benunit.loc[
                benunit.benunit_id.isin(root_benunits.benunit_id + offset)
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                root_benunits.drop(columns=["benunit_id"]),
                copy_benunits.drop(columns=["benunit_id"]),
            )
    # Incumbent rows are untouched apart from the two new columns.
    incoming = frame.table("household")
    pd.testing.assert_frame_equal(
        incoming.reset_index(drop=True),
        household.iloc[: len(incoming)]
        .drop(
            columns=[
                HOUSEHOLD_IS_CGT_SUPPORT_COPY,
                CGT_SUPPORT_COPIES_COLUMN,
                CGT_SUPPORT_COPY_INDEX_COLUMN,
            ]
        )
        .reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        frame.table("person").reset_index(drop=True),
        person.iloc[: len(frame.table("person"))].reset_index(drop=True),
    )


def test_split_is_stable_under_person_order():
    reference = _scenario_split()
    reversed_people = _scenario_split(person_order=list(range(6, -1, -1)))
    shuffled_people = _scenario_split(person_order=[3, 0, 6, 1, 5, 2, 4])

    for other in (reversed_people, shuffled_people):
        assert other.evidence() == reference.evidence()
        assert (
            other.frame.table("household")["household_id"].tolist()
            == reference.frame.table("household")["household_id"].tolist()
        )
        assert (
            other.frame.weights_for("household").values.tolist()
            == reference.frame.weights_for("household").values.tolist()
        )
        assert sorted(other.frame.table("person")["person_id"]) == sorted(
            reference.frame.table("person")["person_id"]
        )
        assert (
            other.frame.table("benunit")["benunit_id"].tolist()
            == reference.frame.table("benunit")["benunit_id"].tolist()
        )


def test_split_records_pool_exhaustion_and_empty_bands_without_refusing():
    # Band 0 asks for 4,000 against a pool of 351; band 125,140 asks for 40
    # with no household at all.
    result = split_cgt_support_households(
        _scenario_frame(),
        distribution=support_distribution(
            {0: {250_000: 1_000.0}, 125_140: {250_000: 10.0}}
        ),
        parameters=PARAMETERS,
        headroom=SCENARIO_HEADROOM,
    )
    bands = {row["income_lower_bound"]: row for row in result.band_rows}

    assert bands[0]["pool_exhausted"] is True
    assert bands[0]["households_selected"] == 4
    assert bands[0]["selected_mass"] == 351.0
    assert bands[0]["copies_created"] == (2 + 2 + 1 + 0)
    assert bands[0]["wealth_threshold"] == 100.0
    assert bands[0]["heaviest_selected_weight"] == 130.0
    assert bands[0]["heaviest_copy_weight"] == 130.0 / 3
    assert bands[125_140] == {
        "income_lower_bound": 125_140,
        "published_top_band_taxpayers": 10.0,
        "suppressed_cells": 0,
        "support_mass": 40.0,
        "pool_households": 0,
        "pool_mass": 0.0,
        "households_selected": 0,
        "copies_created": 0,
        "selected_mass": 0.0,
        "wealth_threshold": None,
        "heaviest_selected_weight": 0.0,
        "heaviest_copy_weight": 0.0,
        "pool_exhausted": True,
    }
    assert bands[37_700]["households_selected"] == 0
    assert bands[37_700]["pool_exhausted"] is False
    assert result.frame.weights_for("household").total == 502.0


def test_split_falls_back_when_every_incumbent_is_selected():
    frame = support_frame(
        weights=[130.0, 121.0], wealth=[5.0, 6.0], incomes=[BAND_INCOMES[0]] * 2
    )
    result = split_cgt_support_households(
        frame,
        distribution=support_distribution({0: {250_000: 1_000.0}}),
        parameters=PARAMETERS,
        headroom=SCENARIO_HEADROOM,
    )

    assert result.households_selected == 2
    assert result.copies_created == 4
    assert result.frame.weights_for("household").total == 251.0
    assert result.band_rows[0]["pool_exhausted"] is True


def test_split_counts_suppressed_cells_and_reports_channels_when_present():
    result = split_cgt_support_households(
        support_frame(
            weights=[100.0, 100.0], wealth=[1.0, 2.0], incomes=[BAND_INCOMES[0]] * 2
        ),
        distribution=support_distribution(
            {0: {250_000: None, 500_000: None, 1_000_000: 5.0}}
        ),
        parameters=PARAMETERS,
        headroom=SCENARIO_HEADROOM,
    )

    assert result.suppressed_cells == 2
    assert result.band_rows[0]["suppressed_cells"] == 2
    assert result.band_rows[0]["published_top_band_taxpayers"] == 5.0
    assert result.band_rows[0]["support_mass"] == 20.0
    assert result.households_selected == 1
    # No support-channel column: both counts are zero rather than missing.
    assert result.evidence()["support_channel_split"] == {"frs": 0, "spi": 0}


def test_split_refuses_a_cloned_or_already_split_frame():
    person, benunit, household, weights = support_tables(
        weights=SCENARIO_WEIGHTS, wealth=SCENARIO_WEALTH, incomes=SCENARIO_INCOMES
    )
    for column, message in (
        (HOUSEHOLD_IS_CGT_CLONE, "before cgt_incidence_clone"),
        (HOUSEHOLD_IS_CGT_SUPPORT_COPY, "already ran"),
        (CGT_SUPPORT_COPIES_COLUMN, "already ran"),
    ):
        marked = household.copy()
        marked[column] = False if column != CGT_SUPPORT_COPIES_COLUMN else 1
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=marked,
            household_weights=weights,
            time_period="2024",
        )
        with pytest.raises(ValueError, match=message):
            split_cgt_support_households(
                frame,
                distribution=support_distribution(SCENARIO_TOP_CELLS),
                parameters=PARAMETERS,
            )


def test_split_refuses_missing_or_non_finite_wealth_and_missing_income():
    person, benunit, household, weights = support_tables(
        weights=SCENARIO_WEIGHTS, wealth=SCENARIO_WEALTH, incomes=SCENARIO_INCOMES
    )

    def _split(person_table, household_table):
        frame = uk_national_frame(
            person=person_table,
            benunit=benunit,
            household=household_table,
            household_weights=weights,
            time_period="2024",
        )
        return split_cgt_support_households(
            frame,
            distribution=support_distribution(SCENARIO_TOP_CELLS),
            parameters=PARAMETERS,
        )

    with pytest.raises(ValueError, match="lacks"):
        _split(person, household.drop(columns=[UK_CGT_INVESTABLE_WEALTH_COLUMNS[0]]))
    broken = household.copy()
    broken.loc[2, UK_CGT_INVESTABLE_WEALTH_COLUMNS[1]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _split(person, broken)
    with pytest.raises(ValueError, match="components missing"):
        _split(person.drop(columns=["property_income"]), household)
    # Refusals leave nothing behind: a clean frame still splits.
    assert _split(person, household).households_selected == 3


def test_evidence_is_json_serialisable_with_plain_scalars():
    evidence = _scenario_split().evidence()

    json.dumps(evidence)
    assert evidence["stage"] == CGT_SUPPORT_SPLIT_STAGE_NAME
    assert set(evidence) == {
        "stage",
        "bands",
        "totals",
        "mass",
        "parameters",
        "support_channel_split",
    }
    assert evidence["totals"] == {
        "published_top_band_taxpayers": 70.0,
        "suppressed_cells": 1,
        "support_mass": 280.0,
        "households_selected": 3,
        "copies_created": 5,
        "selected_mass": 312.0,
        "households_before": 7,
        "households_after": 12,
        "zero_weight_excluded": 1,
    }
    assert evidence["mass"] == {"old_total": 502.0, "new_total": 502.0}
    assert evidence["parameters"] == {
        "clone_split_factor": 2,
        "headroom": 2.0,
        "maximum_copy_weight": 60.0,
    }
    assert evidence["support_channel_split"] == {"frs": 2, "spi": 1}
    assert [row["income_lower_bound"] for row in evidence["bands"]] == list(
        CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS
    )
    assert set(evidence["bands"][0]) == {
        "income_lower_bound",
        "published_top_band_taxpayers",
        "suppressed_cells",
        "support_mass",
        "pool_households",
        "pool_mass",
        "households_selected",
        "copies_created",
        "selected_mass",
        "wealth_threshold",
        "heaviest_selected_weight",
        "heaviest_copy_weight",
        "pool_exhausted",
    }

    def _leaves(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from _leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from _leaves(item)
        else:
            yield value

    assert all(
        type(leaf) in {int, float, bool, str, type(None)} for leaf in _leaves(evidence)
    )


# --- the closed-world dictionary and the transform ------------------------------


def test_operation_dictionary_restates_the_reviewed_design():
    ((kind, payload),) = cgt_support_split_operation_parameters()

    assert (
        kind
        == CGT_SUPPORT_SPLIT_OPERATION_KIND
        == "split_top_wealth_support_households"
    )
    assert payload["joint_resource"] == "hmrc_cgt_conditioning_facts.json"
    assert payload["joint_record_set_prefix"] == "hmrc.cgt_table3_2026.ty2024."
    assert payload["joint_vintage"] == "2024-25"
    assert payload["taxpayer_concept"] == "hmrc.cgt_taxpayers_individuals"
    assert payload["income_band_lower_bounds"] == [
        0,
        37_700,
        50_000,
        100_000,
        125_140,
        200_000,
    ]
    assert payload["minimum_gain_band_lower"] == 250_000
    assert payload["published_top_band_taxpayers"] == 58_000.0
    assert payload["clone_split_factor"] == 2
    assert payload["headroom"] == 1.25
    assert payload["expected_support_mass"] == 145_000.0
    assert payload["income_proxy_components"] == list(
        UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS
    )
    assert payload["allowance_subtraction"] is True
    assert payload["adult_minimum_age"] == 16
    assert payload["investable_wealth_columns"] == list(
        UK_CGT_INVESTABLE_WEALTH_COLUMNS
    )
    assert payload["maximum_copy_weight"] == 60.0
    assert payload["flag_column"] == HOUSEHOLD_IS_CGT_SUPPORT_COPY
    assert payload["copies_column"] == CGT_SUPPORT_COPIES_COLUMN
    assert payload["copy_index_column"] == CGT_SUPPORT_COPY_INDEX_COLUMN
    assert payload["weight_kind_out"] == "importance"
    assert payload["conservation"] == "exact_total"
    assert payload["declared_factor"] == 1.0
    assert payload["reason"] == CGT_SUPPORT_MASS_CHANGE_REASON
    assert list(payload) == [
        "joint_resource",
        "joint_record_set_prefix",
        "joint_vintage",
        "taxpayer_concept",
        "income_band_lower_bounds",
        "minimum_gain_band_lower",
        "published_top_band_taxpayers",
        "suppressed_cells",
        "clone_split_factor",
        "headroom",
        "expected_support_mass",
        "support_mass_rule",
        "income_band_measure",
        "income_proxy_components",
        "allowance_subtraction",
        "adult_minimum_age",
        "carrier",
        "wealth_measure",
        "investable_wealth_columns",
        "selection",
        "maximum_copy_weight",
        "copy_rule",
        "id_remapping",
        "flag_column",
        "copies_column",
        "copy_index_column",
        "values",
        "weight_kind_out",
        "conservation",
        "declared_factor",
        "reason",
    ]
    json.dumps(payload)


@pytest.mark.parametrize(
    "parameter",
    [
        name
        for _, payload in cgt_support_split_operation_parameters()
        for name in payload
    ],
)
def test_drift_assert_covers_every_reviewed_parameter(parameter: str) -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_cgt_support_split_stage_parameters(drift(support_stage(), parameter))


def test_drift_assert_rejects_extra_keys_operations_and_stage_shape() -> None:
    stage = support_stage()
    _assert_cgt_support_split_stage_parameters(stage)
    with pytest.raises(ValueError, match="drifted"):
        _assert_cgt_support_split_stage_parameters(drift(stage, "undeclared_extra_key"))
    with pytest.raises(ValueError, match="operation order drifted"):
        _assert_cgt_support_split_stage_parameters(
            replace(stage, operations=(*stage.operations, stage.operations[-1]))
        )
    with pytest.raises(ValueError, match="operation order drifted"):
        _assert_cgt_support_split_stage_parameters(
            replace(
                stage,
                operations=(
                    SourceOperationSpec(
                        kind="stack_band_donor_households",
                        parameters=dict(stage.operations[0].parameters),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="Expected stage"):
        _assert_cgt_support_split_stage_parameters(
            support_stage(stage_name="cgt_band_donors")
        )
    with pytest.raises(ValueError, match="grain"):
        _assert_cgt_support_split_stage_parameters(support_stage(grain="person"))
    with pytest.raises(ValueError, match="outputs"):
        _assert_cgt_support_split_stage_parameters(
            support_stage(outputs=(HOUSEHOLD_IS_CGT_SUPPORT_COPY,))
        )
    with pytest.raises(ValueError, match="outputs"):
        _assert_cgt_support_split_stage_parameters(
            support_stage(
                outputs=(CGT_SUPPORT_COPIES_COLUMN, HOUSEHOLD_IS_CGT_SUPPORT_COPY)
            )
        )
    with pytest.raises(ValueError, match="rewrites"):
        _assert_cgt_support_split_stage_parameters(
            support_stage(rewrites=("capital_gains",))
        )
    facts, policy = stage.artifacts
    with pytest.raises(ValueError, match="cgt_conditioning_facts"):
        _assert_cgt_support_split_stage_parameters(support_stage(artifacts=(policy,)))
    with pytest.raises(ValueError, match="cgt_conditioning_facts"):
        _assert_cgt_support_split_stage_parameters(
            support_stage(
                artifacts=({**facts, "runtime_sha256_required": False}, policy)
            )
        )
    with pytest.raises(ValueError, match="policy-parameter"):
        _assert_cgt_support_split_stage_parameters(support_stage(artifacts=(facts,)))
    with pytest.raises(ValueError, match="policy-parameter"):
        _assert_cgt_support_split_stage_parameters(
            support_stage(
                artifacts=(
                    facts,
                    {**policy, "parameters": ["gov.hmrc.cgt.annual_exempt_amount"]},
                )
            )
        )


def test_pin_assert_rejects_a_moved_vendored_joint(monkeypatch) -> None:
    from microcosm.build.uk_runtime import cgt_support

    moved = support_distribution({0: {250_000: 57_000.0}})
    monkeypatch.setattr(cgt_support, "load_hmrc_cgt_joint_distribution", lambda: moved)
    with pytest.raises(ValueError, match="drifted from the reviewed 2024-25 pins"):
        _assert_cgt_support_split_stage_parameters(support_stage())


def test_transform_binds_the_stage_and_exposes_the_receipt() -> None:
    stage = support_stage()
    with pytest.raises(ValueError, match="drifted"):
        UKCGTSupportSplitStageTransform(stage=drift(stage, "headroom"))

    transform = UKCGTSupportSplitStageTransform(
        stage=stage,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
    )
    assert transform.stage is stage
    assert transform.output_columns() == (
        HOUSEHOLD_IS_CGT_SUPPORT_COPY,
        CGT_SUPPORT_COPIES_COLUMN,
        CGT_SUPPORT_COPY_INDEX_COLUMN,
    )
    assert transform.output_columns() == tuple(stage.outputs)
    assert transform.last_result is None
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()

    frame = _scenario_frame()
    split = transform(frame)

    assert split.weights_for("household").total == frame.weights_for("household").total
    assert transform.last_result is not None
    assert transform.last_result.frame is split
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence == transform.last_result.evidence()
    assert evidence["stage"] == CGT_SUPPORT_SPLIT_STAGE_NAME
    assert evidence["totals"]["copies_created"] == 5
    # A default distribution is the vendored joint: nothing on this small
    # frame reaches the published support masses, so every band exhausts.
    default = UKCGTSupportSplitStageTransform(stage=stage, parameters=PARAMETERS)
    assert default.distribution is None
    default(frame)
    assert default.last_result is not None
    assert default.last_result.published_top_band_taxpayers == 58_000.0
    assert default.last_result.households_selected == 6
    assert all(
        row["pool_exhausted"]
        for row in default.last_result.band_rows
        if row["pool_households"]
    )


def test_packaged_manifest_declares_the_stage() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stage = spec.sources.stage_map()[CGT_SUPPORT_SPLIT_STAGE_NAME]

    _assert_cgt_support_split_stage_parameters(stage)
    transform = UKCGTSupportSplitStageTransform(
        stage=stage,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
    )
    assert transform.output_columns() == tuple(stage.outputs)
    names = [source.stage for source in spec.sources.stages]
    assert names.index(CGT_SUPPORT_SPLIT_STAGE_NAME) + 1 == names.index(
        "cgt_incidence_clone"
    )
    assert "cgt_band_donors" not in names
