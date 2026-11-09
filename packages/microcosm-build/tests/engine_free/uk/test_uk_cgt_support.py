"""The CGT support split (microcosm#1045): selection, split, receipt, closed world."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime import cgt_support
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_INVESTABLE_WEALTH_COLUMNS,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    uk_cgt_taxable_income_proxy,
)
from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_CLONE_SPLIT_FACTOR,
    CGT_SUPPORT_COPIES_COLUMN,
    CGT_SUPPORT_COPY_BOUND_REFUSAL,
    CGT_SUPPORT_COPY_INDEX_COLUMN,
    CGT_SUPPORT_CORRECTION_CANDIDATES,
    CGT_SUPPORT_EXACT_TOTAL_EXCEPTION,
    CGT_SUPPORT_EXPECTED_MASS,
    CGT_SUPPORT_HEADROOM,
    CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS,
    CGT_SUPPORT_MASS_CHANGE_REASON,
    CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
    CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
    CGT_SUPPORT_ONE_COPY_CHECK,
    CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS,
    CGT_SUPPORT_SPLIT_OPERATION_KIND,
    CGT_SUPPORT_SPLIT_STAGE_NAME,
    HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    UKCGTSupportSplitStageTransform,
    _assert_cgt_support_split_stage_parameters,
    _refuse_copies_beyond_the_copy_id_field,
    cgt_support_copy_counts,
    cgt_support_household_wealth,
    cgt_support_income_band,
    cgt_support_mass_by_income_band,
    cgt_support_split_operation_parameters,
    geographic_clone_count,
    select_cgt_support_households,
    split_cgt_support_households,
)
from microcosm.build.uk_runtime.copy_id_table import ordinal_bound, radix
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_support import (
    _exact_total_correction,
    _importance_weights_with_exact_total,
)
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
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
from test_support.microcosm_build.uk_clone_first_frames import (
    assert_copy_ids_follow_lineage,
    clone_first_frame,
    with_last_household_moved,
)

# --- the reference scenario -------------------------------------------------
#
# Seven one-person households (ids 1-7; copies at k x 10**13). Band 0 (income
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


#: The copy-id table's cgt_support_split radix (microcosm#931).
COPY_RADIX = radix("cgt_support_split")


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
        clone_count=1,
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
    with pytest.raises(ValueError, match="adult carrier") as caught:
        cgt_support_income_band(
            person, household_ids=household["household_id"], parameters=PARAMETERS
        )
    # The refusal counts the households and names none (microcosm#931).
    assert str(caught.value) == (
        "Every CGT household requires an adult carrier; 1 household(s) have none."
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
        [0.0, 1.0, 60.0, 60.0001, 120.0, 121.0, 600.0], 60.0, clone_count=1
    )

    assert counts.tolist() == [1, 1, 1, 2, 2, 3, 10]
    assert counts.dtype == np.dtype("int64")
    assert cgt_support_copy_counts([59.0], clone_count=1).tolist() == [
        math.ceil(59.0 / CGT_SUPPORT_MAXIMUM_COPY_WEIGHT)
    ]
    with pytest.raises(ValueError, match="maximum copy weight"):
        cgt_support_copy_counts([1.0], 0.0, clone_count=1)
    with pytest.raises(ValueError, match="finite"):
        cgt_support_copy_counts([np.nan], 60.0, clone_count=1)


# --- geographic clones (microcosm#931) -----------------------------------------
#
# A geographic clone of the clone-first spine carries 1/K of its source
# household's weight, so the copy count reads K x the row's weight: every
# clone of a household gets the count its source gets at K = 1, and each copy
# weighs at most 60 / K. Every weight here is synthetic.


def _todays_copy_counts(weights) -> np.ndarray:
    """The rule before the clone count, spelled out: ceil(w / 60), at least 1."""

    values = np.asarray(weights, dtype=float)
    return np.maximum(
        np.ceil(values / CGT_SUPPORT_MAXIMUM_COPY_WEIGHT).astype("int64"), 1
    )


def _edge_weights() -> np.ndarray:
    """Exact multiples of 60, one ulp either side, tiny, large and random."""

    multiples = CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * np.arange(1.0, 20_001.0)
    rng = np.random.default_rng(931)
    return np.concatenate(
        [
            [0.0, 5e-324, 1e-300, 1e-12, 0.5, 59.999999, 60.0, 60.000001],
            multiples,
            np.nextafter(multiples, np.inf),
            np.nextafter(multiples, 0.0),
            CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * np.asarray([10_000.0, 1e6, 1e9, 1e12]),
            rng.uniform(0.0, 600_000.0, 200_000),
            rng.lognormal(5.0, 2.0, 200_000),
        ]
    )


def test_copy_counts_at_one_clone_are_todays_ceiling_bit_for_bit():
    weights = _edge_weights()

    counts = cgt_support_copy_counts(weights, clone_count=1)

    assert counts.dtype == np.dtype("int64")
    np.testing.assert_array_equal(counts, _todays_copy_counts(weights))
    # The edges the snap would have moved: one ulp above a multiple of 60 is
    # rounded up at one clone, as before.
    above = np.nextafter(CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * 2.0, np.inf)
    assert cgt_support_copy_counts([above], clone_count=1).tolist() == [3]


@pytest.mark.parametrize("clone_count", [3, 7, 15])
def test_copy_counts_of_clone_replicas_are_their_sources_counts(clone_count: int):
    # Exact multiples of 60, one ulp below them and random weights: each
    # replica at W / K gets the count its source W gets at one clone.
    multiples = CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * np.arange(1.0, 20_001.0)
    rng = np.random.default_rng(clone_count)
    sources = np.concatenate(
        [
            multiples,
            np.nextafter(multiples, 0.0),
            rng.uniform(0.0, 600_000.0, 200_000),
            rng.lognormal(5.0, 2.0, 200_000),
        ]
    )
    replicas = sources / clone_count

    counts = cgt_support_copy_counts(replicas, clone_count=clone_count)

    np.testing.assert_array_equal(
        counts, cgt_support_copy_counts(sources, clone_count=1)
    )
    # Every copy of a replica weighs at most 60 / K, to the gate's tolerance.
    assert (
        clone_count * (replicas / counts)
        <= CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * (1.0 + 1e-12)
    ).all()


def test_the_snap_keeps_exact_multiples_exact_at_seven_clones():
    # At K = 7 the replica 60m / 7 times 7 lands an ulp above 60m for some m
    # (the verifier's case): the unsnapped ceiling adds a copy there.
    multiples = CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * np.arange(1.0, 20_001.0)
    replicas = multiples / 7
    unsnapped = np.ceil((7 * replicas) / CGT_SUPPORT_MAXIMUM_COPY_WEIGHT)
    rounded_up = unsnapped != np.arange(1.0, 20_001.0)
    assert rounded_up.sum() > 0
    assert (
        cgt_support_copy_counts(replicas[rounded_up], clone_count=7).tolist()
        == np.arange(1, 20_001)[rounded_up].tolist()
    )


def test_the_snap_counts_a_quotient_ulps_above_an_integer_as_that_integer():
    # The case the snap gives up, as documented: a source one ulp above 120
    # makes 3 copies at one clone, but its replicas count 2 at K > 1.
    source = np.nextafter(2.0 * CGT_SUPPORT_MAXIMUM_COPY_WEIGHT, np.inf)
    assert cgt_support_copy_counts([source], clone_count=1).tolist() == [3]
    for clone_count in (2, 3, 7):
        replica = source / clone_count
        assert cgt_support_copy_counts([replica], clone_count=clone_count).tolist() == [
            2
        ]


@pytest.mark.parametrize("clone_count", [0, -1, True, 1.0, 2.5, None, "3"])
def test_copy_counts_refuse_a_clone_count_that_is_not_a_positive_integer(
    clone_count,
):
    with pytest.raises(ValueError, match="positive integer geographic clone count"):
        cgt_support_copy_counts([1.0], clone_count=clone_count)
    with pytest.raises(ValueError, match="positive integer geographic clone count"):
        _scenario_split_at(clone_count)


def _scenario_split_at(clone_count):
    return split_cgt_support_households(
        _scenario_frame(),
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        clone_count=clone_count,
        headroom=SCENARIO_HEADROOM,
    )


def test_the_clone_index_column_is_the_lineages_geographic_branch():
    from microcosm.build.uk_runtime.cgt_support import _GEOGRAPHIC_CLONE_INDEX_COLUMN
    from microcosm.build.uk_runtime.household_lineage import (
        GEOGRAPHIC_SUPPORT_BRANCH,
        UK_COPY_LINEAGE,
    )
    from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column

    (branch,) = [
        branch
        for branch in UK_COPY_LINEAGE.branches
        if branch.name == GEOGRAPHIC_SUPPORT_BRANCH
    ]
    assert (
        _GEOGRAPHIC_CLONE_INDEX_COLUMN
        == ladder_clone_index_column("household")
        == branch.index
        == "household_clone_index"
    )


def test_geographic_clone_count_is_one_more_than_the_largest_index():
    assert geographic_clone_count(pd.DataFrame({"household_id": [1, 2]})) == 1
    for indices, expected in (([0, 0], 1), ([0, 1, 2, 1, 0], 3), ([2, 0, 1], 3)):
        household = pd.DataFrame(
            {"household_clone_index": np.asarray(indices, dtype="int64")}
        )
        assert geographic_clone_count(household) == expected
    empty = pd.DataFrame({"household_clone_index": np.zeros(0, dtype="int64")})
    assert geographic_clone_count(empty) == 1


@pytest.mark.parametrize(
    "values,message",
    [
        (
            np.asarray([0, 2, 2, 0], dtype="int64"),
            "1 of the 3 indices from 0 to the largest are carried by no household; "
            "the CGT support split counts K geographic clones as the "
            "'household_clone_index' values 0 to K - 1 (microcosm#931).",
        ),
        (
            np.asarray([0, 3], dtype="int64"),
            "2 of the 4 indices from 0 to the largest are carried by no household; "
            "the CGT support split counts K geographic clones as the "
            "'household_clone_index' values 0 to K - 1 (microcosm#931).",
        ),
        (
            np.asarray([0, -1, 1, -2], dtype="int64"),
            "2 household(s) carry a negative 'household_clone_index'; the CGT "
            "support split counts K geographic clones as the "
            "'household_clone_index' values 0 to K - 1 (microcosm#931).",
        ),
        (
            pd.array([0, None, 1], dtype="Int64"),
            "1 household(s) carry no 'household_clone_index'; the CGT support "
            "split counts K geographic clones as the 'household_clone_index' "
            "values 0 to K - 1 (microcosm#931).",
        ),
        (
            np.asarray([0.0, 1.0]),
            "The CGT support split counts geographic clones on an integer "
            "'household_clone_index' column; it has dtype float64.",
        ),
        (
            np.asarray([False, True]),
            "The CGT support split counts geographic clones on an integer "
            "'household_clone_index' column; it has dtype bool.",
        ),
    ],
)
def test_geographic_clone_count_refuses_indices_other_than_0_to_k_by_count(
    values, message: str
):
    household = pd.DataFrame(
        {
            "household_id": 1_000_003 + np.arange(len(values)),
            "household_clone_index": values,
        }
    )

    with pytest.raises(ValueError) as caught:
        geographic_clone_count(household)

    assert str(caught.value) == message
    assert "1000003" not in str(caught.value)


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
    # in three, id 6 (61) in two, ids offset by k x the declared radix.
    for household_id, weight, copies in ((2, 130.0, 3), (3, 121.0, 3), (6, 61.0, 2)):
        family = [household_id + k * COPY_RADIX for k in range(copies)]
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


def test_split_offsets_every_entity_id_by_k_times_the_declared_radix():
    frame = _scenario_frame(
        ages=[(45,), (50, 30, 8), (45,), (45,), (45,), (60, 58), (45,)],
        benunits=[(0,), (0, 1, 1), (0,), (0,), (0,), (0, 1), (0,)],
    )
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    # The copy-id table's field, not a power of ten above the frame's largest
    # id (microcosm#931).
    multiplier = COPY_RADIX
    assert multiplier == 10**13

    result = split_cgt_support_households(
        frame,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        clone_count=1,
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


def test_split_copies_take_their_roots_ids_plus_k_times_the_declared_radix():
    # Roots come from the input by lineage (each household's council tax is
    # its own here, and copies carry it unchanged), never from the copy's id
    # (microcosm#931).
    frame = _scenario_frame(
        ages=[(45,), (50, 30, 8), (45,), (45,), (45,), (60, 58), (45,)],
        benunits=[(0,), (0, 1, 1), (0,), (0,), (0,), (0, 1), (0,)],
    )
    result = split_cgt_support_households(
        frame,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        clone_count=1,
        headroom=SCENARIO_HEADROOM,
    )

    checked = assert_copy_ids_follow_lineage(
        frame,
        result.frame,
        field="cgt_support_split",
        key_columns=("council_tax",),
        ordinal_column=CGT_SUPPORT_COPY_INDEX_COLUMN,
    )
    assert checked == result.copies_created == 5


def _in_frs_id_spaces(frame):
    """``frame`` in FRS-style id spaces, its SPI-channel rows as SPI copies.

    Households keep their ids, benefit units take household x 100 + slot and
    people household x 1000 + member, so no two id columns share values; an
    SPI-channel household and its members and units move up 10**11, as the
    SPI copies sit after stage 16.
    """

    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    spi = household["household_support_channel"].eq("spi").to_numpy()
    shift = pd.Series(
        np.where(spi, radix("spi_support_channel"), 0),
        index=household["household_id"].to_numpy(),
    )
    sernum = person["person_household_id"]
    offset = sernum.map(shift)
    member = person.groupby("person_household_id").cumcount() + 1
    slot = (
        person.groupby("person_household_id")["person_benunit_id"]
        .rank(method="dense")
        .astype("int64")
    )
    units = dict(
        zip(person["person_benunit_id"], sernum * 100 + slot + offset, strict=True)
    )
    person["person_id"] = sernum * 1000 + member + offset
    person["person_benunit_id"] = person["person_benunit_id"].map(units)
    person["person_household_id"] = sernum + offset
    benunit["benunit_id"] = benunit["benunit_id"].map(units)
    household["household_id"] = household["household_id"] + shift.to_numpy()
    # Group tables stay sorted by id, so the SPI copies follow the base rows.
    order = np.argsort(household["household_id"].to_numpy(), kind="stable")
    return uk_national_frame(
        person=person,
        benunit=benunit.sort_values("benunit_id", ignore_index=True),
        household=household.iloc[order].reset_index(drop=True),
        household_weights=np.asarray(frame.weights_for("household").values)[order],
        weight_kind=frame.weights_for("household").kind,
        time_period="2024",
    )


def test_split_copies_of_spi_rows_take_their_parents_ids_plus_k_times_the_radix():
    # Separate id spaces, and a split household that is an SPI copy at
    # 10**11: a stage that minted one id column from another, or from the
    # SPI copy's root, would miss here (microcosm#931).
    frame = _in_frs_id_spaces(
        _scenario_frame(
            ages=[(45,), (50, 30, 8), (45,), (45,), (45,), (60, 58), (45,)],
            benunits=[(0,), (0, 1, 1), (0,), (0,), (0,), (0, 1), (0,)],
        )
    )
    result = split_cgt_support_households(
        frame,
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        clone_count=1,
        headroom=SCENARIO_HEADROOM,
    )

    checked = assert_copy_ids_follow_lineage(
        frame,
        result.frame,
        field="cgt_support_split",
        key_columns=("council_tax",),
        ordinal_column=CGT_SUPPORT_COPY_INDEX_COLUMN,
    )
    assert checked == result.copies_created == 5
    household = result.frame.table("household")
    copies = household.loc[household[HOUSEHOLD_IS_CGT_SUPPORT_COPY], "household_id"]
    # Household 2, on the SPI channel, is among the split: its copies sit at
    # 2 + 10**11 + k x 10**13.
    assert {2 + radix("spi_support_channel") + k * COPY_RADIX for k in (1, 2)} <= set(
        copies.tolist()
    )


def test_split_refuses_a_household_beyond_the_copy_id_field_with_a_fixed_message():
    # Id 2 would need 10,001 copies of at most 60, one more than the
    # cgt_support_split field's ordinals hold (microcosm#931). The refusal is
    # a fixed sentence: never the household, its weight or its copy count.
    weights = list(SCENARIO_WEIGHTS)
    weights[1] = CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * 10_000 + 1.0

    with pytest.raises(ValueError) as caught:
        _scenario_split(weights=weights)

    assert str(caught.value) == CGT_SUPPORT_COPY_BOUND_REFUSAL
    assert "10001" not in str(caught.value)
    assert "10,001" not in str(caught.value)


def test_a_household_of_exactly_the_fields_bound_in_rows_fits():
    # A weight of exactly 60 x 10,000 divides into 10,000 rows, root
    # included, so its largest copy ordinal is 9,999: the field holds it, and
    # one more row does not (microcosm#931).
    bound = ordinal_bound("cgt_support_split")
    copies = cgt_support_copy_counts(
        [CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * bound, 1.0],
        CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
        clone_count=1,
    )
    assert copies.tolist() == [bound, 1]
    _refuse_copies_beyond_the_copy_id_field(copies)
    _refuse_copies_beyond_the_copy_id_field(np.zeros(0, dtype="int64"))
    with pytest.raises(ValueError) as caught:
        _refuse_copies_beyond_the_copy_id_field(copies + np.asarray([1, 0]))
    assert str(caught.value) == CGT_SUPPORT_COPY_BOUND_REFUSAL


def test_split_refuses_incoming_ids_at_its_copy_id_radix_by_count():
    frame = with_last_household_moved(_scenario_frame(), radix("cgt_support_split"))

    with pytest.raises(
        ValueError,
        match=r"^1 of the incoming person_household_id values reach the "
        r"'cgt_support_split' copy-id field",
    ) as caught:
        split_cgt_support_households(
            frame,
            distribution=support_distribution(SCENARIO_TOP_CELLS),
            parameters=PARAMETERS,
            clone_count=1,
            headroom=SCENARIO_HEADROOM,
        )
    assert str(7 + radix("cgt_support_split")) not in str(caught.value)


def test_split_flags_only_copies_and_writes_the_family_copy_count():
    result = _scenario_split()
    household = result.frame.table("household").set_index("household_id")

    assert household[HOUSEHOLD_IS_CGT_SUPPORT_COPY].dtype == np.dtype(bool)
    assert household[CGT_SUPPORT_COPIES_COLUMN].dtype == np.dtype("int64")
    assert household.index[household[HOUSEHOLD_IS_CGT_SUPPORT_COPY]].tolist() == [
        2 + COPY_RADIX,
        3 + COPY_RADIX,
        6 + COPY_RADIX,
        2 + 2 * COPY_RADIX,
        3 + 2 * COPY_RADIX,
    ]
    assert household[CGT_SUPPORT_COPIES_COLUMN].to_dict() == {
        1: 1,
        2: 3,
        3: 3,
        4: 1,
        5: 1,
        6: 2,
        7: 1,
        2 + COPY_RADIX: 3,
        3 + COPY_RADIX: 3,
        6 + COPY_RADIX: 2,
        2 + 2 * COPY_RADIX: 3,
        3 + 2 * COPY_RADIX: 3,
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
        clone_count=1,
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
            offset = k * COPY_RADIX
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
        clone_count=1,
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


# --- where the exact-total correction lands (microcosm#931) ---------------------
#
# The split leaves whole every positive-weight household it does not divide:
# the unselected ones and the selected ones left at one copy. The correction
# is tried on the 64 heaviest of them, heaviest first, and spi_support's
# shared fallback, whose row can be a root or copy, runs only where none of
# them can carry it. The fallback does not make the one-copy check, so its
# row can also be a one-copy household the check passed over.


@dataclass(frozen=True)
class _CorrectionCall:
    """What the split handed the exact-total correction, and its selection.

    ``reselected`` holds each rerun of the walk the correction made, one per
    selected household left at one copy that could carry it at a weight that
    still counts one copy.
    """

    values: np.ndarray
    target: float
    whole: np.ndarray
    selected: np.ndarray
    reselected: tuple[np.ndarray, ...] = ()


def _recorded_correction(
    monkeypatch, frame, top_cells, *, clone_count: int = 1, headroom=None
):
    """The split on ``frame``, with what it handed the correction recorded."""

    calls: list[tuple[np.ndarray, float, np.ndarray]] = []
    selections: list[np.ndarray] = []
    correct = cgt_support._exact_support_weights
    select = cgt_support.select_cgt_support_households

    def recording_correct(values, *, target, whole, **keywords):
        calls.append((np.array(values), target, np.array(whole)))
        return correct(values, target=target, whole=whole, **keywords)

    def recording_select(*args, **kwargs):
        selected, rows = select(*args, **kwargs)
        selections.append(np.array(selected))
        return selected, rows

    keywords = {} if headroom is None else {"headroom": headroom}
    with monkeypatch.context() as patch:
        patch.setattr(cgt_support, "_exact_support_weights", recording_correct)
        patch.setattr(cgt_support, "select_cgt_support_households", recording_select)
        result = split_cgt_support_households(
            frame,
            distribution=support_distribution(top_cells),
            parameters=PARAMETERS,
            clone_count=clone_count,
            **keywords,
        )
    ((values, target, whole),) = calls
    # The first selection is the stage's; any later one reruns the walk for
    # a selected household left at one copy that could carry the correction.
    return result, _CorrectionCall(
        values, target, whole, selections[0], reselected=tuple(selections[1:])
    )


def _moved_rows(result, call: _CorrectionCall) -> list[int]:
    """The rows whose weight the correction moved, as positions."""

    weights = np.asarray(result.frame.weights_for("household").values)
    assert result.frame.weights_for("household").total == call.target
    return np.flatnonzero(weights != call.values).tolist()


def _correction_before(call: _CorrectionCall) -> tuple[np.ndarray, bool]:
    """The correction before microcosm#931's rule, and whether its one
    candidate carried it: the heaviest unselected household (the first of
    tied weights), else spi_support's shared fallback."""

    unselected = call.whole[~call.selected[call.whole]]
    if unselected.size:
        heaviest = int(unselected[np.argmax(call.values[unselected])])
        corrected = _exact_total_correction(
            call.values, target=call.target, correction_index=heaviest
        )
        if corrected is not None:
            return corrected, True
    fallback = _importance_weights_with_exact_total(call.values, call.target)
    return np.asarray(fallback.values), False


def _carries(call: _CorrectionCall, row: int) -> bool:
    return (
        _exact_total_correction(call.values, target=call.target, correction_index=row)
        is not None
    )


def test_split_falls_back_when_it_leaves_no_household_whole(monkeypatch):
    # Both households are divided, so no household is left whole to carry
    # the exact-total correction, and spi_support's shared fallback places
    # it: on the first root here, off w / n by a rounding amount.
    frame = support_frame(
        weights=[130.0, 121.0], wealth=[5.0, 6.0], incomes=[BAND_INCOMES[0]] * 2
    )
    result, call = _recorded_correction(
        monkeypatch,
        frame,
        {0: {250_000: 1_000.0}},
        headroom=SCENARIO_HEADROOM,
    )

    assert result.households_selected == 2
    assert result.copies_created == 4
    assert result.frame.weights_for("household").total == 251.0
    assert result.band_rows[0]["pool_exhausted"] is True
    assert call.whole.size == 0
    assert float(call.values.sum()) != call.target
    assert _moved_rows(result, call) == [0]


def test_a_selected_household_left_at_one_copy_can_carry_the_correction(
    monkeypatch,
):
    # All three are selected and the 40 household is left at one copy, the
    # one household the split leaves whole, so it carries the correction. The
    # rule before tried unselected households only, so the shared fallback
    # corrected the first root, off w / n.
    frame = support_frame(
        weights=[187.9, 91.8, 40.0],
        wealth=[5.0, 6.0, 7.0],
        incomes=[BAND_INCOMES[0]] * 3,
    )
    result, call = _recorded_correction(
        monkeypatch,
        frame,
        {0: {250_000: 1_000.0}},
        headroom=SCENARIO_HEADROOM,
    )

    household = result.frame.table("household")
    assert call.selected.tolist() == [True, True, True]
    assert household[CGT_SUPPORT_COPIES_COLUMN].tolist()[:3] == [4, 2, 1]
    assert call.whole.tolist() == [2]
    assert float(call.values.sum()) != call.target
    before, carried = _correction_before(call)
    assert not carried
    assert np.flatnonzero(before != call.values).tolist() == [0]
    assert _moved_rows(result, call) == [2]
    weights = _household_weights(result.frame)
    assert weights.loc[3] == pytest.approx(40.0, rel=1e-12)
    for household_id, weight, copies in ((1, 187.9, 4), (2, 91.8, 2)):
        family = [household_id + k * COPY_RADIX for k in range(copies)]
        assert weights.loc[family].tolist() == [weight / copies] * copies


def test_a_household_the_correction_would_divide_is_passed_over(monkeypatch):
    # All four are selected. The 60 household is left at one copy, the
    # heaviest the split leaves whole, but the correction it could carry
    # lifts it above 60, where the rule counts two copies, so the rule rerun
    # on the output (as the identity tool reruns it) would divide it. It is
    # passed over, and the 31.5 household, also at one copy, carries it.
    frame = support_frame(
        weights=[400.0, 268.7, 60.0, 31.5],
        wealth=[5.0, 6.0, 7.0, 8.0],
        incomes=[BAND_INCOMES[0]] * 4,
    )
    result, call = _recorded_correction(
        monkeypatch,
        frame,
        {0: {250_000: 1_000.0}},
        headroom=SCENARIO_HEADROOM,
    )

    assert call.selected.all()
    assert call.whole.tolist() == [2, 3]
    corrected = _exact_total_correction(
        call.values, target=call.target, correction_index=2
    )
    assert corrected is not None
    assert cgt_support_copy_counts([corrected[2]], clone_count=1).tolist() == [2]
    assert _moved_rows(result, call) == [3]
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights[2] == 60.0
    assert weights[3] == pytest.approx(31.5, rel=1e-12)
    assert cgt_support_copy_counts([weights[3]], clone_count=1).tolist() == [1]


def test_a_household_whose_correction_would_move_the_walk_is_passed_over(
    monkeypatch,
):
    # Both households are selected: the walk takes the first, an ulp below
    # its band's support mass of 40 (2 x 1.25 x 16), and crosses on the
    # second. The first is left at one copy, the one household the split
    # leaves whole, and the correction it could carry still counts one copy,
    # but it lifts the household onto 40, where the walk rerun on its
    # corrected weight stops at it and leaves the second unselected. So it is
    # passed over (microcosm#931), and the shared fallback corrects the
    # divided root instead.
    below = float(np.nextafter(40.0, -np.inf))
    frame = support_frame(
        weights=[below, 82.4], wealth=[9.0, 8.0], incomes=[BAND_INCOMES[0]] * 2
    )
    result, call = _recorded_correction(monkeypatch, frame, {0: {250_000: 16.0}})

    assert call.selected.tolist() == [True, True]
    assert call.whole.tolist() == [0]
    corrected = _exact_total_correction(
        call.values, target=call.target, correction_index=0
    )
    assert corrected is not None and corrected[0] >= 40.0
    assert cgt_support_copy_counts([corrected[0]], clone_count=1).tolist() == [1]
    (reselected,) = call.reselected
    assert reselected.tolist() == [True, False]
    assert _moved_rows(result, call) == [1]
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights[0] == below
    assert weights[1] == pytest.approx(41.2, rel=1e-12)


def test_the_fallback_does_not_make_the_one_copy_check(monkeypatch):
    # copy_rule names the one-copy check and that the shared fallback does
    # not make it (microcosm#931). In both frames the one household the split
    # leaves whole is selected at one copy and is the heaviest row, so where
    # the check passes it over, the fallback's first candidate is that same
    # household, and it gets the very weight the check refused. The weights
    # are the rule before's bit for bit, since no household is unselected,
    # and the rule rerun on the output divides a different set.
    #
    # The walk: the first household sits an ulp below its band's support
    # mass of 40 (2 x 1.25 x 16) and the second, divided into two copies of
    # 32.075, crosses. The corrected weight still counts one copy but reaches
    # 40, where the walk rerun stops at it, so the rerun on the output misses
    # the second's family.
    below = float(np.nextafter(40.0, -np.inf))
    frame = support_frame(
        weights=[below, 64.15], wealth=[9.0, 8.0], incomes=[BAND_INCOMES[0]] * 2
    )
    result, call = _recorded_correction(monkeypatch, frame, {0: {250_000: 16.0}})

    assert call.selected.tolist() == [True, True]
    assert call.whole.tolist() == [0]
    assert int(np.argmax(call.values)) == 0
    corrected = _exact_total_correction(
        call.values, target=call.target, correction_index=0
    )
    assert corrected is not None and corrected[0] >= 40.0
    assert cgt_support_copy_counts([corrected[0]], clone_count=1).tolist() == [1]
    (reselected,) = call.reselected
    assert reselected.tolist() == [True, False]
    assert _moved_rows(result, call) == [0]
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights[0] == corrected[0]
    assert weights[1:].tolist() == [64.15 / 2] * 2
    before, carried = _correction_before(call)
    assert not carried
    assert weights.tolist() == before.tolist()

    # The count: all three are selected, and the 60 household, left at one
    # copy, outweighs every copy of the other two. Its corrected weight lifts
    # it above 60, where the rule counts two copies, so the check passes it
    # over without a rerun of the walk, and the rerun on the output divides
    # it, a family the stage never made.
    frame = support_frame(
        weights=[400.0, 268.7, 60.0],
        wealth=[5.0, 6.0, 7.0],
        incomes=[BAND_INCOMES[0]] * 3,
    )
    result, call = _recorded_correction(
        monkeypatch,
        frame,
        {0: {250_000: 1_000.0}},
        headroom=SCENARIO_HEADROOM,
    )

    assert call.selected.all()
    assert call.whole.tolist() == [2]
    assert int(np.argmax(call.values)) == 2
    corrected = _exact_total_correction(
        call.values, target=call.target, correction_index=2
    )
    assert corrected is not None
    assert cgt_support_copy_counts([corrected[2]], clone_count=1).tolist() == [2]
    assert call.reselected == ()
    assert _moved_rows(result, call) == [2]
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights[2] == corrected[2]
    household = result.frame.table("household")
    assert household[CGT_SUPPORT_COPIES_COLUMN].tolist()[:3] == [7, 5, 1]
    before, carried = _correction_before(call)
    assert not carried
    assert weights.tolist() == before.tolist()


def test_a_one_copy_household_tied_with_the_heaviest_unselected_takes_it_first(
    monkeypatch,
):
    # Where the rule before's one candidate, the heaviest unselected
    # household, carries the correction, the weights are the ones it gave,
    # but for one case (microcosm#931): a selected household left at one
    # copy comes first where it outweighs that household or, as here, ties
    # it from an earlier row, which needs every unselected household to
    # weigh at most 60. The walk takes the 145.5 household (three copies) and
    # crosses on the first 37.87 one, left at one copy. The unselected 37.87
    # household could carry the correction, but the one-copy household
    # comes first and carries it, keeping the split, so the weights differ
    # from the rule before's on those two rows, and every divided row stays
    # bitwise w / 3.
    frame = support_frame(
        weights=[145.5, 37.87, 7.05, 26.65, 32.73, 37.87],
        wealth=[9e6, 8e6, 4e3, 3e3, 2e3, 1.0],
        incomes=[BAND_INCOMES[0]] * 6,
    )
    result, call = _recorded_correction(monkeypatch, frame, {0: {250_000: 65.77}})

    assert call.selected.tolist() == [True, True, False, False, False, False]
    assert call.whole.tolist() == [1, 2, 3, 4, 5]
    assert call.values[1] == call.values[5]
    before, carried = _correction_before(call)
    assert carried
    assert np.flatnonzero(before != call.values).tolist() == [5]
    assert _moved_rows(result, call) == [1]
    (reselected,) = call.reselected
    assert reselected.tolist() == call.selected.tolist()
    weights = np.asarray(result.frame.weights_for("household").values)
    assert np.flatnonzero(weights != before).tolist() == [1, 5]
    assert weights[1] == pytest.approx(37.87, rel=1e-12)
    household = result.frame.table("household")
    assert household[CGT_SUPPORT_COPIES_COLUMN].tolist() == [3, 1, 1, 1, 1, 1, 3, 3]
    assert weights[[0, 6, 7]].tolist() == [145.5 / 3] * 3


def test_tied_whole_households_take_the_correction_in_row_order(monkeypatch):
    # Tied weights are tried in row order (a stable sort), as np.argmax took
    # the first of them before (microcosm#931). At one clone the earlier of
    # two unselected households tied at the heaviest weight carries the
    # correction, though the later could too, and the weights are the rule
    # before's bit for bit. At three clones a source household's clones tie,
    # and clone 0, the first, carries it.
    top = {0: {250_000: 1_000.0}, 37_700: {250_000: 0.0}}
    frame = support_frame(
        weights=[274.0, 54.34, 42.59, 54.34],
        wealth=[9e6, 3e3, 2e3, 1e3],
        incomes=[BAND_INCOMES[0]] + [BAND_INCOMES[37_700]] * 3,
    )
    result, call = _recorded_correction(monkeypatch, frame, top)

    assert call.whole.tolist() == [1, 2, 3]
    assert not call.selected[call.whole].any()
    assert call.values[1] == call.values[3]
    assert _carries(call, 1) and _carries(call, 3)
    later = _exact_total_correction(call.values, target=call.target, correction_index=3)
    before, carried = _correction_before(call)
    assert carried and before.tobytes() != later.tobytes()
    assert _moved_rows(result, call) == [1]
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights.tobytes() == before.tobytes()

    cloned = clone_first_frame(
        support_frame(
            weights=[273.0, 82.77, 25.07],
            wealth=[9e6, 2e3, 1e3],
            incomes=[BAND_INCOMES[0]] + [BAND_INCOMES[37_700]] * 2,
        ),
        3,
    )
    result, call = _recorded_correction(monkeypatch, cloned, top, clone_count=3)

    household = result.frame.table("household")
    sources = household["source_household_id"].to_numpy()[: len(call.values)]
    clones = np.flatnonzero(sources == 2)
    assert clones.tolist() == [1, 4, 7]
    assert all(_carries(call, row) for row in clones.tolist())
    assert _moved_rows(result, call) == [1]
    assert household["household_clone_index"].iloc[1] == 0
    before, carried = _correction_before(call)
    weights = np.asarray(result.frame.weights_for("household").values)
    assert carried and weights.tobytes() == before.tobytes()

    # At fifteen clones every source household's clones tie fifteen ways.
    # Here forty of eighty unselected households tie, in a fixed golden-ratio
    # scramble, more than an unstable sort keeps in row order, and the
    # earliest tied row still carries the correction.
    keys = ((np.arange(80) + 1) * 0.6180339887) % 1.0
    position = np.argsort(np.argsort(keys, kind="stable"), kind="stable")
    frame = support_frame(
        weights=[274.0, *np.where(position < 40, 54.34, position - 38.5).tolist()],
        wealth=[9e6, *np.arange(1_000.0, 920.0, -1.0).tolist()],
        incomes=[BAND_INCOMES[0]] + [BAND_INCOMES[37_700]] * 80,
    )
    result, call = _recorded_correction(monkeypatch, frame, top)

    tied = call.whole[call.values[call.whole] == call.values[call.whole].max()]
    assert len(tied) == 40 and all(_carries(call, row) for row in tied.tolist())
    assert _moved_rows(result, call) == [int(tied.min())] == [2]
    before, carried = _correction_before(call)
    weights = np.asarray(result.frame.weights_for("household").values)
    assert carried and weights.tobytes() == before.tobytes()


def test_split_counts_suppressed_cells_and_reports_channels_when_present():
    result = split_cgt_support_households(
        support_frame(
            weights=[100.0, 100.0], wealth=[1.0, 2.0], incomes=[BAND_INCOMES[0]] * 2
        ),
        distribution=support_distribution(
            {0: {250_000: None, 500_000: None, 1_000_000: 5.0}}
        ),
        parameters=PARAMETERS,
        clone_count=1,
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
                clone_count=1,
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
            clone_count=1,
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
    # The geographic clone count is the one new receipt key (microcosm#931);
    # a frame without the clone index is one clone.
    assert evidence["parameters"] == {
        "clone_split_factor": 2,
        "headroom": 2.0,
        "maximum_copy_weight": 60.0,
        "clone_count": 1,
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


# --- the split on clone-first frames (microcosm#931) ----------------------------
#
# Nine one-person households. Band 0 holds exact multiples of 60 that the
# unsnapped rule overcounts at seven clones (900, 1,800, 4,020 and 7,620: 15,
# 30, 67 and 127 copies), a weight one ulp below 300 (5 copies), a light one
# (40, uncopied) and a non-multiple (1,234.5, 21 copies); the zero-weight
# household is outside every pool. The synthetic joint asks band 0 for far
# more than its pool, so the whole pool is selected at every clone count and
# the selection nests exactly; band 37,700 asks for nothing, so its household
# (9,000) is the first candidate for the exact-total correction (on clone 0,
# the first heaviest), and the light one, selected at one copy, is the other
# household the split leaves whole. At seven (and 13) clones none of their 14
# (26) clones can carry the correction: all sit in the half of NumPy's
# pairwise sum whose other half carries exactly half an ulp of the total, so
# no change to one of their weights rounds the total onto the target. The
# shared fallback then puts it on the last copy of the 7,620 family's last
# clone, the one exception copy_rule names. That copy is then off w / n by
# about an ulp of the frame's total weight (thousands of ulps of its own, a
# relative error below 1e-12), within the weights' nesting tolerance and the
# gate's; at one, three, 11 and 15 clones no correction is needed.

NESTING_WEIGHTS = [
    900.0,
    1_800.0,
    4_020.0,
    7_620.0,
    float(np.nextafter(300.0, 0.0)),
    40.0,
    1_234.5,
    9_000.0,
    0.0,
]
NESTING_WEALTH = [7e3, 6e3, 5e3, 4e3, 3e3, 2e3, 1e3, 1.0, 1e6]
NESTING_INCOMES = [BAND_INCOMES[0]] * 7 + [BAND_INCOMES[37_700], BAND_INCOMES[0]]
NESTING_TOP_CELLS = {0: {250_000: 1_000_000.0}}
NESTING_COPIES = [15, 30, 67, 127, 5, 1, 21, 1, 1]


def _clone_first_split(clone_count: int, *, weights=NESTING_WEIGHTS):
    """The transform on the nesting scenario as ``clone_count`` clones."""

    frame = clone_first_frame(
        support_frame(weights=weights, wealth=NESTING_WEALTH, incomes=NESTING_INCOMES),
        clone_count,
    )
    transform = UKCGTSupportSplitStageTransform(
        stage=support_stage(),
        distribution=support_distribution(NESTING_TOP_CELLS),
        parameters=PARAMETERS,
    )
    return transform(frame), transform.last_result


def _families(frame) -> pd.DataFrame:
    household = frame.table("household")
    return pd.DataFrame(
        {
            "source": household["source_household_id"].to_numpy(),
            "clone": household["household_clone_index"].to_numpy(),
            "copies": household[CGT_SUPPORT_COPIES_COLUMN].to_numpy(),
            "weight": np.asarray(frame.weights_for("household").values),
        }
    )


def test_split_at_one_indexed_clone_is_the_split_without_the_index():
    plain = _scenario_split()
    indexed = split_cgt_support_households(
        clone_first_frame(_scenario_frame(), 1),
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
        clone_count=1,
        headroom=SCENARIO_HEADROOM,
    )
    _, through_transform = _clone_first_split(1)

    assert indexed.evidence() == plain.evidence()
    assert indexed.frame.weights_for("household").values.tolist() == (
        plain.frame.weights_for("household").values.tolist()
    )
    assert through_transform.evidence()["parameters"]["clone_count"] == 1


@pytest.mark.parametrize("clone_count", [3, 7])
def test_split_gives_every_geographic_clone_its_sources_copies(clone_count: int):
    one_frame, one = _clone_first_split(1)
    many_frame, many = _clone_first_split(clone_count)
    reference = _families(one_frame)
    families = _families(many_frame)

    # The transform read K from the frame, and every count scales by K.
    assert many.evidence()["parameters"]["clone_count"] == clone_count
    assert one.copies_created == sum(n - 1 for n in NESTING_COPIES) == 259
    assert (
        many.households_selected
        == clone_count * one.households_selected
        == (clone_count * 7)
    )
    assert many.copies_created == clone_count * one.copies_created
    rows = families.groupby(["source", "clone"]).size()
    expected = dict(zip(range(1, 10), NESTING_COPIES, strict=True))
    for (source, clone), count in rows.items():
        # Each clone of a household is divided as its source is at K = 1.
        assert count == expected[source], (source, clone)
    assert sorted(set(families["clone"])) == list(range(clone_count))
    for source, n in expected.items():
        member = families["source"] == source
        assert (families.loc[member, "copies"] == n).all()
        # Nested weights: K x every clone's rows is the source's rows.
        source_weights = np.sort(
            reference.loc[reference["source"] == source, "weight"].to_numpy()
        )
        for clone in range(clone_count):
            clone_weights = np.sort(
                families.loc[member & (families["clone"] == clone), "weight"]
            )
            np.testing.assert_allclose(
                clone_count * clone_weights, source_weights, rtol=1e-12, atol=0.0
            )
    # Every copy, root included, weighs at most 60 / K.
    split = families["copies"] > 1
    assert (
        clone_count * families.loc[split, "weight"]
        <= CGT_SUPPORT_MAXIMUM_COPY_WEIGHT * (1.0 + 1e-12)
    ).all()
    assert many_frame.weights_for("household").total == pytest.approx(
        one_frame.weights_for("household").total, rel=1e-12
    )
    # The packaged stage gate passes the receipt the transform wrote.
    evidence = many.evidence()
    gate = uk_stage_health_gate(
        evidence=evidence,
        stage="cgt_support_split",
        check="cgt_support_split",
        parameters=_packaged_split_gate_parameters(),
    )
    assert gate.passed, gate.failures
    if clone_count == 7:
        # Without the snap, the four multiples of 60 would gain a copy here.
        replicas = np.asarray(NESTING_WEIGHTS[:4]) / 7
        unsnapped = np.ceil((7 * replicas) / CGT_SUPPORT_MAXIMUM_COPY_WEIGHT)
        assert unsnapped.tolist() == [n + 1 for n in NESTING_COPIES[:4]]
        # Seven times a rounded copy weight of 60 / 7 lands above 60, so the
        # gate passes only on its copy check's relative tolerance of 1e-12.
        assert (
            7 * max(row["heaviest_copy_weight"] for row in evidence["bands"])
            > CGT_SUPPORT_MAXIMUM_COPY_WEIGHT
        )


def test_the_walk_can_stop_partway_through_a_households_clones():
    # Band 0 holds households of 900, 1,800 and 600, wealthiest first, with a
    # support mass of 1,500 (2 x 1.25 x 600 published). At one clone the walk
    # takes the first two, at 15 and 30 copies. At three it accumulates each
    # clone's own weight: the first household's three clones (900 together)
    # and then the second's clone 0 (600) reach 1,500, so the second's clones
    # 1 and 2 stay whole. Selection nests only approximately (microcosm#931);
    # every clone the walk selects gets its source's count.
    def counts_by_clone(clone_count: int):
        frame = clone_first_frame(
            support_frame(
                weights=[900.0, 1_800.0, 600.0],
                wealth=[3e3, 2e3, 1e3],
                incomes=[BAND_INCOMES[0]] * 3,
            ),
            clone_count,
        )
        transform = UKCGTSupportSplitStageTransform(
            stage=support_stage(),
            distribution=support_distribution({0: {250_000: 600.0}}),
            parameters=PARAMETERS,
        )
        families = _families(transform(frame))
        counts = families.groupby(["source", "clone"])["copies"].first()
        selected = transform.last_result.households_selected
        return {key: int(count) for key, count in counts.items()}, selected

    one, one_selected = counts_by_clone(1)
    three, three_selected = counts_by_clone(3)

    assert one == {(1, 0): 15, (2, 0): 30, (3, 0): 1}
    assert one_selected == 2
    assert three == {
        **{(1, clone): 15 for clone in range(3)},
        (2, 0): 30,
        (2, 1): 1,
        (2, 2): 1,
        **{(3, clone): 1 for clone in range(3)},
    }
    assert three_selected == 4


def test_the_transform_refuses_a_gap_in_the_clone_indices_by_count():
    frame = clone_first_frame(_scenario_frame(), 2)
    household = frame.table("household").copy()
    household.loc[household["household_clone_index"] == 1, "household_clone_index"] = 2
    gapped = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        household_weights=frame.weights_for("household").values,
        time_period="2024",
    )
    transform = UKCGTSupportSplitStageTransform(
        stage=support_stage(),
        distribution=support_distribution(SCENARIO_TOP_CELLS),
        parameters=PARAMETERS,
    )

    with pytest.raises(ValueError) as caught:
        transform(gapped)

    # A count and the rule, never a household.
    assert str(caught.value) == (
        "1 of the 3 indices from 0 to the largest are carried by no household; "
        "the CGT support split counts K geographic clones as the "
        "'household_clone_index' values 0 to K - 1 (microcosm#931)."
    )
    assert transform.last_result is None


def _packaged_split_gate_parameters() -> dict[str, object]:
    from importlib.resources import files

    gates = json.loads(
        files("microcosm.build.uk").joinpath("gates.json").read_text(encoding="utf-8")
    )["gates"]
    (gate,) = [g for g in gates if g["id"] == "uk_stage_cgt_support_split_mass"]
    return dict(gate["parameters"])


def test_the_stage_gate_reads_the_receipts_clone_count():
    # Three clones of households at 150, 120 and 100: every clone weighs at
    # most 60 (50, 40 and 33.3), but three times its weight does not, so the
    # copies are made. The packaged gate passes the receipt and would fail it
    # if it ignored the clone count.
    weights = [150.0, 120.0, 100.0, 0.0, 0.0, 0.0, 0.0, 90.0, 0.0]
    frame, result = _clone_first_split(3, weights=weights)
    evidence = result.evidence()
    parameters = _packaged_split_gate_parameters()
    assert evidence["parameters"]["clone_count"] == 3
    assert max(row["heaviest_selected_weight"] for row in evidence["bands"]) == 50.0
    assert evidence["totals"]["copies_created"] == 3 * (2 + 1 + 1)

    passed = uk_stage_health_gate(
        evidence=evidence,
        stage="cgt_support_split",
        check="cgt_support_split",
        parameters=parameters,
    )
    ignored = uk_stage_health_gate(
        evidence={
            **evidence,
            "parameters": {**evidence["parameters"], "clone_count": 1},
        },
        stage="cgt_support_split",
        check="cgt_support_split",
        parameters=parameters,
    )

    assert passed.passed, passed.failures
    assert ignored.passed is False
    assert ignored.failures == (
        "cgt_support_split: column from 0.0 created 12 copies although no selected "
        "household exceeds the maximum weight.",
    )
    assert frame.weights_for("household").total == pytest.approx(460.0, rel=1e-12)


# --- whole households past the first candidate (microcosm#931) ------------------
#
# The cap frames put the households the split leaves whole where NumPy's
# pairwise sum decides whether they can carry the correction. Heavy households
# sit in band 37,700, which asks for no support, so the split leaves them
# whole, and so is the light one after them; the last household, in the
# exhausting band 0, is divided into 67 rows (4,004.74 / 60 = 66.7). The
# 132-row vector the split corrects is summed as two halves split at row 64:
# every heavy household sits in the first, the light one, the divided root
# and its copies in the second. The first half sums inside the total's binade
# and the second carries exactly half an ulp of the total, so round half to
# even keeps every change to a heavy household's weight off the target, while
# the light household and the copies can carry it. Each test asserts that
# premise before the rule.

_CAP_HEAVY = [1_100.0 + index for index in range(64)]
_CAP_LIGHT = 1_000.0
_CAP_DIVIDED = 4_004.74
_CAP_COPIES = 67


def _cap_frame(heavy):
    """``heavy`` whole households, the light one, then the divided one."""

    weights = [*heavy, _CAP_LIGHT, _CAP_DIVIDED]
    return support_frame(
        weights=weights,
        wealth=[1.0] * (len(weights) - 1) + [1e6],
        incomes=[BAND_INCOMES[37_700]] * (len(weights) - 1) + [BAND_INCOMES[0]],
    )


def test_where_the_first_candidate_carries_it_the_weights_are_as_before(
    monkeypatch,
):
    # The first candidate is the heaviest unselected household, the one the
    # rule before corrected, wherever no selected household left at one copy
    # outweighs it (such a household weighs at most 60 / K). So wherever it
    # can carry the correction, the weights are the ones the rule before
    # gave, bit for bit: on the reference scenario (id 5) and on random
    # one-clone frames that ask each band for about 30% of its mass.
    rng = np.random.default_rng(7)
    frames = [(_scenario_frame(), SCENARIO_TOP_CELLS, SCENARIO_HEADROOM)]
    for _ in range(12):
        weights = rng.lognormal(4.5, 1.0, 120)
        wealth = rng.lognormal(11.0, 1.5, 120)
        bands = rng.choice([0, 37_700], size=120)
        top = {
            int(band): {
                250_000: float(np.round(0.3 * weights[bands == band].sum() / 2.5, 1))
            }
            for band in (0, 37_700)
        }
        frame = support_frame(
            weights=weights.tolist(),
            wealth=wealth.tolist(),
            incomes=[BAND_INCOMES[int(band)] for band in bands],
        )
        frames.append((frame, top, None))
    carried = 0
    for frame, top, headroom in frames:
        result, call = _recorded_correction(monkeypatch, frame, top, headroom=headroom)
        before, first_carried = _correction_before(call)
        weights = np.asarray(result.frame.weights_for("household").values)
        assert result.frame.weights_for("household").total == call.target
        if not first_carried or float(call.values.sum()) == call.target:
            continue
        carried += 1
        unselected = call.whole[~call.selected[call.whole]]
        first = call.whole[np.lexsort((call.whole, -call.values[call.whole]))][0]
        assert first == unselected[np.argmax(call.values[unselected])]
        assert weights.tobytes() == before.tobytes()
    assert carried >= 3


def test_the_correction_passes_over_whole_households_that_cannot_carry_it(
    monkeypatch,
):
    # 63 heavy households and a zero-weight one fill the first half, so the
    # light household is the 64th and last whole one the rule tries. The rule
    # before tried the heaviest unselected household only, and the shared
    # fallback then corrected the last copy; now the light one carries it and
    # every row of the divided household stays bitwise w / n.
    result, call = _recorded_correction(
        monkeypatch, _cap_frame([*_CAP_HEAVY[1:], 0.0]), NESTING_TOP_CELLS
    )

    assert call.whole.tolist() == [*range(63), 64]
    assert not call.selected[call.whole].any()
    assert float(call.values.sum()) != call.target
    assert not any(_carries(call, row) for row in range(63))
    assert _carries(call, 64)
    before, carried = _correction_before(call)
    assert not carried
    assert np.flatnonzero(before != call.values).tolist() == [len(call.values) - 1]

    assert _moved_rows(result, call) == [64]
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights[64] == pytest.approx(_CAP_LIGHT, rel=1e-12)
    assert weights[65:].tolist() == [_CAP_DIVIDED / _CAP_COPIES] * _CAP_COPIES


def test_the_correction_tries_only_the_64_heaviest_whole_households(monkeypatch):
    # With all 64 heavy households before it, the light one is the 65th
    # whole household, past the cap: none of the 64 can carry the
    # correction, so the shared fallback places it, on the last copy, though
    # the 65th could have. One candidate more and the 65th carries it.
    assert CGT_SUPPORT_CORRECTION_CANDIDATES == 64
    result, call = _recorded_correction(
        monkeypatch, _cap_frame(_CAP_HEAVY), NESTING_TOP_CELLS
    )

    assert call.whole.tolist() == list(range(65))
    assert float(call.values.sum()) != call.target
    assert not any(_carries(call, row) for row in range(64))
    assert _carries(call, 64)
    assert _moved_rows(result, call) == [len(call.values) - 1]
    fallback = _importance_weights_with_exact_total(call.values, call.target)
    weights = np.asarray(result.frame.weights_for("household").values)
    assert weights.tobytes() == np.asarray(fallback.values).tobytes()
    assert weights[-1] == pytest.approx(_CAP_DIVIDED / _CAP_COPIES, rel=1e-12)

    monkeypatch.setattr(cgt_support, "CGT_SUPPORT_CORRECTION_CANDIDATES", 65)
    wider, wider_call = _recorded_correction(
        monkeypatch, _cap_frame(_CAP_HEAVY), NESTING_TOP_CELLS
    )
    assert _moved_rows(wider, wider_call) == [64]
    wider_weights = np.asarray(wider.frame.weights_for("household").values)
    assert wider_weights[65:].tolist() == [_CAP_DIVIDED / _CAP_COPIES] * _CAP_COPIES


@pytest.mark.parametrize("clone_count", [1, 3, 7, 11, 13, 15])
def test_the_nesting_scenario_corrects_a_copy_only_where_no_whole_household_can(
    monkeypatch, clone_count: int
):
    # The split leaves two households of each clone whole: the 9,000 one and
    # the light 40 one, selected at one copy. At 7 and 13 clones none of them
    # can carry the correction, so the shared fallback still puts it on the
    # last row, the last copy of the 7,620 family's last clone, and the total
    # stays bitwise on target. At 1, 3, 11 and 15 none is needed.
    frame = clone_first_frame(
        support_frame(
            weights=NESTING_WEIGHTS, wealth=NESTING_WEALTH, incomes=NESTING_INCOMES
        ),
        clone_count,
    )
    result, call = _recorded_correction(
        monkeypatch, frame, NESTING_TOP_CELLS, clone_count=clone_count
    )

    assert call.whole.size == 2 * clone_count
    moved = _moved_rows(result, call)
    if clone_count not in (7, 13):
        assert float(call.values.sum()) == call.target
        assert moved == []
        return
    assert not any(_carries(call, row) for row in call.whole.tolist())
    assert moved == [len(call.values) - 1]
    last = result.frame.table("household").iloc[-1]
    assert last["source_household_id"] == 4
    assert last["household_clone_index"] == clone_count - 1
    assert last[CGT_SUPPORT_COPY_INDEX_COLUMN] == 126


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
    assert payload["copy_rule"] == (
        "copies = ceil(geographic clone count x household weight / "
        "maximum_copy_weight), the quotient snapped to the nearest integer "
        "within clone-count ulps when the clone count exceeds one; every copy "
        "carries weight / copies, except where none of the 64 heaviest "
        "households left whole can carry the exact-total correction, when the "
        "shared fallback can put it on a root or copy, off weight / copies by "
        "about one ulp of the total weight; of those 64, a selected household "
        "left at one copy can carry the correction only where its corrected "
        "weight keeps its one copy and the walk's selection, a check the "
        "fallback does not make"
    )
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


def test_the_packaged_declarations_restate_the_copy_rule() -> None:
    # The copy rule names its one exception, and then the one-copy check and
    # that the fallback does not make it, identically in the code
    # dictionary, sources.yaml and source_stages.json, both clauses built on
    # the cap (microcosm#931).
    ((_, payload),) = cgt_support_split_operation_parameters()
    stage = load_country_spec("uk").sources.stage_map()[CGT_SUPPORT_SPLIT_STAGE_NAME]
    (operation,) = stage.operations
    assert operation.parameters["copy_rule"] == payload["copy_rule"]
    mirror = Path(cgt_support.__file__).parents[1] / "uk" / "source_stages.json"
    packaged = json.loads(mirror.read_text())
    (entry,) = [row for row in packaged["stages"] if row["stage"] == stage.stage]
    assert entry["operations"][0]["copy_rule"] == payload["copy_rule"]
    assert payload["copy_rule"].endswith(
        "; every copy carries weight / copies, "
        + CGT_SUPPORT_EXACT_TOTAL_EXCEPTION
        + "; "
        + CGT_SUPPORT_ONE_COPY_CHECK
    )
    assert CGT_SUPPORT_ONE_COPY_CHECK.startswith(
        f"of those {CGT_SUPPORT_CORRECTION_CANDIDATES}, "
    )
