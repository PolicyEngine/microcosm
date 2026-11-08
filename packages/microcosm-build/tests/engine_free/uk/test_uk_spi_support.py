from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import binomial_share_tolerance
from microcosm.build.uk_runtime import (
    BASE_FRS_SUPPORT_CHANNEL,
    DEFAULT_SPI_PRIOR_MASS_SHARE,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_INCOME_COMPONENT_COLUMNS,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_PRIOR_MASS_CHANGE_REASON,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    build_uk_spi_support_channel,
    create_uk_spi_support_tables,
    fill_support_channel_from_source,
    replace_uk_spi_support_tables,
    spi_support,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.uk_runtime.terminal_gates import (
    UKZeroWeightStratumDeclaration,
    uk_zero_weight_strata_gate,
)
from microcosm.frame import MassChangeRecord, WeightKind
from test_support.microcosm_build.uk_clone_first_frames import (
    assert_copy_ids_follow_lineage,
)


def household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [10.0, 20.0, 30.0],
            "region": ["LONDON", "WALES", "SCOTLAND"],
        }
    )


def person_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1001, 2001, 2002, 3001],
            "person_household_id": [1, 2, 2, 3],
            "person_benunit_id": [101, 201, 201, 301],
            "employment_income": [1.0, 2.0, 3.0, 4.0],
        }
    )


def benunit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benunit_id": [101, 201, 301],
            "would_claim_uc": [True, False, False],
        }
    )


def test_spi_support_creates_zero_weight_copy_with_lineage_and_remapped_ids() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=None,
        source_year=2023,
    )

    # The SPI copies take the copy-id table's spi_support_channel radix, not
    # a power of ten above the frame's largest id (microcosm#931).
    assert result.id_multiplier == 10**11
    assert result.n_spi_households == 3
    assert result.household["household_id"].tolist() == [
        1,
        2,
        3,
        10**11 + 1,
        10**11 + 2,
        10**11 + 3,
    ]
    assert result.household["household_weight"].tolist() == [
        10.0,
        20.0,
        30.0,
        0.0,
        0.0,
        0.0,
    ]
    assert result.household["household_weight"].sum() == pytest.approx(60.0)
    assert result.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert result.household[support_channel_column("household")].tolist() == [
        BASE_FRS_SUPPORT_CHANNEL,
        BASE_FRS_SUPPORT_CHANNEL,
        BASE_FRS_SUPPORT_CHANNEL,
        SPI_SYNTHETIC_SUPPORT_CHANNEL,
        SPI_SYNTHETIC_SUPPORT_CHANNEL,
        SPI_SYNTHETIC_SUPPORT_CHANNEL,
    ]
    assert result.household[support_clone_index_column("household")].tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
    ]
    assert result.household[support_source_id_column("household")].tolist() == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]
    assert result.household["source_household_id"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result.household["source_household_key"].tolist() == [
        "2023:1",
        "2023:2",
        "2023:3",
        "2023:1",
        "2023:2",
        "2023:3",
    ]

    assert result.person["person_id"].tolist() == [
        1001,
        2001,
        2002,
        3001,
        10**11 + 1001,
        10**11 + 2001,
        10**11 + 2002,
        10**11 + 3001,
    ]
    assert result.person["person_household_id"].tolist() == [
        1,
        2,
        2,
        3,
        10**11 + 1,
        10**11 + 2,
        10**11 + 2,
        10**11 + 3,
    ]
    assert result.person["person_benunit_id"].tolist() == [
        101,
        201,
        201,
        301,
        10**11 + 101,
        10**11 + 201,
        10**11 + 201,
        10**11 + 301,
    ]
    assert result.benunit["benunit_id"].tolist() == [
        101,
        201,
        301,
        10**11 + 101,
        10**11 + 201,
        10**11 + 301,
    ]


def test_spi_copies_take_their_roots_ids_plus_the_declared_radix() -> None:
    # Roots come from the input by lineage (region names one household here),
    # never from the copy's id (microcosm#931).
    before = {
        "person": person_frame(),
        "benunit": benunit_frame(),
        "household": household_frame(),
    }
    result = create_uk_spi_support_tables(
        **before, expected_spi_households=None, source_year=2023
    )
    after = {
        "person": result.person,
        "benunit": result.benunit,
        "household": result.household,
    }

    assert (
        assert_copy_ids_follow_lineage(
            before,
            after,
            field="spi_support_channel",
            key_columns=("region",),
            ordinal_column=support_clone_index_column("household"),
        )
        == 3
    )


def test_spi_support_refuses_incoming_ids_that_reach_its_copy_id_field() -> None:
    household = household_frame()
    household.loc[2, "household_id"] = 10**11 + 3
    person = person_frame()
    person.loc[3, "person_household_id"] = 10**11 + 3

    with pytest.raises(
        ValueError, match=r"^1 of the incoming household_id values reach"
    ) as caught:
        create_uk_spi_support_tables(
            person=person,
            benunit=benunit_frame(),
            household=household,
            expected_spi_households=None,
            source_year=2023,
        )
    # A count, never the id.
    assert str(10**11 + 3) not in str(caught.value)


def test_spi_support_can_subsample_without_rescaling_or_reordering_base() -> None:
    # An expected count of 2 over 3 source households draws each at 2/3;
    # seed 0 keys households 1 and 2 in and 3 out (microcosm#931).
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=2,
        seed=0,
    )

    assert result.household.head(3)["household_id"].tolist() == [1, 2, 3]
    assert result.household.head(3)["household_weight"].tolist() == [10.0, 20.0, 30.0]
    spi_households = result.household[
        result.household[support_channel_column("household")]
        == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ]
    assert result.spi_household_ids == (1, 2)
    assert spi_households["household_weight"].eq(0).all()
    assert tuple(spi_households[support_source_id_column("household")]) == (
        result.spi_household_ids
    )
    assert result.expected_spi_households == 2.0
    assert result.spi_source_households == 3
    assert result.spi_inclusion_probability == 2 / 3


def test_spi_fill_only_updates_spi_channel_and_can_initialize_new_columns() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=None,
    )
    donor = pd.DataFrame(
        {
            "person_id": [1001, 2001, 2002, 3001],
            "employment_income": [900.0, 1_000.0, 1_100.0, 1_200.0],
            "gift_aid": [9.0, 10.0, 11.0, 12.0],
        }
    )

    filled = fill_support_channel_from_source(
        result.person,
        donor,
        entity="person",
        columns=["employment_income", "gift_aid"],
    )

    base = filled[filled[support_channel_column("person")] == BASE_FRS_SUPPORT_CHANNEL]
    spi = filled[
        filled[support_channel_column("person")] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ]
    assert base["employment_income"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert base["gift_aid"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert spi["employment_income"].tolist() == [900.0, 1_000.0, 1_100.0, 1_200.0]
    assert spi["gift_aid"].tolist() == [9.0, 10.0, 11.0, 12.0]


def test_spi_variable_surfaces_include_efrs_stage1_and_stage2_fixes() -> None:
    assert SPI_INCOME_COMPONENT_COLUMNS == (
        "employment_income",
        "self_employment_income",
        "savings_interest_income",
        "dividend_income",
        "private_pension_income",
        "property_income",
        "other_investment_income",
    )
    assert "gift_aid" in SPI_INCOME_IMPUTATION_COLUMNS
    assert "charitable_investment_gifts" in SPI_INCOME_IMPUTATION_COLUMNS
    assert FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS == (
        "age",
        "gender",
        "region",
        "employment_income",
        "self_employment_income",
        "savings_interest_income",
        "dividend_income",
        "private_pension_income",
        "property_income",
    )
    assert "other_investment_income" not in FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS
    assert "universal_credit_reported" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    assert "housing_benefit_reported" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    assert "employee_pension_contributions" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    assert (
        "individual_savings_account_interest_income" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    )
    assert "tax_free_savings_income" not in FRS_ONLY_SPI_FILL_PERSON_COLUMNS


def _certified_like_dead_spi_support():
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 9, dtype="int64"),
            "household_weight": np.arange(10.0, 90.0, 10.0),
            "region": ["LONDON", "LONDON", "WALES", "WALES"] * 2,
            "clone_index": [0] * 4 + [1] * 4,
            "household_is_capital_gains_clone": [False, False, True, True] * 2,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 109, dtype="int64"),
            "person_household_id": np.arange(1, 9, dtype="int64"),
            "person_benunit_id": np.arange(201, 209, dtype="int64"),
            "employment_income": np.arange(1_000.0, 9_000.0, 1_000.0),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(201, 209, dtype="int64")})
    return create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        selected_household_ids=(1, 3, 5, 7),
        source_year=2023,
    )


def test_replace_spi_support_preserves_quotas_and_allocates_real_mass() -> None:
    dead = _certified_like_dead_spi_support()
    dead_spi = dead.household[dead.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]]
    original_total = float(dead.household["household_weight"].sum())

    result = replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=dead.household,
        seed=7,
        source_year=2023,
    )

    assert result.replaced_spi_households == len(dead_spi) == 4
    assert len(result.household) == len(dead.household)
    assert result.spi_prior_mass_share == DEFAULT_SPI_PRIOR_MASS_SHARE == 0.5
    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert result.household["household_weight"].sum() == original_total

    channel = result.household[support_channel_column("household")]
    base = result.household[channel == BASE_FRS_SUPPORT_CHANNEL]
    spi = result.household[channel == SPI_SYNTHETIC_SUPPORT_CHANNEL]
    assert len(spi) == 4
    assert spi["household_weight"].gt(0).all()
    assert base["household_weight"].sum() == pytest.approx(original_total * 0.5)
    assert spi["household_weight"].sum() == pytest.approx(original_total * 0.5)
    assert set(channel) == {BASE_FRS_SUPPORT_CHANNEL, SPI_SYNTHETIC_SUPPORT_CHANNEL}

    strata = ["clone_index", "household_is_capital_gains_clone", "region"]
    pd.testing.assert_series_equal(
        dead_spi.groupby(strata, dropna=False).size(),
        spi.groupby(strata, dropna=False).size(),
    )
    incoming_base = dead.household[~dead.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]]
    pd.testing.assert_series_equal(
        result.household.groupby(strata, dropna=False)["household_weight"].sum(),
        incoming_base.groupby(strata, dropna=False)["household_weight"].sum(),
    )
    assert spi.groupby(strata, dropna=False)["household_weight"].nunique().eq(1).all()
    assert len(result.mass_log) == 1
    record = result.mass_log[-1]
    assert record.entity == "household"
    assert record.old_total == original_total
    assert record.new_total == original_total
    assert record.declared_factor == 1.0
    assert record.reason == SPI_PRIOR_MASS_CHANGE_REASON


def test_build_spi_support_channel_fresh_stack_allocates_region_mass() -> None:
    prior_record = MassChangeRecord(
        entity="household",
        old_total=30.0,
        new_total=60.0,
        declared_factor=2.0,
        reason="prior raw FRS calibration",
    )

    # An expected count of 3 over 3 source households copies every one.
    result = build_uk_spi_support_channel(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=3,
        seed=42,
        source_year=2023,
        strata_columns=("region",),
        input_weight_kind=WeightKind.DESIGN,
        mass_log=(prior_record,),
        zero_weight_declarations=(
            UKZeroWeightStratumDeclaration(
                name="e7_spi_synthetic_share",
                selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                maximum_zero_weight_rows=3,
                reason="synthetic test declaration",
            ),
        ),
    )

    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert result.mass_log[:-1] == (prior_record,)
    assert result.mass_log[-1].old_total == 60.0
    assert result.mass_log[-1].new_total == 60.0
    channel = support_channel_column("household")
    base = result.household[result.household[channel] == BASE_FRS_SUPPORT_CHANNEL]
    spi = result.household[result.household[channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL]
    assert base["household_weight"].sum() == pytest.approx(30.0)
    assert spi["household_weight"].sum() == pytest.approx(30.0)
    pd.testing.assert_series_equal(
        result.household.groupby("region")["household_weight"].sum(),
        household_frame().groupby("region")["household_weight"].sum(),
    )


def test_build_spi_support_channel_fails_on_undeclared_zero_weight_row() -> None:
    with pytest.raises(ValueError, match="match no declared stratum"):
        build_uk_spi_support_channel(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household_frame(),
            expected_spi_households=3,
            zero_weight_declarations=(
                UKZeroWeightStratumDeclaration(
                    name="nonmatching",
                    selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: False},
                    maximum_zero_weight_rows=0,
                    reason="synthetic test declaration",
                ),
            ),
        )


def test_build_spi_support_channel_fails_missing_positive_region_support() -> None:
    # Each household is its own region here, and seed 0 draws household 3 out
    # of the share at 2/3, so SCOTLAND keeps base mass and no copy. The
    # refusal counts strata and names none (microcosm#931).
    with pytest.raises(ValueError, match="positive-mass UK base stratum") as caught:
        build_uk_spi_support_channel(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household_frame(),
            expected_spi_households=2,
            seed=0,
            strata_columns=("region",),
            zero_weight_declarations=(
                UKZeroWeightStratumDeclaration(
                    name="e7_spi_synthetic_share",
                    selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                    maximum_zero_weight_rows=2,
                    reason="synthetic test declaration",
                ),
            ),
        )
    assert str(caught.value) == (
        "Every positive-mass UK base stratum must have rebuilt SPI support; 1 of "
        "the 3 positive-mass strata have no SPI copy."
    )
    assert "SCOTLAND" not in str(caught.value)


def test_replace_spi_support_builds_importance_pool_from_calibrated_base() -> None:
    dead = _certified_like_dead_spi_support()
    original_total = float(dead.household["household_weight"].sum())
    prior_record = MassChangeRecord(
        entity="household",
        old_total=original_total / 2.0,
        new_total=original_total,
        declared_factor=2.0,
        reason="prior certified-base calibration",
    )

    result = replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=dead.household,
        seed=7,
        source_year=2023,
        input_weight_kind=WeightKind.CALIBRATED,
        mass_log=(prior_record,),
    )

    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert result.household["household_weight"].sum() == original_total
    assert result.mass_log[:-1] == (prior_record,)
    assert result.mass_log[-1].reason == SPI_PRIOR_MASS_CHANGE_REASON
    assert result.mass_log[-1].old_total == original_total
    assert result.mass_log[-1].new_total == original_total


def _repeated_outlier_lineage_support():
    """Two extreme donor lineages repeated across ten real-shaped clone strata."""

    household_rows: list[dict[str, object]] = []
    dead_source_ids: list[int] = []
    person_rows: list[dict[str, int]] = []
    benunit_rows: list[dict[str, int]] = []
    next_household_id = 1
    for clone_index in range(10):
        for region, lineage, outlier_weight in (
            ("LONDON", 11_852, 1_000.0),
            ("SCOTLAND", 14_876, 800.0),
        ):
            for position in range(100):
                household_id = next_household_id
                next_household_id += 1
                source_lineage = lineage if position == 0 else household_id
                household_rows.append(
                    {
                        "household_id": household_id,
                        "household_weight": outlier_weight if position == 0 else 1.0,
                        "region": region,
                        "clone_index": clone_index,
                        "household_is_capital_gains_clone": False,
                        "source_household_id": source_lineage,
                    }
                )
                if position < 60:
                    dead_source_ids.append(household_id)
                person_rows.append(
                    {
                        "person_id": household_id,
                        "person_household_id": household_id,
                        "person_benunit_id": household_id,
                    }
                )
                benunit_rows.append({"benunit_id": household_id})
    return create_uk_spi_support_tables(
        person=pd.DataFrame(person_rows),
        benunit=pd.DataFrame(benunit_rows),
        household=pd.DataFrame(household_rows),
        selected_household_ids=dead_source_ids,
        source_year=2023,
    )


def test_spi_prior_spreads_stratum_mass_instead_of_propagating_outliers() -> None:
    """#630: selected descendants must not each inherit an extreme donor weight."""

    dead = _repeated_outlier_lineage_support()
    result = replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=dead.household,
        seed=42,
        source_year=2023,
    )
    channel = support_channel_column("household")
    source_id = support_source_id_column("household")
    spi = result.household[result.household[channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL]
    incoming_base = dead.household[~dead.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]]
    incoming_weights = incoming_base.set_index("household_id")["household_weight"]
    selected_source_weights = spi[source_id].map(incoming_weights).to_numpy(dtype=float)
    old_total = float(incoming_weights.sum())

    # Reconstruct the removed behavior: copy selected calibrated weights and
    # normalize them once across the whole SPI channel. Seed 42 selects twelve
    # descendants of the two extreme lineages, matching the real defect's
    # 11-12-row shape, and the legacy prior breaches the certified June fence.
    legacy = np.concatenate(
        (
            incoming_weights.to_numpy(dtype=float) * 0.5,
            selected_source_weights
            * (old_total * 0.5 / float(selected_source_weights.sum())),
        )
    )
    legacy_median = float(np.median(legacy[legacy > 0.0]))
    old_fence = 1151.2542195939373
    assert int((legacy > old_fence * legacy_median).sum()) == 12
    assert float(legacy.max() / legacy_median) > old_fence

    fixed = result.household["household_weight"].to_numpy(dtype=float)
    fixed_median = float(np.median(fixed[fixed > 0.0]))
    assert float(fixed.max() / fixed_median) < old_fence
    strata = ["clone_index", "household_is_capital_gains_clone", "region"]
    assert spi.groupby(strata, dropna=False)["household_weight"].nunique().eq(1).all()
    pd.testing.assert_series_equal(
        result.household.groupby(strata, dropna=False)["household_weight"].sum(),
        incoming_base.groupby(strata, dropna=False)["household_weight"].sum(),
    )


def test_replace_spi_support_corrects_rounding_residue_to_exact_mass() -> None:
    dead = _certified_like_dead_spi_support()
    household = dead.household.copy()
    synthetic = household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]
    household.loc[~synthetic, "household_weight"] = np.asarray(
        [
            float.fromhex("0x1.1b6fc8a1fb5d7p+16"),
            float.fromhex("0x1.5fd6d9b527953p+13"),
            float.fromhex("0x1.d3e48ce1ae317p+15"),
            float.fromhex("0x1.8a6db6475255fp+14"),
            1.0,
            3.0,
            7.0,
            11.0,
        ]
    )
    original_total = float(household["household_weight"].sum())

    result = replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=household,
        seed=7,
        source_year=2023,
    )

    assert result.household["household_weight"].sum() == original_total
    assert result.mass_log[-1].new_total == original_total


def test_exact_mass_correction_handles_one_ulp_bounce() -> None:
    weights = spi_support._importance_weights_with_exact_total(
        np.asarray([0.05, 0.1, 0.15, 0.3]),
        0.6,
    )

    assert weights.total == 0.6


def test_replace_spi_support_refuses_a_base_whose_ids_reach_the_spi_field() -> None:
    # The rebuilt copies mint in the spi_support_channel field, so a surviving
    # base household already at or above its radix (here a synthetic id at a
    # high-value copy's offset) is refused by count, never given colliding
    # ids (microcosm#931).
    dead = _certified_like_dead_spi_support()
    person, benunit, household = (
        dead.person.copy(),
        dead.benunit.copy(),
        dead.household.copy(),
    )
    shift = 10**12
    members = person["person_household_id"] == 2
    units = set(person.loc[members, "person_benunit_id"])
    household.loc[household["household_id"] == 2, "household_id"] += shift
    person.loc[members, ["person_id", "person_household_id", "person_benunit_id"]] += (
        shift
    )
    benunit.loc[benunit["benunit_id"].isin(units), "benunit_id"] += shift

    with pytest.raises(
        ValueError,
        match=r"^1 of the incoming household_id values reach the "
        r"'spi_support_channel' copy-id field",
    ) as caught:
        replace_uk_spi_support_tables(
            person=person,
            benunit=benunit,
            household=household,
            seed=7,
            source_year=2023,
        )
    assert str(2 + shift) not in str(caught.value)


def test_replace_spi_support_refuses_to_discard_positive_spi_mass() -> None:
    dead = _certified_like_dead_spi_support()
    household = dead.household.copy()
    synthetic = household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]
    household.loc[synthetic, "household_weight"] = 1.0

    with pytest.raises(ValueError, match="refusing to discard live population mass"):
        replace_uk_spi_support_tables(
            person=dead.person,
            benunit=dead.benunit,
            household=household,
        )


def test_spi_source_lineage_keeps_source_support_from_doubling() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=None,
        source_year=2023,
    )

    # The SPI support stage duplicates each household (base + synthetic
    # support clone), but the source lineage key must keep pointing at the
    # original household so downstream per-area source-support counts do
    # not double: 6 households, 3 distinct sources.
    household = result.household
    assert len(household) == 6
    assert household["source_household_key"].nunique() == 3
    counts = household.groupby("source_household_key").size()
    assert counts.tolist() == [2, 2, 2]


def test_spi_support_preserves_existing_rowwise_lineage_metadata() -> None:
    household = household_frame()
    household["source_household_id"] = ["2022-1", "2022-2", "2022-3"]
    household["source_year"] = [2022, 2022, 2022]
    household["source_household_key"] = ["2022:1", "2022:2", "2022:3"]
    household["clone_index"] = [0, 1, 2]

    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household,
        expected_spi_households=None,
        source_year=2023,
    )

    assert result.household["source_household_id"].tolist() == [
        "2022-1",
        "2022-2",
        "2022-3",
        "2022-1",
        "2022-2",
        "2022-3",
    ]
    assert result.household["source_year"].tolist() == [2022] * 6
    assert result.household["source_household_key"].tolist() == [
        "2022:1",
        "2022:2",
        "2022:3",
        "2022:1",
        "2022:2",
        "2022:3",
    ]
    assert result.household["clone_index"].tolist() == [0, 1, 2, 0, 1, 2]


def test_spi_support_refuses_to_run_twice_and_caps_the_share_at_one() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=1,
    )

    with pytest.raises(ValueError, match="should run exactly once"):
        create_uk_spi_support_tables(
            person=result.person,
            benunit=result.benunit,
            household=result.household,
            expected_spi_households=1,
        )

    # An expected count above the source households is a share of one: every
    # household is copied, where the positional draw refused (microcosm#931).
    capped = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=4,
    )
    assert capped.spi_inclusion_probability == 1.0
    assert capped.spi_household_ids == (1, 2, 3)
    for value in (0, -1.0, float("nan"), float("inf"), True, "3"):
        with pytest.raises(ValueError, match="expected_spi_households must be"):
            create_uk_spi_support_tables(
                person=person_frame(),
                benunit=benunit_frame(),
                household=household_frame(),
                expected_spi_households=value,
            )


def test_spi_fill_rejects_missing_source_ids() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        expected_spi_households=None,
    )
    donor = pd.DataFrame(
        {
            "person_id": [1001, 2001, 3001],
            "employment_income": [900.0, 1_000.0, 1_200.0],
        }
    )

    with pytest.raises(ValueError, match="missing source ID") as caught:
        fill_support_channel_from_source(
            result.person,
            donor,
            entity="person",
            columns=["employment_income"],
        )
    # Person 2002's copy names a source the donor lacks; the refusal counts
    # the copy rows and names no id (microcosm#931).
    assert str(caught.value).endswith("1 channel row(s) name a source it lacks.")
    assert "2002" not in str(caught.value)

    repeated = pd.concat([donor, donor.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique") as caught:
        fill_support_channel_from_source(
            result.person,
            repeated,
            entity="person",
            columns=["employment_income"],
        )
    assert str(caught.value) == (
        "donor.person_id must be unique; 1 row(s) repeat an earlier row's id."
    )


def test_spi_support_input_refusals_count_rows_and_name_no_id() -> None:
    """The support stage's uniqueness, link and selection refusals print counts
    (microcosm#931); every id here is synthetic."""

    person = person_frame()
    person.loc[3, "person_household_id"] = 777_777
    with pytest.raises(ValueError, match="absent from household") as caught:
        create_uk_spi_support_tables(
            person=person,
            benunit=benunit_frame(),
            household=household_frame(),
            expected_spi_households=None,
        )
    assert "on 1 person row(s)" in str(caught.value)
    assert "777777" not in str(caught.value)

    household = household_frame()
    household.loc[2, "household_id"] = 2
    with pytest.raises(ValueError, match="must be unique") as caught:
        create_uk_spi_support_tables(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household,
            expected_spi_households=None,
        )
    assert str(caught.value) == (
        "household.household_id must be unique; 1 row(s) repeat an earlier row's value."
    )

    with pytest.raises(ValueError, match="absent from household") as caught:
        create_uk_spi_support_tables(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household_frame(),
            selected_household_ids=(1, 888_888),
        )
    assert "contains 1 value(s)" in str(caught.value)
    assert "888888" not in str(caught.value)


def _pension_age_frames():
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [10.0, 30.0, 20.0, 40.0],
            "region": ["LONDON"] * 4,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [1001, 2001, 3001, 3002, 4001],
            "person_household_id": [1, 2, 3, 3, 4],
            "person_benunit_id": [101, 201, 301, 301, 401],
            "age": [70, 40, 66, 30, 65],
            "employment_income": [0.0] * 5,
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101, 201, 301, 401]})
    return person, benunit, household


def test_build_spi_support_channel_gives_pension_age_households_their_own_share() -> (
    None
):
    person, benunit, household = _pension_age_frames()

    result = build_uk_spi_support_channel(
        person=person,
        benunit=benunit,
        household=household,
        expected_spi_households=4,
        seed=42,
        source_year=2023,
        strata_columns=("region",),
        pension_age_share=0.2,
        state_pension_age=66,
    )

    # Households 1 and 3 have a member at or over 66 (30 of the 100 prior
    # mass): the SPI channel takes 20% of it and 50% of the other 70.
    channel = support_channel_column("household")
    table = result.household.set_index(["source_household_id", channel])[
        "household_weight"
    ]
    spi, base = SPI_SYNTHETIC_SUPPORT_CHANNEL, BASE_FRS_SUPPORT_CHANNEL
    assert table[(1, base)] == pytest.approx(8.0)
    assert table[(3, base)] == pytest.approx(16.0)
    assert table[(1, spi)] + table[(3, spi)] == pytest.approx(6.0)
    assert table[(2, base)] == pytest.approx(15.0)
    assert table[(4, base)] == pytest.approx(20.0)
    assert table[(2, spi)] + table[(4, spi)] == pytest.approx(35.0)
    assert result.household["household_weight"].sum() == 100.0
    assert spi_support.SPI_PENSION_AGE_STRATUM_COLUMN not in result.household
    assert result.pension_age_spi_prior_mass_share == 0.2
    assert result.state_pension_age == 66
    assert result.pension_age_allocation == {
        "pension_age_households": 2,
        "pension_age_spi_households": 2,
        "pension_age_prior_mass": 30.0,
        "pension_age_spi_mass": pytest.approx(6.0),
        "other_prior_mass": 70.0,
        "other_spi_mass": pytest.approx(35.0),
    }
    assert "0.2 in households with" in result.mass_log[-1].reason


def test_build_spi_support_channel_without_a_pension_age_share_is_unchanged() -> None:
    person, benunit, household = _pension_age_frames()

    result = build_uk_spi_support_channel(
        person=person,
        benunit=benunit,
        household=household,
        expected_spi_households=4,
        seed=42,
        source_year=2023,
        strata_columns=("region",),
    )

    assert result.pension_age_spi_prior_mass_share is None
    assert result.pension_age_allocation is None
    assert result.mass_log[-1].reason == SPI_PRIOR_MASS_CHANGE_REASON


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pension_age_share": 1.0, "state_pension_age": 66}, "finite and in"),
        ({"pension_age_share": 0.2}, "positive integer state_pension_age"),
        (
            {
                "pension_age_share": 0.2,
                "state_pension_age": 66,
                "strata_columns": ("region", "household_has_state_pension_age_member"),
            },
            "derived by the allocation",
        ),
    ],
)
def test_build_spi_support_channel_refuses_invalid_pension_age_arguments(
    kwargs: dict, message: str
) -> None:
    person, benunit, household = _pension_age_frames()
    arguments = {"strata_columns": ("region",), **kwargs}
    with pytest.raises(ValueError, match=message):
        build_uk_spi_support_channel(
            person=person,
            benunit=benunit,
            household=household,
            expected_spi_households=4,
            seed=42,
            source_year=2023,
            **arguments,
        )


# The keyed Bernoulli share (microcosm#931). Every frame below is synthetic.

_SHARE_RULE = {"rule": spi_support.SPI_SHARE_GATE_RULE, "sigma": 4}
_SHARE_GATE = {
    "name": "e7_spi_synthetic_share",
    "selector": {HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
    "maximum_zero_weight_rows_rule": _SHARE_RULE,
    "reason": "synthetic share gate",
}
_REGIONS = np.asarray(["LONDON", "WALES", "SCOTLAND", "NORTH_EAST"])


def _share_frames(n_households: int, *, seed: int = 0):
    """``n_households`` one-person households over four regions."""

    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_households + 1, dtype="int64")
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": rng.uniform(50.0, 500.0, n_households),
            "region": _REGIONS[ids % 4],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 1_000 + 1,
            "person_household_id": ids,
            "person_benunit_id": ids * 100 + 1,
            "age": 30 + ids % 50,
        }
    )
    benunit = pd.DataFrame({"benunit_id": ids * 100 + 1})
    return person, benunit, household


def _cloned_share_tables(n_households: int, *, n_clones: int):
    """The share frames as ``n_clones`` geographic clones, clone-major."""

    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import engine_tables
    from test_support.microcosm_build.uk_clone_first_frames import clone_first_frame

    person, benunit, household = _share_frames(n_households)
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )
    tables = engine_tables(clone_first_frame(frame, n_clones))
    return tables["person"], tables["benunit"], tables["household"]


def test_the_share_draw_reads_each_households_own_id_alone() -> None:
    from microcosm.build.stochastic_assignment import stable_identity_uniforms

    ids = np.arange(1, 2_001, dtype="int64")
    drawn = spi_support.spi_share_inclusion(ids, inclusion_probability=0.3, seed=42)
    assert 0 < int(drawn.sum()) < ids.size
    # Each household is drawn when its keyed uniform falls below the share.
    np.testing.assert_array_equal(
        drawn,
        stable_identity_uniforms(
            ids, seed=42, salt=spi_support.SPI_SUPPORT_INCLUSION_SALT
        )
        < 0.3,
    )
    # Permuting the rows permutes the draws with them.
    order = np.random.default_rng(7).permutation(ids.size)
    np.testing.assert_array_equal(
        spi_support.spi_share_inclusion(ids[order], inclusion_probability=0.3, seed=42),
        drawn[order],
    )
    # Unrelated households appended at the same probability leave every
    # existing household's draw where it was (the stage's probability divides
    # by the source count, so it is held fixed here).
    more = np.concatenate([ids, np.arange(5_001, 6_001, dtype="int64")])
    np.testing.assert_array_equal(
        spi_support.spi_share_inclusion(more, inclusion_probability=0.3, seed=42)[
            : ids.size
        ],
        drawn,
    )
    assert not np.array_equal(
        spi_support.spi_share_inclusion(ids, inclusion_probability=0.3, seed=43), drawn
    )
    with pytest.raises(TypeError, match="integer ids"):
        spi_support.spi_share_inclusion(
            ids.astype(float), inclusion_probability=0.3, seed=42
        )
    with pytest.raises(ValueError, match="each record once"):
        spi_support.spi_share_inclusion(
            np.r_[ids, ids[:1]], inclusion_probability=0.3, seed=42
        )
    for probability in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="must lie in"):
            spi_support.spi_share_inclusion(
                ids, inclusion_probability=probability, seed=42
            )


def test_a_row_permutation_leaves_the_stage_selection_unchanged() -> None:
    person, benunit, household = _share_frames(600)
    first = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        expected_spi_households=150,
    )
    order = np.random.default_rng(3).permutation(len(household))
    shuffled_household = household.iloc[order].reset_index(drop=True)
    shuffled = create_uk_spi_support_tables(
        person=person.iloc[::-1].reset_index(drop=True),
        benunit=benunit,
        household=shuffled_household,
        expected_spi_households=150,
    )

    assert first.spi_inclusion_probability == shuffled.spi_inclusion_probability
    assert first.spi_inclusion_probability == 0.25
    assert set(shuffled.spi_household_ids) == set(first.spi_household_ids)
    # The copies keep the frame's order.
    position = {
        value: index for index, value in enumerate(shuffled_household["household_id"])
    }
    assert list(shuffled.spi_household_ids) == sorted(
        shuffled.spi_household_ids, key=position.__getitem__
    )


def test_the_share_count_stays_inside_the_four_sigma_band() -> None:
    n, share = 1_000, 0.3
    ids = np.arange(1, n + 1, dtype="int64")
    band = n * binomial_share_tolerance(target=share, rows=n, sigma=4.0)
    counts = [
        int(
            spi_support.spi_share_inclusion(
                ids, inclusion_probability=share, seed=seed
            ).sum()
        )
        for seed in range(200)
    ]

    assert all(abs(count - n * share) <= band for count in counts)
    # A draw, not a quota: the count varies, around n x share.
    assert len(set(counts)) > 10
    assert abs(np.mean(counts) - n * share) <= 4.0 * math.sqrt(
        n * share * (1.0 - share) / len(counts)
    )


def test_a_share_of_one_copies_every_household_whatever_its_uniform(
    monkeypatch,
) -> None:
    ids = np.arange(1, 11, dtype="int64")
    # A keyed uniform is an integer over 2**64, which can round up to exactly
    # one; at a share of one the household is copied all the same.
    monkeypatch.setattr(
        spi_support,
        "stable_identity_uniforms",
        lambda keys, *, seed, salt: np.ones(len(keys)),
    )
    assert spi_support.spi_share_inclusion(ids, inclusion_probability=1.0, seed=0).all()
    assert not spi_support.spi_share_inclusion(
        ids, inclusion_probability=0.999, seed=0
    ).any()
    person, benunit, household = _share_frames(10)
    result = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        expected_spi_households=10,
    )
    assert result.spi_inclusion_probability == 1.0
    assert result.spi_household_ids == tuple(range(1, 11))


def test_the_share_gate_resolves_to_the_share_plus_its_binomial_band() -> None:
    n, share = 1_000, 0.3
    maximum = spi_support.spi_share_gate_maximum(
        _SHARE_RULE, inclusion_probability=share, household_rows=n
    )

    assert maximum == math.ceil(n * (share + 4.0 * math.sqrt(share * 0.7 / n)))
    assert (
        spi_support.spi_share_gate_maximum(
            _SHARE_RULE, inclusion_probability=1.0, household_rows=n
        )
        == n
    )
    # The unchanged gate passes the resolved maximum and refuses one more.
    for zero_rows, passed in ((maximum, True), (maximum + 1, False)):
        synthetic = np.arange(n) < zero_rows
        gate = uk_zero_weight_strata_gate(
            pd.DataFrame(
                {
                    "household_id": np.arange(1, n + 1),
                    "household_weight": np.where(synthetic, 0.0, 1.0),
                    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: synthetic,
                }
            ),
            declarations=spi_support._coerce_zero_weight_declarations(
                (_SHARE_GATE,), inclusion_probability=share, household_rows=n
            ),
        )
        assert gate.passed is passed


@pytest.mark.parametrize(
    ("declaration", "probability", "message"),
    [
        (
            {**_SHARE_GATE, "maximum_zero_weight_rows_rule": {"rule": "normal"}},
            0.3,
            "maximum rule must be",
        ),
        (
            {
                **_SHARE_GATE,
                "maximum_zero_weight_rows_rule": {**_SHARE_RULE, "sigma": 0},
            },
            0.3,
            "sigma must be positive",
        ),
        (
            {
                **_SHARE_GATE,
                "maximum_zero_weight_rows_rule": {**_SHARE_RULE, "sigma": True},
            },
            0.3,
            "sigma must be positive",
        ),
        (
            {**_SHARE_GATE, "maximum_zero_weight_rows": 10},
            0.3,
            "both a maximum and a maximum rule",
        ),
        (_SHARE_GATE, None, "named copies are not a share"),
    ],
)
def test_the_share_gate_refuses_a_malformed_or_unresolvable_rule(
    declaration, probability, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        spi_support._coerce_zero_weight_declarations(
            (declaration,), inclusion_probability=probability, household_rows=100
        )


def test_the_share_gate_passes_each_draw_and_refuses_one_copy_above_it(
    monkeypatch,
) -> None:
    person, benunit, household = _share_frames(400)
    band = 400 * binomial_share_tolerance(target=0.25, rows=400, sigma=4.0)
    for seed in range(40):
        result = build_uk_spi_support_channel(
            person,
            benunit,
            household,
            expected_spi_households=100,
            seed=seed,
            zero_weight_declarations=(_SHARE_GATE,),
        )
        assert abs(result.n_spi_households - 100) <= band
    maximum = spi_support.spi_share_gate_maximum(
        _SHARE_RULE, inclusion_probability=0.25, household_rows=400
    )
    # One copy more than the bound, injected through the draw, is refused, by
    # count.
    monkeypatch.setattr(
        spi_support,
        "spi_share_inclusion",
        lambda ids, *, inclusion_probability, seed: np.arange(len(ids)) <= maximum,
    )
    with pytest.raises(ValueError) as caught:
        build_uk_spi_support_channel(
            person,
            benunit,
            household,
            expected_spi_households=100,
            zero_weight_declarations=(_SHARE_GATE,),
        )
    assert str(caught.value) == (
        "UK SPI support zero-weight gate failed: ['e7_spi_synthetic_share: "
        f"{maximum + 1} zero-weight rows exceed the declared maximum {maximum}.']."
    )


def test_an_unmatched_zero_weight_base_row_still_fails_the_share_gate() -> None:
    person, benunit, household = _share_frames(400)
    household.loc[0, "household_weight"] = 0.0

    with pytest.raises(ValueError, match=r"1 zero-weight household row\(s\) match no"):
        build_uk_spi_support_channel(
            person,
            benunit,
            household,
            expected_spi_households=100,
            zero_weight_declarations=(_SHARE_GATE,),
        )


def test_the_share_counts_distinct_sources_and_refuses_an_unsourced_row() -> None:
    person, benunit, household = _share_frames(8)
    # Two rows per source: the share divides by the four sources.
    household["source_household_id"] = (household["household_id"] + 1) // 2
    result = create_uk_spi_support_tables(
        person=person, benunit=benunit, household=household, expected_spi_households=2
    )
    assert result.spi_source_households == 4
    assert result.spi_inclusion_probability == 0.5

    household["source_household_id"] = household["source_household_id"].astype("Int64")
    household.loc[3, "source_household_id"] = pd.NA
    with pytest.raises(ValueError) as caught:
        create_uk_spi_support_tables(
            person=person,
            benunit=benunit,
            household=household,
            expected_spi_households=2,
        )
    assert str(caught.value) == (
        "1 household row(s) carry no source_household_id; the SPI share counts "
        "distinct source households."
    )


def test_the_clone_stratum_reads_the_lineages_geographic_index() -> None:
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
        spi_support.SPI_GEOGRAPHIC_CLONE_INDEX_COLUMN
        == _GEOGRAPHIC_CLONE_INDEX_COLUMN
        == ladder_clone_index_column("household")
        == branch.index
    )


def test_one_indexed_clone_allocates_bit_for_bit_as_no_index() -> None:
    person, benunit, household = _share_frames(400)
    arguments = {
        "expected_spi_households": 100,
        "pension_age_share": 0.2,
        "state_pension_age": 66,
    }
    plain = build_uk_spi_support_channel(person, benunit, household, **arguments)
    indexed = build_uk_spi_support_channel(
        person,
        benunit,
        household.assign(household_clone_index=np.zeros(len(household), "int64")),
        **arguments,
    )

    assert indexed.spi_household_ids == plain.spi_household_ids
    np.testing.assert_array_equal(
        indexed.household["household_weight"].to_numpy(),
        plain.household["household_weight"].to_numpy(),
    )
    assert indexed.mass_log == plain.mass_log
    assert spi_support.SPI_GEOGRAPHIC_CLONE_STRATUM_COLUMN not in indexed.household


def test_each_geographic_clone_keeps_its_own_spi_prior_mass() -> None:
    person, benunit, household = _cloned_share_tables(400, n_clones=3)
    result = build_uk_spi_support_channel(
        person, benunit, household, expected_spi_households=100
    )
    out = result.household
    spi = out[support_channel_column("household")].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)

    # Every clone and region keeps its mass, half of it on its own copies.
    for (clone, region), group in household.groupby(
        ["household_clone_index", "region"]
    ):
        members = out["household_clone_index"].eq(clone) & out["region"].eq(region)
        mass = float(group["household_weight"].sum())
        assert float(out.loc[members, "household_weight"].sum()) == pytest.approx(
            mass, rel=1e-12
        )
        assert float(out.loc[members & spi, "household_weight"].sum()) == (
            pytest.approx(0.5 * mass, rel=1e-12)
        )
    assert spi_support.SPI_GEOGRAPHIC_CLONE_STRATUM_COLUMN not in out

    # Without the per-clone stratum the region's copies share one weight, so
    # a clone's copies carry the region's share of its copies, not of its mass.
    pooled = build_uk_spi_support_channel(
        person,
        benunit,
        household.drop(columns=["household_clone_index"]),
        expected_spi_households=100,
    ).household
    pooled_spi = pooled[support_channel_column("household")].eq(
        SPI_SYNTHETIC_SUPPORT_CHANNEL
    )
    clone_of = household.set_index("household_id")["household_clone_index"]
    pooled_clone = pooled["household_source_id"].map(clone_of)
    pooled_shares = [
        float(pooled.loc[pooled_spi & pooled_clone.eq(clone), "household_weight"].sum())
        / float(household.loc[clone_of.to_numpy() == clone, "household_weight"].sum())
        for clone in range(3)
    ]
    assert max(abs(share - 0.5) for share in pooled_shares) > 1e-3


def test_an_empty_clone_stratum_is_refused_by_count() -> None:
    person, benunit, household = _cloned_share_tables(8, n_clones=2)
    # Every clone-0 household is copied, and clone 1's but its LONDON ones:
    # LONDON keeps copies in clone 0 only.
    london = household["region"].eq("LONDON").to_numpy()
    clone = household["household_clone_index"].to_numpy()
    named = tuple(household.loc[~(london & (clone == 1)), "household_id"])
    stacked = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        selected_household_ids=named,
    )
    arguments = {
        "spi_prior_mass_share": 0.5,
        "input_weight_kind": WeightKind.DESIGN,
        "mass_log": (),
        "strata_columns": ("region",),
    }

    with pytest.raises(ValueError) as caught:
        spi_support._allocate_spi_prior_mass(stacked, **arguments)
    assert str(caught.value) == (
        "Every positive-mass UK base stratum must have rebuilt SPI support; 1 of "
        "the 8 positive-mass strata have no SPI copy."
    )
    # Pooled over clones, the region has copies.
    pooled = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household.drop(columns=["household_clone_index"]),
        selected_household_ids=named,
    )
    spi_support._allocate_spi_prior_mass(pooled, **arguments)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            np.asarray([0.0, 1.0] * 4),
            "The SPI prior allocation stratifies geographic clones on an integer "
            "'household_clone_index' column; it has dtype float64.",
        ),
        (
            pd.array([0, None, 1, 0, 1, 0, 1, 0], dtype="Int64"),
            "1 household row(s) carry no 'household_clone_index'; the SPI prior "
            "allocation stratifies each geographic clone.",
        ),
        (
            np.asarray([0, -1, 1, -2, 0, 1, 0, 1], dtype="int64"),
            "2 household row(s) carry a negative 'household_clone_index'; the SPI "
            "prior allocation stratifies each geographic clone.",
        ),
    ],
)
def test_the_clone_stratum_refuses_a_malformed_index_by_count(
    values, message: str
) -> None:
    person, benunit, household = _share_frames(8)
    with pytest.raises(ValueError) as caught:
        build_uk_spi_support_channel(
            person,
            benunit,
            household.assign(household_clone_index=values),
            expected_spi_households=8,
        )
    assert str(caught.value) == message


def test_the_derived_clone_stratum_is_not_a_declared_stratum() -> None:
    person, benunit, household = _share_frames(8)
    column = spi_support.SPI_GEOGRAPHIC_CLONE_STRATUM_COLUMN

    with pytest.raises(ValueError, match="derived by the allocation"):
        build_uk_spi_support_channel(
            person,
            benunit,
            household.assign(**{column: 0}),
            expected_spi_households=8,
            strata_columns=("region", column),
        )
