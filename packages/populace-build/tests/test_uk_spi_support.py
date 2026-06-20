from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk import (
    BASE_FRS_SUPPORT_CHANNEL,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_INCOME_COMPONENT_COLUMNS,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    area_support_summary,
    create_uk_spi_support_tables,
    fill_support_channel_from_source,
    stacked_weights_to_long,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
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
        spi_household_count=None,
        source_year=2023,
    )

    assert result.id_multiplier == 10_000
    assert result.n_spi_households == 3
    assert result.household["household_id"].tolist() == [
        1,
        2,
        3,
        10001,
        10002,
        10003,
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
        11001,
        12001,
        12002,
        13001,
    ]
    assert result.person["person_household_id"].tolist() == [
        1,
        2,
        2,
        3,
        10001,
        10002,
        10002,
        10003,
    ]
    assert result.person["person_benunit_id"].tolist() == [
        101,
        201,
        201,
        301,
        10101,
        10201,
        10201,
        10301,
    ]
    assert result.benunit["benunit_id"].tolist() == [101, 201, 301, 10101, 10201, 10301]


def test_spi_support_can_subsample_without_rescaling_or_reordering_base() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        spi_household_count=2,
        seed=1,
    )

    assert result.household.head(3)["household_id"].tolist() == [1, 2, 3]
    assert result.household.head(3)["household_weight"].tolist() == [10.0, 20.0, 30.0]
    spi_households = result.household[
        result.household[support_channel_column("household")]
        == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ]
    assert len(spi_households) == 2
    assert spi_households["household_weight"].eq(0).all()
    assert tuple(spi_households[support_source_id_column("household")]) == (
        result.spi_household_ids
    )


def test_spi_fill_only_updates_spi_channel_and_can_initialize_new_columns() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        spi_household_count=None,
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


def test_spi_variable_surfaces_include_recent_stage1_and_stage2_fixes() -> None:
    assert SPI_INCOME_COMPONENT_COLUMNS == (
        "employment_income",
        "self_employment_income",
        "savings_interest_income",
        "dividend_income",
        "private_pension_income",
        "property_income",
    )
    assert "gift_aid" in SPI_INCOME_IMPUTATION_COLUMNS
    assert "charitable_investment_gifts" in SPI_INCOME_IMPUTATION_COLUMNS
    assert set(SPI_INCOME_COMPONENT_COLUMNS).issubset(
        FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS
    )
    assert "universal_credit_reported" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    assert "housing_benefit_reported" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    assert "employee_pension_contributions" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    assert "tax_free_savings_income" in FRS_ONLY_SPI_FILL_PERSON_COLUMNS


def test_spi_source_lineage_keeps_longwise_source_support_from_doubling() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        spi_household_count=None,
        source_year=2023,
    )
    weights = np.ones(2 * len(result.household))

    long = stacked_weights_to_long(
        weights,
        area_codes=["A", "B"],
        household_ids=result.household["household_id"],
        household_frame=result.household,
        area_type="local_authority",
    )
    summary = area_support_summary(long)

    assert summary["nonzero_households"].tolist() == [6, 6]
    assert summary["nonzero_source_households"].tolist() == [3, 3]


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
        spi_household_count=None,
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


def test_spi_support_refuses_to_run_twice_or_oversample() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        spi_household_count=1,
    )

    with pytest.raises(ValueError, match="should run exactly once"):
        create_uk_spi_support_tables(
            person=result.person,
            benunit=result.benunit,
            household=result.household,
            spi_household_count=1,
        )

    with pytest.raises(ValueError, match="cannot exceed"):
        create_uk_spi_support_tables(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household_frame(),
            spi_household_count=4,
        )


def test_spi_fill_rejects_missing_source_ids() -> None:
    result = create_uk_spi_support_tables(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        spi_household_count=None,
    )
    donor = pd.DataFrame(
        {
            "person_id": [1001, 2001, 3001],
            "employment_income": [900.0, 1_000.0, 1_200.0],
        }
    )

    with pytest.raises(ValueError, match="missing source ID"):
        fill_support_channel_from_source(
            result.person,
            donor,
            entity="person",
            columns=["employment_income"],
        )
