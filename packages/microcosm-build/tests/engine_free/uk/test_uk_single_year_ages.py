"""A built file's single years of age against ONS mid-2025 (microcosm#1095)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.single_year_ages import (
    UK_OPEN_AGE,
    UK_SINGLE_YEAR_AGE_SEXES,
    load_ons_single_year_ages,
    single_year_age_report,
)

ONS = {(sex, age): 1_000.0 for sex in UK_SINGLE_YEAR_AGE_SEXES for age in range(91)}


def _file(ages, genders, households, sources, weights):
    person = pd.DataFrame(
        {
            "person_household_id": households,
            "age": ages,
            "gender": genders,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, len(sources) + 1),
            "source_household_id": sources,
        }
    )
    return person, household, np.asarray(weights, dtype=float)


def test_the_ons_estimates_carry_both_sexes_at_every_single_year():
    ons = load_ons_single_year_ages()

    assert len(ons) == 2 * (UK_OPEN_AGE + 1)
    assert all(value > 0 for value in ons.values())


def test_the_report_counts_weighted_persons_and_withholds_thin_cells():
    # Households 1 and 2 are copies of one source household.
    person, household, weights = _file(
        ages=[30, 30, 30.5, 95, 90, 81],
        genders=["FEMALE", "FEMALE", "FEMALE", "FEMALE", "FEMALE", "MALE"],
        households=[1, 2, 3, 4, 5, 6],
        sources=[1, 1, 2, 3, 4, 5],
        weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    report = single_year_age_report(
        person, household, weights, ons=ONS, minimum_carriers=2
    )

    female = report["sexes"]["female"]
    assert female["single_years"]["30"] == {
        "source_households": 2,
        "weighted_persons": 6.0,
        "ons_persons": 1_000.0,
        "ratio_to_ons": 0.006,
    }
    # 95 is in ONS's open age; the band of 90 and over is that one cell.
    assert female["single_years"]["90_plus"]["weighted_persons"] == 9.0
    assert female["tail_bands"]["90_plus"]["weighted_persons"] == 9.0
    assert female["tail_bands"]["90_plus"]["ons_persons"] == 1_000.0
    # An age with fewer carriers than the floor, none included, is withheld,
    # and so is every band that holds it.
    assert female["single_years"]["85"] == {"source_households": "<2"}
    assert female["tail_bands"]["85_89"] == {"source_households": "<2"}
    assert female["tail_bands"]["80_plus"] == {"source_households": "<2"}
    male = report["sexes"]["male"]
    assert male["single_years"]["81"] == {"source_households": "<2"}
    assert male["tail_bands"]["80_84"] == {"source_households": "<2"}
    assert report["minimum_source_households"] == 2


def test_the_default_floor_is_ten_source_households():
    person, household, weights = _file(
        ages=[40] * 9,
        genders=["MALE"] * 9,
        households=list(range(1, 10)),
        sources=list(range(1, 10)),
        weights=[1.0] * 9,
    )

    report = single_year_age_report(person, household, weights, ons=ONS)

    assert report["sexes"]["male"]["single_years"]["40"] == {"source_households": "<10"}


def test_the_report_refuses_a_file_it_cannot_count_carriers_in():
    person, household, weights = _file(
        ages=[40],
        genders=["MALE"],
        households=[1],
        sources=[1],
        weights=[1.0],
    )

    with pytest.raises(ValueError, match="source households"):
        single_year_age_report(
            person, household.drop(columns="source_household_id"), weights, ons=ONS
        )
    with pytest.raises(ValueError, match="one final weight"):
        single_year_age_report(person, household, np.ones(2), ons=ONS)
