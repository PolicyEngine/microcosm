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


def _two_carriers_at_every_age(gender, first_household):
    """Two source households carry each single year, at weight 1 each."""

    ages = [age for age in range(UK_OPEN_AGE) for _ in range(2)] + [90, 95]
    households = list(range(first_household, first_household + len(ages)))
    return ages, [gender] * len(ages), households


def test_the_report_counts_weighted_persons_and_withholds_thin_cells():
    ages, genders, households = _two_carriers_at_every_age("FEMALE", 1)
    # Two more copies of one female source household at 30, weighted 2 and 3.
    ages += [30, 30.5]
    genders += ["FEMALE", "FEMALE"]
    households += [len(households) + 1, len(households) + 2]
    sources = list(range(1, len(households) - 1)) + [61, 61]
    weights = [1.0] * (len(households) - 2) + [2.0, 3.0]
    # One man of 81: every other year of his sex has no carrier at all.
    ages.append(81)
    genders.append("MALE")
    households.append(len(households) + 1)
    sources.append(len(households) + 1)
    weights.append(6.0)
    person, household, weights = _file(ages, genders, households, sources, weights)

    report = single_year_age_report(
        person, household, weights, ons=ONS, minimum_carriers=2
    )

    female = report["sexes"]["female"]
    # Age 30's two persons of the first pass are source households 61 and 62;
    # the copies of 61 add weight, not carriers.
    assert female["single_years"]["30"] == {
        "source_households": 2,
        "weighted_persons": 7.0,
        "ons_persons": 1_000.0,
        "ratio_to_ons": 0.007,
    }
    # 95 is in ONS's open age; the band of 90 and over is that one cell.
    assert female["single_years"]["90_plus"]["weighted_persons"] == 2.0
    assert female["tail_bands"]["90_plus"]["weighted_persons"] == 2.0
    assert female["tail_bands"]["90_plus"]["ons_persons"] == 1_000.0
    assert female["tail_bands"]["85_89"]["weighted_persons"] == 10.0
    assert female["tail_bands"]["85_89"]["ons_persons"] == 5_000.0
    # A year with fewer carriers than the floor withholds every cell of its
    # sex, so the sex's persons less the shown years cannot give it back.
    male = report["sexes"]["male"]
    assert male["single_years"]["81"] == {"source_households": "<2"}
    assert all(
        cell == {"source_households": "<2"}
        for part in ("single_years", "tail_bands")
        for cell in male[part].values()
    )
    assert set(male["single_years"]) == set(female["single_years"])
    assert set(male["tail_bands"]) == set(female["tail_bands"])
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
