"""policyengine-uk derives the stored PIP categories from the stage's points (microcosm#1095, P1).

The pip_points stage's own rules are tested engine-free in
tests/engine_free/uk/test_uk_pip_points.py. Here, on the locked engine
(2.177.4), the points it draws give back every stored category in every year
from 2024 to 2030, with no category supplied: the release drops the categories
(UK_RELEASE_EXPORT_DROPPED_COLUMNS) and carries the points into every projected
year, as policyengine-uk's single-year extension copies a stored column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.pip_points import (
    PIP_AGE_GROUPS,
    PIP_ENHANCED_POINTS,
    PIP_NOT_ASSESSED,
    PIP_STANDARD_POINTS,
    load_pip_points_distributions,
    store_pip_points,
)
from microcosm.frame import WeightKind

YEARS = tuple(range(2024, 2031))
AWARDS = ("NONE", "STANDARD", "ENHANCED")


def test_the_engine_reads_the_points_on_the_stages_thresholds() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    for year in YEARS:
        pip = system.parameters(year).gov.dwp.pip
        for component in (pip.daily_living, pip.mobility):
            thresholds = component.points_thresholds
            assert (thresholds.standard, thresholds.enhanced) == (
                PIP_STANDARD_POINTS,
                PIP_ENHANCED_POINTS,
            )
    for name in ("pip_dl_points", "pip_m_points"):
        assert system.variables[name].default_value == PIP_NOT_ASSESSED


def _stage_frame():
    # Every pair of awards at an age in every group, three times over.
    ages = [17, 25, 35, 45, 55, 70]
    assert len(ages) == len(PIP_AGE_GROUPS)
    rows = [(dl, m, age) for dl in AWARDS for m in AWARDS for age in ages] * 3
    count = len(rows)
    person = pd.DataFrame(
        {
            "person_id": [1_000 + i for i in range(count)],
            "person_benunit_id": list(range(1, count + 1)),
            "person_household_id": list(range(1, count + 1)),
            "age": [age for _, _, age in rows],
            "pip_dl_category": [dl for dl, _, _ in rows],
            "pip_m_category": [m for _, m, _ in rows],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": list(range(1, count + 1)),
            "region": ["WALES", "SCOTLAND", "LONDON"] * (count // 3),
            "household_source_id": list(range(count)),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": list(range(1, count + 1))}),
        household=household,
        household_weights=np.ones(count),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_the_engine_derives_every_stored_category_from_the_stage_points() -> None:
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKMultiYearDataset, UKSingleYearDataset

    stored = store_pip_points(
        _stage_frame(), distributions=load_pip_points_distributions()
    ).frame.table("person")
    count = len(stored)
    released = pd.DataFrame(
        {
            "person_id": stored["person_id"].to_numpy(),
            "person_benunit_id": stored["person_benunit_id"].to_numpy(),
            "person_household_id": stored["person_household_id"].to_numpy(),
            "age": stored["age"].to_numpy(),
            # The release's columns: the points, and no category.
            "pip_dl_points": stored["pip_dl_points"].to_numpy(),
            "pip_m_points": stored["pip_m_points"].to_numpy(),
        }
    )

    def single(year: int) -> UKSingleYearDataset:
        return UKSingleYearDataset(
            person=released.copy(),
            benunit=pd.DataFrame({"benunit_id": released["person_benunit_id"]}),
            household=pd.DataFrame(
                {
                    "household_id": released["person_household_id"],
                    "household_weight": np.ones(count),
                }
            ),
            fiscal_year=year,
        )

    simulation = Microsimulation(
        dataset=UKMultiYearDataset(datasets=[single(year) for year in YEARS])
    )
    for year in YEARS:
        for category in ("pip_dl_category", "pip_m_category"):
            engine = np.asarray(simulation.calculate(category, year).values).astype(str)
            assert (engine == stored[category].to_numpy().astype(str)).all(), (
                year,
                category,
            )
