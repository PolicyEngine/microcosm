"""Synthetic APS donors and frame tables for the APS occupation stage tests.

Within every major group an employee's APS sub-major group follows pay: the
group's first sub-major group above the group's median weekly pay, one of the
others below it. Everyone else draws a sub-major group at random. The frame's
FRS households hold an adult and a partner with a major group and a child with
none; each SPI household copies one, with its adult an SPI income recipient on
a high drawn pay (``hmrc_spi_pay``) and its partner and child exact copies of
the twin's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.occupation_constants import (
    UK_SOC2020_MAJOR_GROUPS,
    UK_SOC2020_SUB_MAJOR_GROUPS,
)

#: APS government office region codes the donors live in: Merseyside (3) and
#: the nations included.
_GOVTOF = (1, 2, 3, 8, 11, 12, 13)
_REGIONS = ("NORTH_EAST", "NORTH_WEST", "LONDON", "WALES", "SCOTLAND")
#: SIC 2007 divisions the frame's adults work in; 99 (section U) is a section
#: no donor holds.
_DIVISIONS = (47, 85, 86, 62, 43)


def sub_major_codes(major: int) -> list[int]:
    """A major group's sub-major group codes, in code order."""
    return [code for code in UK_SOC2020_SUB_MAJOR_GROUPS if code // 10 == major]


def aps_raw_table(n: int = 5400, seed: int = 0) -> pd.DataFrame:
    """APS person rows under the codebook's names, with the planted pay rule.

    Besides the donors there are under-16s, adults without a minor group and
    employees whose pay is unknown, which the cleaning drops.
    """

    rng = np.random.default_rng(seed)
    major = rng.integers(1, 10, n)
    employed = rng.random(n) < 0.8
    status = np.where(
        employed,
        np.where(rng.random(n) < 0.85, 1, 2),
        rng.choice([16, 24, 5, 20], n),
    )
    weekly = rng.lognormal(6.3, 0.6, n)
    employee = status == 1
    pay_known = employee & (rng.random(n) < 0.85)
    sub = np.empty(n, dtype=np.int64)
    for group in range(1, 10):
        codes = np.asarray(sub_major_codes(group))
        rows = major == group
        high = weekly[rows] > np.median(weekly[rows & pay_known])
        planted = np.where(high, codes[0], rng.choice(codes[1:], rows.sum()))
        random = rng.choice(codes, rows.sum())
        noisy = rng.random(rows.sum()) < 0.1
        sub[rows] = np.where(pay_known[rows] & ~noisy, planted, random)
    minor = sub * 10 + 1
    coded = rng.random(n) < 0.95
    age = rng.integers(16, 86, n)
    child = rng.random(n) < 0.03
    section = rng.integers(1, 21, n)  # never 21, section U
    return pd.DataFrame(
        {
            "AGE": np.where(child, 15, age),
            "SEX": rng.integers(1, 3, n),
            "GOVTOF": rng.choice(_GOVTOF, n),
            "ILODEFR": np.where(employed, 1, 3),
            "INECAC05": np.where(child, 26, status),
            "SC20MMN": np.where(employed & coded, minor, -9),
            "SC20LMN": np.where(~employed & coded, minor, -9),
            "GRSSWK": np.where(
                pay_known, np.round(weekly, 2), np.where(employee, -8, -9)
            ),
            "PWTA22": rng.uniform(500.0, 1500.0, n),
            "PIWTA22": np.where(pay_known, rng.uniform(800.0, 2000.0, n), -9.0),
            "INDS07M": np.where(employed, section, -9),
            "INDS07L": np.where(~employed, section, -9),
            "CASENO": np.arange(n),
        }
    )


def occupation_frame_tables(n_frs: int = 900, n_spi: int = 150, seed: int = 0):
    """FRS households and SPI copies of some, as the employment shell leaves them.

    Returns ``(person, household, household_weights)``. FRS rows' source ids
    are their own; a copy's ``person_source_id`` is its twin's person id and
    its ``household_source_id`` the twin's household.
    """

    rng = np.random.default_rng(seed)
    majors = np.asarray(list(UK_SOC2020_MAJOR_GROUPS.values()), dtype=object)
    household_id = np.arange(1, n_frs + 1)
    household = pd.DataFrame(
        {
            "household_id": household_id,
            "household_support_channel": "frs",
            "household_source_id": household_id,
            "region": rng.choice(_REGIONS, n_frs),
        }
    )

    def adults(status_draw: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "person_household_id": household_id,
                "age": rng.integers(18, 90, n_frs),
                "gender": rng.choice(["MALE", "FEMALE"], n_frs),
                "employment_status": status_draw,
                "employment_income": np.where(
                    status_draw == "FT_EMPLOYED", rng.lognormal(10.25, 0.6, n_frs), 0.0
                ),
                "sic_industry_division": rng.choice(_DIVISIONS, n_frs).astype(np.int64),
                "soc2020_major_group": rng.choice(majors, n_frs),
                "person_is_spi_income_recipient": False,
                "hmrc_spi_pay": 0.0,
            }
        )

    statuses = np.array(["FT_EMPLOYED", "FT_SELF_EMPLOYED", "RETIRED", "UNEMPLOYED"])
    adult = adults(statuses[rng.choice(4, n_frs, p=[0.75, 0.1, 0.1, 0.05])])
    partner = adults(statuses[rng.choice(4, n_frs, p=[0.75, 0.1, 0.1, 0.05])])
    child = pd.DataFrame(
        {
            "person_household_id": household_id,
            "age": np.full(n_frs, 9),
            "gender": rng.choice(["MALE", "FEMALE"], n_frs),
            "employment_status": "CHILD",
            "employment_income": 0.0,
            "sic_industry_division": np.zeros(n_frs, dtype=np.int64),
            "soc2020_major_group": "NONE",
            "person_is_spi_income_recipient": False,
            "hmrc_spi_pay": 0.0,
        }
    )
    frs = pd.concat([adult, partner, child], ignore_index=True)
    frs.insert(0, "person_id", np.arange(1, len(frs) + 1) * 10)
    frs["person_source_id"] = frs["person_id"]

    twins = rng.choice(household_id, n_spi, replace=False)
    copies = []
    for index, twin in enumerate(twins):
        copy = frs.loc[frs["person_household_id"] == twin].copy()
        copy["person_source_id"] = copy["person_id"]
        copy["person_household_id"] = n_frs + index + 1
        lead = copy.index[0]
        copy.loc[lead, "person_is_spi_income_recipient"] = True
        copy.loc[lead, "employment_status"] = "FT_EMPLOYED"
        # The tape's PAY, and an employment_income that adds benefits and
        # termination pay to it.
        copy.loc[lead, "hmrc_spi_pay"] = rng.lognormal(12.3, 0.4)
        copy.loc[lead, "employment_income"] = copy.loc[lead, "hmrc_spi_pay"] + 5000.0
        copies.append(copy)
    spi = pd.concat(copies, ignore_index=True)
    spi["person_id"] = 100_000 + np.arange(len(spi)) * 10
    person = pd.concat([frs, spi], ignore_index=True)
    spi_households = pd.DataFrame(
        {
            "household_id": n_frs + np.arange(1, n_spi + 1),
            "household_support_channel": "spi",
            "household_source_id": twins,
            "region": household.set_index("household_id")
            .loc[twins, "region"]
            .to_numpy(),
        }
    )
    household = pd.concat([household, spi_households], ignore_index=True)
    weights = np.concatenate(
        [rng.uniform(500.0, 1500.0, n_frs), rng.uniform(5.0, 50.0, n_spi)]
    )
    return person, household, weights
