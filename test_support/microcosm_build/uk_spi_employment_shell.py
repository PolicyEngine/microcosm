"""Synthetic FRS and SPI tables for the SPI employment shell tests.

FRS adults' occupation follows their pay, and their industry and sector follow
their occupation. SPI-channel copies draw high pay but keep a low-paid twin's
occupation, industry and sector, so a stage that conditions on the recipient's
own pay moves them into the occupations that pay implies and gives them an
industry from an adult in that occupation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REGIONS = ("LONDON", "NORTH_EAST", "WALES")
#: Pay rank (low to high) -> the planted occupation, industry division, sector.
_LADDER = (
    ("ELEMENTARY", 47, "PRIVATE"),
    ("SALES_AND_CUSTOMER_SERVICE", 47, "PRIVATE"),
    ("SKILLED_TRADES", 43, "PRIVATE"),
    ("ASSOCIATE_PROFESSIONAL", 86, "PUBLIC"),
    ("PROFESSIONAL", 85, "PUBLIC"),
    ("MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS", 70, "PRIVATE"),
)


def employment_shell_tables(n_frs: int = 1500, n_spi: int = 300, seed: int = 0):
    """FRS adults whose occupation follows pay; SPI copies with a low-paid twin's.

    Returns ``(person, household, household_weights)``. Each household has one
    adult and one child; every SPI household copies an FRS household (its
    ``household_source_id``), its adult is an SPI income recipient with high
    drawn pay and the twin's low-paid occupation, industry and sector. A few
    SPI adults draw no pay or profit and had no occupation, so they keep NONE.
    """

    rng = np.random.default_rng(seed)
    frs_pay = rng.lognormal(10.0, 0.7, n_frs)
    rank = np.searchsorted(np.sort(frs_pay), frs_pay) / n_frs
    level = np.minimum((rank * len(_LADDER)).astype(int), len(_LADDER) - 1)
    noisy = np.where(
        rng.random(n_frs) < 0.1, rng.integers(0, len(_LADDER), n_frs), level
    )
    twins = rng.integers(0, n_frs, n_spi)
    # The copies' twins are low paid: their occupation sits at the bottom.
    twins = np.argsort(frs_pay)[: max(n_spi, 1)][rng.integers(0, max(n_spi, 1), n_spi)]
    n = n_frs + n_spi
    household_id = np.arange(1, n + 1)
    channel = np.array(["frs"] * n_frs + ["spi"] * n_spi)
    source = np.concatenate([household_id[:n_frs], household_id[twins]])
    region = np.concatenate(
        [rng.choice(_REGIONS, n_frs), rng.choice(_REGIONS, n_frs)[twins]]
    )
    household = pd.DataFrame(
        {
            "household_id": household_id,
            "household_support_channel": channel,
            "household_support_clone_index": np.where(channel == "spi", 1, 0),
            "household_source_id": source,
            # A copy's true source is the FRS household it copies.
            "source_household_id": source,
            "region": region,
        }
    )
    weights = np.concatenate([rng.uniform(500, 1500, n_frs), rng.uniform(5, 50, n_spi)])

    occupation = np.array([step[0] for step in _LADDER], dtype=object)[noisy]
    division = np.array([step[1] for step in _LADDER])[noisy]
    sector = np.array([step[2] for step in _LADDER], dtype=object)[noisy]
    spi_pay = rng.lognormal(12.3, 0.4, n_spi)
    workless = np.zeros(n_spi, dtype=bool)
    workless[:5] = True
    adult = pd.DataFrame(
        {
            "person_household_id": household_id,
            "age": np.concatenate(
                [rng.integers(20, 70, n_frs), rng.integers(30, 70, n_spi)]
            ),
            "gender": rng.choice(["MALE", "FEMALE"], n),
            "employment_status": np.where(
                rng.random(n) < 0.85, "FT_EMPLOYED", "FT_SELF_EMPLOYED"
            ),
            "employment_income": np.concatenate(
                [frs_pay, np.where(workless, 0.0, spi_pay)]
            ),
            "hmrc_spi_pay": np.concatenate(
                [np.zeros(n_frs), np.where(workless, 0.0, spi_pay)]
            ),
            "self_employment_income": np.zeros(n),
            "savings_interest_income": rng.exponential(200.0, n),
            "dividend_income": np.zeros(n),
            "private_pension_income": np.zeros(n),
            "person_is_spi_income_recipient": channel == "spi",
            "soc2020_major_group": np.concatenate(
                [occupation, np.where(workless, "NONE", occupation[twins])]
            ),
            "sic_industry_division": np.concatenate(
                [division, np.where(workless, 0, division[twins])]
            ).astype(np.int64),
            "employment_sector": np.concatenate(
                [sector, np.where(workless, "NOT_EMPLOYED", sector[twins])]
            ),
        }
    )
    adult.loc[n_frs + np.flatnonzero(workless), "employment_status"] = "OTHER_INACTIVE"
    child = pd.DataFrame(
        {
            "person_household_id": household_id,
            "age": np.full(n, 8),
            "gender": rng.choice(["MALE", "FEMALE"], n),
            "employment_status": "CHILD",
            "employment_income": 0.0,
            "hmrc_spi_pay": 0.0,
            "self_employment_income": 0.0,
            "savings_interest_income": 0.0,
            "dividend_income": 0.0,
            "private_pension_income": 0.0,
            "person_is_spi_income_recipient": False,
            "soc2020_major_group": "NONE",
            "sic_industry_division": np.zeros(n, dtype=np.int64),
            "employment_sector": "NOT_EMPLOYED",
        }
    )
    person = pd.concat([adult, child], ignore_index=True)
    person.insert(0, "person_id", np.arange(1, len(person) + 1) * 10)
    return person, household, weights
