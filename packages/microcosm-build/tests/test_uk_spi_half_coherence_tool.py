"""The SPI-half coherence instrument pairs support rows with their FRS parents."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_TOOL_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "measure_uk_spi_half_coherence.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "measure_uk_spi_half_coherence", _TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spine(n: int = 40, *, spi_copies_parent_wealth: bool):
    """``n`` one-adult FRS households, each copied once into the SPI channel."""

    rng = np.random.default_rng(0)
    frs_income = rng.lognormal(10, 1, n)
    spi_income = rng.lognormal(12, 1, n)
    frs_wealth = frs_income * 3
    household = pd.DataFrame(
        {
            "household_id": np.r_[np.arange(1, n + 1), np.arange(1, n + 1) + 1000],
            "household_source_id": np.r_[np.arange(1, n + 1), np.arange(1, n + 1)],
            "household_support_channel": ["frs"] * n + ["spi"] * n,
            "household_is_capital_gains_clone": False,
            "household_is_cgt_band_donor": False,
            "household_weight": 1.0,
            "region": "LONDON",
            "gross_financial_wealth": np.r_[
                frs_wealth, frs_wealth if spi_copies_parent_wealth else spi_income * 3
            ],
            "corporate_wealth": 0.0,
            "property_wealth": 0.0,
            "savings": 0.0,
            "food_and_non_alcoholic_beverages_consumption": 1.0,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.r_[np.arange(1, n + 1), np.arange(1, n + 1) + 1000],
            "person_source_id": np.r_[np.arange(1, n + 1), np.arange(1, n + 1)],
            "person_household_id": household["household_id"],
            "person_support_channel": household["household_support_channel"],
            "age": 40,
            "gender": "MALE",
            "employment_income": np.r_[frs_income, spi_income],
        }
    )
    for column in ("self_employment_income", "private_pension_income"):
        person[column] = 0.0
    person["dividend_income"] = np.r_[frs_income, spi_income] / 10
    person["savings_interest_income"] = 0.0
    person["property_income"] = 0.0
    return person, household


def test_inherited_wealth_is_detected_and_coherence_measured() -> None:
    tool = _load_tool()
    person, household = _spine(spi_copies_parent_wealth=True)
    halves = tool.split_halves(person, household)

    inherited = tool.inherited_from_parent(halves)
    assert inherited["spi_households"] == 40
    assert inherited["household"]["gross_financial_wealth"]["identical_share"] == 1.0

    coherence = tool.coherence(halves)
    assert (
        coherence["frs_half"]["spearman_investment_income_gross_financial_wealth"]
        == 1.0
    )
    # Copied parent wealth is unrelated to the SPI draws.
    assert (
        abs(coherence["spi_half"]["spearman_investment_income_gross_financial_wealth"])
        < 0.5
    )


def test_reimputed_wealth_is_not_inherited() -> None:
    tool = _load_tool()
    person, household = _spine(spi_copies_parent_wealth=False)
    halves = tool.split_halves(person, household)

    inherited = tool.inherited_from_parent(halves)
    assert inherited["household"]["gross_financial_wealth"]["identical_share"] == 0.0
    coherence = tool.coherence(halves)
    assert (
        coherence["spi_half"]["spearman_investment_income_gross_financial_wealth"]
        == 1.0
    )


def test_later_cgt_copies_are_left_out() -> None:
    tool = _load_tool()
    person, household = _spine(spi_copies_parent_wealth=True)
    household.loc[household.index[-1], "household_is_capital_gains_clone"] = True
    halves = tool.split_halves(person, household)
    assert len(halves["spi_households"]) == 39
    assert len(halves["spi_persons"]) == 39


def test_income_pairs_report_the_placebo() -> None:
    tool = _load_tool()
    person, household = _spine(spi_copies_parent_wealth=True)
    halves = tool.split_halves(person, household)

    result = tool.income_vs_parent(halves, household, placebo_reps=3, seed=0)
    assert result["spi_adults"] == 40
    employment = result["items"]["employment_income"]
    assert employment["zero_status_agreement"] == 1.0
    assert employment["placebo_spearman_mean"] is not None
    # Columns that are zero everywhere carry no rank information.
    assert result["items"]["property_income"]["spearman"] is None
