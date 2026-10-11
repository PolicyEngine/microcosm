"""Keep Form 5498 contribution benchmarks separate from SOI deductions."""

from __future__ import annotations

import json
import re
from importlib.resources import files

import pytest


def _benchmarks() -> dict:
    return json.loads(
        files("microcosm.build.us").joinpath("ira_benchmarks.json").read_text()
    )


def test_gross_contributions_and_deductible_payments_have_separate_grains() -> None:
    data = _benchmarks()
    assert data["use"] == "validation_only"
    assert data["country"] == "us"
    gross = data["gross_contributions"]
    deductions = data["model_derived_deductions"]
    assert {row["year"] for row in gross} == {2022, 2023}
    assert {row["year"] for row in deductions} == {2022, 2023}
    assert {row["model_variable"] for row in gross} == {
        "traditional_ira_contributions",
        "roth_ira_contributions",
    }
    assert {row["model_variable"] for row in deductions} == {
        "traditional_ira_deduction"
    }
    for row in gross:
        assert row["count_unit"] == "taxpayer"
        assert row["number_of_taxpayers"] > 0
        assert "number_of_returns" not in row
    for row in deductions:
        assert row["count_unit"] == "return"
        assert row["number_of_returns"] > 0
        assert "number_of_taxpayers" not in row


@pytest.mark.parametrize(
    ("year", "account_type", "contributors", "amount_thousands"),
    [
        (2022, "traditional", 5_900_726, 25_807_585),
        (2022, "roth", 9_170_792, 30_584_013),
        (2023, "traditional", 5_873_053, 26_901_749),
        (2023, "roth", 9_488_414, 32_809_393),
    ],
)
def test_gross_benchmarks_bind_june_2026_all_taxpayers_observations(
    year: int, account_type: str, contributors: int, amount_thousands: int
) -> None:
    row = next(
        row
        for row in _benchmarks()["gross_contributions"]
        if row["year"] == year and row["account_type"] == account_type
    )
    assert row["number_of_taxpayers"] == contributors
    assert row["amount_thousands"] == amount_thousands
    source = row["source"]
    assert source["publication_vintage"] == "June 2026"
    assert source["row_label"] == "All taxpayers"
    assert source["row_label_cell"] == "A7"
    assert source["count_cell"] == "B7"
    assert source["amount_cell"] == "C7"


@pytest.mark.parametrize(
    ("year", "returns", "amount_thousands"),
    [(2022, 2_430_935, 13_166_590), (2023, 2_503_934, 13_771_289)],
)
def test_deduction_benchmarks_bind_ira_payments_column(
    year: int, returns: int, amount_thousands: int
) -> None:
    row = next(
        row
        for row in _benchmarks()["model_derived_deductions"]
        if row["year"] == year
    )
    assert row["number_of_returns"] == returns
    assert row["amount_thousands"] == amount_thousands
    source = row["source"]
    assert source["measure_label"] == "IRA payments"
    assert source["measure_label_cell"] == "DT4"
    assert source["count_cell"] == "DT9"
    assert source["amount_cell"] == "DU9"


def test_all_sources_bind_workbook_bytes_and_thousands_of_dollars() -> None:
    data = _benchmarks()
    rows = data["gross_contributions"] + data["model_derived_deductions"]
    assert len(rows) == 6
    for row in rows:
        source = row["source"]
        assert source["url"].startswith("https://www.irs.gov/pub/irs-soi/")
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        assert str(row["year"]) in source["title"]
        assert source["money_unit"] == "thousand_usd"
        assert source["money_unit_cell"] == "A2"
        assert isinstance(row["amount_thousands"], int)
        assert row["amount_thousands"] > 0
