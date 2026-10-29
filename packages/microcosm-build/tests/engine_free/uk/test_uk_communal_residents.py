"""Census communal residents by area and age band (microcosm#1095, #1278).

Synthetic facts only: the shares come from census residence-type counts by
area and age, and Northern Ireland's from its communal count over its mid-year
population, spread over single years.
"""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.communal_residents import (
    communal_cells,
    is_communal_fact_line,
    private_household_shares,
)

_HOUSEHOLD = "Lives in a household"
_COMMUNAL = "Lives in a communal establishment"


def _england_and_wales(level: str, area: str, age: str, residence: str, value: float):
    return {
        "observed_measure": {"source_concept": "ons.census2021_residents"},
        "geography": {"level": level, "id": area},
        "dimensions": {
            "C2021_AGE_24_NAME": age,
            "C2021_RESTYPE_3_NAME": residence,
            "C_SEX_NAME": "All persons",
        },
        "value": value,
    }


def _facts() -> list[dict]:
    facts = [
        _england_and_wales(
            "constituency", "E14000001", "Aged 20 to 24 years", _HOUSEHOLD, 600.0
        ),
        _england_and_wales(
            "constituency", "E14000001", "Aged 20 to 24 years", _COMMUNAL, 300.0
        ),
        _england_and_wales(
            "constituency", "E14000001", "Aged 25 to 29 years", _HOUSEHOLD, 100.0
        ),
        _england_and_wales(
            "local_authority", "E06000001", "Aged 70 to 74 years", _HOUSEHOLD, 99.0
        ),
        _england_and_wales(
            "local_authority", "E06000001", "Aged 75 to 79 years", _COMMUNAL, 1.0
        ),
        {
            "observed_measure": {
                "source_concept": "nisra.census2021_communal_residents"
            },
            "geography": {"level": "local_authority", "id": "N09000003"},
            "dimensions": {"Age": "20-29 years", "Sex": "All", "position": "resident"},
            "value": 50.0,
        },
    ]
    facts += [
        {
            "observed_measure": {"source_concept": "ons.mid_year_population_estimate"},
            "geography": {
                "level": "local_authority",
                "id": "N09000003",
                "vintage": "lgd_2014",
            },
            "period": {"type": "calendar_year", "value": 2025},
            "dimensions": {"age": age, "sex": "female"},
            "value": 100.0,
        }
        for age in range(20, 30)
    ]
    return facts


def test_private_household_shares_by_level_area_and_band() -> None:
    shares = private_household_shares(communal_cells(_facts()))

    assert shares[("constituency", "E14000001")]["20_30"] == pytest.approx(0.3)
    assert shares[("local_authority", "E06000001")]["70_80"] == pytest.approx(0.01)
    # Northern Ireland's district share is approximate: 50 communal residents
    # aged 20-29 over its 1,000 mid-2025 residents of those ages.
    assert shares[("local_authority", "N09000003")]["20_30"] == pytest.approx(0.05)
    # A band with no census count has no share, not a zero.
    assert "30_40" not in shares[("constituency", "E14000001")]


def test_the_line_filter_keeps_every_fact_the_cells_read() -> None:
    import json

    for fact in _facts():
        assert is_communal_fact_line(json.dumps(fact))
    other = {"observed_measure": {"source_concept": "hmrc.spi_income"}, "value": 1.0}
    assert not is_communal_fact_line(json.dumps(other))
