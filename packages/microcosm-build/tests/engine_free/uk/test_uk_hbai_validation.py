"""The HBAI validation scorecard: report-only, against DWP's 95% intervals (microcosm#731)."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.hbai_validation import (
    UK_HBAI_VALIDATION_PERIOD,
    UK_HBAI_VALIDATION_STATISTICS,
    classify_uk_hbai_statistics,
    uk_hbai_validation_references,
)

#: Fixture estimates and intervals, shaped as the feed carries them.
_FIXTURE = {
    "median_equivalised_income_bhc_gbp_week": (700.0, 690.0, 710.0, "gbp_per_week"),
    "median_equivalised_income_ahc_gbp_week": (600.0, 590.0, 610.0, "gbp_per_week"),
    "relative_low_income_bhc_pct": (16.0, 15.0, 17.0, "percent"),
    "relative_low_income_ahc_pct": (21.0, 20.0, 22.0, "percent"),
}


def _fact(record_set: str, measure: str, value: float, unit: str) -> dict:
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{record_set}:{measure}",
        "source_release_key": "ledger.source_release.v1:hbai-fye2025",
        "period": {"type": "fiscal_year", "value": 2024},
        "observed_measure": {"unit": unit},
        "layout": {"record_set_id": record_set, "measure_id": measure},
        "value": value,
    }


def _facts(**overrides) -> list[dict]:
    rows = []
    for statistic, (kind, housing, measure) in UK_HBAI_VALIDATION_STATISTICS.items():
        estimate, lower, upper, unit = overrides.get(statistic, _FIXTURE[statistic])
        base = f"dwp.hbai_fye2025.{kind}.{housing}.fy2024"
        bounds = f"dwp.hbai_fye2025.{kind}_ci95.{housing}.fy2024"
        rows += [
            _fact(base, measure, estimate, unit),
            _fact(bounds, f"{measure}_ci_95_lower_bound", lower, unit),
            _fact(bounds, f"{measure}_ci_95_upper_bound", upper, unit),
        ]
    # A later year of the same series is not FYE 2025.
    rows.append(
        _fact(
            "dwp.hbai_fye2025.median_income.before_housing_costs.fy2023",
            "median_equivalised_household_income",
            650.0,
            "gbp_per_week",
        )
    )
    return rows


def test_the_comparison_year_is_frs_2024_25():
    # Period 2024 is FRS FY2024/25, HBAI's FYE 2025, not FYE 2024.
    assert UK_HBAI_VALIDATION_PERIOD == 2024
    assert set(UK_HBAI_VALIDATION_STATISTICS) == set(_FIXTURE)


def test_references_take_each_estimate_and_its_interval():
    references = uk_hbai_validation_references(_facts())
    bhc = references["median_equivalised_income_bhc_gbp_week"]
    assert (bhc["estimate"], bhc["ci_95_lower"], bhc["ci_95_upper"]) == (
        700.0,
        690.0,
        710.0,
    )
    assert bhc["unit"] == "gbp_per_week"
    assert bhc["source_release_key"] == "ledger.source_release.v1:hbai-fye2025"
    assert bhc["period"] == {"type": "fiscal_year", "value": 2024}
    assert len(bhc["fact_keys"]) == 3


def test_a_statistic_on_a_bound_is_inside_and_one_past_it_is_not():
    references = uk_hbai_validation_references(_facts())
    report = classify_uk_hbai_statistics(
        {
            "median_equivalised_income_bhc_gbp_week": 710.0,
            "median_equivalised_income_ahc_gbp_week": 589.0,
            "relative_low_income_bhc_pct": 15.0,
            "relative_low_income_ahc_pct": 22.5,
        },
        references,
    )
    assert {name: row["inside_ci_95"] for name, row in report.items()} == {
        "median_equivalised_income_bhc_gbp_week": True,
        "median_equivalised_income_ahc_gbp_week": False,
        "relative_low_income_bhc_pct": True,
        "relative_low_income_ahc_pct": False,
    }
    ahc = report["median_equivalised_income_ahc_gbp_week"]
    assert ahc["difference"] == pytest.approx(-11.0)
    assert ahc["relative_difference"] == pytest.approx(-11.0 / 600.0)
    assert ahc["ci_95"] == [590.0, 610.0]


def test_a_missing_statistic_or_fact_is_refused():
    references = uk_hbai_validation_references(_facts())
    with pytest.raises(ValueError, match="missing"):
        classify_uk_hbai_statistics({"relative_low_income_bhc_pct": 16.0}, references)
    with pytest.raises(ValueError, match="not finite"):
        classify_uk_hbai_statistics(
            {statistic: float("nan") for statistic in _FIXTURE}, references
        )
    facts = [
        row
        for row in _facts()
        if row["layout"]["measure_id"] != "relative_low_income_share_ci_95_upper_bound"
    ]
    with pytest.raises(ValueError, match="carries 0"):
        uk_hbai_validation_references(facts)
    with pytest.raises(ValueError, match="carries 2"):
        uk_hbai_validation_references([*_facts(), _facts()[0]])


def test_an_interval_that_does_not_hold_its_estimate_is_refused():
    with pytest.raises(ValueError, match="does not hold"):
        uk_hbai_validation_references(
            _facts(relative_low_income_bhc_pct=(18.0, 15.0, 17.0, "percent"))
        )
