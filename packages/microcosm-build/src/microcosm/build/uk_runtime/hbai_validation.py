"""HBAI validation of a UK dataset against DWP's 95% intervals (microcosm#731, uk-data#487).

Report-only (María's rulings of 2026-10-16 and 2026-10-20): four statistics,
the median equivalised household income and the share of individuals in
relative low income (below 60 percent of the contemporary median), each before
and after housing costs, are measured on a dataset and set beside DWP's
Households Below Average Income estimate for FYE 2025 and the 95 percent
confidence interval DWP publishes for it. A statistic outside its interval is
reported; it never binds and never fails a build.

Microcosm's period 2024 is FRS FY2024/25, which HBAI calls FYE 2025
(``docs/uk-uc-paid-target-contract.md``), so the dataset is measured at engine
year 2024. The estimates and intervals are read from the pinned Chronicle feed
(PolicyEngine/chronicle#359) with the release each was read from: DWP revises
FYE 2022 to FYE 2025 in March 2027, so a report names its vintage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

#: Engine year of the comparison: FRS FY2024/25, HBAI FYE 2025.
UK_HBAI_VALIDATION_PERIOD = 2024
_RECORD_SET_PREFIX = "dwp.hbai_fye2025"
_RECORD_SET_YEAR = "fy2024"
#: Each statistic: the record-set kind, the housing-cost basis and the measure.
UK_HBAI_VALIDATION_STATISTICS: Mapping[str, tuple[str, str, str]] = {
    "median_equivalised_income_bhc_gbp_week": (
        "median_income",
        "before_housing_costs",
        "median_equivalised_household_income",
    ),
    "median_equivalised_income_ahc_gbp_week": (
        "median_income",
        "after_housing_costs",
        "median_equivalised_household_income",
    ),
    "relative_low_income_bhc_pct": (
        "relative_low_income",
        "before_housing_costs",
        "relative_low_income_share",
    ),
    "relative_low_income_ahc_pct": (
        "relative_low_income",
        "after_housing_costs",
        "relative_low_income_share",
    ),
}
_BOUNDS = ("ci_95_lower_bound", "ci_95_upper_bound")


def _record_set(kind: str, housing: str) -> str:
    return f"{_RECORD_SET_PREFIX}.{kind}.{housing}.{_RECORD_SET_YEAR}"


def _one(
    facts: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    record_set: str,
    measure: str,
) -> Mapping[str, Any]:
    rows = facts.get((record_set, measure), [])
    if len(rows) != 1:
        raise ValueError(
            f"HBAI validation needs one fact for {record_set} {measure}, the feed "
            f"carries {len(rows)}."
        )
    return rows[0]


def uk_hbai_validation_references(
    facts: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """The four statistics' FYE 2025 estimates and 95% intervals from feed facts.

    Refuses a statistic whose estimate or either bound is missing or repeated,
    and an interval that does not hold its estimate.
    """

    wanted = {
        _record_set(kind, housing)
        for kind, housing, _ in UK_HBAI_VALIDATION_STATISTICS.values()
    } | {
        _record_set(f"{kind}_ci95", housing)
        for kind, housing, _ in UK_HBAI_VALIDATION_STATISTICS.values()
    }
    selected: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for fact in facts:
        layout = fact.get("layout") or {}
        record_set = layout.get("record_set_id")
        if record_set in wanted:
            key = (str(record_set), str(layout.get("measure_id")))
            selected.setdefault(key, []).append(fact)
    references: dict[str, dict[str, Any]] = {}
    for statistic, (kind, housing, measure) in UK_HBAI_VALIDATION_STATISTICS.items():
        estimate = _one(selected, _record_set(kind, housing), measure)
        lower, upper = (
            _one(selected, _record_set(f"{kind}_ci95", housing), f"{measure}_{bound}")
            for bound in _BOUNDS
        )
        value, low, high = (float(fact["value"]) for fact in (estimate, lower, upper))
        if not (math.isfinite(low) and math.isfinite(high) and low <= value <= high):
            raise ValueError(
                f"HBAI validation: the interval for {statistic} does not hold its "
                "estimate."
            )
        references[statistic] = {
            "estimate": value,
            "ci_95_lower": low,
            "ci_95_upper": high,
            "unit": str((estimate.get("observed_measure") or {}).get("unit", "")),
            "period": dict(estimate.get("period") or {}),
            "record_set_id": _record_set(kind, housing),
            "source_release_key": str(estimate.get("source_release_key", "")),
            "fact_keys": [
                str(fact.get("aggregate_fact_key", ""))
                for fact in (estimate, lower, upper)
            ],
        }
    return references


def classify_uk_hbai_statistics(
    model: Mapping[str, float],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Each measured statistic beside its HBAI estimate, inside its interval or not.

    ``model`` carries the four statistics in the references' units (pounds a
    week for the medians, percent of individuals for the shares). The interval
    is closed: a statistic on a bound is inside. Nothing here binds or fails;
    a missing or non-finite statistic is refused, since a scorecard that drops
    a row reads as one that passed it.
    """

    missing = sorted(set(UK_HBAI_VALIDATION_STATISTICS) - set(model))
    if missing:
        raise ValueError(f"HBAI validation needs every statistic; missing {missing}.")
    report: dict[str, dict[str, Any]] = {}
    for statistic in UK_HBAI_VALIDATION_STATISTICS:
        value = float(model[statistic])
        if not math.isfinite(value):
            raise ValueError(f"HBAI validation: {statistic} is not finite.")
        reference = references[statistic]
        low, high = float(reference["ci_95_lower"]), float(reference["ci_95_upper"])
        estimate = float(reference["estimate"])
        report[statistic] = {
            "model": value,
            "hbai": estimate,
            "ci_95": [low, high],
            "inside_ci_95": low <= value <= high,
            "difference": value - estimate,
            "relative_difference": value / estimate - 1.0 if estimate else None,
            "unit": reference["unit"],
            "source_release_key": reference["source_release_key"],
        }
    return report
