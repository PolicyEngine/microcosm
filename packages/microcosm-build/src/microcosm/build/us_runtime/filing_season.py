"""Compare IRS returns at the same filing-season processing week in two years."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from math import isfinite
from typing import Any


def filing_season_comparison(
    facts: Iterable[Mapping[str, Any]], *, first_year: int, last_year: int, week: int
) -> dict[str, Any]:
    """Compare all published measures and AGI bands, retaining one-year-only cells.

    Chronicle's partial-cohort facts carry the processing cutoff and week as
    dimensions. Totals cover returns processed through the specified week.
    """
    if first_year >= last_year or not 1 <= week <= 53:
        raise ValueError(
            "Pass increasing tax years and a processing week from 1 to 53."
        )
    years = (first_year, last_year)
    cells: dict[tuple[str, str], dict[int, int | float]] = {}
    units: dict[str, str] = {}
    cutoffs: dict[int, str] = {}
    for fact in facts:
        dims = fact.get("dimensions", {})
        measure = fact.get("observed_measure", {})
        period = fact.get("period", {})
        if (
            measure.get("source_name") != "irs_soi"
            or period.get("type") != "tax_year"
            or str(dims.get("filing_season_week")) != str(week)
            or fact.get("geography", {}).get("id") != "0100000US"
        ):
            continue
        year = int(period["value"])
        if year not in years:
            continue
        cutoff = date.fromisoformat(dims["processing_cutoff"])
        if (
            cutoff.year != year + 1
            or cutoffs.setdefault(year, cutoff.isoformat()) != cutoff.isoformat()
        ):
            raise ValueError(
                f"Inconsistent processing cutoff for TY{year}, week {week}."
            )
        measure_id = measure["source_measure_id"]
        unit = measure["unit"]
        if units.setdefault(measure_id, unit) != unit:
            raise ValueError(f"Inconsistent units for {measure_id}.")
        income_range = dims["income_range"]
        value = fact["value"]
        if type(value) not in (float, int) or not isfinite(value):
            raise ValueError(f"Non-numeric observation for {measure_id} in TY{year}.")
        cell = cells.setdefault((income_range, measure_id), {})
        if year in cell:
            raise ValueError(
                f"Duplicate observation for {(year, income_range, measure_id)}."
            )
        cell[year] = value
    if set(cutoffs) != set(years):
        raise ValueError(f"Both tax years must have facts for processing week {week}.")

    rows = []
    for (income_range, measure), values in sorted(cells.items()):
        first, last = (values.get(year) for year in years)
        row = {
            "income_range": income_range,
            "measure": measure,
            "unit": units[measure],
            "first": first,
            "last": last,
            "difference": last - first
            if first is not None and last is not None
            else None,
        }
        if measure == "total_itemized_deductions_returns":
            denominators = cells.get((income_range, "return_count"), {})
            rates = []
            for year in years:
                count, total = values.get(year), denominators.get(year)
                if count is None:
                    rates.append(None)
                    continue
                if total is None or not 0 <= count <= total:
                    raise ValueError(
                        f"Missing or inconsistent all-return count for {(year, income_range)}."
                    )
                rates.append(100 * count / total if total else None)
            row.update(
                first_percent_itemizing=rates[0],
                last_percent_itemizing=rates[1],
                change_percentage_points=(
                    rates[1] - rates[0] if None not in rates else None
                ),
            )
        rows.append(row)
    return {
        "filing_season_week": week,
        "first_tax_year": first_year,
        "first_processing_cutoff": cutoffs[first_year],
        "last_tax_year": last_year,
        "last_processing_cutoff": cutoffs[last_year],
        "population": "Current-tax-year returns first processed through the stated cutoff",
        "rows": rows,
    }
