"""Markdown tables from a pilot run's ``summary.json`` and ``per_county.csv``.

Fetch a run's outputs from the volume, then render::

    modal volume get lwi-county-pilot run1 experiments/us-county-pilot-20261024/out/
    python experiments/us-county-pilot-20261024/report.py \
        experiments/us-county-pilot-20261024/out/run1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

FAMILY_LABELS = {
    "county_households": "Households (ACS B11001)",
    "county_population": "Population (PEP V2024)",
    "county_earnings": "Aggregate earnings (ACS B19061)",
    "cd_households": "District households (held)",
    "cd_population": "District population (held)",
    "county_earning_households": "Households with earnings (ACS B19051)",
    "county_soi_wages": "Wages (IRS SOI A00200)",
    "county_income": "Aggregate household income (ACS B19025)",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render(run: Path) -> str:
    summary = json.loads((run / "summary.json").read_text())
    per_county = pd.read_csv(run / "per_county.csv", dtype={"county_fips": str})
    lines = [
        f"Run `{run.name}`: {summary['households']:,} households, "
        f"{summary['counties']:,} counties, {summary['trained_rows']:,} trained rows, "
        f"{summary['settings']['epochs']} epochs, {summary['seconds'] / 60:.0f} min.",
        "",
        "| Family | Role | Median abs. error before | after | Within 10% before | after |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for family, label in FAMILY_LABELS.items():
        entry = summary["fit"].get(family)
        if entry is None:
            continue
        before, after = entry["before"], entry["after"]
        lines.append(
            f"| {label} | {entry['role']} | {pct(before['median_abs_rel_error'])} | "
            f"{pct(after['median_abs_rel_error'])} | {pct(before['within_10pct'])} | "
            f"{pct(after['within_10pct'])} |"
        )
    lines += [
        "",
        "Median absolute county error by county population (trained families):",
        "",
        "| Family | County population | Counties | Before | After |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family in ("county_households", "county_population", "county_earnings"):
        bands = summary["fit"][family]["median_abs_rel_error_by_county_population"]
        for band, entry in bands.items():
            lines.append(
                f"| {FAMILY_LABELS[family]} | {band} | {entry['counties']:,} | "
                f"{pct(entry['before'])} | {pct(entry['after'])} |"
            )
    grades = summary["reliability_grades"]
    lines += [
        "",
        "| Grade (county ESS) | Counties | Share of US population |",
        "| --- | ---: | ---: |",
    ]
    for label, floor in grades["thresholds_ess"].items():
        lines.append(
            f"| {label} (≥ {floor:g}) | {grades['counties'].get(label, 0):,} | "
            f"{pct(grades['population_share'].get(label, 0.0))} |"
        )
    q = summary["weight_ratio_quantiles"]
    lines += [
        "",
        f"Weight ratio (calibrated / release), quantiles 0.1% / 1% / 50% / 99% / 99.9%: "
        f"{q['0.001']:.2f} / {q['0.01']:.2f} / {q['0.5']:.2f} / {q['0.99']:.2f} / "
        f"{q['0.999']:.2f}.",
        f"Total Kish ESS: {summary['ess_total_before']:,.0f} before, "
        f"{summary['ess_total_after']:,.0f} after.",
        f"Counties with no household records: {summary['counties_without_records']}.",
        "Median absolute gap between the approximate weighted median household income "
        f"and ACS B19013: {pct(summary['median_income_vs_acs_b19013']['before'])} before, "
        f"{pct(summary['median_income_vs_acs_b19013']['after'])} after.",
    ]
    worst = (
        per_county.assign(
            households_error=lambda d: (
                d["households_after"] / d["households_target"] - 1
            )
        )
        .assign(abs_error=lambda d: d["households_error"].abs())
        .nlargest(5, "abs_error")
    )
    lines += ["", "Largest remaining county household errors:", ""]
    for _, row in worst.iterrows():
        lines.append(
            f"- {row['county_fips']}: {pct(row['households_error'])}, "
            f"{int(row['records']) if pd.notna(row['records']) else 0} records, "
            f"ESS {row['ess_after']:.0f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
