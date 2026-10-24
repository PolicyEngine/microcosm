"""Markdown tables from a pilot run's ``summary.json`` and ``per_county.csv``.

Fetch a run's outputs from the volume, then render::

    modal volume get lwi-county-pilot run1 experiments/us-county-pilot-20261024/out/
    python experiments/us-county-pilot-20261024/report.py \
        experiments/us-county-pilot-20261024/out/run1

When the run folder also holds ``weights.npy`` and the extract's
``households.parquet`` is given as a second argument, the report adds the
weight share and Kish ESS of each household spine, nationally and in the
largest counties.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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


def kish(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (weights**2).sum())


def spine_section(run: Path, households_path: Path, largest: int = 5) -> list[str]:
    households = pd.read_parquet(households_path)
    before = households["weight"].to_numpy(np.float64)
    after = np.load(run / "weights.npy")
    spine = households["household_spine"].astype(str).to_numpy()
    county = households["county_fips_2023"].astype(str).str.zfill(5).to_numpy()
    lines = [
        "",
        "| Spine | Records | Weight share before | after | Kish ESS before | after |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(set(spine)):
        mask = spine == name
        lines.append(
            f"| {name} | {int(mask.sum()):,} | {pct(before[mask].sum() / before.sum())} | "
            f"{pct(after[mask].sum() / after.sum())} | {kish(before[mask]):,.0f} | "
            f"{kish(after[mask]):,.0f} |"
        )
    totals = pd.Series(before).groupby(county).sum().nlargest(largest).index
    lines += [
        "",
        "| County | Spine | Records | Weight share after | Kish ESS after |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for code in totals:
        in_county = county == code
        for name in sorted(set(spine[in_county])):
            mask = in_county & (spine == name)
            lines.append(
                f"| {code} | {name} | {int(mask.sum()):,} | "
                f"{pct(after[mask].sum() / after[in_county].sum())} | {kish(after[mask]):,.0f} |"
            )
    return lines


def render(run: Path, households_path: Path | None = None) -> str:
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
    if households_path is not None and (run / "weights.npy").exists():
        lines += spine_section(run, households_path)
    return "\n".join(lines)


if __name__ == "__main__":
    print(
        render(
            Path(sys.argv[1]),
            Path(sys.argv[2]) if len(sys.argv) > 2 else None,
        )
    )
