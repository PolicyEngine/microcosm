"""Pensioner taxpayer measures on a Microcosm UK national H5 (microcosm#1069).

For one H5 at the calibration year, on the measures the bound targets read:

- State Pension recipients (``state_pension > 0``, the DWP and DfC rows'
  measure, and ``state_pension_annual > 0``, the SPI band rows' measure),
  split into those with ``income_tax_before_winter_fuel_payment_charge_annual
  > 0`` and the rest;
- Income Tax payers of State Pension age (``is_SP_age``) and aged 65 and over
  in each of the twelve region-tier areas, the ``hmrc_itl_age`` rows.

Weights are each person's household weight, as ``person_count`` mapped to the
household reads them. Writes one JSON of aggregates; no record leaves the run.

    .venv/bin/python docs/evidence/uk-pensioner-taxpayers-1069/scripts/measure_pensioner_taxpayers.py \
        --h5 <national.h5> --label candidate --output <receipt.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TAX = "income_tax_before_winter_fuel_payment_charge_annual"


def weighted_partition(recipient, taxpayer, weight) -> dict[str, float]:
    """Split the weighted recipients into taxpayers and the rest.

    The two masks ``recipient & taxpayer`` and ``recipient & ~taxpayer`` are
    disjoint and cover ``recipient``, so taxpaying + non-taxpaying equals the
    recipients up to floating-point summation.
    """

    recipient = np.asarray(recipient, dtype=bool)
    taxpayer = np.asarray(taxpayer, dtype=bool)
    weight = np.asarray(weight, dtype=float)
    if not (recipient.shape == taxpayer.shape == weight.shape):
        raise ValueError("recipient, taxpayer and weight must align.")
    if np.any(weight < 0) or not np.all(np.isfinite(weight)):
        raise ValueError("weights must be finite and non-negative.")
    return {
        "recipients": float(weight[recipient].sum()),
        "taxpaying": float(weight[recipient & taxpayer].sum()),
        "non_taxpaying": float(weight[recipient & ~taxpayer].sum()),
    }


def _enum_names(values) -> np.ndarray:
    names = np.asarray(values).astype(str)
    return np.array([name.rsplit(".", 1)[-1] for name in names])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(h5: Path, *, year: int, scratch: Path, current_income_reason: str | None):
    from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
    from microcosm.calibrate.geography_constants import (
        UK_REGION_TIER,
        UK_REGION_TIER_ENUM,
    )

    resolver = UKMeasureResolver(
        simulation_source=h5, scratch_dir=scratch, year=year, frame=None
    )
    if current_income_reason:
        resolver.use_current_income_for_annual_measures(reason=current_income_reason)

    def person(variable: str) -> np.ndarray:
        values, _route = resolver.compute("person", variable)
        return np.asarray(values)

    weight = person("household_weight").astype(float)
    tax = person(TAX).astype(float)
    taxpayer = tax > 0
    state_pension = person("state_pension").astype(float)
    state_pension_annual = person("state_pension_annual").astype(float)
    sp_age = person("is_SP_age").astype(bool)
    age = person("age").astype(float)
    region = _enum_names(person("region"))

    partitions = {
        "state_pension": weighted_partition(state_pension > 0, taxpayer, weight),
        "state_pension_annual": weighted_partition(
            state_pension_annual > 0, taxpayer, weight
        ),
    }
    for name, row in partitions.items():
        gap = row["taxpaying"] + row["non_taxpaying"] - row["recipients"]
        if abs(gap) > 1e-9 * max(row["recipients"], 1.0):
            raise AssertionError(f"{name}: partition gap {gap}")

    areas = {}
    for _level, code in UK_REGION_TIER:
        in_area = region == UK_REGION_TIER_ENUM[code]
        areas[code] = {
            "state_pension_age": float(weight[in_area & taxpayer & sp_age].sum()),
            "age_65_and_over": float(weight[in_area & taxpayer & (age >= 65)].sum()),
        }
    unassigned = ~np.isin(region, list(UK_REGION_TIER_ENUM.values()))
    return {
        "year": year,
        "people": int(weight.size),
        "weighted_people": float(weight.sum()),
        "people_outside_region_tier": int(unassigned.sum()),
        "state_pension_recipients": partitions,
        "taxpayers_by_area": areas,
        "taxpayers_region_tier_total": {
            measure: float(sum(row[measure] for row in areas.values()))
            for measure in ("state_pension_age", "age_65_and_over")
        },
        "state_pension_age_taxpayers_without_state_pension": float(
            weight[taxpayer & sp_age & ~(state_pension > 0)].sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--current-income-for-annual-measures",
        metavar="REASON",
        help="For a file without annual income descriptors (the incumbent).",
    )
    args = parser.parse_args()
    scratch = args.output.parent / f"scratch-{args.label}"
    result = measure(
        args.h5,
        year=args.year,
        scratch=scratch,
        current_income_reason=args.current_income_for_annual_measures,
    )
    receipt = {
        "label": args.label,
        "h5": args.h5.name,
        "h5_sha256": _sha256(args.h5),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["state_pension_recipients"], indent=1))
    print(json.dumps(receipt["taxpayers_region_tier_total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
