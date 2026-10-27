"""Score a UK national H5 on both definitions of the SPI pension band rows.

Main binds the four SPI 2023-24 Table 3.6 pension rows at calendar 2025: the
values projected by Table 2.5 growth, measured on 2025 incomes. The pilot
(microcosm#1413) binds the published 2023-24 values, measured on the dated
``spi_2023_*`` measures. This script measures one H5 both ways, so each build
is also scored on the definition it was not fitted to:

    .venv/bin/python docs/evidence/uk-spi-dated-bands/scripts/spi_cross_fit.py \
        --h5 <national.h5> --label <baseline|pilot> \
        --targets-2025 <baseline calibration_diagnostics.json> \
        --targets-2023 <pilot calibration_diagnostics.json> \
        --native <2025|2023> --output <json>

Weights are each person's household weight, as ``person_count`` mapped to
the household reads them. A band is ``[lower, upper)`` on total income, as
the materializer reads a band without declared semantics. On the definition
the file was fitted to (``--native``) its own diagnostics carry the solver's
``final_estimate``; the report gives the largest gap between that and this
script's reading, as a check on the script. Writes one JSON of aggregates; no record leaves the run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

FAMILIES = {
    "hmrc/state_pension_count_income_band_": ("count", "state_pension"),
    "hmrc/state_pension_income_band_": ("amount", "state_pension"),
    "hmrc/private_pension_income_count_income_band_": ("count", "private_pension"),
    "hmrc/private_pension_income_income_band_": ("amount", "private_pension"),
}
_BAND = re.compile(r"(\d[\d_]*)_to_(\d[\d_]*|inf)$")
TAX = "income_tax_before_winter_fuel_payment_charge_annual"
CURRENT = {
    "total_income": "total_income_after_loss_relief_annual",
    "tax": TAX,
    "state_pension": "state_pension_annual",
    "private_pension": "private_pension_income_annual",
}
DATED = {
    "total_income": "spi_2023_total_income",
    "tax": "spi_2023_income_tax",
    "state_pension": "spi_2023_state_pension",
    "private_pension": "spi_2023_private_pension_income",
}


def band_of(name: str) -> tuple[float, float]:
    match = _BAND.search(name)
    if match is None:
        raise ValueError(f"no band in {name!r}")
    lower = float(match.group(1).replace("_", ""))
    upper = match.group(2)
    return lower, math.inf if upper == "inf" else float(upper.replace("_", ""))


def family_of(name: str) -> tuple[str, tuple[str, str]] | None:
    for prefix, family in FAMILIES.items():
        if name.startswith(prefix) and _BAND.search(name[len(prefix) :]):
            return prefix, family
    return None


def targets(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    for row in json.loads(path.read_text())["targets"]:
        name = str(row["target_name"])
        if family_of(name) is not None:
            if name in rows:
                raise ValueError(f"{name} appears twice in {path}")
            rows[name] = {
                "target": float(row["target"]),
                "estimate": float(row["final_estimate"]),
            }
    return rows


def band_estimate(
    columns: dict[str, np.ndarray],
    weight: np.ndarray,
    *,
    kind: str,
    income: str,
    band: tuple[float, float],
) -> float:
    lower, upper = band
    total = columns["total_income"]
    mask = (columns["tax"] > 0) & (total >= lower) & (total < upper)
    value = columns[income]
    if kind == "count":
        mask &= value > 0
        return math.fsum((weight * columns["count"])[mask].tolist())
    return math.fsum((weight * value)[mask].tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--targets-2025", required=True, type=Path)
    parser.add_argument("--targets-2023", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native", required=True, choices=("2025", "2023"))
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver

    resolver = UKMeasureResolver(
        simulation_source=args.h5,
        scratch_dir=args.output.parent / f"scratch-cross-{args.label}",
        year=args.year,
        frame=None,
    )

    def person(variable: str) -> np.ndarray:
        values, _route = resolver.compute("person", variable)
        return np.asarray(values, dtype=float)

    weight = person("household_weight")
    current = {key: person(variable) for key, variable in CURRENT.items()}
    current["count"] = np.ones_like(weight)
    dated = {key: person(variable) for key, variable in DATED.items()}
    dated["count"] = person("spi_2023_person_count")

    by_definition = {
        "2025": targets(args.targets_2025),
        "2023": targets(args.targets_2023),
    }
    names = sorted(set(by_definition["2025"]) | set(by_definition["2023"]))
    rows, families, native_gaps = [], {}, []
    for name in names:
        prefix, (kind, income) = family_of(name)
        band = band_of(name[len(prefix) :])
        entry = {
            "name": name,
            "kind": kind,
            "income": income,
            "band": [band[0], band[1]],
        }
        for year, columns in (("2025", current), ("2023", dated)):
            target = by_definition[year].get(name)
            if target is None:
                continue
            estimate = band_estimate(
                columns, weight, kind=kind, income=income, band=band
            )
            entry[year] = {
                "target": target["target"],
                "estimate": estimate,
                "relative_error": estimate / target["target"] - 1,
            }
            if year == args.native:
                gap = (
                    abs(estimate / target["estimate"] - 1)
                    if target["estimate"]
                    else 0.0
                )
                native_gaps.append(gap)
            family = families.setdefault(
                f"{kind}:{income}:{year}",
                {"target": 0.0, "estimate": 0.0, "abs_errors": []},
            )
            family["target"] += target["target"]
            family["estimate"] += estimate
            family["abs_errors"].append(abs(estimate / target["target"] - 1))
        rows.append(entry)
    summary = {
        key: {
            "target": family["target"],
            "estimate": family["estimate"],
            "total_relative_error": family["estimate"] / family["target"] - 1,
            "mean_abs_relative_error": math.fsum(family["abs_errors"])
            / len(family["abs_errors"]),
            "rows": len(family["abs_errors"]),
        }
        for key, family in sorted(families.items())
    }
    report = {
        "label": args.label,
        "h5": str(args.h5),
        "year": args.year,
        "native": args.native,
        "native_largest_relative_gap": max(native_gaps, default=None),
        "families": summary,
        "rows": rows,
        "receipt": resolver.receipt().get("spi_dated_measures"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
