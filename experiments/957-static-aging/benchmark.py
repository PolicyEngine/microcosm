"""PR #957 integration benchmark, pinned to the original model/data comparison.

Run preflight, then one cell per process to release each simulation's memory.
All population statistics use MicroSeries methods; no derived weight variables.
Required inputs: --h5, --ssa and --output. Use --help for the command surface.
"""

# ruff: noqa: E402 -- parse --help and configure thread limits before runtime imports.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", choices=("preflight", "engine", "aging"))
    parser.add_argument("year", nargs="?", type=int)
    parser.add_argument(
        "--h5", type=Path, required=True, help="Certified populace_us_2024.h5 input"
    )
    parser.add_argument(
        "--ssa", type=Path, required=True, help="SSA SSPopJul_TR2024.csv input"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for aggregate receipts and logs",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Microcosm checkout supplying editable imports (default: this script's checkout)",
    )
    args = parser.parse_args()
    if args.path == "preflight" and args.year is not None:
        parser.error("preflight does not take a year")
    if args.path != "preflight" and (
        args.year not in (2024, 2025, 2030, 2035)
        or (args.path == "aging" and args.year == 2024)
    ):
        parser.error("use engine 2024 or engine/aging 2025, 2030, 2035")
    return args


# Process --help before importing the numerical runtime.
ARGS = parse_args() if __name__ == "__main__" else None

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from microdf import MicroSeries
from policyengine_core.errors.cycle_error import CycleError
from policyengine_us import CountryTaxBenefitSystem, Microsimulation
from policyengine_us.data import USSingleYearDataset
from policyengine_us.data.economic_assumptions import MICRODATA_UPRATING_OVERRIDES

from microcosm.calibrate import (
    SeriesProjection,
    ssa_population_projection,
    static_aging,
)
from microcosm.frame import US_SCHEMA, Frame, SignedScale, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import multi_year_dataset, uprating_series
from microcosm.frame.materialize import read_frame_table

ROOT: Path
REPO: Path
H5: Path
SSA: Path
BASE = 2024
YEARS = (2025, 2030, 2035)
AGE_BANDS = {80: 84}
EXPECTED_H5 = "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84"
EXPECTED_SSA = "cb6ab96eba35554e92e279e839662106602078ba7ff0dfaa306a8a0ca5a919e5"
SOURCE_FILES = {
    "static_aging.py": Path(inspect.getsourcefile(static_aging)).resolve(),
    "frame/scaling.py": Path(inspect.getsourcefile(SignedScale)).resolve(),
    "adapter/policyengine_us.py": Path(
        inspect.getsourcefile(multi_year_dataset)
    ).resolve(),
}


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


IMPORTED_SOURCE_HASHES = {name: sha(path) for name, path in SOURCE_FILES.items()}


def assert_sources_unchanged():
    assert {
        name: sha(path) for name, path in SOURCE_FILES.items()
    } == IMPORTED_SOURCE_HASHES, "Imported source files changed during this run"


def serialize_factor(factor):
    if isinstance(factor, SignedScale):
        return {"positive": factor.positive, "negative": factor.negative}
    return float(factor)


def write(name, value):
    target = ROOT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def provenance():
    assert md.version("policyengine-us") == "2.6.10"
    assert sha(H5) == EXPECTED_H5
    assert sha(SSA) == EXPECTED_SSA
    snapshots = ROOT / "runner-snapshots"
    snapshots.mkdir(exist_ok=True)
    (snapshots / (sha(__file__) + ".py")).write_bytes(Path(__file__).read_bytes())
    assert_sources_unchanged()
    source_snapshots = ROOT / "source-snapshots"
    source_snapshots.mkdir(exist_ok=True)
    for name, path in SOURCE_FILES.items():
        assert path.is_relative_to(REPO), f"Wrong source checkout: {path}"
        (source_snapshots / (IMPORTED_SOURCE_HASHES[name] + ".py")).write_bytes(
            path.read_bytes()
        )
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--binary", "HEAD"]
    )
    diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    (source_snapshots / (diff_sha256 + ".patch")).write_bytes(tracked_diff)
    return {
        "head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {
            p: md.version(p)
            for p in (
                "policyengine",
                "policyengine-us",
                "policyengine-core",
                "spm-calculator",
                "microdf-python",
                "numpy",
                "pandas",
                "torch",
            )
        },
        "h5_sha256": EXPECTED_H5,
        "h5_release": "populace-us-2024-spm-20260909",
        "ssa_sha256": EXPECTED_SSA,
        "runner_sha256": sha(__file__),
        "source_sha256": IMPORTED_SOURCE_HASHES,
        "source_paths": {name: str(path) for name, path in SOURCE_FILES.items()},
        "tracked_diff_sha256": diff_sha256,
        "tracked_diff_scope": "git diff --binary HEAD; untracked helper content recorded separately by source_sha256",
        "git_status": subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--short"], text=True
        ).splitlines(),
        "seed": 0,
        "epochs": 300,
        "max_weight_ratio": 5.0,
        "anchor": "frame",
        "ssa_age_bands": AGE_BANDS,
        "ssa_age_top": 85,
        "cps_age_coding_source": "https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar24.pdf#page=35",
        "runtime_scope": "PR integration benchmark: PE-US 2.6.10 and custom projected datasets, outside wrapper 6.0.0 model pin 2.2.1",
    }


def load_frame():
    with pd.HDFStore(H5, mode="r") as store:
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
    weight = tables["household"].pop("household_weight").to_numpy(dtype=float)
    assert not any(
        c.endswith("_weight") for entity, t in tables.items() for c in t.columns
    )
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weight, WeightKind.CALIBRATED)}
    )


def project(frame, year):
    system = CountryTaxBenefitSystem()
    columns = [
        c
        for entity in frame.entities
        for c in frame.table(entity)
        if c in system.variables
    ]
    totals, indices, mapping = uprating_series(columns, (BASE, year), system=system)
    demo = ssa_population_projection(SSA, age_top=85, age_bands=AGE_BANDS)
    assert frame.person["age"].max() <= 85
    result = static_aging(
        frame,
        base_year=BASE,
        years=(year,),
        demographics=demo,
        series=SeriesProjection(totals=totals, indices=indices),
        column_series=mapping,
        epochs=300,
        max_weight_ratio=5.0,
    )
    return result, totals, indices, mapping


def raw_total(frame, column):
    """Audit the Frame's stored inputs before engine input normalization."""
    entity = frame.column_entity(column)
    return float(
        MicroSeries(
            frame.table(entity)[column], weights=frame.resolve_weights(entity).values
        ).sum()
    )


def raw_components(frame, column):
    entity = frame.column_entity(column)
    values = MicroSeries(
        frame.table(entity)[column], weights=frame.resolve_weights(entity).values
    )
    return {
        "positive": float(values[values > 0].sum()),
        "negative": float(values[values < 0].sum()),
    }


def preflight():
    started = time.perf_counter()
    frame = load_frame()
    # Prove the adapter preserves every base table, value, row and dtype.
    exported = multi_year_dataset(frame, BASE, {}).datasets[BASE]
    native = USSingleYearDataset(file_path=str(H5), time_period=BASE)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            getattr(exported, entity), getattr(native, entity), check_exact=True
        )
    report = {
        "provenance": provenance(),
        "base_export_exact": True,
        "rows": {entity: frame.n(entity) for entity in frame.entities},
        "years": {},
    }
    for year in YEARS:
        result, totals, indices, mapping = project(frame, year)
        projected = result.frame_for(frame, year)
        projection = result.year(year)
        factors = projection.factors
        output = multi_year_dataset(
            frame, BASE, {year: (projection.weights.values, dict(factors))}
        ).datasets[year]
        output_tables = {
            entity: getattr(output, entity).copy() for entity in frame.entities
        }
        output_weights = output_tables["household"].pop("household_weight").to_numpy()
        np.testing.assert_array_equal(output_weights, projection.weights.values)
        output_frame = Frame(
            output_tables,
            frame.schema,
            {"household": Weights(output_weights, WeightKind.CALIBRATED)},
        )
        audit = []
        for column, name in mapping.items():
            entity = frame.column_entity(column)
            before = frame.table(entity)[column].to_numpy(dtype=float)
            after = projected.table(entity)[column].to_numpy(dtype=float)
            factor = factors[column]
            # Independent expected arrays: deliberately do not call apply_scale.
            expected = np.empty_like(before)
            negative = before < 0
            if isinstance(factor, SignedScale):
                expected[negative] = before[negative] * factor.negative
                expected[~negative] = before[~negative] * factor.positive
            else:
                expected[:] = before * float(factor)
            assert projected.table(entity)[column].dtype == np.dtype("float64")
            assert output_tables[entity][column].dtype == np.dtype("float64")
            np.testing.assert_allclose(
                after, expected, rtol=1e-12, atol=1e-9, err_msg=column
            )
            np.testing.assert_allclose(
                output_tables[entity][column],
                expected,
                rtol=1e-12,
                atol=1e-9,
                err_msg=column,
            )
            np.testing.assert_array_equal(
                np.sign(after), np.sign(before), err_msg=column
            )
            np.testing.assert_array_equal(
                np.sign(output_tables[entity][column]), np.sign(before), err_msg=column
            )
            base_total, year_total = (
                raw_total(frame, column),
                raw_total(output_frame, column),
            )
            values = totals.get(name, indices.get(name))
            growth = values[year] / values[BASE]
            if name in totals:
                np.testing.assert_allclose(
                    year_total, base_total * growth, rtol=1e-9, atol=1e-4
                )
                base_components = raw_components(frame, column)
                output_components = raw_components(output_frame, column)
                for sign in ("positive", "negative"):
                    np.testing.assert_allclose(
                        output_components[sign],
                        base_components[sign] * growth,
                        rtol=1e-9,
                        atol=1e-4,
                        err_msg=f"{column} {sign}",
                    )
                if (before > 0).any() and negative.any():
                    assert isinstance(factor, SignedScale), (
                        f"{column}: mixed-sign total requires SignedScale"
                    )
            else:
                assert not isinstance(factor, SignedScale), (
                    f"{column}: index must use a scalar"
                )
                np.testing.assert_allclose(factor, growth, rtol=1e-12)
                base_components = output_components = None
            audit.append(
                {
                    "column": column,
                    "series": name,
                    "kind": "total" if name in totals else "index",
                    "factor": serialize_factor(factor),
                    "series_growth": growth,
                    "base_total": base_total,
                    "projected_total": year_total,
                    "dataset_override": column in MICRODATA_UPRATING_OVERRIDES,
                    "base_gross_components": base_components,
                    "projected_gross_components": output_components,
                    "frame_dtype": str(projected.table(entity)[column].dtype),
                }
            )
        for entity in frame.entities:
            unchanged = [c for c in frame.table(entity) if c not in mapping]
            pd.testing.assert_frame_equal(
                frame.table(entity)[unchanged],
                projected.table(entity)[unchanged],
                check_exact=True,
            )
            pd.testing.assert_frame_equal(
                frame.table(entity)[unchanged],
                output_tables[entity][unchanged],
                check_exact=True,
            )
        fit = result.year(year).demographic_fit
        assert int((fit["base"] == 0).sum()) == 0, (
            "Certified CPS frame should support every pooled age/sex cell"
        )
        supported = fit["target"] > 0
        wmape = float(
            (fit.loc[supported, "achieved"] - fit.loc[supported, "target"]).abs().sum()
            / fit.loc[supported, "target"].sum()
        )
        weight_ratios = (
            result.year(year).weights.values / frame.weights_for("household").values
        )
        assert (
            np.isfinite(weight_ratios).all()
            and weight_ratios.min() >= 0
            and weight_ratios.max() <= 5 + 1e-7
        )
        fit.to_csv(ROOT / f"demographic-fit-{year}.csv", index=False)
        assert_sources_unchanged()
        report["years"][year] = {
            "demographic_wmape": wmape,
            "unsupported_cells": int((fit["base"] == 0).sum()),
            "weight_ratio_min": float(weight_ratios.min()),
            "weight_ratio_max": float(weight_ratios.max()),
            "audited_columns": audit,
        }
        write("preflight.json", report)
        print(
            f"preflight {year}: {len(audit)} mapped columns passed; cell WMAPE {wmape:.6%}",
            flush=True,
        )
    report["seconds"] = time.perf_counter() - started
    report["complete"] = True
    assert_sources_unchanged()
    write("preflight.json", report)


def cell(path, year):
    started = time.perf_counter()
    receipt = provenance()
    if path == "engine":
        sim = Microsimulation(dataset=str(H5))
    else:
        frame = load_frame()
        result, _, _, _ = project(frame, year)
        p = result.year(year)
        dataset = multi_year_dataset(
            frame, BASE, {year: (p.weights.values, dict(p.factors))}
        )
        sim = Microsimulation(dataset=dataset)
    print(
        f"{path} {year}: simulation initialized at {time.perf_counter() - started:.1f}s",
        flush=True,
    )

    def calc(variable, entity="person"):
        result = sim.calculate(variable, year, map_to=entity)
        assert isinstance(result, MicroSeries)
        return result

    age = calc("age")
    wages = calc("employment_income_before_lsr")
    ss = calc("social_security_retirement")
    row = {
        "path": path,
        "year": year,
        "population_m": float(age.count()) / 1e6,
        "share_65plus": float((age >= 65).mean()),
        "employment_income_b": float(wages.sum()) / 1e9,
        "ss_retirement_b": float(ss.sum()) / 1e9,
        "ss_recipients_m": float((ss > 0).sum()) / 1e6,
        "ss_per_recipient": float(ss[ss > 0].mean()),
        "partnership_income_b": float(calc("partnership_income").sum()) / 1e9,
        "s_corp_income_b": float(calc("s_corp_income").sum()) / 1e9,
        "partnership_self_employment_net_earnings_b": float(
            calc("partnership_self_employment_net_earnings").sum()
        )
        / 1e9,
        "poverty_rate": None,
        "poverty_65plus": None,
        "poverty_child": None,
        "income_tax_b": None,
        "poverty_error": None,
    }
    write(f"{path}-{year}.json", {"provenance": receipt, "row": row, "complete": False})
    try:
        poor = calc("spm_unit_is_in_spm_poverty")
        row.update(
            poverty_rate=float(poor.mean()),
            poverty_65plus=float(poor[age >= 65].mean()),
            poverty_child=float(poor[age < 18].mean()),
        )
    except CycleError as exc:
        row["poverty_error"] = f"{type(exc).__name__}: {exc}"
        (ROOT / f"{path}-{year}-poverty-error.txt").write_text(traceback.format_exc())
        print(
            f"{path} {year}: poverty raised CycleError; preserved full trace",
            flush=True,
        )
    # Avoid contaminated caches after a recursive formula error.
    if row["poverty_error"] is None:
        row["income_tax_b"] = float(calc("income_tax", "tax_unit").sum()) / 1e9
    row["seconds"] = time.perf_counter() - started
    assert_sources_unchanged()
    write(f"{path}-{year}.json", {"provenance": receipt, "row": row, "complete": True})
    print(
        json.dumps(
            {k: v for k, v in row.items() if k != "poverty_error"}, allow_nan=False
        ),
        flush=True,
    )


def main():
    global ROOT, REPO, H5, SSA
    args = ARGS if ARGS is not None else parse_args()
    ROOT, REPO, H5, SSA = (
        path.expanduser().resolve()
        for path in (args.output, args.repo, args.h5, args.ssa)
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    # Validate pins before any calibration or simulation.
    assert md.version("policyengine-us") == "2.6.10"
    assert sha(H5) == EXPECTED_H5
    assert sha(SSA) == EXPECTED_SSA
    if args.path == "preflight":
        preflight()
    else:
        cell(args.path, args.year)


if __name__ == "__main__":
    main()
