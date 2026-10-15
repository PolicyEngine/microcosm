#!/usr/bin/env python3
"""Compare gross IRA contributions and model-derived deductions on US H5 files.

Run after rebuilding and calibrating a candidate with the documented Route A
driver, using a release environment whose PolicyEngine-US includes #10032:

    ~/reviews/us-hub/scripts/heavy_run.sh ira-participation-validation \
      uv run --no-sync python tools/validate_us_ira_participation.py \
      --before-h5 /path/to/incumbent.h5 --after-h5 /path/to/candidate.h5 \
      --year 2025 --maximum-microsim-batch-size 2000 \
      --out /path/to/untracked-evidence/ira-validation.json

Both artifacts are rescored under the same installed model through the release
builder's household-batched scorer. Both must carry base _time_period 2024;
annual artifacts already aged to another year are refused. The output is diagnostic evidence, not
release certification. Source benchmarks are kept separate: gross contributions
use Form 5498 accumulation statistics, while deductions use SOI IRA payments.
The 2022/2023 benchmarks are cross-year comparisons when another year is scored.
This command does not publish or change either input artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

REQUIRED_VARIABLES = (
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "traditional_ira_deduction",
    "dependent_traditional_ira_deduction",
    "ira_active_participant",
    "tax_unit_is_filer",
)
SCORER_BASE_PERIOD = 2024


def _read_base_period(path: Path) -> int:
    """Read only the H5's small period metadata before constructing an engine."""
    import pandas as pd

    with pd.HDFStore(path, mode="r") as store:
        if "/_time_period" not in store.keys():
            raise ValueError(f"{path}: missing _time_period metadata")
        values = np.asarray(store.get("_time_period")).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{path}: _time_period must contain exactly one base year")
    value = values[0]
    try:
        period = int(value)
        valid = not isinstance(value, (bool, np.bool_)) and float(value) == period
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{path}: _time_period must be an integer base year")
    return period


def _load_builder():
    name = "ira_validation_release_builder"
    path = Path(__file__).with_name("build_us_fiscal_refresh_release.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load maintained release scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _model_system():
    from policyengine_us import Microsimulation

    return Microsimulation.default_tax_benefit_system()


def _require_model_variables(system) -> None:
    missing = sorted(set(REQUIRED_VARIABLES) - set(system.variables))
    if missing:
        raise RuntimeError(
            "The installed PolicyEngine-US cannot validate the #10032 IRA "
            f"phase-out: missing variables {', '.join(missing)}. "
            "Use a reviewed release environment with the merged #10032 model; "
            "an older uv.lock cannot supply these deduction calculations."
        )


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in (
        "policyengine-us",
        "policyengine-core",
        "spm-calculator",
        "microcosm-build",
        "numpy",
        "pandas",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _arrays(result, *, name: str, reference_weights=None):
    values = np.asarray(result.values, dtype=np.float64)
    weights = np.asarray(result.weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError(f"{name}: one aligned entity weight is required per value")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError(f"{name}: values and weights must be finite")
    if (values < 0).any() or (weights < 0).any():
        raise ValueError(f"{name}: values and weights must be nonnegative")
    if reference_weights is not None and not np.array_equal(weights, reference_weights):
        raise ValueError(f"{name}: entity weights differ from the aligned plan")
    return values, weights


def _flag(values, *, name: str):
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name}: expected a boolean participation or filing flag")
    return values.astype(bool)


def _contribution_metrics(values, weights) -> dict[str, int | float]:
    positive = values > 0
    return {
        "contributor_records": int(positive.sum()),
        "weighted_contributors": float(weights[positive].sum()),
        "amount_dollars": float(np.sum(values * weights, dtype=np.float64)),
    }


def _score_artifact(
    path: Path, *, year: int, batch_size: int, builder, verified_base_period: int
) -> dict:
    if verified_base_period != SCORER_BASE_PERIOD:
        raise ValueError(
            "The maintained scorer requires verified base-period 2024 inputs"
        )

    def load_formula_free_frame(dataset_path, *, expected_sha256):
        frame = builder._load_frame(dataset_path, expected_sha256=expected_sha256)
        stored_model_results = {
            "traditional_ira_deduction",
            "dependent_traditional_ira_deduction",
            "ira_active_participant",
            "traditional_ira_contributions",
            "roth_ira_contributions",
        }
        forbidden = [
            f"{entity}.{column}"
            for entity in frame.entities
            for column in sorted(
                stored_model_results & set(frame.table(entity).columns)
            )
        ]
        if forbidden:
            raise ValueError(
                "IRA validation requires model-derived participation, capped "
                "contributions and deductions; stored model-result inputs are "
                f"forbidden: {', '.join(forbidden)}"
            )
        return frame

    plan = (
        ("traditional_ira_contributions", year, None),
        ("roth_ira_contributions", year, None),
        ("ira_active_participant", year, None),
        ("traditional_ira_deduction", year, "tax_unit"),
        ("dependent_traditional_ira_deduction", year, "tax_unit"),
        ("tax_unit_is_filer", year, None),
    )
    scorer = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=batch_size, load_frame=load_formula_free_frame
    )
    try:
        consumer = scorer.open_consumer("ira_participation_validation", plan)
        simulation = consumer.simulate(None)
        traditional, person_weights = _arrays(
            simulation.calculate("traditional_ira_contributions", year),
            name="traditional_ira_contributions",
        )
        roth, _ = _arrays(
            simulation.calculate("roth_ira_contributions", year),
            name="roth_ira_contributions",
            reference_weights=person_weights,
        )
        active_values, _ = _arrays(
            simulation.calculate("ira_active_participant", year),
            name="ira_active_participant",
            reference_weights=person_weights,
        )
        active = _flag(active_values, name="ira_active_participant")
        deduction, tax_unit_weights = _arrays(
            simulation.calculate("traditional_ira_deduction", year, "tax_unit"),
            name="traditional_ira_deduction mapped to tax_unit",
        )
        dependent_deduction, _ = _arrays(
            simulation.calculate(
                "dependent_traditional_ira_deduction", year, "tax_unit"
            ),
            name="dependent_traditional_ira_deduction mapped to tax_unit",
            reference_weights=tax_unit_weights,
        )
        parent_deduction = deduction - dependent_deduction
        if (parent_deduction < 0).any():
            raise ValueError(
                "Dependent own-return deductions exceed the corresponding "
                "all-person deductions aggregated to a parent tax unit"
            )
        filer_values, _ = _arrays(
            simulation.calculate("tax_unit_is_filer", year),
            name="tax_unit_is_filer",
            reference_weights=tax_unit_weights,
        )
        filer = _flag(filer_values, name="tax_unit_is_filer")
        traditional_contributor = traditional > 0
        denominator = float(person_weights[traditional_contributor].sum())
        numerator = float(person_weights[traditional_contributor & active].sum())
        positive_deduction = parent_deduction > 0
        filing_deduction = positive_deduction & filer
        result = {
            "artifact": {
                "path": str(path.resolve()),
                "sha256": scorer.dataset_sha256,
                "size_bytes": path.stat().st_size,
                "verified_base_period": verified_base_period,
            },
            "gross_contributions": {
                "grain": "person",
                "traditional": _contribution_metrics(traditional, person_weights),
                "roth": _contribution_metrics(roth, person_weights),
            },
            "active_participation": {
                "variable": "ira_active_participant",
                "traditional_contributor_records": int(traditional_contributor.sum()),
                "active_traditional_contributor_records": int(
                    (traditional_contributor & active).sum()
                ),
                "weighted_traditional_contributors": denominator,
                "weighted_active_traditional_contributors": numerator,
                "share_of_traditional_contributors_active": (
                    numerator / denominator if denominator else None
                ),
            },
            "model_derived_deduction": {
                "variable": "traditional_ira_deduction",
                "grain": "tax_unit",
                "positive_tax_unit_records": int(positive_deduction.sum()),
                "weighted_positive_tax_units": float(
                    tax_unit_weights[positive_deduction].sum()
                ),
                "all_person_amount_dollars_including_dependent_own_returns": float(
                    np.sum(deduction * tax_unit_weights, dtype=np.float64)
                ),
                "dependent_own_return_amount_dollars": float(
                    np.sum(dependent_deduction * tax_unit_weights, dtype=np.float64)
                ),
                "parent_tax_unit_amount_dollars": float(
                    np.sum(parent_deduction * tax_unit_weights, dtype=np.float64)
                ),
                "dependent_own_return_counts_available": False,
                "soi_comparison_scope": (
                    "Modeled filing parent tax units; dependent own returns are "
                    "excluded because their independent filing/count evidence is unavailable"
                ),
                "positive_filing_tax_unit_records": int(filing_deduction.sum()),
                "weighted_deducting_returns": float(
                    tax_unit_weights[filing_deduction].sum()
                ),
                "filing_tax_unit_amount_dollars": float(
                    np.sum(
                        parent_deduction[filer] * tax_unit_weights[filer],
                        dtype=np.float64,
                    )
                ),
            },
        }
        result["post_export_scoring"] = scorer.finish_consumer(consumer)
        return result
    finally:
        scorer.close()


def _benchmark_comparisons(metrics: dict, benchmarks: dict, year: int) -> list[dict]:
    comparisons = []
    for row in benchmarks["gross_contributions"]:
        observed = metrics["gross_contributions"][row["account_type"]]
        benchmark_amount = row["amount_thousands"] * 1000
        comparisons.append(
            {
                "measure": row["model_variable"],
                "benchmark_year": row["year"],
                "same_tax_year": row["year"] == year,
                "benchmark_count_unit": row["count_unit"],
                "benchmark_count": row["number_of_taxpayers"],
                "benchmark_amount_dollars": benchmark_amount,
                "contributor_count_ratio": (
                    observed["weighted_contributors"] / row["number_of_taxpayers"]
                ),
                "amount_ratio": observed["amount_dollars"] / benchmark_amount,
            }
        )
    observed = metrics["model_derived_deduction"]
    for row in benchmarks["model_derived_deductions"]:
        benchmark_amount = row["amount_thousands"] * 1000
        comparisons.append(
            {
                "measure": row["model_variable"],
                "benchmark_year": row["year"],
                "same_tax_year": row["year"] == year,
                "benchmark_count_unit": row["count_unit"],
                "modeled_count_scope": observed["soi_comparison_scope"],
                "benchmark_count": row["number_of_returns"],
                "benchmark_amount_dollars": benchmark_amount,
                "deducting_return_count_ratio": (
                    observed["weighted_deducting_returns"] / row["number_of_returns"]
                ),
                "amount_ratio": (
                    observed["filing_tax_unit_amount_dollars"] / benchmark_amount
                ),
            }
        )
    return comparisons


def validate_ira_participation(
    *,
    before_h5: Path,
    after_h5: Path,
    year: int,
    maximum_microsim_batch_size: int = 2000,
) -> dict:
    if year < SCORER_BASE_PERIOD:
        raise ValueError(
            f"--year must be {SCORER_BASE_PERIOD} or later; the maintained "
            "scorer cannot backcast 2024 base inputs into earlier model years"
        )
    if maximum_microsim_batch_size < 1:
        raise ValueError("maximum_microsim_batch_size must be positive")
    for path in (before_h5, after_h5):
        if not path.is_file():
            raise FileNotFoundError(f"IRA validation artifact does not exist: {path}")
    base_periods = {}
    for path in (before_h5, after_h5):
        base_period = _read_base_period(path)
        if base_period != SCORER_BASE_PERIOD:
            raise ValueError(
                f"{path}: _time_period is {base_period}; the maintained release "
                f"scorer materializes inputs at {SCORER_BASE_PERIOD}. Supply an "
                "unaged 2024 base artifact to avoid aging annual inputs twice."
            )
        base_periods[path] = base_period
    _require_model_variables(_model_system())
    builder = _load_builder()
    benchmark_bytes = (
        files("microcosm.build")
        .joinpath("us_runtime", "data", "ira_benchmarks.json")
        .read_bytes()
    )
    benchmarks = json.loads(benchmark_bytes)
    report = {
        "schema_version": 1,
        "purpose": "diagnostic_validation_not_release_certification",
        "year": year,
        "packages": _package_versions(),
        "python_version": sys.version.split()[0],
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmarks_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "benchmarks": benchmarks,
        "comparison_notes": [
            "Both artifacts use the same installed model. Before is not a reconstruction of the pre-#10032 deduction formula.",
            "Gross contributions count people across the modeled population; the Form 5498 benchmark is the matched Form 1040/5498/1099-R taxpayer study. Count-universe differences remain diagnostic.",
            "SOI deduction comparisons subtract dependent_traditional_ira_deduction from parent tax-unit aggregates before restricting to modeled tax_unit_is_filer. Dependent own-return amounts are reported separately; independent return counts/filing status are unavailable, so the parent-return comparison omits them.",
            "Both H5 files must carry _time_period 2024 because the maintained scorer materializes inputs at that base period. Already-aged annual artifacts are refused to prevent double aging.",
            "The benchmarks are tax years 2022 and 2023. Comparisons to another model year are cross-year diagnostics.",
            "The participation repair does not add employer-only allocations or defined-benefit coverage (option A2), and does not promise to reproduce SOI deductions or Form 5498 totals.",
        ],
    }
    for label, path in (("before", before_h5), ("after", after_h5)):
        metrics = _score_artifact(
            path,
            year=year,
            batch_size=maximum_microsim_batch_size,
            builder=builder,
            verified_base_period=base_periods[path],
        )
        metrics["benchmark_comparisons"] = _benchmark_comparisons(
            metrics, benchmarks, year
        )
        report[label] = metrics
    report["after_minus_before"] = {
        "traditional_contribution_amount_dollars": (
            report["after"]["gross_contributions"]["traditional"]["amount_dollars"]
            - report["before"]["gross_contributions"]["traditional"]["amount_dollars"]
        ),
        "all_person_deduction_amount_dollars_including_dependent_own_returns": (
            report["after"]["model_derived_deduction"][
                "all_person_amount_dollars_including_dependent_own_returns"
            ]
            - report["before"]["model_derived_deduction"][
                "all_person_amount_dollars_including_dependent_own_returns"
            ]
        ),
        "filing_tax_unit_deduction_amount_dollars": (
            report["after"]["model_derived_deduction"]["filing_tax_unit_amount_dollars"]
            - report["before"]["model_derived_deduction"][
                "filing_tax_unit_amount_dollars"
            ]
        ),
    }
    return report


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--before-h5", type=Path, required=True)
    parser.add_argument("--after-h5", type=Path, required=True)
    parser.add_argument("--year", type=_positive_integer, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--maximum-microsim-batch-size", type=_positive_integer, default=2000
    )
    args = parser.parse_args(argv)
    if args.out.resolve() in {args.before_h5.resolve(), args.after_h5.resolve()} or (
        args.out.exists()
        and any(
            path.exists() and args.out.samefile(path)
            for path in (args.before_h5, args.after_h5)
        )
    ):
        parser.error("--out must not overwrite either input H5 artifact")
    try:
        report = validate_ira_participation(
            before_h5=args.before_h5,
            after_h5=args.after_h5,
            year=args.year,
            maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"IRA validation written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
