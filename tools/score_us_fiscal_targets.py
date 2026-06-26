"""Score a Populace US H5 against the current fiscal target surface.

This is the read-only companion to ``build_us_fiscal_refresh_release.py``:
it materializes the same target frame, compiles the same constraint matrix,
and evaluates the existing household weights without optimizing them.
Use it to compare an incumbent release to a candidate on the exact same
target surface before deciding whether a rebuild is actually better.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import build_us_fiscal_refresh_release as release

from populace.calibrate import score_targets
from populace.calibrate.diagnostics import write_calibration_diagnostics
from populace.calibrate.solve import CalibrationResult


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5",
        type=Path,
        help="Populace US H5 to score. Defaults to HF latest.",
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Ledger consumer facts JSONL used to compile the fiscal target surface.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--maximum-microsim-batch-size",
        "--maximum-microsimulation-batch-size",
        dest="maximum_microsim_batch_size",
        type=int,
        default=release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    )
    parser.add_argument(
        "--diagnostic-skip-tax-expenditure-targets",
        action="store_true",
        help="Diagnostic only: drop JCT reform targets to avoid reform simulations.",
    )
    parser.add_argument(
        "--allow-legacy-formula-owned-inputs",
        action="store_true",
        help=(
            "Score an old H5 that contains formula/adds-owned PE output columns "
            "by dropping those columns before target materialization. This is "
            "read-only compatibility for historical incumbents; release builds "
            "still fail on these columns."
        ),
    )
    parser.add_argument(
        "--include-congressional-district-targets",
        action="store_true",
        help=(
            "Opt into SOI congressional-district target rows. Requires the "
            "scored H5 to contain household congressional_district_geoid."
        ),
    )
    return parser.parse_args()


def _drop_legacy_formula_owned_inputs(frame) -> tuple[object, dict[str, list[str]]]:
    adapter = release.PolicyEngineUSEngine()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    formula_owned = adapter._engine_computed_columns(tables, period=release.PERIOD)
    dropped: dict[str, list[str]] = {}
    if not formula_owned:
        return frame, dropped

    cleaned_tables = {}
    for entity in frame.entities:
        table = frame.table(entity)
        columns = sorted(column for column in formula_owned if column in table.columns)
        if columns:
            dropped[entity] = columns
            cleaned_tables[entity] = table.drop(columns=columns)
        else:
            cleaned_tables[entity] = table.copy()
    cleaned_weights = {
        entity: frame.weights_for(entity) for entity in frame.weighted_entities
    }
    return (
        release.Frame(
            cleaned_tables,
            frame.schema,
            cleaned_weights,
            frame.strata,
            mass_log=frame.mass_log,
        ),
        dropped,
    )


def score_frame(
    *,
    h5: Path,
    ledger_facts: Path,
    maximum_microsim_batch_size: int
    | None = release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    diagnostic_skip_tax_expenditure_targets: bool = False,
    allow_legacy_formula_owned_inputs: bool = False,
    include_congressional_district_targets: bool = False,
) -> tuple[CalibrationResult, object, dict[str, object], dict[str, object]]:
    target_registry = release.compile_us_fiscal_target_registry(
        release._load_ledger_facts(ledger_facts),
        target_period=release.PERIOD,
        include_congressional_district_targets=include_congressional_district_targets,
    )
    target_specs = target_registry.specs
    if diagnostic_skip_tax_expenditure_targets:
        tax_expenditure_measures = {
            reform_spec.measure
            for reform_spec in release.US_JCT_TAX_EXPENDITURE_REFORMS
        }
        target_specs = tuple(
            spec
            for spec in target_specs
            if spec.measure not in tax_expenditure_measures
        )

    target_profile_gate = release.target_profile_coverage_gate(
        target_specs,
        release.US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    base_frame = release._load_frame(h5)
    legacy_formula_owned_inputs: dict[str, list[str]] = {}
    if allow_legacy_formula_owned_inputs:
        base_frame, legacy_formula_owned_inputs = _drop_legacy_formula_owned_inputs(
            base_frame
        )
    base_frame, base_population_repair = release._with_base_population_mass_repair(
        base_frame
    )
    base_population_gate = release._base_population_scale_gate(
        base_frame,
        mass_repair=base_population_repair,
    )
    health_input_gate = release._health_input_signal_gate(base_frame)
    target_frame, registry, compilation = release._materialize_target_frame(
        base_frame,
        target_specs,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    result = score_targets(
        target_frame,
        registry.to_target_set(),
        target_loss_weights=release._fiscal_target_loss_weights(registry),
        target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
        options={
            "mass": "existing_weights",
            "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
            "maximum_microsim_batch_size": maximum_microsim_batch_size,
        },
    )
    gates = {
        "target_profile_coverage": {
            "passed": target_profile_gate.passed,
            "failures": list(target_profile_gate.failures),
            "details": dict(target_profile_gate.details),
        },
        "health_input_signal": {
            "passed": health_input_gate.passed,
            "failures": list(health_input_gate.failures),
            "details": dict(health_input_gate.details),
        },
        "base_population_scale": {
            "passed": base_population_gate.passed,
            "failures": list(base_population_gate.failures),
            "details": dict(base_population_gate.details),
        },
    }
    if legacy_formula_owned_inputs:
        compilation = {
            **dict(compilation),
            "legacy_formula_owned_inputs_dropped": legacy_formula_owned_inputs,
        }
    return result, registry, compilation, gates


def _summary_payload(
    *,
    h5: Path,
    ledger_facts: Path,
    result: CalibrationResult,
    registry,
    compilation: dict[str, object],
    gates: dict[str, object],
) -> dict[str, object]:
    return {
        "scored_at": datetime.now(UTC).isoformat(),
        "h5": str(h5),
        "h5_sha256": release._sha256(h5),
        "ledger_facts": str(ledger_facts),
        "ledger_facts_sha256": release._sha256(ledger_facts),
        "final_loss": result.final_loss,
        "initial_loss": result.initial_loss,
        "fraction_within_10pct": result.fraction_within_10pct,
        "n_records": int(result.weights.shape[0]),
        "n_nonzero": int(result.n_nonzero),
        "target_registry": {
            "country": registry.country,
            "version": registry.version,
            "n_specs": len(registry.specs),
        },
        "target_compilation": compilation,
        "gates": gates,
    }


def main() -> None:
    args = _parse_args()
    h5 = args.h5 or release._download_base_h5()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result, registry, compilation, gates = score_frame(
        h5=h5,
        ledger_facts=args.ledger_facts,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        diagnostic_skip_tax_expenditure_targets=args.diagnostic_skip_tax_expenditure_targets,
        allow_legacy_formula_owned_inputs=args.allow_legacy_formula_owned_inputs,
        include_congressional_district_targets=args.include_congressional_district_targets,
    )
    write_calibration_diagnostics(
        result,
        out / "calibration_diagnostics.json",
        target_registry=registry,
        build={
            "mode": "score_only",
            "base_dataset_sha256": release._sha256(h5),
            "target_compilation": compilation,
            "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
            "target_loss_cap": release.US_FISCAL_TARGET_LOSS_CAP,
            "gates": gates,
        },
    )
    summary = _summary_payload(
        h5=h5,
        ledger_facts=args.ledger_facts,
        result=result,
        registry=registry,
        compilation=compilation,
        gates=gates,
    )
    (out / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
