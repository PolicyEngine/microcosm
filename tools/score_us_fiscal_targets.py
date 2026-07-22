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
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import build_us_fiscal_refresh_release as release
import h5py
import numpy as np
import pandas as pd

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
        "--age-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compile the target surface with cbo_growth_factor_aging, "
            "matching a release built with aging on. Defaults off so "
            "incumbent releases score against the un-aged surface they "
            "were calibrated to."
        ),
    )
    parser.add_argument(
        "--allow-unaged-dollar-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Waive the period contract for cross-period observation dollar "
            "levels (PolicyEngine/ledger#71). Defaults on for scoring, since "
            "incumbent releases were compiled before the contract existed."
        ),
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
        "--target-materialization-cache-dir",
        type=Path,
        help=(
            "Optional content-addressed cache for expensive target "
            "materialization, such as JCT reform income-tax vectors."
        ),
    )
    parser.add_argument(
        "--no-target-materialization-cache",
        action="store_true",
        help="Disable the scorer target-materialization cache.",
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
        "--allow-legacy-cd-provenance",
        action="store_true",
        help=(
            "Score an old H5 that contains current congressional-district "
            "geoids but predates CD vintage provenance attributes. The scorer "
            "still requires the observed area artifact surface to be complete "
            "and current. Release builds remain provenance-strict."
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
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        help=(
            "Optional source-to-current congressional-district crosswalk "
            "artifact with source_geography_id, target_geography_id, and "
            "weight columns. Used to score old-vintage SOI CD targets on the "
            "current regional surface."
        ),
    )
    parser.add_argument(
        "--legacy-pe-flat-h5",
        action="store_true",
        help=(
            "Read-only compatibility mode for legacy flat H5 files that "
            "store variables as root-level variable/<period> datasets instead "
            "of Populace entity tables."
        ),
    )
    return parser.parse_args()


_LEGACY_STRUCTURAL_ENTITY_BY_COLUMN = {
    "person_id": "person",
    "person_household_id": "person",
    "person_tax_unit_id": "person",
    "person_spm_unit_id": "person",
    "person_family_id": "person",
    "person_marital_unit_id": "person",
    "household_id": "household",
    "household_weight": "household",
    "tax_unit_id": "tax_unit",
    "spm_unit_id": "spm_unit",
    "family_id": "family",
    "marital_unit_id": "marital_unit",
}


def _legacy_period_dataset(
    root: h5py.File,
    name: str,
    *,
    period: int = release.PERIOD,
) -> h5py.Dataset:
    obj = root[name]
    if isinstance(obj, h5py.Dataset):
        return obj
    period_key = str(period)
    if period_key in obj:
        child = obj[period_key]
        if isinstance(child, h5py.Dataset):
            return child
    dataset_children = [
        child for child in obj.values() if isinstance(child, h5py.Dataset)
    ]
    if len(dataset_children) == 1:
        return dataset_children[0]
    raise ValueError(
        f"legacy flat H5 root key {name!r} does not contain a usable "
        f"{period_key!r} dataset"
    )


def _legacy_h5_values(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[()])
    if values.ndim != 1:
        raise ValueError(
            f"legacy flat H5 dataset {dataset.name!r} is not one-dimensional: "
            f"shape={values.shape}"
        )
    if values.dtype.kind == "S":
        return values.astype(str)
    return values


def _policyengine_variable_entity_map() -> dict[str, str]:
    from policyengine_us import CountryTaxBenefitSystem

    return {
        name: variable.entity.key
        for name, variable in CountryTaxBenefitSystem().variables.items()
    }


def _legacy_entity_lengths(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    required = {
        "person": "person_id",
        "household": "household_id",
        "tax_unit": "tax_unit_id",
        "spm_unit": "spm_unit_id",
        "family": "family_id",
        "marital_unit": "marital_unit_id",
    }
    missing = [column for column in required.values() if column not in arrays]
    if missing:
        raise ValueError(
            "legacy flat H5 is missing required structural columns: "
            + ", ".join(sorted(missing))
        )
    return {entity: len(arrays[column]) for entity, column in required.items()}


def _infer_legacy_variable_entity(
    *,
    name: str,
    values: np.ndarray,
    variable_entity_by_name: Mapping[str, str],
    lengths_by_entity: Mapping[str, int],
) -> tuple[str | None, str | None]:
    if name in _LEGACY_STRUCTURAL_ENTITY_BY_COLUMN:
        return _LEGACY_STRUCTURAL_ENTITY_BY_COLUMN[name], None
    if name in variable_entity_by_name:
        entity = variable_entity_by_name[name]
        expected = lengths_by_entity.get(entity)
        if expected is None:
            return None, f"PolicyEngine entity {entity!r} is not in the US schema"
        if len(values) != expected:
            return (
                None,
                f"PolicyEngine entity {entity!r} expects {expected} rows, "
                f"found {len(values)}",
            )
        return entity, None

    matches = [
        entity
        for entity, expected in lengths_by_entity.items()
        if len(values) == expected
    ]
    if len(matches) == 1:
        return matches[0], "inferred_by_unique_row_count"
    if len(matches) > 1:
        return None, f"ambiguous row count {len(values)} matches {matches}"
    return None, f"row count {len(values)} matches no US entity"


def _load_legacy_pe_flat_frame(
    path: Path,
    *,
    period: int = release.PERIOD,
    variable_entity_by_name: Mapping[str, str] | None = None,
) -> tuple[release.Frame, dict[str, object]]:
    """Load the historical flat PE H5 layout for scoring.

    Old eCPS-style artifacts predate Populace entity-table H5s and store each
    variable as ``/<variable>/<period>``. This compatibility path exists only so
    the read-only scorer can compare old artifacts against the current target
    surface before replacing them.
    """

    variable_entity_by_name = (
        variable_entity_by_name or _policyengine_variable_entity_map()
    )
    arrays: dict[str, np.ndarray] = {}
    skipped: list[dict[str, str]] = []
    with h5py.File(path, "r") as h5:
        for name in h5.keys():
            try:
                dataset = _legacy_period_dataset(h5, name, period=period)
                arrays[name] = _legacy_h5_values(dataset)
            except ValueError as exc:
                skipped.append({"column": name, "reason": str(exc)})

    lengths_by_entity = _legacy_entity_lengths(arrays)
    table_columns: dict[str, dict[str, np.ndarray]] = {
        entity: {} for entity in release.US_SCHEMA.entities
    }
    inferred_unknown_columns: dict[str, list[str]] = {}
    skipped_columns = list(skipped)

    for name, values in arrays.items():
        entity, note = _infer_legacy_variable_entity(
            name=name,
            values=values,
            variable_entity_by_name=variable_entity_by_name,
            lengths_by_entity=lengths_by_entity,
        )
        if entity is None:
            skipped_columns.append({"column": name, "reason": note or "unknown"})
            continue
        table_columns[entity][name] = values
        if note == "inferred_by_unique_row_count":
            inferred_unknown_columns.setdefault(entity, []).append(name)

    missing_weight = "household_weight" not in table_columns["household"]
    if missing_weight:
        raise ValueError("legacy flat H5 is missing household_weight")
    weights = np.asarray(
        table_columns["household"].pop("household_weight"), dtype=np.float64
    )
    tables = {
        entity: pd.DataFrame(columns) for entity, columns in table_columns.items()
    }
    frame = release.Frame(
        tables,
        release.US_SCHEMA,
        {"household": release.Weights(weights, release.WeightKind.CALIBRATED)},
    )
    positive_household_weights = weights > 0
    dropped_zero_weight_households = int((~positive_household_weights).sum())
    dropped_zero_weight_persons = 0
    if dropped_zero_weight_households:
        household_id_column = release.US_SCHEMA.id_column("household")
        positive_household_ids = set(
            tables["household"].loc[positive_household_weights, household_id_column]
        )
        person_household_column = release.US_SCHEMA.membership_column("household")
        person_mask = (
            tables["person"][person_household_column]
            .isin(positive_household_ids)
            .to_numpy()
        )
        dropped_zero_weight_persons = int((~person_mask).sum())
        frame = frame.select(person_mask)
    metadata = {
        "layout": "legacy_pe_flat_h5",
        "period": period,
        "loaded_columns_by_entity": {
            entity: sorted(columns) for entity, columns in table_columns.items()
        },
        "inferred_unknown_columns_by_entity": {
            entity: sorted(columns)
            for entity, columns in inferred_unknown_columns.items()
        },
        "skipped_columns": skipped_columns,
        "entity_row_counts": dict(lengths_by_entity),
        "dropped_zero_weight_households": dropped_zero_weight_households,
        "dropped_zero_weight_persons": dropped_zero_weight_persons,
    }
    return frame, metadata


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


def _assert_legacy_cd_provenance_options(
    *,
    allow_legacy_cd_provenance: bool,
    congressional_district_vintage_crosswalk_metadata: dict[str, object] | None,
) -> None:
    if allow_legacy_cd_provenance and (
        congressional_district_vintage_crosswalk_metadata is None
    ):
        raise ValueError(
            "--allow-legacy-cd-provenance requires "
            "--congressional-district-vintage-crosswalk so the scorer can "
            "verify the H5 has the current congressional-district surface."
        )


def score_frame(
    *,
    h5: Path,
    ledger_facts: Path,
    age_targets: bool = False,
    allow_unaged_dollar_targets: bool = True,
    maximum_microsim_batch_size: int
    | None = release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    diagnostic_skip_tax_expenditure_targets: bool = False,
    allow_legacy_formula_owned_inputs: bool = False,
    allow_legacy_cd_provenance: bool = False,
    include_congressional_district_targets: bool = False,
    congressional_district_vintage_crosswalk: Path | None = None,
    target_materialization_cache_dir: Path | None = None,
    legacy_pe_flat_h5: bool = False,
) -> tuple[CalibrationResult, object, dict[str, object], dict[str, object]]:
    congressional_district_vintage_crosswalk_metadata = (
        {
            "path": str(congressional_district_vintage_crosswalk.resolve()),
            "sha256": release._sha256(congressional_district_vintage_crosswalk),
        }
        if congressional_district_vintage_crosswalk is not None
        else None
    )
    _assert_legacy_cd_provenance_options(
        allow_legacy_cd_provenance=allow_legacy_cd_provenance,
        congressional_district_vintage_crosswalk_metadata=(
            congressional_district_vintage_crosswalk_metadata
        ),
    )
    if not allow_legacy_cd_provenance:
        release._assert_cd_vintage_support_matches(
            h5, congressional_district_vintage_crosswalk_metadata
        )
    target_registry = release.compile_us_fiscal_target_registry(
        release.load_ledger_consumer_artifact(ledger_facts).facts,
        target_period=release.PERIOD,
        age_targets=age_targets,
        allow_unaged_dollar_targets=allow_unaged_dollar_targets,
        include_congressional_district_targets=include_congressional_district_targets,
        congressional_district_vintage_crosswalk=(
            release.load_congressional_district_vintage_crosswalk(
                congressional_district_vintage_crosswalk
            )
            if congressional_district_vintage_crosswalk is not None
            else None
        ),
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
    legacy_h5_layout: dict[str, object] = {}
    if legacy_pe_flat_h5:
        base_frame, legacy_h5_layout = _load_legacy_pe_flat_frame(h5)
    else:
        base_frame = release._load_frame(h5)
    if (
        allow_legacy_cd_provenance
        and congressional_district_vintage_crosswalk_metadata is not None
    ):
        release._strict_area_artifact_specs(base_frame)
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
    target_materialization_cache_context = (
        {
            "base_dataset_sha256": release._sha256(h5),
            "build_commit": release._git_output("rev-parse", "HEAD"),
            "policyengine_us_version": release._package_or_workspace_version(
                "policyengine-us"
            ),
            "seed": 0,
            "target_period": release.PERIOD,
            "target_registry_version": target_registry.version,
            "congressional_district_vintage_crosswalk_sha256": (
                congressional_district_vintage_crosswalk_metadata or {}
            ).get("sha256"),
        }
        if target_materialization_cache_dir is not None
        else None
    )
    target_frame, registry, compilation = release._materialize_target_frame(
        base_frame,
        target_specs,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
        target_materialization_cache_dir=target_materialization_cache_dir,
        target_materialization_cache_context=target_materialization_cache_context,
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
    if legacy_h5_layout:
        compilation = {
            **dict(compilation),
            "legacy_pe_flat_h5": legacy_h5_layout,
        }
    if allow_legacy_cd_provenance:
        compilation = {
            **dict(compilation),
            "legacy_cd_provenance_allowed": {
                "reason": (
                    "read-only incumbent scoring for an H5 that predates CD "
                    "vintage provenance attributes"
                ),
                "area_artifact_surface_checked": True,
            },
        }
    return result, registry, compilation, gates


def _summary_payload(
    *,
    h5: Path,
    ledger_facts: Path,
    congressional_district_vintage_crosswalk: Path | None,
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
        "congressional_district_vintage_crosswalk": (
            str(congressional_district_vintage_crosswalk)
            if congressional_district_vintage_crosswalk is not None
            else None
        ),
        "congressional_district_vintage_crosswalk_sha256": (
            release._sha256(congressional_district_vintage_crosswalk)
            if congressional_district_vintage_crosswalk is not None
            else None
        ),
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
        age_targets=args.age_targets,
        allow_unaged_dollar_targets=args.allow_unaged_dollar_targets,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        diagnostic_skip_tax_expenditure_targets=args.diagnostic_skip_tax_expenditure_targets,
        allow_legacy_formula_owned_inputs=args.allow_legacy_formula_owned_inputs,
        allow_legacy_cd_provenance=args.allow_legacy_cd_provenance,
        include_congressional_district_targets=args.include_congressional_district_targets,
        congressional_district_vintage_crosswalk=(
            args.congressional_district_vintage_crosswalk
        ),
        legacy_pe_flat_h5=args.legacy_pe_flat_h5,
        target_materialization_cache_dir=(
            None
            if args.no_target_materialization_cache
            else (
                args.target_materialization_cache_dir
                or out / "target_materialization_cache"
            )
        ),
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
        congressional_district_vintage_crosswalk=(
            args.congressional_district_vintage_crosswalk
        ),
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
