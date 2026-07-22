"""Score a local directory of legacy US state H5 files.

This is a no-download eval harness. It expects the 51 state artifacts to
already exist on disk, stitches them into one national frame with disjoint
state strata, and scores that stitched frame against the same fiscal target
surface used by the single-H5 scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_STATE_ABBREVIATIONS = (
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
)
DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE: int | None = None


def _load_runtime():
    """Import the heavy scoring stack only after local state files exist."""

    import build_us_fiscal_refresh_release as release
    import score_us_fiscal_targets as fiscal_scorer

    from populace.calibrate import score_targets
    from populace.calibrate.diagnostics import write_calibration_diagnostics

    return release, fiscal_scorer, score_targets, write_calibration_diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-h5-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the already-local state H5 files. The path "
            "may point either at the state-file directory itself or at a "
            "parent directory containing states/."
        ),
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
        default=DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    )
    parser.add_argument(
        "--target-materialization-cache-dir",
        type=Path,
        help="Optional content-addressed cache for expensive target materialization.",
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
            "Score old state H5 files that contain formula/adds-owned PE output "
            "columns by dropping those columns before target materialization. "
            "This is read-only compatibility for historical evals."
        ),
    )
    parser.add_argument(
        "--legacy-pe-flat-h5",
        action="store_true",
        help=(
            "Read-only compatibility mode for legacy flat H5 files that store "
            "variables as root-level variable/<period> datasets instead of "
            "Populace entity tables."
        ),
    )
    parser.add_argument(
        "--write-required-file-list",
        type=Path,
        help=(
            "Write a JSON inventory of expected, present, and missing state "
            "files before failing fast on missing inputs."
        ),
    )
    return parser.parse_args()


def _state_h5_root(path: Path) -> Path:
    nested = path / "states"
    if nested.is_dir() and not any(path.glob("*.h5")):
        return nested
    return path


def required_state_file_inventory(state_h5_dir: Path) -> dict[str, object]:
    root = _state_h5_root(state_h5_dir)
    expected = {
        abbreviation: root / f"{abbreviation}.h5"
        for abbreviation in EXPECTED_STATE_ABBREVIATIONS
    }
    present = [str(path) for path in expected.values() if path.is_file()]
    missing = [str(path) for path in expected.values() if not path.is_file()]
    return {
        "state_h5_dir": str(root),
        "expected_state_count": len(expected),
        "expected_files": [str(path) for path in expected.values()],
        "present_files": present,
        "missing_files": missing,
    }


def assert_required_state_files_present(inventory: Mapping[str, object]) -> None:
    missing = list(inventory.get("missing_files") or [])
    if not missing:
        return
    sample = ", ".join(str(path) for path in missing[:10])
    suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
    raise SystemExit(
        "Missing required legacy state file(s); this scorer never downloads "
        f"inputs. Missing: {sample}{suffix}"
    )


def _write_required_file_list(
    inventory: Mapping[str, object],
    path: Path | None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(inventory), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frame_with_state_stratum(
    frame: Any,
    state_abbreviation: str,
) -> Any:
    import pandas as pd

    tables = {entity: frame.table(entity) for entity in frame.entities}
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    strata = pd.Series(
        f"legacy_state:{state_abbreviation}",
        index=frame.person.index,
        dtype=object,
        name="stratum",
    )
    return frame.__class__(
        tables,
        frame.schema,
        weights,
        strata,
        mass_log=frame.mass_log,
    )


def _load_state_frame(
    path: Path,
    *,
    release,
    fiscal_scorer,
    state_abbreviation: str,
    legacy_pe_flat_h5: bool,
) -> tuple[Any, dict[str, object]]:
    if legacy_pe_flat_h5:
        frame, layout_metadata = fiscal_scorer._load_legacy_pe_flat_frame(path)
    else:
        frame = release._load_frame(path)
        layout_metadata = {"layout": "populace_entity_h5"}
    tagged = _frame_with_state_stratum(frame, state_abbreviation)
    return tagged, {
        "state": state_abbreviation,
        "path": str(path),
        "sha256": release._sha256(path),
        "n_households": int(tagged.n("household")),
        "n_persons": int(tagged.n("person")),
        **layout_metadata,
    }


def _collection_sha256(state_files: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for state_file in sorted(state_files, key=lambda item: str(item["state"])):
        digest.update(str(state_file["state"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(state_file["sha256"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_state_collection(
    state_h5_dir: Path,
    *,
    release,
    fiscal_scorer,
    legacy_pe_flat_h5: bool,
) -> tuple[Any, list[dict[str, object]], str]:
    root = _state_h5_root(state_h5_dir)
    combined: Any | None = None
    state_files: list[dict[str, object]] = []
    for state_abbreviation in EXPECTED_STATE_ABBREVIATIONS:
        path = root / f"{state_abbreviation}.h5"
        frame, metadata = _load_state_frame(
            path,
            release=release,
            fiscal_scorer=fiscal_scorer,
            state_abbreviation=state_abbreviation,
            legacy_pe_flat_h5=legacy_pe_flat_h5,
        )
        if combined is None:
            combined = frame
        else:
            try:
                combined = combined.concat(frame)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot stitch state {state_abbreviation} from {path}: {exc}"
                ) from exc
        state_files.append(metadata)
    if combined is None:  # pragma: no cover - defensive; expected set is nonempty.
        raise ValueError("No state files were loaded.")
    return combined, state_files, _collection_sha256(state_files)


def score_state_files(
    *,
    state_h5_dir: Path,
    ledger_facts: Path,
    age_targets: bool = False,
    allow_unaged_dollar_targets: bool = True,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    diagnostic_skip_tax_expenditure_targets: bool = False,
    allow_legacy_formula_owned_inputs: bool = False,
    legacy_pe_flat_h5: bool = False,
    target_materialization_cache_dir: Path | None = None,
) -> tuple[Any, object, dict[str, object], dict[str, object]]:
    assert_required_state_files_present(required_state_file_inventory(state_h5_dir))
    release, fiscal_scorer, score_targets, _ = _load_runtime()
    if maximum_microsim_batch_size is None:
        maximum_microsim_batch_size = release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    base_frame, state_files, collection_sha256 = _load_state_collection(
        state_h5_dir,
        release=release,
        fiscal_scorer=fiscal_scorer,
        legacy_pe_flat_h5=legacy_pe_flat_h5,
    )
    target_registry = release.compile_us_fiscal_target_registry(
        release.load_ledger_consumer_artifact(ledger_facts).facts,
        target_period=release.PERIOD,
        age_targets=age_targets,
        allow_unaged_dollar_targets=allow_unaged_dollar_targets,
        include_congressional_district_targets=False,
        congressional_district_vintage_crosswalk=None,
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
    legacy_formula_owned_inputs: dict[str, list[str]] = {}
    if allow_legacy_formula_owned_inputs:
        base_frame, legacy_formula_owned_inputs = (
            fiscal_scorer._drop_legacy_formula_owned_inputs(base_frame)
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
            "base_dataset_sha256": collection_sha256,
            "build_commit": release._git_output("rev-parse", "HEAD"),
            "policyengine_us_version": release._package_or_workspace_version(
                "policyengine-us"
            ),
            "seed": 0,
            "target_period": release.PERIOD,
            "target_registry_version": target_registry.version,
            "state_file_collection": collection_sha256,
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
        target_loss_weights=release._fiscal_target_loss_weights(
            registry,
            critical_target_loss_multiplier=(
                release.US_CRITICAL_TARGET_LOSS_MULTIPLIER
            ),
        ),
        target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
        options={
            "mass": "existing_weights",
            "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
            "critical_target_loss_multiplier": (
                release.US_CRITICAL_TARGET_LOSS_MULTIPLIER
            ),
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
    compilation = {
        **dict(compilation),
        "scored_dataset": {
            "layout": (
                "legacy_pe_flat_state_h5_collection"
                if legacy_pe_flat_h5
                else "state_h5_collection"
            ),
            "state_h5_dir": str(_state_h5_root(state_h5_dir)),
            "state_file_collection_sha256": collection_sha256,
            "state_files": state_files,
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
    release,
    state_h5_dir: Path,
    ledger_facts: Path,
    result: Any,
    registry,
    compilation: Mapping[str, object],
    gates: Mapping[str, object],
) -> dict[str, object]:
    scored_dataset = dict(compilation.get("scored_dataset") or {})
    return {
        "scored_at": datetime.now(UTC).isoformat(),
        "state_h5_dir": str(_state_h5_root(state_h5_dir)),
        "state_file_collection_sha256": scored_dataset.get(
            "state_file_collection_sha256"
        ),
        "ledger_facts": str(ledger_facts),
        "ledger_facts_sha256": release._sha256(ledger_facts),
        "final_loss": result.final_loss,
        "initial_loss": result.initial_loss,
        "critical_target_loss_multiplier": (release.US_CRITICAL_TARGET_LOSS_MULTIPLIER),
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
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = required_state_file_inventory(args.state_h5_dir)
    _write_required_file_list(inventory, args.write_required_file_list)
    assert_required_state_files_present(inventory)
    result, registry, compilation, gates = score_state_files(
        state_h5_dir=args.state_h5_dir,
        ledger_facts=args.ledger_facts,
        age_targets=args.age_targets,
        allow_unaged_dollar_targets=args.allow_unaged_dollar_targets,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        diagnostic_skip_tax_expenditure_targets=(
            args.diagnostic_skip_tax_expenditure_targets
        ),
        allow_legacy_formula_owned_inputs=args.allow_legacy_formula_owned_inputs,
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
    release, _, _, write_calibration_diagnostics = _load_runtime()
    write_calibration_diagnostics(
        result,
        out / "calibration_diagnostics.json",
        target_registry=registry,
        build={
            "mode": "score_state_files",
            "base_dataset_sha256": compilation["scored_dataset"][
                "state_file_collection_sha256"
            ],
            "target_compilation": compilation,
            "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
            "critical_target_loss_multiplier": (
                release.US_CRITICAL_TARGET_LOSS_MULTIPLIER
            ),
            "target_loss_cap": release.US_FISCAL_TARGET_LOSS_CAP,
            "gates": gates,
        },
    )
    summary = _summary_payload(
        release=release,
        state_h5_dir=args.state_h5_dir,
        ledger_facts=args.ledger_facts,
        result=result,
        registry=registry,
        compilation=compilation,
        gates=gates,
    )
    (out / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
