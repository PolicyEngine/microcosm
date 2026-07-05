"""Score a local directory of legacy US state H5 files.

This is a no-download eval harness. It expects the 51 state artifacts to
already exist on disk and scores them as a streamed national collection against
the same fiscal target surface used by the single-H5 scorer. Streaming keeps
legacy PE-US-data evals memory-bounded even though the state shards are large
full microdata files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
STATE_FIPS_BY_ABBREVIATION = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}
DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE: int | None = None


def _collect_batch_garbage() -> None:
    """Keep streaming loops tidy without traversing the full object graph."""

    gc.collect(0)


def _load_runtime():
    """Import the heavy scoring stack only after local state files exist."""

    import build_us_fiscal_refresh_release as release
    import score_us_fiscal_targets as fiscal_scorer

    from populace.calibrate import score_targets
    from populace.calibrate.diagnostics import write_calibration_diagnostics

    return release, fiscal_scorer, score_targets, write_calibration_diagnostics


@dataclass(frozen=True)
class _ShapeOnlyArray:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _ShapeOnlySparseMatrix:
    shape: tuple[int, int]
    nnz: int = 0


@dataclass(frozen=True)
class _StreamingScoreProblem:
    target_vector: np.ndarray
    names: tuple[str, ...]
    targets: tuple[object, ...]
    matrix: _ShapeOnlySparseMatrix
    skipped: tuple[object, ...] = ()


@dataclass(frozen=True)
class _StreamingScoreResult:
    weight_entity: str
    weights: _ShapeOnlyArray
    initial_weights: _ShapeOnlyArray
    diagnostics: tuple[object, ...]
    loss_trajectory: np.ndarray
    skipped: tuple[object, ...]
    problem: _StreamingScoreProblem
    l0_lambda: float
    n_nonzero: int
    closing_loss: float
    options: Mapping[str, object]

    @property
    def initial_loss(self) -> float:
        return float(self.loss_trajectory[0])

    @property
    def final_loss(self) -> float:
        return float(self.closing_loss)

    @property
    def fraction_within_10pct(self) -> float:
        if not self.diagnostics:
            return 0.0
        hits = sum(
            abs(float(diagnostic.relative_error)) <= 0.10
            for diagnostic in self.diagnostics
        )
        return hits / len(self.diagnostics)


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
        "--ledger-facts",
        type=Path,
        required=True,
        help="Ledger consumer facts JSONL used to compile the fiscal target surface.",
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        required=True,
        help=(
            "Source-to-current congressional-district crosswalk artifact. "
            "Required because the standard fiscal target surface includes "
            "congressional-district rows."
        ),
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
        "--allow-legacy-cd-provenance",
        action="store_true",
        help=(
            "Score old state H5 files that contain current congressional-district "
            "geoids but predate CD vintage provenance attributes. This is "
            "read-only compatibility for historical evals."
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
    allow_legacy_cd_provenance: bool,
    congressional_district_vintage_crosswalk_metadata: Mapping[str, object],
) -> tuple[Any, dict[str, object]]:
    if not allow_legacy_cd_provenance:
        release._assert_cd_vintage_support_matches(
            path,
            congressional_district_vintage_crosswalk_metadata,
        )
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
        "legacy_cd_provenance_allowed": bool(allow_legacy_cd_provenance),
    }


def _collection_sha256(state_files: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for state_file in sorted(state_files, key=lambda item: str(item["state"])):
        digest.update(str(state_file["state"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(state_file["sha256"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _row_name(target: object) -> str:
    row_name = getattr(target, "row_name", None)
    if row_name is not None:
        return str(row_name)
    return f"{target.name}@{target.period}"


def _target_estimates(
    target_frame: Any,
    target_set: Any,
) -> np.ndarray:
    from populace.calibrate.matrix import build_constraint_matrix
    from populace.calibrate.target import TargetSet

    if not isinstance(target_set, TargetSet):
        target_set = TargetSet(tuple(target_set))
    problem = build_constraint_matrix(target_frame, target_set, "household")
    if problem.skipped:
        sample = "; ".join(
            f"{skipped.target.row_name}: {skipped.reason}"
            for skipped in problem.skipped[:5]
        )
        suffix = (
            "" if len(problem.skipped) <= 5 else f"; ... {len(problem.skipped)} skipped"
        )
        raise RuntimeError(
            f"Streaming state target estimates skipped target(s): {sample}{suffix}"
        )
    expected_names = tuple(_row_name(target) for target in target_set.targets)
    if tuple(problem.names) != expected_names:
        raise RuntimeError(
            "Streaming state target estimate rows changed during matrix "
            f"compilation. Expected first rows {expected_names[:3]!r}, got "
            f"{tuple(problem.names[:3])!r}."
        )
    weights = target_frame.resolve_weights("household").values
    return np.asarray(problem.estimates(weights), dtype=np.float64)


def _target_diagnostics_from_estimates(
    *,
    targets: Sequence[object],
    estimates: np.ndarray,
):
    from populace.calibrate.solve import TargetDiagnostic

    diagnostics: list[object] = []
    for target, estimate in zip(targets, estimates, strict=True):
        target_value = float(target.value)
        relative_error = (
            (float(estimate) - target_value) / target_value
            if target_value != 0.0
            else float(estimate) - target_value
        )
        within_tolerance = (
            None
            if target.tolerance is None
            else abs(float(estimate) - target_value) <= target.tolerance
        )
        diagnostics.append(
            TargetDiagnostic(
                name=_row_name(target),
                target=target_value,
                initial_estimate=float(estimate),
                final_estimate=float(estimate),
                relative_error=float(relative_error),
                within_tolerance=within_tolerance,
            )
        )
    return tuple(diagnostics)


def _state_fips(state_abbreviation: str) -> str:
    try:
        return STATE_FIPS_BY_ABBREVIATION[state_abbreviation]
    except KeyError as exc:
        raise ValueError(f"Unknown state abbreviation {state_abbreviation!r}.") from exc


def _metadata_state_fips(value: object) -> str:
    return str(value).strip().zfill(2)


def _metadata_cd_state_fips(value: object) -> str:
    return str(value).strip().zfill(4)[:2]


def _target_spec_applies_to_state(spec: object, state_abbreviation: str) -> bool:
    metadata = getattr(spec, "metadata", {}) or {}
    state_fips = _state_fips(state_abbreviation)
    spec_state_fips = metadata.get("state_fips")
    if spec_state_fips and _metadata_state_fips(spec_state_fips) != state_fips:
        return False
    congressional_district_geoid = metadata.get("congressional_district_geoid")
    if (
        congressional_district_geoid
        and _metadata_cd_state_fips(congressional_district_geoid) != state_fips
    ):
        return False
    return True


def _target_specs_for_state(
    target_specs: Sequence[object],
    state_abbreviation: str,
) -> tuple[object, ...]:
    return tuple(
        spec
        for spec in target_specs
        if _target_spec_applies_to_state(spec, state_abbreviation)
    )


def _streaming_score_result(
    *,
    registry,
    estimates: np.ndarray,
    n_records: int,
    release,
    maximum_microsim_batch_size: int,
) -> _StreamingScoreResult:
    from populace.calibrate.solve import (
        default_target_loss_scales,
        relative_error_loss,
    )

    target_set = registry.to_target_set()
    targets = tuple(target_set.targets)
    names = tuple(_row_name(target) for target in targets)
    target_vector = np.asarray([float(target.value) for target in targets])
    loss_weights = release._fiscal_target_loss_weights(registry)
    loss = relative_error_loss(
        estimates,
        target_vector,
        target_loss_weights=loss_weights,
        target_loss_scales=default_target_loss_scales(target_vector),
        target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
    )
    diagnostics = _target_diagnostics_from_estimates(
        targets=targets,
        estimates=estimates,
    )
    shape = (int(n_records),)
    return _StreamingScoreResult(
        weight_entity="household",
        weights=_ShapeOnlyArray(shape),
        initial_weights=_ShapeOnlyArray(shape),
        diagnostics=diagnostics,
        loss_trajectory=np.asarray([loss], dtype=np.float64),
        skipped=(),
        problem=_StreamingScoreProblem(
            target_vector=target_vector,
            names=names,
            targets=targets,
            matrix=_ShapeOnlySparseMatrix((len(targets), int(n_records))),
        ),
        l0_lambda=0.0,
        n_nonzero=int(n_records),
        closing_loss=float(loss),
        options={
            "method": "streaming_state_file_score_only",
            "mass": "existing_weights",
            "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
            "target_loss_cap": release.US_FISCAL_TARGET_LOSS_CAP,
            "target_loss_weights": "provided",
            "maximum_microsim_batch_size": maximum_microsim_batch_size,
        },
    )


def _apply_household_weight_factor(
    frame: Any,
    *,
    factor: float,
    release,
) -> Any:
    if factor == 1.0:
        return frame
    weights = frame.weights_for("household")
    return frame.with_weights(
        "household",
        weights.with_values(weights.values * factor, weights.kind),
        mass=release.MassChange(
            factor=factor,
            reason=release.US_BASE_PERSON_POPULATION_REPAIR_REASON,
        ),
    )


def _base_population_repair_payload(
    *,
    initial_population: float,
    release,
) -> tuple[float, dict[str, object]]:
    benchmark = release.US_BASE_PERSON_POPULATION_BENCHMARK
    initial_relative_error = release._base_population_relative_error(initial_population)
    if initial_relative_error is None:
        raise RuntimeError(
            "Streaming state-file scoring requires positive finite weighted "
            f"person population; got {initial_population!r}."
        )
    factor = benchmark / initial_population
    repaired_population = initial_population * factor
    repaired_relative_error = release._base_population_relative_error(
        repaired_population
    )
    return factor, {
        "method": "streaming_rescale_household_weights_to_census_person_population",
        "applied": not np.isclose(factor, 1.0, rtol=1e-12, atol=0.0),
        "reason": release.US_BASE_PERSON_POPULATION_REPAIR_REASON,
        "initial_population": initial_population,
        "benchmark": benchmark,
        "factor": factor,
        "initial_relative_error": initial_relative_error,
        "repaired_population": repaired_population,
        "repaired_relative_error": repaired_relative_error,
    }


def _streaming_base_population_gate(
    repair: Mapping[str, object],
    release,
) -> dict[str, object]:
    relative_error = repair.get("repaired_relative_error")
    passed = (
        relative_error is not None
        and abs(float(relative_error))
        <= release.US_BASE_PERSON_POPULATION_MAX_ABS_RELATIVE_ERROR
    )
    failures = [] if passed else [f"relative error {relative_error!r} exceeds gate"]
    return {
        "passed": passed,
        "failures": failures,
        "details": {
            "method": "streaming_state_collection",
            "benchmark": repair.get("benchmark"),
            "relative_error": relative_error,
            "mass_repair": dict(repair),
        },
    }


def _score_state_files_streaming(
    *,
    state_h5_dir: Path,
    ledger_facts: Path,
    congressional_district_vintage_crosswalk: Path,
    maximum_microsim_batch_size: int,
    diagnostic_skip_tax_expenditure_targets: bool,
    allow_legacy_formula_owned_inputs: bool,
    allow_legacy_cd_provenance: bool,
    legacy_pe_flat_h5: bool,
    target_materialization_cache_dir: Path | None,
) -> tuple[Any, object, dict[str, object], dict[str, object]]:
    release, fiscal_scorer, _, _ = _load_runtime()
    congressional_district_vintage_crosswalk_metadata = {
        "path": str(congressional_district_vintage_crosswalk.resolve()),
        "sha256": release._sha256(congressional_district_vintage_crosswalk),
    }
    fiscal_scorer._assert_legacy_cd_provenance_options(
        allow_legacy_cd_provenance=allow_legacy_cd_provenance,
        congressional_district_vintage_crosswalk_metadata=(
            congressional_district_vintage_crosswalk_metadata
        ),
    )
    target_registry = release.compile_us_fiscal_target_registry(
        release._load_ledger_facts(ledger_facts),
        target_period=release.PERIOD,
        congressional_district_vintage_crosswalk=(
            release.load_congressional_district_vintage_crosswalk(
                congressional_district_vintage_crosswalk
            )
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

    root = _state_h5_root(state_h5_dir)
    state_files: list[dict[str, object]] = []
    initial_person_population = 0.0
    total_households = 0
    for state_abbreviation in EXPECTED_STATE_ABBREVIATIONS:
        path = root / f"{state_abbreviation}.h5"
        print(
            f"Pre-scanning legacy state file {state_abbreviation}: {path}", flush=True
        )
        frame, metadata = _load_state_frame(
            path,
            release=release,
            fiscal_scorer=fiscal_scorer,
            state_abbreviation=state_abbreviation,
            legacy_pe_flat_h5=legacy_pe_flat_h5,
            allow_legacy_cd_provenance=allow_legacy_cd_provenance,
            congressional_district_vintage_crosswalk_metadata=(
                congressional_district_vintage_crosswalk_metadata
            ),
        )
        initial_person_population += release._person_population(frame)
        total_households += int(frame.n("household"))
        state_files.append(metadata)
        del frame
        _collect_batch_garbage()

    collection_sha256 = _collection_sha256(state_files)
    population_factor, base_population_repair = _base_population_repair_payload(
        initial_population=initial_person_population,
        release=release,
    )
    base_population_gate = _streaming_base_population_gate(
        base_population_repair,
        release,
    )

    global_target_set = target_registry.to_target_set()
    global_targets = tuple(global_target_set.targets)
    global_name_to_index = {
        _row_name(target): index for index, target in enumerate(global_targets)
    }
    estimates = np.zeros(len(global_targets), dtype=np.float64)
    target_compilations: list[dict[str, object]] = []
    legacy_formula_owned_inputs: dict[str, dict[str, list[str]]] = {}
    for state_file in state_files:
        state_abbreviation = str(state_file["state"])
        path = Path(str(state_file["path"]))
        print(f"Scoring legacy state file {state_abbreviation}: {path}", flush=True)
        frame, _ = _load_state_frame(
            path,
            release=release,
            fiscal_scorer=fiscal_scorer,
            state_abbreviation=state_abbreviation,
            legacy_pe_flat_h5=legacy_pe_flat_h5,
            allow_legacy_cd_provenance=allow_legacy_cd_provenance,
            congressional_district_vintage_crosswalk_metadata=(
                congressional_district_vintage_crosswalk_metadata
            ),
        )
        if allow_legacy_formula_owned_inputs:
            frame, dropped = fiscal_scorer._drop_legacy_formula_owned_inputs(frame)
            if dropped:
                legacy_formula_owned_inputs[state_abbreviation] = dropped
        frame = _apply_household_weight_factor(
            frame,
            factor=population_factor,
            release=release,
        )
        cache_context = (
            {
                "base_dataset_sha256": state_file["sha256"],
                "build_commit": release._git_output("rev-parse", "HEAD"),
                "policyengine_us_version": release._package_or_workspace_version(
                    "policyengine-us"
                ),
                "seed": 0,
                "target_period": release.PERIOD,
                "target_registry_version": target_registry.version,
                "state_file_collection": collection_sha256,
                "state": state_abbreviation,
                "congressional_district_vintage_crosswalk_sha256": (
                    congressional_district_vintage_crosswalk_metadata["sha256"]
                ),
                "legacy_cd_provenance_allowed": bool(allow_legacy_cd_provenance),
                "streaming_state_collection": True,
            }
            if target_materialization_cache_dir is not None
            else None
        )
        state_target_specs = _target_specs_for_state(target_specs, state_abbreviation)
        target_frame, registry, compilation = release._materialize_target_frame(
            frame,
            state_target_specs,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
            target_materialization_cache_dir=target_materialization_cache_dir,
            target_materialization_cache_context=cache_context,
        )
        release._assert_complete_target_compilation(
            compilation,
            context=f"State-file fiscal target materialization for {state_abbreviation}",
        )
        target_set = registry.to_target_set()
        local_estimates = _target_estimates(target_frame, target_set)
        for local_index, target in enumerate(target_set.targets):
            name = _row_name(target)
            try:
                global_index = global_name_to_index[name]
            except KeyError as exc:
                raise RuntimeError(
                    "Streaming state target registry produced target "
                    f"{name!r} for {state_abbreviation}, which is not in the "
                    "global target surface."
                ) from exc
            estimates[global_index] += local_estimates[local_index]
        target_compilations.append(
            {
                "state": state_abbreviation,
                "declared_global_targets": len(target_specs),
                "declared_targets": compilation["declared_targets"],
                "compiled_candidate_targets": compilation["compiled_candidate_targets"],
                "dropped_target_names": compilation["dropped_target_names"],
                "target_materialization_cache": compilation.get(
                    "target_materialization_cache"
                ),
            }
        )
        del frame, target_frame
        _collect_batch_garbage()

    result = _streaming_score_result(
        registry=target_registry,
        estimates=estimates,
        n_records=total_households,
        release=release,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    gates = {
        "target_profile_coverage": {
            "passed": target_profile_gate.passed,
            "failures": list(target_profile_gate.failures),
            "details": dict(target_profile_gate.details),
        },
        "health_input_signal": {
            "passed": True,
            "failures": [],
            "details": {
                "method": "not_evaluated_for_streaming_legacy_state_collection",
                "reason": (
                    "The memory-bounded legacy scorer evaluates target fit "
                    "state by state and does not build the national frame "
                    "required by the health input signal gate."
                ),
            },
        },
        "base_population_scale": base_population_gate,
    }
    compilation = {
        "declared_targets": len(target_specs),
        "compiled_candidate_targets": len(target_registry.specs),
        "dropped_target_names": [],
        "streamed_state_target_compilations": target_compilations,
        "scored_dataset": {
            "layout": (
                "streamed_legacy_pe_flat_state_h5_collection"
                if legacy_pe_flat_h5
                else "streamed_state_h5_collection"
            ),
            "state_h5_dir": str(root),
            "state_file_collection_sha256": collection_sha256,
            "state_files": state_files,
            "congressional_district_vintage_crosswalk": {
                "path": congressional_district_vintage_crosswalk_metadata["path"],
                "sha256": congressional_district_vintage_crosswalk_metadata["sha256"],
            },
            "legacy_cd_provenance_allowed": bool(allow_legacy_cd_provenance),
            "streaming": True,
        },
        "base_population_repair": base_population_repair,
    }
    if legacy_formula_owned_inputs:
        compilation = {
            **compilation,
            "legacy_formula_owned_inputs_dropped": legacy_formula_owned_inputs,
        }
    return result, target_registry, compilation, gates


def score_state_files(
    *,
    state_h5_dir: Path,
    ledger_facts: Path,
    congressional_district_vintage_crosswalk: Path,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    diagnostic_skip_tax_expenditure_targets: bool = False,
    allow_legacy_formula_owned_inputs: bool = False,
    allow_legacy_cd_provenance: bool = False,
    legacy_pe_flat_h5: bool = False,
    target_materialization_cache_dir: Path | None = None,
) -> tuple[Any, object, dict[str, object], dict[str, object]]:
    assert_required_state_files_present(required_state_file_inventory(state_h5_dir))
    release, _, _, _ = _load_runtime()
    if maximum_microsim_batch_size is None:
        maximum_microsim_batch_size = release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    return _score_state_files_streaming(
        state_h5_dir=state_h5_dir,
        ledger_facts=ledger_facts,
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
        maximum_microsim_batch_size=maximum_microsim_batch_size,
        diagnostic_skip_tax_expenditure_targets=diagnostic_skip_tax_expenditure_targets,
        allow_legacy_formula_owned_inputs=allow_legacy_formula_owned_inputs,
        allow_legacy_cd_provenance=allow_legacy_cd_provenance,
        legacy_pe_flat_h5=legacy_pe_flat_h5,
        target_materialization_cache_dir=target_materialization_cache_dir,
    )


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
        congressional_district_vintage_crosswalk=(
            args.congressional_district_vintage_crosswalk
        ),
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        diagnostic_skip_tax_expenditure_targets=(
            args.diagnostic_skip_tax_expenditure_targets
        ),
        allow_legacy_formula_owned_inputs=args.allow_legacy_formula_owned_inputs,
        allow_legacy_cd_provenance=args.allow_legacy_cd_provenance,
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
