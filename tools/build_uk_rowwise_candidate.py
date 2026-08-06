"""Build the first calibrated UK rowwise candidate (#495 increment 6d).

This command is the narrow adjudicated-partial candidate path from a national
staging H5 to a ladder-assigned, rowwise-calibrated artifact. It binds exactly
one target family, ``census_households/constituency``, using the constant-one
household metric and constituency household controls derived from the same
loaded OA ladder used for assignment. The solve always runs under the reviewed
UK local doctrine; no per-target or doctrine override is exposed.

The command appends the canonical calibration mass-change record, advances the
household weight kind to ``CALIBRATED``, re-gates the calibrated frame, and
publishes the atomic H5 with diagnostics and manifest evidence. A dry run runs
the load, clone, gate, pairing, and matrix fences in memory, then prints a plan
without solving or writing any file.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.uk_runtime import (
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_TARGET_LOSS_CAP,
    StackedLocalSolveResult,
    UKLadderRowwiseDatasetResult,
    UkOaLadder,
    UKRowwiseLocalMatrix,
    build_uk_rowwise_local_matrix,
    clone_uk_dataset_with_ladder_geography,
    constituency_household_targets,
    ladder_target_provenance,
    load_uk_national_frame,
    load_uk_oa_ladder,
    rowwise_area_support_summary,
    rowwise_calibration_mass_record,
    solve_uk_rowwise_weights_under_doctrine,
    uk_geography_ladder_gate,
    uk_time_period,
    write_uk_rowwise_dataset,
)
from populace.frame import MassChangeRecord, WeightKind, assert_kind_transition

BOUND_TARGET_FAMILIES = ("census_households/constituency",)
CANDIDATE_FILENAME_TEMPLATE = "populace_uk_{source_year}_rowwise_candidate.h5"
MANIFEST_FILENAME = "rowwise_candidate_manifest.json"
SOLVE_DIAGNOSTICS_FILENAME = "solve_diagnostics.csv"
AREA_SUPPORT_FILENAME = "area_support_summary.csv"
PAST_CAP_FILENAME = "past_cap_census.json"

_CONSERVE_MASS = False
_TARGET_RECORDS: int | None = None
_L0_LAMBDA = 0.0
_MIN_INITIAL_WEIGHT = 1e-4
_BUDGET_ITERS = 10
_PAST_CAP_COUNT_KEYS = (
    "n_targets",
    "past_at_init",
    "past_at_final",
    "escaped",
    "frozen",
    "pushed_out",
)


class _LadderAssignment:
    """A clone paired in memory with the exact ladder object that produced it."""

    def __init__(
        self,
        result: UKLadderRowwiseDatasetResult,
        ladder: UkOaLadder,
    ) -> None:
        self.result = result
        self.ladder = ladder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="National Populace UK staging H5.",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        required=True,
        help="Full-UK OA geography ladder NPZ.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the candidate H5 and evidence sidecars.",
    )
    parser.add_argument("--n-clones", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-year", type=int)
    parser.add_argument("--source-lineage-modulus", type=int)
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument(
        "--expected-constituency-vintage",
        default="2024_pcon",
        help="Constituency vintage required from the ladder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the fenced clone/matrix plan without solving or writing any file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the rowwise candidate build."""

    args = _parse_args(argv)
    _validate_cli_args(args)
    input_h5 = _require_file(args.input_h5, label="--input-h5")
    ladder_path = _require_file(args.ladder, label="--ladder")
    out_dir = args.out.expanduser().resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out must be a directory path, got {out_dir}.")

    input_artifact = _artifact_info(input_h5)
    ladder_artifact = _artifact_info(ladder_path)
    national_frame, _national_provenance = load_uk_national_frame(input_h5)
    source_year = _source_year(
        args.source_year,
        time_period=uk_time_period(national_frame),
    )
    output_paths = _output_paths(out_dir, source_year=source_year)
    _validate_output_paths(
        output_paths,
        input_h5=input_h5,
        ladder_path=ladder_path,
    )
    ladder = load_uk_oa_ladder(ladder_path)
    target_provenance = ladder_target_provenance(ladder)

    print("cloning through the ladder route...", file=sys.stderr, flush=True)
    assignment = _clone_with_ladder_binding(
        national_frame,
        ladder,
        n_clones=args.n_clones,
        seed=args.seed,
        source_year=source_year,
        expected_constituency_vintage=args.expected_constituency_vintage,
        source_lineage_modulus=args.source_lineage_modulus,
    )
    clone = assignment.result

    print("binding census household targets...", file=sys.stderr, flush=True)
    household, problem = _build_bound_problem(
        assignment,
        target_ladder=ladder,
    )

    if args.dry_run:
        _assert_artifacts_unchanged(
            input_h5=input_h5,
            input_artifact=input_artifact,
            ladder_path=ladder_path,
            ladder_artifact=ladder_artifact,
        )
        plan = _dry_run_plan(
            args,
            clone=clone,
            problem=problem,
            source_year=source_year,
            input_artifact=input_artifact,
            ladder_artifact=ladder_artifact,
            target_provenance=target_provenance,
        )
        print(_json_text(plan), end="")
        return 0

    print(
        f"solving {problem.matrix.shape[0]} targets x "
        f"{problem.matrix.shape[1]} households under the doctrine...",
        file=sys.stderr,
        flush=True,
    )
    base_weights = household["household_weight"].to_numpy(dtype=np.float64)
    solve = solve_uk_rowwise_weights_under_doctrine(
        problem,
        base_weights,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        conserve_mass=_CONSERVE_MASS,
        target_records=_TARGET_RECORDS,
        l0_lambda=_L0_LAMBDA,
        min_initial_weight=_MIN_INITIAL_WEIGHT,
        budget_iters=_BUDGET_ITERS,
        seed=args.seed,
    )
    _validate_solve_result(solve, problem=problem)

    calibration_record = rowwise_calibration_mass_record(
        base_weights,
        solve.weights,
        bound_families=BOUND_TARGET_FAMILIES,
    )
    calibrated_household = household.copy()
    calibrated_household["household_weight"] = solve.weights
    candidate_gate = uk_geography_ladder_gate(
        calibrated_household,
        np.asarray(solve.weights, dtype=np.float64),
    )
    if not candidate_gate.passed:
        raise ValueError(
            "UK geography ladder gate failed on calibrated candidate weights: "
            + "; ".join(candidate_gate.failures)
        )

    assert_kind_transition(
        clone.household_weight_kind,
        WeightKind.CALIBRATED,
    )
    candidate = dataclasses.replace(
        clone,
        household=calibrated_household,
        gate=candidate_gate,
        household_weight_kind=WeightKind.CALIBRATED,
        mass_log=(*clone.mass_log, calibration_record),
        output_path=None,
    )
    support = rowwise_area_support_summary(
        problem,
        solve.weights,
        source_household_ids=household["source_household_id"].tolist(),
    )
    _validate_support_summary(support)
    _assert_artifacts_unchanged(
        input_h5=input_h5,
        input_artifact=input_artifact,
        ladder_path=ladder_path,
        ladder_artifact=ladder_artifact,
    )

    manifest = _write_output_bundle(
        args,
        candidate=candidate,
        clone=clone,
        problem=problem,
        solve=solve,
        support=support,
        calibration_record=calibration_record,
        source_year=source_year,
        output_paths=output_paths,
        input_artifact=input_artifact,
        ladder_artifact=ladder_artifact,
        target_provenance=target_provenance,
    )
    print(_json_text(manifest), end="")
    return 0


def _clone_with_ladder_binding(
    dataset: Any,
    ladder: UkOaLadder,
    *,
    n_clones: int,
    seed: int,
    source_year: int,
    expected_constituency_vintage: str | None,
    source_lineage_modulus: int | None,
) -> _LadderAssignment:
    clone = clone_uk_dataset_with_ladder_geography(
        dataset,
        ladder,
        n_clones=n_clones,
        seed=seed,
        source_year=source_year,
        expected_constituency_vintage=expected_constituency_vintage,
        source_lineage_modulus=source_lineage_modulus,
    )
    return _LadderAssignment(clone, ladder)


def _build_bound_problem(
    assignment: _LadderAssignment,
    *,
    target_ladder: UkOaLadder,
) -> tuple[pd.DataFrame, UKRowwiseLocalMatrix]:
    """Bind the one target family, refusing separately loaded ladders."""

    if assignment.ladder is not target_ladder:
        raise ValueError(
            "assignment and targets must come from the same loaded UK OA ladder object."
        )
    clone = assignment.result
    household = clone.household.reset_index(drop=True)
    household_index = pd.Index(
        household["household_id"],
        name="household_id",
    )
    metrics = pd.DataFrame(
        {"households": np.ones(len(household), dtype=np.float64)},
        index=household_index,
    )
    assigned = pd.Series(
        household["constituency_code"].astype(str).to_numpy(),
        index=household_index,
        name="constituency_code",
    )
    targets = constituency_household_targets(target_ladder)
    problem = build_uk_rowwise_local_matrix(
        metrics,
        assigned,
        targets,
        area_type="constituency",
        code_column="code",
    )
    return household, problem


def _dry_run_plan(
    args: argparse.Namespace,
    *,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "build_kind": "uk_rowwise_calibrated_candidate_plan",
        "dry_run": True,
        "candidate_scope": "adjudicated_partial",
        "bound_target_families": list(BOUND_TARGET_FAMILIES),
        "ladder_target_provenance": dict(target_provenance),
        "inputs": {
            "dataset": dict(input_artifact),
            "ladder": dict(ladder_artifact),
        },
        "parameters": _parameters(args, source_year=source_year),
        "shapes": {
            "person": list(clone.person.shape),
            "benunit": list(clone.benunit.shape),
            "household": list(clone.household.shape),
            "local_matrix": list(problem.matrix.shape),
        },
        "target_count": int(len(problem.targets)),
        "gate": _gate_payload(clone.gate, phase="post_clone"),
    }


def _write_output_bundle(
    args: argparse.Namespace,
    *,
    candidate: UKLadderRowwiseDatasetResult,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    solve: StackedLocalSolveResult,
    support: pd.DataFrame,
    calibration_record: MassChangeRecord,
    source_year: int,
    output_paths: Mapping[str, Path],
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage the complete bundle, then publish atomically per file."""

    out_dir = output_paths["manifest"].parent
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{out_dir.name}.rowwise-candidate.",
            dir=out_dir.parent,
        )
    )
    try:
        staged = {key: staging_dir / path.name for key, path in output_paths.items()}
        print(
            f"staging candidate for {output_paths['dataset']}...",
            file=sys.stderr,
            flush=True,
        )
        write_uk_rowwise_dataset(candidate, staged["dataset"])
        solve.diagnostics.to_csv(staged["diagnostics"], index=False)
        support.to_csv(staged["support"], index=False)
        staged["past_cap"].write_text(_json_text(dict(solve.past_cap_census or {})))

        outputs = {
            "dataset": _artifact_info(
                staged["dataset"],
                reported_path=output_paths["dataset"],
            ),
            "solve_diagnostics": _artifact_info(
                staged["diagnostics"],
                reported_path=output_paths["diagnostics"],
            ),
            "area_support_summary": _artifact_info(
                staged["support"],
                reported_path=output_paths["support"],
            ),
            "past_cap_census": _artifact_info(
                staged["past_cap"],
                reported_path=output_paths["past_cap"],
            ),
        }
        manifest = _manifest(
            args,
            candidate=candidate,
            clone=clone,
            problem=problem,
            solve=solve,
            support=support,
            calibration_record=calibration_record,
            source_year=source_year,
            input_artifact=input_artifact,
            ladder_artifact=ladder_artifact,
            target_provenance=target_provenance,
            outputs=outputs,
        )
        staged["manifest"].write_text(_json_text(manifest))
        _publish_staged_files(staged, output_paths)
        return manifest
    finally:
        shutil.rmtree(staging_dir)


def _manifest(
    args: argparse.Namespace,
    *,
    candidate: UKLadderRowwiseDatasetResult,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    solve: StackedLocalSolveResult,
    support: pd.DataFrame,
    calibration_record: MassChangeRecord,
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    abs_errors = solve.diagnostics["abs_relative_error"].to_numpy(dtype=np.float64)
    old_total = float(calibration_record.old_total)
    new_total = float(calibration_record.new_total)
    past_cap = dict(solve.past_cap_census or {})
    return {
        "schema_version": 1,
        "build_kind": "uk_rowwise_calibrated_candidate",
        "candidate_scope": "adjudicated_partial",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "bound_target_families": list(BOUND_TARGET_FAMILIES),
        "ladder_target_provenance": dict(target_provenance),
        "parameters": _parameters(args, source_year=source_year),
        "inputs": {
            "dataset": dict(input_artifact),
            "ladder": dict(ladder_artifact),
        },
        "outputs": dict(outputs),
        "gate": _gate_payload(candidate.gate, phase="post_calibration"),
        "weights": {
            "household_weight_kind": candidate.household_weight_kind.value,
            "household_weight_kind_chain": [
                {
                    "stage": "staging",
                    "kind": clone.household_weight_kind.value,
                },
                {
                    "stage": "ladder_clone",
                    "kind": clone.household_weight_kind.value,
                },
                {
                    "stage": "rowwise_calibration",
                    "kind": candidate.household_weight_kind.value,
                },
            ],
            "mass_log_records_before_calibration": len(clone.mass_log),
            "mass_log_records": len(candidate.mass_log),
            "calibration_mass_change": {
                "entity": str(calibration_record.entity),
                "old_total": old_total,
                "new_total": new_total,
                "relative_shift": (new_total - old_total) / old_total,
                "declared_factor": calibration_record.declared_factor,
                "reason": str(calibration_record.reason),
            },
        },
        "solve": {
            "n_targets": int(len(problem.targets)),
            "n_households": int(problem.n_households),
            "initial_loss": float(solve.initial_loss),
            "final_loss": float(solve.final_loss),
            "max_abs_relative_error": float(abs_errors.max()),
            "median_abs_relative_error": float(np.median(abs_errors)),
            "n_nonzero": int(solve.n_nonzero),
            "past_cap": {key: int(past_cap[key]) for key in _PAST_CAP_COUNT_KEYS},
        },
        "support": {
            "min_assigned_households": int(support["assigned_households"].min()),
            "min_nonzero_households": int(support["nonzero_households"].min()),
            "min_effective_sample_size": float(support["effective_sample_size"].min()),
        },
    }


def _parameters(args: argparse.Namespace, *, source_year: int) -> dict[str, Any]:
    return {
        "n_clones": int(args.n_clones),
        "seed": int(args.seed),
        "source_year": source_year,
        "source_lineage_modulus": args.source_lineage_modulus,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "expected_constituency_vintage": str(args.expected_constituency_vintage),
        "doctrine": _doctrine_bounds(),
        "solve_options": {
            "conserve_mass": _CONSERVE_MASS,
            "target_records": _TARGET_RECORDS,
            "l0_lambda": _L0_LAMBDA,
            "min_initial_weight": _MIN_INITIAL_WEIGHT,
            "budget_iters": _BUDGET_ITERS,
        },
    }


def _doctrine_bounds() -> dict[str, Any]:
    return {
        "target_loss_cap": float(UK_LOCAL_TARGET_LOSS_CAP),
        "max_weight_ratio": float(UK_LOCAL_MAX_WEIGHT_RATIO),
        "scale_rule": UK_LOCAL_SOLVE_DOCTRINE.scale_rule,
        "target_weight_rule": UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
    }


def _gate_payload(gate: GateResult, *, phase: str) -> dict[str, Any]:
    return {
        "name": str(gate.name),
        "passed": bool(gate.passed),
        "failures": list(gate.failures),
        "details": dict(gate.details),
        "phase": phase,
    }


def _validate_solve_result(
    solve: StackedLocalSolveResult,
    *,
    problem: UKRowwiseLocalMatrix,
) -> None:
    if solve.past_cap_census is None:
        raise RuntimeError(
            "doctrine solve returned no past-cap census; refusing candidate."
        )
    if len(solve.weights) != problem.n_households:
        raise RuntimeError(
            "doctrine solve returned a weight vector with the wrong length."
        )
    weights = np.asarray(solve.weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError(
            "doctrine solve returned non-finite or negative household weights."
        )
    if not np.isfinite([solve.initial_loss, solve.final_loss]).all():
        raise RuntimeError("doctrine solve returned a non-finite loss.")
    errors = solve.diagnostics["abs_relative_error"].to_numpy(dtype=np.float64)
    if len(errors) != len(problem.targets) or not np.isfinite(errors).all():
        raise RuntimeError(
            "doctrine solve returned incomplete or non-finite diagnostics."
        )
    missing_counts = sorted(set(_PAST_CAP_COUNT_KEYS) - set(solve.past_cap_census))
    if missing_counts:
        raise RuntimeError(
            f"past-cap census is missing count field(s): {missing_counts}."
        )


def _validate_support_summary(support: pd.DataFrame) -> None:
    required = {
        "assigned_households",
        "nonzero_households",
        "effective_sample_size",
    }
    missing = sorted(required - set(support.columns))
    if missing or support.empty:
        raise RuntimeError(
            f"area support summary is empty or missing required columns: {missing}."
        )
    values = support[list(required)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("area support summary contains invalid values.")


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.n_clones <= 0:
        raise ValueError("--n-clones must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.source_year is not None and args.source_year <= 0:
        raise ValueError("--source-year must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite.")
    if not str(args.expected_constituency_vintage).strip():
        raise ValueError("--expected-constituency-vintage must be non-empty.")


def _output_paths(out_dir: Path, *, source_year: int) -> dict[str, Path]:
    return {
        "dataset": out_dir
        / CANDIDATE_FILENAME_TEMPLATE.format(source_year=source_year),
        "manifest": out_dir / MANIFEST_FILENAME,
        "diagnostics": out_dir / SOLVE_DIAGNOSTICS_FILENAME,
        "support": out_dir / AREA_SUPPORT_FILENAME,
        "past_cap": out_dir / PAST_CAP_FILENAME,
    }


def _validate_output_paths(
    output_paths: Mapping[str, Path],
    *,
    input_h5: Path,
    ladder_path: Path,
) -> None:
    resolved = {name: path.resolve() for name, path in output_paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("candidate output paths must be distinct.")
    protected = {input_h5.resolve(), ladder_path.resolve()}
    collisions = sorted(str(path) for path in resolved.values() if path in protected)
    if collisions:
        raise ValueError(
            "candidate outputs must differ from --input-h5 and --ladder; "
            f"collision(s): {collisions}."
        )
    existing = sorted(str(path) for path in resolved.values() if path.exists())
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing candidate artifact(s): {existing}."
        )


def _publish_staged_files(
    staged: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> None:
    out_dir = output_paths["manifest"].parent
    created_out_dir = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    publish_order = (
        "dataset",
        "diagnostics",
        "support",
        "past_cap",
        "manifest",
    )
    published: list[Path] = []
    succeeded = False
    try:
        for key in publish_order:
            destination = output_paths[key]
            if destination.exists():
                raise FileExistsError(
                    "candidate output appeared during publication; refusing "
                    f"to overwrite {destination}."
                )
            staged[key].replace(destination)
            published.append(destination)
        succeeded = True
    finally:
        if not succeeded:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            if created_out_dir:
                try:
                    out_dir.rmdir()
                except OSError:
                    pass


def _assert_artifacts_unchanged(
    *,
    input_h5: Path,
    input_artifact: Mapping[str, Any],
    ladder_path: Path,
    ladder_artifact: Mapping[str, Any],
) -> None:
    for label, path, before in (
        ("input H5", input_h5, input_artifact),
        ("ladder", ladder_path, ladder_artifact),
    ):
        after = _artifact_info(path)
        if after["sha256"] != before["sha256"] or after["bytes"] != before["bytes"]:
            raise RuntimeError(
                f"{label} changed during the candidate build; refusing to "
                "bind mixed source bytes."
            )


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {resolved}.")
    return resolved


def _source_year(requested: int | None, *, time_period: str) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--source-year must be positive.")
        return requested
    prefix = str(time_period).strip()[:4]
    if len(prefix) != 4 or not prefix.isdigit():
        raise ValueError(
            "Could not infer source year from input H5 time_period; pass --source-year."
        )
    return int(prefix)


def _artifact_info(
    path: Path,
    *,
    reported_path: Path | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str((reported_path or path).resolve()),
        "sha256": digest.hexdigest(),
        "bytes": int(path.stat().st_size),
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _json_text(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
