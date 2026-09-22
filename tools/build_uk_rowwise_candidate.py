"""Build a joint local, ladder, and national UK rowwise candidate.

Pinned Ledger facts supply the local and national registries. The command
samples before cloning, resolves both local grains and national measures on the
cloned frame, and calibrates every row in one doctrine solve. A dry run compiles
the registries and reports analytical matrix/support evidence without running
the policy engine, solving, or writing output files.

The pinned Ledger arguments are mandatory; ``--households-only`` binds only
the Chronicle census-household constituency targets from the same registry.

``--release-role`` declares which UK dataset line the run builds and is
required: ``dense`` is the joint K-clone surface described above under the
local doctrine; ``national`` builds the certified national line without
cloning, on national targets only, under the calibration-seam doctrine
(microcosm#823). The role fixes every solve default and refuses the other
role's flags, so the declared role is checked against the parameters.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
)
from microcosm.build.gates import GateResult
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    atomic_write_json,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
    write_error_receipt,
)
from microcosm.build.staging_cli import (
    add_staged_dataset_arguments,
    add_staging_arguments,
    validate_staged_dataset_arguments,
    validate_staging_arguments,
)
from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    StagedDatasetBundle,
    disabled_staged_dataset,
    local_only_staged_dataset,
    parse_sha256sums,
    refresh_sha256sums_entry,
    stage_bundle,
    write_sidecars,
)
from microcosm.build.staging_storage import HuggingFaceDatasetStorage
from microcosm.build.staging_v2 import (
    StagingContractError,
    StagingTelemetryV2,
    disabled_staging_delivery,
)
from microcosm.build.target_materialization import resolve_target_measures
from microcosm.build.uk_runtime import (
    UK_GATE_REGISTRY,
    CalibrationFrameAdapter,
    UKLadderRowwiseDatasetResult,
    UkOaLadder,
    UKRowwiseDoctrineSolve,
    UKRowwiseLocalMatrix,
    UKRowwiseNationalRows,
    build_uk_rowwise_local_matrix,
    build_uk_rowwise_local_surface_matrix,
    clone_uk_dataset_with_ladder_geography,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    compute_household_metrics,
    drop_injected_measure_inputs,
    inject_measure_inputs,
    ladder_clone_index_column,
    ladder_target_provenance,
    ladder_vs_chronicle_household_dispersion,
    load_bound_spine_sidecar,
    load_uk_local_area_crosswalk,
    load_uk_national_frame,
    load_uk_oa_ladder,
    local_target_census,
    materialize_uk_ledger_targets,
    require_adjudicated_uk_local_binding,
    rotated_uk_local_holdout,
    runtime_provenance,
    solve_uk_rowwise_weights_under_doctrine,
    spine_provenance_from_sidecar,
    uk_area_region_codes,
    uk_census_household_uprating,
    uk_fit_by_family,
    uk_household_weight_kind,
    uk_ladder_area_support_summary,
    uk_ledger_households_total,
    uk_local_doctrine_with_overrides,
    uk_local_target_surface,
    uk_support_limited_misses,
    uk_time_period,
    uk_weight_summary,
    write_uk_calibration_diagnostics,
    write_uk_rowwise_dataset,
)
from microcosm.build.uk_runtime.calibration_run import (
    UK_LOCAL_GATE_SCOPE,
    UKCalibrationRunPaths,
    finalize_uk_scoped_gate_report,
    new_uk_calibration_attempt_id,
    run_uk_calibration,
    uk_local_gate_scope_exclusions,
    uk_scoped_gate_manifest,
)
from microcosm.build.uk_runtime.chronicle_feed import (
    require_committed_uk_chronicle_feed_pin,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import _spec_geography
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_doctrine import uk_doctrine_with_overrides
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    UK_SAMPLE_SEED_DEFAULT,
    sample_uk_spine_frame,
)
from microcosm.build.uk_runtime.rowwise_posture import (
    UK_ROWWISE_DENSE_POSTURE,
    UK_ROWWISE_RELEASE_ROLES,
    UKRowwisePosture,
    uk_rowwise_posture,
)
from microcosm.build.uk_runtime.staging import (
    UK_STAGED_DATASET_PREFIX,
    UK_STAGED_DATASET_REPOSITORY,
    UK_STAGING_REPOSITORY,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import Frame, MassChangeRecord

BOUND_TARGET_FAMILIES = ("census_households/constituency",)
BOUND_NATIONAL_TARGETS: tuple[str, ...] = ()
MANIFEST_FILENAME = "rowwise_candidate_manifest.json"
SOLVE_DIAGNOSTICS_FILENAME = "solve_diagnostics.csv"
CALIBRATION_DIAGNOSTICS_FILENAME = "calibration_diagnostics.json"
AREA_SUPPORT_FILENAME = "area_support_summary.csv"
PAST_CAP_FILENAME = "past_cap_census.json"
LOCAL_REGISTRY_FILENAME = "local_target_registry.json"
#: National-role outputs (the calibration seam's evidence shape).
BUILD_RECORD_FILENAME = "build_record.json"
NATIONAL_REGISTRY_FILENAME = "national_target_registry.json"
NATIONAL_CONTRACT_REGISTRY_FILENAME = "national_contract_registry.json"
SCORE_RECEIPT_FILENAME = "score_vs_incumbent.json"
DENSE_REFERENCE_DIAGNOSTICS_FILENAME = "dense_reference_diagnostics.csv"
DATASET_SIZE_SELECTION_FILENAME = "dataset_size_selection.csv"

#: Outputs a run writes only when ``--dataset-households`` is set.
_SIZE_RUN_ONLY_OUTPUTS = frozenset({"dense_reference", "selection"})

_CONSERVE_MASS = False
_TARGET_RECORDS: int | None = None
_L0_LAMBDA = 0.0
_BUDGET_ITERS = 10
# The dense role's Logbook pipeline and gate-policy suffix, kept as module
# names for the Logbook helpers and the contract-pin tests; the posture
# record (``rowwise_posture.py``) is the source of truth for both roles.
_UK_CANDIDATE_PIPELINE = UK_ROWWISE_DENSE_POSTURE.pipeline
_LOCAL_GATE_POLICY_SUFFIX = UK_ROWWISE_DENSE_POSTURE.gate_policy_suffix
# Best-effort telemetry upload cadence. The Hub allows about 128 commits per
# hour per repository and one cycle is up to eight single-file commits, so
# the shared 30-second default exhausts the budget on a multi-hour solve
# and loses uploads (the v20 national run did); five minutes keeps a
# 1,500-epoch run well inside it.
_STAGING_UPLOAD_INTERVAL_SECONDS = 300.0
# Staging telemetry keeps one row per forwarded epoch in
# calibration_progress.json and one event in events.ndjson, both under the
# contract's 5 MiB remote cap. A size run at 2,000 epochs solves the dense
# pool, up to ten full-length L0 probes and the refit: about 24,000 epochs,
# which would breach the cap mid-run. Forwarding every tenth epoch and the
# last epoch of each phase keeps the loss curve and stays near 0.8 MB.
_STAGING_EPOCH_EVERY = 10
# ...and never more than this many forwarded epochs per run, whatever --epochs
# says: the stride grows with the run so the cap holds by construction.
_STAGING_MAX_EPOCH_ROWS = 2400
_STAGED_DATASET_PHASES = {
    "uploaded": "dataset_staged",
    "already_staged": "dataset_staged",
    "failed": "dataset_stage_failed",
    "skipped": "dataset_stage_skipped",
}
_REPOSITORY = Path(__file__).resolve().parents[1]
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


def _new_candidate_build_id(
    *, seed: int, timestamp: datetime, rung: str = "f100"
) -> str:
    """The dense role's attempt id; the national role mints the seam's."""

    instant = timestamp.astimezone(UTC)
    return (
        f"{UK_ROWWISE_DENSE_POSTURE.build_id_prefix}{rung}-s{seed}-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _posture_of(args: argparse.Namespace) -> UKRowwisePosture:
    """The release-role posture bound to parsed arguments."""

    posture = getattr(args, "_posture", None)
    if not isinstance(posture, UKRowwisePosture):
        raise RuntimeError("arguments carry no release-role posture; parse them first.")
    return posture


def _candidate_clone_counts_argument(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not value.strip() or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "candidate clone counts must be a non-empty comma list of positive integers"
        )
    try:
        counts = [int(part.strip()) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate clone counts must be a comma list of positive integers"
        ) from error
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "candidate clone counts must all be positive integers"
        )
    return tuple(sorted(set(counts)))


def _sample_candidate_frame(
    frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Sample spine families below f100; keep the full rung untouched."""

    pre_count = len(frame.table("household"))
    if fraction == 1.0:
        return frame, {
            "fraction": 1.0,
            "seed": int(seed),
            "rung_token": UK_SAMPLE_RUNG_TOKENS[fraction],
            "sampled": False,
            "pre_household_count": int(pre_count),
            "post_household_count": int(pre_count),
        }

    sampled, receipt = sample_uk_spine_frame(
        frame,
        fraction=fraction,
        seed=seed,
    )
    return sampled, {"sampled": True, **receipt}


def _resolve_candidate_engine_surface(
    frame,
    national_registry,
    *,
    period: int,
    scratch_dir: Path,
    band_edge_registry=None,
    resolver_factory=UKMeasureResolver,
    blocks: int = 1,
) -> tuple[Any, Any, UKRowwiseNationalRows, dict[str, pd.DataFrame], dict[str, Any]]:
    """Resolve national inputs and local metrics on the cloned frame.

    ``blocks=1`` uses one scratch-mode engine for the whole clone.  The
    reviewed escape hatch ``blocks=K`` resolves each clone index separately,
    then rejoins every entity-level prepared column by its stable entity id so
    the full-frame target materialization and single solve retain frame order.
    """

    household = frame.table("household")
    if blocks < 1:
        raise ValueError("engine resolution blocks must be positive.")
    if blocks == 1:
        block_frames = [(None, frame)]
    else:
        clone_column = ladder_clone_index_column("household")
        if clone_column not in household.columns:
            raise ValueError(f"per-clone engine resolution requires {clone_column}.")
        clone_indices = tuple(sorted(household[clone_column].unique().tolist()))
        if len(clone_indices) != blocks:
            raise ValueError(
                "engine resolution blocks must match the realized clone indices: "
                f"requested {blocks}, found {clone_indices}."
            )
        person = frame.table("person")
        block_frames = []
        for clone_index in clone_indices:
            household_ids = set(
                household.loc[
                    household[clone_column] == clone_index,
                    "household_id",
                ].tolist()
            )
            person_mask = person["person_household_id"].isin(household_ids)
            block = frame.select(person_mask)
            # The block carries a K-th of the cloned mass while its log still
            # ends on the full-clone record, and the scratch export validates
            # the chain. Declare the subset explicitly: old = the cloned
            # total, new = the block total, reason naming the block. The block
            # frame is engine scratch and is discarded after resolution.
            block_weights = block.weights_for("household")
            full_total = float(frame.weights_for("household").total)
            block_total = float(block_weights.total)
            subset_record = MassChangeRecord(
                entity="household",
                old_total=full_total,
                new_total=block_total,
                declared_factor=block_total / full_total,
                reason=(
                    f"engine resolution block {clone_index} of {blocks}: "
                    "scratch subset of the cloned frame for measure "
                    "resolution only, discarded after resolution"
                ),
            )
            block = Frame(
                {
                    **{name: block.table(name) for name in block.entities},
                    **{name: block.link(name) for name in block.links},
                },
                block.schema,
                {
                    entity: block.weights_for(entity)
                    for entity in block.weighted_entities
                },
                block.strata,
                mass_log=(*block.mass_log, subset_record),
                metadata=block.metadata,
            )
            block_frames.append((clone_index, block))

    measure_parts: dict[tuple[str, str], list[pd.Series]] = {}
    metric_parts: dict[str, list[pd.DataFrame]] = {
        "constituency": [],
        "la": [],
    }
    resolver_receipts: list[Mapping[str, Any]] = []
    national_input_keys: set[tuple[str, str]] | None = None
    for clone_index, block_frame in block_frames:
        block_scratch = (
            scratch_dir if clone_index is None else scratch_dir / f"clone-{clone_index}"
        )
        resolver = resolver_factory(
            simulation_source=None,
            scratch_dir=block_scratch,
            year=period,
            frame=block_frame,
        )
        resolution = resolve_target_measures(
            lambda block_frame=block_frame: CalibrationFrameAdapter(block_frame),
            national_registry,
            resolver,
            period=period,
        )
        keys = set(resolution.measure_inputs)
        if national_input_keys is None:
            national_input_keys = keys
        elif keys != national_input_keys:
            raise RuntimeError(
                "per-clone engine resolution returned inconsistent national inputs."
            )
        for (entity, variable), values in resolution.measure_inputs.items():
            entity_table = block_frame.table(entity)
            entity_id = f"{entity}_id"
            measure_parts.setdefault((entity, variable), []).append(
                pd.Series(
                    np.asarray(values),
                    index=entity_table[entity_id].tolist(),
                )
            )
        block_household_ids = block_frame.table("household")["household_id"].tolist()
        for area_type in metric_parts:
            metric_parts[area_type].append(
                compute_household_metrics(
                    resolver.simulation,
                    area_type,
                    period=period,
                    household_ids=block_household_ids,
                )
            )
        resolver_receipts.append(resolver.receipt())
        del resolver
        simulation_input = block_scratch / "simulation-input.h5"
        simulation_input.unlink(missing_ok=True)
        try:
            block_scratch.rmdir()
        except OSError:
            pass

    measure_inputs: dict[tuple[str, str], np.ndarray] = {}
    for (entity, variable), parts in measure_parts.items():
        combined = pd.concat(parts)
        if combined.index.has_duplicates:
            raise RuntimeError(
                f"per-clone engine resolution duplicated {entity} ids for {variable}."
            )
        ordered_ids = frame.table(entity)[f"{entity}_id"]
        ordered = combined.reindex(ordered_ids.tolist())
        if ordered.isna().any():
            raise RuntimeError(
                f"per-clone engine resolution missed {entity} rows for {variable}."
            )
        measure_inputs[(entity, variable)] = ordered.to_numpy()

    full_household_ids = household["household_id"].tolist()
    local_metrics = {}
    for area_type, parts in metric_parts.items():
        combined = pd.concat(parts)
        if combined.index.has_duplicates:
            raise RuntimeError(
                f"per-clone engine resolution duplicated {area_type} household ids."
            )
        ordered = combined.reindex(full_household_ids)
        if ordered.isna().any().any():
            raise RuntimeError(
                f"per-clone engine resolution missed {area_type} household rows."
            )
        local_metrics[area_type] = ordered

    adapter = CalibrationFrameAdapter(frame)
    # Injected engine inputs are scratch state for materialization only:
    # they must be dropped before the prepared frame is assembled, or the
    # flattening rule refuses columns that now exist on two entities
    # (region, esa_* on the live spine). Same lifecycle as the national stage.
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    inject_measure_inputs(adapter, measure_inputs)
    materialized = materialize_uk_ledger_targets(
        adapter,
        national_registry,
        period=period,
        band_edge_registry=(
            national_registry if band_edge_registry is None else band_edge_registry
        ),
    )
    if materialized.skipped:
        raise RuntimeError(
            "candidate national target materialization skipped row(s): "
            f"{[skip.__dict__ for skip in materialized.skipped]}."
        )
    modes = {receipt.get("mode") for receipt in resolver_receipts}
    versions = {receipt.get("policyengine_uk_version") for receipt in resolver_receipts}
    if len(modes) != 1 or len(versions) != 1:
        raise RuntimeError("per-clone engine resolver provenance is inconsistent.")
    cgt_period_contract = resolver_receipts[0].get("cgt_period_contract")
    if any(
        block_receipt.get("cgt_period_contract") != cgt_period_contract
        for block_receipt in resolver_receipts[1:]
    ):
        raise RuntimeError("per-clone CGT period contract is inconsistent.")
    receipt = {
        "mode": next(iter(modes)),
        "engine_version": next(iter(versions)),
        "households": len(frame.table("household")),
        "persons": len(frame.table("person")),
        "benunits": len(frame.table("benunit")),
        "national_inputs": len(measure_inputs),
        "local_metrics": {
            area_type: len(metrics.columns)
            for area_type, metrics in local_metrics.items()
        },
        "blocks": blocks,
    }
    if cgt_period_contract is not None:
        receipt["cgt_period_contract"] = cgt_period_contract
    if blocks > 1:
        receipt["deviation"] = "per_clone_block_engine_resolution"
        present = sorted(
            column
            for column in UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            if column in measure_inputs
        )
        receipt["block_sensitivity"] = {
            "known_population_normalised_measures": list(
                UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            ),
            "present_in_this_run": present,
            "caveat": (
                "per-block engine resolution mis-measures population-normalised "
                "formulas (each block reproduces a national aggregate); rows "
                "on these measures are not evidence for adjudication from this "
                "run. Resolve in a single block before ruling on them."
            ),
        }
    try:
        scratch_dir.rmdir()
    except OSError:
        pass
    drop_injected_measure_inputs(adapter, measure_inputs, original_columns)
    national_rows = UKRowwiseNationalRows(
        targets=national_registry.to_target_set(),
        registry=national_registry,
        families=tuple(sorted({spec.family for spec in national_registry.specs})),
    )
    return (
        adapter.prepared_frame(),
        adapter.restore,
        national_rows,
        local_metrics,
        receipt,
    )


def _pin_from_artifact(info: Mapping[str, Any]) -> dict[str, object]:
    return {
        "sha256": str(info["sha256"]),
        "size_bytes": int(info["bytes"]),
    }


def _stderr_progress(line: str) -> None:
    """Solver progress (epoch losses, budget probes, the search verdict)."""
    print(line, file=sys.stderr, flush=True)


def _refuse_stale_size_checkpoint(args: argparse.Namespace, out_dir: Path) -> None:
    """Refuse an --out holding a checkpoint before the solve, not after it.

    The checkpoint writer refuses to overwrite, but it runs after the dense
    solve and the search; a stale checkpoint in --out must fail here, before
    the hours are spent.
    """
    if args.dataset_households is None or args.no_size_checkpoint:
        return
    if args.resume_size_checkpoint is not None:
        return
    from microcosm.build.uk_runtime.size_checkpoint import (
        SIZE_CHECKPOINT_ARRAYS_FILENAME,
        SIZE_CHECKPOINT_MANIFEST_FILENAME,
    )

    existing = sorted(
        str(out_dir / name)
        for name in (SIZE_CHECKPOINT_ARRAYS_FILENAME, SIZE_CHECKPOINT_MANIFEST_FILENAME)
        if (out_dir / name).exists()
    )
    if existing:
        raise FileExistsError(
            "refusing to run into an --out that already holds a size checkpoint: "
            f"{existing}. Resume from it with --resume-size-checkpoint, or choose "
            "another --out."
        )


def _size_checkpoint_identity(
    args: argparse.Namespace,
    *,
    pins: Mapping[str, Mapping[str, object]],
    source_year: int,
) -> dict[str, object]:
    """Everything a size checkpoint must share with the run that resumes it.

    The pool (spine, ladder, clones, seed, sampling), the target surface
    (ledger digests, year, rule, engine blocks) and the solve settings the
    checkpointed dense solve and search were made with. The draw threshold is
    deliberately absent: re-drawing at another threshold is the point.
    """
    posture = _posture_of(args)
    return {
        "release_role": posture.role,
        "dataset_pin": dict(pins["dataset"]),
        "ladder_pin": dict(pins["ladder"]),
        "ledger_facts_sha256": args.ledger_facts_sha256,
        "ledger_manifest_sha256": args.ledger_manifest_sha256,
        "seed": int(args.seed),
        "selection_seed": int(
            args.seed if args.selection_seed is None else args.selection_seed
        ),
        "n_clones": None if args.n_clones is None else int(args.n_clones),
        "dataset_households": args.dataset_households,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "sample_fraction": float(args.sample_fraction),
        "sample_seed": int(args.sample_seed),
        "source_year": int(source_year),
        "source_lineage_modulus": args.source_lineage_modulus,
        "calibration_year": getattr(args, "_calibration_year", None),
        "target_weight_rule": args.target_weight_rule,
        "engine_blocks": int(args.engine_blocks),
        "measure_exclusions": (
            None if args.measure_exclusions is None else str(args.measure_exclusions)
        ),
        # The solve doctrine the dense solve and the search run under: a
        # resume after a doctrine change must refuse, not run under the old
        # bound while the manifest declares the new one.
        "doctrine": _doctrine_bounds(posture),
    }


def _candidate_identity_digest(
    *,
    pins: dict[str, dict[str, object]],
    args: argparse.Namespace,
    source_year: int,
) -> str:
    payload = {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "inputs": pins,
        "parameters": _parameters(args, source_year=source_year),
        "source_year": source_year,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record_candidate_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    spool_dir: Path,
    rung: str = "f100",
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_UK_CANDIDATE_PIPELINE,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _record_candidate_error(
    *,
    error: BaseException,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int,
    code_pin: str,
    predecessor: str | None,
    base_dir: Path,
    spool_dir: Path,
    rung: str = "f100",
) -> None:
    error_path = write_error_receipt(
        error_receipt_path(base_dir, build_id=state.build_id),
        state=state,
        pipeline=_UK_CANDIDATE_PIPELINE,
        error=error,
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(error_path, repository_hint=_REPOSITORY)}#/error_type",
    )
    _record_candidate_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        seed=seed,
        code_pin=code_pin,
        disposition="failed",
        predecessor=predecessor,
        spool_dir=spool_dir,
        rung=rung,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-role",
        choices=UK_ROWWISE_RELEASE_ROLES,
        required=True,
        help=(
            "Which UK dataset line this run builds: 'national' (no cloning, "
            "national targets only, the calibration-seam doctrine) or 'dense' "
            "(the K-clone joint national + local surface under the local "
            "doctrine). The role supplies every unset solve default and "
            "refuses the other role's flags."
        ),
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="National Microcosm UK staging H5.",
    )
    parser.add_argument(
        "--input-sha256",
        type=sha256_argument,
        help="Pinned SHA-256 of --input-h5 (required for the joint registry path).",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        help="Full-UK OA geography ladder NPZ (required by the dense role).",
    )
    parser.add_argument(
        "--ladder-sha256",
        type=sha256_argument,
        help="Pinned SHA-256 of --ladder (required for the joint registry path).",
    )
    parser.add_argument("--ledger-facts", type=Path)
    parser.add_argument("--ledger-facts-sha256", type=sha256_argument)
    parser.add_argument("--ledger-manifest-sha256", type=sha256_argument)
    parser.add_argument("--measure-exclusions", type=Path)
    parser.add_argument("--register-json", type=Path)
    parser.add_argument(
        "--target-weight-rule",
        choices=("uniform", "grain_equal", "family_equal"),
        help=(
            "Target-weighting rule; defaults to the role's doctrine "
            "(dense: grain_equal, national: family_equal). Any other admitted "
            "rule is a receipted override."
        ),
    )
    parser.add_argument(
        "--target-loss-cap",
        type=float,
        help=(
            "National role only: receipted override of the seam doctrine's "
            "per-target loss cap."
        ),
    )
    parser.add_argument(
        "--allow-unpinned-feed",
        action="store_true",
        help=(
            "National role only: allow a Ledger artifact whose feed commit is "
            "not the committed Chronicle pin (development runs)."
        ),
    )
    parser.add_argument(
        "--incumbent-h5",
        type=Path,
        help=(
            "National role only: the incumbent dataset the finished candidate "
            "is evaluated against (microcosm#578 rule 1 on the surface both "
            "can materialize). The evaluation runs after the bundle is staged "
            "and never blocks the build; its receipt is what the release-cut "
            "certifier reads."
        ),
    )
    parser.add_argument(
        "--incumbent-sha256",
        help="National role only: the incumbent's SHA-256, verified before it is read.",
    )
    parser.add_argument(
        "--incumbent-label",
        default="enhanced_frs_2024_25",
        help="National role only: the incumbent's label in the score receipt.",
    )
    parser.add_argument("--release-candidate", action="store_true")
    parser.add_argument(
        "--households-only",
        action="store_true",
        help="Bind only Chronicle census-household constituency targets.",
    )
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the candidate H5 and evidence sidecars.",
    )
    parser.add_argument(
        "--dataset-households",
        type=int,
        help="Exact output household count after informed L0 and refit; pool clone K is unchanged. Candidate-only until size certification.",
    )
    parser.add_argument(
        "--n-clones",
        type=int,
        help="Dense role only; defaults to the doctrine clone count.",
    )
    parser.add_argument(
        "--candidate-clone-counts",
        type=_candidate_clone_counts_argument,
        help="Dry-run only comma-separated candidate clone counts.",
    )
    parser.add_argument("--seed", type=int, help="Defaults to the role's seed.")
    parser.add_argument(
        "--selection-seed",
        type=int,
        help=(
            "Seed for the size selection only (informed L0 search, exact-count "
            "draw, refit); defaults to --seed. The pool, ladder assignment and "
            "dense reference stay on --seed, so two selections compare on one "
            "pool. Requires --dataset-households."
        ),
    )
    parser.add_argument(
        "--selection-pi-hi",
        type=float,
        default=1.0,
        help=(
            "Certainty threshold of the exact-count draw: gates whose learned open "
            "probability reaches it are taken with certainty. 1.0 (default) keeps "
            "only the protected carriers certain; a lower value promotes learned "
            "near-certain gates (the US exact-k ladder runs 0.95). Candidate-only; "
            "recorded in the size receipt. Requires --dataset-households."
        ),
    )
    parser.add_argument(
        "--baseline-pi-floor",
        type=float,
        default=0.0,
        help=(
            "Floor on the inclusion probability the refit's Horvitz-Thompson "
            "baseline divides each selected row's dense weight by: a boundary "
            "row drawn at a few in a million otherwise starts at millions of "
            "households and starves every other row under the stretch bound "
            "(microcosm#355, Q50 2026-09-10). 0 (default) is the untrimmed "
            "baseline. Candidate-only; recorded in the size receipt with the "
            "rows it trimmed. Requires --dataset-households; a resumed "
            "checkpoint may use a different floor."
        ),
    )
    parser.add_argument(
        "--no-size-checkpoint",
        action="store_true",
        help=(
            "Do not persist the dense solve and the informed L0 search before the "
            "exact-count draw. By default a --dataset-households run writes "
            "size_selection_checkpoint.{npz,json} into --out so a draw refusal "
            "costs a re-draw, not the pool solve (microcosm#355)."
        ),
    )
    parser.add_argument(
        "--resume-size-checkpoint",
        type=Path,
        help=(
            "Directory holding a size_selection_checkpoint written by an earlier "
            "--dataset-households run on the same inputs: the pool and the target "
            "surface are re-derived and verified, the dense solve and the search "
            "are restored, and the run continues at the exact-count draw "
            "(--selection-pi-hi may differ; both thresholds are recorded). "
            "Requires --dataset-households and the same seeds, epochs and pins."
        ),
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Spine sampling rung: 0.01, 0.10, or 1.0.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        help=f"Dense role only; defaults to {UK_SAMPLE_SEED_DEFAULT}.",
    )
    parser.add_argument(
        "--engine-blocks",
        type=int,
        default=1,
        help="Resolve one engine or one block per clone (must equal --n-clones).",
    )
    parser.add_argument(
        "--source-year",
        type=int,
        help="Survey year recorded for lineage (calibration uses the FRS release year).",
    )
    parser.add_argument("--source-lineage-modulus", type=int)
    parser.add_argument(
        "--epochs", type=int, help="Defaults to the role's doctrine solve length."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Defaults to the role's learning rate (dense 0.15, national 0.02).",
    )
    parser.add_argument(
        "--expected-constituency-vintage",
        help="Dense role only: constituency vintage required from the ladder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the fenced clone/matrix plan without solving or writing any file."
        ),
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help=(
            "Optional current Logbook chain head. If omitted, "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    add_staging_arguments(
        parser,
        repository=UK_STAGING_REPOSITORY,
        default_upload_interval_seconds=_STAGING_UPLOAD_INTERVAL_SECONDS,
    )
    add_staged_dataset_arguments(parser, repository=UK_STAGED_DATASET_REPOSITORY)
    args = parser.parse_args(argv)
    validate_staging_arguments(parser, args)
    validate_staged_dataset_arguments(parser, args)
    _resolve_role_arguments(args)
    return args


#: Solve arguments whose argparse default is ``None`` so an explicit value can
#: be told from the role's default: the other role's refusal table keys on
#: what was actually given.
_ROLE_DEFAULTED_ARGUMENTS = (
    "n_clones",
    "seed",
    "sample_seed",
    "epochs",
    "learning_rate",
    "target_weight_rule",
    "expected_constituency_vintage",
)


def _resolve_role_arguments(args: argparse.Namespace) -> UKRowwisePosture:
    """Bind the declared role's posture and fill its defaults into unset arguments."""

    posture = uk_rowwise_posture(args.release_role)
    args._explicit_arguments = frozenset(
        name for name in _ROLE_DEFAULTED_ARGUMENTS if getattr(args, name) is not None
    )
    if args.n_clones is None:
        args.n_clones = posture.clone_count
    if args.seed is None:
        args.seed = posture.seed
    if args.sample_seed is None:
        args.sample_seed = UK_SAMPLE_SEED_DEFAULT
    if args.epochs is None:
        args.epochs = posture.epochs
    if args.learning_rate is None:
        args.learning_rate = posture.learning_rate
    if args.target_weight_rule is None:
        args.target_weight_rule = posture.target_weight_rule
    if args.expected_constituency_vintage is None:
        args.expected_constituency_vintage = posture.expected_constituency_vintage
    args._posture = posture
    return posture


def main(argv: list[str] | None = None) -> int:
    """Run the rowwise candidate build."""

    args = _parse_args(argv)
    _validate_cli_args(args)
    posture = _posture_of(args)
    if posture.role == "national":
        if args.dry_run:
            return _national_dry_run(args)
        # Argument refusals above cost nothing; the credential check reaches
        # the Hub, so it runs last, still before any input is read.
        _preflight_staged_dataset(args)
        return _run_national_role(args)
    if args.candidate_clone_counts is not None and not args.dry_run:
        raise ValueError("--candidate-clone-counts is valid only with --dry-run.")
    if _CONSERVE_MASS:
        raise NotImplementedError(
            "the candidate manifest's calibration_mass_change block reads "
            "the kernel's free-mass record; a conserve-mass doctrine run "
            "appends no record and needs its own reviewed manifest shape "
            "before this constant may flip."
        )
    if args.dry_run:
        # Dry runs plan without solving or writing and record no Logbook
        # row on any path, so they need no chain configuration.
        return _run_candidate(args, attempt=None)
    # Argument refusals above cost nothing; the credential check reaches the
    # Hub, so it runs last, still before any input is read.
    _preflight_staged_dataset(args)
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    digest = preflight_digest(posture.pipeline)
    state = AttemptState(
        build_id=_new_candidate_build_id(
            seed=args.seed,
            timestamp=started_ts,
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        ),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )
    # Logbook chain configuration is validated before any terminal work: a
    # malformed or conflicting head refuses the run with no row and no side
    # effects (#666 adversarial-review finding).
    predecessor = resolve_predecessor(args.logbook_prev_row_digest)
    # Staging telemetry opens with the attempt, as in the spine builder, so an
    # early refusal still leaves a failed run under runs/<build_id>/.
    telemetry = _create_staging_telemetry(args, build_id=state.build_id)
    try:
        return _run_candidate(
            args,
            attempt={
                "state": state,
                "started_at": started_at,
                "started_ts": started_ts,
                "code_pin": "unresolved-local-git-code-pin",
                "predecessor": predecessor,
            },
            telemetry=telemetry,
        )
    except BaseException as error:
        _fail_staging_telemetry(telemetry, error)
        raise


def _run_candidate(
    args: argparse.Namespace,
    *,
    attempt: dict[str, object] | None,
    telemetry: StagingTelemetryV2 | None = None,
) -> int:
    """Build the candidate, recording every non-dry terminal outcome.

    The recording envelope opens before input verification so that setup
    failures — unreadable inputs, frame or ladder load errors, clone and
    target-binding refusals — still spool a failed row (#666
    adversarial-review finding). Dry runs pass ``attempt=None`` and record
    nothing.
    """

    out_dir = args.out.expanduser().resolve()
    try:
        input_h5 = _require_file(args.input_h5, label="--input-h5")
        ladder_path = _require_file(args.ladder, label="--ladder")
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"--out must be a directory path, got {out_dir}.")

        input_artifact = _artifact_info(input_h5)
        ladder_artifact = _artifact_info(ladder_path)
        _verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
        _verify_requested_pin("--ladder", ladder_artifact, requested=args.ladder_sha256)
        pins = {
            "dataset": _pin_from_artifact(input_artifact),
            "ladder": _pin_from_artifact(ladder_artifact),
        }
        _stage(
            telemetry,
            "input_pinning",
            "completed",
            dataset_sha256=input_artifact["sha256"],
            ladder_sha256=ladder_artifact["sha256"],
        )
        state: AttemptState | None = None
        if attempt is not None:
            unpacked_state = attempt["state"]
            assert isinstance(unpacked_state, AttemptState)
            state = unpacked_state
            attempt["code_pin"] = git_code_pin(_REPOSITORY)
            state.input_pins_digest = role_pins_digest(pins)
            append_phase(state, "configured")
            append_phase(state, "inputs_pinned")
        national_frame, _national_provenance = load_uk_national_frame(input_h5)
        frs_release = load_uk_frs_release()
        calibration_year = int(frs_release.calibration_year)
        args._calibration_year = calibration_year
        args._frs_vintage = str(frs_release.vintage)
        if args.households_only:
            args._spine_provenance = {}
        else:
            spine_sidecar_path = input_h5.with_suffix(".build.json")
            spine_sidecar = load_bound_spine_sidecar(
                spine_sidecar_path,
                national_frame,
            )
            args._spine_provenance = spine_provenance_from_sidecar(
                spine_sidecar_path,
                spine_sidecar,
            )
        national_frame, sampling = _sample_candidate_frame(
            national_frame,
            fraction=args.sample_fraction,
            seed=args.sample_seed,
        )
        args._sampling_receipt = sampling
        if telemetry is not None and args.sample_fraction == 1.0:
            # The contract's only sampling statement is "full"; a rung below
            # f100 stages a null sample, as the spine builder does.
            telemetry.set_sample({"mode": "full"})
        source_year = _source_year(
            args.source_year,
            time_period=uk_time_period(national_frame),
        )
        if state is not None:
            # The identity digest waits on the frame-derived source year;
            # earlier failures record with the preflight placeholder.
            state.identity_digest = _candidate_identity_digest(
                pins=pins,
                args=args,
                source_year=source_year,
            )
        output_paths = _output_paths(
            out_dir,
            posture=_posture_of(args),
            vintage=args._frs_vintage,
        )
        _validate_output_paths(
            output_paths,
            input_h5=input_h5,
            ladder_path=ladder_path,
        )
        _refuse_stale_size_checkpoint(args, out_dir)
        ladder = load_uk_oa_ladder(ladder_path)
        target_provenance = ladder_target_provenance(ladder)
        _stage(telemetry, "target_compilation", "started")
        joint_inputs = _load_joint_target_inputs(args)
        facts = getattr(joint_inputs.get("artifact"), "facts", None)
        if facts is None:
            joint_inputs["census_household_uprating"] = {
                "applied": False,
                "reason": "the joint target inputs carry no Ledger facts.",
            }
        else:
            joint_inputs["census_household_uprating"] = uk_census_household_uprating(
                joint_inputs["local_registry"],
                uk_ledger_households_total(
                    facts, period=joint_inputs["calibration_year"]
                ),
                period=joint_inputs["calibration_year"],
            )
        joint_inputs["household_dispersion"] = ladder_vs_chronicle_household_dispersion(
            ladder, joint_inputs["local_registry"].specs
        )
        _stage(
            telemetry,
            "target_compilation",
            "completed",
            local_target_count=len(joint_inputs["local_registry"].specs),
            national_target_count=len(joint_inputs["national_registry"].specs),
            census_household_uprating_applied=bool(
                joint_inputs["census_household_uprating"].get("applied")
            ),
        )
        if args.release_candidate and not joint_inputs["census_household_uprating"].get(
            "applied"
        ):
            raise SystemExit(
                "error: --release-candidate requires the A15 census household "
                "uprating: "
                + str(joint_inputs["census_household_uprating"].get("reason"))
            )
        posture = _posture_of(args)
        doctrine, doctrine_override = uk_local_doctrine_with_overrides(
            posture.doctrine,
            (
                {}
                if args.target_weight_rule == posture.target_weight_rule
                else {"target_weight_rule": args.target_weight_rule}
            ),
        )
        args._doctrine_override_receipt = doctrine_override
        if doctrine.target_weight_rule != args.target_weight_rule:
            raise RuntimeError(
                "local doctrine override did not bind the requested rule."
            )

        print("cloning through the ladder route...", file=sys.stderr, flush=True)
        _stage(telemetry, "cloning", "started", clone_count=int(args.n_clones))
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
        if (
            args.dataset_households is not None
            and args.dataset_households > clone.frame.n("household")
        ):
            raise ValueError(
                "--dataset-households exceeds the cloned pool; selection never clamps the request."
            )
        _stage(
            telemetry,
            "cloning",
            "completed",
            clone_count=int(args.n_clones),
            pool_rows=int(clone.frame.n("household")),
        )
        if state is not None:
            append_phase(state, "cloned")

        if not args.households_only and args.dry_run:
            plan = _joint_dry_run_plan(
                args,
                clone=clone,
                sampled_spine=national_frame,
                ladder=ladder,
                joint_inputs=joint_inputs,
                source_year=source_year,
                input_artifact=input_artifact,
                ladder_artifact=ladder_artifact,
                target_provenance=target_provenance,
            )
            _assert_artifacts_unchanged(
                input_h5=input_h5,
                input_artifact=input_artifact,
                ladder_path=ladder_path,
                ladder_artifact=ladder_artifact,
            )
            print(_json_text(plan), end="")
            return 0

        _stage(telemetry, "surface_resolution", "started")
        if args.households_only:
            print(
                "binding Chronicle census household targets...",
                file=sys.stderr,
                flush=True,
            )
            household, problem, cross_grain = _build_bound_problem(
                assignment,
                local_registry=joint_inputs["local_registry"],
                period=joint_inputs["calibration_year"],
                census_household_uprating=joint_inputs.get("census_household_uprating"),
            )
            solve_frame = clone.frame
            restore = None
            national_rows = None
            bound_families = BOUND_TARGET_FAMILIES
            measure_resolution: Mapping[str, Any] = {}
            args._rung_surface = {
                "fraction": float(args.sample_fraction),
                "dropped_cells": 0,
                "dropped_by_grain": {},
                "dropped_by_family": {},
            }
        else:
            print("resolving joint local and national surface...", file=sys.stderr)
            (
                solve_frame,
                restore,
                national_rows,
                local_metrics,
                measure_resolution,
            ) = _resolve_candidate_engine_surface(
                clone.frame,
                joint_inputs["national_registry"],
                period=joint_inputs["calibration_year"],
                scratch_dir=out_dir.parent
                / f".{out_dir.name}.candidate-engine-scratch",
                band_edge_registry=joint_inputs["band_edge_registry"],
                blocks=args.engine_blocks,
            )
            (
                household,
                problem,
                cross_grain,
                bound_families,
                rung_surface,
            ) = _build_joint_problem(
                assignment,
                local_registry=joint_inputs["local_registry"],
                national_registry=joint_inputs["national_registry"],
                local_metrics=local_metrics,
                period=joint_inputs["calibration_year"],
                sample_fraction=args.sample_fraction,
                reviewed_unbound_higher_targets=joint_inputs[
                    "reviewed_unbound_higher_targets"
                ],
                census_household_uprating=joint_inputs.get("census_household_uprating"),
            )
            args._rung_surface = rung_surface
        args._bound_families = tuple(bound_families)
        args._joint_inputs_receipt = joint_inputs
        args._measure_resolution = dict(measure_resolution)
        _stage(
            telemetry,
            "surface_resolution",
            "completed",
            target_count=int(problem.matrix.shape[0]),
            bound_family_count=len(bound_families),
            engine_blocks=int(args.engine_blocks),
        )
        if state is not None:
            append_phase(state, "targets_bound")

        if args.dry_run:
            binding_adjudications = require_adjudicated_uk_local_binding(
                bound_families,
                problem.target_frame,
            )
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
                binding_adjudications=binding_adjudications,
                cross_grain=cross_grain,
            )
            print(_json_text(plan), end="")
            return 0

        assert attempt is not None
        assert state is not None
        started_at = attempt["started_at"]
        started_ts = attempt["started_ts"]
        assert isinstance(started_at, float)
        assert isinstance(started_ts, datetime)
        predecessor = attempt["predecessor"]
        assert predecessor is None or isinstance(predecessor, str)
        code_pin = str(attempt["code_pin"])

        print(
            f"solving {problem.matrix.shape[0]} targets x "
            f"{problem.matrix.shape[1]} households under the doctrine...",
            file=sys.stderr,
            flush=True,
        )
        checkpoint_identity = _size_checkpoint_identity(
            args, pins=pins, source_year=source_year
        )
        resume_checkpoint = (
            None
            if args.resume_size_checkpoint is None
            else args.resume_size_checkpoint.expanduser().resolve()
        )
        write_checkpoint = (
            args.dataset_households is not None
            and not args.no_size_checkpoint
            and resume_checkpoint is None
        )
        if resume_checkpoint is not None:
            print(
                f"resuming the size selection from {resume_checkpoint}...",
                file=sys.stderr,
                flush=True,
            )
        _stage(
            telemetry,
            "calibration",
            "started",
            target_count=int(problem.matrix.shape[0]),
            pool_rows=int(problem.matrix.shape[1]),
            dataset_households=args.dataset_households,
            epochs=int(args.epochs),
            epoch_every=_staging_epoch_every(args),
            resumed_from_checkpoint=resume_checkpoint is not None,
        )
        solve = solve_uk_rowwise_weights_under_doctrine(
            solve_frame,
            problem,
            bound_families=bound_families,
            national_rows=national_rows,
            target_weight_rule=args.target_weight_rule,
            restore=restore,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            conserve_mass=_CONSERVE_MASS,
            target_records=_TARGET_RECORDS,
            dataset_households=args.dataset_households,
            l0_lambda=_L0_LAMBDA,
            budget_iters=_BUDGET_ITERS,
            seed=args.seed,
            selection_seed=args.selection_seed,
            selection_pi_hi=args.selection_pi_hi,
            baseline_pi_floor=args.baseline_pi_floor,
            size_checkpoint_dir=out_dir if write_checkpoint else None,
            resume_size_checkpoint=resume_checkpoint,
            checkpoint_identity=checkpoint_identity,
            checkpoint_provenance={"code_pin": code_pin, "build_id": state.build_id},
            progress=_stderr_progress,
            progress_events=(
                None
                if telemetry is None
                else _thinned_epochs(
                    telemetry.calibration_progress, every=_staging_epoch_every(args)
                )
            ),
        )
        _validate_solve_result(solve, problem=problem)
        if solve.size_receipt is not None and solve.size_receipt.get("checkpoint"):
            checkpoint = solve.size_receipt["checkpoint"]
            if "written" in checkpoint:
                append_phase(state, "size_selection_checkpointed")
                print(
                    f"size selection checkpoint written to {out_dir}",
                    file=sys.stderr,
                    flush=True,
                )
            elif "resumed_from" in checkpoint:
                append_phase(state, "size_selection_resumed")
        append_phase(state, "solved")
        _stage(
            telemetry,
            "calibration",
            "completed",
            final_loss=float(solve.final_loss),
            n_nonzero=int(solve.n_nonzero),
            realized_households=int(solve.frame.n("household")),
            size_checkpoint=_size_checkpoint_state(solve),
        )

        # The kernel minted the calibration mass record inside calibrate() (the
        # CALIBRATED kind transition is enforced there too); the record names
        # the bound families via the doctrine's mass reason.
        calibration_record = solve.frame.mass_log[-1]
        if "calibration" not in calibration_record.reason:
            raise ValueError(
                "calibrated frame's latest mass record is not the calibration "
                f"record: {calibration_record.reason!r}."
            )
        support = _candidate_area_support(
            solve.frame.table("household"),
            ladder,
            weights=solve.weights,
        )
        _validate_support_summary(support)
        local_diagnostics = _local_gate_diagnostics(solve.diagnostics)
        target_registry, target_geography_levels = _local_diagnostics_registry(
            solve,
            problem,
            national_registry=(
                None if args.households_only else joint_inputs["national_registry"]
            ),
        )
        _stage(telemetry, "gate_battery", "started")
        try:
            gate_report, candidate_gate = _run_local_gate_battery(
                frame=solve.frame,
                support=support,
                diagnostics=local_diagnostics,
                report_path=output_paths["local_gates"],
                release_id=state.build_id,
                evaluated_on=started_ts.date(),
                enforce_only=(
                    None
                    if args.sample_fraction == 1.0
                    else ("uk_local_geography_ladder_post_calibration",)
                ),
                # The battery attests the posture it ran under: a release
                # candidate blocks on absent evidence and can be shippable.
                release_candidate=bool(args.release_candidate),
            )
        except GateBatteryBlockedError:
            # Write-then-block, extended to the whole evidence bundle: a
            # release-blocking failure at f100 still writes the diagnostics,
            # manifest and artifact (marked unreleasable) so the block can be
            # reviewed; only a non-passing ladder verdict is structural and
            # re-raises. The Logbook row records the attempt as failed.
            gate_report = json.loads(
                output_paths["local_gates"].read_text(encoding="utf-8")
            )
            _apply_gate_verdicts(state, gate_report, output_paths["local_gates"])
            ladder_entry = gate_report["gates"].get(
                "uk_local_geography_ladder_post_calibration", {}
            )
            if ladder_entry.get("status") != "passed":
                raise
            candidate_gate = clone.gate
            blocked_failures, diagnostic_failures = _gate_failures_by_criticality(
                gate_report
            )
            if not blocked_failures:
                # The battery blocked, yet the persisted report names no
                # failed release-blocking entry: the report and the error
                # disagree, which is structural.
                raise
            unenforced_failures = []
        else:
            _apply_gate_verdicts(state, gate_report, output_paths["local_gates"])
            # Nothing blocked. Release-blocking entries can still hold a
            # failure here: below f100 only the ladder gate is enforced, and
            # a dev build tolerates absent evidence. Those lines are reported
            # as not enforced, never as a block.
            unenforced_failures, diagnostic_failures = _gate_failures_by_criticality(
                gate_report
            )
            blocked_failures = []
        args._gate_report = gate_report
        args._blocked_failures = blocked_failures
        args._diagnostic_failures = diagnostic_failures
        args._unenforced_release_failures = (
            [] if blocked_failures else unenforced_failures
        )
        append_phase(
            state, "candidate_gated" if not blocked_failures else "candidate_blocked"
        )
        _stage(
            telemetry,
            "gate_battery",
            "completed",
            gate_statuses=_gate_statuses(gate_report),
            blocking_failure_count=len(blocked_failures),
            diagnostic_failure_count=len(diagnostic_failures),
        )

        _stage(telemetry, "holdout", "started", skipped=bool(args.skip_holdout))
        if args.skip_holdout:
            rotated_holdout = {"skipped": True}
        else:
            rotated_holdout = rotated_uk_local_holdout(
                solve_frame,
                problem,
                bound_families=bound_families,
                national_rows=national_rows,
                target_weight_rule=args.target_weight_rule,
                restore=restore,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                conserve_mass=_CONSERVE_MASS,
                target_records=_TARGET_RECORDS,
                dataset_households=args.dataset_households,
                l0_lambda=_L0_LAMBDA,
                budget_iters=_BUDGET_ITERS,
                solve_seed=args.seed,
                selection_seed=args.selection_seed,
                selection_pi_hi=args.selection_pi_hi,
                baseline_pi_floor=args.baseline_pi_floor,
            )
        args._rotated_holdout = rotated_holdout
        _stage(telemetry, "holdout", "completed", skipped=bool(args.skip_holdout))

        candidate = dataclasses.replace(
            clone,
            frame=solve.frame,
            gate=candidate_gate,
            output_path=None,
        )
        support_by_grain = {
            ("la" if grain == "local_authority" else str(grain)): rows.reset_index(
                drop=True
            )
            for grain, rows in support.groupby("geography_level", sort=True)
        }
        args._support_limited_misses = uk_support_limited_misses(
            solve.diagnostics,
            support_by_grain,
            max_abs_relative_error=0.25,
        )
        _assert_artifacts_unchanged(
            input_h5=input_h5,
            input_artifact=input_artifact,
            ladder_path=ladder_path,
            ladder_artifact=ladder_artifact,
        )

        _stage(telemetry, "output_bundle", "started")
        manifest = _write_output_bundle(
            args,
            candidate=candidate,
            clone=clone,
            problem=problem,
            solve=solve,
            local_diagnostics=local_diagnostics,
            target_registry=target_registry,
            target_geography_levels=target_geography_levels,
            rotated_holdout=rotated_holdout,
            support=support,
            calibration_record=calibration_record,
            source_year=source_year,
            output_paths=output_paths,
            input_artifact=input_artifact,
            ladder_artifact=ladder_artifact,
            target_provenance=target_provenance,
            cross_grain=cross_grain,
        )
        _stage(
            telemetry,
            "output_bundle",
            "completed",
            output_bytes={
                key: int(entry["bytes"]) for key, entry in manifest["outputs"].items()
            },
        )
        append_phase(state, "published")
        # The staged dataset and the telemetry receipt are evidence about the
        # published bundle, so they are appended to the manifest after it is
        # on disk (the national build record is rewritten the same way); the
        # copy inside the staged bundle predates them and staged_manifest.json
        # describes the remote side.
        staged_dataset = _stage_dataset(
            args,
            manifest=manifest,
            output_paths=output_paths,
            run_id=state.build_id if telemetry is None else telemetry.run_id,
            telemetry=telemetry,
        )
        append_phase(state, _STAGED_DATASET_PHASES[staged_dataset["status"]])
        try:
            _finalize_staging_telemetry(args, telemetry)
        finally:
            manifest["staging_delivery"] = _staging_delivery(telemetry)
            manifest["staged_dataset"] = staged_dataset
            _replace_manifest(output_paths["manifest"], manifest)
            if (output_paths["manifest"].parent / SHA256SUMS_FILENAME).is_file():
                # The uploaded copy lists the manifest as uploaded; the local
                # copy lists the manifest as it now is, evidence included.
                refresh_sha256sums_entry(
                    output_paths["manifest"].parent, output_paths["manifest"].name
                )
        state.artifact_location = local_artifact_reference(
            output_paths["dataset"],
            repository_hint=_REPOSITORY,
        )
        spool_path = _record_candidate_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            seed=args.seed,
            code_pin=code_pin,
            disposition="failed" if blocked_failures else "iterating",
            predecessor=predecessor,
            spool_dir=out_dir / "logbook-spool",
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        )
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
        print(_json_text(manifest), end="")
        if blocked_failures:
            print(
                "Gate battery blocked the artifact at f100; evidence bundle "
                f"written, artifact unreleasable: {blocked_failures[:5]}",
                file=sys.stderr,
            )
            return 1
        return 0
    except Exception as error:
        if attempt is None:
            # Dry runs record no row on any path, including failures.
            raise
        failed_state = attempt["state"]
        assert isinstance(failed_state, AttemptState)
        failed_started_at = attempt["started_at"]
        failed_started_ts = attempt["started_ts"]
        assert isinstance(failed_started_at, float)
        assert isinstance(failed_started_ts, datetime)
        failed_predecessor = attempt["predecessor"]
        assert failed_predecessor is None or isinstance(failed_predecessor, str)
        _record_candidate_error(
            error=error,
            state=failed_state,
            started_at=failed_started_at,
            started_ts=failed_started_ts,
            seed=args.seed,
            code_pin=str(attempt["code_pin"]),
            predecessor=failed_predecessor,
            base_dir=out_dir,
            spool_dir=out_dir / "logbook-spool",
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        )
        raise


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a JSON object.")
    return payload


def _ledger_facts_pin(artifact: Any) -> dict[str, object]:
    facts_path = (
        artifact.path / "consumer_facts.jsonl"
        if artifact.path.is_dir()
        else artifact.path
    )
    return {"sha256": artifact.facts_sha256, "size_bytes": facts_path.stat().st_size}


def _load_national_target_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """The national role's target surface: the pinned Ledger artifact, compiled.

    The artifact must be the committed Chronicle feed pin unless
    ``--allow-unpinned-feed`` records a reviewed diagnostic run; the
    compiled register, less the measure exclusions, is the solve surface and
    the full compiled register keeps the band edges; ``--register-json``
    requires the re-derived register to be the frozen scoring surface.
    """

    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    pin = require_committed_uk_chronicle_feed_pin(
        artifact.facts_sha256,
        manifest_sha256=artifact.manifest_sha256,
        allow_unpinned_feed=bool(args.allow_unpinned_feed),
    )
    calibration_year = int(load_uk_frs_release().calibration_year)
    compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if compilation.unsupported:
        raise SystemExit(
            f"{len(compilation.unsupported)} national target references "
            "failed to compile"
        )
    exclusions = load_uk_calibration_measure_exclusions(args.measure_exclusions)
    registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        compilation.registry, exclusions
    )
    if args.register_json is not None:
        try:
            frozen = TargetRegistry.from_json(args.register_json)
        except ValueError as error:
            raise SystemExit(
                f"error: frozen scoring register is unusable: {error}"
            ) from error
        if frozen.version != registry.version:
            raise SystemExit(
                "re-derived register differs from the frozen scoring register: "
                f"{registry.version} vs {frozen.version}"
            )
    return {
        "artifact": artifact,
        "calibration_year": calibration_year,
        "national_registry": registry,
        "band_edge_registry": compilation.registry,
        "measure_exclusions": exclusion_receipt,
        "chronicle_feed_pin": pin.to_dict(),
    }


def _national_doctrine_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The receipted per-run overrides of the seam doctrine, explicit flags only."""

    explicit = args._explicit_arguments
    overrides: dict[str, Any] = {}
    if "epochs" in explicit:
        overrides["epochs"] = int(args.epochs)
    if "learning_rate" in explicit:
        overrides["learning_rate"] = float(args.learning_rate)
    if "target_weight_rule" in explicit:
        overrides["target_weight_rule"] = str(args.target_weight_rule)
    if args.target_loss_cap is not None:
        overrides["target_loss_cap"] = float(args.target_loss_cap)
    return overrides


def _national_dry_run(args: argparse.Namespace) -> int:
    """Compile the national target surface and print the plan; write nothing."""

    posture = _posture_of(args)
    input_h5 = _require_file(args.input_h5, label="--input-h5")
    input_artifact = _artifact_info(input_h5)
    _verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
    frs_release = load_uk_frs_release()
    inputs = _load_national_target_inputs(args)
    doctrine, doctrine_overrides = uk_doctrine_with_overrides(
        **_national_doctrine_overrides(args)
    )
    plan = {
        "schema_version": 3,
        "build_kind": "uk_national_calibrated_candidate_plan",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "dry_run": True,
        "calibration_year": inputs["calibration_year"],
        "inputs": {"dataset": dict(input_artifact)},
        "targets": {
            "chronicle": inputs["artifact"].provenance(),
            "compiled": len(inputs["band_edge_registry"].specs),
            "active": len(inputs["national_registry"].specs),
            "excluded": len(inputs["measure_exclusions"]),
            "register_sha256": inputs["national_registry"].version,
        },
        "doctrine": {
            field: getattr(doctrine, field)
            for field in (
                "epochs",
                "learning_rate",
                "max_weight_ratio",
                "seed",
                "target_loss_cap",
                "scale_rule",
                "target_weight_rule",
                "mass_rule",
                "l0_lambda",
            )
        },
        "doctrine_overrides": dict(doctrine_overrides),
        "parameters": _parameters(
            args,
            source_year=_source_year(
                args.source_year, time_period=str(frs_release.time_period)
            ),
        ),
        "engine": "not_run",
        "incumbent": _incumbent_arguments(args),
        "releasable": False,
    }
    print(_json_text(plan))
    return 0


def _run_national_role(args: argparse.Namespace) -> int:
    """Build the national line: the calibration seam under the driver's posture.

    The seam library (:func:`run_uk_calibration`) resolves the measures from
    the input file, solves under the seam doctrine, runs the six
    calibration-seam gates, writes the H5, the diagnostics, the signed gate
    report, the build record and the Logbook row exactly as the retiring
    seam command did, so a national cut built here is bit-for-bit the seam's.
    The driver adds what the dense role has: the pinned input, the role's
    doctrine and overrides, staging telemetry under the attempt id, the
    rowwise manifest beside the seam's evidence, and the staged bundle.
    """

    posture = _posture_of(args)
    out_dir = args.out.expanduser().resolve()
    input_h5 = _require_file(args.input_h5, label="--input-h5")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out must be a directory path, got {out_dir}.")
    input_artifact = _artifact_info(input_h5)
    _verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
    incumbent = _incumbent_arguments(args)
    # The attempt id is minted before telemetry opens so the staging run id
    # and the Logbook row agree, as on the dense role.
    build_id = new_uk_calibration_attempt_id(timestamp=datetime.now(UTC))
    telemetry = _create_staging_telemetry(args, build_id=build_id)
    try:
        return _run_national_attempt(
            args,
            posture=posture,
            out_dir=out_dir,
            input_h5=input_h5,
            input_artifact=input_artifact,
            build_id=build_id,
            telemetry=telemetry,
            incumbent=incumbent,
        )
    except BaseException as error:
        _fail_staging_telemetry(telemetry, error)
        raise


def _run_national_attempt(
    args: argparse.Namespace,
    *,
    posture: UKRowwisePosture,
    out_dir: Path,
    input_h5: Path,
    input_artifact: Mapping[str, Any],
    build_id: str,
    telemetry: StagingTelemetryV2 | None,
    incumbent: Mapping[str, Any] | None = None,
) -> int:
    _stage(
        telemetry,
        "input_pinning",
        "completed",
        dataset_sha256=input_artifact["sha256"],
    )
    if telemetry is not None:
        telemetry.set_sample({"mode": "full"})
    frs_release = load_uk_frs_release()
    calibration_year = int(frs_release.calibration_year)
    args._calibration_year = calibration_year
    args._frs_vintage = str(frs_release.vintage)
    source_year = _source_year(
        args.source_year, time_period=str(frs_release.time_period)
    )
    output_paths = _output_paths(out_dir, posture=posture, vintage=args._frs_vintage)
    _validate_output_paths(output_paths, input_h5=input_h5, ladder_path=None)
    _stage(telemetry, "target_compilation", "started")
    inputs = _load_national_target_inputs(args)
    _stage(
        telemetry,
        "target_compilation",
        "completed",
        compiled_target_count=len(inputs["band_edge_registry"].specs),
        active_target_count=len(inputs["national_registry"].specs),
    )
    doctrine, doctrine_overrides = uk_doctrine_with_overrides(
        **_national_doctrine_overrides(args)
    )
    if doctrine.target_weight_rule != args.target_weight_rule:
        raise RuntimeError(
            "national doctrine override did not bind the requested rule."
        )
    args._doctrine_override_receipt = doctrine_overrides
    out_dir.mkdir(parents=True, exist_ok=True)
    # The frozen register is the scorer's input: the same artifact this run
    # solved against, by content hash.
    inputs["national_registry"].to_json(output_paths["national_registry"])
    # The full compiled register beside it: the band edges a pruned scoring
    # surface must never redraw (#803), for the end-of-build evaluation and
    # for a re-score by hand (--band-edge-registry-json).
    inputs["band_edge_registry"].to_json(output_paths["contract_registry"])
    resolver = UKMeasureResolver(
        simulation_source=input_h5,
        scratch_dir=out_dir,
        year=calibration_year,
        frame=None,
    )
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=output_paths["dataset"],
        diagnostics_json=output_paths["calibration_diagnostics"],
        build_record_json=output_paths["build_record"],
        terminal_gate_json=output_paths["terminal_gates"],
    )
    evidence: dict[str, Any] = {}

    def publish_manifest() -> None:
        # The seam has written its evidence; the manifest describes it and the
        # staged bundle (every manifest-registered output) follows, as on the
        # dense role. The staged copy predates the evidence blocks appended
        # below; staged_manifest.json describes the remote side.
        args._gate_report = _read_json(paths.terminal_gate_json)
        manifest = _national_manifest(
            args,
            posture=posture,
            build_id=build_id,
            build_record=_read_json(paths.build_record_json),
            build_record_path=paths.build_record_json,
            gate_report=args._gate_report,
            diagnostics=_read_json(paths.diagnostics_json),
            inputs=inputs,
            doctrine_overrides=doctrine_overrides,
            output_paths=output_paths,
            input_artifact=input_artifact,
            source_year=source_year,
        )
        _replace_manifest(output_paths["manifest"], manifest)
        evidence["manifest"] = manifest
        evidence["staged_dataset"] = _stage_dataset(
            args,
            manifest=manifest,
            output_paths=output_paths,
            run_id=build_id if telemetry is None else telemetry.run_id,
            telemetry=telemetry,
        )

    def evaluate() -> None:
        # After the bundle is staged (the evaluation never blocks staging),
        # before the telemetry completes (the receipt rides it as an artifact).
        evidence["evaluation"] = _evaluate_against_incumbent(
            args,
            incumbent=incumbent,
            inputs=inputs,
            output_paths=output_paths,
            telemetry=telemetry,
            calibration_year=calibration_year,
            out_dir=out_dir,
        )

    def finalize_staging() -> None:
        publish_manifest()
        evaluate()
        _finalize_staging_telemetry(args, telemetry)

    def event_callback(stage_id: str, status: str, details: Mapping[str, Any]) -> None:
        _stage(telemetry, stage_id, status, **dict(details))

    result = run_uk_calibration(
        paths=paths,
        build_id=build_id,
        input_sha256=str(args.input_sha256),
        ledger_artifact=inputs["artifact"],
        register_registry=inputs["national_registry"],
        band_edge_registry=inputs["band_edge_registry"],
        calibration_year=calibration_year,
        exclusion_receipt=inputs["measure_exclusions"],
        doctrine=doctrine,
        doctrine_overrides=doctrine_overrides,
        measure_resolver=resolver,
        source_pins={
            "input_h5": {
                "sha256": str(args.input_sha256),
                "size_bytes": int(input_artifact["bytes"]),
            },
            "ledger_facts": _ledger_facts_pin(inputs["artifact"]),
        },
        run_config_extra={
            "release_role": posture.role,
            "calibration_year": calibration_year,
            "allow_unpinned_feed": bool(args.allow_unpinned_feed),
            "chronicle_feed_pin": inputs["chronicle_feed_pin"],
            "rowwise_driver_parameters": _parameters(args, source_year=source_year),
        },
        release_id=posture.release_id,
        logbook_prev_row_digest=args.logbook_prev_row_digest,
        progress_callback=(
            None
            if telemetry is None
            else _thinned_epochs(
                telemetry.calibration_progress, every=_staging_epoch_every(args)
            )
        ),
        event_callback=None if telemetry is None else event_callback,
        staging_delivery=_staging_delivery(telemetry),
        staging_finalizer=None if telemetry is None else finalize_staging,
        staging_delivery_provider=(
            None if telemetry is None else (lambda: telemetry.delivery_summary)
        ),
    )
    if telemetry is None:
        publish_manifest()
        evaluate()
    manifest = evidence["manifest"]
    # The seam rewrote its record with the delivery summary after the
    # finalizer; the manifest binds the record as it now is.
    manifest["outputs"]["build_record"] = _artifact_info(paths.build_record_json)
    manifest["build_record"] = {
        "path": str(paths.build_record_json),
        "sha256": result.build_record_sha256,
    }
    manifest["staging_delivery"] = _staging_delivery(telemetry)
    manifest["staged_dataset"] = evidence["staged_dataset"]
    evaluation = evidence.get("evaluation", {"status": "not_requested"})
    manifest["evaluation"] = evaluation
    if evaluation.get("status") == "completed":
        manifest["outputs"]["score_receipt"] = evaluation["receipt"]
    _replace_manifest(output_paths["manifest"], manifest)
    if (out_dir / SHA256SUMS_FILENAME).is_file():
        # The uploaded copies list the files as uploaded; the local sums
        # list the record, the receipt and the manifest as they now are,
        # evidence included.
        refresh_sha256sums_entry(out_dir, paths.build_record_json.name)
        if evaluation.get("status") == "completed":
            _list_sha256sums_entry(out_dir, output_paths["score_receipt"].name)
        refresh_sha256sums_entry(out_dir, output_paths["manifest"].name)
    print(_json_text(manifest))
    return 0


def _national_fit_by_family(
    diagnostics: Mapping[str, Any], registry: TargetRegistry
) -> list[dict[str, object]]:
    families = {str(spec.name): str(spec.family or "") for spec in registry.specs}
    rows = []
    for row in diagnostics.get("targets", []):
        if not isinstance(row, Mapping):
            continue
        error = row.get("relative_error")
        if error is None:
            continue
        # The diagnostics name a target by its register spec name and, on
        # period-suffixed rows, by a materialized name; the family lives on
        # the spec.
        labels = [
            str(label)
            for label in (row.get("target_name"), row.get("name"))
            if label is not None
        ]
        family = next((families[label] for label in labels if label in families), "")
        rows.append(
            {
                "target_name": labels[0] if labels else "",
                "family": family,
                "abs_relative_error": abs(float(error)),
            }
        )
    if not rows:
        return []
    return uk_fit_by_family(pd.DataFrame(rows))


def _national_manifest(
    args: argparse.Namespace,
    *,
    posture: UKRowwisePosture,
    build_id: str,
    build_record: Mapping[str, Any],
    build_record_path: Path,
    gate_report: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    inputs: Mapping[str, Any],
    doctrine_overrides: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    input_artifact: Mapping[str, Any],
    source_year: int,
) -> dict[str, Any]:
    """The rowwise manifest of a national-role run, over the seam's evidence."""

    calibration = build_record["calibration"]
    solve = calibration["solve"]
    statuses = _gate_statuses(gate_report)
    abs_errors = np.asarray(
        [
            abs(float(row["relative_error"]))
            for row in diagnostics.get("targets", [])
            if isinstance(row, Mapping) and row.get("relative_error") is not None
        ],
        dtype=np.float64,
    )
    fit_rows = _national_fit_by_family(diagnostics, inputs["national_registry"])
    outputs = {
        "dataset": _artifact_info(output_paths["dataset"]),
        "calibration_diagnostics": _artifact_info(
            output_paths["calibration_diagnostics"]
        ),
        "build_record": _artifact_info(build_record_path),
        "terminal_gate_report": _artifact_info(output_paths["terminal_gates"]),
        "national_target_registry": _artifact_info(output_paths["national_registry"]),
        "national_contract_registry": _artifact_info(output_paths["contract_registry"]),
    }
    return {
        "schema_version": 4,
        "build_kind": "uk_national_calibrated_candidate",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "build_id": build_id,
        "candidate_scope": "national",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "parameters": _parameters(args, source_year=source_year),
        "inputs": {"dataset": dict(input_artifact)},
        "identity": {
            "spine": {
                **dict(input_artifact),
                "spine_provenance": dict(build_record["spine_provenance"]),
            },
            "targets": {"chronicle": inputs["artifact"].provenance()},
            "code": {"git_commit": _git_commit(), "git_dirty": _git_dirty()},
            "runtime": runtime_provenance(),
            "sampling": {"mode": "full"},
            "survey_year": source_year,
            "calibration_year": int(inputs["calibration_year"]),
        },
        "sampling": {"mode": "full"},
        "outputs": outputs,
        "weights": dict(calibration["weights"]),
        "solve": {
            "n_targets": int(solve["n_targets"]),
            "n_targets_by_kind": {
                "national": int(solve["n_targets"]),
                "local": 0,
                "ladder": 0,
            },
            "n_households": int(solve["n_households"]),
            "pool_households": int(solve["n_households"]),
            "initial_loss": float(solve["initial_loss"]),
            "final_loss": float(solve["final_loss"]),
            "n_nonzero": int(solve["n_nonzero"]),
            "max_abs_relative_error": (
                float(abs_errors.max()) if abs_errors.size else None
            ),
            "median_abs_relative_error": (
                float(np.median(abs_errors)) if abs_errors.size else None
            ),
            "effective_sample_size": calibration.get("effective_sample_size"),
            "max_weight_ratio": calibration.get("max_weight_ratio"),
            "target_weight_rule": str(args.target_weight_rule),
            "target_weight_rule_override": dict(
                doctrine_overrides.get("target_weight_rule", {})
            ),
            "doctrine_overrides": dict(doctrine_overrides),
            "measure_resolution": calibration.get("measure_resolution"),
        },
        "fit": {
            "national_by_family": fit_rows,
            "weakest_families": sorted(
                fit_rows,
                key=lambda row: (
                    -float(row["worst_abs_relative_error"]),
                    row["family"],
                ),
            )[:10],
        },
        "gate": {
            "scope": list(posture.gate_scope),
            "posture": posture.gate_posture,
            "release_id": gate_report.get("release_id"),
            "statuses": statuses,
        },
        "failing_gate_ids": sorted(
            gate_id for gate_id, status in statuses.items() if status != "passed"
        ),
        "releasable": False,
        "release_posture": {
            "release_candidate": False,
            "shippable_by": "tools/certify_uk_release_cut.py",
            "calibration_seam_gates_passed": bool(statuses)
            and all(status == "passed" for status in statuses.values()),
        },
        "measure_exclusions": dict(inputs["measure_exclusions"]),
        "build_record": {
            "path": str(build_record_path),
            "sha256": outputs["build_record"]["sha256"],
        },
    }


def _hub_api() -> Any:
    """The Hub client used for telemetry and the staged dataset (test seam)."""

    from huggingface_hub import HfApi

    return HfApi()


def _hub_token() -> str | None:
    """The ambient Hub credential, if any (test seam)."""

    from huggingface_hub import get_token

    return get_token()


def _staged_dataset_mode(args: argparse.Namespace) -> str:
    if args.no_staging or args.no_staged_dataset:
        return "disabled"
    if args.staging_local_only:
        return "local_only"
    return "local_and_remote"


def _preflight_staged_dataset(args: argparse.Namespace) -> None:
    """Refuse a remote dataset stage the run could not complete.

    The bundle upload is the last step of a multi-hour run, so the credential
    and the repository are checked before the spine is read. Telemetry stays
    best-effort with no pre-flight, as on the national command.
    """

    if args.dry_run or _staged_dataset_mode(args) != "local_and_remote":
        return
    repo_id = str(args.staged_dataset_repo_id).strip()
    hint = "pass --staging-local-only or --no-staged-dataset to keep the bundle local"
    if not _hub_token():
        raise ValueError(
            f"remote dataset staging to {repo_id} needs a Hugging Face write "
            f"credential (HF_TOKEN or `hf auth login`); {hint}."
        )
    storage = HuggingFaceDatasetStorage(repo_id, api=_hub_api())
    try:
        storage.head_revision()
    except Exception as error:
        # The transport's own message is not chained: it can carry request
        # URLs and identifiers, and the type name is enough to act on.
        raise ValueError(
            f"remote dataset staging cannot reach {repo_id} "
            f"({type(error).__name__}); {hint}."
        ) from None
    _require_write_credential(storage, hint=hint)


def _require_write_credential(storage: HuggingFaceDatasetStorage, *, hint: str) -> None:
    """Refuse a credential that can see the repository but cannot write it.

    A read token, or a fine-grained token scoped to another owner, passes the
    reachability check and is refused by the Hub with 403 only when the upload
    starts, hours later. The scope is read from the Hub's own description of
    the token; when it cannot be read the upload itself is the proof.
    """

    try:
        can_write = storage.credential_can_write()
    except Exception:
        can_write = None
    if can_write is False:
        raise ValueError(
            f"remote dataset staging to {storage.repo_id} needs a write credential: "
            "the ambient Hugging Face token is read-only or is not scoped to this "
            "repository or its owner (a fine-grained token needs repo.write on "
            f"{storage.repo_id} or on {storage.repo_id.split('/', 1)[0]}); {hint}."
        )
    if can_write is None:
        print(
            "warning: the Hugging Face credential's write scope could not be read; "
            "the upload at the end of the run will prove it.",
            file=sys.stderr,
            flush=True,
        )


def _create_staging_telemetry(
    args: argparse.Namespace, *, build_id: str
) -> StagingTelemetryV2 | None:
    if args.no_staging:
        return None
    local_only = bool(args.staging_local_only)
    out_dir = args.out.expanduser().resolve()
    posture = _posture_of(args)
    return StagingTelemetryV2(
        run_id=args.staging_run_id or build_id,
        country_code="GB",
        operation_id=posture.staging_operation_id,
        pipeline_id=posture.pipeline,
        pipeline_version=metadata.version("microcosm-build"),
        candidate_id=args.staging_candidate_id or build_id,
        local_dir=args.staging_dir or out_dir / "staging",
        run_kind="calibration",
        delivery_mode="local_only" if local_only else "local_and_remote",
        repo_id=None if local_only else args.staging_repo_id,
        upload_interval_seconds=args.staging_upload_interval_seconds,
        api=None if local_only else _hub_api(),
    )


def _stage(
    telemetry: StagingTelemetryV2 | None,
    stage_id: str,
    event_status: str = "started",
    **details: Any,
) -> None:
    """Forward one stage event to the best-effort telemetry.

    A contract or content refusal of the event is reported and the event
    dropped; the build must never abort on its own progress report. Each
    event is judged on its own details, so a refused event does not silence
    the ones that follow.
    """

    if telemetry is None:
        return
    try:
        telemetry.stage(stage_id, event_status=event_status, **details)
    except StagingContractError as error:
        print(
            f"warning: staging telemetry refused the {stage_id!r} stage event "
            f"({type(error).__name__}: {error}); the event is not staged, the "
            "build continues.",
            file=sys.stderr,
            flush=True,
        )


def _staging_epoch_every(args: argparse.Namespace) -> int:
    """The epoch stride that keeps the forwarded rows under the row budget.

    A dense run solves once; a size run solves the pool, up to ``budget_iters``
    full-length probes and the refit. The stride is at least
    ``_STAGING_EPOCH_EVERY`` and grows so at most ``_STAGING_MAX_EPOCH_ROWS``
    epochs are forwarded, keeping ``calibration_progress.json`` and
    ``events.ndjson`` under the contract's 5 MiB cap for any ``--epochs``.
    """

    solves = 1 if args.dataset_households is None else 2 + _BUDGET_ITERS
    total = int(args.epochs) * solves
    return max(_STAGING_EPOCH_EVERY, -(-total // _STAGING_MAX_EPOCH_ROWS))


def _thinned_epochs(
    sink: Callable[[Mapping[str, Any]], None], *, every: int = _STAGING_EPOCH_EVERY
) -> Callable[[dict[str, object]], None]:
    """Forward every ``every``-th epoch and each phase's last epoch to ``sink``.

    The kernel flags probe epochs with ``budget_search: True``; the staging
    contract records that field as an integer or null, so the flag becomes 1
    (the national command never runs a budget search and never met this). A
    contract or content refusal from the telemetry is reported once and stops
    the forwarding: the solve must never abort on its own progress report.
    """

    disabled = False

    def callback(event: dict[str, object]) -> None:
        nonlocal disabled
        if disabled or event.get("kind") != "calibration_epoch":
            return
        epoch = int(event["epoch"])
        epochs = int(event["epochs"])
        if epoch % every != 0 and epoch != epochs:
            return
        forwarded = dict(event)
        budget_search = forwarded.get("budget_search")
        if isinstance(budget_search, bool):
            forwarded["budget_search"] = 1 if budget_search else None
        try:
            sink(forwarded)
        except StagingContractError as error:
            disabled = True
            print(
                "warning: staging telemetry refused a calibration progress row "
                f"({type(error).__name__}); epoch progress is no longer forwarded, "
                "the solve continues.",
                file=sys.stderr,
                flush=True,
            )

    return callback


def _size_checkpoint_state(solve: UKRowwiseDoctrineSolve) -> str | None:
    if solve.size_receipt is None or not solve.size_receipt.get("checkpoint"):
        return None
    checkpoint = solve.size_receipt["checkpoint"]
    if "written" in checkpoint:
        return "written"
    if "resumed_from" in checkpoint:
        return "resumed"
    return None


def _gate_statuses(gate_report: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(gate_id): str(entry.get("status"))
        for gate_id, entry in gate_report.get("gates", {}).items()
        if isinstance(entry, Mapping)
    }


def _fail_staging_telemetry(
    telemetry: StagingTelemetryV2 | None, error: BaseException
) -> None:
    if telemetry is None or telemetry.status != "running":
        return
    try:
        telemetry.fail(error)
        telemetry.validate_local_bundle()
    except Exception:
        pass


def _finalize_staging_telemetry(
    args: argparse.Namespace, telemetry: StagingTelemetryV2 | None
) -> None:
    if telemetry is None:
        return
    try:
        telemetry.complete(message="UK rowwise candidate staging run completed.")
    except StagingContractError as error:
        _warn_telemetry("could not close the staging run", error)
        return
    try:
        if args.staging_read_back:
            # Requested explicitly, so a failed read-back is the run's failure,
            # as on the national command.
            telemetry.verify_remote()
    finally:
        try:
            telemetry.validate_local_bundle()
        except StagingContractError as error:
            _warn_telemetry("the local staging bundle does not validate", error)


def _warn_telemetry(what: str, error: BaseException) -> None:
    print(
        f"warning: {what} ({type(error).__name__}: {error}); the build's own "
        "evidence is unaffected.",
        file=sys.stderr,
        flush=True,
    )


def _staging_delivery(telemetry: StagingTelemetryV2 | None) -> dict[str, Any]:
    if telemetry is None:
        return disabled_staging_delivery("--no-staging")
    return telemetry.delivery_summary


def _stage_dataset(
    args: argparse.Namespace,
    *,
    manifest: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    run_id: str,
    telemetry: StagingTelemetryV2 | None,
) -> dict[str, Any]:
    """Stage the published bundle under ``staged/<run_id>/``; record, never raise.

    The bundle is every file the manifest registers as an output plus the
    manifest and two sidecars, verified from disk against the manifest's own
    digests. Nothing else in the run directory is eligible.
    """

    mode = _staged_dataset_mode(args)
    if mode == "disabled":
        return disabled_staged_dataset(
            "--no-staging" if args.no_staging else "--no-staged-dataset"
        )
    repository = (
        None if mode == "local_only" else str(args.staged_dataset_repo_id).strip()
    )
    _stage(telemetry, "dataset_staging", "started", mode=mode, repository=repository)
    gate_statuses = _gate_statuses(getattr(args, "_gate_report", {}) or {})
    bundle = StagedDatasetBundle.from_manifest(
        output_paths["manifest"].parent,
        run_id=run_id,
        manifest_name=MANIFEST_FILENAME,
        extra_summary={
            "dataset_households": manifest["parameters"]["dataset_households"],
            "pool_rows": manifest["solve"]["pool_households"],
            "realized_households": manifest["solve"]["n_households"],
            "final_loss": manifest["solve"]["final_loss"],
            "release_posture": manifest["release_posture"],
            "gate_statuses": gate_statuses,
        },
    )
    telemetry_reference = (
        None
        if telemetry is None
        else {
            "repository": telemetry.repo_id,
            "prefix": telemetry.repo_run_prefix,
            "mode": telemetry.delivery_mode,
        }
    )
    write_sidecars(
        bundle,
        repository=repository,
        prefix=UK_STAGED_DATASET_PREFIX,
        telemetry=telemetry_reference,
    )
    if mode == "local_only":
        delivery = local_only_staged_dataset(bundle, prefix=UK_STAGED_DATASET_PREFIX)
    else:
        print(
            f"staging the dataset bundle to {repository} under "
            f"{bundle.remote_prefix(UK_STAGED_DATASET_PREFIX)}...",
            file=sys.stderr,
            flush=True,
        )
        storage = HuggingFaceDatasetStorage(repository, api=_hub_api())
        delivery = stage_bundle(
            bundle, storage=storage, prefix=UK_STAGED_DATASET_PREFIX
        )
    print(_staged_dataset_line(delivery), file=sys.stderr, flush=True)
    _stage(
        telemetry,
        "dataset_staging",
        "completed",
        status=delivery["status"],
        repository=delivery["repository"],
        revision=delivery["revision"],
        error_code=delivery["error_code"],
        file_count=len(delivery["files"]),
    )
    _add_staging_artifact(
        telemetry,
        "staged_dataset",
        delivery,
        artifact_kind="build_metadata",
        classification="non_row_level",
    )
    _add_staging_artifact(
        telemetry,
        "fit_summary",
        _fit_summary(
            manifest,
            run_id=run_id,
            gate_statuses=gate_statuses,
            staged_dataset=delivery,
        ),
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    return delivery


def _staged_dataset_line(delivery: Mapping[str, Any]) -> str:
    status = delivery["status"]
    if status in ("uploaded", "already_staged"):
        return (
            f"staged dataset: {status} at {delivery['repository']}/"
            f"{delivery['prefix']} (revision {delivery['revision']})"
        )
    if status == "failed":
        return (
            f"staged dataset: failed ({delivery['error_code']}); the bundle and "
            "its sidecars stay local and can be re-staged with "
            "tools/stage_uk_rowwise_candidate.py"
        )
    return (
        f"staged dataset: skipped ({delivery['mode']}); sidecars written beside "
        "the bundle"
    )


def _add_staging_artifact(
    telemetry: StagingTelemetryV2 | None,
    logical_name: str,
    payload: Mapping[str, Any],
    *,
    artifact_kind: str,
    classification: str,
) -> None:
    """Attach a reviewed aggregate JSON artifact to the telemetry run.

    A content-policy refusal is reported and skipped: the telemetry is
    best-effort and must never fail a finished build.
    """

    if telemetry is None:
        return
    with tempfile.TemporaryDirectory(prefix=".staging-artifact.") as scratch:
        source = Path(scratch) / f"{logical_name}.json"
        source.write_text(_json_text(payload), encoding="utf-8")
        try:
            telemetry.add_artifact(
                logical_name,
                source,
                artifact_kind=artifact_kind,
                classification=classification,
            )
        except StagingContractError as error:
            print(
                f"warning: staging artifact {logical_name} was refused by the "
                f"content policy and is not staged: {error}",
                file=sys.stderr,
                flush=True,
            )


def _fit_summary(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    gate_statuses: Mapping[str, str],
    staged_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate fit, gate and size evidence shaped for a reviewed artifact.

    The content policy admits JSON objects without arrays of objects, so the
    per-family rows become mappings keyed by family and anything row-shaped
    is dropped by :func:`_aggregate_only`.
    """

    solve = manifest["solve"]
    fit = manifest.get("fit") or {}
    size = solve.get("dataset_size")
    parameters = manifest["parameters"]
    return {
        "schema_name": "microcosm.uk.rowwise-fit-summary",
        "schema_version": 1,
        "run_id": run_id,
        "build_kind": manifest["build_kind"],
        "releasable": manifest["releasable"],
        "release_posture": _aggregate_only(manifest["release_posture"]),
        "git_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
        "parameters": {
            key: parameters.get(key)
            for key in (
                "n_clones",
                "dataset_households",
                "seed",
                "epochs",
                "sample_fraction",
                "release_candidate",
                "skip_holdout",
                "target_weight_rule",
            )
        },
        "targets": {
            "count": solve["n_targets"],
            "by_kind": _aggregate_only(solve["n_targets_by_kind"]),
        },
        "pool_rows": solve["pool_households"],
        "realized_households": solve["n_households"],
        "loss": {
            "initial": solve["initial_loss"],
            "final": solve["final_loss"],
            "max_abs_relative_error": solve["max_abs_relative_error"],
            "median_abs_relative_error": solve["median_abs_relative_error"],
        },
        "fit_by_family": {
            "local": _rows_by_key(fit.get("local_by_family"), key="family"),
            "national": _rows_by_key(fit.get("national_by_family"), key="family"),
        },
        "weakest_areas_by_fit": _aggregate_only(fit.get("weakest_areas_by_fit")),
        "rotated_holdout": _aggregate_only(fit.get("rotated_holdout")),
        "gates": dict(gate_statuses),
        "failing_gate_ids": list(manifest.get("failing_gate_ids", [])),
        "blocking_failure_count": len(manifest.get("blocking_failures", [])),
        # The receipt's per-row arrays (pool_row_indices, inclusion
        # probabilities) live in dataset_size_selection.csv and would push the
        # artifact past the 5 MiB cap on a real run.
        "dataset_size": None
        if size is None
        else _aggregate_only(
            {
                key: value
                for key, value in size.items()
                if key not in ("pool_row_indices", "inclusion_probabilities")
            }
        ),
        "staged_dataset": {
            key: staged_dataset[key]
            for key in ("repository", "prefix", "revision", "status", "error_code")
        },
    }


def _rows_by_key(rows: Any, *, key: str) -> dict[str, Any]:
    """Turn a list of row mappings into a mapping keyed by ``row[key]``."""

    if not isinstance(rows, list):
        return {}
    keyed: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping) or key not in row:
            continue
        keyed[str(row[key])] = _aggregate_only(
            {name: value for name, value in row.items() if name != key}
        )
    return keyed


def _aggregate_only(value: Any) -> Any:
    """Drop row-shaped data (lists holding mappings) recursively."""

    if isinstance(value, Mapping):
        kept = {}
        for name, item in value.items():
            cleaned = _aggregate_only(item)
            if cleaned is not _DROPPED:
                kept[str(name)] = cleaned
        return kept
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, Mapping) for item in value):
            return _DROPPED
        return [
            item
            for item in (_aggregate_only(entry) for entry in value)
            if item is not _DROPPED
        ]
    return value


_DROPPED = object()


def _replace_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Rewrite the published manifest atomically with appended evidence."""

    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with open(handle, "w", encoding="utf-8") as stream:
            stream.write(_json_text(manifest))
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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


def _verify_requested_pin(
    label: str,
    artifact: Mapping[str, Any],
    *,
    requested: str | None,
) -> None:
    if requested is None:
        artifact["pin_verified"] = False
        return
    measured = str(artifact["sha256"])
    if measured != requested:
        raise SystemExit(
            f"error: {label} sha mismatch: measured {measured}, pinned {requested}"
        )
    artifact["pin_verified"] = True


def _load_joint_target_inputs(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    calibration_year = int(load_uk_frs_release().calibration_year)
    national_compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if national_compilation.unsupported:
        raise SystemExit(
            f"{len(national_compilation.unsupported)} national target references "
            "failed to compile"
        )
    local_compilation = compile_uk_local_target_registry(
        artifact.facts,
        target_period=calibration_year,
        crosswalk=load_uk_local_area_crosswalk(),
    )
    if local_compilation.unsupported:
        raise SystemExit(
            f"{len(local_compilation.unsupported)} local target references "
            "failed to compile"
        )
    exclusions = load_uk_calibration_measure_exclusions(args.measure_exclusions)
    national_registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        national_compilation.registry, exclusions
    )
    national_specs_by_name = {
        spec.name: spec for spec in national_compilation.registry.specs
    }
    reviewed_unbound_higher_targets = {
        str(
            national_specs_by_name[name].metadata.get(
                "contract_target_id", national_specs_by_name[name].name
            )
        ): record
        for name, record in exclusion_receipt.items()
    }
    if args.register_json is not None:
        try:
            frozen = TargetRegistry.from_json(args.register_json)
        except ValueError as error:
            raise SystemExit(
                f"error: frozen scoring register is unusable: {error}"
            ) from error
        if frozen.version != national_registry.version:
            raise SystemExit(
                "re-derived register differs from the frozen scoring register: "
                f"{national_registry.version} vs {frozen.version}"
            )
    return {
        "artifact": artifact,
        "calibration_year": calibration_year,
        "national_registry": national_registry,
        "band_edge_registry": national_compilation.registry,
        "local_registry": local_compilation.registry,
        "measure_exclusions": exclusion_receipt,
        "reviewed_unbound_higher_targets": reviewed_unbound_higher_targets,
    }


def _national_contract_target_ids(registry: TargetRegistry) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(spec.metadata.get("contract_target_id", spec.name))
                for spec in registry.specs
            }
        )
    )


def _joint_surface_registry(
    local_registry: TargetRegistry,
    national_registry: TargetRegistry,
) -> TargetRegistry:
    """Put national controls beside local cells for cross-grain reconciliation."""

    return TargetRegistry(
        [*local_registry.specs, *national_registry.specs],
        country="uk",
    )


def _build_joint_problem(
    assignment: _LadderAssignment,
    *,
    local_registry: TargetRegistry,
    national_registry: TargetRegistry,
    local_metrics: Mapping[str, pd.DataFrame],
    period: int,
    sample_fraction: float,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]],
    census_household_uprating: Mapping[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    UKRowwiseLocalMatrix,
    dict[str, Any],
    tuple[str, ...],
    dict[str, Any],
]:
    household = assignment.result.frame.table("household").reset_index(drop=True)
    household_index = pd.Index(household["household_id"], name="household_id")
    metrics = {
        grain: frame.set_axis(household_index, axis="index")
        for grain, frame in local_metrics.items()
    }
    assigned = {
        "constituency": pd.Series(
            household["constituency_code"].astype(str).to_numpy(),
            index=household_index,
        ),
        "la": pd.Series(
            household["local_authority_code"].astype(str).to_numpy(),
            index=household_index,
        ),
    }
    national_ids = _national_contract_target_ids(national_registry)
    surface, cross_grain = uk_local_target_surface(
        _joint_surface_registry(local_registry, national_registry),
        bound_national_target_ids=national_ids,
        period=period,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        census_household_uprating=census_household_uprating,
        area_region_codes=uk_area_region_codes(assignment.ladder),
    )
    covered = {
        grain: set(values.astype(str).tolist()) for grain, values in assigned.items()
    }
    covered_mask = pd.Series(
        [
            str(row.area_code) in covered[str(row.area_type)]
            for row in surface.itertuples(index=False)
        ],
        index=surface.index,
        dtype=bool,
    )
    dropped = surface.loc[~covered_mask]
    if sample_fraction < 1.0:
        surface = surface.loc[covered_mask].reset_index(drop=True)
    # Below f100 a covered area can still carry a nonzero cell with no metric
    # support in the sample (no self-employed household among three drawn
    # rows). The builder refuses such a cell at every rung; at development
    # rungs the cell is dropped here and receipted instead. f100 stays strict.
    unreachable = surface.iloc[0:0]
    if sample_fraction < 1.0 and len(surface):
        nonzero_by_grain = {
            grain: (metrics[grain] != 0).groupby(assigned[grain]).sum()
            for grain in metrics
        }
        unreachable_mask = pd.Series(
            [
                float(row.value) != 0.0
                and str(row.metric) in nonzero_by_grain[str(row.area_type)].columns
                and str(row.area_code) in nonzero_by_grain[str(row.area_type)].index
                and int(
                    nonzero_by_grain[str(row.area_type)].loc[
                        str(row.area_code), str(row.metric)
                    ]
                )
                == 0
                for row in surface.itertuples(index=False)
            ],
            index=surface.index,
            dtype=bool,
        )
        unreachable = surface.loc[unreachable_mask]
        surface = surface.loc[~unreachable_mask].reset_index(drop=True)
    rung_surface = {
        "dropped_unreachable_cells": int(len(unreachable)),
        "dropped_unreachable_by_grain": {
            str(key): int(value)
            for key, value in unreachable.groupby("area_type").size().items()
        },
        "dropped_unreachable_by_family": {
            str(key): int(value)
            for key, value in unreachable.groupby("family").size().items()
        },
        "fraction": float(sample_fraction),
        "dropped_cells": int(len(dropped) if sample_fraction < 1.0 else 0),
        "dropped_by_grain": (
            {
                str(key): int(value)
                for key, value in dropped.groupby("area_type").size().items()
            }
            if sample_fraction < 1.0
            else {}
        ),
        "dropped_by_family": (
            {
                str(key): int(value)
                for key, value in dropped.groupby("family").size().items()
            }
            if sample_fraction < 1.0
            else {}
        ),
    }
    rosters = {
        "constituency": tuple(map(str, np.unique(assignment.ladder.constituency_code))),
        "la": tuple(map(str, np.unique(assignment.ladder.local_authority_code))),
    }
    problem = build_uk_rowwise_local_surface_matrix(
        metrics,
        assigned,
        surface,
        area_codes_by_grain=rosters,
        require_every_assigned_area_covered=(sample_fraction == 1.0),
    )
    local_bound = tuple(
        sorted(
            {
                f"{row.family}/{row.area_type}"
                for row in surface[["family", "area_type"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
        )
    )
    national_bound = tuple(
        f"national/{family}"
        for family in sorted({spec.family for spec in national_registry.specs})
    )
    return (
        household,
        problem,
        cross_grain,
        (*local_bound, *national_bound),
        rung_surface,
    )


def _joint_dry_run_plan(
    args: argparse.Namespace,
    *,
    clone: UKLadderRowwiseDatasetResult,
    sampled_spine: Any,
    ladder: UkOaLadder,
    joint_inputs: Mapping[str, Any],
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    national_registry = joint_inputs["national_registry"]
    surface, cross_grain = uk_local_target_surface(
        _joint_surface_registry(
            joint_inputs["local_registry"],
            national_registry,
        ),
        bound_national_target_ids=_national_contract_target_ids(national_registry),
        period=joint_inputs["calibration_year"],
        reviewed_unbound_higher_targets=joint_inputs["reviewed_unbound_higher_targets"],
        area_region_codes=uk_area_region_codes(ladder),
        census_household_uprating=joint_inputs.get("census_household_uprating"),
    )
    household = clone.frame.table("household")
    covered = {
        "constituency": set(household["constituency_code"].astype(str)),
        "la": set(household["local_authority_code"].astype(str)),
    }
    covered_mask = pd.Series(
        [
            str(row.area_code) in covered[str(row.area_type)]
            for row in surface.itertuples(index=False)
        ],
        index=surface.index,
        dtype=bool,
    )
    dropped = surface.loc[~covered_mask]
    active_surface = (
        surface.loc[covered_mask].reset_index(drop=True)
        if args.sample_fraction < 1.0
        else surface
    )
    household_count = len(clone.frame.table("household"))
    clone_support: dict[str, object] = {}
    for clone_count in args.candidate_clone_counts or (args.n_clones,):
        candidate = (
            clone
            if clone_count == args.n_clones
            else _clone_with_ladder_binding(
                sampled_spine,
                ladder,
                n_clones=clone_count,
                seed=args.seed,
                source_year=source_year,
                expected_constituency_vintage=args.expected_constituency_vintage,
                source_lineage_modulus=args.source_lineage_modulus,
            ).result
        )
        # The typed frame weights are the authority; the persisted
        # household_weight column is an export artefact the loaded spine
        # does not carry, so attach them the way the real support path does.
        candidate_household = candidate.frame.table("household").copy()
        candidate_household["household_weight"] = np.asarray(
            candidate.frame.weights_for("household").values, dtype=np.float64
        )
        summaries = uk_ladder_area_support_summary(candidate_household, ladder)
        clone_support[str(clone_count)] = {
            grain: {
                "minimum_rows": int(rows["nonzero_households"].min()),
                "minimum_effective_sample_size": float(
                    rows["effective_sample_size"].min()
                ),
                "minimum_distinct_sources": int(
                    rows["nonzero_source_households"].min()
                ),
            }
            for grain, rows in summaries.items()
        }
    return {
        "schema_version": 3,
        "build_kind": "uk_rowwise_calibrated_candidate_plan",
        "release_role": _posture_of(args).role,
        "dry_run": True,
        "survey_year": source_year,
        "calibration_year": joint_inputs["calibration_year"],
        "identity": {
            "spine": dict(input_artifact),
            "ladder": dict(ladder_artifact),
            "targets": {
                "chronicle": joint_inputs["artifact"].provenance(),
                "paired_ladder_sha256": str(ladder_artifact["sha256"]),
            },
        },
        "sampling": dict(args._sampling_receipt),
        "rung_surface": {
            "rung": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            "fraction": args.sample_fraction,
            "dropped_cells": int(len(dropped) if args.sample_fraction < 1.0 else 0),
            "dropped_by_grain": (
                {
                    str(key): int(value)
                    for key, value in dropped.groupby("area_type").size().items()
                }
                if args.sample_fraction < 1.0
                else {}
            ),
            "dropped_by_family": (
                {
                    str(key): int(value)
                    for key, value in dropped.groupby("family").size().items()
                }
                if args.sample_fraction < 1.0
                else {}
            ),
            "unreachable_check": "deferred_to_build",
        },
        "vintages": _local_vintage_census(joint_inputs["local_registry"]),
        "cross_grain": cross_grain,
        "matrix": {
            "rows": int(len(active_surface) + len(national_registry.specs)),
            "columns": household_count,
            "local_rows": len(active_surface),
            "national_rows": len(national_registry.specs),
        },
        "candidate_clone_counts": list(args.candidate_clone_counts or (args.n_clones,)),
        "candidate_clone_support": clone_support,
        "parameters": _parameters(args, source_year=source_year),
        "releasable": False,
        "engine": "not_run",
        "ladder_assignment_provenance": dict(target_provenance),
        "household_dispersion": dict(joint_inputs["household_dispersion"]),
    }


def _local_vintage_census(registry: TargetRegistry) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for spec in registry.specs:
        resolved = str(spec.metadata.get("ledger_fact_period", ""))
        target = str(spec.period)
        if not resolved or resolved == target:
            continue
        level, _ = _spec_geography(spec)
        key = (spec.family, level, resolved, target)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "family": family,
            "geography_level": level,
            "resolved_period": resolved,
            "target_period": target,
            "cells": cells,
        }
        for (family, level, resolved, target), cells in sorted(counts.items())
    ]


def _build_bound_problem(
    assignment: _LadderAssignment,
    *,
    local_registry: TargetRegistry,
    period: int | str,
    census_household_uprating: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, UKRowwiseLocalMatrix, dict[str, Any]]:
    """Bind Chronicle census household targets at constituency grain only.

    The constituency cells go through the same ``uk_local_target_surface``
    pass as the joint scope, so the per-grain A15 factor applies here too and
    its receipt reaches the manifest; the scope has no national controls.
    """
    clone = assignment.result
    household = clone.frame.table("household").reset_index(drop=True)
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
    household_specs = sorted(
        (
            spec
            for spec in local_registry.specs
            if spec.name.startswith("ons.census.households@")
            and _spec_geography(spec)[0] == "constituency"
        ),
        key=lambda spec: _spec_geography(spec)[1],
    )
    if not household_specs:
        raise ValueError(
            "households-only binding requires Chronicle constituency household specs."
        )
    surface, cross_grain = uk_local_target_surface(
        TargetRegistry(household_specs, country="uk"),
        bound_national_target_ids=BOUND_NATIONAL_TARGETS,
        period=period,
        census_household_uprating=census_household_uprating,
        area_region_codes=uk_area_region_codes(assignment.ladder),
    )
    surface = surface.sort_values("area_code", kind="mergesort").reset_index(drop=True)
    targets = pd.DataFrame(
        {
            "code": surface["area_code"].astype(str).to_numpy(),
            "households": surface["value"].to_numpy(dtype=np.float64),
        }
    )
    problem = build_uk_rowwise_local_matrix(
        metrics,
        assigned,
        targets,
        area_type="constituency",
        code_column="code",
    )
    target_identity = surface.set_index(surface["area_code"].astype(str))[
        ["target_name", "contract_target_id", "hierarchy"]
    ]
    joined_identity = problem.target_frame[["area_code"]].join(
        target_identity,
        on="area_code",
        validate="many_to_one",
    )
    if (
        joined_identity[["target_name", "contract_target_id", "hierarchy"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "households-only target identities do not cover every matrix area code."
        )
    problem.target_frame["target_name"] = joined_identity["target_name"].to_numpy()
    problem.target_frame["contract_target_id"] = joined_identity[
        "contract_target_id"
    ].to_numpy()
    problem.target_frame["hierarchy"] = joined_identity["hierarchy"].to_numpy()
    return (
        household,
        problem,
        {"bound_national_targets": list(BOUND_NATIONAL_TARGETS), **cross_grain},
    )


def _candidate_area_support(
    household: pd.DataFrame,
    ladder: UkOaLadder,
    *,
    weights: np.ndarray,
) -> pd.DataFrame:
    weighted_household = household.copy()
    weighted_household["household_weight"] = np.asarray(weights, dtype=np.float64)
    summaries = uk_ladder_area_support_summary(weighted_household, ladder)
    return pd.concat(
        (
            summaries["constituency"].assign(geography_level="constituency"),
            summaries["la"].assign(geography_level="local_authority"),
        ),
        ignore_index=True,
    )[
        [
            "geography_level",
            "area_code",
            "assigned_households",
            "nonzero_households",
            "nonzero_source_households",
            "weight_sum",
            "max_weight",
            "effective_sample_size",
        ]
    ]


def _local_gate_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    result = diagnostics.copy()
    required = {"family", "area_type", "area_code", "metric"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"local diagnostics are missing binding columns {missing}.")
    if result[list(required)].isna().any().any():
        raise ValueError("local diagnostics contain unclassified binding rows.")
    return result


def _local_diagnostics_registry(
    solve: UKRowwiseDoctrineSolve,
    problem: UKRowwiseLocalMatrix,
    *,
    national_registry: TargetRegistry | None = None,
) -> tuple[TargetRegistry, dict[str, str]]:
    targets = tuple(solve.calibration_result.problem.targets)
    expected = len(problem.target_frame) + (
        0 if national_registry is None else len(national_registry.specs)
    )
    if len(targets) != expected:
        raise RuntimeError(
            "candidate diagnostics registry is not aligned to the solve."
        )
    specs: list[TargetSpec] = []
    geography: dict[str, str] = {}
    for target, row in zip(
        targets[: len(problem.target_frame)],
        problem.target_frame.itertuples(index=False),
        strict=True,
    ):
        metric = str(row.metric)
        family = (
            str(row.family)
            if "family" in problem.target_frame.columns
            else local_target_census.family_for_metric(metric)
        )
        spec = TargetSpec(
            name=str(target.name),
            entity=str(target.entity),
            value=float(target.value),
            measure=f"rowwise_metric:{metric}",
            filter=f"rowwise_area:{row.area_code}",
            period=target.period,
            source=str(target.source),
            family=family,
            metadata={key: str(value) for key, value in target.metadata.items()},
            hierarchy=target.hierarchy,
        )
        specs.append(spec)
        geography[spec.to_target().row_name] = str(row.area_type)
    if national_registry is not None:
        specs.extend(national_registry.specs)
        for spec in national_registry.specs:
            level, _ = _spec_geography(spec)
            geography[spec.to_target().row_name] = level
    return TargetRegistry(specs, country="uk"), geography


#: Measure columns whose policyengine-uk formulas normalise by a population
#: total (a fixed national aggregate allocated by each household's share of
#: total weighted corporate wealth, or a term scaled by a weight sum). Under
#: ``--engine-blocks K`` the engine sees one clone block at a time, so each
#: block reproduces the whole aggregate and the column comes out K× (receipt
#: R15 in ``experiments/762-uk-rowwise-candidate-receipts.md``: corporate
#: land value ×15.000 at K=15). Evidence from a per-block run must not
#: adjudicate these rows; the release posture is single-block.
UK_BLOCK_SENSITIVE_MEASURE_COLUMNS = (
    "ons/corporate_land_value",
    "ons/land_value",
    "slc/student_loan_repayment/england",
)


def _run_local_gate_battery(
    *,
    frame: Any,
    support: pd.DataFrame,
    diagnostics: pd.DataFrame,
    report_path: Path,
    release_id: str,
    evaluated_on: date,
    enforce_only: tuple[str, ...] | None = None,
    release_candidate: bool = False,
) -> tuple[dict[str, object], GateResult]:
    manifest = uk_scoped_gate_manifest(
        UK_LOCAL_GATE_SCOPE,
        phases=("terminal",),
        policy_suffix=_LOCAL_GATE_POLICY_SUFFIX,
    )
    battery = GateBatteryRun(
        manifest,
        release_id=release_id,
        report_path=report_path,
        release_candidate=release_candidate,
        registry=UK_GATE_REGISTRY,
    )
    phase = battery.run_phase(
        "terminal",
        EvidenceContext(
            frame=frame,
            artifacts={
                "uk_area_support_summary": support,
                "local_target_diagnostics": diagnostics,
                # The run's own clock, not the wall clock: the same artifact
                # must reproduce the same exclusion verdicts.
                "exclusions_evaluated_on": evaluated_on,
            },
        ),
    )
    if enforce_only is None:
        try:
            battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
        except GateBatteryBlockedError:
            payload = battery.report_payload()
            finalize_uk_scoped_gate_report(
                payload,
                posture="local_candidate",
                scope_exclusions=uk_local_gate_scope_exclusions(),
                aggregate_admin_measurement=None,
            )
            atomic_write_json(report_path, payload)
            raise
    else:
        unknown = sorted(set(enforce_only) - set(UK_LOCAL_GATE_SCOPE))
        if unknown:
            raise ValueError(f"enforce_only names unknown local gates: {unknown}.")
        selected_blocking = [
            outcome
            for outcome in phase.blocking_outcomes(release_candidate=release_candidate)
            if outcome.entry.id in enforce_only
        ]
        if selected_blocking:
            payload = battery.report_payload()
            finalize_uk_scoped_gate_report(
                payload,
                posture="local_candidate",
                scope_exclusions=uk_local_gate_scope_exclusions(),
                aggregate_admin_measurement=None,
            )
            atomic_write_json(report_path, payload)
            failures = [
                failure
                for outcome in selected_blocking
                if outcome.result is not None
                for failure in outcome.result.failures
            ]
            raise GateBatteryBlockedError("terminal", failures, report_path)
    payload = battery.report_payload()
    finalize_uk_scoped_gate_report(
        payload,
        posture="local_candidate",
        scope_exclusions=uk_local_gate_scope_exclusions(),
        aggregate_admin_measurement=None,
    )
    atomic_write_json(report_path, payload)
    ladder = next(
        outcome
        for outcome in phase.outcomes
        if outcome.entry.id == "uk_local_geography_ladder_post_calibration"
    )
    if ladder.result is None or not ladder.result.passed:
        raise RuntimeError(
            "a non-passing local geography-ladder result escaped battery enforcement."
        )
    return payload, ladder.result


def _gate_failures_by_criticality(
    gate_report: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Split a persisted battery report's failure lines by criticality.

    Returns ``(release_blocking, diagnostic)``, each entry-prefixed like
    :class:`GateBatteryBlockedError`'s lines. Only ``failed`` and
    ``evidence_absent`` entries are failures; ``not_applicable`` and
    ``unreached`` entries are not.
    """

    blocking: list[str] = []
    diagnostic: list[str] = []
    gates = gate_report.get("gates", {})
    if not isinstance(gates, Mapping):
        return blocking, diagnostic
    for gate_id, payload in gates.items():
        if not isinstance(payload, Mapping):
            continue
        status = payload.get("status")
        if status not in {"failed", "evidence_absent"}:
            continue
        lines = [f"[{gate_id}] {line}" for line in payload.get("failures") or ()]
        if not lines:
            lines = [f"[{gate_id}] {payload.get('reason') or status}"]
        bucket = blocking if _is_release_blocking(payload) else diagnostic
        bucket.extend(lines)
    return blocking, diagnostic


def _is_release_blocking(payload: Mapping[str, Any]) -> bool:
    """Fail-closed criticality read.

    Only an entry that explicitly declares ``criticality: diagnostic`` is
    exempt from vetoing the release; a missing or unknown criticality is
    treated as release-blocking, so partial schema drift on one persisted
    entry cannot drop a failed gate out of both the blocking list and
    ``all_gates_passed``.
    """

    return payload.get("criticality") != "diagnostic"


def _release_verdict(
    *,
    sample_fraction: float,
    engine_blocks: int,
    release_blocking_gates_passed: bool,
) -> tuple[bool, dict[str, bool]]:
    """``releasable`` needs the full rung, a single-block engine resolution and
    every release-blocking gate passed.

    Per-block engine resolution mis-measures population-normalised formulas
    (each block reproduces a national aggregate: the ×K land-value artefact
    behind the #736 erratum), so a run resolved in more than one block is
    diagnostic-only whatever its gates say. The posture is written beside the
    verdict so a reader sees which leg failed.
    """

    posture = {
        "full_rung": float(sample_fraction) == 1.0,
        "single_block_engine": int(engine_blocks) == 1,
        "release_blocking_gates_passed": bool(release_blocking_gates_passed),
    }
    return all(posture.values()), posture


def _apply_gate_verdicts(
    state: AttemptState,
    report: Mapping[str, object],
    report_path: Path,
) -> None:
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(UK_LOCAL_GATE_SCOPE):
        raise RuntimeError("local gate report does not cover the declared scope.")
    receipt = local_artifact_reference(report_path, repository_hint=_REPOSITORY)
    state.gate_verdicts = {
        gate_id: {
            "verdict": str(payload["status"]),
            "receipt": f"{receipt}#/gates/{gate_id}",
        }
        for gate_id, payload in gates.items()
        if isinstance(payload, Mapping)
    }
    if set(state.gate_verdicts) != set(UK_LOCAL_GATE_SCOPE):
        raise RuntimeError("local gate verdicts are malformed.")


def _dry_run_plan(
    args: argparse.Namespace,
    *,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    binding_adjudications: Mapping[str, Any],
    cross_grain: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "build_kind": "uk_rowwise_calibrated_candidate_plan",
        "release_role": _posture_of(args).role,
        "dry_run": True,
        "candidate_scope": "adjudicated_partial",
        "bound_target_families": list(args._bound_families),
        "binding_adjudications": dict(binding_adjudications),
        "cross_grain": dict(cross_grain),
        "ladder_assignment_provenance": dict(target_provenance),
        "household_dispersion": dict(
            args._joint_inputs_receipt["household_dispersion"]
        ),
        "identity": {
            "targets": {
                "chronicle": args._joint_inputs_receipt["artifact"].provenance(),
                "paired_ladder_sha256": str(ladder_artifact["sha256"]),
            }
        },
        "inputs": {
            "dataset": dict(input_artifact),
            "ladder": dict(ladder_artifact),
        },
        "sampling": dict(args._sampling_receipt),
        "survey_year": source_year,
        "calibration_year": (
            args._joint_inputs_receipt["calibration_year"]
            if args._joint_inputs_receipt is not None
            else source_year
        ),
        "rung_surface": {
            "rung": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            "fraction": args.sample_fraction,
            "unreachable_check": "completed",
        },
        "releasable": args.sample_fraction == 1.0
        and args.engine_blocks == 1
        and args.dataset_households is None,
        "parameters": _parameters(args, source_year=source_year),
        "shapes": {
            "person": list(clone.frame.table("person").shape),
            "benunit": list(clone.frame.table("benunit").shape),
            "household": list(clone.frame.table("household").shape),
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
    solve: UKRowwiseDoctrineSolve,
    local_diagnostics: pd.DataFrame,
    target_registry: TargetRegistry,
    target_geography_levels: Mapping[str, str],
    rotated_holdout: Mapping[str, object],
    support: pd.DataFrame,
    calibration_record: MassChangeRecord,
    source_year: int,
    output_paths: Mapping[str, Path],
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    cross_grain: Mapping[str, Any],
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
        if solve.dense_reference is not None:
            _dense_reference_diagnostics_frame(solve).to_csv(
                staged["dense_reference"], index=False
            )
            _dataset_size_selection_frame(solve, problem=problem, clone=clone).to_csv(
                staged["selection"], index=False
            )
        support = support.copy()
        support["support_below_floor"] = (
            (support["assigned_households"] < 50)
            | (support["effective_sample_size"] < 50.0)
            | (support["nonzero_source_households"] < 50)
        )
        support.to_csv(staged["support"], index=False)
        staged["past_cap"].write_text(_json_text(dict(solve.past_cap_census or {})))
        local_registry = _local_output_registry(
            solve,
            problem,
            period=(
                args._joint_inputs_receipt["calibration_year"]
                if args._joint_inputs_receipt is not None
                else source_year
            ),
        )
        local_registry.to_json(staged["local_registry"])
        write_uk_calibration_diagnostics(
            solve.calibration_result,
            staged["calibration_diagnostics"],
            solve.frame,
            target_geography_levels=target_geography_levels,
            target_registry=target_registry,
            local_area_support=support,
            rotated_holdout=rotated_holdout,
            build={
                "build_kind": "uk_rowwise_calibrated_candidate",
                "candidate_scope": "adjudicated_partial",
            },
        )
        calibration_diagnostics = json.loads(
            staged["calibration_diagnostics"].read_text(encoding="utf-8")
        )
        args._weakest_areas_by_fit = calibration_diagnostics["uk_diagnostics"][
            "weakest_areas_by_fit"
        ]

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
            "calibration_diagnostics": _artifact_info(
                staged["calibration_diagnostics"],
                reported_path=output_paths["calibration_diagnostics"],
            ),
            "local_gate_report": _artifact_info(output_paths["local_gates"]),
            "local_target_registry": _artifact_info(
                staged["local_registry"],
                reported_path=output_paths["local_registry"],
            ),
        }
        if solve.dense_reference is not None:
            outputs["dense_reference_diagnostics"] = _artifact_info(
                staged["dense_reference"],
                reported_path=output_paths["dense_reference"],
            )
            outputs["dataset_size_selection"] = _artifact_info(
                staged["selection"],
                reported_path=output_paths["selection"],
            )
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
            cross_grain=cross_grain,
            calibration_diagnostics=calibration_diagnostics,
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
    solve: UKRowwiseDoctrineSolve,
    support: pd.DataFrame,
    calibration_record: MassChangeRecord,
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    cross_grain: Mapping[str, Any],
    calibration_diagnostics: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    abs_errors = solve.diagnostics["abs_relative_error"].to_numpy(dtype=np.float64)
    old_total = float(calibration_record.old_total)
    new_total = float(calibration_record.new_total)
    past_cap = dict(solve.past_cap_census or {})
    gate_rows = args._gate_report.get("gates", {})
    # ``releasable`` follows the battery's own doctrine: release-blocking
    # entries decide, diagnostic entries (target_fit, weight_ratio, ...) are
    # reported but never veto. ``failing_gate_ids`` still lists every
    # non-passing entry of either criticality.
    release_gate_rows = {
        gate_id: payload
        for gate_id, payload in gate_rows.items()
        if isinstance(payload, Mapping) and _is_release_blocking(payload)
    }
    all_gates_passed = bool(release_gate_rows) and all(
        payload.get("status") == "passed" for payload in release_gate_rows.values()
    )
    releasable, release_posture = _release_verdict(
        sample_fraction=args.sample_fraction,
        engine_blocks=args.engine_blocks,
        release_blocking_gates_passed=all_gates_passed,
    )
    area_gate = gate_rows.get("uk_local_area_support", {})
    area_exclusion_details = (
        area_gate.get("details", {}) if isinstance(area_gate, Mapping) else {}
    )
    ladder_rows = int(
        problem.target_frame["target_name"]
        .astype(str)
        .str.startswith("ons.census.households@")
        .sum()
    )
    local_rows = int(len(problem.target_frame) - ladder_rows)
    sample_stage = (
        []
        if args.sample_fraction == 1.0
        else [
            {
                "stage": "sample",
                "kind": uk_household_weight_kind(clone.frame).value,
            }
        ]
    )
    posture = _posture_of(args)
    return {
        "schema_version": 4,
        "build_kind": "uk_rowwise_calibrated_candidate",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "candidate_scope": "adjudicated_partial",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "bound_target_families": list(args._bound_families),
        "binding_adjudications": dict(solve.binding_adjudications),
        "cross_grain": dict(cross_grain),
        "ladder_assignment_provenance": dict(target_provenance),
        "household_dispersion": dict(
            args._joint_inputs_receipt["household_dispersion"]
        ),
        "parameters": _parameters(args, source_year=source_year),
        "inputs": {
            "dataset": dict(input_artifact),
            "ladder": dict(ladder_artifact),
        },
        "identity": {
            "spine": {
                **dict(input_artifact),
                "spine_provenance": dict(args._spine_provenance),
            },
            "ladder": {
                **dict(ladder_artifact),
                "layer_vintages": dict(target_provenance),
                "matches_local_area_crosswalk_pin": True,
            },
            "targets": {
                "chronicle": args._joint_inputs_receipt["artifact"].provenance(),
                "paired_ladder_sha256": str(ladder_artifact["sha256"]),
            },
            "code": {"git_commit": _git_commit(), "git_dirty": _git_dirty()},
            "runtime": runtime_provenance(),
            "sampling": dict(args._sampling_receipt),
            "survey_year": source_year,
            "calibration_year": (
                args._joint_inputs_receipt["calibration_year"]
                if args._joint_inputs_receipt is not None
                else source_year
            ),
        },
        "sampling": dict(args._sampling_receipt),
        "rung_surface": {
            **dict(args._rung_surface),
            "rung": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            "fraction": args.sample_fraction,
            "unreachable_check": "completed",
        },
        "outputs": dict(outputs),
        "geography": {
            "constituencies_assigned": int(
                support.loc[
                    support["geography_level"] == "constituency",
                    "area_code",
                ].nunique()
            ),
            "local_authorities_assigned": int(
                support.loc[
                    support["geography_level"] == "local_authority",
                    "area_code",
                ].nunique()
            ),
            "missing_geography_rows": 0,
            "ladder_gate": _gate_payload(candidate.gate, phase="post_calibration"),
        },
        "gate": _gate_payload(candidate.gate, phase="post_calibration"),
        "weights": {
            "household_weight_kind": uk_household_weight_kind(candidate.frame).value,
            "household_weight_kind_chain": [
                {
                    "stage": "staging",
                    "kind": uk_household_weight_kind(clone.frame).value,
                },
                *sample_stage,
                {
                    "stage": "ladder_clone",
                    "kind": uk_household_weight_kind(clone.frame).value,
                },
                {
                    "stage": "rowwise_calibration",
                    "kind": uk_household_weight_kind(candidate.frame).value,
                },
            ],
            "mass_log_records_before_calibration": len(clone.frame.mass_log),
            "mass_log_records": len(candidate.frame.mass_log),
            "calibration_mass_change": {
                "entity": str(calibration_record.entity),
                "old_total": old_total,
                "new_total": new_total,
                "relative_shift": (new_total - old_total) / old_total,
                "declared_factor": calibration_record.declared_factor,
                "reason": str(calibration_record.reason),
            },
            "abs_delta": abs(new_total - old_total),
            "declared_stretch_bound": float(posture.doctrine.max_weight_ratio),
            "stretch_reference": "pool_design"
            if solve.size_receipt is None
            else "normalized_horvitz_thompson_w_over_q",
            # Against the frame the refit started from (the pool design on a
            # dense run, the Horvitz-Thompson baseline on a size run)...
            "realized_max_weight_ratio_vs_stretch_reference": float(
                np.max(
                    np.divide(
                        np.asarray(solve.weights, dtype=np.float64),
                        np.asarray(solve.initial_weights),
                    )
                )
            ),
            # ...and always against the pool design weights themselves.
            "realized_max_weight_ratio_vs_design": float(
                np.max(
                    np.divide(
                        np.asarray(solve.weights, dtype=np.float64),
                        np.asarray(_design_weights_for(solve), dtype=np.float64),
                    )
                )
            ),
        },
        "solve": {
            "n_targets": int(len(problem.targets) + len(solve.national_diagnostics)),
            "n_targets_by_kind": {
                "local": local_rows,
                "ladder": ladder_rows,
                "national": int(len(solve.national_diagnostics)),
            },
            "n_households": int(solve.frame.n("household")),
            "pool_households": int(problem.n_households),
            "dataset_size": None
            if solve.size_receipt is None
            else {
                **dict(solve.size_receipt),
                "dense_reference": _dense_reference_summary(solve),
            },
            "initial_loss": float(solve.initial_loss),
            "final_loss": float(solve.final_loss),
            "max_abs_relative_error": float(abs_errors.max()),
            "median_abs_relative_error": float(np.median(abs_errors)),
            "n_nonzero": int(solve.n_nonzero),
            "past_cap": {key: int(past_cap[key]) for key in _PAST_CAP_COUNT_KEYS},
            "loss_shape": "capped_relative_error",
            "target_weight_rule": args.target_weight_rule,
            "target_weight_rule_override": dict(args._doctrine_override_receipt),
            "measure_resolution": dict(args._measure_resolution),
            "cross_grain": dict(cross_grain),
            "binding_adjudications": dict(solve.binding_adjudications),
            "area_support_exclusions": {
                "resource": "local_area_support_exclusions.json",
                "entries_stood_on": sorted(
                    area_exclusion_details.get("reviewed_exclusions", {})
                ),
                "stale": list(area_exclusion_details.get("stale_exclusions", [])),
                "unknown": list(area_exclusion_details.get("unknown_exclusions", [])),
            },
            "past_cap_by_kind": {
                "local": dict(solve.past_cap_census or {}),
                "national": dict(solve.national_past_cap_census or {}),
                "all": dict(solve.all_past_cap_census or {}),
            },
        },
        "diagnostics": {
            "schema_version": calibration_diagnostics["schema_version"],
            "target_registry": calibration_diagnostics["target_registry"],
            "weakest_families": calibration_diagnostics["uk_diagnostics"][
                "weakest_families"
            ],
            "weakest_areas_by_fit": calibration_diagnostics["uk_diagnostics"][
                "weakest_areas_by_fit"
            ],
            "rotated_holdout": calibration_diagnostics["uk_diagnostics"][
                "rotated_holdout"
            ],
        },
        "support": {
            "min_assigned_households": int(support["assigned_households"].min()),
            "min_nonzero_households": int(support["nonzero_households"].min()),
            "min_effective_sample_size": float(support["effective_sample_size"].min()),
            "by_geography_level": {
                str(level): {
                    "min_assigned_households": int(rows["assigned_households"].min()),
                    "min_nonzero_households": int(rows["nonzero_households"].min()),
                    "min_effective_sample_size": float(
                        rows["effective_sample_size"].min()
                    ),
                    "min_nonzero_source_households": int(
                        rows["nonzero_source_households"].min()
                    ),
                }
                for level, rows in support.groupby("geography_level", sort=True)
            },
        },
        "fit": {
            "local_by_family": uk_fit_by_family(solve.diagnostics),
            "national_by_family": uk_fit_by_family(solve.national_diagnostics),
            "weakest_families": sorted(
                [
                    *uk_fit_by_family(solve.diagnostics),
                    *uk_fit_by_family(solve.national_diagnostics),
                ],
                key=lambda row: (
                    -float(row["worst_abs_relative_error"]),
                    row["family"],
                ),
            )[:10],
            "weakest_areas_by_fit": dict(args._weakest_areas_by_fit),
            "support_limited_misses": dict(args._support_limited_misses),
            "rotated_holdout": dict(args._rotated_holdout),
        },
        "vintages": (
            _local_vintage_census(args._joint_inputs_receipt["local_registry"])
            if args._joint_inputs_receipt is not None
            else []
        ),
        "failing_gate_ids": sorted(
            gate_id
            for gate_id, payload in gate_rows.items()
            if not isinstance(payload, Mapping) or payload.get("status") != "passed"
        ),
        "releasable": releasable and args.dataset_households is None,
        "release_posture": {
            **release_posture,
            **(
                {}
                if args.dataset_households is None
                else {"size_certification_present": False}
            ),
        },
        "census_household_uprating": dict(
            cross_grain.get("census_household_uprating")
            or {"applied": False, "reason": "no cross-grain receipt"}
        ),
        # The reviewed measure exclusions the national compile stood on
        # (name -> register record), so the narrowing is in the evidence.
        "measure_exclusions": {
            str(name): dict(record)
            for name, record in sorted(
                (
                    (getattr(args, "_joint_inputs_receipt", None) or {}).get(
                        "measure_exclusions"
                    )
                    or {}
                ).items()
            )
        },
        "blocked_at_f100": bool(getattr(args, "_blocked_failures", [])),
        "blocking_failures": list(getattr(args, "_blocked_failures", [])),
        "diagnostic_failures": list(getattr(args, "_diagnostic_failures", [])),
        "release_gate_failures_not_enforced": list(
            getattr(args, "_unenforced_release_failures", [])
        ),
    }


def _parameters(args: argparse.Namespace, *, source_year: int) -> dict[str, Any]:
    posture = _posture_of(args)
    return {
        "release_role": posture.role,
        "n_clones": None if args.n_clones is None else int(args.n_clones),
        "dataset_households": args.dataset_households,
        "seed": int(args.seed),
        "selection_seed": None
        if args.dataset_households is None
        else int(args.seed if args.selection_seed is None else args.selection_seed),
        "selection_pi_hi": None
        if args.dataset_households is None
        else float(args.selection_pi_hi),
        "baseline_pi_floor": None
        if args.dataset_households is None
        else float(args.baseline_pi_floor),
        "size_checkpoint": bool(
            args.dataset_households is not None
            and not args.no_size_checkpoint
            and args.resume_size_checkpoint is None
        ),
        "resume_size_checkpoint": None
        if args.resume_size_checkpoint is None
        else str(args.resume_size_checkpoint.expanduser().resolve()),
        "source_year": source_year,
        "source_lineage_modulus": args.source_lineage_modulus,
        "sample_fraction": float(args.sample_fraction),
        "sample_seed": int(args.sample_seed),
        "engine_blocks": int(args.engine_blocks),
        "target_weight_rule": args.target_weight_rule,
        "release_candidate": bool(args.release_candidate),
        "skip_holdout": bool(args.skip_holdout),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "expected_constituency_vintage": (
            None
            if args.expected_constituency_vintage is None
            else str(args.expected_constituency_vintage)
        ),
        "doctrine": _doctrine_bounds(posture),
        "solve_options": {
            "conserve_mass": _CONSERVE_MASS,
            "target_records": _TARGET_RECORDS,
            "l0_lambda": _L0_LAMBDA,
            "budget_iters": _BUDGET_ITERS,
        },
    }


def _design_weights_for(solve: UKRowwiseDoctrineSolve) -> np.ndarray:
    """The pool design weights aligned to the solve's exported rows."""
    if solve.selected_support is None or solve.dense_reference is None:
        return np.asarray(solve.initial_weights, dtype=np.float64)
    return np.asarray(solve.dense_reference.initial_weights, dtype=np.float64)[
        np.asarray(solve.selected_support, dtype=np.int64)
    ]


def _dense_reference_summary(solve: UKRowwiseDoctrineSolve) -> dict[str, Any] | None:
    """Manifest-sized evidence of the dense solve a size run was cut from."""

    dense = solve.dense_reference
    if dense is None:
        return None
    local_errors = dense.diagnostics["abs_relative_error"].to_numpy(dtype=np.float64)
    national_errors = dense.national_diagnostics["abs_relative_error"].to_numpy(
        dtype=np.float64
    )
    past_cap = dict(dense.past_cap_census or {})
    return {
        "initial_loss": float(dense.initial_loss),
        "final_loss": float(dense.final_loss),
        "n_nonzero": int(dense.n_nonzero),
        "n_households": int(dense.weights.size),
        "max_abs_relative_error": float(local_errors.max())
        if local_errors.size
        else None,
        "median_abs_relative_error": float(np.median(local_errors))
        if local_errors.size
        else None,
        "national_max_abs_relative_error": float(national_errors.max())
        if national_errors.size
        else None,
        "past_cap": {key: int(past_cap[key]) for key in _PAST_CAP_COUNT_KEYS},
        "weights": uk_weight_summary(dense.weights),
        "local_by_family": uk_fit_by_family(dense.diagnostics),
        "national_by_family": uk_fit_by_family(dense.national_diagnostics),
        "diagnostics_file": DENSE_REFERENCE_DIAGNOSTICS_FILENAME,
    }


def _dense_reference_diagnostics_frame(solve: UKRowwiseDoctrineSolve) -> pd.DataFrame:
    """Every target's dense-reference estimate, local rows then national rows."""

    dense = solve.dense_reference
    assert dense is not None
    local = dense.diagnostics.copy()
    local.insert(0, "grain", local["area_type"].astype(str))
    national = dense.national_diagnostics.copy()
    national.insert(0, "grain", "national")
    return pd.concat([local, national], ignore_index=True, sort=False)


def _dataset_size_selection_frame(
    solve: UKRowwiseDoctrineSolve,
    *,
    problem: UKRowwiseLocalMatrix,
    clone: UKLadderRowwiseDatasetResult,
) -> pd.DataFrame:
    """One row per selected pool household: identity, design, draw and refit."""

    dense = solve.dense_reference
    receipt = solve.size_receipt
    assert dense is not None and receipt is not None
    support = np.asarray(solve.selected_support, dtype=np.int64)
    household = clone.frame.table("household")
    clone_column = ladder_clone_index_column("household")
    ids = household["household_id"].to_numpy()[support]
    expected = np.asarray([problem.household_ids[i] for i in support])
    if not np.array_equal(ids, expected):
        raise RuntimeError(
            "the cloned pool's household order does not match the solve's "
            "matrix columns; the selection sidecar would misattribute rows."
        )
    inclusion = np.asarray(receipt["inclusion_probabilities"], dtype=np.float64)
    if inclusion.shape != support.shape:
        raise RuntimeError("selection receipt inclusion probabilities are misaligned.")
    return pd.DataFrame(
        {
            "pool_row_index": support,
            "household_id": ids,
            "clone_index": household[clone_column].to_numpy()[support]
            if clone_column in household.columns
            else np.zeros(support.size, dtype=np.int64),
            "design_weight": dense.initial_weights[support],
            "inclusion_probability": inclusion,
            "certainty": inclusion >= 1.0,
            "ht_baseline_weight": np.asarray(solve.initial_weights, dtype=np.float64),
            "refit_weight": np.asarray(solve.weights, dtype=np.float64),
        }
    )


def _local_output_registry(
    solve: UKRowwiseDoctrineSolve,
    problem: UKRowwiseLocalMatrix,
    *,
    period: int,
) -> TargetRegistry:
    specs = []
    targets = tuple(solve.calibration_result.problem.targets)
    if len(targets) < len(problem.target_frame):
        raise RuntimeError("candidate target set is shorter than its local surface.")
    for target, row in zip(
        targets[: len(problem.target_frame)],
        problem.target_frame.itertuples(index=False),
        strict=True,
    ):
        payload = row._asdict()
        specs.append(
            TargetSpec(
                name=str(payload["target_name"]),
                entity="household",
                value=float(payload["value"]),
                measure=str(payload["metric"]),
                period=int(payload.get("period", period)),
                source=str(payload.get("source", "uk_rowwise_local_surface")),
                family=str(payload["family"]),
                metadata={
                    "area_type": str(payload["area_type"]),
                    "area_code": str(payload["area_code"]),
                    "metric": str(payload["metric"]),
                },
                hierarchy=target.hierarchy,
            )
        )
    return TargetRegistry(specs, country="uk")


def _doctrine_bounds(posture: UKRowwisePosture) -> dict[str, Any]:
    return posture.doctrine_bounds()


def _gate_payload(gate: GateResult, *, phase: str) -> dict[str, Any]:
    return {
        "name": str(gate.name),
        "passed": bool(gate.passed),
        "failures": list(gate.failures),
        "details": dict(gate.details),
        "phase": phase,
    }


def _validate_solve_result(
    solve: UKRowwiseDoctrineSolve,
    *,
    problem: UKRowwiseLocalMatrix,
) -> None:
    if solve.past_cap_census is None:
        raise RuntimeError(
            "doctrine solve returned no past-cap census; refusing candidate."
        )
    expected_count = (
        problem.n_households
        if solve.selected_support is None
        else len(solve.selected_support)
    )
    if len(solve.weights) != expected_count:
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
        "geography_level",
        "area_code",
        "assigned_households",
        "nonzero_households",
        "nonzero_source_households",
        "effective_sample_size",
    }
    missing = sorted(required - set(support.columns))
    if missing or support.empty:
        raise RuntimeError(
            f"area support summary is empty or missing required columns: {missing}."
        )
    numeric = sorted(required - {"geography_level", "area_code"})
    values = support[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("area support summary contains invalid values.")


def _validate_cli_args(args: argparse.Namespace) -> None:
    posture = _posture_of(args)
    # The declared role is checked against the parameters first: the other
    # role's flags are refused by name before any value is range-checked.
    if posture.role == "national":
        _refuse_dense_role_arguments(args, posture)
    else:
        _refuse_national_role_arguments(args, posture)
    if args.selection_seed is not None and args.dataset_households is None:
        raise ValueError("--selection-seed requires --dataset-households.")
    if not (0.0 < args.selection_pi_hi <= 1.0):
        raise ValueError("--selection-pi-hi must be in (0, 1].")
    if args.selection_pi_hi != 1.0 and args.dataset_households is None:
        raise ValueError("--selection-pi-hi requires --dataset-households.")
    if not (0.0 <= args.baseline_pi_floor <= 1.0):
        raise ValueError("--baseline-pi-floor must be in [0, 1].")
    if args.baseline_pi_floor != 0.0 and args.dataset_households is None:
        raise ValueError("--baseline-pi-floor requires --dataset-households.")
    if args.no_size_checkpoint and args.dataset_households is None:
        raise ValueError("--no-size-checkpoint requires --dataset-households.")
    if args.resume_size_checkpoint is not None:
        if args.dataset_households is None:
            raise ValueError("--resume-size-checkpoint requires --dataset-households.")
        if args.no_size_checkpoint:
            raise ValueError(
                "--resume-size-checkpoint already implies no new checkpoint; "
                "drop --no-size-checkpoint."
            )
    if args.dataset_households is not None:
        if args.dataset_households <= 0:
            raise ValueError("--dataset-households must be positive.")
        if args.release_candidate:
            raise ValueError(
                "--dataset-households is candidate-only: size-specific matched comparison and promotion scorecard are required before release."
            )
    # Size-selection arguments are validated first so their refusals name
    # the size flag at fault; the pinned Ledger inputs are then mandatory.
    ledger_values = (
        args.ledger_facts,
        args.ledger_facts_sha256,
        args.ledger_manifest_sha256,
    )
    if not all(value is not None for value in ledger_values):
        raise ValueError(
            "--ledger-facts, --ledger-facts-sha256, and "
            "--ledger-manifest-sha256 are mandatory and must be supplied together."
        )
    if posture.ladder_required:
        if args.ladder is None:
            raise ValueError("--release-role dense requires --ladder.")
        if args.input_sha256 is None or args.ladder_sha256 is None:
            raise ValueError(
                "the joint registry path requires --input-sha256 and --ladder-sha256."
            )
    elif args.input_sha256 is None:
        raise ValueError("--release-role national requires --input-sha256.")
    if args.release_candidate:
        required_release = {
            "--input-sha256": args.input_sha256,
            "--ladder-sha256": args.ladder_sha256,
            "--ledger-facts": args.ledger_facts,
            "--ledger-facts-sha256": args.ledger_facts_sha256,
            "--ledger-manifest-sha256": args.ledger_manifest_sha256,
        }
        missing_release = [
            name for name, value in required_release.items() if value is None
        ]
        if missing_release:
            raise ValueError(
                "--release-candidate requires pinned joint inputs: "
                + ", ".join(missing_release)
            )
        refused = []
        if args.target_weight_rule != posture.target_weight_rule:
            refused.append("--target-weight-rule")
        if args.epochs != posture.epochs:
            refused.append(f"--epochs != doctrine {posture.epochs}")
        if args.n_clones != posture.clone_count:
            refused.append(f"--n-clones != doctrine {posture.clone_count}")
        if args.measure_exclusions is not None:
            refused.append("--measure-exclusions")
        if args.skip_holdout:
            refused.append("--skip-holdout")
        if args.engine_blocks > 1:
            refused.append("--engine-blocks > 1")
        if args.sample_fraction != 1.0:
            refused.append("--sample-fraction != 1.0")
        if refused:
            raise ValueError(
                "--release-candidate refuses non-release settings: "
                + ", ".join(refused)
            )
    if args.n_clones is not None and args.n_clones <= 0:
        raise ValueError("--n-clones must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.sample_fraction not in UK_SAMPLE_RUNG_TOKENS:
        raise ValueError(
            "--sample-fraction must be one of "
            f"{sorted(UK_SAMPLE_RUNG_TOKENS)}, got {args.sample_fraction!r}."
        )
    if args.sample_seed < 0:
        raise ValueError("--sample-seed must be non-negative.")
    if args.engine_blocks <= 0:
        raise ValueError("--engine-blocks must be positive.")
    if args.engine_blocks > 1 and args.engine_blocks != args.n_clones:
        raise ValueError("--engine-blocks greater than one must equal --n-clones.")
    if args.source_year is not None and args.source_year <= 0:
        raise ValueError("--source-year must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite.")
    if args.target_loss_cap is not None and (
        not np.isfinite(args.target_loss_cap) or args.target_loss_cap <= 0
    ):
        raise ValueError("--target-loss-cap must be positive and finite.")
    if (
        args.expected_constituency_vintage is not None
        and not str(args.expected_constituency_vintage).strip()
    ):
        raise ValueError("--expected-constituency-vintage must be non-empty.")


def _refuse_dense_role_arguments(
    args: argparse.Namespace, posture: UKRowwisePosture
) -> None:
    """The national role's refusal table: nothing of the clone surface may be given.

    ``--release-candidate`` is refused outright with the seam's own reason: the
    calibration-seam battery covers six of the declared entries and must never
    sign a shippability claim; a national cut's verdict comes only from the
    release-cut certification producer (``tools/certify_uk_release_cut.py``).
    """

    if args.release_candidate:
        raise ValueError(
            "--release-candidate is refused on the national role: the "
            "calibration seam's scoped battery cannot sign shippability; run "
            "the release-cut certification producer "
            "(tools/certify_uk_release_cut.py) on the finished build instead."
        )
    explicit = args._explicit_arguments
    refused: list[str] = []
    if args.ladder is not None:
        refused.append("--ladder")
    if args.ladder_sha256 is not None:
        refused.append("--ladder-sha256")
    if "expected_constituency_vintage" in explicit:
        refused.append("--expected-constituency-vintage")
    if args.source_year is not None:
        refused.append("--source-year")
    if args.source_lineage_modulus is not None:
        refused.append("--source-lineage-modulus")
    if "n_clones" in explicit:
        refused.append("--n-clones")
    if args.candidate_clone_counts is not None:
        refused.append("--candidate-clone-counts")
    if args.engine_blocks != 1:
        refused.append("--engine-blocks")
    if args.households_only:
        refused.append("--households-only")
    if args.skip_holdout:
        refused.append("--skip-holdout")
    if args.dataset_households is not None:
        refused.append("--dataset-households")
    if args.selection_seed is not None:
        refused.append("--selection-seed")
    if args.selection_pi_hi != 1.0:
        refused.append("--selection-pi-hi")
    if args.baseline_pi_floor != 0.0:
        refused.append("--baseline-pi-floor")
    if args.no_size_checkpoint:
        refused.append("--no-size-checkpoint")
    if args.resume_size_checkpoint is not None:
        refused.append("--resume-size-checkpoint")
    if args.sample_fraction != 1.0:
        refused.append("--sample-fraction")
    if "sample_seed" in explicit:
        refused.append("--sample-seed")
    if "seed" in explicit and args.seed != posture.seed:
        # The seam doctrine's seed is a reviewed constant, not a knob.
        refused.append(f"--seed != doctrine {posture.seed}")
    if args.target_weight_rule not in posture.allowed_target_weight_rules:
        refused.append(f"--target-weight-rule {args.target_weight_rule}")
    if refused:
        raise ValueError(
            "--release-role national refuses the dense role's arguments: "
            + ", ".join(refused)
        )


def _refuse_national_role_arguments(
    args: argparse.Namespace, posture: UKRowwisePosture
) -> None:
    """The dense role's refusal table: the seam's knobs are not its own."""

    refused: list[str] = []
    if args.target_loss_cap is not None:
        refused.append("--target-loss-cap")
    if args.allow_unpinned_feed:
        refused.append("--allow-unpinned-feed")
    if args.incumbent_h5 is not None or args.incumbent_sha256 is not None:
        refused.append("--incumbent-h5/--incumbent-sha256")
    if args.target_weight_rule not in posture.allowed_target_weight_rules:
        refused.append(f"--target-weight-rule {args.target_weight_rule}")
    if refused:
        raise ValueError(
            "--release-role dense refuses the national role's arguments: "
            + ", ".join(refused)
        )


def _load_candidate_evaluator(importer=importlib.import_module):
    """The common-surface scorer (microcosm#967), loaded when an incumbent is given.

    Lazy so the driver imports without it; a build asked to evaluate refuses
    up front, before any solve, when the scorer is not in the tree.
    """

    try:
        return importer("microcosm.build.uk_runtime.candidate_score")
    except ImportError as error:
        raise ValueError(
            "--incumbent-h5 needs the common-surface scorer (microcosm#967): "
            "microcosm.build.uk_runtime.candidate_score is not in this tree."
        ) from error


def _incumbent_arguments(args: argparse.Namespace) -> dict[str, Any] | None:
    """The national role's optional incumbent, pinned and verified up front."""

    if args.incumbent_h5 is None and args.incumbent_sha256 is None:
        return None
    if args.incumbent_h5 is None or args.incumbent_sha256 is None:
        raise ValueError(
            "--incumbent-h5 and --incumbent-sha256 must be given together."
        )
    _load_candidate_evaluator()
    incumbent_h5 = _require_file(args.incumbent_h5, label="--incumbent-h5")
    info = _artifact_info(incumbent_h5)
    _verify_requested_pin("--incumbent-h5", info, requested=args.incumbent_sha256)
    return {
        "path": str(incumbent_h5),
        "sha256": info["sha256"],
        "bytes": info["bytes"],
        "label": str(args.incumbent_label),
    }


def _list_sha256sums_entry(out_dir: Path, name: str) -> None:
    """List a file the build wrote after the sidecars in the local sums."""

    sums_path = out_dir / SHA256SUMS_FILENAME
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    if name in {listed for _, listed in entries}:
        refresh_sha256sums_entry(out_dir, name)
        return
    digest = _artifact_info(out_dir / name)["sha256"]
    sums_path.write_text(
        sums_path.read_text(encoding="utf-8") + f"{digest}  {name}\n",
        encoding="utf-8",
    )


#: Receipt blocks that are record arrays (lists of mappings): the reviewed
#: telemetry artifact policy refuses them, and the verdict, the pruned block
#: and the aggregates carry everything a reviewer reads. The full receipt stays
#: beside the outputs and in the staged bundle.
_SCORE_RECEIPT_TELEMETRY_EXCLUDED = (
    "target_drift",
    "signed_asymmetries",
    "measure_resolution",
)


def _score_receipt_telemetry_summary(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in score.items()
        if key not in _SCORE_RECEIPT_TELEMETRY_EXCLUDED
    }


def _evaluate_against_incumbent(
    args: argparse.Namespace,
    *,
    incumbent: Mapping[str, Any] | None,
    inputs: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    telemetry: StagingTelemetryV2 | None,
    calibration_year: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Score the finished candidate against the incumbent; never fail the build.

    Runs after the staged bundle is on the Hub and before the telemetry
    completes, so the receipt rides the run as a reviewed artifact. Rows the
    incumbent cannot materialize are pruned from both arms and warned about
    by name; the verdict (microcosm#578 rule 1 on the common surface) is
    what the release-cut certifier requires to be ``passed``. An error is
    recorded and warned, never raised: staging and the exit code stand.
    """

    if incumbent is None:
        return {
            "status": "not_requested",
            "note": (
                "no --incumbent-h5: the rule-1 score receipt the release-cut "
                "certifier needs was not produced; score the candidate with "
                "tools/score_uk_national_candidate.py before certification."
            ),
        }
    _stage(
        telemetry,
        "incumbent_evaluation",
        "started",
        incumbent_sha256=incumbent["sha256"],
    )
    candidate = _artifact_info(output_paths["dataset"])
    try:
        module = _load_candidate_evaluator()
        score = module.evaluate_uk_candidate_against_incumbent(
            candidate_h5=output_paths["dataset"],
            incumbent_h5=Path(incumbent["path"]),
            candidate_sha256=candidate["sha256"],
            incumbent_sha256=incumbent["sha256"],
            target_registry=inputs["national_registry"],
            calibration_year=calibration_year,
            measure_resolver_factory=module.uk_default_measure_resolver_factory(
                out_dir, calibration_year
            ),
            candidate_label=output_paths["dataset"].stem,
            incumbent_label=incumbent["label"],
            band_edge_registry=inputs["band_edge_registry"],
        )
        atomic_write_json(output_paths["score_receipt"], score)
    except Exception as error:  # noqa: BLE001 - the evaluation never fails a finished build
        message = f"{type(error).__name__}: {error}"[:600]
        print(
            f"warning: the incumbent evaluation failed ({message}); the build's "
            "evidence and staging are unaffected, and the candidate cannot be "
            "certified until it is re-scored with "
            "tools/score_uk_national_candidate.py.",
            file=sys.stderr,
            flush=True,
        )
        _stage(telemetry, "incumbent_evaluation", "failed", error=message)
        return {
            "status": "error",
            "error": message,
            "incumbent": dict(incumbent),
            "receipt": None,
        }
    warning = module.pruned_warning(score)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)
    evaluation = score["evaluation"]
    pruned = score["incumbent_unresolvable_pruned"]
    _add_staging_artifact(
        telemetry,
        "score_vs_incumbent",
        _score_receipt_telemetry_summary(score),
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    _stage(
        telemetry,
        "incumbent_evaluation",
        "completed",
        verdict=evaluation["verdict"],
        n_scored=int(evaluation["scored_surface"]["n_scored"]),
        n_pruned=int(evaluation["scored_surface"]["n_pruned"]),
    )
    return {
        "status": "completed",
        "verdict": evaluation["verdict"],
        "rule_1": dict(evaluation["rule_1"]),
        "scored_surface": dict(evaluation["scored_surface"]),
        "pruned_measures": list(pruned["measures"]),
        "pruned_families": dict(pruned["families"]),
        "receipt": _artifact_info(output_paths["score_receipt"]),
        "incumbent": dict(incumbent),
    }


def _output_paths(
    out_dir: Path,
    *,
    posture: UKRowwisePosture,
    vintage: str,
) -> dict[str, Path]:
    """The role's output paths for one FRS release vintage (``2024_25``)."""

    dataset = out_dir / posture.dataset_filename(vintage)
    if posture.role == "national":
        return {
            "dataset": dataset,
            "manifest": out_dir / MANIFEST_FILENAME,
            "calibration_diagnostics": out_dir / CALIBRATION_DIAGNOSTICS_FILENAME,
            "build_record": out_dir / BUILD_RECORD_FILENAME,
            "terminal_gates": out_dir / posture.gate_report_filename(vintage),
            "national_registry": out_dir / NATIONAL_REGISTRY_FILENAME,
            "contract_registry": out_dir / NATIONAL_CONTRACT_REGISTRY_FILENAME,
            "score_receipt": out_dir / SCORE_RECEIPT_FILENAME,
        }
    return {
        "dataset": dataset,
        "manifest": out_dir / MANIFEST_FILENAME,
        "diagnostics": out_dir / SOLVE_DIAGNOSTICS_FILENAME,
        "support": out_dir / AREA_SUPPORT_FILENAME,
        "past_cap": out_dir / PAST_CAP_FILENAME,
        "calibration_diagnostics": out_dir / CALIBRATION_DIAGNOSTICS_FILENAME,
        "local_gates": out_dir / posture.gate_report_filename(vintage),
        "local_registry": out_dir / LOCAL_REGISTRY_FILENAME,
        "dense_reference": out_dir / DENSE_REFERENCE_DIAGNOSTICS_FILENAME,
        "selection": out_dir / DATASET_SIZE_SELECTION_FILENAME,
    }


def _validate_output_paths(
    output_paths: Mapping[str, Path],
    *,
    input_h5: Path,
    ladder_path: Path | None,
) -> None:
    resolved = {name: path.resolve() for name, path in output_paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("candidate output paths must be distinct.")
    protected = {input_h5.resolve()}
    if ladder_path is not None:
        protected.add(ladder_path.resolve())
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
        "calibration_diagnostics",
        "local_registry",
        "dense_reference",
        "selection",
        "manifest",
    )
    published: list[Path] = []
    succeeded = False
    try:
        for key in publish_order:
            if key in _SIZE_RUN_ONLY_OUTPUTS and not staged[key].exists():
                continue
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


def _git_dirty() -> bool | None:
    """Measured, not asserted: tracked modifications in the working tree.

    ``None`` when git cannot answer (no repository), so a downstream
    assembler records the pin as unmeasured rather than clean.
    """

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


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
