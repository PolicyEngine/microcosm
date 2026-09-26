"""The UK rowwise release roles' shared command surface (microcosm#823).

Both UK dataset lines are built by one command whose ``--release-role``
fixes every solve default and refuses the other role's flags. The helpers
here are the object-free part of that surface: the role-defaulted argument
resolution, the posture-aware validator and the two refusal tables, the
recorded run parameters, the role's output filenames and the dense role's
Logbook attempt helpers. They were moved from
``tools/build_uk_rowwise_candidate.py`` so the graph full-build driver
(:mod:`microcosm.build.uk_runtime.full_build_cli`) and the rowwise tool
parse, default, refuse and record identically; the tool imports them back.

The private spellings (``_validate_cli_args`` and friends) are kept as
aliases so the drivers' tests can patch and call the names they always did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    apply_error_verdict,
    error_receipt_path,
    local_artifact_reference,
    record_terminal_attempt,
    write_error_receipt,
)
from microcosm.build.uk_runtime.ledger_targets import _spec_geography
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    UK_SAMPLE_SEED_DEFAULT,
)
from microcosm.build.uk_runtime.rowwise_posture import (
    UK_ROWWISE_DENSE_POSTURE,
    UKRowwisePosture,
    uk_rowwise_posture,
)
from microcosm.calibrate import TargetRegistry

__all__ = [
    "AREA_SUPPORT_FILENAME",
    "BUDGET_ITERS",
    "BUILD_RECORD_FILENAME",
    "CALIBRATION_DIAGNOSTICS_FILENAME",
    "CONSERVE_MASS",
    "DATASET_SIZE_SELECTION_FILENAME",
    "DENSE_REFERENCE_DIAGNOSTICS_FILENAME",
    "L0_LAMBDA",
    "LOCAL_REGISTRY_FILENAME",
    "MANIFEST_FILENAME",
    "NATIONAL_CONTRACT_REGISTRY_FILENAME",
    "NATIONAL_REGISTRY_FILENAME",
    "PAST_CAP_FILENAME",
    "REPOSITORY",
    "ROLE_DEFAULTED_ARGUMENTS",
    "SCORE_RECEIPT_FILENAME",
    "SIZE_RUN_ONLY_OUTPUTS",
    "SOLVE_DIAGNOSTICS_FILENAME",
    "TARGET_RECORDS",
    "UK_CANDIDATE_PIPELINE",
    "candidate_clone_counts_argument",
    "candidate_identity_digest",
    "doctrine_bounds",
    "git_commit",
    "git_dirty",
    "gate_failures_by_criticality",
    "is_release_blocking",
    "local_vintage_census",
    "json_text",
    "new_candidate_build_id",
    "output_paths",
    "posture_of",
    "record_candidate_attempt",
    "record_candidate_error",
    "refuse_dense_role_arguments",
    "refuse_national_role_arguments",
    "release_verdict",
    "resolve_role_arguments",
    "rowwise_parameters",
    "validate_cli_args",
]

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
SIZE_RUN_ONLY_OUTPUTS = frozenset({"dense_reference", "selection"})
_SIZE_RUN_ONLY_OUTPUTS = SIZE_RUN_ONLY_OUTPUTS

#: The solve options every rowwise run records beside its doctrine.
CONSERVE_MASS = False
TARGET_RECORDS: int | None = None
L0_LAMBDA = 0.0
BUDGET_ITERS = 10
_CONSERVE_MASS = CONSERVE_MASS
_TARGET_RECORDS = TARGET_RECORDS
_L0_LAMBDA = L0_LAMBDA
_BUDGET_ITERS = BUDGET_ITERS

# The dense role's Logbook pipeline, kept as a module name for the Logbook
# helpers and the contract-pin tests; the posture record
# (``rowwise_posture.py``) is the source of truth for both roles.
UK_CANDIDATE_PIPELINE = UK_ROWWISE_DENSE_POSTURE.pipeline
_UK_CANDIDATE_PIPELINE = UK_CANDIDATE_PIPELINE

#: The checkout the Logbook references anchor to (the workspace root above
#: ``packages/``); the current directory when the package runs installed.
REPOSITORY = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").is_file() and (parent / "packages").is_dir()
    ),
    Path.cwd(),
)
_REPOSITORY = REPOSITORY


def json_text(payload: Any) -> str:
    """The evidence files' JSON rendering: sorted keys, two-space indent."""

    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


_json_text = json_text


def new_candidate_build_id(
    *, seed: int, timestamp: datetime, rung: str = "f100"
) -> str:
    """The dense role's attempt id; the national role mints the seam's."""

    instant = timestamp.astimezone(UTC)
    return (
        f"{UK_ROWWISE_DENSE_POSTURE.build_id_prefix}{rung}-s{seed}-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


_new_candidate_build_id = new_candidate_build_id


def posture_of(args: argparse.Namespace) -> UKRowwisePosture:
    """The release-role posture bound to parsed arguments."""

    posture = getattr(args, "_posture", None)
    if not isinstance(posture, UKRowwisePosture):
        raise RuntimeError("arguments carry no release-role posture; parse them first.")
    return posture


_posture_of = posture_of


def candidate_clone_counts_argument(value: str) -> tuple[int, ...]:
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


_candidate_clone_counts_argument = candidate_clone_counts_argument


def doctrine_bounds(posture: UKRowwisePosture) -> dict[str, Any]:
    return posture.doctrine_bounds()


_doctrine_bounds = doctrine_bounds


def rowwise_parameters(args: argparse.Namespace, *, source_year: int) -> dict[str, Any]:
    """The run parameters a rowwise manifest, plan and identity digest record."""

    posture = posture_of(args)
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
        "doctrine": doctrine_bounds(posture),
        "solve_options": {
            "conserve_mass": CONSERVE_MASS,
            "target_records": TARGET_RECORDS,
            "l0_lambda": L0_LAMBDA,
            "budget_iters": BUDGET_ITERS,
        },
    }


_parameters = rowwise_parameters


def candidate_identity_digest(
    *,
    pins: dict[str, dict[str, object]],
    args: argparse.Namespace,
    source_year: int,
) -> str:
    payload = {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "inputs": pins,
        "parameters": rowwise_parameters(args, source_year=source_year),
        "source_year": source_year,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


_candidate_identity_digest = candidate_identity_digest


def record_candidate_attempt(
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
        pipeline=UK_CANDIDATE_PIPELINE,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


_record_candidate_attempt = record_candidate_attempt


def record_candidate_error(
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
        pipeline=UK_CANDIDATE_PIPELINE,
        error=error,
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(error_path, repository_hint=REPOSITORY)}#/error_type",
    )
    record_candidate_attempt(
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


_record_candidate_error = record_candidate_error


#: Solve arguments whose argparse default is ``None`` so an explicit value can
#: be told from the role's default: the other role's refusal table keys on
#: what was actually given.
ROLE_DEFAULTED_ARGUMENTS = (
    "n_clones",
    "seed",
    "sample_seed",
    "epochs",
    "learning_rate",
    "target_weight_rule",
    "expected_constituency_vintage",
)
_ROLE_DEFAULTED_ARGUMENTS = ROLE_DEFAULTED_ARGUMENTS


def resolve_role_arguments(args: argparse.Namespace) -> UKRowwisePosture:
    """Bind the declared role's posture and fill its defaults into unset arguments."""

    posture = uk_rowwise_posture(args.release_role)
    args._explicit_arguments = frozenset(
        name for name in ROLE_DEFAULTED_ARGUMENTS if getattr(args, name) is not None
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


_resolve_role_arguments = resolve_role_arguments


def validate_cli_args(args: argparse.Namespace) -> None:
    posture = posture_of(args)
    # The declared role is checked against the parameters first: the other
    # role's flags are refused by name before any value is range-checked.
    if posture.role == "national":
        refuse_dense_role_arguments(args, posture)
    else:
        refuse_national_role_arguments(args, posture)
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
    # The input pin is required only when an input H5 is given: a build
    # from a spine request executes the spine stages in its own graph and
    # has no checkpoint H5 to pin.
    input_pin_required = args.input_h5 is not None and args.input_sha256 is None
    if posture.ladder_required:
        if args.ladder is None:
            raise ValueError("--release-role dense requires --ladder.")
        if input_pin_required or args.ladder_sha256 is None:
            raise ValueError(
                "the joint registry path requires --input-sha256 and --ladder-sha256."
            )
    elif input_pin_required:
        raise ValueError("--release-role national requires --input-sha256.")
    if args.release_candidate:
        required_release = {
            "--ladder-sha256": args.ladder_sha256,
            "--ledger-facts": args.ledger_facts,
            "--ledger-facts-sha256": args.ledger_facts_sha256,
            "--ledger-manifest-sha256": args.ledger_manifest_sha256,
        }
        if args.input_h5 is not None:
            required_release = {"--input-sha256": args.input_sha256, **required_release}
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


_validate_cli_args = validate_cli_args


def refuse_dense_role_arguments(
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


_refuse_dense_role_arguments = refuse_dense_role_arguments


def refuse_national_role_arguments(
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


_refuse_national_role_arguments = refuse_national_role_arguments


def output_paths(
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


_output_paths = output_paths


def gate_failures_by_criticality(
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
        bucket = blocking if is_release_blocking(payload) else diagnostic
        bucket.extend(lines)
    return blocking, diagnostic


_gate_failures_by_criticality = gate_failures_by_criticality


def is_release_blocking(payload: Mapping[str, Any]) -> bool:
    """Fail-closed criticality read.

    Only an entry that explicitly declares ``criticality: diagnostic`` is
    exempt from vetoing the release; a missing or unknown criticality is
    treated as release-blocking, so partial schema drift on one persisted
    entry cannot drop a failed gate out of both the blocking list and
    ``all_gates_passed``.
    """

    return payload.get("criticality") != "diagnostic"


_is_release_blocking = is_release_blocking


def release_verdict(
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


_release_verdict = release_verdict


def local_vintage_census(registry: TargetRegistry) -> list[dict[str, object]]:
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


_local_vintage_census = local_vintage_census


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


_git_commit = git_commit


def git_dirty() -> bool | None:
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


_git_dirty = git_dirty
