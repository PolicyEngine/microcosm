"""Launch one immutable US exact-record-count ladder release.

The launcher owns only strict configuration, artifact pins, ladder-point
resolution, and the non-publishing package receipt.  Target materialization,
checkpointing, calibration gates, diagnostics, and release manifests remain in
``build_us_fiscal_refresh_release.py`` so the ladder cannot drift from the
incumbent's frozen-register release path.

Example::

    uv run python tools/build_us_exact_k_ladder_release.py \
        --pool-manifest /artifacts/pool.manifest.json \
        --config configs/us_exact_k_57240.json \
        --out build/us-exact-k

Schema-v2 configuration (paths are resolved relative to the config file)::

    {
      "schema_version": 2,
      "family": {"id": "12345678-1234-4234-9234-123456789abc"},
      "pool": {"release_id": "...", "manifest_sha256": "<sha256>"},
      "ladder": {"k": 57240, "seed": 17, "pi_hi": 0.95},
      "targets": {
        "ledger_facts": "...", "ledger_facts_sha256": "<sha256>",
        "ledger_manifest_sha256": "<sha256>",
        "incumbent_diagnostics": "...",
        "incumbent_diagnostics_sha256": "<sha256>",
        "target_surface_sha256": "<sha256>"
      },
      "calibration": {
        "epochs": 256, "learning_rate": 0.02, "max_weight_ratio": 20,
        "l0_refit_lambda_share": 0.8, "l2_lambda": 0,
        "refit_l2_lambda": 0
      },
      "release": {"id": "populace-us-2024-k57240-...",
                  "repo_id": "policyengine/populace-us"}
    }

``ladder.k`` accepts exactly ``"N"``, ``57240``, or ``20000``. ``targets``
may additionally carry the paired
``ssi_take_up_prior_weight_basis`` and
``ssi_take_up_prior_weight_basis_sha256`` fields for the house one-retry
delivery-gate protocol.
"""

# The sibling house builder is importable only after its tools directory is
# installed below.
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import build_us_fiscal_refresh_release as fiscal_release
from microcosm.build.logbook import (
    LogbookWriteResult,
    canonical_json_bytes,
    record_build_attempt,
)
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    attempt_receipt_dir,
    atomic_write_json,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    resolve_predecessor,
    sha256_argument,
    write_error_receipt,
)
from microcosm.build.logbook_family import (
    FamilyMember,
    LogbookFamily,
    reconcile_logbook_spool,
    record_family,
    record_family_member,
)
from microcosm.build.us_runtime.h5_io import (
    load_simulation_ready_us_multispine_pool_manifest,
)

CONFIG_SCHEMA_VERSION = 2
RATIFIED_SPARSE_K = fiscal_release.RATIFIED_EXACT_K_COUNTS
US_RELEASE_REPO_ID = "policyengine/populace-us"
_LOGBOOK_PIPELINE = "us-2024-release"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_RELEASE_ID = re.compile(r"[A-Za-z0-9-]+")


@dataclass(frozen=True)
class LadderReleaseConfig:
    """Validated, path-resolved launcher configuration."""

    family_id: str
    pool_release_id: str
    pool_manifest_sha256: str
    requested_k: str | int
    seed: int
    pi_hi: float
    ledger_facts: Path
    ledger_facts_sha256: str
    ledger_manifest_sha256: str
    incumbent_diagnostics: Path
    incumbent_diagnostics_sha256: str
    target_surface_sha256: str
    ssi_take_up_prior_weight_basis: Path | None
    ssi_take_up_prior_weight_basis_sha256: str | None
    epochs: int
    learning_rate: float
    max_weight_ratio: float
    l0_refit_lambda_share: float
    l2_lambda: float
    refit_l2_lambda: float
    release_id: str
    repo_id: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        required=True,
        help="Simulation-ready manifest produced by build_us_multispine_pool.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Strict schema-v2 JSON release configuration.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help=(
            "Current US Logbook row checksum. May instead be supplied through "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST."
        ),
    )
    return parser.parse_args(argv)


def _read_config(path: Path) -> LadderReleaseConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ladder config {path} is not valid JSON: {exc}.") from exc
    root = _object(payload, label=f"ladder config {path}")
    _keys(
        root,
        required={
            "schema_version",
            "family",
            "pool",
            "ladder",
            "targets",
            "calibration",
            "release",
        },
        label=f"ladder config {path}",
    )
    if (
        isinstance(root["schema_version"], bool)
        or root["schema_version"] != CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Ladder config {path} schema_version must be "
            f"{CONFIG_SCHEMA_VERSION}, got {root['schema_version']!r}."
        )

    family = _object(root["family"], label="family")
    _keys(family, required={"id"}, label="family")
    pool = _object(root["pool"], label="pool")
    _keys(pool, required={"release_id", "manifest_sha256"}, label="pool")
    ladder = _object(root["ladder"], label="ladder")
    _keys(ladder, required={"k", "seed", "pi_hi"}, label="ladder")
    targets = _object(root["targets"], label="targets")
    _keys(
        targets,
        required={
            "ledger_facts",
            "ledger_facts_sha256",
            "ledger_manifest_sha256",
            "incumbent_diagnostics",
            "incumbent_diagnostics_sha256",
            "target_surface_sha256",
        },
        optional={
            "ssi_take_up_prior_weight_basis",
            "ssi_take_up_prior_weight_basis_sha256",
        },
        label="targets",
    )
    calibration = _object(root["calibration"], label="calibration")
    _keys(
        calibration,
        required={
            "epochs",
            "learning_rate",
            "max_weight_ratio",
            "l0_refit_lambda_share",
            "l2_lambda",
            "refit_l2_lambda",
        },
        label="calibration",
    )
    release = _object(root["release"], label="release")
    _keys(release, required={"id", "repo_id"}, label="release")

    requested_k = ladder["k"]
    if requested_k != "N" and (
        isinstance(requested_k, bool)
        or not isinstance(requested_k, int)
        or requested_k not in RATIFIED_SPARSE_K
    ):
        raise ValueError(
            f"ladder.k must be exactly 'N', 57240, or 20000; got {requested_k!r}."
        )
    seed = _nonnegative_int(ladder["seed"], label="ladder.seed")
    pi_hi = _finite_number(ladder["pi_hi"], label="ladder.pi_hi")
    if not 0.0 <= pi_hi <= 1.0:
        raise ValueError(f"ladder.pi_hi must be in [0, 1], got {pi_hi!r}.")

    config_dir = path.resolve().parent
    ledger_facts = _resolve_path(targets["ledger_facts"], config_dir, "ledger_facts")
    incumbent = _resolve_path(
        targets["incumbent_diagnostics"],
        config_dir,
        "incumbent_diagnostics",
    )
    prior_basis_value = targets.get("ssi_take_up_prior_weight_basis")
    prior_basis_sha_value = targets.get("ssi_take_up_prior_weight_basis_sha256")
    if (prior_basis_value is None) != (prior_basis_sha_value is None):
        raise ValueError(
            "targets.ssi_take_up_prior_weight_basis and its SHA-256 pin must "
            "be provided together."
        )
    prior_basis = (
        None
        if prior_basis_value is None
        else _resolve_path(
            prior_basis_value,
            config_dir,
            "ssi_take_up_prior_weight_basis",
        )
    )
    prior_basis_sha = (
        None
        if prior_basis_sha_value is None
        else _sha256_value(
            prior_basis_sha_value,
            label="targets.ssi_take_up_prior_weight_basis_sha256",
        )
    )
    epochs = _positive_int(calibration["epochs"], label="calibration.epochs")
    learning_rate = _positive_number(
        calibration["learning_rate"], label="calibration.learning_rate"
    )
    max_weight_ratio = _finite_number(
        calibration["max_weight_ratio"], label="calibration.max_weight_ratio"
    )
    if max_weight_ratio < 1.0:
        raise ValueError("calibration.max_weight_ratio must be at least 1.")
    l0_share = _positive_number(
        calibration["l0_refit_lambda_share"],
        label="calibration.l0_refit_lambda_share",
    )
    l2_lambda = _nonnegative_number(
        calibration["l2_lambda"], label="calibration.l2_lambda"
    )
    refit_l2_lambda = _nonnegative_number(
        calibration["refit_l2_lambda"],
        label="calibration.refit_l2_lambda",
    )

    pool_release_id = _nonempty_string(pool["release_id"], label="pool.release_id")
    release_id = _nonempty_string(release["id"], label="release.id")
    if _RELEASE_ID.fullmatch(release_id) is None:
        raise ValueError(
            "release.id may contain only ASCII letters, digits, and hyphens."
        )
    repo_id = _nonempty_string(release["repo_id"], label="release.repo_id")
    if repo_id != US_RELEASE_REPO_ID:
        raise ValueError(
            f"release.repo_id must be {US_RELEASE_REPO_ID!r}, got {repo_id!r}."
        )

    return LadderReleaseConfig(
        family_id=_uuid_value(family["id"], label="family.id"),
        pool_release_id=pool_release_id,
        pool_manifest_sha256=_sha256_value(
            pool["manifest_sha256"], label="pool.manifest_sha256"
        ),
        requested_k=requested_k,
        seed=seed,
        pi_hi=pi_hi,
        ledger_facts=ledger_facts,
        ledger_facts_sha256=_sha256_value(
            targets["ledger_facts_sha256"], label="targets.ledger_facts_sha256"
        ),
        ledger_manifest_sha256=_sha256_value(
            targets["ledger_manifest_sha256"],
            label="targets.ledger_manifest_sha256",
        ),
        incumbent_diagnostics=incumbent,
        incumbent_diagnostics_sha256=_sha256_value(
            targets["incumbent_diagnostics_sha256"],
            label="targets.incumbent_diagnostics_sha256",
        ),
        target_surface_sha256=_sha256_value(
            targets["target_surface_sha256"],
            label="targets.target_surface_sha256",
        ),
        ssi_take_up_prior_weight_basis=prior_basis,
        ssi_take_up_prior_weight_basis_sha256=prior_basis_sha,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=max_weight_ratio,
        l0_refit_lambda_share=l0_share,
        l2_lambda=l2_lambda,
        refit_l2_lambda=refit_l2_lambda,
        release_id=release_id,
        repo_id=repo_id,
    )


def _validate_pins_and_resolve_k(
    *,
    config: LadderReleaseConfig,
    pool_manifest_path: Path,
) -> tuple[int, Mapping[str, object]]:
    pool_manifest = load_simulation_ready_us_multispine_pool_manifest(
        pool_manifest_path,
        expected_manifest_sha256=config.pool_manifest_sha256,
    )
    fiscal_release._assert_pool_release_identity(
        config.pool_release_id,
        pool_manifest,
    )
    agreement_gate = _object(
        pool_manifest.get("agreement_gate"), label="pool manifest agreement_gate"
    )
    if agreement_gate.get("passed") is not True:
        raise ValueError("Pool manifest has no passing agreement-gate verdict.")
    counts = _object(
        pool_manifest.get("provenance_counts"),
        label="pool manifest provenance_counts",
    )
    household_counts = _object(
        counts.get("household"),
        label="pool manifest provenance_counts.household",
    )
    pool_size = _positive_int(
        household_counts.get("rows"),
        label="pool manifest provenance_counts.household.rows",
    )
    k = pool_size if config.requested_k == "N" else int(config.requested_k)
    if k > pool_size:
        raise ValueError(
            f"k={k} exceeds the pool size {pool_size}; ladder selection never "
            "clamps the requested cardinality."
        )
    expected_prefix = f"populace-us-2024-k{k}-"
    if not config.release_id.startswith(expected_prefix):
        raise ValueError(
            f"release.id must start with {expected_prefix!r} for resolved k={k}; "
            f"got {config.release_id!r}."
        )

    if not config.ledger_facts.exists():
        raise FileNotFoundError(
            f"targets.ledger_facts does not exist: {config.ledger_facts}"
        )
    if not config.incumbent_diagnostics.is_file():
        raise FileNotFoundError(
            "targets.incumbent_diagnostics is not a file: "
            f"{config.incumbent_diagnostics}"
        )
    incumbent_bytes = config.incumbent_diagnostics.read_bytes()
    observed_incumbent_sha256 = hashlib.sha256(incumbent_bytes).hexdigest()
    if observed_incumbent_sha256 != config.incumbent_diagnostics_sha256:
        raise ValueError(
            "Incumbent diagnostics SHA-256 mismatch: got "
            f"{observed_incumbent_sha256}, expected "
            f"{config.incumbent_diagnostics_sha256}."
        )
    incumbent = _object(
        json.loads(incumbent_bytes),
        label="incumbent diagnostics",
    )
    target_surface = _object(
        incumbent.get("target_surface"),
        label="incumbent diagnostics target_surface",
    )
    if target_surface.get("sha256") != config.target_surface_sha256:
        raise ValueError(
            "Incumbent diagnostics target-surface SHA-256 mismatch: got "
            f"{target_surface.get('sha256')!r}, expected "
            f"{config.target_surface_sha256!r}."
        )
    if config.ssi_take_up_prior_weight_basis is not None:
        if not config.ssi_take_up_prior_weight_basis.is_file():
            raise FileNotFoundError(
                "targets.ssi_take_up_prior_weight_basis is not a file: "
                f"{config.ssi_take_up_prior_weight_basis}"
            )
        observed_prior_basis_sha256 = _sha256(config.ssi_take_up_prior_weight_basis)
        if observed_prior_basis_sha256 != (
            config.ssi_take_up_prior_weight_basis_sha256
        ):
            raise ValueError(
                "SSI take-up prior-weight basis SHA-256 mismatch: got "
                f"{observed_prior_basis_sha256}, expected "
                f"{config.ssi_take_up_prior_weight_basis_sha256}."
            )
    return k, pool_manifest


def _builder_argv(
    *,
    config: LadderReleaseConfig,
    pool_manifest: Path,
    out: Path,
    k: str | int,
) -> list[str]:
    argv = [
        "--pool-manifest",
        str(pool_manifest),
        "--pool-manifest-sha256",
        config.pool_manifest_sha256,
        "--pool-release-id",
        config.pool_release_id,
        "--exact-k",
        str(k),
        "--exact-k-pi-hi",
        str(config.pi_hi),
        "--ledger-facts",
        str(config.ledger_facts),
        "--ledger-facts-sha256",
        config.ledger_facts_sha256,
        "--ledger-manifest-sha256",
        config.ledger_manifest_sha256,
        "--incumbent-diagnostics",
        str(config.incumbent_diagnostics),
        "--incumbent-diagnostics-sha256",
        config.incumbent_diagnostics_sha256,
        "--frozen-target-surface-sha256",
        config.target_surface_sha256,
        "--out",
        str(out),
        "--release-id",
        config.release_id,
        "--seed",
        str(config.seed),
        "--epochs",
        str(config.epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--max-weight-ratio",
        str(config.max_weight_ratio),
        "--l0-refit-lambda-share",
        str(config.l0_refit_lambda_share),
        "--l2-lambda",
        str(config.l2_lambda),
        "--refit-l2-lambda",
        str(config.refit_l2_lambda),
        "--no-staging",
    ]
    if config.ssi_take_up_prior_weight_basis is not None:
        argv.extend(
            [
                "--ssi-take-up-prior-weight-basis",
                str(config.ssi_take_up_prior_weight_basis),
                "--ssi-take-up-prior-weight-basis-sha256",
                str(config.ssi_take_up_prior_weight_basis_sha256),
            ]
        )
    return argv


def _logbook_input_pins_digest(config: LadderReleaseConfig) -> str:
    payload = {
        "pool_manifest_sha256": config.pool_manifest_sha256,
        "ledger_facts_sha256": config.ledger_facts_sha256,
        "ledger_manifest_sha256": config.ledger_manifest_sha256,
        "incumbent_diagnostics_sha256": config.incumbent_diagnostics_sha256,
        "target_surface_sha256": config.target_surface_sha256,
        "ssi_take_up_prior_weight_basis_sha256": (
            config.ssi_take_up_prior_weight_basis_sha256
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _logbook_identity_digest(
    config: LadderReleaseConfig,
    *,
    resolved_k: int | None,
) -> str:
    payload = {
        "pipeline": _LOGBOOK_PIPELINE,
        "build_id": config.release_id,
        "family_id": config.family_id,
        "source_pool_sha256": config.pool_manifest_sha256,
        "requested_k_input": config.requested_k,
        "requested_k_resolved": resolved_k,
        "record_unit": "household" if resolved_k is not None else None,
        "seed": config.seed,
        "pi_hi": config.pi_hi,
        "calibration": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "max_weight_ratio": config.max_weight_ratio,
            "l0_refit_lambda_share": config.l0_refit_lambda_share,
            "l2_lambda": config.l2_lambda,
            "refit_l2_lambda": config.refit_l2_lambda,
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record_exact_k_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    code_pin: str,
    seed: int,
    predecessor: str | None,
    spool_dir: Path,
    requested_k: int | None,
    realized_k: int | None,
    disposition: str,
) -> LogbookWriteResult:
    return record_build_attempt(
        build_id=state.build_id,
        ts=started_ts,
        pipeline=_LOGBOOK_PIPELINE,
        rung=None,
        seed=seed,
        code_pin=code_pin,
        input_pins_digest=state.input_pins_digest,
        identity_digest=state.identity_digest,
        phases_reached=state.phases_reached,
        gate_verdicts=state.gate_verdicts,
        wall_seconds=time.perf_counter() - started_at,
        cost_usd=None,
        artifact_location=state.artifact_location,
        disposition=disposition,
        prediction_id=None,
        prev_row_digest=predecessor,
        row_format_version=2,
        requested_k=requested_k,
        realized_k=realized_k,
        record_unit="household" if requested_k is not None else None,
        spool_dir=spool_dir,
        post_remote=False,
    )


def launch(
    *,
    pool_manifest: Path,
    config_path: Path,
    out: Path,
    logbook_prev_row_digest: str | None = None,
    release_builder: Callable[[Sequence[str] | None], object] = fiscal_release.main,
) -> dict[str, object]:
    """Validate pins, build, persist Logbook records, and write a receipt."""

    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    config = _read_config(config_path)
    resolved_pool_manifest = pool_manifest.resolve()
    resolved_out = out.resolve()
    spool_dir = resolved_out / "logbook-spool"
    predecessor = resolve_predecessor(logbook_prev_row_digest)
    code_pin = git_code_pin(_REPOSITORY_ROOT)
    requested_k = (
        int(config.requested_k) if isinstance(config.requested_k, int) else None
    )
    success_receipt = (
        attempt_receipt_dir(resolved_out, build_id=config.release_id) / "release.json"
    )
    success_receipt_reference = local_artifact_reference(
        success_receipt,
        repository_hint=_REPOSITORY_ROOT,
    )
    state = AttemptState(
        build_id=config.release_id,
        identity_digest=_logbook_identity_digest(
            config=config,
            resolved_k=requested_k,
        ),
        input_pins_digest=_logbook_input_pins_digest(config),
        phases_reached=["attempt_started"],
        gate_verdicts={
            "exact_k_build": {
                "verdict": "running",
                "receipt": success_receipt_reference,
            }
        },
    )
    family = LogbookFamily.create(
        family_id=config.family_id,
        chain_scope="us",
        source_pool_sha256=config.pool_manifest_sha256,
    )

    try:
        k, _ = _validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=resolved_pool_manifest,
        )
        requested_k = k
        state.identity_digest = _logbook_identity_digest(
            config,
            resolved_k=requested_k,
        )
        append_phase(state, "source_pool_verified")
        record_family(family, spool_dir=spool_dir, post_remote=False)
        append_phase(state, "family_record_spooled")
        release_builder(
            _builder_argv(
                config=config,
                pool_manifest=resolved_pool_manifest,
                out=resolved_out,
                k=config.requested_k,
            )
        )
        append_phase(state, "dataset_built")
        build = {
            "release_id": config.release_id,
            "release_dir": str(resolved_out / "releases" / config.release_id),
            "artifact_root": str(resolved_out / "artifacts"),
        }
        atomic_write_json(
            success_receipt,
            {
                "artifact_kind": "populace_exact_k_release_receipt",
                "schema_version": 1,
                **build,
                "family_id": config.family_id,
                "source_pool_sha256": config.pool_manifest_sha256,
                "requested_k": requested_k,
                "realized_k": requested_k,
                "record_unit": "household",
                "rung": None,
            },
        )
        state.gate_verdicts = {
            "exact_k_build": {
                "verdict": "passed",
                "receipt": success_receipt_reference,
            }
        }
        state.artifact_location = local_artifact_reference(
            Path(build["release_dir"]),
            repository_hint=_REPOSITORY_ROOT,
        )
    except BaseException as error:
        try:
            failure_path = write_error_receipt(
                error_receipt_path(resolved_out, build_id=config.release_id),
                state=state,
                pipeline=_LOGBOOK_PIPELINE,
                error=error,
            )
            failure_reference = local_artifact_reference(
                failure_path,
                repository_hint=_REPOSITORY_ROOT,
            )
            state.gate_verdicts = {
                "exact_k_build": {
                    "verdict": "error",
                    "receipt": failure_reference,
                }
            }
            state.artifact_location = failure_reference
            append_phase(state, "error")
            _record_exact_k_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                code_pin=code_pin,
                seed=config.seed,
                predecessor=predecessor,
                spool_dir=spool_dir,
                requested_k=requested_k,
                realized_k=None,
                disposition="failed",
            )
            reconcile_logbook_spool(spool_dir)
        except Exception as recording_error:
            error.add_note(
                "Exact-k Logbook failure recording also failed: "
                f"{type(recording_error).__name__}: {recording_error}"
            )
        raise

    assert requested_k is not None
    write_result = _record_exact_k_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        code_pin=code_pin,
        seed=config.seed,
        predecessor=predecessor,
        spool_dir=spool_dir,
        requested_k=requested_k,
        realized_k=requested_k,
        disposition="iterating",
    )
    member = FamilyMember.create(
        family_id=config.family_id,
        build_id=config.release_id,
    )
    record_family_member(member, spool_dir=spool_dir, post_remote=False)
    reconcile_logbook_spool(spool_dir)

    publish_argv = [
        "tools/publish_release.sh",
        build["release_dir"],
        "--repo-id",
        config.repo_id,
        "--artifact-root",
        build["artifact_root"],
        "--create-tag",
        "--no-latest",
        "--tag-only",
    ]
    result: dict[str, object] = {
        **build,
        "k": k,
        "family_id": config.family_id,
        "requested_k": requested_k,
        "realized_k": requested_k,
        "record_unit": "household",
        "rung": None,
        "logbook_row_digest": write_result.row.row_digest,
        "seed": config.seed,
        "automatic_publish": False,
        "pointer_update": False,
        "pointer_updates": {
            "production": {
                "repo_id": config.repo_id,
                "pointer_update": False,
            },
            "staging": {
                "repo_id": os.environ.get(
                    "POPULACE_STAGING_REPO_ID",
                    "policyengine/populace-us-staging",
                ),
                "pointer_update": False,
            },
        },
        "publish_argv": publish_argv,
        "publish_command": shlex.join(publish_argv),
    }
    package_result = resolved_out / "package_result.json"
    atomic_write_json(package_result, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        raise ValueError(
            f"{label} keys do not match schema; missing={missing}, unknown={unknown}."
        )


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _sha256_value(value: object, *, label: str) -> str:
    parsed = _nonempty_string(value, label=label)
    if _LOWERCASE_SHA256.fullmatch(parsed) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256.")
    return parsed


def _uuid_value(value: object, *, label: str) -> str:
    parsed = _nonempty_string(value, label=label)
    try:
        normalized = str(uuid.UUID(parsed))
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UUID.") from exc
    if parsed != normalized:
        raise ValueError(f"{label} must use canonical lowercase UUID text.")
    return normalized


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _positive_int(value: object, *, label: str) -> int:
    parsed = _nonnegative_int(value, label=label)
    if parsed == 0:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    return parsed


def _positive_number(value: object, *, label: str) -> float:
    parsed = _finite_number(value, label=label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive, got {parsed!r}.")
    return parsed


def _nonnegative_number(value: object, *, label: str) -> float:
    parsed = _finite_number(value, label=label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be non-negative, got {parsed!r}.")
    return parsed


def _resolve_path(value: object, base: Path, label: str) -> Path:
    raw = _nonempty_string(value, label=f"targets.{label}")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    return launch(
        pool_manifest=args.pool_manifest,
        config_path=args.config,
        out=args.out,
        logbook_prev_row_digest=args.logbook_prev_row_digest,
    )


if __name__ == "__main__":
    main()
