"""Staging telemetry and staged-dataset delivery of the UK rowwise roles.

Two destinations, one run id: reviewed aggregate telemetry goes to the
staging repository under ``runs/<run_id>/``, the finished bundle a manifest
vouches for to the private artifact repository under ``staged/<run_id>/``.
Both are best-effort evidence about the build, never a release, and the
build must never abort on its own progress report.

These helpers were moved from ``tools/build_uk_rowwise_candidate.py`` so the
graph full-build driver and the rowwise tool stage identically; the tool
imports them back. ``_hub_api`` and ``_hub_token`` are the test seams the
drivers' tests patch on this module.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from microcosm.build.staging_dataset import (
    StagedDatasetBundle,
    disabled_staged_dataset,
    local_only_staged_dataset,
    stage_bundle,
    write_sidecars,
)
from microcosm.build.staging_storage import HuggingFaceDatasetStorage
from microcosm.build.staging_v2 import (
    StagingContractError,
    StagingTelemetryV2,
    disabled_staging_delivery,
)
from microcosm.build.uk_runtime.rowwise_cli import (
    BUDGET_ITERS,
    MANIFEST_FILENAME,
    SIZE_RUN_ONLY_OUTPUTS,
    json_text,
    posture_of,
)
from microcosm.build.uk_runtime.staging import UK_STAGED_DATASET_PREFIX

__all__ = [
    "STAGED_DATASET_PHASES",
    "STAGING_EPOCH_EVERY",
    "STAGING_MAX_EPOCH_ROWS",
    "STAGING_UPLOAD_INTERVAL_SECONDS",
    "add_staging_artifact",
    "create_staging_telemetry",
    "fail_staging_telemetry",
    "finalize_staging_telemetry",
    "fit_summary",
    "gate_statuses",
    "preflight_staged_dataset",
    "publish_staged_files",
    "replace_manifest",
    "stage",
    "stage_dataset",
    "staged_dataset_mode",
    "staging_delivery",
    "staging_epoch_every",
    "thinned_epochs",
]

# Best-effort telemetry upload cadence. The Hub allows about 128 commits per
# hour per repository and one cycle is up to eight single-file commits, so
# the shared 30-second default exhausts the budget on a multi-hour solve
# and loses uploads (the v20 national run did); five minutes keeps a
# 1,500-epoch run well inside it.
STAGING_UPLOAD_INTERVAL_SECONDS = 300.0
# Staging telemetry keeps one row per forwarded epoch in
# calibration_progress.json and one event in events.ndjson, both under the
# contract's 5 MiB remote cap. A size run at 2,000 epochs solves the dense
# pool, up to ten full-length L0 probes and the refit: about 24,000 epochs,
# which would breach the cap mid-run. Forwarding every tenth epoch and the
# last epoch of each phase keeps the loss curve and stays near 0.8 MB.
STAGING_EPOCH_EVERY = 10
# ...and never more than this many forwarded epochs per run, whatever --epochs
# says: the stride grows with the run so the cap holds by construction.
STAGING_MAX_EPOCH_ROWS = 2400
STAGED_DATASET_PHASES = {
    "uploaded": "dataset_staged",
    "already_staged": "dataset_staged",
    "failed": "dataset_stage_failed",
    "skipped": "dataset_stage_skipped",
}
_STAGING_UPLOAD_INTERVAL_SECONDS = STAGING_UPLOAD_INTERVAL_SECONDS
_STAGING_EPOCH_EVERY = STAGING_EPOCH_EVERY
_STAGING_MAX_EPOCH_ROWS = STAGING_MAX_EPOCH_ROWS
_STAGED_DATASET_PHASES = STAGED_DATASET_PHASES


def _hub_api() -> Any:
    """The Hub client used for telemetry and the staged dataset (test seam)."""

    from huggingface_hub import HfApi

    return HfApi()


def _hub_token() -> str | None:
    """The ambient Hub credential, if any (test seam)."""

    from huggingface_hub import get_token

    return get_token()


def staged_dataset_mode(args: argparse.Namespace) -> str:
    if args.no_staging or args.no_staged_dataset:
        return "disabled"
    if args.staging_local_only:
        return "local_only"
    return "local_and_remote"


_staged_dataset_mode = staged_dataset_mode


def preflight_staged_dataset(args: argparse.Namespace) -> None:
    """Refuse a remote dataset stage the run could not complete.

    The bundle upload is the last step of a multi-hour run, so the credential
    and the repository are checked before the spine is read. Telemetry stays
    best-effort with no pre-flight, as on the national command.
    """

    if args.dry_run or staged_dataset_mode(args) != "local_and_remote":
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


_preflight_staged_dataset = preflight_staged_dataset


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


def create_staging_telemetry(
    args: argparse.Namespace, *, build_id: str
) -> StagingTelemetryV2 | None:
    if args.no_staging:
        return None
    local_only = bool(args.staging_local_only)
    out_dir = args.out.expanduser().resolve()
    posture = posture_of(args)
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


_create_staging_telemetry = create_staging_telemetry


def stage(
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


_stage = stage


def staging_epoch_every(args: argparse.Namespace) -> int:
    """The epoch stride that keeps the forwarded rows under the row budget.

    A dense run solves once; a size run solves the pool, up to ``budget_iters``
    full-length probes and the refit. The stride is at least
    ``STAGING_EPOCH_EVERY`` and grows so at most ``STAGING_MAX_EPOCH_ROWS``
    epochs are forwarded, keeping ``calibration_progress.json`` and
    ``events.ndjson`` under the contract's 5 MiB cap for any ``--epochs``.
    """

    solves = 1 if args.dataset_households is None else 2 + BUDGET_ITERS
    total = int(args.epochs) * solves
    return max(STAGING_EPOCH_EVERY, -(-total // STAGING_MAX_EPOCH_ROWS))


_staging_epoch_every = staging_epoch_every


def thinned_epochs(
    sink: Callable[[Mapping[str, Any]], None], *, every: int = STAGING_EPOCH_EVERY
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


_thinned_epochs = thinned_epochs


def gate_statuses(gate_report: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(gate_id): str(entry.get("status"))
        for gate_id, entry in gate_report.get("gates", {}).items()
        if isinstance(entry, Mapping)
    }


_gate_statuses = gate_statuses


def fail_staging_telemetry(
    telemetry: StagingTelemetryV2 | None, error: BaseException
) -> None:
    if telemetry is None or telemetry.status != "running":
        return
    try:
        telemetry.fail(error)
        telemetry.validate_local_bundle()
    except Exception:
        pass


_fail_staging_telemetry = fail_staging_telemetry


def finalize_staging_telemetry(
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


_finalize_staging_telemetry = finalize_staging_telemetry


def _warn_telemetry(what: str, error: BaseException) -> None:
    print(
        f"warning: {what} ({type(error).__name__}: {error}); the build's own "
        "evidence is unaffected.",
        file=sys.stderr,
        flush=True,
    )


def staging_delivery(telemetry: StagingTelemetryV2 | None) -> dict[str, Any]:
    if telemetry is None:
        return disabled_staging_delivery("--no-staging")
    return telemetry.delivery_summary


_staging_delivery = staging_delivery


def stage_dataset(
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

    mode = staged_dataset_mode(args)
    if mode == "disabled":
        return disabled_staged_dataset(
            "--no-staging" if args.no_staging else "--no-staged-dataset"
        )
    repository = (
        None if mode == "local_only" else str(args.staged_dataset_repo_id).strip()
    )
    stage(telemetry, "dataset_staging", "started", mode=mode, repository=repository)
    statuses = gate_statuses(getattr(args, "_gate_report", {}) or {})
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
            "gate_statuses": statuses,
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
    stage(
        telemetry,
        "dataset_staging",
        "completed",
        status=delivery["status"],
        repository=delivery["repository"],
        revision=delivery["revision"],
        error_code=delivery["error_code"],
        file_count=len(delivery["files"]),
    )
    add_staging_artifact(
        telemetry,
        "staged_dataset",
        delivery,
        artifact_kind="build_metadata",
        classification="non_row_level",
    )
    add_staging_artifact(
        telemetry,
        "fit_summary",
        fit_summary(
            manifest,
            run_id=run_id,
            gate_statuses=statuses,
            staged_dataset=delivery,
        ),
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    return delivery


_stage_dataset = stage_dataset


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


def add_staging_artifact(
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
        source.write_text(json_text(payload), encoding="utf-8")
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


_add_staging_artifact = add_staging_artifact


def fit_summary(
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


_fit_summary = fit_summary


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


def replace_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Rewrite the published manifest atomically with appended evidence."""

    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with open(handle, "w", encoding="utf-8") as stream:
            stream.write(json_text(manifest))
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


_replace_manifest = replace_manifest


def publish_staged_files(
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
            if key in SIZE_RUN_ONLY_OUTPUTS and not staged[key].exists():
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


_publish_staged_files = publish_staged_files
