"""Assemble one publishable UK release from a signed national staging run.

Modern UK release identity is deliberately not the retired June pipeline's
``<uk-data7>-<populace7>-<timestamp>`` tuple.  The current build and data
contracts ratify the exact-cardinality form
``populace-uk-<year>-<tier>-k<N>``: ``year`` and the licensed spine ``tier``
are explicit operator metadata, while ``N`` is derived from the completed
staging build record and cross-checked against the diagnostics, signed report,
and checkpoint run context.  There is no dead ``uk-data`` component.

The output follows the US builders' local layout.  ``--releases-root`` gets
``<release_id>/`` with the contract files; its sibling ``artifacts/`` (or an
explicit ``--artifact-root``) gets the canonical H5 and calibration receipt.
That directory is the value to pass as ``microcosm-publish-release
--artifact-root``.  The staging H5 remains the source of record-level weights;
the emitted NPZ is a compact, schema-versioned receipt of the weight summary
and target diagnostics bound into the signed report.

Example::

    # Mint the immutable id before starting the release-candidate build.
    uv run python tools/build_uk_release_dir.py \
        --print-release-id \
        --candidate-h5 /gated/populace_uk_2023.h5 \
        --year 2023 \
        --tier frs

    uv run python tools/build_uk_release_dir.py \
        --staging-dir build/uk-national \
        --calibration-diagnostics build/calibration_diagnostics.json \
        --releases-root build/releases \
        --year 2023 \
        --tier frs \
        --built-at 2026-08-20T18:00:00Z \
        --git-commit <staging-builder-commit>

``--built-at`` and ``--git-commit`` are fallbacks for staging receipts that do
not record those values; when the build record or run context already carries
either value, an explicit value must agree.  Recorded runtime versions are
preferred.  The current schema-3 staging receipt records Python and the numeric
stack but not PolicyEngine UK/Core, so those two versions are captured from the
assembler environment, matching the established US release-builder behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from packaging.version import InvalidVersion, Version

from microcosm.build.uk_runtime.hmrc_restoration import (
    verify_certified_uk_candidate,
)
from microcosm.build.uk_runtime.release_identity import (
    UK_RELEASE_TIERS,
    UKReleaseIdentity,
)
from microcosm.data import RELEASE_MANIFEST_SCHEMA_VERSION, validate_release_dir

UK_RELEASE_REPO_ID = "policyengine/populace-uk-private"
UK_DATASET_KEY_PREFIX = "populace_uk_"
UK_TERMINAL_REPORT = "terminal_gates.json"
UK_STAGING_BUILD_SCHEMA_VERSION = 3
UK_GATE_BATTERY_SCHEMA_VERSION = 4
UK_STAGE_CONTEXT_SCHEMA_VERSION = 2
UK_STAGE_CONTEXT_ARTIFACT_KIND = "populace_outer_stage_run_context"
UK_CALIBRATION_RECEIPT_SCHEMA_VERSION = 1
UK_HOUSEHOLD_TABLE_PATH = "household/table"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-release-id",
        action="store_true",
        help=(
            "Verify --candidate-h5, mint the canonical exact-k release id from "
            "its household-row count, print the id, and exit without assembling."
        ),
    )
    parser.add_argument(
        "--candidate-h5",
        type=Path,
        help=(
            "Certified candidate H5 used only with --print-release-id. The same "
            "candidate must be passed to the release-candidate staging build."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help=(
            "Directory containing one *_staged.h5 and its .terminal_gates.json, "
            ".build.json, .hmrc_income.json, and .hmrc_replay.json sidecars."
        ),
    )
    parser.add_argument(
        "--calibration-diagnostics",
        type=Path,
        help="Exact calibration_diagnostics.json whose SHA is signed by the run.",
    )
    parser.add_argument(
        "--releases-root",
        type=Path,
        help="Output releases/ directory; the release-id directory is created here.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Root-artifact output directory. Defaults to a sibling artifacts/ "
            "directory beside --releases-root."
        ),
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--tier",
        required=True,
        choices=sorted(UK_RELEASE_TIERS),
        help="Licensed UK spine tier encoded in the canonical release id.",
    )
    parser.add_argument(
        "--built-at",
        help=(
            "UTC build timestamp for a legacy staging receipt that did not record "
            "one. Refused when it disagrees with recorded provenance."
        ),
    )
    parser.add_argument(
        "--git-commit",
        help=(
            "Full staging-builder commit for a legacy receipt that did not record "
            "one. Refused when it disagrees with recorded provenance."
        ),
    )
    args = parser.parse_args(argv)
    if args.print_release_id:
        if args.candidate_h5 is None:
            parser.error("--print-release-id requires --candidate-h5.")
        return args
    if args.candidate_h5 is not None:
        parser.error("--candidate-h5 is only valid with --print-release-id.")
    missing = [
        option
        for option, value in (
            ("--staging-dir", args.staging_dir),
            ("--calibration-diagnostics", args.calibration_diagnostics),
            ("--releases-root", args.releases_root),
        )
        if value is None
    ]
    if missing:
        parser.error("release assembly requires " + ", ".join(missing) + ".")
    return args


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_bad_json)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {path}: {error}.") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}.")
    return payload


def _bad_json(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _discover_staging_paths(staging_dir: Path) -> dict[str, Path]:
    root = staging_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UK staging output directory is missing: {root}.")
    staged = sorted(root.glob("*_staged.h5"))
    if len(staged) != 1:
        raise ValueError(
            f"UK staging directory {root} must contain exactly one *_staged.h5; "
            f"found {[path.name for path in staged]}."
        )
    h5 = staged[0]
    return {
        "staging_h5": h5,
        "terminal_gates": h5.with_suffix(".terminal_gates.json"),
        "build_record": h5.with_suffix(".build.json"),
        "hmrc_evidence": h5.with_suffix(".hmrc_income.json"),
        "hmrc_replay": h5.with_suffix(".hmrc_replay.json"),
        "run_context": root / "checkpoints" / "stage_run_context.json",
    }


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return value


def _h5_household_count(path: Path, *, label: str) -> int:
    """Read one PolicyEngine H5 household-table cardinality cheaply."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - the UK extra carries h5py.
        raise RuntimeError(
            "Checking UK release cardinality requires h5py; install the UK build extra."
        ) from error

    try:
        with h5py.File(path, "r") as h5:
            if UK_HOUSEHOLD_TABLE_PATH not in h5:
                raise ValueError(
                    f"{label} {path} has no /{UK_HOUSEHOLD_TABLE_PATH} dataset."
                )
            shape = h5[UK_HOUSEHOLD_TABLE_PATH].shape
    except OSError as error:
        raise ValueError(f"{label} is not a readable HDF5 file: {path}.") from error
    if not shape:
        raise ValueError(f"{label} /{UK_HOUSEHOLD_TABLE_PATH} has no row axis.")
    return _positive_int(
        int(shape[0]),
        label=f"{label} /{UK_HOUSEHOLD_TABLE_PATH} row count",
    )


def _candidate_household_count(candidate_h5: Path) -> int:
    """Read the certified candidate's household-table cardinality cheaply."""

    return _h5_household_count(candidate_h5, label="Certified candidate")


def mint_uk_release_id(*, candidate_h5: Path, year: int, tier: str) -> str:
    """Verify a certified candidate and mint its immutable pre-build id.

    Full-national UK staging preserves household cardinality.  The staging
    assembler nevertheless derives the count again from the completed build
    record, so any future cardinality-changing stage fails closed instead of
    silently relabelling a report that already signed this id.
    """

    candidate = verify_certified_uk_candidate(candidate_h5)
    if candidate.tier != tier:
        raise ValueError(
            f"Certified candidate tier is {candidate.tier!r}, not requested "
            f"tier {tier!r}."
        )
    before = candidate.path.stat()
    record_count = _candidate_household_count(candidate.path)
    after = candidate.path.stat()
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_fingerprint != after_fingerprint:
        raise RuntimeError(
            "Certified candidate changed while its household count was read."
        )
    return UKReleaseIdentity(
        year=year,
        tier=tier,
        record_count=record_count,
    ).release_id


def _record_count(build_record: Mapping[str, object]) -> int:
    dataset = _mapping(build_record.get("dataset"), label="staging build dataset")
    entity_rows = _mapping(
        dataset.get("entity_rows"), label="staging build dataset.entity_rows"
    )
    return _positive_int(
        entity_rows.get("household"),
        label="staging build dataset.entity_rows.household",
    )


def _check_record_year(build_record: Mapping[str, object], *, year: int) -> None:
    dataset = _mapping(build_record.get("dataset"), label="staging build dataset")
    recorded = dataset.get("time_period")
    if isinstance(recorded, bool):
        recorded_year = None
    elif isinstance(recorded, int):
        recorded_year = recorded
    elif isinstance(recorded, str) and recorded.isdigit():
        recorded_year = int(recorded)
    else:
        recorded_year = None
    if recorded_year != year:
        raise ValueError(
            "Staging build dataset.time_period must match the requested release "
            f"year {year}, got {recorded!r}."
        )


def _check_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256.")
    return value


def _check_staging_artifact(
    build_record: Mapping[str, object],
    *,
    role: str,
    path: Path,
) -> tuple[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed UK staging run is missing {role}: {path}.")
    artifacts = _mapping(build_record.get("artifacts"), label="staging build artifacts")
    entry = _mapping(artifacts.get(role), label=f"staging build artifacts.{role}")
    expected_sha = _check_sha(
        entry.get("sha256"), label=f"staging build artifacts.{role}.sha256"
    )
    expected_size = entry.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError(
            f"staging build artifacts.{role}.size_bytes must be a non-negative "
            f"integer, got {expected_size!r}."
        )
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ValueError(
            f"Staging artifact {role!r} size mismatch: observed {observed_size}, "
            f"build record declares {expected_size}."
        )
    observed_sha = _sha256(path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"Staging artifact {role!r} SHA-256 mismatch: observed {observed_sha}, "
            f"build record declares {expected_sha}."
        )
    return expected_sha, expected_size


def _check_completed_staging_record(build_record: Mapping[str, object]) -> None:
    if build_record.get("schema_version") != UK_STAGING_BUILD_SCHEMA_VERSION:
        raise ValueError(
            "UK staging build record schema_version must be "
            f"{UK_STAGING_BUILD_SCHEMA_VERSION}, got "
            f"{build_record.get('schema_version')!r}."
        )
    if build_record.get("build_kind") != "uk_national_staging_dataset":
        raise ValueError(
            "UK staging build record build_kind must be 'uk_national_staging_dataset'."
        )
    if build_record.get("status") != "passed":
        raise ValueError("UK staging build record status must be 'passed'.")
    parameters = _mapping(
        build_record.get("parameters"), label="staging build parameters"
    )
    if parameters.get("sample_fraction") != 1.0:
        raise ValueError(
            "UK releases require a full national staging run: "
            "parameters.sample_fraction must be 1.0."
        )
    if parameters.get("rung_token") != "f100":
        raise ValueError(
            "UK releases require the f100 staging rung, got "
            f"{parameters.get('rung_token')!r}."
        )


def _check_run_context(
    context: Mapping[str, object],
    *,
    release_id: str,
    diagnostics_sha256: str,
    build_record: Mapping[str, object],
) -> Mapping[str, object]:
    if context.get("artifact_kind") != UK_STAGE_CONTEXT_ARTIFACT_KIND:
        raise ValueError(
            "Checkpoint stage_run_context.json has the wrong artifact_kind."
        )
    if context.get("schema_version") != UK_STAGE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint stage_run_context.json schema_version must be "
            f"{UK_STAGE_CONTEXT_SCHEMA_VERSION}."
        )
    pipeline = _mapping(context.get("pipeline"), label="checkpoint pipeline")
    if context.get("pipeline_sha256") != _canonical_sha256(pipeline):
        raise ValueError(
            "Checkpoint stage_run_context.json pipeline_sha256 is invalid."
        )
    raw_pipeline_stages = _list(
        pipeline.get("stages"), label="checkpoint pipeline.stages"
    )
    pipeline_names: list[str] = []
    for index, entry in enumerate(raw_pipeline_stages):
        stage = _mapping(entry, label=f"checkpoint pipeline.stages[{index}]")
        name = stage.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"checkpoint pipeline.stages[{index}].name must be non-empty."
            )
        pipeline_names.append(name)
    completed = _list(context.get("completed"), label="checkpoint completed")
    if completed != pipeline_names:
        raise ValueError(
            "Checkpoint stage_run_context.json is incomplete: completed must "
            "equal the full pipeline stage order."
        )
    record_stages = _list(build_record.get("stages"), label="staging build stages")
    if record_stages != completed:
        raise ValueError(
            "Staging build record stages do not match the completed checkpoint "
            "pipeline."
        )
    stage_records = _list(
        context.get("stage_records"), label="checkpoint stage_records"
    )
    if len(stage_records) != len(completed):
        raise ValueError(
            "Checkpoint stage_run_context.json must carry one stage record per "
            "completed stage."
        )
    for index, (raw_record, stage_name) in enumerate(
        zip(stage_records, completed, strict=True)
    ):
        record = _mapping(raw_record, label=f"checkpoint stage_records[{index}]")
        if record.get("stage") != stage_name:
            raise ValueError(
                f"checkpoint stage_records[{index}].stage does not match completed."
            )

    run_config = _mapping(context.get("run_config"), label="checkpoint run_config")
    if run_config.get("build_kind") != "uk_national_staging_dataset":
        raise ValueError("Checkpoint run_config has the wrong build_kind.")
    if run_config.get("release_id") != release_id:
        raise ValueError(
            "Checkpoint run_config.release_id does not match the canonical "
            f"release id {release_id!r}."
        )
    if run_config.get("calibration_diagnostics_sha256") != diagnostics_sha256:
        raise ValueError(
            "Checkpoint run_config.calibration_diagnostics_sha256 does not "
            "match the supplied diagnostics bytes."
        )
    sampling = _mapping(
        run_config.get("sampling"), label="checkpoint run_config.sampling"
    )
    if sampling.get("sample_fraction") not in ("1.0", 1.0):
        raise ValueError("Checkpoint run_config sampling must be the full 1.0 rung.")
    if sampling.get("rung_token") != "f100":
        raise ValueError("Checkpoint run_config sampling.rung_token must be 'f100'.")
    return run_config


def _check_certified_candidate_binding(
    run_config: Mapping[str, object],
    build_record: Mapping[str, object],
) -> None:
    """Hold the checkpoint candidate pin equal to the recorded staging input."""

    certified = _mapping(
        run_config.get("certified_candidate"),
        label="checkpoint run_config.certified_candidate",
    )
    artifacts = _mapping(build_record.get("artifacts"), label="staging build artifacts")
    input_h5 = _mapping(
        artifacts.get("input_h5"), label="staging build artifacts.input_h5"
    )
    certified_sha = _check_sha(
        certified.get("sha256"),
        label="checkpoint run_config.certified_candidate.sha256",
    )
    input_sha = _check_sha(
        input_h5.get("sha256"), label="staging build artifacts.input_h5.sha256"
    )
    if certified_sha != input_sha:
        raise ValueError(
            "Checkpoint certified-candidate SHA-256 does not match the staging "
            "build record input_h5 SHA-256."
        )
    certified_size = certified.get("size_bytes")
    input_size = input_h5.get("size_bytes")
    for value, label in (
        (certified_size, "checkpoint run_config.certified_candidate.size_bytes"),
        (input_size, "staging build artifacts.input_h5.size_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    if certified_size != input_size:
        raise ValueError(
            "Checkpoint certified-candidate size does not match the staging "
            "build record input_h5 size."
        )


def _check_terminal_report(
    report: Mapping[str, object],
    *,
    release_id: str,
    diagnostics_sha256: str,
    build_record: Mapping[str, object],
) -> Mapping[str, object]:
    if report.get("schema_version") != UK_GATE_BATTERY_SCHEMA_VERSION:
        raise ValueError(
            "UK release assembly requires the signed schema-4 gate-battery "
            f"report, got schema_version {report.get('schema_version')!r}."
        )
    if report.get("shippable") is not True:
        raise ValueError(
            "Terminal report shippable must be true; refusing a non-shippable "
            "staging run."
        )
    if report.get("release_id") != release_id:
        raise ValueError(
            f"Terminal report release_id must be {release_id!r}, got "
            f"{report.get('release_id')!r}."
        )
    if report.get("release_candidate") is not True:
        raise ValueError(
            "Terminal report release_candidate must be true; development runs "
            "cannot be assembled into releases."
        )
    if report.get("blocked_at_phase") is not None:
        raise ValueError("Terminal report is blocked and cannot be released.")
    attestation = _mapping(report.get("attestation"), label="terminal attestation")
    signature = attestation.get("signature")
    if not isinstance(signature, str) or not signature:
        raise ValueError(
            "Terminal report is unsigned: attestation.signature must be non-empty."
        )
    if "signing_error" in attestation:
        raise ValueError(
            "Terminal report records a signing_error and is not release-authentic."
        )
    release_evidence = _mapping(
        report.get("release_evidence"), label="terminal report release_evidence"
    )
    bound_sha = release_evidence.get("calibration_diagnostics_sha256")
    if bound_sha != diagnostics_sha256:
        raise ValueError(
            "Calibration diagnostics SHA-256 mismatch: terminal report binds "
            f"{bound_sha!r}, supplied bytes hash to {diagnostics_sha256}."
        )
    if build_record.get("calibration_diagnostics_sha256") != diagnostics_sha256:
        raise ValueError(
            "Calibration diagnostics SHA-256 mismatch: staging build record does "
            "not bind the supplied bytes."
        )
    embedded = build_record.get("terminal_gates")
    if embedded != report:
        raise ValueError(
            "Staging build record terminal_gates does not exactly match the "
            "terminal report sidecar."
        )
    return _mapping(report.get("evidence_sha256"), label="terminal evidence_sha256")


def _recorded_value(
    candidates: Sequence[Mapping[str, object]],
    field: str,
) -> object | None:
    values = [candidate[field] for candidate in candidates if field in candidate]
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Recorded staging provenance disagrees on {field!r}.")
    return values[0]


def _runtime_provenance(
    build_record: Mapping[str, object], run_config: Mapping[str, object]
) -> dict[str, str]:
    candidates: list[Mapping[str, object]] = []
    for owner, value in (
        ("staging build runtime", build_record.get("runtime")),
        ("checkpoint run_config.runtime", run_config.get("runtime")),
    ):
        if value is not None:
            candidates.append(_mapping(value, label=owner))
    code_identity = _mapping(
        run_config.get("code_identity"), label="checkpoint run_config.code_identity"
    )
    dependencies = code_identity.get("dependency_versions")
    if dependencies is not None:
        candidates.append(
            _mapping(
                dependencies,
                label="checkpoint run_config.code_identity.dependency_versions",
            )
        )
    python = _recorded_value(candidates, "python")
    if python is None:
        python = code_identity.get("python")
    if not isinstance(python, str) or not python.strip():
        raise ValueError("Staging build record/run context records no Python version.")
    python_version = python.split()[0]

    runtime = {"python": python_version}
    for package in ("policyengine-core", "policyengine-uk"):
        version = _recorded_value(candidates, package)
        if version is None:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError(
                    "Staging build record/run context records no resolved "
                    f"{package} version and the assembler environment does not "
                    "provide one."
                ) from error
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Resolved {package} version must be non-empty.")
        try:
            Version(version)
        except InvalidVersion as error:
            raise ValueError(
                f"Recorded {package} version {version!r} is not valid PEP 440."
            ) from error
        runtime[package] = version
    data_version = _recorded_value(candidates, "microcosm-data")
    runtime["microcosm-data"] = (
        data_version
        if isinstance(data_version, str) and data_version.strip()
        else importlib.metadata.version("microcosm-data")
    )
    return runtime


def _normalize_built_at(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp.")
    rendered = value.strip()
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO-8601 timestamp: {value!r}.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must carry an explicit UTC offset.")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _build_timestamp(
    build_record: Mapping[str, object],
    run_config: Mapping[str, object],
    explicit: str | None,
) -> str:
    candidates = [
        value
        for value in (
            build_record.get("built_at"),
            build_record.get("created_at"),
            run_config.get("built_at"),
            run_config.get("created_at"),
        )
        if value is not None
    ]
    normalized = [
        _normalize_built_at(value, label="recorded built_at") for value in candidates
    ]
    if normalized and any(value != normalized[0] for value in normalized[1:]):
        raise ValueError("Staging build record/run context disagrees on built_at.")
    explicit_value = (
        None if explicit is None else _normalize_built_at(explicit, label="--built-at")
    )
    if normalized:
        if explicit_value is not None and explicit_value != normalized[0]:
            raise ValueError("--built-at disagrees with recorded staging provenance.")
        return normalized[0]
    if explicit_value is None:
        raise ValueError(
            "Staging provenance records no built_at timestamp; pass --built-at "
            "with the completed staging run's UTC build time."
        )
    return explicit_value


def _code_provenance(
    build_record: Mapping[str, object],
    run_config: Mapping[str, object],
    explicit_commit: str | None,
) -> dict[str, object]:
    raw_candidates = [build_record.get("code"), run_config.get("code")]
    candidates = [
        _mapping(value, label="recorded staging code")
        for value in raw_candidates
        if value is not None
    ]
    code_identity = _mapping(
        run_config.get("code_identity"), label="checkpoint run_config.code_identity"
    )
    identity_code = {
        field: code_identity[field]
        for field in ("git_commit", "git_dirty", "repository")
        if field in code_identity
    }
    if identity_code:
        candidates.append(identity_code)
    recorded_commit = _recorded_value(candidates, "git_commit")
    if (
        explicit_commit is not None
        and _GIT_COMMIT_RE.fullmatch(explicit_commit) is None
    ):
        raise ValueError("--git-commit must be a full 40-character lowercase commit.")
    if recorded_commit is not None:
        if (
            not isinstance(recorded_commit, str)
            or _GIT_COMMIT_RE.fullmatch(recorded_commit) is None
        ):
            raise ValueError("Recorded staging git_commit is malformed.")
        if explicit_commit is not None and explicit_commit != recorded_commit:
            raise ValueError("--git-commit disagrees with recorded staging provenance.")
        commit = recorded_commit
    elif explicit_commit is not None:
        commit = explicit_commit
    else:
        raise ValueError(
            "Staging provenance records no git_commit; pass --git-commit with "
            "the exact staging-builder commit."
        )
    dirty = _recorded_value(candidates, "git_dirty")
    if dirty not in (None, False):
        raise ValueError("Refusing a staging run recorded from a dirty worktree.")
    repository = _recorded_value(candidates, "repository")
    if repository is None:
        repository = "PolicyEngine/microcosm"
    if not isinstance(repository, str) or not repository:
        raise ValueError("Recorded staging code.repository must be non-empty.")
    result: dict[str, object] = {
        "repository": repository,
        "git_commit": commit,
        "git_dirty": False,
    }
    source_sha = code_identity.get("source_sha256")
    if source_sha is not None:
        result["source_sha256"] = _check_sha(
            source_sha,
            label="checkpoint run_config.code_identity.source_sha256",
        )
    return result


def _strict_json_text(payload: object) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_strict_json_text(payload), encoding="utf-8")


def _write_calibration_receipt(path: Path, diagnostics: Mapping[str, object]) -> None:
    uk = _mapping(
        diagnostics.get("uk_diagnostics"),
        label="calibration diagnostics uk_diagnostics",
    )
    weights = _mapping(
        uk.get("weights"), label="calibration diagnostics uk_diagnostics.weights"
    )
    target_surface = _mapping(
        diagnostics.get("target_surface"),
        label="calibration diagnostics target_surface",
    )
    target_registry = _mapping(
        diagnostics.get("target_registry"),
        label="calibration diagnostics target_registry",
    )
    targets = _list(diagnostics.get("targets"), label="calibration diagnostics targets")
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(UK_CALIBRATION_RECEIPT_SCHEMA_VERSION),
            weight_entity=np.asarray(
                str(diagnostics.get("weight_entity", "household"))
            ),
            weight_summary_json=np.asarray(
                _canonical_json_bytes(dict(weights)).decode("utf-8")
            ),
            target_surface_sha256=np.asarray(str(target_surface.get("sha256", ""))),
            target_registry_version=np.asarray(str(target_registry.get("version", ""))),
            target_diagnostics_json=np.asarray(
                _canonical_json_bytes(targets).decode("utf-8")
            ),
        )


def _artifact_entry(
    *,
    path: str,
    sha256: str,
    size_bytes: int,
    kind: str,
    revision: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "path": path,
        "repo_id": UK_RELEASE_REPO_ID,
        "revision": revision,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _release_manifests(
    *,
    identity: UKReleaseIdentity,
    built_at: str,
    runtime: Mapping[str, str],
    code: Mapping[str, object],
    diagnostics: Mapping[str, object],
    evidence: Mapping[str, object],
    dataset_sha256: str,
    dataset_size: int,
    calibration_sha256: str,
    calibration_size: int,
    diagnostics_sha256: str,
    diagnostics_size: int,
    terminal_sha256: str,
    terminal_size: int,
    build_record_sha256: str,
    run_context_sha256: str,
    hmrc_evidence_sha256: str,
    hmrc_replay_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    release_id = identity.release_id
    dataset_key = f"{UK_DATASET_KEY_PREFIX}{identity.year}"
    dataset_filename = f"{dataset_key}.h5"
    calibration_key = f"{dataset_key}_calibration"
    calibration_filename = f"{calibration_key}.npz"
    target_surface = _mapping(
        diagnostics.get("target_surface"),
        label="calibration diagnostics target_surface",
    )
    target_registry = _mapping(
        diagnostics.get("target_registry"),
        label="calibration diagnostics target_registry",
    )

    build_manifest: dict[str, object] = {
        "build_id": release_id,
        "builder": "microcosm",
        "build_sha": str(code["git_commit"])[:7],
        "built_at": built_at,
        "created_at": built_at,
        "code": dict(code),
        "runtime": dict(runtime),
        "country": "uk",
        "year": identity.year,
        "n_records": identity.record_count,
        "dataset": {
            "filename": dataset_filename,
            "sha256": dataset_sha256,
            "size_bytes": dataset_size,
        },
        "calibration": {
            "filename": calibration_filename,
            "sha256": calibration_sha256,
            "size_bytes": calibration_size,
            "target_surface": {
                "sha256": target_surface.get("sha256"),
                "n_targets": target_surface.get("n_targets"),
            },
            "target_registry": {
                "version": target_registry.get("version"),
                "n_specs": target_registry.get("n_specs"),
            },
        },
        "terminal_gate_evidence": dict(evidence),
        "gates": {
            "uk_terminal": {
                "passed": True,
                "path": UK_TERMINAL_REPORT,
                "sha256": terminal_sha256,
            }
        },
        "staging_provenance": {
            "build_record_sha256": build_record_sha256,
            "run_context_sha256": run_context_sha256,
            "hmrc_evidence_sha256": hmrc_evidence_sha256,
            "hmrc_replay_sha256": hmrc_replay_sha256,
        },
    }

    core_version = runtime["policyengine-core"]
    model_version = runtime["policyengine-uk"]
    release_manifest: dict[str, object] = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "country": "uk",
        "year": identity.year,
        "tier": identity.tier,
        "record_count": identity.record_count,
        "n_records": identity.record_count,
        "data_package": {
            "name": "microcosm-data",
            "version": runtime["microcosm-data"],
        },
        "default_datasets": {"national": dataset_key},
        "build": {
            "build_id": release_id,
            "built_at": built_at,
            "builder": "microcosm",
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": core_version,
            },
            "built_with_model_package": {
                "name": "policyengine-uk",
                "version": model_version,
            },
        },
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": f"=={core_version}"}
        ],
        "compatible_model_packages": [
            {"name": "policyengine-uk", "specifier": f"=={model_version}"}
        ],
        "metadata": {
            "region_datasets": {
                "national": {"path_template": dataset_filename},
            }
        },
        "artifacts": {
            dataset_key: _artifact_entry(
                path=dataset_filename,
                sha256=dataset_sha256,
                size_bytes=dataset_size,
                kind="microdata",
                revision=release_id,
            ),
            calibration_key: _artifact_entry(
                path=calibration_filename,
                sha256=calibration_sha256,
                size_bytes=calibration_size,
                kind="calibration",
                revision=release_id,
            ),
            "calibration_diagnostics": _artifact_entry(
                path="calibration_diagnostics.json",
                sha256=diagnostics_sha256,
                size_bytes=diagnostics_size,
                kind="diagnostics",
                revision=release_id,
            ),
            "terminal_gates": _artifact_entry(
                path=UK_TERMINAL_REPORT,
                sha256=terminal_sha256,
                size_bytes=terminal_size,
                kind="diagnostics",
                revision=release_id,
            ),
        },
    }
    return build_manifest, release_manifest


def assemble_uk_release_dir(
    *,
    staging_dir: Path,
    calibration_diagnostics: Path,
    releases_root: Path,
    year: int,
    tier: str,
    artifact_root: Path | None = None,
    built_at: str | None = None,
    git_commit: str | None = None,
) -> Path:
    """Validate one staging run and atomically install its release directory."""

    paths = _discover_staging_paths(Path(staging_dir))
    diagnostics_path = Path(calibration_diagnostics).resolve()
    diagnostics = _load_json(
        diagnostics_path, label="calibration diagnostics supplied for release"
    )
    diagnostics_sha256 = _sha256(diagnostics_path)
    build_record = _load_json(paths["build_record"], label="UK staging build record")
    _check_completed_staging_record(build_record)
    record_count = _record_count(build_record)
    _check_record_year(build_record, year=year)
    identity = UKReleaseIdentity(year=year, tier=tier, record_count=record_count)
    release_id = identity.release_id

    terminal_report = _load_json(
        paths["terminal_gates"], label="UK terminal gate report"
    )
    evidence = _check_terminal_report(
        terminal_report,
        release_id=release_id,
        diagnostics_sha256=diagnostics_sha256,
        build_record=build_record,
    )
    context = _load_json(paths["run_context"], label="UK checkpoint run context")
    run_config = _check_run_context(
        context,
        release_id=release_id,
        diagnostics_sha256=diagnostics_sha256,
        build_record=build_record,
    )
    _check_certified_candidate_binding(run_config, build_record)
    runtime = _runtime_provenance(build_record, run_config)
    code = _code_provenance(build_record, run_config, git_commit)
    resolved_built_at = _build_timestamp(build_record, run_config, built_at)

    staging_bindings = {
        role: _check_staging_artifact(build_record, role=role, path=paths[role])
        for role in (
            "staging_h5",
            "terminal_gates",
            "hmrc_evidence",
            "hmrc_replay",
        )
    }
    staged_count = _h5_household_count(
        paths["staging_h5"], label="Completed UK staging dataset"
    )
    if staged_count != record_count:
        raise ValueError(
            "Completed UK staging dataset household count does not match the "
            f"build record: observed {staged_count}, recorded {record_count}."
        )

    output_releases = Path(releases_root).resolve()
    output_artifacts = (
        output_releases.parent / "artifacts"
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    release_dir = output_releases / release_id
    dataset_filename = f"{UK_DATASET_KEY_PREFIX}{year}.h5"
    calibration_filename = f"{UK_DATASET_KEY_PREFIX}{year}_calibration.npz"
    dataset_output = output_artifacts / dataset_filename
    calibration_output = output_artifacts / calibration_filename
    occupied = [
        path
        for path in (release_dir, dataset_output, calibration_output)
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite an assembled UK release artifact: "
            + ", ".join(map(str, occupied))
            + "."
        )

    output_releases.mkdir(parents=True, exist_ok=True)
    output_artifacts.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".assemble-{release_id}-", dir=output_releases)
    )
    temporary_release = temporary_parent / release_id
    temporary_release.mkdir()
    temporary_dataset = output_artifacts / f".{dataset_filename}.{uuid.uuid4().hex}.tmp"
    temporary_calibration = (
        output_artifacts / f".{calibration_filename}.{uuid.uuid4().hex}.tmp"
    )
    installed_artifacts: list[Path] = []
    try:
        shutil.copyfile(paths["staging_h5"], temporary_dataset)
        _write_calibration_receipt(temporary_calibration, diagnostics)
        shutil.copyfile(
            diagnostics_path, temporary_release / "calibration_diagnostics.json"
        )
        shutil.copyfile(paths["terminal_gates"], temporary_release / UK_TERMINAL_REPORT)

        dataset_sha256 = _sha256(temporary_dataset)
        calibration_sha256 = _sha256(temporary_calibration)
        terminal_sha256 = _sha256(temporary_release / UK_TERMINAL_REPORT)
        copied_staging_binding = (
            dataset_sha256,
            temporary_dataset.stat().st_size,
        )
        if copied_staging_binding != staging_bindings["staging_h5"]:
            raise RuntimeError(
                "UK staging dataset changed between verification and release copy."
            )
        copied_terminal_binding = (
            terminal_sha256,
            (temporary_release / UK_TERMINAL_REPORT).stat().st_size,
        )
        if copied_terminal_binding != staging_bindings["terminal_gates"]:
            raise RuntimeError(
                "UK terminal report changed between verification and release copy."
            )
        build_manifest, release_manifest = _release_manifests(
            identity=identity,
            built_at=resolved_built_at,
            runtime=runtime,
            code=code,
            diagnostics=diagnostics,
            evidence=evidence,
            dataset_sha256=dataset_sha256,
            dataset_size=temporary_dataset.stat().st_size,
            calibration_sha256=calibration_sha256,
            calibration_size=temporary_calibration.stat().st_size,
            diagnostics_sha256=diagnostics_sha256,
            diagnostics_size=diagnostics_path.stat().st_size,
            terminal_sha256=terminal_sha256,
            terminal_size=(temporary_release / UK_TERMINAL_REPORT).stat().st_size,
            build_record_sha256=_sha256(paths["build_record"]),
            run_context_sha256=_sha256(paths["run_context"]),
            hmrc_evidence_sha256=staging_bindings["hmrc_evidence"][0],
            hmrc_replay_sha256=staging_bindings["hmrc_replay"][0],
        )
        _write_json(temporary_release / "build_manifest.json", build_manifest)
        _write_json(temporary_release / "release_manifest.json", release_manifest)

        # This is the acceptance oracle, including HMAC verification through
        # MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY.  No output is installed if
        # the out-of-band trust root is absent or the report is forged.
        validate_release_dir(temporary_release)

        for temporary, final in (
            (temporary_dataset, dataset_output),
            (temporary_calibration, calibration_output),
        ):
            os.link(temporary, final)
            installed_artifacts.append(final)
            temporary.unlink()
        temporary_release.rename(release_dir)
        temporary_parent.rmdir()
    except Exception:
        for path in reversed(installed_artifacts):
            path.unlink(missing_ok=True)
        shutil.rmtree(temporary_parent, ignore_errors=True)
        temporary_dataset.unlink(missing_ok=True)
        temporary_calibration.unlink(missing_ok=True)
        raise
    return release_dir


def main(argv: Sequence[str] | None = None) -> Path | str:
    args = _parse_args(argv)
    if args.print_release_id:
        release_id = mint_uk_release_id(
            candidate_h5=args.candidate_h5,
            year=args.year,
            tier=args.tier,
        )
        print(release_id)
        return release_id
    release_dir = assemble_uk_release_dir(
        staging_dir=args.staging_dir,
        calibration_diagnostics=args.calibration_diagnostics,
        releases_root=args.releases_root,
        year=args.year,
        tier=args.tier,
        artifact_root=args.artifact_root,
        built_at=args.built_at,
        git_commit=args.git_commit,
    )
    artifact_root = (
        args.releases_root.resolve().parent / "artifacts"
        if args.artifact_root is None
        else args.artifact_root.resolve()
    )
    result = {
        "release_id": release_dir.name,
        "release_dir": str(release_dir),
        "artifact_root": str(artifact_root),
        "automatic_publish": False,
        "publish_argv": [
            "tools/publish_release.sh",
            str(release_dir),
            "--repo-id",
            UK_RELEASE_REPO_ID,
            "--artifact-root",
            str(artifact_root),
            "--create-tag",
            "--no-latest",
            "--tag-only",
        ],
    }
    print(_strict_json_text(result), end="")
    return release_dir


if __name__ == "__main__":
    main()
