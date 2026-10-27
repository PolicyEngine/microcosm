import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.staging_cli import (
    add_staged_dataset_arguments,
    add_staging_arguments,
    validate_staged_dataset_arguments,
    validate_staging_arguments,
)
from microcosm.build.staging_storage import (
    HuggingFaceDatasetStorage,
    StagingRepositoryConfig,
)
from microcosm.build.staging_v2 import (
    CALIBRATION_PROGRESS_SCHEMA,
    EVENT_SCHEMA,
    PROGRESS_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    STAGING_CONTRACT_VERSION,
    STAGING_DOCUMENT_VERSION,
    StagingContentError,
    StagingContractError,
    StagingReadBackError,
    StagingRunBundleWriterV2,
    disabled_staging_delivery,
    validate_staging_bundle,
    validate_staging_delivery,
    validate_v2_bundle,
    validate_v2_document,
)
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
UK_STAGING = "policyengine/populace-uk-staging"

ROOT = _TEST_PATHS.repository
FIXTURE_ROOT = _TEST_PATHS.tests / "fixtures" / "staging"


class Clock:
    def __init__(self) -> None:
        self.second = 0

    def __call__(self) -> str:
        value = f"2026-01-02T00:00:{self.second:02d}+00:00"
        self.second += 1
        return value


class MemoryApi:
    def __init__(
        self,
        root: Path,
        repo_id: str = "policyengine/populace-uk-staging",
    ) -> None:
        self.root = root
        self.repo_id = repo_id
        self.files: dict[str, bytes] = {}
        self.downloaded: list[str] = []

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        assert repo_id == self.repo_id
        assert repo_type == "dataset"
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def hf_hub_download(self, *, filename, repo_id, repo_type, **kwargs):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        self.downloaded.append(filename)
        destination = self.root / "download" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)


def _sample() -> dict:
    return {"mode": "full"}


def _recorder(tmp_path, **kwargs) -> StagingRunBundleWriterV2:
    defaults = {
        "run_id": "uk-smoke-5-42",
        "country_code": "GB",
        "operation_id": "uk_frs_spine",
        "pipeline_id": "uk_national_spine",
        "pipeline_version": "2026.09",
        "candidate_id": "uk-smoke-5-42",
        "local_dir": tmp_path / "telemetry",
        "release_id": None,
        "run_kind": "smoke",
        "delivery_mode": "local_only",
        "repo_id": None,
        "clock": Clock(),
    }
    defaults.update(kwargs)
    return StagingRunBundleWriterV2(**defaults)


def _remote_recorder(tmp_path, emitter=None, **kwargs):
    """A remote-mode recorder whose run files go to an in-memory emitter."""
    emitter = emitter if emitter is not None else FakeEmitterClient()
    options = {
        "delivery_mode": "local_and_remote",
        "repo_id": UK_STAGING,
        "emitter_client": emitter,
        "upload_interval_seconds": 0,
        **kwargs,
    }
    return _recorder(tmp_path, **options), emitter


def _remote_files(emitter, repository=UK_STAGING) -> dict[str, bytes]:
    return {
        path: content
        for (repo, path), content in emitter.hub_files.items()
        if repo == repository
    }


class _AlteringEmitter(FakeEmitterClient):
    """An emitter whose Hub copy of one path differs once it reads back."""

    def __init__(self, path, alter) -> None:
        super().__init__()
        self.altered_path = path
        self.alter = alter

    def read_back_run_files(self, repository, run_id, *, wait_seconds=30.0):
        key = (repository, self.altered_path)
        self.hub_files[key] = self.alter(self.hub_files[key])
        return super().read_back_run_files(
            repository, run_id, wait_seconds=wait_seconds
        )


def test_staging_run_bundle_writer_sends_no_telemetry_events(tmp_path) -> None:
    """The writer hands over its files; hosted telemetry stays the build's."""
    assert "emitter" not in inspect.signature(StagingRunBundleWriterV2).parameters
    telemetry, emitter = _remote_recorder(tmp_path)
    telemetry.stage("input_verification", force_upload=True)
    telemetry.complete()
    telemetry.verify_remote()

    assert emitter.run_file_requests
    assert set(emitter.calls) <= {
        "stage_run_files",
        "run_files_receipt",
        "read_back_run_files",
    }
    assert emitter.telemetry_messages == []


def test_version_2_local_bundle_has_explicit_schemas_and_ordered_events(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.set_sample(_sample())
    telemetry.stage("input_verification", event_status="completed", files=3)
    telemetry.stage("sampling", event_status="completed", source_families=7)
    telemetry.complete()

    bundle = telemetry.validate_local_bundle()

    assert set(bundle) == {"run_manifest", "progress", "events"}
    assert bundle["run_manifest"]["schema_name"] == RUN_MANIFEST_SCHEMA
    assert bundle["progress"]["schema_name"] == PROGRESS_SCHEMA
    assert {event["schema_name"] for event in bundle["events"]} == {EVENT_SCHEMA}
    assert [event["sequence"] for event in bundle["events"]] == [1, 2, 3, 4]
    assert bundle["progress"]["status"] == "completed"
    assert bundle["run_manifest"]["release_id"] is None
    assert bundle["run_manifest"]["non_release"] is True


def test_calibration_events_are_independently_versioned(tmp_path):
    telemetry = _recorder(tmp_path, run_kind="calibration")
    telemetry.calibration_progress(
        {
            "kind": "calibration_epoch",
            "epoch": 1,
            "epochs": 5,
            "phase": "solve",
            "loss": 1.25,
            "iteration": 4,
        }
    )

    bundle = validate_v2_bundle(telemetry.local_dir, telemetry.run_id)

    assert bundle["calibration_progress"]["schema_name"] == (
        CALIBRATION_PROGRESS_SCHEMA
    )
    assert bundle["calibration_progress"]["events"][0]["loss"] == 1.25
    assert bundle["events"][-1]["event_type"] == "calibration"


def test_typed_artifacts_accept_aggregate_json_and_refuse_population_data(tmp_path):
    telemetry = _recorder(tmp_path)
    diagnostic = tmp_path / "aggregate.json"
    diagnostic.write_text(json.dumps({"target_count": 3, "mean_loss": 0.25}))

    artifact = telemetry.add_artifact(
        "aggregate-diagnostics",
        diagnostic,
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )

    assert artifact["contract_relative_path"] == (
        "artifacts/aggregate-diagnostics.json"
    )
    assert len(artifact["sha256"]) == 64

    population = tmp_path / "population.h5"
    population.write_bytes(b"not really h5")
    with pytest.raises(StagingContentError, match="must be JSON"):
        telemetry.add_artifact(
            "population",
            population,
            artifact_kind="build_metadata",
            classification="non_row_level",
        )

    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"person_id": [1, 2]}))
    with pytest.raises(StagingContentError, match="row-level field"):
        telemetry.add_artifact(
            "rows",
            rows,
            artifact_kind="aggregate_diagnostics",
            classification="aggregate",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"apiToken": "secret-value"},
        {"credentials": {"username": "operator"}},
        {"environment": {"HOME": "/licensed/input"}},
        {"rows": [{"value": 1}]},
        [{"name": "Example person", "age": 42, "income": 50_000}],
        {"data": [{"years_old": 42, "earnings": 50_000}]},
        {"age": [42], "income": [50_000]},
    ],
)
def test_typed_artifacts_reject_sensitive_or_row_level_json(tmp_path, payload):
    telemetry = _recorder(tmp_path)
    artifact = tmp_path / "unsafe.json"
    artifact.write_text(json.dumps(payload))

    with pytest.raises(StagingContentError, match="Prohibited"):
        telemetry.add_artifact(
            "unsafe",
            artifact,
            artifact_kind="aggregate_diagnostics",
            classification="aggregate",
        )


def test_failure_uses_sanitized_contract_fields(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.stage("input_verification")
    telemetry.fail(
        RuntimeError("/licensed/frs/adult.tab token=secret-value household 123"),
        local_diagnostic_reference="diagnostics/local-error.txt",
    )

    progress = telemetry.validate_local_bundle()["progress"]
    serialized = json.dumps(progress)

    assert progress["status"] == "failed"
    assert progress["failure"] == {
        "error_code": "BUILD_FAILED",
        "error_type": "RuntimeError",
        "message": "The build failed during input_verification.",
        "local_diagnostic_reference": "diagnostics/local-error.txt",
        "failure_class": None,
    }
    assert progress["block"] is None
    assert "licensed" not in serialized
    assert "secret-value" not in serialized
    assert "traceback" not in serialized.lower()


def test_delivery_validation_rejects_contradictions():
    local_only = {
        "contract_version": 2,
        "enabled": True,
        "mode": "local_only",
        "run_id": "run-a",
        "configured_repository": None,
        "upload_attempts": 0,
        "upload_successes": 0,
        "read_back": "not_requested",
        "last_error_code": None,
        "opt_out_reason": None,
    }
    assert validate_staging_delivery(local_only) == local_only
    assert disabled_staging_delivery("--no-staging")["enabled"] is False

    with pytest.raises(StagingContractError, match="cannot name a repository"):
        validate_staging_delivery(
            {**local_only, "configured_repository": "policyengine/example"}
        )
    with pytest.raises(StagingContractError, match="cannot exceed"):
        validate_staging_delivery(
            {**local_only, "upload_attempts": 0, "upload_successes": 1}
        )

    invalid_local_states = [
        {"upload_attempts": 1},
        {"read_back": "passed"},
        {"last_error_code": "UPLOAD_FAILED"},
    ]
    for overrides in invalid_local_states:
        with pytest.raises(StagingContractError, match="remote delivery activity"):
            validate_staging_delivery({**local_only, **overrides})

    disabled = disabled_staging_delivery("--no-staging")
    for overrides in (
        {"read_back": "passed"},
        {"last_error_code": "UPLOAD_FAILED"},
        {"opt_out_reason": "   "},
    ):
        with pytest.raises(StagingContractError, match="Disabled staging"):
            validate_staging_delivery({**disabled, **overrides})

    remote = {
        **local_only,
        "mode": "local_and_remote",
        "configured_repository": "policyengine/example",
    }
    with pytest.raises(StagingContractError, match="configured repository"):
        validate_staging_delivery({**remote, "configured_repository": "   "})
    with pytest.raises(StagingContractError, match="Successful read-back"):
        validate_staging_delivery({**remote, "read_back": "passed"})
    with pytest.raises(StagingContractError, match="contradicts"):
        validate_staging_delivery({**remote, "read_back": "failed"})


def test_unknown_schema_version_is_incompatible():
    with pytest.raises(StagingContractError, match="Unsupported staging schema"):
        validate_v2_document(
            {
                "schema_name": RUN_MANIFEST_SCHEMA,
                "schema_version": 999,
            }
        )


def test_running_document_cannot_report_completed_remote_read_back(tmp_path):
    telemetry, _ = _remote_recorder(tmp_path)
    telemetry.complete()
    telemetry.verify_remote()
    manifest = telemetry.validate_local_bundle()["run_manifest"]
    manifest["status"] = "running"

    with pytest.raises(StagingContractError, match="Running staging runs"):
        validate_v2_document(manifest)


@pytest.mark.parametrize("field", ["sample", "delivery", "failure"])
def test_bundle_validation_rejects_shared_document_disagreement(tmp_path, field):
    telemetry = _recorder(tmp_path)
    telemetry.stage("input_verification")
    telemetry.fail(RuntimeError("build failed"))

    progress_path = telemetry.run_dir / "progress.json"
    progress = json.loads(progress_path.read_text())
    if field == "sample":
        progress[field] = _sample()
    elif field == "delivery":
        progress[field]["run_id"] = "different-run"
    else:
        progress[field]["error_code"] = "DIFFERENT_FAILURE"
    progress_path.write_text(json.dumps(progress))

    with pytest.raises(StagingContractError, match=f"Bundle disagrees on {field}"):
        telemetry.validate_local_bundle()


def test_identifiers_and_paths_cannot_escape_contract_root(tmp_path):
    with pytest.raises(StagingContractError, match="run_id"):
        _recorder(tmp_path, run_id="../outside")
    with pytest.raises(StagingContractError, match="Local-only"):
        _recorder(tmp_path, repo_id="policyengine/example")
    with pytest.raises(StagingContractError, match="repository identifier"):
        _recorder(tmp_path, delivery_mode="local_and_remote")


def test_remote_failures_are_best_effort_and_preserve_repository(tmp_path, capsys):
    telemetry, emitter = _remote_recorder(
        tmp_path, FakeEmitterClient(accept_run_files=False)
    )

    telemetry.stage("input_verification", force_upload=True)

    delivery = telemetry.delivery_summary
    assert delivery["configured_repository"] == UK_STAGING
    # The three files the emitter did not accept count as attempts.
    assert len(emitter.run_file_requests[-1].files) == 3
    assert delivery["upload_attempts"] == 3
    assert delivery["upload_successes"] == 0
    assert delivery["last_error_code"] == "UPLOAD_FAILED"
    assert telemetry.validate_local_bundle()["progress"]["status"] == "running"
    assert "staging hand-over failed" in capsys.readouterr().err


def test_a_build_without_an_emitter_records_the_failed_hand_over(tmp_path):
    telemetry, _ = _remote_recorder(tmp_path, emitter_client=None)
    telemetry.complete()

    delivery = telemetry.delivery_summary
    assert delivery["mode"] == "local_and_remote"
    assert delivery["upload_successes"] == 0
    assert delivery["last_error_code"] == "UPLOAD_FAILED"
    assert telemetry.validate_local_bundle()["progress"]["status"] == "completed"


def test_remote_read_back_validates_the_written_run(tmp_path):
    telemetry, emitter = _remote_recorder(tmp_path)
    telemetry.set_sample(_sample())
    telemetry.complete()

    telemetry.verify_remote()

    assert "read_back_run_files" in emitter.calls
    assert {
        (request.repository, request.run_id, request.prefix)
        for request in emitter.run_file_requests
    } == {(UK_STAGING, "uk-smoke-5-42", "runs")}
    assert all(request.latest_pointer is None for request in emitter.run_file_requests)
    assert all(request.run_index_entry is None for request in emitter.run_file_requests)
    assert telemetry.delivery_summary["read_back"] == "passed"
    assert telemetry.uploads_succeeded > 0
    remote = _remote_files(emitter)
    assert "runs/uk-smoke-5-42/run_manifest.json" in remote
    assert all(path.startswith("runs/uk-smoke-5-42/") for path in remote)
    remote_manifest = json.loads(remote["runs/uk-smoke-5-42/run_manifest.json"])
    assert remote_manifest["delivery"]["read_back"] == "passed"


def test_remote_read_back_requires_a_final_run_status(tmp_path):
    telemetry, emitter = _remote_recorder(tmp_path)

    with pytest.raises(StagingContractError, match="completed, blocked or failed"):
        telemetry.verify_remote()

    assert telemetry.delivery_summary["read_back"] == "not_requested"
    assert emitter.hub_files == {}
    assert emitter.calls == []


@pytest.mark.parametrize("final_status", ["completed", "failed"])
def test_finalized_run_rejects_content_changes(tmp_path, final_status):
    telemetry = _recorder(tmp_path)
    artifact = tmp_path / "aggregate.json"
    artifact.write_text(json.dumps({"target_count": 3}))
    if final_status == "completed":
        telemetry.complete()
    else:
        telemetry.fail(RuntimeError("build failed"))
    expected_bundle = telemetry.validate_local_bundle()

    mutations = [
        lambda: telemetry.set_sample(_sample()),
        lambda: telemetry.stage("later_stage"),
        lambda: telemetry.calibration_progress(
            {"kind": "calibration_epoch", "epoch": 1}
        ),
        lambda: telemetry.add_artifact(
            "aggregate-diagnostics",
            artifact,
            artifact_kind="aggregate_diagnostics",
            classification="aggregate",
        ),
        lambda: telemetry.complete(),
        lambda: telemetry.fail(RuntimeError("later failure")),
    ]
    for mutate in mutations:
        with pytest.raises(StagingContractError, match="status"):
            mutate()

    assert telemetry.validate_local_bundle() == expected_bundle
    assert not (telemetry.run_dir / "artifacts").exists()


@pytest.mark.parametrize(
    ("path", "mutate"),
    [
        (
            "runs/uk-smoke-5-42/run_manifest.json",
            lambda document: document.update(candidate_id="altered-candidate"),
        ),
        (
            "runs/uk-smoke-5-42/progress.json",
            lambda document: document.update(message="Altered remote progress."),
        ),
    ],
)
def test_remote_read_back_rejects_schema_valid_content_changes(
    tmp_path,
    path,
    mutate,
):
    def altered(data):
        document = json.loads(data)
        mutate(document)
        return json.dumps(document).encode()

    telemetry, _ = _remote_recorder(tmp_path, _AlteringEmitter(path, altered))
    telemetry.complete()

    with pytest.raises(StagingReadBackError, match=f"differ.*{path}"):
        telemetry.verify_remote()

    assert telemetry.delivery_summary["read_back"] == "failed"
    assert telemetry.delivery_summary["last_error_code"] == "READ_BACK_FAILED"


def test_remote_producers_have_disjoint_run_scoped_upload_paths(tmp_path):
    emitter = FakeEmitterClient()
    first, _ = _remote_recorder(
        tmp_path / "first", emitter, run_id="first", candidate_id="first"
    )
    second, _ = _remote_recorder(
        tmp_path / "second", emitter, run_id="second", candidate_id="second"
    )

    first_paths = {remote for _, remote in first._upload_paths()}
    second_paths = {remote for _, remote in second._upload_paths()}

    assert first_paths.isdisjoint(second_paths)
    assert all(path.startswith("runs/first/") for path in first_paths)
    assert all(path.startswith("runs/second/") for path in second_paths)


def test_remote_delivery_matches_local_delivery_after_completion(tmp_path):
    telemetry, emitter = _remote_recorder(tmp_path)

    telemetry.complete()

    # The documents carry the emitter's counts after the hand-over that
    # delivered them; this emitter commits at once.
    expected = telemetry.delivery_summary
    assert expected["upload_attempts"] == len(telemetry._upload_paths())
    assert expected["upload_successes"] == expected["upload_attempts"]
    remote = _remote_files(emitter)
    for filename in ("run_manifest.json", "progress.json"):
        document = json.loads(remote[f"runs/uk-smoke-5-42/{filename}"])
        assert document["delivery"] == expected


def test_delivery_counts_are_what_the_emitter_reported(tmp_path):
    """Files the emitter has not committed yet do not count as delivered."""

    class NotYetCommitted(FakeEmitterClient):
        def _receipt(self, repository, run_id):
            return {
                **super()._receipt(repository, run_id),
                "pending_files": 3,
                "files_attempted": 0,
                "files_committed": 0,
                "commit_attempts": 0,
                "commits": 0,
            }

    telemetry, emitter = _remote_recorder(tmp_path, NotYetCommitted())
    telemetry.stage("input_verification", force_upload=True)
    telemetry.complete()

    delivery = telemetry.delivery_summary
    assert len(emitter.run_file_requests) >= 2
    assert delivery["upload_attempts"] == delivery["upload_successes"] == 0
    assert delivery["last_error_code"] is None
    telemetry.validate_local_bundle()


def test_remote_delivery_matches_local_delivery_after_read_back(tmp_path):
    telemetry, emitter = _remote_recorder(tmp_path)
    telemetry.complete()

    telemetry.verify_remote()

    expected = telemetry.delivery_summary
    assert expected["read_back"] == "passed"
    assert expected["upload_successes"] == expected["upload_attempts"] > 0
    remote = _remote_files(emitter)
    for filename in ("run_manifest.json", "progress.json"):
        document = json.loads(remote[f"runs/uk-smoke-5-42/{filename}"])
        assert document["delivery"] == expected


@pytest.mark.parametrize(
    "answer",
    [
        {
            "status": "failed",
            "files": 3,
            "mismatched_paths": [],
            "error_code": "REPOSITORY_UNAVAILABLE",
        },
        {"status": "pending", "files": 0, "mismatched_paths": [], "error_code": None},
        {
            "status": "unavailable",
            "files": 0,
            "mismatched_paths": [],
            "error_code": None,
        },
    ],
)
def test_remote_read_back_failure_is_explicit(tmp_path, answer):
    emitter = FakeEmitterClient()
    emitter.read_back_answer = answer
    telemetry, _ = _remote_recorder(tmp_path, emitter)
    telemetry.complete()

    with pytest.raises(StagingReadBackError, match="read-back failed"):
        telemetry.verify_remote()
    assert telemetry.delivery_summary["read_back"] == "failed"
    assert telemetry.delivery_summary["last_error_code"] == "READ_BACK_FAILED"


def test_remote_read_back_without_an_emitter_fails(tmp_path):
    telemetry, _ = _remote_recorder(tmp_path, emitter_client=None)
    telemetry.complete()

    with pytest.raises(StagingReadBackError, match="read-back failed"):
        telemetry.verify_remote()
    assert telemetry.delivery_summary["read_back"] == "failed"


def test_a_failed_refresh_after_a_passed_read_back_keeps_the_verified_delivery(
    tmp_path, capsys
):
    """The read-back verdict owns the error code (microcosm#1301).

    The refresh upload after a passed read-back is best effort. When it
    fails, the delivery stays verified with no error code, the miss shows as
    an attempt without a success, and the remote copy still shows the
    delivery as it was before read-back; the run is not failed.
    """

    class RefreshFails(FakeEmitterClient):
        def read_back_run_files(self, repository, run_id, *, wait_seconds=30.0):
            answer = super().read_back_run_files(
                repository, run_id, wait_seconds=wait_seconds
            )
            self.accept_run_files = False
            return answer

    telemetry, emitter = _remote_recorder(tmp_path, RefreshFails())
    telemetry.set_sample(_sample())
    telemetry.complete()

    telemetry.verify_remote()

    delivery = telemetry.delivery_summary
    assert delivery["read_back"] == "passed"
    assert delivery["last_error_code"] is None
    assert delivery["upload_attempts"] > delivery["upload_successes"] > 0
    bundle = telemetry.validate_local_bundle()
    assert bundle["run_manifest"]["delivery"] == delivery
    remote_manifest = json.loads(
        _remote_files(emitter)["runs/uk-smoke-5-42/run_manifest.json"]
    )
    assert remote_manifest["delivery"]["read_back"] == "not_requested"
    assert "staging hand-over failed" in capsys.readouterr().err


def test_a_failed_read_back_keeps_its_error_code_through_later_uploads(tmp_path):
    emitter = FakeEmitterClient()
    emitter.read_back_answer = {
        "status": "failed",
        "files": 3,
        "mismatched_paths": [],
        "error_code": "REPOSITORY_UNAVAILABLE",
    }
    telemetry, _ = _remote_recorder(tmp_path, emitter)
    telemetry.complete()
    with pytest.raises(StagingReadBackError):
        telemetry.verify_remote()

    # A successful upload afterwards cannot clear the failed verdict's code.
    telemetry._maybe_upload(force=True)

    delivery = telemetry.delivery_summary
    assert delivery["read_back"] == "failed"
    assert delivery["last_error_code"] == "READ_BACK_FAILED"
    assert telemetry.validate_local_bundle()["run_manifest"]["delivery"] == delivery


def test_bundle_validation_checks_reviewed_artifact_digest(tmp_path):
    telemetry = _recorder(tmp_path)
    diagnostic = tmp_path / "aggregate.json"
    diagnostic.write_text(json.dumps({"target_count": 3}))
    artifact = telemetry.add_artifact(
        "aggregate-diagnostics",
        diagnostic,
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    artifact_path = telemetry.run_dir / artifact["contract_relative_path"]
    artifact_path.write_text(json.dumps({"target_count": 4}))

    with pytest.raises(StagingContractError, match="wrong digest"):
        telemetry.validate_local_bundle()


def test_canonical_version_2_fixture_cases_validate():
    v2 = FIXTURE_ROOT / "v2"
    completed = validate_v2_bundle(v2 / "completed-spine", "uk-spine-v2-fixture")
    calibration = validate_v2_bundle(v2 / "calibration", "uk-calibration-v2-fixture")
    failed = validate_v2_bundle(v2 / "failed", "uk-failed-v2-fixture")
    cases = json.loads((v2 / "contract-cases.json").read_text())

    assert completed["progress"]["status"] == "completed"
    assert calibration["calibration_progress"]["events"][0]["phase"] == "solve"
    assert failed["progress"]["failure"]["message"] == (
        "The build failed during input_verification."
    )
    assert validate_staging_delivery(cases["delivery_success"])["upload_successes"] == 6
    assert (
        validate_staging_delivery(cases["delivery_failure"])["last_error_code"]
        == "READ_BACK_FAILED"
    )
    assert validate_staging_delivery(cases["deliberate_opt_out"])["enabled"] is False
    with pytest.raises(StagingContractError, match="Unsupported staging schema"):
        validate_v2_document(cases["unknown_version"])


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_contract_fixture_checksums_are_pinned(version):
    fixture_dir = FIXTURE_ROOT / version
    for line in (fixture_dir / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split("  ", 1)
        assert hashlib.sha256(
            (fixture_dir / relative_path).read_bytes()
        ).hexdigest() == (expected)


def test_canonical_version_3_fixture_cases_validate():
    v3 = FIXTURE_ROOT / "v3"
    completed = validate_staging_bundle(v3 / "completed-spine", "uk-spine-v3-fixture")
    calibration = validate_staging_bundle(
        v3 / "calibration", "uk-calibration-v3-fixture"
    )
    failed = validate_staging_bundle(v3 / "failed", "uk-failed-v3-fixture")
    blocked = validate_staging_bundle(v3 / "blocked", "uk-blocked-v3-fixture")

    assert completed["progress"]["schema_version"] == STAGING_DOCUMENT_VERSION == 3
    assert completed["progress"]["delivery"]["contract_version"] == 2
    assert completed["progress"]["block"] is None
    assert calibration["calibration_progress"]["schema_version"] == 3
    assert failed["progress"]["failure"]["failure_class"] == "error"
    assert failed["events"][-1]["details"]["failure_class"] == "error"
    assert blocked["progress"]["status"] == "blocked"
    assert blocked["progress"]["failure"] is None
    assert blocked["run_manifest"]["block"] == {
        "phase": "terminal",
        "blocking_failure_count": 1,
        "blocking_gate_ids": ["uk_target_fit"],
    }
    terminal = blocked["events"][-1]
    assert (terminal["stage_id"], terminal["status"]) == ("blocked", "blocked")
    assert terminal["details"]["gate_statuses"]["uk_target_fit"] == "failed"


def test_delivery_contract_stays_version_2_while_documents_move_to_3():
    """Publishers and release assemblers pin the delivery summary's version."""

    assert STAGING_CONTRACT_VERSION == 2
    assert disabled_staging_delivery("--no-staging")["contract_version"] == 2


def test_blocked_run_closes_with_its_gate_phase(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.stage("gate_battery", event_status="completed")
    telemetry.block(
        phase="preflight",
        blocking_gate_ids=["uk_target_surface", "uk_support"],
        gate_statuses={"uk_target_surface": "failed", "uk_support": "failed"},
    )

    bundle = telemetry.validate_local_bundle()
    assert bundle["progress"]["status"] == "blocked"
    assert bundle["progress"]["block"]["phase"] == "preflight"
    assert bundle["progress"]["block"]["blocking_failure_count"] == 2
    with pytest.raises(StagingContractError, match="status is 'blocked'"):
        telemetry.complete()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": "terminal", "blocking_gate_ids": []},
        {"phase": "terminal", "blocking_gate_ids": [""]},
        {"phase": "../x", "blocking_gate_ids": ["uk_target_fit"]},
        {"phase": "terminal", "blocking_gate_ids": ["a"], "blocking_failure_count": 0},
        {
            "phase": "terminal",
            "blocking_gate_ids": ["token"],
            "gate_statuses": {"token": "failed"},
        },
    ],
)
def test_block_refuses_an_empty_or_unsafe_block(tmp_path, kwargs):
    telemetry = _recorder(tmp_path)
    with pytest.raises(StagingContractError):
        telemetry.block(**kwargs)
    # Refused before any state changed: the run is still running and can
    # still close another way.
    assert telemetry.status == "running"


def test_a_refused_block_leaves_the_run_able_to_close_failed(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.stage("gate_battery", event_status="completed")
    with pytest.raises(StagingContractError):
        telemetry.block(
            phase="terminal",
            blocking_gate_ids=["token"],
            gate_statuses={"token": "failed"},
        )
    telemetry.fail(
        RuntimeError("the block could not be recorded"),
        error_code="GATE_BLOCK_UNRECORDED",
        failure_class="unrecorded_gate_block",
    )
    progress = telemetry.validate_local_bundle()["progress"]
    assert progress["status"] == "failed" and progress["block"] is None
    assert progress["failure"]["error_code"] == "GATE_BLOCK_UNRECORDED"


def test_a_block_may_name_no_gate_when_it_counts_its_failures(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.block(phase="assembled", blocking_gate_ids=[], blocking_failure_count=3)
    block = telemetry.validate_local_bundle()["progress"]["block"]
    assert block == {
        "phase": "assembled",
        "blocking_failure_count": 3,
        "blocking_gate_ids": [],
    }


def test_failure_class_is_recorded_and_checked(tmp_path):
    telemetry = _recorder(tmp_path)
    with pytest.raises(StagingContractError, match="failure_class"):
        telemetry.fail(RuntimeError("x"), failure_class="Not A Class")
    telemetry.fail(
        KeyboardInterrupt(), error_code="INTERRUPTED", failure_class="interrupted"
    )
    failure = telemetry.validate_local_bundle()["progress"]["failure"]
    assert (failure["error_code"], failure["failure_class"]) == (
        "INTERRUPTED",
        "interrupted",
    )


def test_version_3_documents_keep_failure_and_block_apart(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.block(phase="terminal", blocking_gate_ids=["uk_target_fit"])
    manifest = telemetry.validate_local_bundle()["run_manifest"]

    with pytest.raises(StagingContractError, match="Blocked runs require"):
        validate_v2_document({**manifest, "block": None})
    failed = {
        **manifest,
        "status": "failed",
        "failure": {
            "error_code": "BUILD_FAILED",
            "error_type": "RuntimeError",
            "message": "The build failed during gate_battery.",
            "local_diagnostic_reference": None,
            "failure_class": None,
        },
    }
    with pytest.raises(StagingContractError, match="Non-blocked runs"):
        validate_v2_document(failed)
    with pytest.raises(StagingContractError, match="failure"):
        validate_v2_document(
            {**failed, "block": None, "failure": {**failed["failure"], "extra": 1}}
        )


def test_a_run_cannot_mix_contract_versions(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.complete()
    progress_path = telemetry.run_dir / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["schema_version"] = 2
    progress.pop("block")
    progress_path.write_text(json.dumps(progress))

    with pytest.raises(StagingContractError, match="mix contract versions"):
        telemetry.validate_local_bundle()


def test_canonical_version_3_fixtures_are_reproducible():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_staging_contract_fixtures.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_remote_read_back_covers_every_run_scoped_file(tmp_path):
    telemetry, emitter = _remote_recorder(tmp_path)
    diagnostic = tmp_path / "aggregate.json"
    diagnostic.write_text(json.dumps({"target_count": 3}))
    telemetry.add_artifact(
        "aggregate-diagnostics",
        diagnostic,
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    telemetry.calibration_progress(
        {
            "kind": "calibration_epoch",
            "epoch": 1,
            "epochs": 2,
            "phase": "solve",
            "loss": 1.0,
            "iteration": 1,
        }
    )
    telemetry.complete()

    telemetry.verify_remote()

    # The emitter compares each file it committed for the run with the Hub.
    expected = {remote for _, remote in telemetry._upload_paths()}
    assert set(_remote_files(emitter)) == expected
    assert emitter.committed_paths(UK_STAGING, "uk-smoke-5-42") == expected
    assert telemetry.delivery_summary["read_back"] == "passed"


@pytest.mark.parametrize(
    "corrupt_path",
    [
        "runs/uk-smoke-5-42/events.ndjson",
        "runs/uk-smoke-5-42/calibration_progress.json",
        "runs/uk-smoke-5-42/artifacts/aggregate-diagnostics.json",
    ],
)
def test_remote_read_back_rejects_changed_declared_file(tmp_path, corrupt_path):
    telemetry, _ = _remote_recorder(
        tmp_path, _AlteringEmitter(corrupt_path, lambda data: b"{}\n")
    )
    diagnostic = tmp_path / "aggregate.json"
    diagnostic.write_text(json.dumps({"target_count": 3}))
    telemetry.add_artifact(
        "aggregate-diagnostics",
        diagnostic,
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    telemetry.calibration_progress(
        {
            "kind": "calibration_epoch",
            "epoch": 1,
            "epochs": 2,
            "phase": "solve",
            "loss": 1.0,
            "iteration": 1,
        }
    )
    telemetry.complete()

    with pytest.raises(StagingReadBackError, match=f"differ.*{corrupt_path}"):
        telemetry.verify_remote()

    assert telemetry.delivery_summary["read_back"] == "failed"


def test_version_2_storage_prefix_is_fixed_to_runs(tmp_path):
    telemetry = _recorder(tmp_path)

    assert telemetry.run_dir == tmp_path / "telemetry" / "runs" / telemetry.run_id
    assert telemetry.repo_run_prefix == f"runs/{telemetry.run_id}"
    with pytest.raises(TypeError, match="path_prefix"):
        _recorder(tmp_path, path_prefix="candidate-runs")
    with pytest.raises(TypeError, match="path_prefix"):
        validate_v2_bundle(tmp_path, telemetry.run_id, path_prefix="candidate-runs")


def test_version_2_accepts_explicit_configuration_for_another_country(tmp_path):
    repo_id = "example/populace-ca-staging"
    telemetry, emitter = _remote_recorder(tmp_path, country_code="CA", repo_id=repo_id)

    telemetry.complete()

    manifest = telemetry.validate_local_bundle()["run_manifest"]
    assert manifest["country_code"] == "CA"
    assert manifest["delivery"]["configured_repository"] == repo_id
    assert _remote_files(emitter, repo_id)


def test_version_2_cli_uses_country_owned_repository_configuration(monkeypatch):
    repository = StagingRepositoryConfig(
        default_repo_id="example/default-staging",
        repo_id_environment_variable="EXAMPLE_STAGING_REPO_ID",
    )
    monkeypatch.setenv("EXAMPLE_STAGING_REPO_ID", "example/configured-staging")
    parser = argparse.ArgumentParser()
    add_staging_arguments(parser, repository=repository)

    args = parser.parse_args([])
    assert args.staging_repo_id == "example/configured-staging"
    assert args.staging_upload_interval_seconds == 30.0
    assert not hasattr(args, "staging_prefix")
    with pytest.raises(SystemExit):
        parser.parse_args(["--staging-prefix", "candidate-runs"])


def test_version_2_cli_lets_a_long_running_command_slow_its_upload_cadence():
    """A multi-hour solve at 30 s exhausts the Hub's commit budget (v20 run)."""

    repository = StagingRepositoryConfig(
        default_repo_id="example/default-staging",
        repo_id_environment_variable="EXAMPLE_STAGING_REPO_ID",
    )
    parser = argparse.ArgumentParser()
    add_staging_arguments(
        parser, repository=repository, default_upload_interval_seconds=300
    )
    assert parser.parse_args([]).staging_upload_interval_seconds == 300.0
    assert (
        parser.parse_args(
            ["--staging-upload-interval-seconds", "45"]
        ).staging_upload_interval_seconds
        == 45.0
    )


def test_typed_artifacts_reject_one_scalar_individual_record(tmp_path):
    telemetry = _recorder(tmp_path)
    artifact = tmp_path / "individual.json"
    artifact.write_text(
        json.dumps({"name": "Example person", "age": 42, "income": 50_000})
    )

    with pytest.raises(StagingContentError, match="individual record"):
        telemetry.add_artifact(
            "individual",
            artifact,
            artifact_kind="aggregate_diagnostics",
            classification="aggregate",
        )


def test_staged_dataset_cli_follows_the_staging_mode_switch(monkeypatch):
    telemetry_repository = StagingRepositoryConfig(
        default_repo_id="example/default-staging",
        repo_id_environment_variable="EXAMPLE_STAGING_REPO_ID",
    )
    dataset_repository = StagingRepositoryConfig(
        default_repo_id="example/default-private",
        repo_id_environment_variable="EXAMPLE_STAGED_DATASET_REPO_ID",
    )
    monkeypatch.setenv("EXAMPLE_STAGED_DATASET_REPO_ID", "example/configured-private")

    def parse(argv):
        parser = argparse.ArgumentParser()
        add_staging_arguments(parser, repository=telemetry_repository)
        add_staged_dataset_arguments(parser, repository=dataset_repository)
        args = parser.parse_args(argv)
        validate_staging_arguments(parser, args)
        validate_staged_dataset_arguments(parser, args)
        return args

    args = parse([])
    assert args.staged_dataset_repo_id == "example/configured-private"
    assert args.no_staged_dataset is False
    assert parse(["--no-staged-dataset"]).no_staged_dataset is True
    assert parse(
        ["--staging-local-only", "--staged-dataset-repo-id", ""]
    ).staging_local_only
    with pytest.raises(SystemExit):
        parse(["--no-staging", "--no-staged-dataset"])
    with pytest.raises(SystemExit):
        parse(["--staged-dataset-repo-id", " "])


class CommitApi(MemoryApi):
    """MemoryApi plus the commit surface the staged-dataset lane uses."""

    def __init__(self, root: Path, repo_id: str = "example/private") -> None:
        super().__init__(root, repo_id)
        self.sha = "1" * 40
        self.commits: list[tuple[str, str | None, list[str]]] = []

    def file_exists(self, *, repo_id, filename, repo_type):
        assert repo_id == self.repo_id and repo_type == "dataset"
        return filename in self.files

    def repo_info(self, *, repo_id, repo_type):
        assert repo_id == self.repo_id and repo_type == "dataset"
        return {"sha": self.sha}

    def create_commit(
        self, *, repo_id, operations, commit_message, repo_type, parent_commit
    ):
        assert repo_id == self.repo_id and repo_type == "dataset"
        paths = []
        for operation in operations:
            self.files[operation.path_in_repo] = Path(
                operation.path_or_fileobj
            ).read_bytes()
            paths.append(operation.path_in_repo)
        self.commits.append((commit_message, parent_commit, paths))
        self.sha = "2" * 40
        return {"oid": self.sha}


def test_storage_commits_several_files_at_once_and_reports_the_revision(tmp_path):
    from huggingface_hub import CommitOperationAdd

    api = CommitApi(tmp_path)
    storage = HuggingFaceDatasetStorage("example/private", api=api)
    first = tmp_path / "a.json"
    second = tmp_path / "b.h5"
    first.write_text("{}")
    second.write_bytes(b"binary")

    assert storage.head_revision() == "1" * 40
    assert storage.file_exists("staged/run/a.json") is False
    revision = storage.commit(
        [
            CommitOperationAdd(
                path_in_repo="staged/run/a.json", path_or_fileobj=str(first)
            ),
            CommitOperationAdd(
                path_in_repo="staged/run/b.h5", path_or_fileobj=str(second)
            ),
        ],
        message="Stage run",
        parent_commit="1" * 40,
    )
    assert revision == "2" * 40
    assert api.commits == [
        ("Stage run", "1" * 40, ["staged/run/a.json", "staged/run/b.h5"])
    ]
    assert storage.file_exists("staged/run/b.h5") is True
    assert storage.download_file("staged/run/b.h5").read_bytes() == b"binary"


def test_storage_reports_the_credential_role_when_the_backend_can(tmp_path):
    class RoleApi(CommitApi):
        def whoami(self):
            return {"name": "x", "auth": {"accessToken": {"role": "read"}}}

    assert (
        HuggingFaceDatasetStorage(
            "example/private", api=RoleApi(tmp_path)
        ).credential_role()
        == "read"
    )
    assert (
        HuggingFaceDatasetStorage(
            "example/private", api=CommitApi(tmp_path)
        ).credential_role()
        is None
    )


def test_storage_commit_refuses_a_revisionless_backend(tmp_path):
    class NoRevision(CommitApi):
        def create_commit(self, **kwargs):
            return {"oid": ""}

    storage = HuggingFaceDatasetStorage("example/private", api=NoRevision(tmp_path))
    with pytest.raises(RuntimeError, match="no revision"):
        storage.commit([], message="empty")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ({"role": "read"}, False),
        ({"role": "write"}, True),
        (
            {
                "role": "fineGrained",
                "fineGrained": {"global": ["repo.write"], "scoped": []},
            },
            True,
        ),
        (
            {
                "role": "fineGrained",
                "fineGrained": {
                    "global": [],
                    "scoped": [
                        {
                            "entity": {"type": "org", "name": "example"},
                            "permissions": ["repo.write"],
                        }
                    ],
                },
            },
            True,
        ),
        (
            {
                "role": "fineGrained",
                "fineGrained": {
                    "global": [],
                    "scoped": [
                        {
                            "entity": {"type": "dataset", "name": "example/private"},
                            "permissions": ["repo.content.read", "repo.write"],
                        }
                    ],
                },
            },
            True,
        ),
        (
            {
                "role": "fineGrained",
                "fineGrained": {
                    "global": ["discussion.write"],
                    "scoped": [
                        {
                            "entity": {"type": "user", "name": "someone"},
                            "permissions": ["repo.write"],
                        }
                    ],
                },
            },
            False,
        ),
        (
            {
                "role": "fineGrained",
                "fineGrained": {
                    "global": [],
                    "scoped": [
                        {
                            "entity": {"type": "org", "name": "example"},
                            "permissions": ["repo.content.read"],
                        }
                    ],
                },
            },
            False,
        ),
        ({"role": "mystery"}, None),
    ],
)
def test_storage_reads_whether_the_credential_can_write_this_repository(
    tmp_path, token, expected
):
    class TokenApi(CommitApi):
        def whoami(self):
            return {"name": "x", "auth": {"accessToken": token}}

    storage = HuggingFaceDatasetStorage("example/private", api=TokenApi(tmp_path))
    assert storage.credential_can_write() is expected
    # No whoami on the backend: the scope is unknown, never assumed.
    assert (
        HuggingFaceDatasetStorage(
            "example/private", api=CommitApi(tmp_path)
        ).credential_can_write()
        is None
    )


def _paused_remote_recorder(tmp_path):
    emitter = FakeEmitterClient(accept_run_files=False)
    telemetry, _ = _remote_recorder(tmp_path, emitter)
    telemetry.set_sample(_sample())
    # Three consecutive hand-overs the emitter does not accept pause the
    # run's hand-overs until its final state.
    for stage in ("input_verification", "sampling", "calibrating"):
        telemetry.stage(stage, force_upload=True)
    assert telemetry.delivery_summary["last_error_code"] == "UPLOAD_FAILED"
    requests_paused = len(emitter.run_file_requests)
    telemetry.stage("diagnostics", force_upload=True)
    assert len(emitter.run_file_requests) == requests_paused == 3
    return telemetry, emitter


def test_completion_hands_over_the_final_state_after_paused_hand_overs(tmp_path):
    """The v20 last-mile gap: a paused run left its remote copy 'running'."""

    telemetry, emitter = _paused_remote_recorder(tmp_path)
    emitter.accept_run_files = True
    telemetry.complete()

    remote = _remote_files(emitter)
    prefix = telemetry.repo_run_prefix
    assert json.loads(remote[f"{prefix}/progress.json"])["status"] == "completed"
    assert {f"{prefix}/run_manifest.json", f"{prefix}/events.ndjson"} <= set(remote)
    assert emitter.run_file_requests[-2].flush is True
    delivery = telemetry.delivery_summary
    assert delivery["last_error_code"] is None
    assert delivery["upload_successes"] >= 3
    assert delivery["upload_attempts"] > delivery["upload_successes"]


def test_failure_hands_over_the_final_state_after_paused_hand_overs(tmp_path):
    telemetry, emitter = _paused_remote_recorder(tmp_path)
    emitter.accept_run_files = True
    telemetry.fail(RuntimeError("solver exploded"))
    prefix = telemetry.repo_run_prefix
    remote = _remote_files(emitter)
    assert json.loads(remote[f"{prefix}/progress.json"])["status"] == "failed"
    assert telemetry.delivery_summary["last_error_code"] is None


def test_a_final_state_the_emitter_does_not_accept_stays_local(tmp_path, capsys):
    telemetry, emitter = _paused_remote_recorder(tmp_path)
    telemetry.complete()
    assert len(emitter.run_file_requests) == 4
    assert telemetry.repo_run_prefix + "/progress.json" not in _remote_files(emitter)
    delivery = telemetry.delivery_summary
    assert delivery["last_error_code"] == "UPLOAD_FAILED"
    err = capsys.readouterr().err
    assert "final state was not handed over" in err
    # The local bundle is complete whatever the emitter did.
    assert telemetry.validate_local_bundle()["progress"]["status"] == "completed"
