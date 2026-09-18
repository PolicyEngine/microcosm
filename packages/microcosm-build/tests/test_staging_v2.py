import argparse
import hashlib
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
    BestEffortUploadSession,
    HuggingFaceDatasetStorage,
    StagingRepositoryConfig,
)
from microcosm.build.staging_v2 import (
    CALIBRATION_PROGRESS_SCHEMA,
    EVENT_SCHEMA,
    PROGRESS_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    StagingContentError,
    StagingContractError,
    StagingReadBackError,
    StagingTelemetryV2,
    disabled_staging_delivery,
    validate_staging_delivery,
    validate_v2_bundle,
    validate_v2_document,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "staging"


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


def _recorder(tmp_path, **kwargs) -> StagingTelemetryV2:
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
    return StagingTelemetryV2(**defaults)


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
    }
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
                "schema_version": 3,
            }
        )


def test_running_document_cannot_report_completed_remote_read_back(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
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
    class FailingApi:
        def upload_file(self, **kwargs):
            raise RuntimeError("401 token=do-not-record")

    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=FailingApi(),
        upload_interval_seconds=0,
    )

    telemetry.stage("input_verification", force_upload=True)

    delivery = telemetry.delivery_summary
    assert delivery["configured_repository"] == "policyengine/populace-uk-staging"
    assert delivery["upload_attempts"] == 3
    assert delivery["upload_successes"] == 0
    assert delivery["last_error_code"] == "UPLOAD_FAILED"
    assert telemetry.validate_local_bundle()["progress"]["status"] == "running"
    assert "do-not-record" not in capsys.readouterr().err


def test_remote_read_back_validates_the_written_run(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    telemetry.set_sample(_sample())
    telemetry.complete()

    telemetry.verify_remote()

    assert isinstance(telemetry._upload_session, BestEffortUploadSession)
    assert telemetry.delivery_summary["read_back"] == "passed"
    assert telemetry.uploads_succeeded > 0
    assert "runs/uk-smoke-5-42/run_manifest.json" in api.files
    assert all(path.startswith("runs/uk-smoke-5-42/") for path in api.files)
    remote_manifest = json.loads(api.files["runs/uk-smoke-5-42/run_manifest.json"])
    assert remote_manifest["delivery"]["read_back"] == "passed"


def test_remote_read_back_requires_a_final_run_status(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )

    with pytest.raises(StagingContractError, match="completed or failed"):
        telemetry.verify_remote()

    assert telemetry.delivery_summary["read_back"] == "not_requested"
    assert api.files == {}
    assert api.downloaded == []


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
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    telemetry.complete()
    download = telemetry._transport.download

    def altered_download(path_in_repo, **kwargs):
        data = download(path_in_repo, **kwargs)
        if path_in_repo == path:
            document = json.loads(data)
            mutate(document)
            return json.dumps(document).encode()
        return data

    telemetry._transport.download = altered_download

    with pytest.raises(StagingReadBackError, match="does not match local"):
        telemetry.verify_remote()

    assert telemetry.delivery_summary["read_back"] == "failed"


def test_remote_producers_have_disjoint_run_scoped_upload_paths(tmp_path):
    api = MemoryApi(tmp_path)
    first = _recorder(
        tmp_path / "first",
        run_id="first",
        candidate_id="first",
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    second = _recorder(
        tmp_path / "second",
        run_id="second",
        candidate_id="second",
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )

    first_paths = {remote for _, remote in first._upload_paths()}
    second_paths = {remote for _, remote in second._upload_paths()}

    assert first_paths.isdisjoint(second_paths)
    assert all(path.startswith("runs/first/") for path in first_paths)
    assert all(path.startswith("runs/second/") for path in second_paths)


def test_remote_delivery_matches_local_delivery_after_completion(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )

    telemetry.complete()

    expected = telemetry.delivery_summary
    assert expected["upload_attempts"] == len(telemetry._upload_paths())
    assert expected["upload_successes"] == expected["upload_attempts"]
    for filename in ("run_manifest.json", "progress.json"):
        remote = json.loads(api.files[f"runs/uk-smoke-5-42/{filename}"])
        assert remote["delivery"] == expected


def test_remote_delivery_matches_local_delivery_after_read_back(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    telemetry.complete()

    counted_paths = len(telemetry._upload_paths())
    telemetry.verify_remote()

    expected = telemetry.delivery_summary
    assert expected["upload_attempts"] == 3 * counted_paths
    assert expected["upload_successes"] == expected["upload_attempts"]
    for filename in ("run_manifest.json", "progress.json"):
        remote = json.loads(api.files[f"runs/uk-smoke-5-42/{filename}"])
        assert remote["delivery"] == expected


def test_remote_read_back_failure_is_explicit(tmp_path):
    class UploadOnlyApi(MemoryApi):
        def hf_hub_download(self, **kwargs):
            raise RuntimeError("unavailable")

    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=UploadOnlyApi(tmp_path),
        upload_interval_seconds=0,
    )
    telemetry.complete()

    with pytest.raises(StagingReadBackError, match="read-back failed"):
        telemetry.verify_remote()
    assert telemetry.delivery_summary["read_back"] == "failed"


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


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_contract_fixture_checksums_are_pinned(version):
    fixture_dir = FIXTURE_ROOT / version
    for line in (fixture_dir / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split("  ", 1)
        assert hashlib.sha256(
            (fixture_dir / relative_path).read_bytes()
        ).hexdigest() == (expected)


def test_canonical_version_2_fixtures_are_reproducible():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_staging_contract_fixtures.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_remote_read_back_downloads_every_run_scoped_file(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
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

    telemetry.verify_remote()

    expected = {remote for _, remote in telemetry._upload_paths()}
    assert set(api.downloaded) == expected


@pytest.mark.parametrize(
    "corrupt_path",
    [
        "runs/uk-smoke-5-42/events.ndjson",
        "runs/uk-smoke-5-42/calibration_progress.json",
        "runs/uk-smoke-5-42/artifacts/aggregate-diagnostics.json",
    ],
)
def test_remote_read_back_rejects_changed_declared_file(tmp_path, corrupt_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
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
    download = telemetry._transport.download

    def altered_download(path_in_repo):
        data = download(path_in_repo)
        return b"{}\n" if path_in_repo == corrupt_path else data

    telemetry._transport.download = altered_download

    with pytest.raises(StagingReadBackError, match="does not match local"):
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
    api = MemoryApi(tmp_path, repo_id=repo_id)
    telemetry = _recorder(
        tmp_path,
        country_code="CA",
        delivery_mode="local_and_remote",
        repo_id=repo_id,
        api=api,
        upload_interval_seconds=0,
    )

    telemetry.complete()

    manifest = telemetry.validate_local_bundle()["run_manifest"]
    assert manifest["country_code"] == "CA"
    assert manifest["delivery"]["configured_repository"] == repo_id
    assert api.files


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
    assert not hasattr(args, "staging_prefix")
    with pytest.raises(SystemExit):
        parser.parse_args(["--staging-prefix", "candidate-runs"])


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
