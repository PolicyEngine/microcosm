"""US builds write their staging run files with the version 2 writer (#1135).

The build and its certification each open a staging run through
``_BuildProgress``; these tests read the run's local files and what the
in-memory emitter committed, and check them against the version 2 contract.
"""

from __future__ import annotations

import importlib.util
import json
from importlib import metadata
from types import SimpleNamespace

import pytest

from microcosm.build.staging_v2 import (
    EVENT_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    STAGING_DOCUMENT_VERSION,
    validate_staging_bundle,
    validate_staging_delivery,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
US_STAGING = "policyengine/populace-us-staging"
BUILD = "rel-1"


def _load_builder_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


def _hosted_telemetry(client=None):
    from microcosm.build.telemetry_emitter import TelemetryRun
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    return FakeTelemetryEmitter(
        TelemetryRun(BUILD, "US", "us_fiscal_refresh"), client=client
    )


def _staging_arguments(builder, *, repository=US_STAGING):
    """The build's staging options with their defaults: no local directory."""
    return SimpleNamespace(
        no_staging=False,
        staging_dir=None,
        staging_repo_id=repository,
        staging_run_id=None,
        staging_prefix="runs",
        epochs=builder.DEFAULT_US_FISCAL_CALIBRATION_EPOCHS,
        dense_default_dataset=False,
    )


def _remote_build_progress(builder, tmp_path, client=None):
    emitter = _hosted_telemetry(client)
    progress = builder._build_progress(
        _staging_arguments(builder),
        release_root=tmp_path,
        release_id=BUILD,
        emitter=emitter,
    )
    return progress, emitter


def _run_dir(tmp_path, run_id=BUILD):
    """Where a run's files are written when no ``--staging-dir`` is given."""
    return tmp_path / "staging" / "runs" / run_id


def _read(run_dir, name):
    return json.loads((run_dir / name).read_text())


def _events(run_dir):
    return [
        json.loads(text)
        for text in (run_dir / "events.ndjson").read_text().splitlines()
        if text
    ]


def _run_events(emitter):
    return [event for event in emitter.events if event["event_type"] == "run"]


def test_a_us_build_writes_version_2_documents_with_its_identity(
    builder, tmp_path
) -> None:
    progress, _ = _remote_build_progress(builder, tmp_path)
    progress.stage("target_registry", message="Compiled targets.", n_targets=3)

    progress.complete()

    run_dir = _run_dir(tmp_path)
    manifest = _read(run_dir, "run_manifest.json")
    assert (manifest.get("schema_name"), manifest.get("schema_version")) == (
        RUN_MANIFEST_SCHEMA,
        STAGING_DOCUMENT_VERSION,
    )
    identity = (
        "run_id",
        "country_code",
        "operation_id",
        "pipeline",
        "candidate_id",
        "release_id",
        "run_kind",
        "non_release",
        "status",
        "current_stage",
        "sample",
    )
    assert {field: manifest.get(field) for field in identity} == {
        "run_id": BUILD,
        "country_code": "US",
        "operation_id": "us_fiscal_refresh_build",
        "pipeline": {
            "id": "us_fiscal_refresh",
            "version": metadata.version("microcosm-build"),
        },
        "candidate_id": BUILD,
        "release_id": BUILD,
        "run_kind": "release",
        "non_release": False,
        "status": "completed",
        "current_stage": "complete",
        "sample": {"mode": "full"},
    }
    documents = validate_staging_bundle(tmp_path / "staging", BUILD)
    delivery = documents["progress"]["delivery"]
    assert (delivery["mode"], delivery["configured_repository"]) == (
        "local_and_remote",
        US_STAGING,
    )
    assert delivery["upload_successes"] >= 1


def test_us_stage_statuses_become_version_2_event_statuses(builder, tmp_path) -> None:
    """A failed check is an event, not the run's end; other outcomes stay in the details."""
    progress, emitter = _remote_build_progress(builder, tmp_path)

    progress.stage("load_base_frame", message="Loading base population H5.")
    progress.stage(
        "qbi_input_gate",
        status="failed",
        message="QBI input signal check failed.",
        failures=["qbi share is zero"],
    )
    progress.stage(
        "base_frame_spm_composition",
        status="skipped",
        message="Base pool carries no readable SPM composition inputs.",
        error="no SPM columns",
    )
    progress.stage(
        "base_frame_spm_composition",
        status="retained",
        message="Classified the base pool's SPM measurement composition.",
    )

    run_dir = _run_dir(tmp_path)
    events = _events(run_dir)
    assert [
        (event.get("schema_name"), event.get("stage_id"), event.get("status"))
        for event in events[1:]
    ] == [
        (EVENT_SCHEMA, "load_base_frame", "started"),
        (EVENT_SCHEMA, "qbi_input_gate", "failed"),
        (EVENT_SCHEMA, "base_frame_spm_composition", "completed"),
        (EVENT_SCHEMA, "base_frame_spm_composition", "completed"),
    ]
    assert events[2]["details"]["failures"] == ["qbi share is zero"]
    assert [event["details"].get("stage_status") for event in events[3:]] == [
        "skipped",
        "retained",
    ]
    assert _read(run_dir, "progress.json").get("status") == "running"
    # Hosted telemetry keeps the statuses the build reported.
    assert [
        event["status"]
        for event in emitter.events
        if event["stage_id"] == "qbi_input_gate"
    ][0] == "failed"


def test_a_failed_us_build_records_a_version_2_failure_without_its_text(
    builder, tmp_path
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    progress.stage("export_dataset", message="Exporting the dataset.")

    progress.fail(RuntimeError("export failed at /restricted/asec.h5 token=abc"))

    run_dir = _run_dir(tmp_path)
    document = _read(run_dir, "progress.json")
    assert document.get("status") == "failed"
    assert document.get("failure") == {
        "error_code": "BUILD_FAILED",
        "error_type": "RuntimeError",
        "message": "The build failed during export_dataset.",
        "local_diagnostic_reference": None,
        "failure_class": "error",
    }
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert "token=abc" not in path.read_text(), path.name
    validate_staging_bundle(tmp_path / "staging", BUILD)
    assert [event["status"] for event in _run_events(emitter)] == ["failed"]


def test_a_terminated_us_build_records_a_termination(builder, tmp_path) -> None:
    from microcosm.build.termination import BuildTerminatedError

    progress, _ = _remote_build_progress(builder, tmp_path)

    progress.fail(BuildTerminatedError())

    failure = _read(_run_dir(tmp_path), "progress.json").get("failure") or {}
    assert (failure.get("error_code"), failure.get("failure_class")) == (
        "TERMINATED",
        "terminated",
    )


def test_a_us_run_writes_no_latest_pointer_or_runs_index(builder, tmp_path) -> None:
    """Version 2 runs are found by their folders, as UK runs are."""
    progress, emitter = _remote_build_progress(builder, tmp_path)

    progress.complete()

    requests = emitter.client.run_file_requests
    assert requests
    assert [
        (request.latest_pointer, request.run_index_entry) for request in requests
    ] == [(None, None)] * len(requests)
    run_dir = _run_dir(tmp_path)
    assert not (run_dir / "latest_staging.json").exists()
    assert not (run_dir / "runs.json").exists()
    assert {"latest_staging.json", "runs.json"}.isdisjoint(
        emitter.client.paths(US_STAGING)
    )


def test_a_hand_over_the_emitter_refuses_is_recorded_and_the_build_goes_on(
    builder, tmp_path
) -> None:
    from test_support.microcosm_build.emitter import FakeEmitterClient

    progress, emitter = _remote_build_progress(
        builder, tmp_path, client=FakeEmitterClient(accept_run_files=False)
    )
    progress.stage("target_registry", message="Compiled targets.")

    progress.complete()

    delivery = _read(_run_dir(tmp_path), "progress.json").get("delivery") or {}
    assert (delivery.get("upload_successes"), delivery.get("last_error_code")) == (
        0,
        "UPLOAD_FAILED",
    )
    assert [event["status"] for event in _run_events(emitter)] == ["completed"]


def test_a_passing_result_file_is_staged_as_a_version_2_artifact(
    builder, tmp_path
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    result_file = tmp_path / "stored_inputs.json"
    result_file.write_text(json.dumps({"status": "passed", "checked_columns": 12}))

    progress.attach_artifact("stored_inputs", result_file)

    manifest = _read(_run_dir(tmp_path), "run_manifest.json")
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    assert [
        (artifact["logical_name"], artifact["contract_relative_path"])
        for artifact in artifacts
    ] == [("stored_inputs", "artifacts/stored_inputs.json")]
    assert (
        emitter.client.hub_files[
            (US_STAGING, f"runs/{BUILD}/artifacts/stored_inputs.json")
        ]
        == result_file.read_bytes()
    )


def test_a_result_file_the_content_checks_refuse_stays_out_and_the_build_goes_on(
    builder, tmp_path, capsys
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    refused = tmp_path / "input_coverage.json"
    refused.write_text(json.dumps({"token": "not for the staging repository"}))
    passing = tmp_path / "stored_inputs.json"
    passing.write_text(json.dumps({"status": "passed"}))

    progress.attach_artifact("input_coverage", refused)
    progress.attach_artifact("stored_inputs", passing)
    progress.complete()

    artifacts = _read(_run_dir(tmp_path), "run_manifest.json").get("artifacts")
    assert isinstance(artifacts, list)
    assert [artifact["logical_name"] for artifact in artifacts] == ["stored_inputs"]
    assert not any(
        "not for the staging repository" in content.decode(errors="replace")
        for content in emitter.client.hub_files.values()
    )
    assert "input_coverage" in capsys.readouterr().err
    assert [event["status"] for event in _run_events(emitter)] == ["completed"]


def test_a_staging_error_fails_neither_the_build_nor_its_hosted_telemetry(
    builder, capsys
) -> None:
    emitter = _hosted_telemetry()

    class BrokenBundle:
        remote_staging = False

        def stage(self, *args, **kwargs):
            raise OSError("staging disk unavailable")

        attach_artifact = calibration_progress = complete = stage

    progress = builder._BuildProgress(
        run_id=BUILD, emitter=emitter, staging_bundle=BrokenBundle()
    )

    try:
        progress.stage("target_compilation", message="Compiling targets.")
        progress.complete()
    except OSError:
        pytest.fail("a staging error stopped the build")

    assert [
        (event["stage_id"], event["status"])
        for event in emitter.events
        if event["stage_id"] == "target_compilation"
    ][0] == ("target_compilation", "started")
    assert [event["status"] for event in _run_events(emitter)] == ["completed"]
    assert "staging disk unavailable" in capsys.readouterr().err


def test_the_build_manifest_records_the_version_2_delivery_summary(
    builder, tmp_path
) -> None:
    progress, _ = _remote_build_progress(builder, tmp_path)
    progress.complete()

    block = builder._staging_manifest_block(progress)

    assert block.get("contract_version") == 2
    assert validate_staging_delivery(block) == block
    assert (block["mode"], block["run_id"]) == ("local_and_remote", BUILD)
    assert block["upload_successes"] >= 1
    opted_out = builder._staging_manifest_block(None)
    assert validate_staging_delivery(opted_out) == opted_out
    assert (opted_out["enabled"], opted_out["opt_out_reason"]) == (
        False,
        "--no-staging",
    )


def test_certification_opens_a_version_2_run_of_its_own(builder, tmp_path) -> None:
    emitter = _hosted_telemetry()
    record = SimpleNamespace(
        build=BUILD,
        release_root=tmp_path,
        staging={"run_id": BUILD},
        staging_settings={"no_staging": False, "repo_id": US_STAGING, "prefix": "runs"},
    )

    progress = builder._certify_progress(
        SimpleNamespace(no_staging=False, staging_dir=None),
        record=record,
        build=f"{BUILD}-evidence",
        run_id=f"{BUILD}-certify",
        emitter=emitter,
    )
    progress.complete()

    manifest = _read(_run_dir(tmp_path, f"{BUILD}-certify"), "run_manifest.json")
    assert {
        field: manifest.get(field)
        for field in (
            "run_id",
            "operation_id",
            "candidate_id",
            "release_id",
            "run_kind",
        )
    } == {
        "run_id": f"{BUILD}-certify",
        "operation_id": "us_fiscal_refresh_certification",
        "candidate_id": BUILD,
        "release_id": f"{BUILD}-evidence",
        "run_kind": "certification",
    }
    validate_staging_bundle(tmp_path / "staging", f"{BUILD}-certify")
