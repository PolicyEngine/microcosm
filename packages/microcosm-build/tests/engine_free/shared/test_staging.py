import inspect
import json

import pytest

import microcosm.build.staging as staging_module
from microcosm.build.staging import StagingRunBundleWriter
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

V1_FIXTURE = _TEST_PATHS.tests / "fixtures" / "staging" / "v1"
US_STAGING = "policyengine/populace-us-staging"


def test_staging_run_bundle_writer_sends_no_telemetry_events(tmp_path) -> None:
    """The writer hands over its files; hosted telemetry stays the build's."""
    assert "emitter" not in inspect.signature(StagingRunBundleWriter).parameters
    emitter = FakeEmitterClient()
    telemetry = StagingRunBundleWriter(
        run_id="run-a",
        candidate_release_id="run-a",
        run_dir=tmp_path / "run-a",
        repo_id=US_STAGING,
        emitter_client=emitter,
        upload_interval_seconds=0,
    )
    telemetry.stage("calibrating", force_upload=True)
    telemetry.fail(RuntimeError("synthetic"))

    assert emitter.run_file_requests
    assert set(emitter.calls) <= {"stage_run_files", "run_files_receipt"}
    assert emitter.telemetry_messages == []


def test_staging_run_bundle_writer_writes_local_progress(tmp_path) -> None:
    telemetry = StagingRunBundleWriter(
        run_id="run-a",
        candidate_release_id="populace-us-2024-abc-20260618T000000Z",
        run_dir=tmp_path / "run-a",
    )

    telemetry.stage("calibrating", message="Calibration started.", n_targets=3)
    telemetry.calibration_progress(
        {"kind": "calibration_epoch", "epoch": 1, "epochs": 5, "loss": 12.5}
    )
    telemetry.complete()

    progress = json.loads((tmp_path / "run-a" / "progress.json").read_text())
    calibration = json.loads(
        (tmp_path / "run-a" / "calibration_progress.json").read_text()
    )
    events = (tmp_path / "run-a" / "events.ndjson").read_text().strip().splitlines()

    assert progress["status"] == "passed"
    assert progress["stage"] == "complete"
    assert calibration["events"][0]["loss"] == 12.5
    assert len(events) >= 3


def test_staging_run_bundle_writer_hands_repo_paths_to_the_emitter(tmp_path) -> None:
    emitter = FakeEmitterClient()
    run_dir = tmp_path / "run-b"
    telemetry = StagingRunBundleWriter(
        run_id="run-b",
        candidate_release_id="populace-us-2024-def-20260618T000000Z",
        run_dir=run_dir,
        repo_id=US_STAGING,
        emitter_client=emitter,
        upload_interval_seconds=0,
    )
    telemetry.stage("target_compilation", force_upload=True)

    request = emitter.run_file_requests[-1]
    assert request.repository == US_STAGING
    assert request.run_id == "run-b"
    assert request.prefix == "runs"
    assert request.flush is True
    assert request.commit_interval_seconds == 0
    assert {item.relative_path: item.local_path for item in request.files} == {
        "run_manifest.json": run_dir / "run_manifest.json",
        "progress.json": run_dir / "progress.json",
        "events.ndjson": run_dir / "events.ndjson",
    }
    assert request.latest_pointer == run_dir / "latest_staging.json"
    assert request.run_index_entry == telemetry.run_summary()
    assert {
        "runs/run-b/progress.json",
        "runs/run-b/run_manifest.json",
        "latest_staging.json",
        "runs.json",
    } <= set(emitter.paths(US_STAGING))
    # Unforced changes wait for the writer's own interval, then go as one
    # hand-over the emitter commits after its interval.
    assert all(not item.flush for item in emitter.run_file_requests[:-1])


def test_the_emitter_merges_this_run_into_the_existing_runs_index(tmp_path) -> None:
    emitter = FakeEmitterClient()
    # An older run is already recorded in the repo's index.
    emitter.hub_files[(US_STAGING, "runs.json")] = json.dumps(
        {
            "schema_version": 1,
            "updated_at": "2026-06-19T00:00:00+00:00",
            "runs": [
                {
                    "run_id": "older",
                    "candidate_release_id": "populace-us-2024-old-20260619T000000Z",
                    "status": "running",
                    "stage": "write_calibration_npz",
                    "updated_at": "2026-06-19T00:00:00+00:00",
                }
            ],
        }
    ).encode()

    telemetry = StagingRunBundleWriter(
        run_id="newer",
        candidate_release_id="populace-us-2024-new-20260620T000000Z",
        run_dir=tmp_path / "newer",
        repo_id=US_STAGING,
        emitter_client=emitter,
        upload_interval_seconds=0,
    )
    telemetry.complete()

    # The writer sends only this run's entry; the emitter merges it at commit
    # time, so both the pre-existing run and this run survive.
    remote = emitter.hub_json(US_STAGING, "runs.json")
    assert [run["run_id"] for run in remote["runs"]] == ["newer", "older"]
    assert emitter.run_file_requests[-1].run_index_entry["status"] == "passed"
    local = json.loads((tmp_path / "newer" / "runs.json").read_text())
    assert [run["run_id"] for run in local["runs"]] == ["newer"]


def test_hand_over_failures_never_raise_and_disable_after_three(tmp_path, capsys):
    emitter = FakeEmitterClient(accept_run_files=False)
    telemetry = StagingRunBundleWriter(
        run_id="run-1",
        candidate_release_id="run-1",
        run_dir=tmp_path / "run",
        repo_id="org/staging",
        emitter_client=emitter,
        upload_interval_seconds=0.0,
    )
    # Staging telemetry is best-effort: hand-overs the emitter does not accept
    # never raise, and after three in a row the run stops handing files over.
    # The writer's own "created" stage was the first.
    telemetry.stage("one", force_upload=True)
    telemetry.stage("two", force_upload=True)
    assert telemetry.repo_id is None
    requests = len(emitter.run_file_requests)
    telemetry.stage("three", force_upload=True)
    assert len(emitter.run_file_requests) == requests == 3
    err = capsys.readouterr().err
    assert "staging hand-over failed" in err
    assert "disabling staging hand-overs" in err
    # Nothing reached the repo, and the run says so. A configured destination
    # is not evidence of delivery.
    assert telemetry.uploads_succeeded == 0


def test_a_build_without_an_emitter_keeps_its_staging_files_local(
    tmp_path, capsys, monkeypatch
):
    from microcosm.build import staging_storage

    monkeypatch.setattr(
        staging_storage.HuggingFaceDatasetStorage,
        "_api",
        lambda storage: pytest.fail("the writer contacted the Hub itself"),
    )
    telemetry = StagingRunBundleWriter(
        run_id="run-1",
        candidate_release_id="run-1",
        run_dir=tmp_path / "run",
        repo_id="org/staging",
        upload_interval_seconds=0.0,
    )
    telemetry.complete()

    assert telemetry.uploads_succeeded == 0
    assert json.loads((tmp_path / "run" / "progress.json").read_text())["status"] == (
        "passed"
    )
    assert "staging hand-over failed" in capsys.readouterr().err


def test_uploads_succeeded_counts_files_the_emitter_committed(tmp_path):
    emitter = FakeEmitterClient()
    telemetry = StagingRunBundleWriter(
        run_id="run-c",
        candidate_release_id="run-c",
        run_dir=tmp_path / "run-c",
        repo_id="org/staging",
        emitter_client=emitter,
        upload_interval_seconds=0.0,
    )

    telemetry.stage("target_compilation", force_upload=True)

    receipt = emitter.run_files_receipt("org/staging", "run-c")
    assert telemetry.uploads_succeeded == receipt["files_committed"]
    assert telemetry.uploads_succeeded > 0
    # The count stays what the emitter last reported once it has shut down.
    emitter.close()
    assert telemetry.uploads_succeeded == receipt["files_committed"]


def test_an_attached_artifact_is_handed_over_once_beside_the_run(tmp_path):
    emitter = FakeEmitterClient()
    telemetry = StagingRunBundleWriter(
        run_id="run-g",
        candidate_release_id="run-g",
        run_dir=tmp_path / "run-g",
        repo_id="org/staging",
        emitter_client=emitter,
        upload_interval_seconds=0.0,
    )
    artifact = tmp_path / "diagnostics.json"
    artifact.write_text(json.dumps({"targets": 3}))

    telemetry.attach_artifact("calibration_diagnostics", artifact)
    telemetry.stage("validation", force_upload=True)

    carrying = [
        request
        for request in emitter.run_file_requests
        if any(item.relative_path == "diagnostics.json" for item in request.files)
    ]
    assert len(carrying) == 1
    assert [item.local_path for item in carrying[0].files] == [artifact]
    assert carrying[0].flush is True
    assert emitter.hub_files[("org/staging", "runs/run-g/diagnostics.json")] == (
        artifact.read_bytes()
    )


def test_blank_path_prefix_falls_back_to_the_default(tmp_path):
    # A blank or slash-only prefix would put run files at the repo root, where
    # the dashboard's runs/<run_id> paths cannot find them.
    for blank in ("", "   ", "/", " / "):
        telemetry = StagingRunBundleWriter(
            run_id="run-e",
            candidate_release_id="run-e",
            run_dir=tmp_path / "run-e",
            path_prefix=blank,
        )
        assert telemetry.path_prefix == "runs"
        assert telemetry.repo_run_prefix == "runs/run-e"


def test_path_prefix_is_trimmed_but_otherwise_respected(tmp_path):
    telemetry = StagingRunBundleWriter(
        run_id="run-f",
        candidate_release_id="run-f",
        run_dir=tmp_path / "run-f",
        path_prefix="  /candidate-runs/  ",
    )
    assert telemetry.path_prefix == "candidate-runs"
    assert telemetry.repo_run_prefix == "candidate-runs/run-f"


def test_uploads_succeeded_is_zero_for_a_local_only_run(tmp_path):
    telemetry = StagingRunBundleWriter(
        run_id="run-d",
        candidate_release_id="run-d",
        run_dir=tmp_path / "run-d",
    )

    telemetry.stage("target_compilation", force_upload=True)

    assert telemetry.uploads_succeeded == 0
    assert (tmp_path / "run-d" / "progress.json").is_file()


def test_us_version_1_bundle_matches_fixed_fixture(tmp_path, monkeypatch):
    timestamps = iter(
        [
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:03+00:00",
            "2026-01-01T00:00:04+00:00",
            "2026-01-01T00:00:05+00:00",
            "2026-01-01T00:00:06+00:00",
            "2026-01-01T00:00:07+00:00",
        ]
    )
    monkeypatch.setattr(staging_module, "_now", lambda: next(timestamps))

    emitter = FakeEmitterClient()
    run_dir = tmp_path / "v1-us-fixture"
    telemetry = StagingRunBundleWriter(
        run_id="v1-us-fixture",
        candidate_release_id="populace-us-2024-v1-fixture",
        run_dir=run_dir,
        repo_id=US_STAGING,
        emitter_client=emitter,
        upload_interval_seconds=float("inf"),
        started_at="2026-01-01T00:00:00+00:00",
    )
    telemetry.stage("calibrating", message="Calibration started.", n_targets=2)
    telemetry.calibration_progress(
        {"kind": "calibration_epoch", "epoch": 1, "epochs": 2, "loss": 1.5}
    )
    telemetry.complete()

    for filename in (
        "run_manifest.json",
        "progress.json",
        "calibration_progress.json",
        "latest_staging.json",
        "runs.json",
    ):
        assert json.loads((run_dir / filename).read_text()) == json.loads(
            (V1_FIXTURE / filename).read_text()
        )
    assert (run_dir / "events.ndjson").read_text().splitlines() == (
        V1_FIXTURE / "events.ndjson"
    ).read_text().splitlines()
    # An infinite interval hands over only the forced final state.
    assert len(emitter.run_file_requests) == 1
    assert (
        emitter.hub_files[(US_STAGING, "latest_staging.json")]
        == (run_dir / "latest_staging.json").read_bytes()
    )
