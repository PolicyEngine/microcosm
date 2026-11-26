import inspect
import json
import time

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
    )
    telemetry.stage("calibrating")
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
    )

    # The run's start is its first hand-over.
    request = emitter.run_file_requests[-1]
    assert request.repository == US_STAGING
    assert request.run_id == "run-b"
    assert request.prefix == "runs"
    assert request.flush is True
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
    assert len(emitter.run_file_requests) == 1


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
    )
    telemetry.complete()

    # The writer sends only this run's entry; the emitter merges it at commit
    # time, so both the pre-existing run and this run survive.
    remote = emitter.hub_json(US_STAGING, "runs.json")
    assert [run["run_id"] for run in remote["runs"]] == ["newer", "older"]
    assert emitter.run_file_requests[-1].run_index_entry["status"] == "passed"
    local = json.loads((tmp_path / "newer" / "runs.json").read_text())
    assert [run["run_id"] for run in local["runs"]] == ["newer"]


def test_hand_over_failures_never_raise_and_pause_after_three(tmp_path, capsys):
    emitter = FakeEmitterClient(accept_run_files=False)
    telemetry = StagingRunBundleWriter(
        run_id="run-1",
        candidate_release_id="run-1",
        run_dir=tmp_path / "run",
        repo_id="org/staging",
        emitter_client=emitter,
    )
    # Staging telemetry is best-effort: hand-overs the emitter does not accept
    # never raise, and after three in a row the run stops handing files over
    # until its final state. The run's start was the first; each result file
    # goes with the run's files in one more.
    for name in ("first", "second"):
        result = tmp_path / f"{name}.json"
        result.write_text(json.dumps({"name": name}))
        telemetry.attach_artifact(name, result)
    # The configured destination stays on record; the delivery counts say
    # what reached it.
    assert telemetry.repo_id == "org/staging"
    requests = len(emitter.run_file_requests)
    third = tmp_path / "third.json"
    third.write_text(json.dumps({"name": "third"}))
    telemetry.attach_artifact("third", third)
    assert len(emitter.run_file_requests) == requests == 3
    err = capsys.readouterr().err
    assert "staging hand-over failed" in err
    assert "pausing staging hand-overs" in err
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
    )

    receipt = emitter.run_files_receipt("org/staging", "run-c")
    assert telemetry.uploads_succeeded == receipt["files_committed"]
    assert telemetry.uploads_succeeded > 0
    # The count stays what the emitter last reported once it has shut down.
    emitter.close()
    assert telemetry.uploads_succeeded == receipt["files_committed"]


def test_an_attached_artifact_goes_with_every_later_hand_over(tmp_path):
    """The emitter skips bytes it already committed, so a result file it
    missed reaches the repo with the next hand-over."""
    emitter = FakeEmitterClient()
    telemetry = StagingRunBundleWriter(
        run_id="run-g",
        candidate_release_id="run-g",
        run_dir=tmp_path / "run-g",
        repo_id="org/staging",
        emitter_client=emitter,
    )
    artifact = tmp_path / "diagnostics.json"
    artifact.write_text(json.dumps({"targets": 3}))

    telemetry.attach_artifact("calibration_diagnostics", artifact)
    telemetry.complete()

    attached, completed = emitter.run_file_requests[1:]
    for request in (attached, completed):
        files = {item.relative_path: item.local_path for item in request.files}
        assert files["diagnostics.json"] == artifact
        assert "run_manifest.json" in files
        assert request.flush is True
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

    telemetry.stage("target_compilation")

    assert telemetry.uploads_succeeded == 0
    assert (tmp_path / "run-d" / "progress.json").is_file()


def test_us_version_1_bundle_matches_fixed_fixture(tmp_path, monkeypatch):
    # The run's start hand-over writes the pointer and the index right after
    # the "created" stage; the final hand-over rewrites both.
    timestamps = iter(
        [
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:01+00:00",
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
    # Only the run's start and its final state are handed over.
    assert len(emitter.run_file_requests) == 2
    assert (
        emitter.hub_files[(US_STAGING, "latest_staging.json")]
        == (run_dir / "latest_staging.json").read_bytes()
    )


class _HourlyClock:
    """A monotonic clock that moves on an hour each time it is read."""

    def __init__(self) -> None:
        self.seconds = 1_000_000.0

    def __call__(self) -> float:
        self.seconds += 3600.0
        return self.seconds


def _remote_writer(tmp_path, emitter, run_id="run-s"):
    return StagingRunBundleWriter(
        run_id=run_id,
        candidate_release_id=run_id,
        run_dir=tmp_path / run_id,
        repo_id=US_STAGING,
        emitter_client=emitter,
    )


def test_the_run_start_is_handed_over_once_and_flushed(tmp_path):
    """The run's folder exists on the Hub from its start, whatever follows."""
    emitter = FakeEmitterClient()
    run_dir = tmp_path / "run-s"

    _remote_writer(tmp_path, emitter)

    assert len(emitter.run_file_requests) == 1
    start = emitter.run_file_requests[0]
    assert start.flush is True
    assert {item.relative_path for item in start.files} == {
        "run_manifest.json",
        "progress.json",
        "events.ndjson",
    }
    assert start.latest_pointer == run_dir / "latest_staging.json"
    progress = emitter.hub_json(US_STAGING, "runs/run-s/progress.json")
    assert (progress["status"], progress["stage"]) == ("running", "created")


def test_only_the_start_result_files_and_end_are_handed_over(tmp_path, monkeypatch):
    """Progress between those points stays local, however long the run takes."""
    monkeypatch.setattr(time, "monotonic", _HourlyClock())
    emitter = FakeEmitterClient()
    telemetry = _remote_writer(tmp_path, emitter)
    after_start = len(emitter.run_file_requests)

    telemetry.stage("target_registry", message="Compiled targets.", n_targets=3)
    for epoch in range(1, 4):
        telemetry.calibration_progress(
            {"kind": "calibration_epoch", "epoch": epoch, "epochs": 3, "loss": 1.0}
        )
    telemetry.stage("export_dataset", message="Exporting.")
    assert len(emitter.run_file_requests) == after_start

    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(json.dumps({"targets": 3}))
    telemetry.attach_artifact("calibration_diagnostics", diagnostics)
    after_result = len(emitter.run_file_requests)
    assert after_result > after_start
    assert emitter.hub_files[
        (US_STAGING, "runs/run-s/calibration_diagnostics.json")
    ] == (diagnostics.read_bytes())
    manifest = emitter.hub_json(US_STAGING, "runs/run-s/run_manifest.json")
    assert "calibration_diagnostics" in manifest["artifacts"]

    telemetry.stage("post_export_scoring", message="Scoring.")
    assert len(emitter.run_file_requests) == after_result

    telemetry.complete()
    assert len(emitter.run_file_requests) > after_result
    assert all(request.flush for request in emitter.run_file_requests)
    progress = emitter.hub_json(US_STAGING, "runs/run-s/progress.json")
    assert (progress["status"], progress["stage"]) == ("passed", "complete")
    calibration = emitter.hub_json(US_STAGING, "runs/run-s/calibration_progress.json")
    assert [row["epoch"] for row in calibration["events"]] == [1, 2, 3]


def _paused_writer(tmp_path, emitter):
    """A run whose hand-overs paused after three the emitter did not accept."""
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-p")
    for name in ("first", "second"):
        result = tmp_path / f"{name}.json"
        result.write_text(json.dumps({"name": name}))
        telemetry.attach_artifact(name, result)
    assert len(emitter.run_file_requests) >= 3
    assert not emitter.paths(US_STAGING)
    return telemetry


def test_completion_hands_over_the_final_state_after_paused_hand_overs(tmp_path):
    emitter = FakeEmitterClient(accept_run_files=False)
    telemetry = _paused_writer(tmp_path, emitter)
    emitter.accept_run_files = True

    telemetry.complete()

    assert "runs/run-p/progress.json" in emitter.paths(US_STAGING)
    progress = emitter.hub_json(US_STAGING, "runs/run-p/progress.json")
    assert (progress["status"], progress["stage"]) == ("passed", "complete")
    assert emitter.run_file_requests[-1].flush is True
    assert {"runs/run-p/run_manifest.json", "runs/run-p/events.ndjson"} <= set(
        emitter.paths(US_STAGING)
    )


def test_failure_hands_over_the_final_state_after_paused_hand_overs(tmp_path):
    emitter = FakeEmitterClient(accept_run_files=False)
    telemetry = _paused_writer(tmp_path, emitter)
    emitter.accept_run_files = True

    telemetry.fail(RuntimeError("solver exploded"))

    assert "runs/run-p/progress.json" in emitter.paths(US_STAGING)
    progress = emitter.hub_json(US_STAGING, "runs/run-p/progress.json")
    assert (progress["status"], progress["stage"]) == ("failed", "failed")
    assert progress["message"] == "solver exploded"


def test_the_writer_has_no_commit_interval_or_forced_stage_option():
    """Every hand-over is forced, so neither an interval nor a flag applies."""
    assert (
        "upload_interval_seconds"
        not in inspect.signature(StagingRunBundleWriter).parameters
    )
    assert (
        "force_upload" not in inspect.signature(StagingRunBundleWriter.stage).parameters
    )
    assert (
        "force_upload"
        not in inspect.signature(StagingRunBundleWriter.attach_artifact).parameters
    )


def test_a_completed_run_confirms_that_the_hub_holds_its_final_state(tmp_path):
    emitter = FakeEmitterClient()
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-c")
    telemetry.complete()
    assert callable(getattr(telemetry, "confirm_final_commit", None))

    check = telemetry.confirm_final_commit(wait_seconds=120.0)

    assert check.confirmed is True
    assert check.reason is None
    assert check.files_compared > 0
    assert emitter.read_back_waits == [120.0]


def test_an_unconfirmed_final_commit_warns_without_raising(tmp_path, capsys):
    emitter = FakeEmitterClient()
    emitter.read_back_answer = {
        "status": "failed",
        "files": 4,
        "mismatched_paths": ["runs/run-u/progress.json"],
        "error_code": None,
    }
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-u")
    telemetry.fail(RuntimeError("solver exploded"))
    assert callable(getattr(telemetry, "confirm_final_commit", None))

    check = telemetry.confirm_final_commit(wait_seconds=30.0)

    assert check.confirmed is False
    assert check.reason == "failed"
    assert emitter.read_back_waits == [30.0]
    err = capsys.readouterr().err
    assert "final commit" in err and "not confirmed" in err


def test_a_final_state_the_emitter_refused_is_not_read_back(tmp_path):
    """Read-back compares the Hub with what the emitter committed, which would
    then be an older state of the run."""
    emitter = FakeEmitterClient()
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-r")
    emitter.accept_run_files = False
    telemetry.complete()
    assert callable(getattr(telemetry, "confirm_final_commit", None))

    check = telemetry.confirm_final_commit(wait_seconds=120.0)

    assert check.confirmed is False
    assert check.reason == "not_handed_over"
    assert "read_back_run_files" not in emitter.calls


def test_a_local_only_run_has_no_final_commit_to_confirm(tmp_path):
    telemetry = StagingRunBundleWriter(
        run_id="run-l", candidate_release_id="run-l", run_dir=tmp_path / "run-l"
    )
    telemetry.complete()
    assert callable(getattr(telemetry, "confirm_final_commit", None))

    assert telemetry.confirm_final_commit(wait_seconds=120.0) is None


class _MissingCredential(FakeEmitterClient):
    """An emitter with no Hugging Face credential: every commit stays pending."""

    def _receipt(self, repository, run_id):
        return {
            **super()._receipt(repository, run_id),
            "pending_files": 3,
            "last_error_code": "CREDENTIAL_MISSING",
        }


def test_a_receipt_check_confirms_without_reading_the_hub_back(tmp_path):
    """The check after an interrupt: the Hub comparison can outlast the grace."""
    emitter = FakeEmitterClient()
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-q")
    telemetry.complete()

    assert "read_back" in inspect.signature(telemetry.confirm_final_commit).parameters
    check = telemetry.confirm_final_commit(wait_seconds=30.0, read_back=False)

    assert check.confirmed is True
    assert check.method == "receipt"
    assert "read_back_run_files" not in emitter.calls


def test_a_missing_credential_ends_the_check_at_once(tmp_path):
    """Waiting cannot help a commit that has no credential."""
    emitter = _MissingCredential()
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-m")
    telemetry.complete()

    check = telemetry.confirm_final_commit(wait_seconds=120.0)

    assert check.confirmed is False
    assert check.reason == "CREDENTIAL_MISSING"
    assert "read_back_run_files" not in emitter.calls


def test_the_final_hand_over_carries_result_files_missed_while_paused(tmp_path):
    emitter = FakeEmitterClient(accept_run_files=False)
    telemetry = _paused_writer(tmp_path, emitter)
    emitter.accept_run_files = True

    telemetry.complete()

    assert {"runs/run-p/first.json", "runs/run-p/second.json"} <= set(
        emitter.paths(US_STAGING)
    )


class _PendingReceipts(FakeEmitterClient):
    """An emitter whose receipts show files pending for the first few queries."""

    def __init__(self, pending_queries: int) -> None:
        super().__init__()
        self.pending_queries = pending_queries

    def run_files_receipt(self, repository, run_id):
        receipt = super().run_files_receipt(repository, run_id)
        if self.pending_queries > 0:
            self.pending_queries -= 1
            return {**receipt, "pending_files": 2}
        return receipt


def test_a_receipt_check_waits_until_no_file_is_pending(tmp_path):
    emitter = _PendingReceipts(pending_queries=3)
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-w")
    telemetry.complete()

    check = telemetry.confirm_final_commit(wait_seconds=30.0, read_back=False)

    assert (check.confirmed, check.method) == (True, "receipt")
    assert emitter.pending_queries == 0


def test_a_receipt_check_gives_up_after_its_wait(tmp_path):
    emitter = _PendingReceipts(pending_queries=1_000)
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-g")
    telemetry.complete()

    check = telemetry.confirm_final_commit(wait_seconds=0.0, read_back=False)

    assert (check.confirmed, check.reason) == (False, "pending")


def test_a_receipt_check_stops_when_the_emitter_stops_answering(tmp_path):
    """A stale receipt from the last hand-over must not be polled for 30 s."""
    from microcosm_emitter_interface.staging import unavailable_run_files_receipt

    class Stopped(FakeEmitterClient):
        def _receipt(self, repository, run_id):
            return {**super()._receipt(repository, run_id), "pending_files": 3}

        def run_files_receipt(self, repository, run_id):
            self.calls.append("run_files_receipt")
            return unavailable_run_files_receipt(repository, run_id)

    emitter = Stopped()
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-x")
    telemetry.complete()

    check = telemetry.confirm_final_commit(wait_seconds=0.5, read_back=False)

    assert (check.confirmed, check.reason) == (False, "unavailable")


def test_a_result_file_too_large_for_the_emitter_stays_out_of_the_hand_over(
    tmp_path, monkeypatch, capsys
):
    """The emitter refuses a whole request with one file over its limit, which
    would refuse the run's documents and its final state with it."""
    monkeypatch.setattr(staging_module, "_MAX_RUN_FILE_BYTES", 100, raising=False)
    emitter = FakeEmitterClient()
    telemetry = _remote_writer(tmp_path, emitter, run_id="run-l")
    large = tmp_path / "large.json"
    large.write_text(json.dumps({"rows": "x" * 200}))
    small = tmp_path / "small.json"
    small.write_text(json.dumps({"rows": 1}))

    telemetry.attach_artifact("large", large)
    telemetry.attach_artifact("small", small)
    telemetry.complete()

    assert not any(
        item.relative_path == "large.json"
        for request in emitter.run_file_requests
        for item in request.files
    )
    paths = set(emitter.paths(US_STAGING))
    assert "runs/run-l/small.json" in paths
    assert emitter.hub_json(US_STAGING, "runs/run-l/progress.json")["status"] == (
        "passed"
    )
    assert "larger than the emitter's limit for one file" in capsys.readouterr().err


class _MeasuringEmitter(FakeEmitterClient):
    """Records each hand-over's bytes as the emitter totals them: every file
    and the latest-run pointer."""

    def __init__(self) -> None:
        super().__init__()
        self.request_bytes: list[int] = []

    def stage_run_files(self, request):
        total = sum(item.local_path.stat().st_size for item in request.files)
        if request.latest_pointer is not None:
            total += request.latest_pointer.stat().st_size
        self.request_bytes.append(total)
        return super().stage_run_files(request)


def test_a_hand_over_counts_the_latest_pointer_in_the_request_limit(
    tmp_path, monkeypatch
):
    """The emitter refuses a whole request whose files and pointer together
    pass its limit, so the writer's budget must count the pointer too."""
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"rows": "x" * 1_000}))
    measured = _MeasuringEmitter()
    telemetry = _remote_writer(tmp_path / "measured", measured, run_id="run-p")
    telemetry.attach_artifact("result", result)
    # The result file's hand-over, pointer included, as the emitter totals it.
    with_result = measured.request_bytes[-1]

    monkeypatch.setattr(
        staging_module, "_MAX_RUN_REQUEST_BYTES", with_result - 1, raising=False
    )
    limited = _MeasuringEmitter()
    telemetry = _remote_writer(tmp_path / "limited", limited, run_id="run-p")
    telemetry.attach_artifact("result", result)

    assert all(total <= with_result - 1 for total in limited.request_bytes)
