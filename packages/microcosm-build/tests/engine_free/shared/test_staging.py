import json
import threading
import time

import microcosm.build.staging as staging_module
from microcosm.build.staging import StagingTelemetry
from microcosm.build.staging_storage import BestEffortUploadSession
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

V1_FIXTURE = _TEST_PATHS.tests / "fixtures" / "staging" / "v1"


class FakeApi:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str, str]] = []

    def upload_file(
        self,
        *,
        path_or_fileobj,
        path_in_repo,
        repo_id,
        repo_type,
    ):
        self.uploads.append((str(path_or_fileobj), path_in_repo, repo_id, repo_type))
        return None


def test_staging_telemetry_writes_local_progress(tmp_path) -> None:
    telemetry = StagingTelemetry(
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


def test_staging_telemetry_uploads_repo_paths(tmp_path) -> None:
    api = FakeApi()
    telemetry = StagingTelemetry(
        run_id="run-b",
        candidate_release_id="populace-us-2024-def-20260618T000000Z",
        run_dir=tmp_path / "run-b",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
    )
    telemetry.stage("target_compilation", force_upload=True)

    assert isinstance(telemetry._upload_session, BestEffortUploadSession)
    uploaded_paths = {upload[1] for upload in api.uploads}
    assert "runs/run-b/progress.json" in uploaded_paths
    assert "runs/run-b/run_manifest.json" in uploaded_paths
    assert "latest_staging.json" in uploaded_paths
    assert "runs.json" in uploaded_paths


def test_runs_index_keeps_existing_runs(tmp_path) -> None:
    # An older run is already recorded in the repo's index.
    existing = tmp_path / "existing.json"
    existing.write_text(
        json.dumps(
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
        )
    )

    class FakeApiWithDownload(FakeApi):
        def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
            return str(existing)

    telemetry = StagingTelemetry(
        run_id="newer",
        candidate_release_id="populace-us-2024-new-20260620T000000Z",
        run_dir=tmp_path / "newer",
        repo_id="policyengine/populace-us-staging",
        api=FakeApiWithDownload(),
        upload_interval_seconds=0,
    )
    telemetry.complete()

    index = json.loads((tmp_path / "newer" / "runs.json").read_text())
    run_ids = [run["run_id"] for run in index["runs"]]
    # Both the pre-existing run and this run survive (no overwrite).
    assert run_ids == ["newer", "older"]


def test_upload_failures_never_raise_and_disable_after_three(tmp_path, capsys):
    class FailingApi:
        def upload_file(self, **kwargs):
            raise RuntimeError("401 Unauthorized")

        def hf_hub_download(self, **kwargs):
            raise RuntimeError("401 Unauthorized")

    telemetry = StagingTelemetry(
        run_id="run-1",
        candidate_release_id="run-1",
        run_dir=tmp_path / "run",
        repo_id="org/staging",
        api=FailingApi(),
        upload_interval_seconds=0.0,
    )
    # Staging telemetry is best-effort: failing uploads must never raise, and
    # after three consecutive failures uploads are disabled for the run.
    telemetry.stage("one", force_upload=True)
    telemetry.stage("two", force_upload=True)
    assert telemetry.repo_id is None
    err = capsys.readouterr().err
    assert "staging upload" in err
    assert "disabling staging uploads" in err
    # Nothing reached the repo, and the run says so. A configured destination
    # is not evidence of delivery.
    assert telemetry.uploads_succeeded == 0


def test_uploads_succeeded_counts_files_that_reached_the_repo(tmp_path):
    api = FakeApi()
    telemetry = StagingTelemetry(
        run_id="run-c",
        candidate_release_id="run-c",
        run_dir=tmp_path / "run-c",
        repo_id="org/staging",
        api=api,
        upload_interval_seconds=0.0,
    )

    telemetry.stage("target_compilation", force_upload=True)

    assert telemetry.uploads_succeeded == len(api.uploads)
    assert telemetry.uploads_succeeded > 0


def test_blank_path_prefix_falls_back_to_the_default(tmp_path):
    # A blank or slash-only prefix would put run files at the repo root, where
    # the dashboard's runs/<run_id> paths cannot find them.
    for blank in ("", "   ", "/", " / "):
        telemetry = StagingTelemetry(
            run_id="run-e",
            candidate_release_id="run-e",
            run_dir=tmp_path / "run-e",
            path_prefix=blank,
        )
        assert telemetry.path_prefix == "runs"
        assert telemetry.repo_run_prefix == "runs/run-e"


def test_path_prefix_is_trimmed_but_otherwise_respected(tmp_path):
    telemetry = StagingTelemetry(
        run_id="run-f",
        candidate_release_id="run-f",
        run_dir=tmp_path / "run-f",
        path_prefix="  /candidate-runs/  ",
    )
    assert telemetry.path_prefix == "candidate-runs"
    assert telemetry.repo_run_prefix == "candidate-runs/run-f"


def test_uploads_succeeded_is_zero_for_a_local_only_run(tmp_path):
    telemetry = StagingTelemetry(
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

    class FixtureApi(FakeApi):
        def hf_hub_download(self, **kwargs):
            raise FileNotFoundError

    run_dir = tmp_path / "v1-us-fixture"
    telemetry = StagingTelemetry(
        run_id="v1-us-fixture",
        candidate_release_id="populace-us-2024-v1-fixture",
        run_dir=run_dir,
        repo_id="policyengine/populace-us-staging",
        api=FixtureApi(),
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


def test_pointer_free_runs_upload_only_under_their_prefix(tmp_path) -> None:
    api = FakeApi()
    telemetry = StagingTelemetry(
        run_id="run-k",
        candidate_release_id="populace-us-2024-k20000-fixture",
        run_dir=tmp_path / "run-k",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
        update_pointers=False,
    )
    telemetry.stage("target_compilation", force_upload=True)
    telemetry.complete()

    uploaded_paths = {upload[1] for upload in api.uploads}
    assert "runs/run-k/progress.json" in uploaded_paths
    assert all(path.startswith("runs/run-k/") for path in uploaded_paths)


class BlockingApi(FakeApi):
    """Holds every upload until released, like a slow or unreachable Hub."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def upload_file(self, **kwargs):
        self.release.wait(timeout=10)
        return super().upload_file(**kwargs)

    def hf_hub_download(self, **kwargs):
        raise FileNotFoundError("no runs index yet")


def test_background_uploads_never_block_the_build(tmp_path) -> None:
    api = BlockingApi()
    telemetry = StagingTelemetry(
        run_id="run-bg",
        candidate_release_id="populace-us-2024-bg-20260618T000000Z",
        run_dir=tmp_path / "run-bg",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
        background_uploads=True,
    )

    # The Hub is stuck, yet the build thread keeps going and writes locally.
    for stage in ("target_registry", "load_base_frame", "target_compilation"):
        telemetry.stage(stage, force_upload=True)
    progress = json.loads((tmp_path / "run-bg" / "progress.json").read_text())
    assert progress["stage"] == "target_compilation"
    assert api.uploads == []

    api.release.set()
    telemetry.complete()

    # The terminal state is awaited and uploaded, from a consistent snapshot.
    uploaded = {path: local for local, path, _, _ in api.uploads}
    final = json.loads(open(uploaded["runs/run-bg/progress.json"]).read())
    assert final["status"] == "passed"
    events = open(uploaded["runs/run-bg/events.ndjson"]).read().splitlines()
    assert json.loads(events[-1])["stage"] == "complete"
    assert telemetry.uploads_succeeded > 0


def test_background_final_upload_wait_is_bounded(tmp_path, capsys) -> None:
    api = BlockingApi()
    telemetry = StagingTelemetry(
        run_id="run-stuck",
        candidate_release_id="populace-us-2024-stuck-20260618T000000Z",
        run_dir=tmp_path / "run-stuck",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
        background_uploads=True,
        final_upload_timeout_seconds=0.2,
    )

    telemetry.complete()

    assert "did not finish within 0.2 s" in capsys.readouterr().err
    progress = json.loads((tmp_path / "run-stuck" / "progress.json").read_text())
    assert progress["status"] == "passed"
    api.release.set()


class RoleApi(FakeApi):
    """A Hub client whose credential has the given role."""

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    def whoami(self):
        return {"auth": {"accessToken": {"role": self.role}}}

    def hf_hub_download(self, **kwargs):
        raise FileNotFoundError("no runs index yet")


def test_a_run_without_write_access_builds_on_and_stays_local(tmp_path, capsys):
    api = RoleApi("read")
    telemetry = StagingTelemetry(
        run_id="run-ro",
        candidate_release_id="populace-us-2024-ro",
        run_dir=tmp_path / "run-ro",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
        check_write_access=True,
    )
    telemetry.stage("target_compilation", force_upload=True)
    telemetry.complete()

    # Nothing is refused: the run completes with full local telemetry.
    progress = json.loads((tmp_path / "run-ro" / "progress.json").read_text())
    assert progress["status"] == "passed"
    assert api.uploads == []
    manifest = json.loads((tmp_path / "run-ro" / "run_manifest.json").read_text())
    assert manifest["delivery_check"] == {
        "uploads": "local_only",
        "repository": "policyengine/populace-us-staging",
        "reason": "the Hugging Face token cannot write this repository",
    }
    assert "restage_us_staging_run.py --run-dir" in capsys.readouterr().err


def test_a_run_with_write_access_uploads(tmp_path):
    api = RoleApi("write")
    telemetry = StagingTelemetry(
        run_id="run-rw",
        candidate_release_id="populace-us-2024-rw",
        run_dir=tmp_path / "run-rw",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
        check_write_access=True,
    )
    telemetry.complete()

    assert "runs/run-rw/progress.json" in {upload[1] for upload in api.uploads}
    manifest = json.loads((tmp_path / "run-rw" / "run_manifest.json").read_text())
    assert "delivery_check" not in manifest


def test_restage_uploads_a_local_run_without_moving_the_pointer(tmp_path):
    run_dir = tmp_path / "run-local"
    telemetry = StagingTelemetry(
        run_id="run-local",
        candidate_release_id="populace-us-2024-local",
        run_dir=run_dir,
    )
    (run_dir / "calibration_diagnostics.json").write_text("{}")
    telemetry.attach_artifact(
        "calibration_diagnostics", run_dir / "calibration_diagnostics.json"
    )
    telemetry.complete()

    api = RoleApi("write")
    written = staging_module.restage_run(
        run_dir, repo_id="policyengine/populace-us-staging", api=api
    )

    assert set(written) == {
        "runs/run-local/run_manifest.json",
        "runs/run-local/progress.json",
        "runs/run-local/events.ndjson",
        "runs/run-local/calibration_diagnostics.json",
        "runs.json",
    }
    assert "latest_staging.json" not in {upload[1] for upload in api.uploads}
    uploaded = {path_in_repo: local for local, path_in_repo, _, _ in api.uploads}
    index = json.loads(open(uploaded["runs.json"]).read())
    assert [run["run_id"] for run in index["runs"]] == ["run-local"]

    api = RoleApi("write")
    written = staging_module.restage_run(
        run_dir,
        repo_id="policyengine/populace-us-staging",
        api=api,
        update_index=False,
    )
    assert "runs.json" not in written


def test_resources_are_recorded_on_events_and_progress(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-r",
        candidate_release_id="populace-us-2024-r",
        run_dir=tmp_path / "run-r",
        record_resources=True,
    )
    telemetry.stage("target_compilation")

    events = [
        json.loads(line)
        for line in (tmp_path / "run-r" / "events.ndjson").read_text().splitlines()
    ]
    resources = events[-1]["resources"]
    assert resources["cpu_user_seconds"] >= 0
    assert resources["peak_rss_bytes"] > 0
    progress = json.loads((tmp_path / "run-r" / "progress.json").read_text())
    assert set(progress["resources"]) >= {"cpu_user_seconds", "peak_rss_bytes"}


def test_work_progress_reports_a_rate_and_clears_on_the_next_stage(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-w",
        candidate_release_id="populace-us-2024-w",
        run_dir=tmp_path / "run-w",
    )
    telemetry.stage("target_compilation")
    telemetry.work_progress(10, 2124, unit="engine batch", pass_name="base")

    progress = json.loads((tmp_path / "run-w" / "progress.json").read_text())
    assert progress["work"]["stage"] == "target_compilation"
    assert (progress["work"]["done"], progress["work"]["total"]) == (10, 2124)
    assert progress["work"]["elapsed_seconds"] >= 0
    assert progress["work"]["details"] == {"pass_name": "base"}
    # Work reports add no events.
    events = (tmp_path / "run-w" / "events.ndjson").read_text().splitlines()
    assert [json.loads(line)["stage"] for line in events] == [
        "created",
        "target_compilation",
    ]

    telemetry.stage("calibrating")
    progress = json.loads((tmp_path / "run-w" / "progress.json").read_text())
    assert "work" not in progress


def test_heartbeat_keeps_a_silent_stage_visibly_alive(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-h",
        candidate_release_id="populace-us-2024-h",
        run_dir=tmp_path / "run-h",
        heartbeat_seconds=0.05,
    )
    telemetry.stage("target_compilation")
    progress_path = tmp_path / "run-h" / "progress.json"
    deadline = time.monotonic() + 5
    while "heartbeat_at" not in json.loads(progress_path.read_text()):
        assert time.monotonic() < deadline, "no heartbeat within 5 s"
        time.sleep(0.02)
    progress = json.loads(progress_path.read_text())
    assert progress["stage"] == "target_compilation"
    assert "resources" in progress

    telemetry.complete()
    assert telemetry._heartbeat_thread is not None
    assert not telemetry._heartbeat_thread.is_alive()


def test_recorded_outcome_classes_the_failure(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-f",
        candidate_release_id="populace-us-2024-f",
        run_dir=tmp_path / "run-f",
        record_outcome=True,
    )
    telemetry.stage("release_gates")
    telemetry.fail(RuntimeError("Release gates failed: QRF tail concentration"))

    events = [
        json.loads(line)
        for line in (tmp_path / "run-f" / "events.ndjson").read_text().splitlines()
    ]
    details = events[-1]["details"]
    assert details["failure_class"] == "gate_refused"
    assert details["failed_during"] == "release_gates"
    assert details["elapsed_seconds"] >= 0


def test_outcome_fields_are_off_by_default(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-d",
        candidate_release_id="populace-us-2024-d",
        run_dir=tmp_path / "run-d",
    )
    telemetry.complete()

    last = (tmp_path / "run-d" / "events.ndjson").read_text().splitlines()[-1]
    assert json.loads(last)["details"] == {}
    assert "resources" not in json.loads(last)


def test_failures_are_classed() -> None:
    class BuildTerminatedError(SystemExit):
        pass

    assert staging_module.classify_failure(BuildTerminatedError(143)) == "terminated"
    assert staging_module.classify_failure(KeyboardInterrupt()) == "interrupted"
    assert staging_module.classify_failure(MemoryError()) == "out_of_memory"
    assert (
        staging_module.classify_failure(RuntimeError("Release gates failed: x"))
        == "gate_refused"
    )
    assert (
        staging_module.classify_failure(SystemExit("The feed pin does not match."))
        == "refused"
    )
    assert staging_module.classify_failure(ValueError("bad column")) == "error"


def test_identity_is_recorded_in_the_run_manifest(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-i",
        candidate_release_id="populace-us-2024-i",
        run_dir=tmp_path / "run-i",
    )
    telemetry.record_identity(git_commit="abc123", host={"cpu_count": 18})

    manifest = json.loads((tmp_path / "run-i" / "run_manifest.json").read_text())
    assert manifest["identity"] == {"git_commit": "abc123", "host": {"cpu_count": 18}}
