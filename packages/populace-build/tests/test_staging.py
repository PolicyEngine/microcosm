import json

from populace.build.staging import StagingTelemetry


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
