"""Staging telemetry for long-running Microcosm builds.

Production releases are contract artifacts: they are only published after the
build completes and validates. Staging telemetry is intentionally weaker and
more incremental: it lets dashboards monitor a candidate build before it is a
release by writing small JSON artifacts under ``runs/<run_id>/`` and,
optionally, uploading them to a staging Hugging Face dataset repo.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGING_SCHEMA_VERSION = 1
LATEST_STAGING_POINTER = "latest_staging.json"
RUNS_INDEX = "runs.json"
DEFAULT_STAGING_PREFIX = "runs"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=1, allow_nan=False))


@dataclass
class StagingTelemetry:
    """Write and optionally upload build-run telemetry.

    Args:
        run_id: Stable id for this build attempt.
        candidate_release_id: Release id this run is expected to produce.
        run_dir: Local directory where staging artifacts are written.
        repo_id: Optional Hugging Face dataset repo for staging uploads.
        path_prefix: Prefix inside the repo. Defaults to ``"runs"`` so files
            live under ``runs/<run_id>/``.
        api: Optional ``huggingface_hub.HfApi``-shaped object for tests.
        upload_interval_seconds: Minimum interval between best-effort progress
            uploads. Final states always upload.
    """

    run_id: str
    candidate_release_id: str
    run_dir: Path | str
    repo_id: str | None = None
    path_prefix: str = DEFAULT_STAGING_PREFIX
    api: Any = None
    upload_interval_seconds: float = 30.0
    started_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Normalize here rather than at the caller: a blank or slash-only
        # prefix would otherwise put run files at the repo root, where the
        # dashboard's runs/<run_id> paths cannot find them. Covers the CLI
        # flag, the environment, and programmatic callers in one place.
        self.path_prefix = self.path_prefix.strip().strip("/").strip()
        if not self.path_prefix:
            self.path_prefix = DEFAULT_STAGING_PREFIX
        self._last_upload_at = 0.0
        self._upload_failures = 0
        self._upload_successes = 0
        self._calibration_events: list[dict[str, Any]] = []
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._progress: dict[str, Any] = {
            "schema_version": STAGING_SCHEMA_VERSION,
            "run_id": self.run_id,
            "candidate_release_id": self.candidate_release_id,
            "status": "running",
            "stage": "created",
            "started_at": self.started_at,
            "updated_at": self.started_at,
        }
        self._write_run_manifest()
        self.stage("created", message="Staging run created.")

    @property
    def repo_run_prefix(self) -> str:
        # path_prefix is normalized non-empty at construction, so there is no
        # root-level fallback here: writing runs to the repo root is the
        # failure this class now refuses, not an alternative layout.
        return f"{self.path_prefix}/{self.run_id}"

    @property
    def uploads_succeeded(self) -> int:
        """How many files actually reached the staging repo.

        Zero on a run that was configured to upload but never managed to --
        no write token, revoked access, a Hub outage. Uploads are best-effort
        and never fail the build, so this is the only signal separating a run
        that staged from one that merely intended to.
        """

        return self._upload_successes

    def _api(self):
        if self.api is not None:
            return self.api
        if not self.repo_id:
            return None
        from huggingface_hub import HfApi

        self.api = HfApi()
        return self.api

    def _upload_file(self, local: Path, path_in_repo: str) -> None:
        api = self._api()
        if api is None or not self.repo_id:
            return
        # Best-effort: staging telemetry must never fail (or stall) a build.
        # After three consecutive failures — e.g. no write token — stop trying
        # for the rest of the run; local staging artifacts are still written.
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self._upload_failures = 0
            self._upload_successes += 1
        except Exception as exc:
            self._upload_failures += 1
            print(
                f"warning: staging upload of {path_in_repo} failed: {exc}",
                file=sys.stderr,
            )
            if self._upload_failures >= 3:
                print(
                    "warning: disabling staging uploads for this run after "
                    "three consecutive failures; local staging artifacts are "
                    "still written.",
                    file=sys.stderr,
                )
                self.repo_id = None

    def _maybe_upload(self, *, force: bool = False) -> None:
        if not self.repo_id:
            return
        now = time.monotonic()
        if not force and now - self._last_upload_at < self.upload_interval_seconds:
            return
        self._last_upload_at = now
        for filename in (
            "run_manifest.json",
            "progress.json",
            "calibration_progress.json",
            "events.ndjson",
        ):
            local = self.run_dir / filename
            if local.exists():
                self._upload_file(local, f"{self.repo_run_prefix}/{filename}")
        self._upload_latest_pointer()
        self._upload_runs_index()

    def _upload_latest_pointer(self) -> None:
        payload = {
            "schema_version": STAGING_SCHEMA_VERSION,
            "run_id": self.run_id,
            "candidate_release_id": self.candidate_release_id,
            "updated_at": _now(),
            "paths": {
                "run_manifest": f"{self.repo_run_prefix}/run_manifest.json",
                "progress": f"{self.repo_run_prefix}/progress.json",
                "calibration_progress": (
                    f"{self.repo_run_prefix}/calibration_progress.json"
                ),
                "events": f"{self.repo_run_prefix}/events.ndjson",
            },
        }
        local = self.run_dir / LATEST_STAGING_POINTER
        _write_json(local, payload)
        self._upload_file(local, LATEST_STAGING_POINTER)

    def _existing_runs(self) -> list[dict[str, Any]]:
        """Best-effort fetch of the runs index already in the repo."""
        api = self._api()
        if api is None or not self.repo_id:
            return []
        download = getattr(api, "hf_hub_download", None)
        if download is None:
            try:
                from huggingface_hub import hf_hub_download as download
            except Exception:
                return []
        try:
            local = download(
                repo_id=self.repo_id,
                filename=RUNS_INDEX,
                repo_type="dataset",
                force_download=True,
            )
            data = json.loads(Path(local).read_text())
        except Exception:
            # Index missing (first run) or unreadable; start from scratch.
            return []
        runs = data.get("runs") if isinstance(data, dict) else None
        if not isinstance(runs, list):
            return []
        return [run for run in runs if isinstance(run, dict) and run.get("run_id")]

    def _upload_runs_index(self) -> None:
        # Upsert this run into the existing index rather than overwriting it, so
        # the index keeps every run instead of only the last one to upload.
        current = self.run_summary()
        runs = [
            run for run in self._existing_runs() if run.get("run_id") != self.run_id
        ]
        runs.append(current)
        runs.sort(
            key=lambda run: str(
                run.get("updated_at") or run.get("started_at") or run.get("run_id")
            ),
            reverse=True,
        )
        local = self.run_dir / RUNS_INDEX
        _write_json(
            local,
            {
                "schema_version": STAGING_SCHEMA_VERSION,
                "updated_at": _now(),
                "runs": runs,
            },
        )
        self._upload_file(local, RUNS_INDEX)

    def run_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "candidate_release_id": self.candidate_release_id,
            "status": self._progress.get("status"),
            "stage": self._progress.get("stage"),
            "started_at": self.started_at,
            "updated_at": self._progress.get("updated_at"),
            "progress_path": f"{self.repo_run_prefix}/progress.json",
            "run_manifest_path": f"{self.repo_run_prefix}/run_manifest.json",
        }

    def _write_run_manifest(self) -> None:
        _write_json(
            self.run_dir / "run_manifest.json",
            {
                "schema_version": STAGING_SCHEMA_VERSION,
                "run_id": self.run_id,
                "candidate_release_id": self.candidate_release_id,
                "started_at": self.started_at,
                "repo_id": self.repo_id,
                "path_prefix": self.path_prefix,
                "artifacts": self._artifacts,
            },
        )

    def _write_progress(self) -> None:
        _write_json(self.run_dir / "progress.json", self._progress)

    def _write_calibration_progress(self) -> None:
        _write_json(
            self.run_dir / "calibration_progress.json",
            {
                "schema_version": STAGING_SCHEMA_VERSION,
                "run_id": self.run_id,
                "candidate_release_id": self.candidate_release_id,
                "updated_at": _now(),
                "events": self._calibration_events,
            },
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        path = self.run_dir / "events.ndjson"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(_jsonable(event), allow_nan=False) + "\n")

    def stage(
        self,
        stage: str,
        *,
        status: str = "running",
        message: str | None = None,
        force_upload: bool = False,
        **details: Any,
    ) -> None:
        updated_at = _now()
        self._progress.update(
            {
                "status": status,
                "stage": stage,
                "message": message,
                "updated_at": updated_at,
                "details": _jsonable(details),
            }
        )
        self._write_progress()
        self._append_event(
            {
                "time": updated_at,
                "type": "stage",
                "status": status,
                "stage": stage,
                "message": message,
                "details": details,
            }
        )
        self._maybe_upload(force=force_upload)

    def calibration_progress(self, event: dict[str, object]) -> None:
        if event.get("kind") != "calibration_epoch":
            return
        row = {
            "epoch": event.get("epoch"),
            "epochs": event.get("epochs"),
            "loss": event.get("loss"),
            "budget_search": event.get("budget_search"),
            "budget_iteration": event.get("budget_iteration"),
            "budget_iters": event.get("budget_iters"),
            "l0_lambda": event.get("l0_lambda"),
            "time": _now(),
        }
        self._calibration_events.append(_jsonable(row))
        self._progress.update(
            {
                "status": "running",
                "stage": "calibrating",
                "updated_at": row["time"],
                "calibration": row,
            }
        )
        self._write_progress()
        self._write_calibration_progress()
        self._maybe_upload()

    def attach_artifact(
        self,
        name: str,
        path: Path | str,
        *,
        copy: bool = False,
        force_upload: bool = True,
    ) -> None:
        source = Path(path)
        local = self.run_dir / source.name if copy else source
        if copy and source.resolve() != local.resolve():
            shutil.copy2(source, local)
        self._artifacts[name] = {
            "path": source.name,
            "staging_path": f"{self.repo_run_prefix}/{source.name}",
        }
        self._write_run_manifest()
        if local.exists():
            self._upload_file(local, f"{self.repo_run_prefix}/{source.name}")
        self._maybe_upload(force=force_upload)

    def fail(self, error: BaseException) -> None:
        self.stage(
            "failed",
            status="failed",
            message=str(error),
            force_upload=True,
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )

    def complete(self) -> None:
        self.stage(
            "complete",
            status="passed",
            message="Staging run completed.",
            force_upload=True,
        )
