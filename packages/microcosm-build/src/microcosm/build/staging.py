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
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.build.staging_storage import (
    BestEffortUploadSession,
    HuggingFaceDatasetStorage,
)

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


def _ambient_hub_token() -> str | None:
    """The token huggingface_hub would use (env or ``hf auth login``), if any."""

    try:
        from huggingface_hub.utils import get_token
    except ImportError:  # pragma: no cover - very old huggingface_hub
        return None
    try:
        return get_token()
    except Exception:  # pragma: no cover - defensive
        return None


RUN_FILES = (
    "run_manifest.json",
    "progress.json",
    "calibration_progress.json",
    "events.ndjson",
)


def restage_run(
    run_dir: Path | str,
    *,
    repo_id: str,
    path_prefix: str = DEFAULT_STAGING_PREFIX,
    update_index: bool = True,
    api: Any = None,
) -> list[str]:
    """Upload a finished local run folder to the staging repository.

    For a run that kept its telemetry local (no write token, a failed
    upload, ``--staging-dir`` only). Uploads the run files and the artifacts
    its manifest lists under ``<path_prefix>/<run_id>/``, and upserts the run
    into ``runs.json`` unless ``update_index`` is false. Never moves
    ``latest_staging.json``. Returns the repository paths written.
    """

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    run_id = str(manifest["run_id"])
    prefix = path_prefix.strip().strip("/") or DEFAULT_STAGING_PREFIX
    storage = HuggingFaceDatasetStorage(repo_id, api=api)
    artifacts = manifest.get("artifacts") or {}
    names = list(RUN_FILES) + [
        str(artifact["path"])
        for artifact in artifacts.values()
        if isinstance(artifact, dict) and artifact.get("path")
    ]
    written: list[str] = []
    for name in dict.fromkeys(names):
        local = run_dir / name
        if not local.is_file():
            continue
        path_in_repo = f"{prefix}/{run_id}/{name}"
        storage.upload(local, path_in_repo)
        written.append(path_in_repo)
    if update_index and (run_dir / "progress.json").is_file():
        progress = json.loads((run_dir / "progress.json").read_text())
        try:
            index = json.loads(storage.download(RUNS_INDEX))
            runs = [
                run
                for run in (index.get("runs") or [])
                if isinstance(run, dict) and run.get("run_id") != run_id
            ]
        except Exception:
            runs = []
        runs.append(
            {
                "run_id": run_id,
                "candidate_release_id": manifest.get("candidate_release_id"),
                "status": progress.get("status"),
                "stage": progress.get("stage"),
                "started_at": manifest.get("started_at"),
                "updated_at": progress.get("updated_at"),
                "progress_path": f"{prefix}/{run_id}/progress.json",
                "run_manifest_path": f"{prefix}/{run_id}/run_manifest.json",
            }
        )
        runs.sort(
            key=lambda run: str(run.get("updated_at") or run.get("started_at") or ""),
            reverse=True,
        )
        local_index = run_dir / ".upload" / RUNS_INDEX
        local_index.parent.mkdir(exist_ok=True)
        _write_json(
            local_index,
            {
                "schema_version": STAGING_SCHEMA_VERSION,
                "updated_at": _now(),
                "runs": runs,
            },
        )
        storage.upload(local_index, RUNS_INDEX)
        written.append(RUNS_INDEX)
    return written


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
        update_pointers: Whether uploads also refresh the repository-level
            ``latest_staging.json`` pointer and ``runs.json`` index. A run
            that must never move a shared pointer (an exact-count release)
            sets this to ``False`` and writes only under its own run prefix.
        background_uploads: Upload from a worker thread instead of the build
            thread, so a slow or unreachable Hub never stalls the build. The
            terminal ``complete``/``fail`` upload is still awaited, for at most
            ``final_upload_timeout_seconds``.
        check_write_access: Check at the start that the ambient Hugging Face
            credential can write ``repo_id``. Without one, the run warns, keeps
            its telemetry local, records why in the run manifest, and builds
            as usual; ``restage_run`` uploads the folder later. A check that
            cannot reach the Hub changes nothing: uploads stay best-effort.
    """

    run_id: str
    candidate_release_id: str
    run_dir: Path | str
    repo_id: str | None = None
    path_prefix: str = DEFAULT_STAGING_PREFIX
    api: Any = None
    upload_interval_seconds: float = 30.0
    update_pointers: bool = True
    background_uploads: bool = False
    final_upload_timeout_seconds: float = 120.0
    check_write_access: bool = False
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
        self._storage = (
            HuggingFaceDatasetStorage(self.repo_id, api=self.api)
            if self.repo_id
            else None
        )
        self._upload_session = (
            BestEffortUploadSession(self._storage) if self._storage else None
        )
        self._delivery_check: dict[str, Any] | None = None
        if self._storage is not None and self.check_write_access:
            self._delivery_check = self._check_write_access()
        self._calibration_events: list[dict[str, Any]] = []
        self._artifacts: dict[str, dict[str, Any]] = {}
        # Local writes and the uploader's snapshots take this lock, so a
        # background upload never ships a half-written file.
        self._io_lock = threading.RLock()
        self._pending_artifacts: list[tuple[Path, str]] = []
        self._upload_requested = threading.Event()
        self._uploads_closing = False
        self._upload_thread: threading.Thread | None = None
        if self._upload_session is not None and self.background_uploads:
            self._upload_thread = threading.Thread(
                target=self._upload_worker,
                name=f"staging-upload-{self.run_id}",
                daemon=True,
            )
            self._upload_thread.start()
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

    def _check_write_access(self) -> dict[str, Any] | None:
        """Keep the run local, loudly, when no credential can write the repo.

        Returns the reason recorded in the run manifest, or ``None`` when
        uploads go ahead (write access confirmed, or not determinable).
        """

        repo_id = self.repo_id
        reason: str | None = None
        if self.api is None and _ambient_hub_token() is None:
            reason = "no Hugging Face token is configured"
        else:
            try:
                can_write = self._storage.credential_can_write()
            except Exception as error:
                print(
                    "warning: could not check write access to the staging "
                    f"repository {repo_id} ({type(error).__name__}); uploads "
                    "stay best-effort.",
                    file=sys.stderr,
                )
                return None
            if can_write is False:
                reason = "the Hugging Face token cannot write this repository"
        if reason is None:
            return None
        self.repo_id = None
        self._storage = None
        self._upload_session = None
        print(
            f"warning: staging uploads are off for this run: {reason} "
            f"({repo_id}). The build continues and its telemetry stays in "
            f"{self.run_dir}. Upload it afterwards with:\n"
            f"  uv run python tools/restage_us_staging_run.py --run-dir {self.run_dir}",
            file=sys.stderr,
        )
        return {
            "uploads": "local_only",
            "repository": repo_id,
            "reason": reason,
        }

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

    def _upload_file(self, local: Path, path_in_repo: str) -> None:
        if self._upload_session is None or not self.repo_id:
            return
        # Best-effort: staging telemetry must never fail (or stall) a build.
        # After three consecutive failures — e.g. no write token — stop trying
        # for the rest of the run; local staging artifacts are still written.
        result = self._upload_session.upload(local, path_in_repo)
        self._upload_failures = self._upload_session.consecutive_failures
        self._upload_successes = self._upload_session.successes
        if not result.succeeded:
            print(
                f"warning: staging upload of {path_in_repo} failed: {result.error}",
                file=sys.stderr,
            )
            if result.became_disabled:
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
        if self._upload_thread is None:
            self._upload_cycle()
        else:
            self._upload_requested.set()

    def _upload_cycle(self) -> None:
        with self._io_lock:
            pending = [
                (self._upload_copy(local), path_in_repo)
                for local, path_in_repo in self._pending_artifacts
                if local.exists()
            ]
            self._pending_artifacts.clear()
            for filename in (
                "run_manifest.json",
                "progress.json",
                "calibration_progress.json",
                "events.ndjson",
            ):
                local = self.run_dir / filename
                if local.exists():
                    pending.append(
                        (self._upload_copy(local), f"{self.repo_run_prefix}/{filename}")
                    )
        for local, path_in_repo in pending:
            self._upload_file(local, path_in_repo)
        if self.update_pointers:
            self._upload_latest_pointer()
            self._upload_runs_index()

    def _upload_copy(self, local: Path) -> Path:
        """The file to upload: a snapshot when a worker thread uploads it.

        The build thread keeps appending to these files while the worker
        uploads, so the worker ships a copy taken under the write lock.
        """

        if self._upload_thread is None:
            return local
        snapshots = self.run_dir / ".upload"
        snapshots.mkdir(exist_ok=True)
        copy = snapshots / local.name
        shutil.copyfile(local, copy)
        return copy

    def _upload_worker(self) -> None:
        while True:
            self._upload_requested.wait()
            self._upload_requested.clear()
            try:
                self._upload_cycle()
            except Exception as error:  # pragma: no cover - defensive
                print(
                    f"warning: staging upload cycle failed: {error}",
                    file=sys.stderr,
                )
            if self._uploads_closing and not self._upload_requested.is_set():
                return

    def _finish_uploads(self) -> None:
        """Wait for the terminal upload when uploads run in the background."""

        thread = self._upload_thread
        if thread is None:
            return
        self._uploads_closing = True
        self._upload_requested.set()
        thread.join(timeout=self.final_upload_timeout_seconds)
        if thread.is_alive():
            print(
                "warning: the final staging upload did not finish within "
                f"{self.final_upload_timeout_seconds:g} s; local staging "
                "artifacts are complete.",
                file=sys.stderr,
            )

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
        if self._storage is None or not self.repo_id:
            return []
        try:
            data = json.loads(self._storage.download(RUNS_INDEX))
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
        with self._io_lock:
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
                    **(
                        {"delivery_check": self._delivery_check}
                        if self._delivery_check
                        else {}
                    ),
                },
            )

    def _write_progress(self) -> None:
        with self._io_lock:
            _write_json(self.run_dir / "progress.json", self._progress)

    def _write_calibration_progress(self) -> None:
        with self._io_lock:
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
        with self._io_lock, path.open("a") as stream:
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
            path_in_repo = f"{self.repo_run_prefix}/{source.name}"
            if self._upload_thread is None:
                self._upload_file(local, path_in_repo)
            else:
                with self._io_lock:
                    self._pending_artifacts.append((local, path_in_repo))
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
        self._finish_uploads()

    def complete(self) -> None:
        self.stage(
            "complete",
            status="passed",
            message="Staging run completed.",
            force_upload=True,
        )
        self._finish_uploads()
