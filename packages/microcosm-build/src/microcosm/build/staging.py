"""Staging telemetry for long-running Microcosm builds.

Production releases are contract artifacts: they are only published after the
build completes and validates. Staging telemetry is intentionally weaker: it
records a candidate build before it is a release by writing small JSON
artifacts under ``runs/<run_id>/`` and, optionally, handing them to the
build's local emitter, which commits them to a staging Hugging Face dataset
repo with the build's own credential.

The writer hands the run's files over at three points only: when the run
starts, when a result file is attached and when the run ends. Every hand-over
is forced, so the emitter commits it at its next cycle, and carries the run's
documents with every attached result file the emitter can take. Stage events
and calibration progress in between are written locally and reach the repo
with the next of those hand-overs; the build's hosted telemetry is the live
record of a running build. Once the run has ended,
:meth:`StagingRunBundleWriter.confirm_final_commit` has the emitter check
that the final state reached the repo.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.build.staging_storage import (
    FinalCommitCheck,
    RunFileEmitter,
    RunFileHandOver,
    warn_final_commit_unconfirmed,
    warn_hand_over_failure,
)

STAGING_SCHEMA_VERSION = 1
#: The emitter refuses a whole hand-over in which one file is larger than
#: this, or whose files together are larger than ``_MAX_RUN_REQUEST_BYTES``.
_MAX_RUN_FILE_BYTES = 32 * 1024 * 1024
_MAX_RUN_REQUEST_BYTES = 64 * 1024 * 1024
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
class StagingRunBundleWriter:
    """Write a build's staging run bundle and hand it to the build's emitter.

    Args:
        run_id: Stable id for this build attempt.
        candidate_release_id: Release id this run is expected to produce.
        run_dir: Local directory where staging artifacts are written.
        repo_id: Optional Hugging Face dataset repo for staging files.
        path_prefix: Prefix inside the repo. Defaults to ``"runs"`` so files
            live under ``runs/<run_id>/``.
        emitter_client: The build's local emitter (the interface's
            ``EmitterClient``), which commits the handed-over files. Without
            one, a configured repository receives nothing.
    """

    run_id: str
    candidate_release_id: str
    run_dir: Path | str
    repo_id: str | None = None
    path_prefix: str = DEFAULT_STAGING_PREFIX
    emitter_client: RunFileEmitter | None = None
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
        self._hand_over = (
            RunFileHandOver(
                self.emitter_client,
                repository=self.repo_id.strip(),
                run_id=self.run_id,
                prefix=self.path_prefix,
            )
            if self.repo_id
            else None
        )
        self._calibration_events: list[dict[str, Any]] = []
        self._artifacts: dict[str, dict[str, Any]] = {}
        #: Each attached result file's local copy, by its name in the run's
        #: folder; every hand-over carries them, and the emitter skips bytes
        #: it has already committed.
        self._artifact_files: dict[str, Path] = {}
        self._too_large_warned: set[str] = set()
        #: Whether completion or failure handed the run's final state over;
        #: until then there is no final commit to check.
        self._final_state_handed_over = False
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
        # The run's start is handed over, so its folder exists in the
        # repository from the start even if the build dies before its next
        # hand-over (a result file or the end).
        self._hand_over_run()

    @property
    def repo_run_prefix(self) -> str:
        # path_prefix is normalized non-empty at construction, so there is no
        # root-level fallback here: writing runs to the repo root is the
        # failure this class now refuses, not an alternative layout.
        return f"{self.path_prefix}/{self.run_id}"

    @property
    def remote_staging(self) -> bool:
        """Whether the run's files go to a repository, through the emitter."""
        return self._hand_over is not None

    @property
    def uploads_succeeded(self) -> int:
        """How many files actually reached the staging repo.

        The count the build's emitter last reported committing for this run.
        Zero on a run that was configured to stage but never managed to: no
        emitter, no write token, revoked access, a Hub outage. Staging is
        best-effort and never fails the build, so this is the only signal
        separating a run that staged from one that merely intended to.
        """

        if self._hand_over is None:
            return 0
        self._hand_over.receipt()
        return self._hand_over.files_committed

    def _hand_over_files(
        self,
        files: list[tuple[str, Path]],
        *,
        latest_pointer: Path | None = None,
        run_index_entry: dict[str, Any] | None = None,
    ) -> None:
        if self._hand_over is None or not self._hand_over.enabled:
            return
        # Best-effort: staging telemetry must never fail (or stall) a build.
        # After three hand-overs in a row the emitter does not accept (no
        # emitter, or one that does not answer), pause until the run's final
        # state; local staging artifacts are still written.
        result = self._hand_over.hand_over(
            files,
            latest_pointer=latest_pointer,
            run_index_entry=run_index_entry,
        )
        if result.accepted:
            return
        warn_hand_over_failure(self._hand_over.repository, result)
        if result.became_paused:
            print(
                "warning: pausing staging hand-overs for this run after three "
                "in a row failed, until its final state; local staging "
                "artifacts are still written.",
                file=sys.stderr,
            )

    def _hand_over_run(self) -> None:
        """Hand the run's folder over as it stands: its start, a result file, its end.

        Every attached result file goes with the run's documents, so the final
        hand-over also carries any the emitter missed while hand-overs paused.
        """
        if self._hand_over is None or not self._hand_over.enabled:
            return
        files = [
            (filename, self.run_dir / filename)
            for filename in (
                "run_manifest.json",
                "progress.json",
                "calibration_progress.json",
                "events.ndjson",
            )
            if (self.run_dir / filename).exists()
        ]
        pointer = self._write_latest_pointer()
        self._write_runs_index()
        # A result file named like one of the run's documents is not sent in
        # its place: the documents describe the run. The emitter totals the
        # files and the pointer against its request limit.
        documents = {filename for filename, _ in files}
        files.extend(
            self._sendable_result_files(
                exclude=documents,
                budget=_MAX_RUN_REQUEST_BYTES
                - sum(local.stat().st_size for _, local in files)
                - pointer.stat().st_size,
            )
        )
        # The emitter merges this run's entry into the repository's runs.json
        # when it commits, keeping every other run.
        self._hand_over_files(
            files,
            latest_pointer=pointer,
            run_index_entry=self.run_summary(),
        )

    def _sendable_result_files(
        self, *, exclude: set[str], budget: int
    ) -> list[tuple[str, Path]]:
        """The attached result files one hand-over can carry with the documents.

        A file the emitter would refuse (over its size limit, or past the
        request's) stays local with a warning, so it cannot get the run's
        documents and final state refused with it.
        """
        sendable: list[tuple[str, Path]] = []
        for filename, local in self._artifact_files.items():
            if filename in exclude or not local.is_file():
                continue
            size = local.stat().st_size
            if size > _MAX_RUN_FILE_BYTES or size > budget:
                if filename not in self._too_large_warned:
                    self._too_large_warned.add(filename)
                    limit = (
                        "the emitter's limit for one file"
                        if size > _MAX_RUN_FILE_BYTES
                        else "what is left of the emitter's limit for one hand-over"
                    )
                    print(
                        f"warning: the staging result file {filename} "
                        f"({size} bytes) is larger than {limit}; it stays in "
                        "the local staging files, though the run manifest "
                        "names it.",
                        file=sys.stderr,
                    )
                continue
            budget -= size
            sendable.append((filename, local))
        return sendable

    def _write_latest_pointer(self) -> Path:
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
        return local

    def _write_runs_index(self) -> None:
        """This run's local index: the repository's copy is the emitter's merge."""
        _write_json(
            self.run_dir / RUNS_INDEX,
            {
                "schema_version": STAGING_SCHEMA_VERSION,
                "updated_at": _now(),
                "runs": [self.run_summary()],
            },
        )

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
        **details: Any,
    ) -> None:
        """Record a stage locally; it reaches the repo with the next hand-over."""
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

    def attach_artifact(
        self,
        name: str,
        path: Path | str,
        *,
        copy: bool = False,
    ) -> None:
        """Record a result file and hand it over with the run's current files."""
        source = Path(path)
        local = self.run_dir / source.name if copy else source
        if copy and source.resolve() != local.resolve():
            shutil.copy2(source, local)
        self._artifacts[name] = {
            "path": source.name,
            "staging_path": f"{self.repo_run_prefix}/{source.name}",
        }
        self._artifact_files[source.name] = local
        self._write_run_manifest()
        self._hand_over_run()

    def _reopen_for_final_state(self) -> None:
        """Hand the final state over even after failures paused hand-overs.

        A run whose remote copy stays ``running`` forever is the gap the v20
        national run met; the emitter retries its commits itself.
        """
        if self._hand_over is not None:
            self._hand_over.reopen()

    def fail(self, error: BaseException) -> None:
        self._reopen_for_final_state()
        self.stage(
            "failed",
            status="failed",
            message=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
        self._hand_over_run()
        self._final_state_handed_over = True

    def complete(self) -> None:
        self._reopen_for_final_state()
        self.stage(
            "complete",
            status="passed",
            message="Staging run completed.",
        )
        self._hand_over_run()
        self._final_state_handed_over = True

    def confirm_final_commit(
        self, *, wait_seconds: float, read_back: bool = True
    ) -> FinalCommitCheck | None:
        """Check that the ended run's final state reached the Hub, through the emitter.

        With ``read_back`` the emitter commits the run's pending files, waiting
        up to ``wait_seconds``, and compares the Hub's copies; without it only
        its receipts are read (see ``RunFileHandOver.confirm_final_commit``).
        A check that fails warns and returns; it never fails the build. A run
        without remote staging has nothing to check.
        """
        if self._hand_over is None:
            return None
        if self._progress.get("status") == "running":
            raise RuntimeError(
                "Only a completed or failed staging run has a final commit."
            )
        if not self._final_state_handed_over:
            check = FinalCommitCheck(
                confirmed=False,
                reason="not_handed_over",
                method="read_back" if read_back else "receipt",
            )
        else:
            check = self._hand_over.confirm_final_commit(
                wait_seconds=wait_seconds, read_back=read_back
            )
        if not check.confirmed:
            warn_final_commit_unconfirmed(self._hand_over.repository, check)
        return check
