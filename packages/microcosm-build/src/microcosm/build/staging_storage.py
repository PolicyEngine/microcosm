"""Country-neutral configuration, remote storage and run-file hand-over.

Staging run files (``runs/<run_id>/...``) go to the build's local emitter,
which commits them to the dataset repository with the build's own Hugging
Face credential (:class:`RunFileHandOver`); so do staged dataset bundles
(:mod:`microcosm.build.staging_dataset`). :class:`HuggingFaceDatasetStorage`
only reads the repository, for the tools that fetch a staged bundle.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from microcosm_emitter_interface import RunFile, StageRunFiles
from microcosm_emitter_interface.staging import (
    ReadBackResult,
    RunFilesReceipt,
    unavailable_read_back,
    unavailable_run_files_receipt,
)

__all__ = [
    "FINAL_COMMIT_CHECK",
    "FINAL_COMMIT_WAIT_SECONDS",
    "INTERRUPTED_FINAL_COMMIT_WAIT_SECONDS",
    "CheckTelemetry",
    "FinalCommitCheck",
    "FinalCommitWriter",
    "HandOverResult",
    "HuggingFaceDatasetStorage",
    "RunFileEmitter",
    "RunFileHandOver",
    "StagingRepositoryConfig",
    "confirm_and_report_final_commit",
    "report_final_commit",
    "warn_final_commit_unconfirmed",
    "warn_hand_over_failure",
]

#: How long a build waits for the emitter to commit its run's final state
#: before reading it back; the emitter spaces two commits to one repository
#: at least 10 seconds apart and retries a failed commit itself. The
#: comparison that follows is one more request of up to 60 seconds.
FINAL_COMMIT_WAIT_SECONDS = 120.0
#: The wait after an interrupt (a SIGTERM, which builds turn into
#: ``BuildTerminatedError``, or Ctrl-C), when the check reads the emitter's
#: receipts and skips the Hub comparison, so that the close-out ends inside
#: the 60 seconds the Modal stage runner allows between SIGTERM and SIGKILL.
#: Each receipt query is a request of up to 5 seconds, so the check can take
#: up to about 40 seconds in all; the emitter finishes its own commits after
#: the build has exited.
INTERRUPTED_FINAL_COMMIT_WAIT_SECONDS = 30.0
#: How a build's hosted telemetry names the check, in the details of the one
#: progress event, outside any stage, that reports it.
FINAL_COMMIT_CHECK = "staging_final_commit"
#: Commit errors that waiting cannot clear.
_PERMANENT_COMMIT_ERRORS = frozenset({"CREDENTIAL_MISSING"})
#: The pause between two receipt queries.
_RECEIPT_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class StagingRepositoryConfig:
    """Country-owned defaults consumed by the shared staging command line."""

    default_repo_id: str
    repo_id_environment_variable: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.default_repo_id, str)
            or not self.default_repo_id.strip()
        ):
            raise ValueError("default_repo_id must be non-empty.")
        if (
            not isinstance(self.repo_id_environment_variable, str)
            or not self.repo_id_environment_variable.strip()
        ):
            raise ValueError("repo_id_environment_variable must be non-empty.")

    def repo_id(self, environment: Mapping[str, str]) -> str:
        """Resolve the repository without embedding a country in shared code."""

        return environment.get(
            self.repo_id_environment_variable,
            self.default_repo_id,
        )


@dataclass
class HuggingFaceDatasetStorage:
    """Read files from one Hugging Face dataset repository."""

    repo_id: str
    api: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id.strip():
            raise ValueError("Remote staging requires a repository identifier.")
        self.repo_id = self.repo_id.strip()

    def _api(self) -> Any:
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi()
        return self.api

    def download_file(self, path_in_repo: str, *, revision: str | None = None) -> Path:
        """Fetch one file through the Hub cache and return its local path.

        The bytes are never read into memory, so it serves population files
        as well as small documents.
        """

        api = self._api()
        download = getattr(api, "hf_hub_download", None)
        if download is None:
            from huggingface_hub import hf_hub_download as download

        local = download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
        )
        return Path(local)


class RunFileEmitter(Protocol):
    """The staging calls of the interface's ``EmitterClient``."""

    def stage_run_files(self, request: StageRunFiles) -> RunFilesReceipt: ...

    def run_files_receipt(self, repository: str, run_id: str) -> RunFilesReceipt: ...

    def read_back_run_files(
        self, repository: str, run_id: str, *, wait_seconds: float = ...
    ) -> ReadBackResult: ...


@dataclass(frozen=True)
class HandOverResult:
    """The emitter's answer to one hand-over of a run's files.

    ``reason`` names why it was not accepted: the receipt's status, or its
    refusal code; never exception text.
    """

    accepted: bool
    receipt: RunFilesReceipt
    paused: bool = False
    became_paused: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class FinalCommitCheck:
    """Whether a closed run's final state reached the Hub, as the emitter says.

    ``method`` is ``read_back`` when the emitter compared the Hub's copy of
    each file with the bytes it committed, and ``receipt`` when only its
    receipt was read (no files pending, at least one committed). ``reason``
    names why it was not confirmed: ``not_handed_over`` when the emitter did
    not accept the final state, ``interrupted``, else the emitter's error
    code or the read-back's status; never exception text. ``files_compared``
    is how many files the emitter compared, when it said.
    """

    confirmed: bool
    reason: str | None = None
    files_compared: int | None = None
    method: str = "read_back"

    def details(self) -> dict[str, Any]:
        """The check as the details of a telemetry event."""
        return {
            "check": FINAL_COMMIT_CHECK,
            "method": self.method,
            "confirmed": self.confirmed,
            "reason": self.reason,
            "files_compared": self.files_compared,
        }


class RunFileHandOver:
    """Hand one staging run's files to the build's local emitter.

    The emitter copies the files' bytes before it answers, so the writer may
    rewrite them at once. Every hand-over is forced: the emitter commits it
    at its next cycle, one Hugging Face commit per repository carrying every
    pending path, with two commits to one repository at least 10 seconds
    apart, and it skips a file whose bytes it has already committed. Its
    receipts say how many files it attempted to commit and how many it
    committed.

    A hand-over the emitter does not accept (no emitter, no answer in time,
    or a refused request) never raises. After ``max_consecutive_failures`` of
    them in a row, the hand-over pauses until :meth:`reopen`, so a missing or
    stalled emitter cannot slow the build. Files in a hand-over that was not
    accepted count as attempted, never as committed.
    """

    def __init__(
        self,
        emitter: RunFileEmitter | None,
        *,
        repository: str,
        run_id: str,
        prefix: str = "runs",
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive.")
        self.emitter = emitter
        self.repository = repository
        self.run_id = run_id
        self.prefix = prefix
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.enabled = True
        #: Whether the most recent hand-over was accepted; None before one.
        self.last_accepted: bool | None = None
        #: The most recent receipt in which the emitter recorded the run.
        self.last_receipt: RunFilesReceipt | None = None
        self._files_not_accepted = 0

    @property
    def files_committed(self) -> int:
        """Files the emitter reported committed, as of its last receipt."""
        receipt = self.last_receipt
        return int(receipt["files_committed"]) if receipt else 0

    @property
    def files_attempted(self) -> int:
        """Files the emitter attempted to commit, plus those it did not accept."""
        receipt = self.last_receipt
        attempted = int(receipt["files_attempted"]) if receipt else 0
        return attempted + self._files_not_accepted

    @property
    def last_error_code(self) -> str | None:
        """The emitter's last commit error for the run, as of its last receipt."""
        receipt = self.last_receipt
        return receipt["last_error_code"] if receipt else None

    def _record(self, receipt: RunFilesReceipt) -> bool:
        if receipt.get("status") != "recorded":
            return False
        self.last_receipt = receipt
        return True

    def receipt(self) -> RunFilesReceipt | None:
        """Ask the emitter for the run's counts; keep the last known otherwise."""
        if self.emitter is not None:
            try:
                self._record(
                    self.emitter.run_files_receipt(self.repository, self.run_id)
                )
            except Exception:
                pass
        return self.last_receipt

    def hand_over(self, files: Sequence[tuple[str, Path]]) -> HandOverResult:
        """Hand over ``(relative path, local path)`` pairs under the run's folder."""
        offered = len(files)
        if not self.enabled:
            return HandOverResult(
                accepted=False,
                receipt=unavailable_run_files_receipt(self.repository, self.run_id),
                paused=True,
                reason="paused",
            )
        receipt: RunFilesReceipt
        try:
            request = StageRunFiles(
                repository=self.repository,
                run_id=self.run_id,
                files=[RunFile(relative, Path(local)) for relative, local in files],
                prefix=self.prefix,
                flush=True,
            )
        except ValueError:
            receipt = {
                **unavailable_run_files_receipt(self.repository, self.run_id),
                "status": "refused",
                "refusal_code": "INVALID_REQUEST",
            }
        else:
            if self.emitter is None:
                receipt = unavailable_run_files_receipt(self.repository, self.run_id)
            else:
                try:
                    receipt = self.emitter.stage_run_files(request)
                except Exception:
                    receipt = unavailable_run_files_receipt(
                        self.repository, self.run_id
                    )
        accepted = self._record(receipt)
        self.last_accepted = accepted
        if accepted:
            self.consecutive_failures = 0
            return HandOverResult(accepted=True, receipt=receipt)
        self._files_not_accepted += offered
        self.consecutive_failures += 1
        became_paused = self.consecutive_failures >= self.max_consecutive_failures
        if became_paused:
            self.enabled = False
        return HandOverResult(
            accepted=False,
            receipt=receipt,
            paused=not self.enabled,
            became_paused=became_paused,
            reason=str(receipt.get("refusal_code") or receipt.get("status")),
        )

    def read_back(self, *, wait_seconds: float) -> ReadBackResult:
        """Ask the emitter to commit the run's pending files and compare the Hub's."""
        if self.emitter is None:
            return unavailable_read_back()
        try:
            return self.emitter.read_back_run_files(
                self.repository, self.run_id, wait_seconds=wait_seconds
            )
        except Exception:
            return unavailable_read_back()

    def confirm_final_commit(
        self, *, wait_seconds: float, read_back: bool = True
    ) -> FinalCommitCheck:
        """Whether the run's last hand-over reached the Hub, as the emitter says.

        With ``read_back`` the emitter commits the run's pending files, waiting
        up to ``wait_seconds``, then compares the Hub's copy of each file with
        the bytes it committed (one more request of up to 60 seconds).
        Without it, the emitter's receipts are read for up to ``wait_seconds``
        until no file is pending. A run whose last hand-over the emitter did
        not accept is not checked: the emitter would compare the Hub with an
        older state of the run. A commit error that waiting cannot clear (no
        credential) ends the check at once.
        """
        method = "read_back" if read_back else "receipt"
        if self.last_accepted is not True:
            return FinalCommitCheck(
                confirmed=False, reason="not_handed_over", method=method
            )
        receipt = self.receipt()
        error = receipt["last_error_code"] if receipt else None
        if error in _PERMANENT_COMMIT_ERRORS:
            return FinalCommitCheck(confirmed=False, reason=error, method=method)
        if not read_back:
            return self._confirm_by_receipt(wait_seconds)
        result = self.read_back(wait_seconds=wait_seconds)
        self.receipt()
        files = result.get("files")
        compared = int(files) if isinstance(files, int) else None
        status = result.get("status")
        if status == "passed" and self.files_committed >= 1:
            return FinalCommitCheck(confirmed=True, files_compared=compared)
        reason = (
            "NO_COMMITTED_FILES"
            if status == "passed"
            else str(result.get("error_code") or status or "unavailable")
        )
        return FinalCommitCheck(confirmed=False, reason=reason, files_compared=compared)

    def _answered_receipt(self) -> RunFilesReceipt | None:
        """The emitter's current receipt for the run, or None if it did not answer."""
        if self.emitter is None:
            return None
        try:
            receipt = self.emitter.run_files_receipt(self.repository, self.run_id)
        except Exception:
            return None
        return receipt if self._record(receipt) else None

    def _confirm_by_receipt(self, wait_seconds: float) -> FinalCommitCheck:
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        # Only receipts the emitter gives now count: the last hand-over's
        # answer would show its files still pending until the wait ran out.
        receipt = self._answered_receipt()
        while (
            receipt is not None
            and receipt["pending_files"]
            and receipt["last_error_code"] not in _PERMANENT_COMMIT_ERRORS
            and time.monotonic() < deadline
        ):
            time.sleep(_RECEIPT_POLL_SECONDS)
            receipt = self._answered_receipt()
        if receipt is None:
            return FinalCommitCheck(
                confirmed=False, reason="unavailable", method="receipt"
            )
        if receipt["pending_files"]:
            return FinalCommitCheck(
                confirmed=False,
                reason=receipt["last_error_code"] or "pending",
                method="receipt",
            )
        if self.files_committed < 1:
            return FinalCommitCheck(
                confirmed=False, reason="NO_COMMITTED_FILES", method="receipt"
            )
        return FinalCommitCheck(confirmed=True, method="receipt")

    def reopen(self) -> None:
        """Hand over again after a pause, for the run's final state."""
        self.consecutive_failures = 0
        self.enabled = True


def warn_hand_over_failure(repository: str, result: HandOverResult) -> None:
    """Report a hand-over the emitter did not accept, without exception text."""
    print(
        "warning: staging hand-over failed: the local emitter did not accept "
        f"the run files for {repository} ({result.reason}); local staging "
        "files are still written.",
        file=sys.stderr,
    )


def warn_final_commit_unconfirmed(repository: str, check: FinalCommitCheck) -> None:
    """Report a final commit the emitter could not confirm, without failing."""
    print(
        "warning: the staging run's final commit to "
        f"{repository} was not confirmed ({check.reason}); the local staging "
        "files are complete and the build's outcome is unchanged.",
        file=sys.stderr,
    )


def _interrupted(error: BaseException | None) -> bool:
    """Whether ``error`` is, or was caused by, Ctrl-C or a SIGTERM."""
    seen: set[int] = set()
    link = error
    while link is not None and id(link) not in seen:
        # BuildTerminatedError, a SIGTERM, subclasses KeyboardInterrupt.
        if isinstance(link, KeyboardInterrupt):
            return True
        seen.add(id(link))
        link = link.__cause__ or link.__context__
    return False


class CheckTelemetry(Protocol):
    """The event call of the build's hosted telemetry (``LocalTelemetryEmitter``)."""

    def emit(
        self,
        *,
        event_type: Any,
        status: Any,
        stage_id: str | None = ...,
        message: str | None = ...,
        details: Mapping[str, Any] | None = ...,
    ) -> None: ...


class FinalCommitWriter(Protocol):
    """A staging writer whose closed run can confirm its final commit."""

    @property
    def remote_staging(self) -> bool: ...

    def confirm_final_commit(
        self, *, wait_seconds: float, read_back: bool = ...
    ) -> FinalCommitCheck | None: ...


def report_final_commit(
    telemetry: CheckTelemetry | None, check: FinalCommitCheck
) -> None:
    """Tell the build's hosted telemetry what the final-commit check found.

    One progress event outside any stage, so it neither ends the build's
    current stage nor becomes the stage a later failure is attributed to.
    The report never changes how the build closes.
    """
    if telemetry is None:
        return
    try:
        telemetry.emit(
            event_type="progress",
            status="completed" if check.confirmed else "failed",
            message=(
                "The staging repository holds the run's final state."
                if check.confirmed
                else "The staging run's final commit was not confirmed."
            ),
            details=check.details(),
        )
    except Exception:
        pass


def confirm_and_report_final_commit(
    writer: FinalCommitWriter,
    telemetry: CheckTelemetry | None,
    *,
    error: BaseException | None = None,
) -> FinalCommitCheck | None:
    """Confirm a closed run's final commit and report it to the build's telemetry.

    Call it after the run's final state is handed over and before the run's
    outcome reaches the emitter, which drops a run's events after its
    outcome. ``error`` is what ended the build, if anything: after an
    interrupt the check reads receipts for up to 30 seconds, otherwise it
    reads the run back, waiting up to 120 seconds. A run without remote
    staging has nothing to confirm. A check that fails, or that an interrupt
    ends, warns and is reported; it never raises, so the close-out goes on
    and the build keeps its own outcome.
    """
    if not writer.remote_staging:
        return None
    interrupted = _interrupted(error)
    method = "receipt" if interrupted else "read_back"
    try:
        check = writer.confirm_final_commit(
            wait_seconds=(
                INTERRUPTED_FINAL_COMMIT_WAIT_SECONDS
                if interrupted
                else FINAL_COMMIT_WAIT_SECONDS
            ),
            read_back=not interrupted,
        )
    except KeyboardInterrupt:
        print(
            "warning: the staging run's final commit check was interrupted, so "
            "the commit was not confirmed; the build's outcome is unchanged.",
            file=sys.stderr,
        )
        check = FinalCommitCheck(confirmed=False, reason="interrupted", method=method)
    except Exception as failure:
        print(
            "warning: the staging run's final commit could not be checked "
            f"({type(failure).__name__}), so it was not confirmed; the "
            "build's outcome is unchanged.",
            file=sys.stderr,
        )
        check = FinalCommitCheck(
            confirmed=False, reason=type(failure).__name__, method=method
        )
    if check is None:
        return None
    report_final_commit(telemetry, check)
    return check
