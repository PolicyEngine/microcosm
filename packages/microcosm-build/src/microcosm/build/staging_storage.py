"""Country-neutral configuration, remote storage and run-file hand-over.

Staging run files (``runs/<run_id>/...``) go to the build's local emitter,
which commits them to the dataset repository with the build's own Hugging
Face credential (:class:`RunFileHandOver`). :class:`HuggingFaceDatasetStorage`
reads and writes the repository directly for the staged dataset bundle and
the tools that fetch it.
"""

from __future__ import annotations

import math
import sys
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
    "HandOverResult",
    "HuggingFaceDatasetStorage",
    "RunFileEmitter",
    "RunFileHandOver",
    "StagingRepositoryConfig",
    "warn_hand_over_failure",
]


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
    """Read and write files in one Hugging Face dataset repository."""

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

    def upload(self, local_path: Path, path_in_repo: str) -> None:
        self._api().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def download(self, path_in_repo: str) -> bytes:
        api = self._api()
        download = getattr(api, "hf_hub_download", None)
        if download is None:
            from huggingface_hub import hf_hub_download as download

        local = download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            force_download=True,
        )
        return Path(local).read_bytes()

    def download_file(self, path_in_repo: str, *, revision: str | None = None) -> Path:
        """Fetch one file through the Hub cache and return its local path.

        Unlike :meth:`download` this never reads the bytes into memory, so it
        serves population files as well as small documents.
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

    def file_exists(self, path_in_repo: str) -> bool:
        return bool(
            self._api().file_exists(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset",
            )
        )

    def _access_token(self) -> Mapping[str, Any] | None:
        whoami = getattr(self._api(), "whoami", None)
        if not callable(whoami):
            return None
        info = whoami()
        auth = info.get("auth") if isinstance(info, Mapping) else None
        token = auth.get("accessToken") if isinstance(auth, Mapping) else None
        return token if isinstance(token, Mapping) else None

    def credential_role(self) -> str | None:
        """The ambient credential's role as the Hub reports it.

        ``read`` and ``write`` are the classic roles; a fine-grained token
        reports ``fineGrained`` and carries its scopes separately (see
        :meth:`credential_can_write`). ``None`` when the backend cannot say.
        """

        token = self._access_token()
        role = token.get("role") if token else None
        return str(role) if role else None

    def credential_can_write(self) -> bool | None:
        """Whether the ambient credential may write this repository.

        ``True`` for a classic write token or a fine-grained token whose
        scopes grant ``repo.write`` on the repository, its owner (user or
        organisation) or globally; ``False`` for a read token or a fine-grained
        token scoped elsewhere; ``None`` when the backend cannot say, in which
        case only the upload itself proves the scope.
        """

        token = self._access_token()
        if token is None:
            return None
        role = str(token.get("role") or "")
        if role == "read":
            return False
        if role == "write":
            return True
        if role != "fineGrained":
            return None
        fine = token.get("fineGrained")
        fine = fine if isinstance(fine, Mapping) else {}
        if "repo.write" in (fine.get("global") or []):
            return True
        owner = self.repo_id.split("/", 1)[0]
        for scope in fine.get("scoped") or []:
            if not isinstance(scope, Mapping):
                continue
            entity = scope.get("entity")
            entity = entity if isinstance(entity, Mapping) else {}
            name = str(entity.get("name") or "")
            kind = str(entity.get("type") or "")
            covers = name == self.repo_id or (kind in {"user", "org"} and name == owner)
            if covers and "repo.write" in (scope.get("permissions") or []):
                return True
        return False

    def last_commit(self, path_in_repo: str) -> str | None:
        """The commit that last changed ``path_in_repo``, when the backend reports it."""

        paths_info = getattr(self._api(), "get_paths_info", None)
        if not callable(paths_info):
            return None
        entries = paths_info(
            repo_id=self.repo_id,
            paths=[path_in_repo],
            expand=True,
            repo_type="dataset",
        )
        for entry in entries or []:
            commit = (
                entry.get("last_commit")
                if isinstance(entry, Mapping)
                else getattr(entry, "last_commit", None)
            )
            oid = (
                commit.get("oid")
                if isinstance(commit, Mapping)
                else getattr(commit, "oid", None)
            )
            if oid:
                return str(oid)
        return None

    def head_revision(self) -> str | None:
        """The default branch's current commit, when the backend reports one."""

        info = self._api().repo_info(repo_id=self.repo_id, repo_type="dataset")
        sha = (
            info.get("sha") if isinstance(info, Mapping) else getattr(info, "sha", None)
        )
        return str(sha) if sha else None

    def commit(
        self,
        operations: Sequence[Any],
        *,
        message: str,
        parent_commit: str | None = None,
    ) -> str:
        """Write several files in one commit and return the new revision.

        ``operations`` are ``huggingface_hub.CommitOperation`` values built by
        the caller; pinning ``parent_commit`` makes a concurrent write fail
        instead of silently interleaving.
        """

        info = self._api().create_commit(
            repo_id=self.repo_id,
            operations=list(operations),
            commit_message=message,
            repo_type="dataset",
            parent_commit=parent_commit,
        )
        oid = (
            info.get("oid") if isinstance(info, Mapping) else getattr(info, "oid", None)
        )
        if not isinstance(oid, str) or not oid.strip():
            raise RuntimeError("The Hub commit reported no revision.")
        return oid.strip()


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


class RunFileHandOver:
    """Hand one staging run's files to the build's local emitter.

    The emitter copies the files' bytes before it answers, so the writer may
    rewrite them at once. It writes every pending path of the repository in
    one Hugging Face commit per cycle: an unforced hand-over one
    ``commit_interval_seconds`` after its last commit to the repository, and
    one with ``flush`` at the next cycle. Its receipts say how many files it
    attempted to commit and how many it committed.

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
        commit_interval_seconds: float = 30.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive.")
        self.emitter = emitter
        self.repository = repository
        self.run_id = run_id
        self.prefix = prefix
        # The request needs a finite interval. A writer with an infinite one
        # hands over only forced states, which the emitter commits at once.
        interval = float(commit_interval_seconds)
        self.commit_interval_seconds = (
            max(0.0, interval) if math.isfinite(interval) else 0.0
        )
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

    def hand_over(
        self,
        files: Sequence[tuple[str, Path]],
        *,
        latest_pointer: Path | None = None,
        run_index_entry: Mapping[str, Any] | None = None,
        flush: bool = False,
    ) -> HandOverResult:
        """Hand over ``(relative path, local path)`` pairs under the run's folder."""
        offered = len(files) + (latest_pointer is not None)
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
                latest_pointer=latest_pointer,
                run_index_entry=run_index_entry,
                flush=flush,
                commit_interval_seconds=self.commit_interval_seconds,
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
