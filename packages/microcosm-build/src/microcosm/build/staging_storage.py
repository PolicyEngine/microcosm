"""Country-neutral configuration and remote storage for staging data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BestEffortUploadSession",
    "HuggingFaceDatasetStorage",
    "StagingRepositoryConfig",
    "UploadResult",
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
            covers = name == self.repo_id or (
                kind in {"user", "org"} and name == owner
            )
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


@dataclass(frozen=True)
class UploadResult:
    """Outcome of one best-effort remote write."""

    attempted: bool
    succeeded: bool
    disabled: bool
    became_disabled: bool
    error: Exception | None = None


class BestEffortUploadSession:
    """Apply common retry-stop accounting without selecting a file contract."""

    def __init__(
        self,
        storage: HuggingFaceDatasetStorage,
        *,
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive.")
        self.storage = storage
        self.max_consecutive_failures = max_consecutive_failures
        self.attempts = 0
        self.successes = 0
        self.consecutive_failures = 0
        self.enabled = True

    def upload(
        self,
        local_path: Path,
        path_in_repo: str,
        *,
        count: bool = True,
    ) -> UploadResult:
        """Attempt one write and stop the session after repeated failures."""

        if not self.enabled:
            return UploadResult(
                attempted=False,
                succeeded=False,
                disabled=True,
                became_disabled=False,
            )
        if count:
            self.attempts += 1
        try:
            self.storage.upload(local_path, path_in_repo)
        except Exception as error:
            self.consecutive_failures += 1
            became_disabled = (
                self.consecutive_failures >= self.max_consecutive_failures
            )
            if became_disabled:
                self.enabled = False
            return UploadResult(
                attempted=True,
                succeeded=False,
                disabled=not self.enabled,
                became_disabled=became_disabled,
                error=error,
            )
        self.consecutive_failures = 0
        if count:
            self.successes += 1
        return UploadResult(
            attempted=True,
            succeeded=True,
            disabled=False,
            became_disabled=False,
        )
