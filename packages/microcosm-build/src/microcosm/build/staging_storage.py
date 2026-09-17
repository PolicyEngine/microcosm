"""Country-neutral configuration and remote storage for staging data."""

from __future__ import annotations

from collections.abc import Mapping
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
            became_disabled = self.consecutive_failures >= self.max_consecutive_failures
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
