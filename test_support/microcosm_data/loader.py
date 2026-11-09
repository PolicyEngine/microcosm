"""Release-native loader resolution, verification, and compatibility checks."""

# ruff: noqa: F401

from __future__ import annotations

import hashlib
import json
import os
import re
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import microcosm.data.loader as loader
from microcosm.data import DEFAULT_VARIANT, download, latest_year, load, resolve
from microcosm.data.release import LATEST_POINTER_PATH, latest_pointer_payload

#: Release id served by the hub fixture. Any well-formed id works: nothing in
#: this suite may assert which release the live Hub currently points at,
#: because re-certification moves ``latest.json`` without a commit here.
RELEASE_ID = "populace-us-2024-buildi-sparse-rmloss100-6e8e929-20260709T034135Z"
TAGGED_ARTIFACT = b"certified release artifact"
MUTABLE_ROOT_ARTIFACT = b"uncertified in-flight root artifact"


def _release_manifest(
    *,
    artifact_content: bytes = TAGGED_ARTIFACT,
    release_id: str = RELEASE_ID,
) -> dict:
    return {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.26.11",
            },
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": "1.764.6",
            },
        },
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": "==3.26.11"}
        ],
        "compatible_model_packages": [
            {"name": "policyengine-us", "specifier": "==1.764.6"}
        ],
        "artifacts": {
            "populace_us_2024": {
                "kind": "microdata",
                "path": "populace_us_2024.h5",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": hashlib.sha256(artifact_content).hexdigest(),
            }
        },
    }


class FakeHubDownload:
    """Serve mutable and release-tagged files while recording every request."""

    def __init__(self, tmp_path: Path, *, manifest: dict | None = None) -> None:
        pointer = latest_pointer_payload(
            RELEASE_ID, updated_at="2026-07-09T04:00:00+00:00"
        )
        manifest = manifest or _release_manifest()
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, str, str | None]] = []
        self.files: dict[tuple[str, str, str | None], Path] = {}
        self._add(
            tmp_path,
            "policyengine/populace-us",
            LATEST_POINTER_PATH,
            None,
            json.dumps(pointer).encode(),
        )
        self._add(
            tmp_path,
            "policyengine/populace-us",
            pointer["paths"]["release_manifest"],
            RELEASE_ID,
            json.dumps(manifest).encode(),
        )
        self._add(
            tmp_path,
            "policyengine/populace-us",
            "populace_us_2024.h5",
            RELEASE_ID,
            TAGGED_ARTIFACT,
        )
        self._add(
            tmp_path,
            "policyengine/populace-us",
            "populace_us_2024.h5",
            None,
            MUTABLE_ROOT_ARTIFACT,
        )

    def _add(
        self,
        tmp_path: Path,
        repo_id: str,
        filename: str,
        revision: str | None,
        content: bytes,
    ) -> None:
        revision_dir = revision or "main"
        local = tmp_path / "hub" / revision_dir / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        self.files[(repo_id, filename, revision)] = local

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str | None = None,
    ) -> str:
        assert repo_type == "dataset"
        self.calls.append((repo_id, filename, revision))
        try:
            return str(self.files[(repo_id, filename, revision)])
        except KeyError as exc:
            raise FileNotFoundError(
                f"{repo_id}/{filename} at revision {revision!r}"
            ) from exc


def _patch_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_version: str = "1.764.6",
    core_version: str = "3.26.11",
) -> list[str]:
    constructed: list[str] = []

    class FakeDataset:
        def __init__(self, *, file_path: str) -> None:
            constructed.append(file_path)
            self.file_path = file_path

    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda _name: SimpleNamespace(USSingleYearDataset=FakeDataset),
    )
    versions = {
        "policyengine-us": model_version,
        "policyengine-core": core_version,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    return constructed


def _engine_available() -> bool:
    try:
        import policyengine_us  # noqa: F401
    except ImportError:
        return False
    return True


def _hf_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}


def _is_offline_error(exc: Exception) -> bool:
    """Recognize transport/offline failures without hiding bad release metadata."""
    offline_names = {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "ProxyError",
        "ReadTimeout",
    }
    closed_after_retry = isinstance(exc, RuntimeError) and str(exc) == (
        "Cannot send a request, as the client has been closed."
    )
    return (
        isinstance(exc, (ConnectionError, TimeoutError))
        or closed_after_retry
        or any(cls.__name__ in offline_names for cls in type(exc).__mro__)
    )


__all__ = [name for name in globals() if not name.startswith("__")]
