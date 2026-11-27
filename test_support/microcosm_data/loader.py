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
    """Recognize transport/offline failures without hiding bad release metadata.

    The check applies to ``exc`` and to each exception in its ``__cause__``
    chain, each once: offline, the loader refuses a ``releases.json`` never
    cached with a ``ValueError`` whose cause is the Hub client's error.
    """
    offline_names = {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "ProxyError",
        "ReadTimeout",
    }
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        closed_after_retry = isinstance(current, RuntimeError) and str(current) == (
            "Cannot send a request, as the client has been closed."
        )
        if (
            isinstance(current, (ConnectionError, TimeoutError))
            or closed_after_retry
            or any(cls.__name__ in offline_names for cls in type(current).__mro__)
        ):
            return True
        current = current.__cause__
    return False


__all__ = [name for name in globals() if not name.startswith("__")]
