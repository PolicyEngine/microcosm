"""Default-off cache helpers for fitted conditional models."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QRFCacheKey:
    """Stable metadata identity for a fitted QRF model."""

    donor_sha256: str
    predictors: tuple[str, ...]
    targets: tuple[str, ...]
    seed: int
    weight_kind: str
    package_versions: dict[str, str]
    extra: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "donor_sha256": self.donor_sha256,
            "predictors": list(self.predictors),
            "targets": list(self.targets),
            "seed": int(self.seed),
            "weight_kind": self.weight_kind,
            "package_versions": dict(sorted(self.package_versions.items())),
            "extra": {} if self.extra is None else self.extra,
        }

    def digest(self) -> str:
        return _digest_payload(self.payload())


def cache_path(cache_dir: str | Path, key: QRFCacheKey) -> Path:
    """Return the model path for ``key`` under an explicit cache directory."""

    return Path(cache_dir).expanduser().resolve() / f"{key.digest()}.pkl"


def save_fitted_qrf_cache(
    model: object,
    cache_dir: str | Path,
    key: QRFCacheKey,
) -> Path:
    """Save a fitted model under ``key`` and return the written path."""

    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": key.payload(),
        "metadata_digest": key.digest(),
        "model": model,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_fitted_qrf_cache(cache_dir: str | Path, key: QRFCacheKey) -> object | None:
    """Load the cached model for ``key``, returning ``None`` on a miss."""

    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"QRF cache file {path} does not contain a payload object.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"QRF cache file {path} has unsupported schema version.")
    if payload.get("metadata_digest") != key.digest():
        return None
    if payload.get("metadata") != key.payload():
        return None
    return payload.get("model")


def _digest_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
