"""Shared verification cache for the immutable full 2023 SIPP donor.

Several US build stages read different column subsets from the same 3.73 GB
file.  Each stage still supplies its own expected SHA-256 so its pinned-source
and transform-audit contracts remain explicit, while this module ensures that
unchanged bytes are scanned only once per process.

The cache key is the file's cheap filesystem fingerprint.  Replacing or
mutating a path changes that fingerprint and forces a fresh hash; a mutation
during hashing is rejected rather than cached.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

__all__ = [
    "FullSIPPDonorMutationError",
    "cache_verified_full_sipp_sha256",
    "clear_full_sipp_sha256_cache",
    "full_sipp_sha256",
]

_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


_SHA256_BY_FINGERPRINT: dict[_FileFingerprint, str] = {}
_CACHE_LOCK = Lock()


class FullSIPPDonorMutationError(RuntimeError):
    """Raised when a full-SIPP file changes while its SHA-256 is computed."""


def _fingerprint(path: Path) -> _FileFingerprint:
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return _FileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _hash_file_contents(path: Path, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_sipp_sha256(
    path: str | Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> str:
    """Return a mutation-aware, process-cached SHA-256 for a full-SIPP file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    source_path = Path(path).expanduser()
    initial = _fingerprint(source_path)
    with _CACHE_LOCK:
        cached = _SHA256_BY_FINGERPRINT.get(initial)
        if cached is not None:
            return cached
        digest = _hash_file_contents(source_path, chunk_size=chunk_size)
        final = _fingerprint(source_path)
        if final != initial:
            raise FullSIPPDonorMutationError(
                "Full SIPP donor changed during SHA-256 verification; refusing "
                "to cache a digest not bound to stable bytes."
            )
        _SHA256_BY_FINGERPRINT[initial] = digest
        return digest


def cache_verified_full_sipp_sha256(
    path: str | Path,
    sha256: str,
) -> None:
    """Seed the cache after a streaming download verified these exact bytes."""

    normalized = str(sha256).lower()
    if len(normalized) != 64:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError("sha256 must contain only hexadecimal characters") from exc
    source_path = Path(path).expanduser()
    fingerprint = _fingerprint(source_path)
    with _CACHE_LOCK:
        _SHA256_BY_FINGERPRINT[fingerprint] = normalized


def clear_full_sipp_sha256_cache() -> None:
    """Clear cached identities (a deterministic seam for focused tests)."""

    with _CACHE_LOCK:
        _SHA256_BY_FINGERPRINT.clear()
