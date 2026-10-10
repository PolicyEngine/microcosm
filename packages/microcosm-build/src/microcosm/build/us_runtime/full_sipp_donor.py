"""Shared verification cache for the immutable full 2023 SIPP donor.

Several US build stages read different column subsets from the same 3.73 GB
file.  Each stage still supplies its own expected SHA-256 so its pinned-source
and transform-audit contracts remain explicit, while this module ensures that
unchanged bytes are scanned only once per process.

The cache key is the file's cheap filesystem fingerprint. Replacing or
mutating a path forces a fresh hash. Verification and parsing share one open
descriptor, whose identity is checked alongside the path before and after
either operation. A downloader's digest is never trusted without verifying
the current bytes.
"""

from __future__ import annotations

import hashlib
import os
import stat as stat_module
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO

__all__ = [
    "FullSIPPDonorMutationError",
    "FullSIPPVerifiedFile",
    "FullSIPPVerifiedFingerprint",
    "cache_verified_full_sipp_sha256",
    "clear_full_sipp_sha256_cache",
    "full_sipp_stream_fingerprint",
    "full_sipp_sha256",
    "open_verified_full_sipp",
]

_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class FullSIPPVerifiedFingerprint:
    """Content identity captured while verified download bytes remain open."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class FullSIPPVerifiedFile:
    """An open full-SIPP stream whose current identity owns ``sha256``."""

    stream: BinaryIO
    sha256: str
    fingerprint: _FileFingerprint


_SHA256_BY_FINGERPRINT: dict[_FileFingerprint, str] = {}
_CACHE_LOCK = RLock()


class FullSIPPDonorMutationError(RuntimeError):
    """Raised when a full-SIPP file changes while its SHA-256 is computed."""


def _fingerprint(path: Path) -> _FileFingerprint:
    stat_result = path.stat()
    if not stat_module.S_ISREG(stat_result.st_mode):
        raise FileNotFoundError(path)
    return _fingerprint_from_stat(stat_result)


def _fingerprint_from_stat(stat_result: os.stat_result) -> _FileFingerprint:
    return _FileFingerprint(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        size_bytes=stat_result.st_size,
        modified_ns=stat_result.st_mtime_ns,
        changed_ns=stat_result.st_ctime_ns,
    )


def _stream_fingerprint(stream: BinaryIO) -> _FileFingerprint:
    return _fingerprint_from_stat(os.fstat(stream.fileno()))


def _verified_fingerprint(
    fingerprint: _FileFingerprint,
) -> FullSIPPVerifiedFingerprint:
    return FullSIPPVerifiedFingerprint(
        device=fingerprint.device,
        inode=fingerprint.inode,
        size_bytes=fingerprint.size_bytes,
        modified_ns=fingerprint.modified_ns,
    )


def full_sipp_stream_fingerprint(stream: BinaryIO) -> FullSIPPVerifiedFingerprint:
    """Capture the content identity of an open, flushed download stream."""

    return _verified_fingerprint(_stream_fingerprint(stream))


def _hash_stream_contents(stream: BinaryIO, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _mutation_error() -> FullSIPPDonorMutationError:
    return FullSIPPDonorMutationError(
        "Full SIPP donor changed during SHA-256 verification; refusing "
        "to cache a digest not bound to stable bytes."
    )


def _initial_open_fingerprint(path: Path, stream: BinaryIO) -> _FileFingerprint:
    descriptor = _stream_fingerprint(stream)
    try:
        current_path = _fingerprint(path)
    except OSError as exc:
        raise _mutation_error() from exc
    if current_path != descriptor:
        raise _mutation_error()
    return descriptor


def _assert_open_identity(
    path: Path,
    stream: BinaryIO,
    expected: _FileFingerprint,
) -> None:
    try:
        descriptor = _stream_fingerprint(stream)
        current_path = _fingerprint(path)
    except OSError as exc:
        raise _mutation_error() from exc
    if descriptor != expected or current_path != expected:
        raise _mutation_error()


def full_sipp_sha256(
    path: str | Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> str:
    """Return a mutation-aware, process-cached SHA-256 for a full-SIPP file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    source_path = Path(path).expanduser()
    with source_path.open("rb") as verified_stream, _CACHE_LOCK:
        initial = _initial_open_fingerprint(source_path, verified_stream)
        cached = _SHA256_BY_FINGERPRINT.get(initial)
        if cached is not None:
            try:
                _assert_open_identity(source_path, verified_stream, initial)
            except FullSIPPDonorMutationError:
                _SHA256_BY_FINGERPRINT.pop(initial, None)
                raise
            return cached
        digest = _hash_stream_contents(verified_stream, chunk_size=chunk_size)
        try:
            _assert_open_identity(source_path, verified_stream, initial)
        except FullSIPPDonorMutationError:
            _SHA256_BY_FINGERPRINT.pop(initial, None)
            raise
        _SHA256_BY_FINGERPRINT[initial] = digest
        return digest


@contextmanager
def open_verified_full_sipp(
    path: str | Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Iterator[FullSIPPVerifiedFile]:
    """Yield the exact open bytes hashed for a full-SIPP parser.

    The descriptor remains open from fingerprinting through parsing. Both its
    identity and the path identity are checked before and after the caller
    consumes it, so replacement or in-place mutation cannot silently separate
    the cached digest from the bytes passed to pandas.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    source_path = Path(path).expanduser()
    with source_path.open("rb") as stream:
        initial = _initial_open_fingerprint(source_path, stream)
        with _CACHE_LOCK:
            digest = _SHA256_BY_FINGERPRINT.get(initial)
            if digest is None:
                digest = _hash_stream_contents(stream, chunk_size=chunk_size)
            try:
                _assert_open_identity(source_path, stream, initial)
            except FullSIPPDonorMutationError:
                _SHA256_BY_FINGERPRINT.pop(initial, None)
                raise
            _SHA256_BY_FINGERPRINT[initial] = digest
        stream.seek(0)
        try:
            yield FullSIPPVerifiedFile(
                stream=stream,
                sha256=digest,
                fingerprint=initial,
            )
        finally:
            try:
                _assert_open_identity(source_path, stream, initial)
            except FullSIPPDonorMutationError:
                with _CACHE_LOCK:
                    _SHA256_BY_FINGERPRINT.pop(initial, None)
                raise


def cache_verified_full_sipp_sha256(
    path: str | Path,
    sha256: str,
    *,
    verified_fingerprint: FullSIPPVerifiedFingerprint | None = None,
) -> None:
    """Seed the cache after a streaming download verified these exact bytes.

    A download fingerprint provides an additional identity check after its
    atomic rename. The target bytes are still verified against the supplied
    digest, so a replacement after download verification cannot seed an old
    digest under a new identity. This can add one scan after a fresh download;
    subsequent consumers reuse the verified cache.
    """

    normalized = str(sha256).lower()
    if len(normalized) != 64:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError("sha256 must contain only hexadecimal characters") from exc
    source_path = Path(path).expanduser()
    if verified_fingerprint is not None:
        if _verified_fingerprint(_fingerprint(source_path)) != verified_fingerprint:
            raise _mutation_error()
    actual = full_sipp_sha256(source_path)
    if actual != normalized:
        raise _mutation_error()


def clear_full_sipp_sha256_cache() -> None:
    """Clear cached identities (a deterministic seam for focused tests)."""

    with _CACHE_LOCK:
        _SHA256_BY_FINGERPRINT.clear()
