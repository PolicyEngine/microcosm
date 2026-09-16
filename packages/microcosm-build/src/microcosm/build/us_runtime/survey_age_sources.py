"""Two explicit S0101-only source captures; no network or target activation.

Callers supply already acquired response bytes. Reading an explicit capture
never discovers files or falls back to a mixed national reference inventory.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from urllib.parse import urlencode

from microcosm.graph.canonical import canonical_json

from .cd_reference_sources import strict_json

MAX_RESPONSE_BYTES = 8 * 1024**2
MAX_DESCRIPTOR_BYTES = 4096
_DESCRIPTOR_FIELDS = frozenset(
    {
        "table",
        "year",
        "dataset",
        "geography",
        "kind",
        "url",
        "sha256",
        "size_bytes",
        "path",
    }
)


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise ValueError("SURVEY_AGE_SOURCE_" + reason)


def survey_age_requests() -> tuple[dict[str, object], dict[str, object]]:
    """Return detached descriptors of exactly two supported national requests."""
    base = "https://api.census.gov/data/2024/acs/acs1/subject"
    common = {
        "table": "S0101",
        "year": 2024,
        "dataset": "acs/acs1/subject",
        "geography": "us:*",
    }
    return (
        {**common, "kind": "metadata", "url": f"{base}/groups/S0101.json"},
        {
            **common,
            "kind": "data",
            "url": base + "?" + urlencode({"get": "group(S0101)", "for": "us:*"}),
        },
    )


def _request(value: object) -> dict[str, object]:
    _require(type(value) is dict, "REQUEST")
    for request in survey_age_requests():
        if set(value) == set(request) and all(
            type(value[k]) is type(v) and value[k] == v for k, v in request.items()
        ):
            return request
    _require(False, "REQUEST")


def _digest(value: object) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "PIN",
    )
    return value


def _path(value: str | Path) -> Path:
    path = Path(value)
    _require(".." not in path.parts, "PATH")
    path = path.absolute()
    for member in (*reversed(path.parents), path):
        _require(not member.is_symlink(), "SYMLINK")
    return path


def _read(path: Path, limit: int) -> bytes:
    path = _path(path)
    try:
        # A FIFO must reach the regular-file check instead of waiting for a
        # writer before fstat can reject it. Nonblocking leaves regular reads unchanged.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), "REGULAR_FILE")
            _require(0 < before.st_size <= limit, "SIZE")
            payload = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            _require(
                len(payload) == before.st_size
                and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "CHANGED_FILE",
            )
            return payload
    except OSError:
        raise ValueError("SURVEY_AGE_SOURCE_FILE") from None


def _write(path: Path, payload: bytes, limit: int) -> None:
    path = _path(path)
    if path.exists():
        _require(_read(path, limit) == payload, "IMMUTABLE")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _path(path)
    with path.open("xb") as stream:
        stream.write(payload)


def capture_survey_age_response(
    root: str | Path, request: dict, payload: bytes
) -> dict:
    """Store one caller-acquired response immutably; grant no target permission."""
    request = _request(request)
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_RESPONSE_BYTES, "SIZE")
    strict_json(payload)
    root = _path(root)
    digest = hashlib.sha256(payload).hexdigest()
    descriptor = {
        **request,
        "sha256": digest,
        "size_bytes": len(payload),
        "path": f"raw/{digest}.json",
    }
    key = hashlib.sha256(request["url"].encode()).hexdigest()
    descriptor_path = root / "requests" / f"{key}.json"
    encoded = canonical_json(descriptor)
    # Refuse replacement before writing even an unreferenced new raw file.
    if _path(descriptor_path).exists():
        _require(_read(descriptor_path, MAX_DESCRIPTOR_BYTES) == encoded, "IMMUTABLE")
    _write(root / descriptor["path"], payload, MAX_RESPONSE_BYTES)
    _write(descriptor_path, encoded, MAX_DESCRIPTOR_BYTES)
    return descriptor


def read_survey_age_response(
    root: str | Path, kind: str, expected_sha256: str
) -> tuple[bytes, object]:
    """Read the fixed named response against a separately reviewed content pin."""
    _require(type(kind) is str and kind in {"metadata", "data"}, "KIND")
    expected_sha256 = _digest(expected_sha256)
    request = next(row for row in survey_age_requests() if row["kind"] == kind)
    root = _path(root)
    key = hashlib.sha256(request["url"].encode()).hexdigest()
    descriptor = strict_json(
        _read(root / "requests" / f"{key}.json", MAX_DESCRIPTOR_BYTES)
    )
    _require(
        type(descriptor) is dict and set(descriptor) == _DESCRIPTOR_FIELDS, "DESCRIPTOR"
    )
    _request({key: descriptor[key] for key in request})
    _require(
        descriptor["sha256"] == expected_sha256
        and descriptor["path"] == f"raw/{expected_sha256}.json",
        "PIN",
    )
    _require(
        type(descriptor["size_bytes"]) is int
        and 0 < descriptor["size_bytes"] <= MAX_RESPONSE_BYTES,
        "SIZE",
    )
    payload = _read(root / f"raw/{expected_sha256}.json", MAX_RESPONSE_BYTES)
    _require(
        len(payload) == descriptor["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == expected_sha256,
        "RESPONSE_PIN",
    )
    return payload, strict_json(payload)
