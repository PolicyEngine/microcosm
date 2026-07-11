"""Hermetic source pins and streaming acquisition for ACS PUMS.

The packaged manifest identifies the two exact Census archives used by the
``acs_2024_1yr`` spine.  Acquisition never trusts an existing cache by name:
both its byte count and SHA-256 digest must match the manifest. Fresh downloads
are streamed through unique same-directory ``.partial`` files, verified from
the closed temporary bytes, and moved into content-addressed cache directories
atomically, so interrupted, concurrent, or changed upstream payloads cannot be
consumed by the loader.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Literal

from populace.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_2024_1YR_VINTAGE,
    AcsPumsSource,
)

__all__ = [
    "AcsSourceArtifact",
    "AcsSourceManifest",
    "fetch_acs_pums_sources",
    "load_acs_source_manifest",
]

_MANIFEST_FILENAME = "acs_2024_1yr_sources.json"
_SOURCE_DIRECTORY = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/"
)
_EXPECTED_FILES = {
    "household": "csv_hus.zip",
    "person": "csv_pus.zip",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HTTP_USER_AGENT = "PolicyEngine-Populace/0.1 (ACS PUMS source fetch)"
_DEFAULT_CHUNK_BYTES = 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 60.0

ArtifactRole = Literal["household", "person"]
_Response = AbstractContextManager[BinaryIO]
_Opener = Callable[[urllib.request.Request], _Response]


@dataclass(frozen=True)
class AcsSourceArtifact:
    """One byte-pinned archive in the ACS source manifest."""

    role: ArtifactRole
    filename: str
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AcsSourceManifest:
    """Versioned declaration of every source artifact for one ACS spine."""

    version: int
    spine: str
    vintage: int
    verified_on: str
    source_directory: str
    artifacts: tuple[AcsSourceArtifact, ...]

    def artifact(self, role: ArtifactRole) -> AcsSourceArtifact:
        """Return the unique artifact assigned to ``role``."""

        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        raise KeyError(role)


def load_acs_source_manifest(path: str | Path | None = None) -> AcsSourceManifest:
    """Load and strictly validate the packaged ACS source declaration."""

    manifest_path = (
        Path(path) if path is not None else Path(__file__).with_name(_MANIFEST_FILENAME)
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ACS source manifest must be a JSON object.")

    required = {
        "version",
        "spine",
        "vintage",
        "verified_on",
        "source_directory",
        "artifacts",
    }
    missing = sorted(required - raw.keys())
    extra = sorted(raw.keys() - required)
    if missing or extra:
        raise ValueError(
            f"ACS source manifest fields differ: missing={missing}, extra={extra}."
        )
    raw_artifacts = raw["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ValueError("ACS source manifest artifacts must be a JSON list.")
    artifacts: list[AcsSourceArtifact] = []
    artifact_fields = {"role", "filename", "url", "sha256", "size_bytes"}
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, dict):
            raise ValueError(f"ACS source artifact {index} must be a JSON object.")
        item_missing = sorted(artifact_fields - item.keys())
        item_extra = sorted(item.keys() - artifact_fields)
        if item_missing or item_extra:
            raise ValueError(
                f"ACS source artifact {index} fields differ: "
                f"missing={item_missing}, extra={item_extra}."
            )
        artifacts.append(AcsSourceArtifact(**item))

    manifest = AcsSourceManifest(
        version=raw["version"],
        spine=raw["spine"],
        vintage=raw["vintage"],
        verified_on=raw["verified_on"],
        source_directory=raw["source_directory"],
        artifacts=tuple(artifacts),
    )
    _validate_manifest(manifest)
    return manifest


def fetch_acs_pums_sources(
    cache_dir: str | Path,
    *,
    manifest: AcsSourceManifest | None = None,
    opener: _Opener | None = None,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> AcsPumsSource:
    """Stream, verify, and cache both archives declared by ``manifest``."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    source_manifest = manifest or load_acs_source_manifest()
    _validate_manifest(source_manifest)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if opener is None:

        def open_url(request: urllib.request.Request) -> _Response:
            return urllib.request.urlopen(  # noqa: S310
                request,
                timeout=timeout_seconds,
            )

    else:
        open_url = opener

    paths: dict[str, Path] = {}
    for artifact in source_manifest.artifacts:
        # The digest directory makes a returned path immutable with respect to
        # other valid manifest overrides. Two callers may publish concurrently,
        # but different payloads can never replace one another by basename.
        destination = root / artifact.sha256 / artifact.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _file_matches(destination, artifact):
            paths[artifact.role] = destination
            continue
        request = urllib.request.Request(
            artifact.url,
            headers={"User-Agent": _HTTP_USER_AGENT},
        )
        partial: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".partial",
                delete=False,
            ) as output:
                partial = Path(output.name)
                downloaded_size = 0
                with open_url(request) as response:
                    for chunk in _iter_chunks(response, chunk_bytes):
                        downloaded_size += len(chunk)
                        if downloaded_size > artifact.size_bytes:
                            raise ValueError(
                                f"ACS source {artifact.filename} exceeded its "
                                f"pinned size of {artifact.size_bytes} bytes."
                            )
                        output.write(chunk)
            size, digest = _file_identity(partial)
            _verify_download(artifact, size=size, digest=digest)
            partial.replace(destination)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
        paths[artifact.role] = destination

    return AcsPumsSource(
        household_zip=paths["household"],
        person_zip=paths["person"],
        vintage=source_manifest.vintage,
    )


def _validate_manifest(manifest: AcsSourceManifest) -> None:
    if type(manifest.version) is not int or manifest.version != 1:
        raise ValueError(f"Unsupported ACS source manifest version {manifest.version}.")
    if manifest.spine != ACS_2024_1YR_SPINE:
        raise ValueError(f"ACS source manifest spine must be {ACS_2024_1YR_SPINE!r}.")
    if type(manifest.vintage) is not int or manifest.vintage != ACS_2024_1YR_VINTAGE:
        raise ValueError(f"ACS source manifest vintage must be {ACS_2024_1YR_VINTAGE}.")
    if manifest.source_directory != _SOURCE_DIRECTORY:
        raise ValueError(
            f"ACS source directory must be the pinned {_SOURCE_DIRECTORY!r}."
        )
    if not isinstance(manifest.verified_on, str):
        raise ValueError("ACS source manifest verified_on must be an ISO date string.")
    try:
        verified_on = date.fromisoformat(manifest.verified_on)
    except ValueError as exc:
        raise ValueError(
            "ACS source manifest verified_on must be an ISO date string."
        ) from exc
    if verified_on.isoformat() != manifest.verified_on:
        raise ValueError("ACS source manifest verified_on must be an ISO date string.")

    roles = [artifact.role for artifact in manifest.artifacts]
    if roles != ["household", "person"]:
        raise ValueError(
            "ACS source manifest must declare household then person exactly once."
        )
    for artifact in manifest.artifacts:
        expected_filename = _EXPECTED_FILES.get(artifact.role)
        if artifact.filename != expected_filename:
            raise ValueError(
                f"ACS {artifact.role} artifact filename must be {expected_filename!r}."
            )
        if Path(artifact.filename).name != artifact.filename:
            raise ValueError("ACS source artifact filenames must be basenames.")
        expected_url = f"{manifest.source_directory}{artifact.filename}"
        if artifact.url != expected_url:
            raise ValueError(
                f"ACS {artifact.role} artifact URL must be {expected_url!r}."
            )
        if not isinstance(artifact.sha256, str) or not _SHA256_PATTERN.fullmatch(
            artifact.sha256
        ):
            raise ValueError(
                "ACS source artifact sha256 must contain 64 lowercase hexadecimal "
                "characters."
            )
        if type(artifact.size_bytes) is not int or artifact.size_bytes <= 0:
            raise ValueError(
                "ACS source artifact size_bytes must be a positive integer."
            )


def _file_matches(path: Path, artifact: AcsSourceArtifact) -> bool:
    if not path.is_file() or path.stat().st_size != artifact.size_bytes:
        return False
    size, digest = _file_identity(path)
    return size == artifact.size_bytes and digest == artifact.sha256


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in _iter_chunks(source, _DEFAULT_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _verify_download(
    artifact: AcsSourceArtifact,
    *,
    size: int,
    digest: str,
) -> None:
    if digest != artifact.sha256:
        raise ValueError(
            f"ACS source sha-256 verification failed for {artifact.filename}: "
            f"expected {artifact.sha256}, got {digest}."
        )
    if size != artifact.size_bytes:
        raise ValueError(
            f"ACS source {artifact.filename} byte-size verification failed: "
            f"expected {artifact.size_bytes}, got {size}."
        )


def _iter_chunks(source: BinaryIO, chunk_bytes: int) -> Iterator[bytes]:
    while chunk := source.read(chunk_bytes):
        yield chunk
