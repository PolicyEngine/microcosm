from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event

import pytest

from populace.build.us_runtime.acs_sources import (
    AcsSourceArtifact,
    AcsSourceManifest,
    fetch_acs_pums_sources,
    load_acs_source_manifest,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    household: bytes = b"household zip fixture",
    person: bytes = b"person zip fixture",
) -> AcsSourceManifest:
    return AcsSourceManifest(
        version=1,
        spine="acs_2024_1yr",
        vintage=2024,
        verified_on="2026-07-10",
        source_directory=(
            "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/"
        ),
        artifacts=(
            AcsSourceArtifact(
                role="household",
                filename="csv_hus.zip",
                url=(
                    "https://www2.census.gov/programs-surveys/acs/data/pums/"
                    "2024/1-Year/csv_hus.zip"
                ),
                sha256=_sha(household),
                size_bytes=len(household),
            ),
            AcsSourceArtifact(
                role="person",
                filename="csv_pus.zip",
                url=(
                    "https://www2.census.gov/programs-surveys/acs/data/pums/"
                    "2024/1-Year/csv_pus.zip"
                ),
                sha256=_sha(person),
                size_bytes=len(person),
            ),
        ),
    )


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _InterruptedResponse(_Response):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads == 2:
            raise OSError("fixture interrupted")
        return super().read(size)


def test_fetch_acs_pums_sources_streams_verifies_and_atomically_caches(
    tmp_path: Path,
) -> None:
    payloads = {
        "csv_hus.zip": b"household zip fixture",
        "csv_pus.zip": b"person zip fixture",
    }
    opened: list[str] = []

    def opener(request):
        filename = request.full_url.rsplit("/", 1)[-1]
        opened.append(filename)
        return _Response(payloads[filename])

    source = fetch_acs_pums_sources(
        tmp_path,
        manifest=_manifest(),
        opener=opener,
        chunk_bytes=3,
    )

    assert source.household_zip.read_bytes() == payloads["csv_hus.zip"]
    assert source.person_zip.read_bytes() == payloads["csv_pus.zip"]
    assert opened == ["csv_hus.zip", "csv_pus.zip"]
    assert list(tmp_path.glob("*.partial")) == []


def test_fetch_acs_pums_sources_reuses_only_verified_cache(tmp_path: Path) -> None:
    manifest = _manifest()
    for artifact in manifest.artifacts:
        (tmp_path / artifact.filename).write_bytes(
            b"household zip fixture"
            if artifact.role == "household"
            else b"person zip fixture"
        )

    def no_network(_request):
        raise AssertionError("verified cache must not make a network request")

    source = fetch_acs_pums_sources(
        tmp_path,
        manifest=manifest,
        opener=no_network,
    )

    assert source.household_zip == tmp_path / "csv_hus.zip"
    assert source.person_zip == tmp_path / "csv_pus.zip"


def test_fetch_acs_pums_sources_rejects_hash_mismatch_and_removes_partial(
    tmp_path: Path,
) -> None:
    manifest = _manifest()

    def corrupted(_request):
        return _Response(b"x" * len(b"household zip fixture"))

    with pytest.raises(ValueError, match="sha-256.*csv_hus.zip"):
        fetch_acs_pums_sources(tmp_path, manifest=manifest, opener=corrupted)

    assert not (tmp_path / "csv_hus.zip").exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_fetch_acs_pums_sources_rejects_oversize_before_writing_chunk(
    tmp_path: Path,
) -> None:
    old = tmp_path / "csv_hus.zip"
    old.write_bytes(b"old unverified cache")

    def oversized(_request):
        return _Response(b"household zip fixture!")

    with pytest.raises(ValueError, match="exceeded.*pinned size"):
        fetch_acs_pums_sources(tmp_path, manifest=_manifest(), opener=oversized)

    assert old.read_bytes() == b"old unverified cache"
    assert list(tmp_path.glob("*.partial")) == []


def test_fetch_acs_pums_sources_preserves_cache_on_interrupted_stream(
    tmp_path: Path,
) -> None:
    old = tmp_path / "csv_hus.zip"
    old.write_bytes(b"old unverified cache")

    def interrupted(_request):
        return _InterruptedResponse(b"household zip fixture")

    with pytest.raises(OSError, match="fixture interrupted"):
        fetch_acs_pums_sources(
            tmp_path,
            manifest=_manifest(),
            opener=interrupted,
            chunk_bytes=3,
        )

    assert old.read_bytes() == b"old unverified cache"
    assert list(tmp_path.glob("*.partial")) == []


def test_concurrent_fetchers_cannot_verify_different_bytes_than_they_publish(
    tmp_path: Path,
) -> None:
    good = b"GOOD"
    evil = b"EVIL"
    person = b"PERSON"
    manifest = _manifest(household=good, person=person)
    a_written = Event()
    b_written = Event()
    a_done = Event()

    class CoordinatedResponse(_Response):
        def __init__(self, payload: bytes, *, caller: str):
            super().__init__(payload)
            self._caller = caller
            self._reads = 0

        def read(self, size: int = -1) -> bytes:
            self._reads += 1
            if self._reads == 2 and self._caller == "a":
                a_written.set()
                assert b_written.wait(5)
            elif self._reads == 2 and self._caller == "b":
                b_written.set()
                assert a_done.wait(5)
            return super().read(size)

    def opener_a(request):
        filename = request.full_url.rsplit("/", 1)[-1]
        return (
            CoordinatedResponse(good, caller="a")
            if filename == "csv_hus.zip"
            else _Response(person)
        )

    def opener_b(_request):
        assert a_written.wait(5)
        return CoordinatedResponse(evil, caller="b")

    def fetch_a():
        try:
            return fetch_acs_pums_sources(
                tmp_path,
                manifest=manifest,
                opener=opener_a,
                chunk_bytes=len(good),
            )
        finally:
            a_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(fetch_a)
        future_b = executor.submit(
            fetch_acs_pums_sources,
            tmp_path,
            manifest=manifest,
            opener=opener_b,
            chunk_bytes=len(evil),
        )
        source = future_a.result(timeout=10)
        with pytest.raises(ValueError, match="sha-256.*csv_hus.zip"):
            future_b.result(timeout=10)

    assert source.household_zip.read_bytes() == good
    assert source.person_zip.read_bytes() == person
    assert list(tmp_path.glob("*.partial")) == []


def test_load_acs_source_manifest_rejects_invalid_hash(tmp_path: Path) -> None:
    payload = asdict(_manifest())
    payload["artifacts"][0]["sha256"] = "g" * 64
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        load_acs_source_manifest(path)


def test_load_acs_source_manifest_rejects_changed_url_and_role_order(
    tmp_path: Path,
) -> None:
    payload = asdict(_manifest())
    payload["artifacts"][0]["url"] = "https://example.com/csv_hus.zip"
    path = tmp_path / "changed-url.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact URL must be"):
        load_acs_source_manifest(path)

    payload = asdict(_manifest())
    payload["artifacts"] = list(reversed(payload["artifacts"]))
    path = tmp_path / "changed-order.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="household then person exactly once"):
        load_acs_source_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", True, "Unsupported ACS source manifest version"),
        ("vintage", 2024.0, "manifest vintage must be 2024"),
        ("verified_on", " ", "verified_on must be an ISO date"),
    ],
)
def test_load_acs_source_manifest_rejects_non_strict_scalar_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = asdict(_manifest())
    payload[field] = value
    path = tmp_path / f"invalid-{field}.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_acs_source_manifest(path)


def test_load_acs_source_manifest_rejects_boolean_size(tmp_path: Path) -> None:
    payload = asdict(_manifest())
    payload["artifacts"][0]["size_bytes"] = True
    path = tmp_path / "invalid-size.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="size_bytes must be a positive integer"):
        load_acs_source_manifest(path)


def test_packaged_acs_source_manifest_pins_exact_two_census_archives() -> None:
    manifest = load_acs_source_manifest()

    assert manifest.spine == "acs_2024_1yr"
    assert manifest.vintage == 2024
    assert manifest.verified_on == "2026-07-10"
    assert [asdict(artifact) for artifact in manifest.artifacts] == [
        {
            "role": "household",
            "filename": "csv_hus.zip",
            "url": (
                "https://www2.census.gov/programs-surveys/acs/data/pums/"
                "2024/1-Year/csv_hus.zip"
            ),
            "sha256": (
                "8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0"
            ),
            "size_bytes": 251_500_587,
        },
        {
            "role": "person",
            "filename": "csv_pus.zip",
            "url": (
                "https://www2.census.gov/programs-surveys/acs/data/pums/"
                "2024/1-Year/csv_pus.zip"
            ),
            "sha256": (
                "afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894"
            ),
            "size_bytes": 602_847_146,
        },
    ]
