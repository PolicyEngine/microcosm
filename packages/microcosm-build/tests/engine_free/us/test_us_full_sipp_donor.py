"""The shared SIPP digest must remain bound to the bytes a parser consumes."""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pytest

from microcosm.build.us_runtime import full_sipp_donor as module
from microcosm.build.us_runtime import (
    sipp_financial_assets,
    sipp_head_start,
    sipp_vehicles,
    ssi_disability_criteria,
    voluntary_filing,
)


@pytest.fixture(autouse=True)
def isolate_digest_cache():
    module.clear_full_sipp_sha256_cache()
    yield
    module.clear_full_sipp_sha256_cache()


def test_hash_cache_reuses_unchanged_open_byte_identity(tmp_path, monkeypatch):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    real_hash = module._hash_stream_contents
    scans = []

    def count_scan(stream, *, chunk_size):
        scans.append(stream.fileno())
        return real_hash(stream, chunk_size=chunk_size)

    monkeypatch.setattr(module, "_hash_stream_contents", count_scan)
    expected = hashlib.sha256(b"good").hexdigest()
    wrappers = (
        sipp_financial_assets._sha256_file,
        sipp_head_start._sha256_file,
        sipp_vehicles._sha256_file,
        ssi_disability_criteria._sha256_file,
        voluntary_filing._sha256_file,
    )
    assert [wrapper(path) for wrapper in wrappers] == [expected] * len(wrappers)
    with module.open_verified_full_sipp(path) as verified:
        assert verified.sha256 == expected
        assert verified.stream.read() == b"good"
    assert len(scans) == 1

    path.write_bytes(b"evil")
    assert module.full_sipp_sha256(path) == hashlib.sha256(b"evil").hexdigest()
    assert len(scans) == 2


def test_cache_hit_refuses_same_size_mutation_after_initial_stat(tmp_path, monkeypatch):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    assert module.full_sipp_sha256(path) == hashlib.sha256(b"good").hexdigest()
    real_fingerprint = module._fingerprint
    mutate = True

    def stat_then_mutate(source_path):
        nonlocal mutate
        fingerprint = real_fingerprint(source_path)
        if mutate:
            source_path.write_bytes(b"evil")
            mutate = False
        return fingerprint

    monkeypatch.setattr(module, "_fingerprint", stat_then_mutate)
    with pytest.raises(module.FullSIPPDonorMutationError):
        module.full_sipp_sha256(path)
    assert not module._SHA256_BY_FINGERPRINT
    assert module.full_sipp_sha256(path) == hashlib.sha256(b"evil").hexdigest()


def test_hash_refuses_change_during_scan(tmp_path, monkeypatch):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    real_hash = module._hash_stream_contents

    def hash_then_mutate(stream, *, chunk_size):
        digest = real_hash(stream, chunk_size=chunk_size)
        path.write_bytes(b"evil")
        return digest

    monkeypatch.setattr(module, "_hash_stream_contents", hash_then_mutate)
    with pytest.raises(module.FullSIPPDonorMutationError):
        module.full_sipp_sha256(path)
    assert not module._SHA256_BY_FINGERPRINT


def test_downloader_digest_cannot_seed_changed_bytes(tmp_path):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    downloaded_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    # The old downloader has finished verification, but a concurrent refresh
    # replaces the same-size bytes before it calls the cache-seeding helper.
    path.write_bytes(b"evil")
    with pytest.raises(module.FullSIPPDonorMutationError):
        module.cache_verified_full_sipp_sha256(path, downloaded_digest)
    assert module.full_sipp_sha256(path) == hashlib.sha256(b"evil").hexdigest()
    assert downloaded_digest not in module._SHA256_BY_FINGERPRINT.values()


def test_verified_stream_refuses_path_replacement_after_cache_lookup(
    tmp_path, monkeypatch
):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    module.full_sipp_sha256(path)
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"evil")

    class ReplaceOnCacheHit(dict):
        def get(self, key, default=None):
            digest = super().get(key, default)
            replacement.replace(path)
            return digest

    monkeypatch.setattr(
        module,
        "_SHA256_BY_FINGERPRINT",
        ReplaceOnCacheHit(module._SHA256_BY_FINGERPRINT),
    )
    with pytest.raises(module.FullSIPPDonorMutationError):
        with module.open_verified_full_sipp(path):
            pytest.fail("Changed donor reached the parser")
    assert not module._SHA256_BY_FINGERPRINT


def test_verified_stream_refuses_inplace_change_while_parsing(tmp_path):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    with pytest.raises(module.FullSIPPDonorMutationError):
        with module.open_verified_full_sipp(path) as verified:
            assert verified.stream.read() == b"good"
            path.write_bytes(b"evil")
    assert not module._SHA256_BY_FINGERPRINT


def test_verified_stream_yields_exact_hashed_bytes(tmp_path, monkeypatch):
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"a,b\n1,2\n")
    path_reads = []
    original_open = Path.open

    def count_open(source_path, *args, **kwargs):
        path_reads.append(source_path)
        return original_open(source_path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_open)
    with module.open_verified_full_sipp(path, chunk_size=3) as verified:
        parsed_bytes = verified.stream.read()
        assert parsed_bytes == b"a,b\n1,2\n"
        assert verified.sha256 == hashlib.sha256(parsed_bytes).hexdigest()
    assert path_reads == [path]


@pytest.mark.parametrize(
    ("loader", "columns"),
    [
        (
            sipp_financial_assets.load_sipp_2023_financial_asset_donor,
            sipp_financial_assets.SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
        ),
        (
            sipp_head_start.load_sipp_2023_head_start_donor,
            sipp_head_start.SIPP_HEAD_START_SOURCE_COLUMNS,
        ),
        (
            sipp_vehicles.load_sipp_2023_vehicle_donor,
            sipp_vehicles.SIPP_VEHICLE_SOURCE_COLUMNS,
        ),
        (
            ssi_disability_criteria.load_sipp_2023_ssi_disability_donor,
            ssi_disability_criteria.SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
        ),
        (
            voluntary_filing.load_sipp_2023_voluntary_filing_donor,
            voluntary_filing.SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
        ),
    ],
    ids=["financial-assets", "head-start", "vehicles", "ssi-criteria", "filing"],
)
def test_each_loader_refuses_replacement_between_header_and_rows(
    tmp_path, monkeypatch, loader, columns
):
    path = tmp_path / "pu2023.csv"
    path.write_text("|".join(columns) + "\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"evil")
    original_read_csv = sipp_financial_assets.pd.read_csv
    inputs = []

    def replace_after_header(source, *args, **kwargs):
        inputs.append(source)
        result = original_read_csv(source, *args, **kwargs)
        if kwargs.get("nrows") == 0:
            replacement.replace(path)
        return result

    monkeypatch.setattr(sipp_financial_assets.pd, "read_csv", replace_after_header)
    with pytest.raises(module.FullSIPPDonorMutationError):
        loader(path, expected_sha256=expected, expected_size_bytes=None)
    assert len(inputs) == 2
    assert inputs[0] is inputs[1]
    assert not isinstance(inputs[0], (str, Path))
    assert not module._SHA256_BY_FINGERPRINT


@pytest.mark.parametrize(
    "fetch",
    [
        sipp_vehicles.fetch_sipp_2023_vehicle_donor,
        voluntary_filing.fetch_sipp_2023_voluntary_filing_donor,
    ],
    ids=["vehicles", "filing"],
)
def test_each_downloader_refuses_digest_from_before_cache_refresh(
    tmp_path, monkeypatch, fetch
):
    original_replace = Path.replace

    def replace_then_refresh(source, target):
        result = original_replace(source, target)
        Path(target).write_bytes(b"evil")
        return result

    monkeypatch.setattr(Path, "replace", replace_then_refresh)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(b"good"))
    with pytest.raises(module.FullSIPPDonorMutationError):
        fetch(
            tmp_path,
            expected_sha256=hashlib.sha256(b"good").hexdigest(),
            expected_size_bytes=4,
            chunk_size=2,
        )
    assert (
        module.full_sipp_sha256(tmp_path / "pu2023.csv")
        == hashlib.sha256(b"evil").hexdigest()
    )
