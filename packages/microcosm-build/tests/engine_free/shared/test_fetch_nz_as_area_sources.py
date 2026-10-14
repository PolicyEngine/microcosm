"""The MC9 source fetcher recreates exactly the pinned inputs, or refuses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
SPEC = REPO / "packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json"
_loader = importlib.util.spec_from_file_location(
    "fetch_nz_as_area_sources", REPO / "tools/fetch_nz_as_area_sources.py"
)
fetcher = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(fetcher)


def _provenance(payloads):
    pages = [
        {
            "file": name,
            "url": f"https://example.test/{name}",
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }
        for name, body in payloads.items()
    ]
    return {"pages": pages, "layers": []}


def test_every_pinned_input_in_the_spec_has_a_url_and_a_digest():
    provenance = json.loads(SPEC.read_text())["provenance"]
    downloads = fetcher.pinned_downloads(provenance)
    assert len(downloads) == len(provenance["pages"]) + len(provenance["layers"])
    for name, url, sha256, _size in downloads:
        assert url.startswith("https://services2.arcgis.com/"), name
        assert len(sha256) == 64, name


def test_metadata_url_is_the_layer_endpoint():
    assert (
        fetcher.metadata_url("https://h/x/FeatureServer/0/query")
        == "https://h/x/FeatureServer/0?f=json"
    )
    with pytest.raises(ValueError):
        fetcher.metadata_url("https://h/x/FeatureServer/0")


def test_fetch_writes_bytes_that_match_their_pins(tmp_path):
    payloads = {"a.json": b"alpha", "b.json": b"beta"}
    served = {f"https://example.test/{k}": v for k, v in payloads.items()}
    fetched = fetcher.fetch_all(
        _provenance(payloads), tmp_path, fetch=served.__getitem__
    )
    assert sorted(fetched) == ["a.json", "b.json"]
    assert (tmp_path / "a.json").read_bytes() == b"alpha"


def test_fetch_refuses_changed_bytes_and_writes_nothing(tmp_path):
    provenance = _provenance({"a.json": b"alpha"})
    with pytest.raises(ValueError, match="SHA-256|bytes"):
        fetcher.fetch_all(provenance, tmp_path, fetch=lambda url: b"alphA")
    assert not (tmp_path / "a.json").exists()


def test_fetch_keeps_verified_files_and_replaces_corrupt_ones(tmp_path):
    payloads = {"a.json": b"alpha", "b.json": b"beta"}
    (tmp_path / "a.json").write_bytes(b"alpha")
    (tmp_path / "b.json").write_bytes(b"corrupt")
    calls = []

    def fetch(url):
        calls.append(url)
        return payloads[url.rsplit("/", 1)[1]]

    fetched = fetcher.fetch_all(_provenance(payloads), tmp_path, fetch=fetch)
    assert fetched == ["b.json"]
    assert calls == ["https://example.test/b.json"]
    assert (tmp_path / "b.json").read_bytes() == b"beta"
