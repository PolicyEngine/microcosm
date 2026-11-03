"""The MC9 source fetcher recreates exactly the pinned inputs, or refuses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from test_support.paths import paths_for

REPO = paths_for("microcosm-build").repository
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


def test_joint_geography_pages_pin_the_complete_regc_concordance():
    provenance = fetcher.JOINT_GEOGRAPHY_PROVENANCE
    downloads = fetcher.pinned_downloads(provenance)
    assert len(downloads) == 29
    assert provenance["layers"] == []
    assert sum(page["features"] for page in provenance["pages"]) == 57575
    for index, (name, url, sha256, size) in enumerate(downloads):
        assert name == f"mbhg-2026-regc-p{index:03d}.json"
        parsed = urlsplit(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "services2.arcgis.com"
        assert parsed.path.endswith(
            "/Meshblock_Higher_Geographies_2026/FeatureServer/0/query"
        )
        query = parse_qs(parsed.query)
        assert query["resultOffset"] == [str(index * 2000)]
        assert query["resultRecordCount"] == ["2000"]
        assert set(query["outFields"][0].split(",")) == {
            "MB2026_V1_00",
            "SA12026_V1_00",
            "TA2026_V1_00",
            "TA2026_V1_00_NAME",
            "REGC2025_V1_00",
            "REGC2025_V1_00_NAME",
            "TALB2026_V1_00",
            "TALB2026_V1_00_NAME",
        }
        assert len(sha256) == 64
        assert set(sha256) <= set("0123456789abcdef")
        assert size > 0


def test_joint_geography_refuses_unpinned_page(tmp_path):
    provenance = fetcher.JOINT_GEOGRAPHY_PROVENANCE
    first_page = provenance["pages"][0]
    with pytest.raises(ValueError, match="SHA-256"):
        fetcher.fetch_all(
            provenance, tmp_path, fetch=lambda _url: b"\0" * first_page["bytes"]
        )
    assert not (tmp_path / first_page["file"]).exists()


@pytest.mark.parametrize("joint_geography", [False, True])
def test_cli_uses_selected_provenance_and_source_environment(
    tmp_path, monkeypatch, joint_geography
):
    monkeypatch.setenv("NZ_AS_AREA_SOURCES", str(tmp_path / "as"))
    monkeypatch.setenv("NZ_JOINT_GEOGRAPHY_SOURCES", str(tmp_path / "joint"))
    captured = []

    def fetch_all(provenance, out_dir):
        captured.append((provenance, out_dir))
        return []

    monkeypatch.setattr(fetcher, "fetch_all", fetch_all)
    args = ["--joint-geography"] if joint_geography else ["--spec", str(SPEC)]
    assert fetcher.main(args) == 0
    provenance, out_dir = captured[0]
    if joint_geography:
        assert provenance is fetcher.JOINT_GEOGRAPHY_PROVENANCE
        assert out_dir == tmp_path / "joint"
    else:
        assert provenance == json.loads(SPEC.read_text())["provenance"]
        assert out_dir == tmp_path / "as"


def test_cli_explicit_output_overrides_joint_geography_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NZ_JOINT_GEOGRAPHY_SOURCES", str(tmp_path / "unused"))
    captured = []

    def fetch_all(_provenance, out_dir):
        captured.append(out_dir)
        return []

    monkeypatch.setattr(fetcher, "fetch_all", fetch_all)
    assert fetcher.main(["--joint-geography", "--out", str(tmp_path)]) == 0
    assert captured == [tmp_path]
