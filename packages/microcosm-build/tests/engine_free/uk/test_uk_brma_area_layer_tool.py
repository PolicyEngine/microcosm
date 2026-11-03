"""The area-to-BRMA layer builder's pins, overrides and staleness check (microcosm#1262).

The build itself needs geopandas and Chronicle's raw files, so it runs outside
the workspace (``uvx``) wherever the inputs are staged; these tests cover what
can be checked without them.
"""

from __future__ import annotations

import importlib.util
import json
import re

import pytest

from microcosm.build.uk_runtime import brma_area_layer as bal
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_UK = _TEST_PATHS.package / "src/microcosm/build/uk"
_TOOL = _TEST_PATHS.repository / "tools/build_uk_brma_area_layer.py"
_SPEC = importlib.util.spec_from_file_location("build_uk_brma_area_layer", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _provenance() -> dict:
    return json.loads((_UK / "brma_area_layer.provenance.json").read_text("utf-8"))


def test_registrations_are_pinned_and_match_the_provenance() -> None:
    recorded = _provenance()["registrations"]
    assert set(recorded) == set(tool.REGISTRATIONS)
    for name, pin in tool.REGISTRATIONS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
        assert recorded[name]["sha256"] == pin["sha256"]
        assert recorded[name]["chronicle_package_id"] == pin["package_id"]
        assert recorded[name]["registration_kind"] == "raw_only"


def test_overrides_are_the_committed_overrides() -> None:
    committed = json.loads((_UK / "brma_area_layer.json").read_text("utf-8"))[
        "overrides"
    ]
    assert committed["ew_lsoa21"] == tool.LSOA_OVERRIDES
    assert committed["scotland_oa22"] == tool.OUTPUT_AREA_OVERRIDES
    layer = bal.load_brma_area_layer((_UK / "brma_area_layer.json").read_bytes())
    for code, brma in tool.LSOA_OVERRIDES.items():
        assert layer.brma_by_area["ew_lsoa21"][code] == brma
    for code, brma in tool.OUTPUT_AREA_OVERRIDES.items():
        assert layer.brma_by_area["scotland_oa22"][code] == brma


def test_scottish_codes_are_the_committed_codes() -> None:
    layer = bal.load_brma_area_layer((_UK / "brma_area_layer.json").read_bytes())
    assert dict(layer.scotland_codes) == tool.SCOTTISH_BRMA_CODES


@pytest.mark.parametrize(
    ("raw", "key"),
    [("G83 8NQ", "G838NQ"), ("eh1 1aa", "EH11AA"), ("G83 8NQA", "G838NQ")],
)
def test_postcode_key_drops_spaces_and_split_suffixes(raw, key) -> None:
    assert tool._postcode_key(raw) == key


def test_postcode_key_refuses_a_non_postcode() -> None:
    with pytest.raises(SystemExit, match="unrecognised postcode"):
        tool._postcode_key("NOT A POSTCODE")


def test_staleness_ignores_the_build_date(tmp_path) -> None:
    payload = {"kind": "x", "built_on": "2026-10-16", "value": 1}
    path = tmp_path / "resource.json"
    path.write_text(tool._render(payload), encoding="utf-8")
    assert not tool._stale(path, {**payload, "built_on": "2027-01-01"})
    assert tool._stale(path, {**payload, "value": 2})
    assert tool._stale(tmp_path / "missing.json", payload)
