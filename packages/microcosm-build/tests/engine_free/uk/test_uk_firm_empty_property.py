"""UK firm empty property: an opt-in draw of unoccupied properties."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import cache

import pytest

from microcosm.build.uk_runtime.firm_empty_property import (
    UK_FIRM_EMPTY_FILES,
    UK_FIRM_EMPTY_STATUS,
    UKFirmEmptyPropertySource,
    read_uk_firm_empty_property_source,
    uk_firm_empty_property_published_figures,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_empty_property_2024_25"
TOOL = PATHS.repository / "tools" / "extract_uk_firm_empty_property_tables.py"


@cache
def _source() -> UKFirmEmptyPropertySource:
    return read_uk_firm_empty_property_source(FIXTURE)


@cache
def _tool():
    name = "extract_uk_firm_empty_property_tables"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_reads_published_values() -> None:
    source = _source()
    dec_2024 = source.hereditaments["2024-12-31"]
    assert dec_2024["hdit_mr_empty"] == 178_507
    assert dec_2024["hdit_mr_emptyind"] == 12_873
    assert dec_2024["hdit_mr_emptyabove"] == 39_759
    assert source.hereditaments["2023-12-31"]["hdit_mr_empty"] == 175_870
    assert source.relief_gbp["2025-26"]["value_empty"] == 1_084_979_229
    assert source.nndr3["empty_premises_relief"] == pytest.approx(1275.504353)
    assert source.nndr3["hereditaments_30_sept"] == 2011
    assert source.nndr3["small_business_multiplier"] == 49.9
    assert source.nndr3["standard_multiplier"] == 54.6


def test_provenance_hashes_match_fixture() -> None:
    record = json.loads((FIXTURE / "provenance.json").read_text(encoding="utf-8"))
    assert record["accessed"] == "2026-11-23"
    assert "none was filled from memory" in record["method"]
    assert set(record["sources"]) == {"nndr1_2024_25", "nndr1_2025_26", "nndr3_2024_25"}
    for name in UK_FIRM_EMPTY_FILES:
        payload = (FIXTURE / name).read_bytes()
        assert record["files"][name]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_supplementary_extraction_keys_on_variable_names() -> None:
    tool = _tool()
    names = (tool.HEREDITAMENT_TOTAL, *tool.HEREDITAMENT_PARTS)
    values = (21, 1, 2, 3, 4, 5, 6)
    hereditaments = [
        ("Hereditaments qualifying ... in England as at 31 December 2024",),
        (None, None, None, None, "Variable name", *names),
        (
            None,
            None,
            None,
            None,
            "Notes",
            "Mandatory Relief for empty premises",
            *"abcdef",
        ),
        ("Eng", "E92000001", "England ", "TE", None, *values),
    ]
    relief_names = (tool.RELIEF_TOTAL, *tool.RELIEF_PARTS)
    relief = [
        ("Estimated cost of empty property relief ... in 2025-26 [note 7]",),
        (None, None, None, None, "Estimated value of empty property relief", *"abcdef"),
        (None, None, None, None, *relief_names),
        ("Eng", "Eng", "England", None, 7, 1, 1, 1, 1, 1, 1),
    ]
    h, r = tool.extract_supplementary(hereditaments, relief)
    assert h[0] == (
        "2024-12-31",
        "hdit_mr_empty",
        "Mandatory Relief for empty premises",
        "21",
    )
    assert r[0][:2] == ("2025-26", "value_empty")
    broken = [*hereditaments[:3], ("Eng", "E", "England", "TE", None, 22, *values[1:])]
    with pytest.raises(ValueError, match="do not sum"):
        tool.extract_supplementary(broken, relief)


def test_published_figures_and_gap() -> None:
    figures = uk_firm_empty_property_published_figures(_source())
    assert figures["status"] == UK_FIRM_EMPTY_STATUS == "not_set"
    assert "firm_property_occupied" in figures["gap"]
    assert figures["empty_relief_hereditaments"] == 178_507
    assert figures["empty_relief_hereditaments_above_threshold"] == 52_632
    assert figures["hereditaments_on_list"] == 2_011_000
    assert figures["empty_relief_share_of_hereditaments"] == pytest.approx(
        178_507 / 2_011_000
    )
    assert figures["empty_relief_above_threshold_forecast_gbp"] == 301_645_624
    assert figures["empty_relief_outturn_gbp"] == pytest.approx(1_275_504_353)
    assert figures["empty_relief_share_of_gross_rates"] == pytest.approx(
        1275.504353 / 35439.30771
    )


def test_module_assigns_nothing() -> None:
    import microcosm.build.uk_runtime.firm_empty_property as module

    assert not [name for name in module.__all__ if name.startswith("assign")]
    assert not hasattr(module, "draw_uk_firm_empty_property")
