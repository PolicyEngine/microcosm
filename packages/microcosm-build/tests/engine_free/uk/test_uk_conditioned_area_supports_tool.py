"""The UK conditioned-support tool on toy supports and invented inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.uk_runtime.atomic_area_support import (
    BAND_WEIGHT_COLUMNS,
    BAND_WEIGHT_SCALE,
    IMD_DECILE_COLUMN,
    SYSTEMS,
    uk_conditioned_support_base,
)
from test_support.microcosm_build.uk_atomic_support_fixtures import toy_support_payloads
from test_support.paths import paths_for

EW, SCOT, NI = SYSTEMS


@pytest.fixture(scope="module")
def tool():
    path = (
        paths_for("microcosm-build").repository
        / "tools/build_uk_conditioned_area_supports.py"
    )
    spec = importlib.util.spec_from_file_location("uk_conditioned_tool_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invented_inputs(bases: dict[str, bytes]):
    """Three postcodes per area, two of them at rank 1; band mixes per LSOA and zone."""
    postcodes = []
    for payload in bases.values():
        for area in decode_atomic_support(payload).arrays["area"].tolist():
            postcodes += [(area, "1"), (area, "1"), (area, "2")]
    frame = pd.DataFrame(postcodes, columns=["oa21cd", "imd20ind"])
    deciles = {nation: {1: "4", 2: "9"} for nation in "EWSN"}
    ew = decode_atomic_support(bases[EW])
    lsoa = {}
    for code in ew.arrays["lsoa_code"].tolist():
        # England: an eighth in each of bands A to H. Wales: a tenth in each of
        # A to H and a fifth in band I.
        cumulative = (
            [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0]
            if code.startswith("E")
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        )
        lsoa[code] = {"all_properties": 40, "cumulative": cumulative}
    voa = {"bands": list("ABCDEFGHI"), "lsoa": lsoa}
    scot = {
        str(zone): dict(zip("abcdefgh", (1, 2, 1, 1, 1, 1, 1, 1), strict=True))
        for zone in decode_atomic_support(bases[SCOT]).arrays["lsoa_code"].tolist()
    }
    resources = {
        "voa_council_tax_band_mix_lsoa.json": {
            "resource": "voa_council_tax_band_mix_lsoa.json",
            "sha256": "1" * 64,
            "publisher_package": "invented-package",
            "publisher_sha256": "2" * 64,
        }
    }
    return frame, deciles, voa, scot, resources


def test_tool_conditions_each_registered_support(tool) -> None:
    bases = toy_support_payloads()
    postcodes, deciles, voa, scot, resources = _invented_inputs(bases)
    payloads, entries = tool.condition_supports(
        bases,
        postcodes,
        deciles,
        voa_mix=voa,
        scotland_counts=scot,
        resources=resources,
    )
    for system in SYSTEMS:
        assert uk_conditioned_support_base(payloads[system]) == bases[system]
        support = decode_atomic_support(payloads[system])
        # Two of every area's three postcodes carry rank 1, decile 4.
        assert set(support.arrays[IMD_DECILE_COLUMN].tolist()) == {"4"}
        receipt = entries[system]["imd_decile"]
        assert receipt["areas_by_rule"] == {"area": len(support.arrays["area"])}
        assert receipt["areas_whose_records_span_codes"] == len(support.arrays["area"])
        households = np.asarray(support.arrays["households"], dtype=np.int64)
        weights = {
            c: np.asarray(support.arrays[c]) for c in BAND_WEIGHT_COLUMNS[system]
        }
        if system == EW:
            english = np.char.startswith(np.asarray(support.arrays["area"]), "E")
            assert np.array_equal(
                weights["households_band_c"][english], households[english] * 125_000
            )
            assert not weights["households_band_i"][english].any()
            assert np.array_equal(
                weights["households_band_i"][~english], households[~english] * 200_000
            )
        if system == SCOT:
            # A ninth in band A, rounded half to even.
            ninth = [round(int(h) * BAND_WEIGHT_SCALE / 9) for h in households]
            assert weights["households_band_a"].tolist() == ninth
        if system == NI:
            assert not weights and entries[system]["band_weights"] is None


def test_tool_refuses_a_rank_outside_its_nation_index(tool) -> None:
    bases = toy_support_payloads()
    postcodes, deciles, voa, scot, resources = _invented_inputs(bases)
    postcodes.loc[0, "imd20ind"] = "77"
    with pytest.raises(ValueError, match="not in the E index"):
        tool.condition_supports(
            bases,
            postcodes,
            deciles,
            voa_mix=voa,
            scotland_counts=scot,
            resources=resources,
        )


def test_rank_deciles_refuse_a_split_tie_and_a_short_index(tool, monkeypatch) -> None:
    monkeypatch.setitem(tool.INDEX_AREAS, "E", 3)
    ok = tool._rank_deciles(pd.Series([1, 1, 3]), pd.Series([1, 1, 2]), nation="E")
    assert ok == {1: "1", 3: "2"}
    with pytest.raises(ValueError, match="no single decile"):
        tool._rank_deciles(pd.Series([1, 1, 3]), pd.Series([1, 2, 2]), nation="E")
    with pytest.raises(ValueError, match="expected 3"):
        tool._rank_deciles(pd.Series([1, 2]), pd.Series([1, 2]), nation="E")


def _nrs_workbook(tool, path, rows):
    """NRS's layout: four title rows, the header on row 5, zones, then a footnote."""
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = tool.NRS_SHEET
    for _ in range(tool.NRS_HEADER_ROW):
        sheet.append(["Dwelling estimates by 2022 Data Zone"])
    sheet.append(
        [
            tool.NRS_ZONE_COLUMN,
            "Data zone name",
            tool.NRS_TOTAL_COLUMN,
            *tool.NRS_BAND_COLUMNS.values(),
        ]
    )
    for zone, total, *bands in rows:
        sheet.append([zone, f"{zone} name", total, *bands])
    sheet.append(["Source: Scottish Assessors' Portal data"])
    workbook.save(path)
    return path


def test_nrs_band_counts_read_each_zone_and_refuse_a_broken_sum(
    tool, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(tool, "NRS_DATA_ZONES", 2)
    rows = [
        ["S01013482", 9, 1, 2, 1, 1, 1, 1, 1, 1],
        ["S01013483", 8, 0, 0, 0, 0, 0, 0, 0, 8],
    ]
    path = tmp_path / "zones.xlsx"
    assert tool._nrs_band_counts(_nrs_workbook(tool, path, rows)) == {
        "S01013482": dict(zip("abcdefgh", (1, 2, 1, 1, 1, 1, 1, 1), strict=True)),
        "S01013483": {**dict.fromkeys("abcdefg", 0), "h": 8},
    }
    with pytest.raises(ValueError, match="expected 2"):
        tool._nrs_band_counts(_nrs_workbook(tool, path, rows[:1]))
    rows[1][1] = 9
    with pytest.raises(ValueError, match="miss its total"):
        tool._nrs_band_counts(_nrs_workbook(tool, path, rows))


def test_scotland_numerators_refuse_a_zone_without_counts(tool) -> None:
    support = decode_atomic_support(toy_support_payloads()[SCOT])
    with pytest.raises(ValueError, match="each data zone A to H"):
        tool._scotland_numerators(support, {"S01999999": dict.fromkeys("abcdefgh", 1)})


def test_northern_ireland_deciles_are_tenths_of_its_soas(tool) -> None:
    deciles = tool._northern_ireland()
    assert len(deciles) == 890
    assert Counter(deciles.values()) == {str(d): 89 for d in range(1, 11)}
    assert (deciles[1], deciles[89], deciles[90], deciles[890]) == ("1", "1", "2", "10")


def test_pinned_sources_refuse_bytes_other_than_the_registration(
    tool, tmp_path
) -> None:
    rows = {}
    for name, spec in tool.SOURCES.items():
        path = tmp_path / str(spec["filename"])
        path.write_bytes(name.encode())
        rows[name] = {
            "path": str(path),
            "url": spec["url"],
            "sha256": spec["sha256"],
            "bytes": spec["size_bytes"],
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "sources": rows}))
    with pytest.raises(ValueError, match="identity mismatch"):
        tool._pinned_sources(manifest)
    real = tmp_path / str(tool.SOURCES["nspl"]["filename"])
    rows["nspl"]["sha256"] = hashlib.sha256(real.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"schema_version": 1, "sources": rows}))
    with pytest.raises(ValueError, match="differs from its registration"):
        tool._pinned_sources(manifest)
