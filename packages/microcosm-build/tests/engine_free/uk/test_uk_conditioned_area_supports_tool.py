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
    """Three postcodes per area, two of them at rank 1; band mixes per LSOA and council."""
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
    council = decode_atomic_support(bases[SCOT]).arrays["local_authority_code"][0]
    scot = {
        "rows": [
            {
                "geography": {"id": str(council)},
                "measure_id": f"band_{band}",
                "value": value,
            }
            for band, value in zip("abcdefgh", (1, 2, 1, 1, 1, 1, 1, 1), strict=True)
        ]
    }
    resources = {
        name: {
            "resource": name,
            "sha256": "1" * 64,
            "publisher_package": "invented-package",
            "publisher_sha256": "2" * 64,
        }
        for name in (
            "voa_council_tax_band_mix_lsoa.json",
            "scotgov_chargeable_dwellings_by_band.json",
        )
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
