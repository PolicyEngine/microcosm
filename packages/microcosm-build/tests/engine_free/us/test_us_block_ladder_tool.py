"""End-to-end tests for tools/build_us_block_ladder_artifact.py on a local cache.

Every source the builder downloads is written into a temporary cache under the
file name the builder caches it as, so ``main()`` runs offline; any network
request fails the test.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import load_us_block_ladder
from microcosm.build.us_runtime.block_ladder_sources import (
    us_block_ladder_cbsa_coverage_failures,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

# Real 2020 blocks (BAF MCD layer): Bridgeport, Shelton, Stamford (all in 2020
# Fairfield County 09001), New Haven (09009), and one Delaware block.
_BLOCKS = {
    "090010701001000": ("08070", "04", 40),
    "090011101001000": ("68170", "03", 30),
    "090010201011000": ("73070", "04", 20),
    "090091401011000": ("52070", "03", 10),
}
_DE_BLOCK = "100010401001000"

# Rows copied from Census's ct_cou_to_cousub_crosswalk.txt, BOM included.
_CT_CROSSWALK = (
    "﻿"
    '"STATEFP\n(INCITS38)"|"OLD_COUNTYFP\n(INCITS31)"|OLD_COUNTY_NAMELSAD|'
    '"NEW_COUNTYFP\n(INCITS31)"|NEW_COUNTY_NAMELSAD|COUSUBFP|OLD_COUSUB_GEOID|'
    'NEW_COUSUB_GEOID|COUSUB_NAMELSAD|"COUSUBNS\n(INCITS446)"|COUSUB_LSAD|'
    "COUSUB_FUNCSTAT|COUSUB_CLASSFP\n"
    "09|001|Fairfield County|120|Greater Bridgeport Planning Region|08070|"
    "0900108070|0912008070|Bridgeport town|00213396|43|C|T5\n"
    "09|001|Fairfield County|120|Greater Bridgeport Planning Region|00000|"
    "0900100000|0912000000|County subdivisions not defined|00000000|00|F|Z9\n"
    "09|001|Fairfield County|140|Naugatuck Valley Planning Region|68170|"
    "0900168170|0914068170|Shelton town|00213504|43|C|T5\n"
    "09|001|Fairfield County|190|Western Connecticut Planning Region|73070|"
    "0900173070|0919073070|Stamford town|00213511|43|C|T5\n"
    "09|009|New Haven County|170|South Central Connecticut Planning Region|52070|"
    "0900952070|0917052070|New Haven town|00213471|43|C|T5\n"
    "\n"
    "\n"
    "GLOSSARY\n"
    "STATEFP = State FIPS Code / ANSI INCITS 38 Code\n"
)
# (CBSA Code, CBSA Title, FIPS State Code, FIPS County Code), as in
# list1_2023.xlsx. Puerto Rico is outside the state spine and never built.
_DELINEATION_ROWS = (
    ("14860", "Bridgeport-Stamford-Danbury, CT", "09", "120"),
    ("47930", "Waterbury-Shelton, CT", "09", "140"),
    ("35300", "New Haven, CT", "09", "170"),
    ("14860", "Bridgeport-Stamford-Danbury, CT", "09", "190"),
    ("20100", "Dover, DE", "10", "001"),
    ("41980", "San Juan-Bayamón-Caguas, PR", "72", "127"),
)


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_block_ladder_artifact.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_block_ladder_artifact", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pl_line(summary_level: str, geocode: str, population: int) -> str:
    fields = [""] * 97
    fields[0], fields[2], fields[9], fields[90] = (
        "PLST",
        summary_level,
        geocode,
        str(population),
    )
    return "|".join(fields) + "\n"


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)


def _write_cache(cache: Path, *, with_ct_mcd: bool = True) -> None:
    import openpyxl

    cache.mkdir()
    _write_zip(
        cache / "cd119.zip",
        {
            "NationalCD119.txt": "GEOID,CDFP\n"
            + "".join(f"{block},{cd}\n" for block, (_, cd, _) in _BLOCKS.items())
            + f"{_DE_BLOCK},00\n"
        },
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(("List 1. CORE BASED STATISTICAL AREAS (CBSAs)",))
    sheet.append(("CBSA Code", "CBSA Title", "FIPS State Code", "FIPS County Code"))
    for row in _DELINEATION_ROWS:
        sheet.append(row)
    workbook.save(cache / "list1_2023.xlsx")
    ct_population = sum(population for _, _, population in _BLOCKS.values())
    _write_zip(
        cache / "ct2020.pl.zip",
        {
            "ctgeo2020.pl": _pl_line("040", "09", ct_population)
            + "".join(
                _pl_line("750", block, population)
                for block, (_, _, population) in _BLOCKS.items()
            )
        },
    )
    _write_zip(
        cache / "de2020.pl.zip",
        {"degeo2020.pl": _pl_line("040", "10", 5) + _pl_line("750", _DE_BLOCK, 5)},
    )
    ct_baf = {
        "BlockAssign_ST09_CT_SLDU.txt": "BLOCKID|DISTRICT\n"
        + "".join(f"{block}|001\n" for block in _BLOCKS),
        "BlockAssign_ST09_CT_SLDL.txt": "BLOCKID|DISTRICT\n"
        + "".join(f"{block}|120\n" for block in _BLOCKS),
        "BlockAssign_ST09_CT_INCPLACE_CDP.txt": "BLOCKID|PLACEFP\n"
        + "".join(f"{block}|\n" for block in _BLOCKS),
    }
    if with_ct_mcd:
        ct_baf["BlockAssign_ST09_CT_MCD.txt"] = "BLOCKID|COUNTYFP|COUSUBFP\n" + "".join(
            f"{block}|{block[2:5]}|{cousub}\n"
            for block, (cousub, _, _) in _BLOCKS.items()
        )
    _write_zip(cache / "BlockAssign_ST09_CT.zip", ct_baf)
    _write_zip(
        cache / "BlockAssign_ST10_DE.zip",
        {"BlockAssign_ST10_DE_SLDU.txt": f"BLOCKID|DISTRICT\n{_DE_BLOCK}|001\n"},
    )
    (cache / "ct_cou_to_cousub_crosswalk.txt").write_text(
        _CT_CROSSWALK, encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected download: {args[0]!r}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_builder_gives_connecticut_blocks_their_planning_region_cbsa(
    tmp_path, capsys
) -> None:
    tool = _load_tool_module()
    cache = tmp_path / "cache"
    _write_cache(cache)
    out = tmp_path / "ladder.npz"

    tool.main(["--out", str(out), "--cache-dir", str(cache), "--states", "09,10"])

    ladder = load_us_block_ladder(out)
    cbsa = dict(
        zip(
            (f"{block:015d}" for block in ladder.block_geoid.tolist()),
            ladder.cbsa_code.tolist(),
            strict=True,
        )
    )
    assert cbsa == {
        "090010201011000": 14860,  # Stamford → Western → Bridgeport-Stamford
        "090010701001000": 14860,  # Bridgeport → Greater Bridgeport
        "090011101001000": 47930,  # Shelton → Naugatuck Valley → Waterbury
        "090091401011000": 35300,  # New Haven → South Central → New Haven
        _DE_BLOCK: 20100,
    }
    assert us_block_ladder_cbsa_coverage_failures(ladder) == []
    metadata = ladder.metadata
    assert metadata["cbsa_delineated_states"] == ["09", "10"]
    assert metadata["layers"]["cbsa"]["vintage"] == "omb_2023_delineations"
    assert (
        metadata["layers"]["cbsa"]["ct_planning_region_crosswalk_url"]
        == tool.CT_PLANNING_REGION_CROSSWALK_URL
    )
    crosswalk_source = metadata["source_files"]["ct_planning_region_crosswalk"]
    assert crosswalk_source["path"] == str(cache / "ct_cou_to_cousub_crosswalk.txt")
    assert len(crosswalk_source["sha256"]) == 64
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["cbsa_population_share_by_state"] == {"09": 1.0, "10": 1.0}
    assert summary["cbsa_delineated_states"] == ["09", "10"]
    capsys.readouterr()


def test_builder_without_connecticut_never_reads_the_crosswalk(
    tmp_path, capsys
) -> None:
    tool = _load_tool_module()
    cache = tmp_path / "cache"
    _write_cache(cache)
    # Absent from the cache: reading it would attempt a (refused) download.
    (cache / "ct_cou_to_cousub_crosswalk.txt").unlink()
    out = tmp_path / "ladder.npz"

    tool.main(["--out", str(out), "--cache-dir", str(cache), "--states", "10"])

    ladder = load_us_block_ladder(out)
    assert ladder.cbsa_code.tolist() == [20100]
    assert ladder.metadata["cbsa_delineated_states"] == ["10"]
    assert "ct_planning_region_crosswalk" not in ladder.metadata["source_files"]
    capsys.readouterr()


def test_builder_refuses_connecticut_without_its_county_subdivision_layer(
    tmp_path,
) -> None:
    tool = _load_tool_module()
    cache = tmp_path / "cache"
    _write_cache(cache, with_ct_mcd=False)

    with pytest.raises(SystemExit, match="BlockAssign_ST09_CT_MCD.txt"):
        tool.main(
            [
                "--out",
                str(tmp_path / "ladder.npz"),
                "--cache-dir",
                str(cache),
                "--states",
                "09,10",
            ]
        )
    assert not (tmp_path / "ladder.npz").exists()


def test_builder_output_matches_the_pre_fix_join_outside_connecticut(
    tmp_path, capsys
) -> None:
    """Differential: outside CT the join equals the 2020-county lookup."""
    tool = _load_tool_module()
    cache = tmp_path / "cache"
    _write_cache(cache)
    out = tmp_path / "ladder.npz"
    tool.main(["--out", str(out), "--cache-dir", str(cache), "--states", "09,10"])
    capsys.readouterr()

    ladder = load_us_block_ladder(out)
    by_county = {
        f"{state}{county}": int(code) for code, _, state, county in _DELINEATION_ROWS
    }
    outside = ladder.block_geoid // 10**13 != 9
    expected = np.asarray(
        [by_county.get(f"{block:015d}"[:5], 0) for block in ladder.block_geoid[outside]]
    )
    np.testing.assert_array_equal(ladder.cbsa_code[outside], expected)
