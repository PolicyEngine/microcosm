"""Offline pinned builds of the three UK atomic-area supports from invented files.

Every publisher format the tool reads (ONS CSVs, Nomis zips, NRS zips and CSVs,
the NISRA GeoJSON zip, table-builder CSVs and the DZ lookup workbook) is
invented here at a few areas per nation; national counts are monkeypatched on
the shared loader module the same way the loader tests do. No download, no
publisher file, no country engine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.uk_runtime.geography_sources as geography_sources
from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.uk_runtime import assemble_uk_oa_ladder, load_uk_oa_ladder
from microcosm.build.uk_runtime.atomic_area_support import (
    SOURCES,
    SYSTEMS,
    uk_atomic_assignment_definition,
)

EW, SCOT, NI = SYSTEMS

# oa, lsoa, msoa, lad22, lad24, region, pcon, ward, itl3, population, households
_EW_ROWS = [
    ("E00000001", "E06000047", "E06000047", "E12000001", "TLC11", 100, 40),
    ("E00000002", "E06000049", "E06000049", "E12000002", "TLD11", 110, 44),
    ("E00000003", "E08000016", "E08000016", "E12000003", "TLE31", 120, 48),
    ("E00000004", "E08000019", "E08000019", "E12000003", "TLE32", 130, 52),
    # A retired LAD22 code whose OA now sits in a 2023 unitary authority.
    ("E00000005", "E07000026", "E06000063", "E12000004", "TLF11", 140, 56),
    ("E00000006", "E06000019", "E06000019", "E12000005", "TLG11", 150, 60),
    ("E00000007", "E06000031", "E06000031", "E12000006", "TLH11", 160, 64),
    ("E00000008", "E09000001", "E09000001", "E12000007", "TLI31", 170, 68),
    ("E00000009", "E06000035", "E06000035", "E12000008", "TLJ11", 180, 72),
    ("E00000010", "E06000022", "E06000022", "E12000009", "TLK11", 190, 76),
    ("W00000001", "W06000001", "W06000001", "", "TLL11", 200, 80),
]
_SCOT_ROWS = [
    ("S00000001", "S30000001", "S12000033", "S14000001", "S13002835", "TLM50", 90),
    ("S00000002", "S30000002", "S12000005", "S14000002", "S13002604", "TLM82", 75),
]
_NI_COUNT = 18


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return buffer.getvalue()


def _csv(header: str, rows: list[str]) -> str:
    return header + "\n" + "\n".join(rows) + "\n"


def _ni_workbook(*, missing_constituency: bool = False) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "DZ2021_Admin_geog_lookup"
    sheet.append(
        [
            "DZ2021_code",
            "PARLCON2024_code",
            "SDZ2021_code",
            "LGD2014_code",
            "DEA2014_code",
        ]
    )
    for index in range(1, _NI_COUNT + 1):
        constituency = f"N050000{index:02d}"
        if missing_constituency and index == _NI_COUNT:
            constituency = "N05000017"
        sheet.append(
            [
                f"N200000{index:02d}",
                constituency,
                f"N210000{index:02d}",
                "N09000001",
                f"N100000{index:02d}",
            ]
        )
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def invented_files(
    *,
    ew_rows: list[tuple] | None = None,
    ni_missing_constituency: bool = False,
) -> dict[str, bytes]:
    """Every publisher file the tool reads, keyed like the tool's SOURCES."""
    ew_rows = _EW_ROWS if ew_rows is None else ew_rows
    hierarchy = [
        f"{oa},{oa[0]}01{oa[3:]},x,{oa[0]}02{oa[3:]},y,{lad22},z"
        for oa, lad22, *_ in ew_rows
    ]
    parncp = [
        f"{oa},{oa[0]}04{oa[3:]},p,{lad24},l,{region},r,{oa[0]}92000001,c"
        for oa, _lad22, lad24, region, *_ in ew_rows
    ]
    pcon = [f"{oa},{oa[0]}14{oa[3:]},n" for oa, *_ in ew_rows]
    pcon[-1] = f"{ew_rows[-1][0]},W07000041,n"
    ward = [f"{oa},{oa[0]}05{oa[3:]},w,IGNORED,i" for oa, *_ in ew_rows]
    itl = [
        f"TLX,x,TLX1,x,{itl3},x,{oa[0]}30{oa[3:]},x,{lad24},x,1"
        for oa, _l, lad24, _r, itl3, *_ in ew_rows
    ]
    itl += [
        f"TLM,s,TLM5,s,{itl3},s,{lau},s,{ca},s,2"
        for _oa, lau, ca, _p, _w, itl3, _n in _SCOT_ROWS
    ]
    itl += ["TLN,n,TLN0,n,TLN0A,n,N30000001,n,N09000001,n,3"]
    population = [f"2021,{oa},{oa},{pop}" for oa, *_, pop, _hh in ew_rows]
    households = [f"2021,{oa},{oa},{hh}" for oa, *_, hh in ew_rows]
    ni_index = range(1, _NI_COUNT + 1)
    ni_features = [
        {
            "type": "Feature",
            "properties": {
                "DZ2021_cd": f"N200000{i:02d}",
                "SDZ2021_cd": f"N210000{i:02d}",
                "LGD2014_cd": "N09000001",
                "DEA2014_cd": f"N100000{i:02d}",
            },
        }
        for i in ni_index
    ]
    return {
        "oa_hierarchy": _csv(
            "OA21CD,LSOA21CD,LSOA21NM,MSOA21CD,MSOA21NM,LAD22CD,LAD22NM", hierarchy
        ).encode(),
        "oa_parncp_lad_region": _csv(
            "OA21CD,PARNCP24CD,PARNCP24NM,LAD24CD,LAD24NM,RGN24CD,RGN24NM,CTRY24CD,CTRY24NM",
            parncp,
        ).encode(),
        "oa_constituency": _csv("OA21CD,PCON25CD,PCON25NM", pcon).encode(),
        "oa_ward": _csv("OA21CD,WD24CD,WD24NM,LAD24CD,LAD24NM", ward).encode(),
        "lad_itl": _csv(
            "ITL125CD,ITL125NM,ITL225CD,ITL225NM,ITL325CD,ITL325NM,LAU125CD,LAU125NM,LAD24CD,LAD24NM,ObjectId",
            itl,
        ).encode(),
        "oa_population": _zip(
            {
                "census2021-ts001-oa.csv": _csv(
                    "date,geography,geography code,Residence type: Total; measures: Value",
                    population,
                )
            }
        ),
        "oa_households": _zip(
            {
                "census2021-ts041-oa.csv": _csv(
                    "date,geography,geography code,Household: Total; measures: Value",
                    households,
                )
            }
        ),
        "scotland_dz_iz": _zip(
            {
                "OA22_DZ22_IZ22.csv": _csv(
                    "OA22,DZ22,IZ22",
                    [f"{oa},S01{oa[3:]},S02{oa[3:]}" for oa, *_ in _SCOT_ROWS],
                )
            }
        ),
        "scotland_lau": _zip(
            {
                "OA22_LAU25_L1.csv": _csv(
                    "OutputArea2022Code,LAU2025Level1Code",
                    [f"{oa},{lau}" for oa, lau, *_ in _SCOT_ROWS],
                ),
                "CA19 - LAU25L1.csv": _csv(
                    "LAU2025Level1Code,CouncilArea2019Code",
                    [f"{lau},{ca}" for _oa, lau, ca, *_ in _SCOT_ROWS],
                ),
            }
        ),
        "scotland_constituency": _zip(
            {
                "OA22_UKPC24.csv": _csv(
                    "OA22,UKPC24",
                    [f"{oa},{pcon}" for oa, _lau, _ca, pcon, *_ in _SCOT_ROWS],
                )
            }
        ),
        "scotland_population": _csv(
            "OutputArea2022,UsualResidentPopulation",
            [f"{oa},{pop}" for oa, *_, pop in _SCOT_ROWS],
        ).encode(),
        "scotland_census_index": _zip(
            {
                "Census_2022_Index/OA_TO_HIGHER_AREAS.csv": _csv(
                    "OA2022,CA2019,EW2022,DZ2011",
                    [
                        f"{oa},{ca},{ward},S01006755"
                        for oa, _lau, ca, _p, ward, *_ in _SCOT_ROWS
                    ],
                ),
                "Census_2022_Index/Postcode_To_OA.csv": _csv(
                    "Postcode,OutputArea2022Code,HouseholdCount,PopulationCount",
                    [
                        f"AB1 1A{i},{oa},{20 + i},{pop}"
                        for i, (oa, *_, pop) in enumerate(_SCOT_ROWS)
                    ]
                    + [f"AB1 1AZ,{_SCOT_ROWS[0][0]},5,{_SCOT_ROWS[0][-1]}"],
                ),
            }
        ),
        "ni_geojson": _zip(
            {
                "DZ2021.geojson": json.dumps(
                    {"type": "FeatureCollection", "features": ni_features}
                )
            }
        ),
        "ni_population": _csv(
            "Census 2021 Data Zone Code,Count",
            [f"N200000{i:02d},{700 + i}" for i in ni_index],
        ).encode(),
        "ni_households": _csv(
            "Census 2021 Data Zone Code,Count",
            [f"N200000{i:02d},{300 + i}" for i in ni_index],
        ).encode(),
        "ni_parlcon24_lookup": _ni_workbook(
            missing_constituency=ni_missing_constituency
        ),
    }


def crosswalk_resource(ew_rows: list[tuple] | None = None) -> dict:
    ew_rows = _EW_ROWS if ew_rows is None else ew_rows
    constituencies = [f"{oa[0]}14{oa[3:]}" for oa, *_ in ew_rows[:-1]] + ["W07000041"]
    constituencies += [pcon for _oa, _lau, _ca, pcon, *_ in _SCOT_ROWS]
    constituencies += [f"N050000{i:02d}" for i in range(1, _NI_COUNT + 1)]
    local_authorities = sorted({lad24 for _oa, _lad22, lad24, *_ in ew_rows})
    local_authorities += [ca for _oa, _lau, ca, *_ in _SCOT_ROWS] + ["N09000001"]
    return {
        "levels": {
            "constituency": {"area_ids": constituencies},
            "local_authority": {"area_ids": local_authorities},
        }
    }


@pytest.fixture(scope="module")
def tool():
    import microcosm.build

    path = (
        Path(microcosm.build.__file__).resolve().parents[5]
        / "tools/build_uk_atomic_area_supports.py"
    )
    if not path.is_file():
        path = (
            Path(__file__).resolve().parents[3]
            / "tools/build_uk_atomic_area_supports.py"
        )
    spec = importlib.util.spec_from_file_location("uk_supports_tool_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def national_counts(monkeypatch):
    monkeypatch.setattr(geography_sources, "ENGLAND_WALES_OA2021_COUNT", len(_EW_ROWS))
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", len(_SCOT_ROWS))
    monkeypatch.setattr(geography_sources, "NI_DZ2021_COUNT", _NI_COUNT)


def write_inputs(
    tmp_path: Path, tool, files: dict[str, bytes], crosswalk: dict
) -> dict:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    manifest_sources = {}
    for key, payload in files.items():
        path = inputs / str(tool.SOURCES[key]["name"])
        path.write_bytes(payload)
        manifest_sources[key] = {
            "path": str(path),
            "url": tool.SOURCES[key]["url"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema_version": 1, "sources": manifest_sources}))
    resource = tmp_path / "local_area_crosswalk.json"
    resource.write_text(json.dumps(crosswalk))
    return {"manifest": manifest, "resource": resource, "sources": manifest_sources}


def cli_args(tmp_path: Path, inputs: dict, *extra: str) -> list[str]:
    return [
        "--out-dir",
        str(tmp_path / "supports"),
        "--source-manifest",
        str(inputs["manifest"]),
        "--crosswalk-resource",
        str(inputs["resource"]),
        "--provenance-json",
        str(tmp_path / "provenance.json"),
        "--cache-dir",
        str(tmp_path / "unused-cache"),
        *extra,
    ]


def test_pinned_offline_tool_builds_all_three_supports(tmp_path, tool, national_counts):
    inputs = write_inputs(tmp_path, tool, invented_files(), crosswalk_resource())
    tool.main(cli_args(tmp_path, inputs))

    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["kind"] == "uk_atomic_area_supports_provenance"
    assert set(provenance["publisher_files"]) == set(tool.SOURCES)
    for key, row in provenance["publisher_files"].items():
        assert row["sha256"] == inputs["sources"][key]["sha256"]
        assert row["size_bytes"] == inputs["sources"][key]["bytes"]
        assert row["chronicle_package_id"] == tool.SOURCES[key]["chronicle_package_id"]
    assert provenance["ladder_reference"] is None and provenance["ladder_diff"] is None
    assert not (tmp_path / "unused-cache").exists()

    payloads = {}
    for system in SYSTEMS:
        path = tmp_path / "supports" / f"{SOURCES[system]}.npz"
        payloads[system] = path.read_bytes()
        support = decode_atomic_support(payloads[system])
        assert provenance["supports"][system]["sha256"] == support.sha256
        assert provenance["supports"][system]["areas"] == len(support.arrays["area"])
    uk_atomic_assignment_definition(payloads, seed=42)

    ew = decode_atomic_support(payloads[EW])
    by_area = dict(
        zip(ew.arrays["area"], ew.arrays["local_authority_code"], strict=True)
    )
    assert by_area["E00000005"] == "E06000063"  # LAD24 from the December 2024 file
    regions = dict(zip(ew.arrays["area"], ew.arrays["region_code"], strict=True))
    assert regions["E00000005"] == "E12000004" and regions["W00000001"] == "W99999999"
    wards = dict(zip(ew.arrays["area"], ew.arrays["ward_code"], strict=True))
    assert wards["E00000001"] == "E05000001"  # WD24; the file's LAD column is ignored
    itl = dict(zip(ew.arrays["area"], ew.arrays["itl3_code"], strict=True))
    assert itl["E00000005"] == "TLF11"
    parncp_sha = inputs["sources"]["oa_parncp_lad_region"]["sha256"]
    assert ew.metadata["columns"]["local_authority_code"] == {
        "kind": "code",
        "source": f"ons-oa21-parncp-lad-rgn-ctry-dec2024-lookup@sha256:{parncp_sha}",
        "vintage": "2023_april_lad",
        "relation": "best_fit",
    }
    assert ew.metadata["columns"]["households"]["basis"] == "census_2021_households"

    scotland = decode_atomic_support(payloads[SCOT])
    assert scotland.arrays["households"].tolist() == [25, 21]  # postcode rows summed
    assert scotland.arrays["region_code"].tolist() == ["S99999999", "S99999999"]
    assert scotland.metadata["columns"]["constituency_code"]["relation"] == "best_fit"

    ni = decode_atomic_support(payloads[NI])
    assert len(ni.arrays["area"]) == _NI_COUNT
    assert np.array_equal(ni.arrays["lsoa_code"], ni.arrays["area"])
    assert (
        ni.metadata["columns"]["constituency_code"]["relation"] == "official_tabulation"
    )
    assert provenance["supports"][NI]["constituencies"] == _NI_COUNT


def _ladder_from_supports(tmp_path: Path, supports_dir: Path) -> Path:
    frames = []
    for system in SYSTEMS:
        support = decode_atomic_support(
            (supports_dir / f"{SOURCES[system]}.npz").read_bytes()
        )
        frame = pd.DataFrame(
            {
                column: support.arrays["area" if column == "oa_code" else column]
                for column in tool_columns()
            }
        )
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)

    def layer(vintage: str) -> dict:
        countries = {"vintage": vintage, "source": "synthetic"}
        return {
            "vintage": vintage,
            "source": "synthetic",
            "countries": {
                "england_and_wales": countries,
                "scotland": countries,
                "northern_ireland": countries,
            },
        }

    metadata = {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "ew:2021;scotland:2022;ni:2021",
        "constituency_sampling_basis": "synthetic",
        "oa_sampling_basis": "synthetic",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("composite"),
            "msoa": layer("composite"),
            "local_authority": layer("composite"),
            "ward": layer("composite"),
            "itl": {"vintage": "2025_itl", "source": "synthetic"},
            "region": layer("composite"),
        },
    }
    path = tmp_path / "ladder.npz"
    np.savez_compressed(path, **assemble_uk_oa_ladder(frame, metadata))
    load_uk_oa_ladder(path)
    return path


def tool_columns() -> tuple[str, ...]:
    from microcosm.build.uk_runtime.oa_ladder_sources import LADDER_OA_COLUMNS

    return LADDER_OA_COLUMNS


def test_ladder_diff_reports_zero_moves_and_refuses_a_moved_local_authority(
    tmp_path, tool, national_counts
):
    inputs = write_inputs(tmp_path, tool, invented_files(), crosswalk_resource())
    tool.main(cli_args(tmp_path, inputs))
    ladder = _ladder_from_supports(tmp_path, tmp_path / "supports")

    tool.main(cli_args(tmp_path, inputs, "--ladder", str(ladder)))
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["ladder_reference"] == {
        "path": "ladder.npz",
        "sha256": hashlib.sha256(ladder.read_bytes()).hexdigest(),
    }
    assert set(provenance["ladder_diff"]["changed_areas_by_column"].values()) == {0}
    assert provenance["ladder_diff"]["unexpected_moves"] == {}

    with np.load(ladder) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["local_authority_code"] = arrays["local_authority_code"].copy()
    arrays["local_authority_code"][0] = "E06000099"
    moved = tmp_path / "moved.npz"
    np.savez_compressed(moved, **arrays)
    with pytest.raises(ValueError, match="unexpected array moves"):
        tool.main(cli_args(tmp_path, inputs, "--ladder", str(moved)))


@pytest.mark.parametrize(
    "defect",
    ["lad_2025_recode", "oa_count", "ni_constituency_set", "zero_population", "roster"],
)
def test_pinned_offline_tool_refuses_source_defects(
    tmp_path, tool, national_counts, defect
):
    files = invented_files(ni_missing_constituency=defect == "ni_constituency_set")
    crosswalk = crosswalk_resource()
    if defect == "lad_2025_recode":
        files["oa_parncp_lad_region"] = files["oa_parncp_lad_region"].replace(
            b"E08000016,l", b"E08000038,l"
        )
        crosswalk["levels"]["local_authority"]["area_ids"].append("E08000038")
        files["lad_itl"] += b"TLE,x,TLE3,x,TLE31,x,E30000038,x,E08000038,x,9\n"
        message = "April 2023 code set"
    elif defect == "oa_count":
        lines = files["oa_parncp_lad_region"].decode().splitlines()
        files["oa_parncp_lad_region"] = ("\n".join(lines[:-1]) + "\n").encode()
        message = "cover every OA2021"
    elif defect == "ni_constituency_set":
        message = "N05000001-N05000018"
    elif defect == "zero_population":
        files["oa_population"] = _zip(
            {
                "census2021-ts001-oa.csv": _csv(
                    "date,geography,geography code,Residence type: Total; measures: Value",
                    [
                        f"2021,{oa},{oa},{0 if oa == 'E00000001' else pop}"
                        for oa, *_, pop, _hh in _EW_ROWS
                    ],
                )
            }
        )
        message = "zero population"
    else:
        crosswalk["levels"]["constituency"]["area_ids"].remove("E14000001")
        message = "constituency roster differs"
    inputs = write_inputs(tmp_path, tool, files, crosswalk)
    with pytest.raises(ValueError, match=message):
        tool.main(cli_args(tmp_path, inputs))
    assert not (tmp_path / "supports").exists()
    assert not (tmp_path / "provenance.json").exists()


@pytest.mark.parametrize("defect", ["sha256", "bytes", "url", "missing", "extra"])
def test_pinned_offline_tool_refuses_bad_envelope(
    tmp_path, tool, national_counts, defect
):
    inputs = write_inputs(tmp_path, tool, invented_files(), crosswalk_resource())
    document = json.loads(inputs["manifest"].read_text())
    row = document["sources"]["oa_hierarchy"]
    if defect == "sha256":
        row["sha256"] = "0" * 64
    elif defect == "bytes":
        row["bytes"] += 1
    elif defect == "url":
        row["url"] = "https://example.invalid/wrong-source"
    elif defect == "missing":
        del document["sources"]["ni_geojson"]
    else:
        document["sources"]["retired_lad_region"] = dict(row)
    inputs["manifest"].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="[Pp]inned source"):
        tool.main(cli_args(tmp_path, inputs))
    assert not (tmp_path / "supports").exists()
    assert not (tmp_path / "unused-cache").exists()


def test_source_register_matches_the_tool_column_tables(tool):
    """Every column source key exists and every publisher file feeds a column."""
    used = set()
    for system in SYSTEMS:
        for key, _vintage, relation in tool.COLUMN_SOURCES[system].values():
            assert key in tool.SOURCES
            assert relation in {"exact", "best_fit", "official_tabulation"}
            used.add(key)
        for key, _basis in tool.COUNT_SOURCES[system].values():
            assert key in tool.SOURCES
            used.add(key)
    assert used == set(tool.SOURCES)
    assert tool.COLUMN_SOURCES[NI]["constituency_code"][2] == "official_tabulation"
    assert all(
        tool.COLUMN_SOURCES[system]["itl3_code"] == ("lad_itl", "2025_itl", "exact")
        for system in SYSTEMS
    )
    for spec in tool.SOURCES.values():
        assert spec["chronicle_source_id"] in {"ons", "nrs", "nisra"}
        assert spec["chronicle_package_id"].startswith(
            spec["chronicle_source_id"] + "-"
        )
