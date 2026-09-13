"""Tests for the pure PUMA-ladder source parsers and assembler."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime.puma_ladder_sources import (
    assemble_us_puma_ladder,
    parse_tract_to_puma_relationship,
)

# Four populated blocks: PUMA 0100100 spans counties 01001/01003 and CDs
# 101/102; PUMA 0100200 sits in county 01003; PUMA 0200100 is the state-02
# at-large district. Block ints drop the leading state zero, exactly as the
# P.L. 94-171 and CD BEF parsers return them.
_BLOCK_POPULATION = {
    10010001001000: 900,  # tract 01001000100, county 01001
    10030001001000: 100,  # tract 01003000100, county 01003
    10030002001000: 500,  # tract 01003000200, county 01003
    20130001001000: 400,  # tract 02013000100, county 02013
}
_CD_BY_BLOCK = {
    10010001001000: 101,
    10030001001000: 102,
    10030002001000: 102,
    20130001001000: 200,
}
_TRACT_TO_PUMA = {
    1001000100: 100100,
    1003000100: 100100,
    1003000200: 100200,
    2013000100: 200100,
}
_METADATA = {
    "schema_version": 2,
    "kind": "us_puma_ladder",
    "puma_vintage": "2020_puma",
    "sampling_basis": "population",
    "layers": {
        "congressional_district": {"vintage": "119th_congress", "source": "cd119"},
        "county": {"vintage": "2020_census", "source": "tract-to-puma"},
        "tract": {"vintage": "2020_census", "source": "tract-to-puma"},
    },
}


def _relationship_lines(extra: list[str] | None = None) -> list[str]:
    lines = [
        "﻿STATEFP,COUNTYFP,TRACTCE,PUMA5CE",
        "01,001,000100,00100",
        "01,003,000100,00100",
        "01,003,000200,00200",
        "02,013,000100,00100",
        "72,001,000100,00100",  # Puerto Rico — filtered out.
    ]
    return lines + (extra or [])


def test_parse_tract_to_puma_filters_territories_and_builds_geoids() -> None:
    mapping = parse_tract_to_puma_relationship(
        _relationship_lines(), allowed_state_fips=frozenset({"01", "02"})
    )

    assert mapping == _TRACT_TO_PUMA
    # The Puerto Rico tract is excluded by the state filter.
    assert 72001000100 not in mapping


def test_parse_tract_to_puma_keeps_all_states_without_a_filter() -> None:
    mapping = parse_tract_to_puma_relationship(_relationship_lines())

    assert 72001000100 in mapping
    assert mapping[72001000100] == 7200100


def test_parse_tract_to_puma_rejects_a_wrong_header() -> None:
    with pytest.raises(ValueError, match="header must be"):
        parse_tract_to_puma_relationship(["STATE,COUNTY,TRACT,PUMA", "01,001,1,1"])


def test_parse_tract_to_puma_rejects_a_malformed_row() -> None:
    with pytest.raises(ValueError, match="four fields"):
        parse_tract_to_puma_relationship(
            ["STATEFP,COUNTYFP,TRACTCE,PUMA5CE", "01,001,000100"]
        )


def test_parse_tract_to_puma_rejects_a_bad_width() -> None:
    with pytest.raises(ValueError, match="5-digit"):
        parse_tract_to_puma_relationship(
            ["STATEFP,COUNTYFP,TRACTCE,PUMA5CE", "01,001,000100,100"]
        )


def test_parse_tract_to_puma_rejects_conflicting_pumas() -> None:
    with pytest.raises(ValueError, match="both PUMA"):
        parse_tract_to_puma_relationship(
            [
                "STATEFP,COUNTYFP,TRACTCE,PUMA5CE",
                "01,001,000100,00100",
                "01,001,000100,00200",
            ]
        )


def test_assemble_builds_conserving_overlap_tables() -> None:
    payload = assemble_us_puma_ladder(
        block_population=_BLOCK_POPULATION,
        cd_by_block=_CD_BY_BLOCK,
        tract_to_puma=_TRACT_TO_PUMA,
        metadata=_METADATA,
    )

    assert payload["puma"].tolist() == [100100, 100200, 200100]
    assert payload["puma_population"].tolist() == [1000, 500, 400]

    # CD overlap sorted by (puma, cd), conserving each PUMA's population.
    assert list(
        zip(
            payload["cd_overlap_puma"].tolist(),
            payload["cd_overlap_cd"].tolist(),
            payload["cd_overlap_population"].tolist(),
            strict=True,
        )
    ) == [
        (100100, 101, 900),
        (100100, 102, 100),
        (100200, 102, 500),
        (200100, 200, 400),
    ]
    # County overlap: PUMA 0100100 straddles counties 01001 and 01003.
    assert list(
        zip(
            payload["county_overlap_puma"].tolist(),
            payload["county_overlap_county"].tolist(),
            payload["county_overlap_population"].tolist(),
            strict=True,
        )
    ) == [
        (100100, 1001, 900),
        (100100, 1003, 100),
        (100200, 1003, 500),
        (200100, 2013, 400),
    ]
    assert payload["tract_overlap_tract"].tolist() == [
        1001000100,
        1003000100,
        1003000200,
        2013000100,
    ]
    metadata = json.loads(str(payload["metadata_json"]))
    assert metadata["kind"] == "us_puma_ladder"


def test_assemble_refuses_a_block_with_no_puma() -> None:
    with pytest.raises(ValueError, match="tract absent from the tract-to-PUMA"):
        assemble_us_puma_ladder(
            block_population={**_BLOCK_POPULATION, 30070001001000: 50},
            cd_by_block={**_CD_BY_BLOCK, 30070001001000: 301},
            tract_to_puma=_TRACT_TO_PUMA,  # no tract 3007000100
            metadata=_METADATA,
        )


def test_assemble_refuses_a_block_with_no_congressional_district() -> None:
    with pytest.raises(ValueError, match="no congressional district"):
        assemble_us_puma_ladder(
            block_population=_BLOCK_POPULATION,
            cd_by_block={
                key: value
                for key, value in _CD_BY_BLOCK.items()
                if key != 20130001001000
            },
            tract_to_puma=_TRACT_TO_PUMA,
            metadata=_METADATA,
        )


def test_assemble_ignores_zero_population_blocks() -> None:
    payload = assemble_us_puma_ladder(
        block_population={**_BLOCK_POPULATION, 10010001009000: 0},
        cd_by_block=_CD_BY_BLOCK,
        tract_to_puma=_TRACT_TO_PUMA,
        metadata=_METADATA,
    )

    # The zero-population block never needs a CD or tract lookup and does not
    # change any PUMA's conserved total.
    assert payload["puma_population"].tolist() == [1000, 500, 400]
    assert np.asarray(payload["cd_overlap_population"]).sum() == 1900


def test_assembly_retains_joint_cells_without_crossing_marginals() -> None:
    payload = assemble_us_puma_ladder(
        block_population=_BLOCK_POPULATION,
        cd_by_block=_CD_BY_BLOCK,
        tract_to_puma=_TRACT_TO_PUMA,
        metadata=_METADATA,
    )
    assert list(
        zip(
            payload["joint_overlap_puma"].tolist(),
            payload["joint_overlap_tract"].tolist(),
            payload["joint_overlap_cd"].tolist(),
            payload["joint_overlap_population"].tolist(),
            strict=True,
        )
    ) == [
        (100100, 1001000100, 101, 900),
        (100100, 1003000100, 102, 100),
        (100200, 1003000200, 102, 500),
        (200100, 2013000100, 200, 400),
    ]


def test_assembly_preserves_a_tract_split_between_districts() -> None:
    block_population = {**_BLOCK_POPULATION, 10010001001001: 80}
    cd_by_block = {**_CD_BY_BLOCK, 10010001001001: 102}
    payload = assemble_us_puma_ladder(
        block_population=block_population,
        cd_by_block=cd_by_block,
        tract_to_puma=_TRACT_TO_PUMA,
        metadata=_METADATA,
    )
    assert payload["joint_overlap_tract"][:2].tolist() == [1001000100, 1001000100]
    assert payload["joint_overlap_cd"][:2].tolist() == [101, 102]
    assert payload["joint_overlap_population"][:2].tolist() == [900, 80]
    assert payload["puma_population"].tolist() == [1080, 500, 400]


def test_assembly_refuses_legacy_schema_label_for_joint_payload() -> None:
    with pytest.raises(ValueError, match="requires schema_version 2"):
        assemble_us_puma_ladder(
            block_population=_BLOCK_POPULATION,
            cd_by_block=_CD_BY_BLOCK,
            tract_to_puma=_TRACT_TO_PUMA,
            metadata={**_METADATA, "schema_version": 1},
        )


@pytest.fixture(scope="module")
def ladder_tool():
    import microcosm.build

    # Resolve the actual source checkout even when ordinary tests are copied
    # to isolated scratch. Wheel tests retain the repository-side tool fixture.
    path = (
        Path(microcosm.build.__file__).resolve().parents[5]
        / "tools/build_us_puma_ladder_artifact.py"
    )
    if not path.is_file():
        path = (
            Path(__file__).resolve().parents[3]
            / "tools/build_us_puma_ladder_artifact.py"
        )
    spec = importlib.util.spec_from_file_location("puma_ladder_tool_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invented_pinned_sources(tmp_path, tool):
    import zipfile

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tract = inputs / "tract.txt"
    tract.write_text("STATEFP,COUNTYFP,TRACTCE,PUMA5CE\n01,001,000100,00100\n")
    cd = inputs / "cd119.zip"
    with zipfile.ZipFile(cd, "w") as archive:
        archive.writestr(
            "NationalCD119.txt",
            "GEOID,CDFP\n010010001001000,01\n010010001001001,02\n",
        )
    pl = inputs / "al2020.pl.zip"

    def line(level, geocode, population):
        fields = [""] * 97
        fields[2], fields[9], fields[90] = level, geocode, str(population)
        return "|".join(fields) + "\n"

    with zipfile.ZipFile(pl, "w") as archive:
        archive.writestr(
            "algeo2020.pl",
            line("040", "01", 100)
            + line("750", "010010001001000", 90)
            + line("750", "010010001001001", 10),
        )
    locators = {
        "tract_to_puma": (tract, tool.TRACT_TO_PUMA_URL),
        "cd119_bef": (cd, tool.CD119_BEF_URL),
        "pl94171_al": (
            pl,
            tool.PL94171_URL_TEMPLATE.format(dirname="Alabama", usps_lower="al"),
        ),
    }
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    name: {
                        "path": str(path),
                        "url": url,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for name, (path, url) in locators.items()
                },
            }
        )
    )
    return manifest, tract


def _pinned_cli_args(tmp_path, manifest):
    return [
        "--out",
        str(tmp_path / "ladder.npz"),
        "--states",
        "01",
        "--source-manifest",
        str(manifest),
        "--cache-dir",
        str(tmp_path / "unused-cache"),
    ]


def test_pinned_offline_tool_builds_actual_joint_npz(tmp_path, ladder_tool) -> None:
    from microcosm.build.us_runtime import load_us_puma_ladder

    manifest, _ = _invented_pinned_sources(tmp_path, ladder_tool)
    ladder_tool.main(_pinned_cli_args(tmp_path, manifest))
    ladder = load_us_puma_ladder(tmp_path / "ladder.npz")
    # One observed source tract can cross two districts within the same PUMA.
    assert ladder.joint_overlap_tract.tolist() == [1001000100, 1001000100]
    assert ladder.joint_overlap_cd.tolist() == [101, 102]
    assert ladder.joint_overlap_population.tolist() == [90, 10]
    assert (
        ladder.metadata["source_manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert not (tmp_path / "unused-cache").exists()


@pytest.mark.parametrize("defect", ["sha256", "bytes", "url", "missing", "extra"])
def test_pinned_offline_tool_refuses_bad_envelope(
    tmp_path, ladder_tool, defect
) -> None:
    manifest, _ = _invented_pinned_sources(tmp_path, ladder_tool)
    document = json.loads(manifest.read_text())
    row = document["sources"]["tract_to_puma"]
    if defect == "sha256":
        row["sha256"] = "0" * 64
    elif defect == "bytes":
        row["bytes"] += 1
    elif defect == "url":
        row["url"] = "https://example.invalid/wrong-source"
    elif defect == "missing":
        del document["sources"]["pl94171_al"]
    else:
        document["sources"]["unselected_state"] = dict(row)
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="[Pp]inned source"):
        ladder_tool.main(_pinned_cli_args(tmp_path, manifest))
    assert not (tmp_path / "ladder.npz").exists()
    assert not (tmp_path / "unused-cache").exists()


@pytest.mark.parametrize("mutation", ["payload", "manifest"])
def test_pinned_offline_tool_rechecks_after_parsing(
    tmp_path, ladder_tool, monkeypatch, mutation
) -> None:
    manifest, tract = _invented_pinned_sources(tmp_path, ladder_tool)
    original = ladder_tool._text_lines

    def mutate_after_read(path):
        yield from original(path)
        target = tract if mutation == "payload" else manifest
        target.write_bytes(target.read_bytes() + b"\n")

    monkeypatch.setattr(ladder_tool, "_text_lines", mutate_after_read)
    with pytest.raises(ValueError, match="Pinned source"):
        ladder_tool.main(_pinned_cli_args(tmp_path, manifest))
    assert not (tmp_path / "ladder.npz").exists()
