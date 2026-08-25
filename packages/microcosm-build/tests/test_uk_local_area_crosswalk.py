"""Guarantees for the UK local-area identity crosswalk resource."""

from __future__ import annotations

import json
from importlib import resources as importlib_resources
from pathlib import Path

from microcosm.build.country_spec import load_country_spec
from tools.generate_uk_local_area_crosswalk import build_local_area_crosswalk

LADDER_ARTIFACT = Path("build/uk/uk_oa_ladder_2021.npz")
LADDER_SUMMARY = Path("build/uk/ladder_summary.json")


def _load() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text()
    )


def test_uk_local_area_crosswalk_is_registered_in_the_country_package() -> None:
    spec = load_country_spec("uk")

    assert "local_area_crosswalk.json" in spec.resources
    assert "local_area_crosswalk.json" in spec.resource_hashes


def test_uk_local_area_crosswalk_matches_generator_output() -> None:
    committed = _load()
    generated = build_local_area_crosswalk(
        ladder_artifact=LADDER_ARTIFACT,
        ladder_summary=LADDER_SUMMARY,
    )

    assert committed == generated


def test_uk_local_area_crosswalk_pins_rosters_and_vintages() -> None:
    resource = _load()

    assert resource["country"] == "uk"
    assert resource["schema_version"] == 1
    assert resource["ladder_artifact_sha256"] == (
        "9c6d56b90d2e975d750106b175020a54c5ec6acf42ef8909d304a9d7fc3868a7"
    )

    constituency = resource["levels"]["constituency"]
    assert constituency["ladder_layer"] == "constituency"
    assert constituency["ladder_code_column"] == "constituency_code"
    assert constituency["ladder_vintage"] == "2024_pcon"
    assert constituency["expected_vintage"] == "pcon_2024"
    assert constituency["area_count"] == 650
    assert len(constituency["area_ids"]) == 650
    assert len(set(constituency["area_ids"])) == 650
    assert {area_id[:3] for area_id in constituency["area_ids"]} == {
        "E14",
        "N05",
        "S14",
        "W07",
    }

    local_authority = resource["levels"]["local_authority"]
    assert local_authority["ladder_layer"] == "local_authority"
    assert local_authority["ladder_code_column"] == "local_authority_code"
    assert local_authority["ladder_vintage"] == (
        "ew:2023_april_lad;scotland:2019_council_area;ni:2014_lgd"
    )
    assert local_authority["expected_vintage"] == {
        "E": "lad_2023",
        "W": "lad_2023",
        "S": "ca_2019",
        "N": "lgd_2014",
    }
    assert local_authority["area_count"] == 361
    assert len(local_authority["area_ids"]) == 361
    assert len(set(local_authority["area_ids"])) == 361
    assert {area_id[:3] for area_id in local_authority["area_ids"]} == {
        "E06",
        "E07",
        "E08",
        "E09",
        "N09",
        "S12",
        "W06",
    }
