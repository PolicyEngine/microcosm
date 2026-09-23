"""Guarantees for the UK local-area identity crosswalk resource."""

from __future__ import annotations

import json
from importlib import resources as importlib_resources
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.calibrate.geography_constants import UK_REGION_TIER
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
    if not LADDER_ARTIFACT.exists():
        pytest.skip("sha-pinned UK OA ladder artifact is not mounted")
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
        "bed3f13d3a82eea2d1f39248b71c0abf5ba6960a446ddd9415ae1dbcb7ae07fd"
    )

    constituency = resource["levels"]["constituency"]
    assert constituency["ladder_layer"] == "constituency"
    assert constituency["ladder_code_column"] == "constituency_code"
    assert constituency["ladder_vintage"] == "2024_pcon"
    assert constituency["ladder_layer_sources"]["northern_ireland"] == {
        "source": (
            "NISRA, Geography Data Zone and Super Data Zone Lookups V3, "
            "DZ2021 to PARLCON2024 published administrative lookup"
        ),
        "url": (
            "https://www.nisra.gov.uk/files/nisra/documents/2025-04/"
            "geography-data-zone-and-super-data-zone-lookups-v3.xlsx"
        ),
        "sha256": ("8e2e1f6daaddd2f5b6887ccab1bf9d0bf179f6d0d2ba60ef8e0ebb495a3e1999"),
        "vintage": "2024_pcon",
    }
    assert constituency["expected_vintage"] == ["pcon_2024"]
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
        "E": ["lad_2023"],
        "W": ["lad_2023"],
        "S": ["ca_2019", "lad_2023"],
        "N": ["lgd_2014", "lad_2023"],
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


def test_uk_local_area_crosswalk_carries_region_tier_membership() -> None:
    resource = _load()
    tier = {code for _, code in UK_REGION_TIER}
    nation_by_prefix = {"W": "W92000004", "S": "S92000003", "N": "N92000002"}
    for payload in resource["levels"].values():
        by_area = payload["region_code_by_area"]
        assert set(by_area) == set(payload["area_ids"])
        assert set(by_area.values()) <= tier
        for area_id, region in by_area.items():
            if area_id.startswith("E"):
                assert region.startswith("E12"), (area_id, region)
            else:
                assert region == nation_by_prefix[area_id[0]], (area_id, region)
    constituency = resource["levels"]["constituency"]["region_code_by_area"]
    local_authority = resource["levels"]["local_authority"]["region_code_by_area"]
    # London: 75 constituencies and 33 authorities (32 boroughs + the City);
    # the North East holds 12 authorities. Nation counts follow the rosters.
    assert sum(region == "E12000007" for region in constituency.values()) == 75
    assert sum(region == "E12000007" for region in local_authority.values()) == 33
    assert sum(region == "E12000001" for region in local_authority.values()) == 12
    assert sum(region == "S92000003" for region in local_authority.values()) == 32
    assert sum(region == "W92000004" for region in local_authority.values()) == 22
    assert sum(region == "N92000002" for region in local_authority.values()) == 11
