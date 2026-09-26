"""The committed UK atomic-area support register agrees with the spec bundle.

`uk/uk_atomic_area_supports.provenance.json` is the sidecar that pins the three
support artifacts and their sixteen publisher inputs; `uk/spec/sources.yaml`
pins the same three artifacts as typed sources with their vintages. The two
must never drift apart. The support bytes themselves are run-time inputs
(``build/`` is not committed); when they are mounted, the last tests check the
committed register against the actual bytes and the pinned OA ladder.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import microcosm.build
from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.country_spec import load_country_spec
from microcosm.build.spec_engine import load_bundle
from microcosm.build.spec_engine.compiler_ir import compile_spec
from microcosm.build.uk_runtime.atomic_area_support import (
    SOURCES,
    SYSTEMS,
    uk_atomic_assignment_definition,
)
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

EW, SCOT, NI = SYSTEMS
_UK = Path(microcosm.build.__file__).resolve().parent / "uk"
_REPOSITORY = Path(microcosm.build.__file__).resolve().parents[5]
_SHA = re.compile(r"^[0-9a-f]{64}$")

#: Pinned mapping classification per support column (publisher's own terms).
RELATIONS = {
    EW: {
        "area": "exact",
        "lsoa_code": "exact",
        "msoa_code": "exact",
        "local_authority_code": "best_fit",
        "region_code": "best_fit",
        "constituency_code": "best_fit",
        "ward_code": "best_fit",
        "itl3_code": "exact",
    },
    SCOT: {
        "area": "exact",
        "lsoa_code": "exact",
        "msoa_code": "exact",
        "local_authority_code": "exact",
        "region_code": "exact",
        "constituency_code": "best_fit",
        "ward_code": "best_fit",
        "itl3_code": "exact",
    },
    NI: {
        "area": "exact",
        "lsoa_code": "exact",
        "msoa_code": "exact",
        "local_authority_code": "exact",
        "region_code": "exact",
        "constituency_code": "official_tabulation",
        "ward_code": "exact",
        "itl3_code": "exact",
    },
}
#: Publisher vintages per support column; the frs_region sentinel is not a
#: publisher vintage and is excluded from the spec-side check.
VINTAGES = {
    EW: {
        "area": "2021_census",
        "lsoa_code": "2021_census",
        "msoa_code": "2021_census",
        "local_authority_code": "2023_april_lad",
        "region_code": "2024_rgn",
        "constituency_code": "2024_pcon",
        "ward_code": "2024_wd",
        "itl3_code": "2025_itl",
    },
    SCOT: {
        "area": "2022_census",
        "lsoa_code": "2022_census",
        "msoa_code": "2022_census",
        "local_authority_code": "2019_council_area",
        "region_code": "frs_region_sentinel",
        "constituency_code": "2024_pcon",
        "ward_code": "2022_ew",
        "itl3_code": "2025_itl",
    },
    NI: {
        "area": "2021_census",
        "lsoa_code": "2021_census",
        "msoa_code": "2021_census",
        "local_authority_code": "2014_lgd",
        "region_code": "frs_region_sentinel",
        "constituency_code": "2024_pcon",
        "ward_code": "2014_dea",
        "itl3_code": "2025_itl",
    },
}
EXPECTED_AREAS = {EW: 188_880, SCOT: 46_363, NI: 3_780}


@pytest.fixture(scope="module")
def provenance() -> dict:
    return json.loads(
        (_UK / "uk_atomic_area_supports.provenance.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def bundle():
    return compile_spec(load_bundle("uk"))


def _spec_sources(bundle) -> dict[str, dict]:
    return {row["id"]: row for row in bundle.resource("sources")["sources"]}


def test_register_shape_and_publisher_files(provenance) -> None:
    assert provenance["schema_version"] == 1
    assert provenance["kind"] == "uk_atomic_area_supports_provenance"
    assert provenance["tool"] == "tools/build_uk_atomic_area_supports.py"
    assert provenance["reviewed_differences"] == []
    files = provenance["publisher_files"]
    assert len(files) == 16
    for key, row in files.items():
        assert _SHA.match(row["sha256"]), key
        assert row["size_bytes"] > 0
        assert row["chronicle_source_id"] in {"ons", "nrs", "nisra"}
        assert row["chronicle_package_id"].startswith(row["chronicle_source_id"] + "-")
        assert row["url"].startswith("https://")
        assert row["vintage"]
    assert files["oa_parncp_lad_region"]["vintage"] == "2023_april_lad"
    assert files["oa_ward"]["vintage"] == "2024_wd"
    assert files["lad_itl"]["vintage"] == "2025_itl"
    assert files["oa_constituency"]["vintage"] == "2024_pcon"
    # Retired products are not inputs.
    assert not {"oa_lad23", "lad_region"} & set(files)
    assert provenance["ladder_reference"]["path"] == "uk_oa_ladder_2021.npz"


def test_register_agrees_with_the_pinned_ladder_and_reports_only_ward_and_itl_moves(
    provenance,
) -> None:
    crosswalk = json.loads((_UK / "local_area_crosswalk.json").read_text())
    assert (
        provenance["ladder_reference"]["sha256"] == crosswalk["ladder_artifact_sha256"]
    )
    diff = provenance["ladder_diff"]
    assert diff["areas"] == sum(EXPECTED_AREAS.values())
    assert diff["unexpected_moves"] == {}
    changed = diff["changed_areas_by_column"]
    for column in (
        "lsoa_code",
        "msoa_code",
        "local_authority_code",
        "constituency_code",
        "region_code",
        "population",
        "households",
    ):
        assert changed[column] == 0, column
    assert changed["ward_code"] > 0 and changed["itl3_code"] > 0


def test_supports_summary_matches_the_national_rosters(provenance) -> None:
    crosswalk = json.loads((_UK / "local_area_crosswalk.json").read_text())
    supports = provenance["supports"]
    assert set(supports) == set(SYSTEMS)
    for system, expected in EXPECTED_AREAS.items():
        entry = supports[system]
        assert entry["filename"] == f"{SOURCES[system]}.npz"
        assert entry["areas"] == expected
        assert _SHA.match(entry["sha256"])
        assert 0 < entry["size_bytes"] <= RAW_BYTES_MAX_BYTES
        assert entry["population_total"] > entry["households_total"] > 0
    assert sum(e["constituencies"] for e in supports.values()) == len(
        crosswalk["levels"]["constituency"]["area_ids"]
    )
    assert sum(e["local_authorities"] for e in supports.values()) == len(
        crosswalk["levels"]["local_authority"]["area_ids"]
    )
    assert supports[NI]["constituencies"] == 18


def test_column_metadata_pins_relations_vintages_and_publisher_shas(provenance) -> None:
    files = provenance["publisher_files"]
    sha_by_package = {
        row["chronicle_package_id"]: row["sha256"] for row in files.values()
    }
    for system in SYSTEMS:
        columns = provenance["supports"][system]["column_metadata"]
        for column, relation in RELATIONS[system].items():
            assert columns[column]["relation"] == relation, (system, column)
            assert columns[column]["vintage"] == VINTAGES[system][column], (
                system,
                column,
            )
            package, _, digest = columns[column]["source"].partition("@sha256:")
            digest = digest.split("#")[0]
            assert sha_by_package[package] == digest, (system, column)
        for column in ("population", "households"):
            package, _, digest = columns[column]["source"].partition("@sha256:")
            assert sha_by_package[package] == digest, (system, column)
        assert columns["itl2_code"]["source"].endswith("#prefix-4")
        assert columns["itl1_code"]["source"].endswith("#prefix-3")
        assert columns["frs_region"]["vintage"] == "frs_region_enum_v1"


def test_spec_sources_pin_the_same_bytes_and_every_column_vintage(
    provenance, bundle
) -> None:
    sources = _spec_sources(bundle)
    records = {row["id"]: row for row in bundle.resource("vintages")["records"]}
    authority_values = {}
    for row in sources.values():
        for authority in row.get("vintage_authorities", []):
            authority_values[authority["id"]] = authority["value"]
    for system in SYSTEMS:
        row = sources[SOURCES[system]]
        entry = provenance["supports"][system]
        assert row["role"] == "uk_atomic_area_support"
        assert row["loader"] == "kernel:load_uk_atomic_area_support"
        assert row["sha256"] == entry["sha256"]
        assert row["byte_size"] == entry["size_bytes"]
        declared = set()
        for reference in row["vintages"]:
            vintage_id = reference.removeprefix("vintage:")
            record = records[vintage_id]
            assert record["kind"] == "geography_vintage_ref"
            assert record["authority_ref"]["authority"] == vintage_id
            declared.add(authority_values[vintage_id])
        column_vintages = {
            vintage
            for vintage in VINTAGES[system].values()
            if vintage != "frs_region_sentinel"
        }
        assert column_vintages <= declared, (system, column_vintages - declared)
    geography = [r for r in records.values() if r["kind"] == "geography_vintage_ref"]
    ids = {r["id"] for r in geography}
    for record in geography:
        assert set(
            map(lambda v: v.removeprefix("vintage:"), record["compatible_with"])
        ) == ids - {record["id"]}


def test_provenance_register_is_a_country_package_resource() -> None:
    spec = load_country_spec("uk")
    name = "uk_atomic_area_supports.provenance.json"
    assert name in spec.resource_hashes
    assert (
        spec.resource_hashes[name]
        == hashlib.sha256((_UK / name).read_bytes()).hexdigest()
    )


def _mounted(system: str) -> Path:
    path = _REPOSITORY / "build/uk/supports" / f"{SOURCES[system]}.npz"
    if not path.is_file():
        pytest.skip(f"{path.name} is not mounted under build/uk/supports")
    return path


@pytest.mark.parametrize("system", SYSTEMS)
def test_mounted_support_bytes_match_the_register(provenance, system) -> None:
    payload = _mounted(system).read_bytes()
    entry = provenance["supports"][system]
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert len(payload) == entry["size_bytes"] <= RAW_BYTES_MAX_BYTES
    support = decode_atomic_support(payload)
    assert len(support.arrays["area"]) == entry["areas"]
    assert int(support.arrays["population"].sum()) == entry["population_total"]
    assert int(support.arrays["households"].sum()) == entry["households_total"]
    assert {k: dict(v) for k, v in support.metadata["columns"].items()} == entry[
        "column_metadata"
    ]
    if system == NI:
        assert np.array_equal(support.arrays["lsoa_code"], support.arrays["area"])
        assert len(np.unique(support.arrays["constituency_code"])) == 18


def test_mounted_supports_form_an_accepted_definition_and_the_committed_rosters(
    provenance,
) -> None:
    payloads = {system: _mounted(system).read_bytes() for system in SYSTEMS}
    definition = uk_atomic_assignment_definition(payloads, seed=42)
    assert definition["systems"] if "systems" in definition else definition
    crosswalk = json.loads((_UK / "local_area_crosswalk.json").read_text())
    constituencies: set[str] = set()
    local_authorities: set[str] = set()
    for payload in payloads.values():
        support = decode_atomic_support(payload)
        constituencies |= set(support.arrays["constituency_code"].tolist())
        local_authorities |= set(support.arrays["local_authority_code"].tolist())
    assert constituencies == set(crosswalk["levels"]["constituency"]["area_ids"])
    assert local_authorities == set(crosswalk["levels"]["local_authority"]["area_ids"])


def test_mounted_supports_carry_the_pinned_ladder_counts_per_area(provenance) -> None:
    from microcosm.build.uk_runtime import load_uk_oa_ladder

    ladder_path = _REPOSITORY / "build/uk/uk_oa_ladder_2021.npz"
    if not ladder_path.is_file():
        pytest.skip("pinned OA ladder is not mounted under build/uk")
    assert (
        hashlib.sha256(ladder_path.read_bytes()).hexdigest()
        == provenance["ladder_reference"]["sha256"]
    )
    ladder = load_uk_oa_ladder(ladder_path)
    reference = {
        code: (int(population), int(households), la, pcon)
        for code, population, households, la, pcon in zip(
            ladder.oa_code,
            ladder.population,
            ladder.households,
            ladder.local_authority_code,
            ladder.constituency_code,
            strict=True,
        )
    }
    seen = 0
    for system in SYSTEMS:
        support = decode_atomic_support(_mounted(system).read_bytes())
        for code, population, households, la, pcon in zip(
            support.arrays["area"],
            support.arrays["population"],
            support.arrays["households"],
            support.arrays["local_authority_code"],
            support.arrays["constituency_code"],
            strict=True,
        ):
            assert reference[str(code)] == (int(population), int(households), la, pcon)
            seen += 1
    assert seen == len(reference)
