"""Tests split from packages/microcosm-build/tests/test_uk_local_authority_input.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_local_authority_input import *


def test_uk_local_authority_names_is_registered_in_the_country_package() -> None:
    spec = load_country_spec("uk")

    assert UK_LOCAL_AUTHORITY_NAMES_RESOURCE in spec.resources
    assert UK_LOCAL_AUTHORITY_NAMES_RESOURCE in spec.resource_hashes


def test_uk_local_authority_names_pins_source_and_roster() -> None:
    resource = _resource()

    assert resource["schema_version"] == 1
    assert resource["country"] == "uk"
    assert resource["kind"] == UK_LOCAL_AUTHORITY_NAMES_KIND
    assert resource["source"] == {
        "name": (
            "Local Authority Districts (April 2023) Names and Codes in the "
            "United Kingdom"
        ),
        "publisher": "ONS Open Geography Portal",
        "item_id": LAD23_NAMES_ITEM_ID,
        "url": LAD23_NAMES_URL,
        "sha256": SOURCE_SHA256,
        "columns": ["LAD23CD", "LAD23NM"],
        "vintage": UK_LOCAL_AUTHORITY_VINTAGE,
    }
    assert LAD23_NAMES_ITEM_ID in LAD23_NAMES_URL
    assert LAD23_NAMES_SHA256 == SOURCE_SHA256
    assert resource["area_count"] == 361 == len(resource["areas"])
    by_nation: dict[str, int] = {}
    for code in resource["areas"]:
        by_nation[code[0]] = by_nation.get(code[0], 0) + 1
    assert by_nation == {"E": 296, "W": 22, "S": 32, "N": 11}
    assert list(resource["areas"]) == sorted(resource["areas"])
    # The validating loader accepts the committed resource as-is.
    assert load_uk_local_authority_names_resource()["area_count"] == 361


def test_uk_local_authority_names_covers_the_crosswalk_roster_exactly() -> None:
    area_ids = _crosswalk_area_ids()

    assert len(area_ids) == 361
    assert set(_resource()["areas"]) == set(area_ids)
    assert set(local_authority_engine_key_by_code()) == set(area_ids)


def test_engine_keys_follow_the_mechanical_rule_except_the_declared_alias() -> None:
    areas = _resource()["areas"]

    for code, entry in areas.items():
        assert entry["engine_key"] == local_authority_engine_key(entry["name"]), code

    def bare_rule(name: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")

    bare_rule_differs = {
        code
        for code, entry in areas.items()
        if entry["engine_key"] != bare_rule(entry["name"])
    }
    assert bare_rule_differs == {"E07000146"}
    assert bare_rule("King's Lynn and West Norfolk") == "KING_S_LYNN_AND_WEST_NORFOLK"
    assert areas["E07000146"] == {
        "name": "King's Lynn and West Norfolk",
        "engine_key": "KINGS_LYNN_AND_WEST_NORFOLK",
    }
    assert dict(LOCAL_AUTHORITY_ENGINE_KEY_ALIASES) == {
        "King's Lynn and West Norfolk": "KINGS_LYNN_AND_WEST_NORFOLK"
    }
    keys = [entry["engine_key"] for entry in areas.values()]
    assert len(set(keys)) == len(keys) == 361


def test_engine_key_rule_examples() -> None:
    assert local_authority_engine_key("Bristol, City of") == "BRISTOL_CITY_OF"
    assert local_authority_engine_key("Na h-Eileanan Siar") == "NA_H_EILEANAN_SIAR"
    assert local_authority_engine_key(" Rhondda Cynon Taf ") == "RHONDDA_CYNON_TAF"
    assert local_authority_engine_key("Herefordshire, County of") == (
        "HEREFORDSHIRE_COUNTY_OF"
    )
    assert local_authority_engine_key("King's Lynn and West Norfolk") == (
        "KINGS_LYNN_AND_WEST_NORFOLK"
    )
    with pytest.raises(ValueError, match="empty engine key"):
        local_authority_engine_key(" , ")


def test_resolve_maps_known_codes_and_refuses_unknown() -> None:
    resolved = resolve_local_authority_engine_keys(
        pd.Series(
            ["E09000001", "E07000146", "S12000013", "N09000011"], index=[7, 3, 9, 1]
        )
    )

    assert resolved.dtype == object
    assert resolved.tolist() == [
        "CITY_OF_LONDON",
        "KINGS_LYNN_AND_WEST_NORFOLK",
        "NA_H_EILEANAN_SIAR",
        "ARDS_AND_NORTH_DOWN",
    ]
    assert resolve_local_authority_engine_keys(np.asarray(["W06000001"])).tolist() == [
        "ISLE_OF_ANGLESEY"
    ]
    assert resolve_local_authority_engine_keys(["E06000063"]).tolist() == ["CUMBERLAND"]

    with pytest.raises(ValueError, match=r"not on the April 2023.*E09999999"):
        resolve_local_authority_engine_keys(["E09000001", "E09999999", ""])
    with pytest.raises(ValueError, match="2 distinct"):
        resolve_local_authority_engine_keys(["E09999999", None, "E09999999"])


def test_consistency_failures_report_blank_unknown_and_mismatched_rows() -> None:
    household = pd.DataFrame(
        {
            "local_authority_code": [
                "E09000001",
                "E09000001",
                "E08000020",
                "W06000001",
                "S12000033",
            ],
            "local_authority": [
                "CITY_OF_LONDON",
                "MAIDSTONE",
                "GATESHEAD",
                "",
                "ABERDEEN_CITY",
            ],
        }
    )

    failures = local_authority_consistency_failures(household)

    assert failures == [
        "local_authority: 1/5 row(s) are blank",
        "local_authority_code: 1/5 row(s) are not on the April 2023 local "
        "authority roster; examples ['E08000020']",
        "local_authority: 1/5 row(s) disagree with local_authority_code; "
        "examples ['E09000001->MAIDSTONE']",
    ]
    assert local_authority_consistency_failures(household.iloc[[0, 4]]) == []
    assert local_authority_consistency_failures(
        household.drop(columns=["local_authority"])
    ) == ["household table is missing local authority column(s): ['local_authority']"]


def test_local_authority_is_a_ladder_column_after_its_code() -> None:
    columns = list(UK_GEOGRAPHY_LADDER_COLUMNS)

    assert columns.index("local_authority") == columns.index("local_authority_code") + 1


def test_resource_loader_refuses_a_drifted_source_digest(monkeypatch) -> None:
    load_uk_local_authority_names_resource.cache_clear()
    monkeypatch.setattr(geography_sources, "LAD23_NAMES_SHA256", "0" * 64)
    try:
        with pytest.raises(ValueError, match="not the pinned LAD23 names digest"):
            load_uk_local_authority_names_resource()
    finally:
        load_uk_local_authority_names_resource.cache_clear()
    monkeypatch.undo()
    assert load_uk_local_authority_names_resource()["area_count"] == 361


def test_uk_local_authority_names_matches_generator_output() -> None:
    if not DEFAULT_SOURCE_CSV.exists():
        pytest.skip("ONS LAD23 names lookup is not mounted under build/uk")

    assert _resource() == build_local_authority_names(source_csv=DEFAULT_SOURCE_CSV)


def test_generator_refuses_wrong_row_count_and_duplicate_keys(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "lad23.csv"
    source.write_text(
        "﻿LAD23CD,LAD23NM,LAD23NMW,ObjectId\n"
        "E06000001,Hartlepool,,1\n"
        "E06000002,Middlesbrough,,2\n",
        encoding="utf-8",
    )
    # A lookup whose bytes differ from the pinned digest is refused before
    # any row is parsed, naming both digests.
    with pytest.raises(ValueError, match=f"expected {SOURCE_SHA256}, got "):
        build_local_authority_names(source_csv=source)

    monkeypatch.setattr(
        geography_sources,
        "LAD23_NAMES_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(geography_sources, "LAD23_COUNT", 3)
    with pytest.raises(ValueError, match="LAD23"):
        build_local_authority_names(source_csv=source)

    monkeypatch.setattr(geography_sources, "LAD23_COUNT", 2)
    built = build_local_authority_names(source_csv=source)
    assert built["area_count"] == 2
    assert built["areas"]["E06000002"] == {
        "name": "Middlesbrough",
        "engine_key": "MIDDLESBROUGH",
    }
    assert built["source"]["url"] == LAD23_NAMES_URL

    source.write_text(
        "﻿LAD23CD,LAD23NM,LAD23NMW,ObjectId\n"
        "E06000001,Somerset,,1\n"
        "E06000066,Somerset!,,2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        geography_sources,
        "LAD23_NAMES_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="one engine key"):
        build_local_authority_names(source_csv=source)
