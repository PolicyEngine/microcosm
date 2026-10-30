"""The Rural Fuel Duty Relief area layer and its household flags (microcosm#1418)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.rfdr_area_layer import (
    KIND,
    METHOD_VERSION,
    SCHEMA_VERSION,
    RFDRPostcodeDefinition,
    area_flags,
    flag_output_areas,
    load_rfdr_area_layer,
    postcode_in_scheme,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_DEFINITION = RFDRPostcodeDefinition.from_mapping(
    {"districts": ["EX35", "HS1"], "sectors": ["PA34 4"]}
)


def _layer(**areas) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "method": METHOD_VERSION,
        "areas": {
            "uk_ew_output_area_2021": ["E00000002"],
            "uk_scotland_output_area_2022": ["S00135400"],
            **areas,
        },
    }


def test_a_postcode_is_in_the_scheme_by_its_district_or_sector() -> None:
    inside = postcode_in_scheme(
        ["EX356AB", "EX36 1AA", "PA344AB", "PA34 5AB", "HS12AA", "HS21AA"],
        _DEFINITION,
    )
    assert inside.tolist() == [True, False, True, False, True, False]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"districts": ["EX35"], "sectors": ["EX35 6"]}, "already hold"),
        ({"districts": ["EX35"], "units": ["EX35 6AB"]}, "already hold"),
        ({"districts": ["EX35"], "excluded_units": ["EX36 1AA"]}, "no included"),
        ({"districts": ["EX35"], "units": ["EX356AB"]}, "malformed"),
        ({"districts": ["ex35"]}, "malformed"),
        ({"districts": ["EX35"], "sectors": ["PA34"]}, "malformed"),
        ({"sectors": ["PA34 4"]}, "no postcode district"),
    ],
)
def test_a_malformed_definition_is_refused(payload, match) -> None:
    with pytest.raises(ValueError, match=match):
        RFDRPostcodeDefinition.from_mapping(payload)


def test_units_join_and_leave_the_scheme_one_by_one() -> None:
    """An island sharing a sector with the mainland joins by its own units, and
    a mainland unit inside an island district is taken out (microcosm#1418)."""

    definition = RFDRPostcodeDefinition.from_mapping(
        {
            "districts": ["TR21", "HS1"],
            "units": {"raasay": ["IV40 8NT"], "seil": ["PA34 4RU"]},
            "excluded_units": ["TR21 0PW"],
        }
    )
    inside = postcode_in_scheme(
        ["IV408NT", "IV40 8DB", "PA34 4RU", "PA344AB", "TR210PW", "TR210JA"],
        definition,
    )
    assert inside.tolist() == [True, False, True, False, False, True]


def test_an_output_area_is_flagged_when_most_of_its_postcodes_are_in() -> None:
    postcodes = pd.DataFrame(
        {
            "postcode": ["EX356AA", "EX356AB", "EX361AA", "EX361AB", "PA344AA"],
            "oa21cd": [
                "E00000001",
                "E00000001",
                "E00000001",
                "E00000002",
                "S00135400",
            ],
        }
    )
    flagged, receipt = flag_output_areas(postcodes, _DEFINITION)
    # Two of E00000001's three postcodes are in; none of E00000002's.
    assert flagged == {
        "uk_ew_output_area_2021": ["E00000001"],
        "uk_scotland_output_area_2022": ["S00135400"],
    }
    assert receipt["postcodes_inside"] == 3
    assert receipt["by_system"]["uk_ew_output_area_2021"] == {
        "areas_flagged": 1,
        "areas_partly_inside": 1,
    }


def test_a_tied_output_area_is_not_flagged() -> None:
    postcodes = pd.DataFrame(
        {"postcode": ["EX356AA", "EX361AA"], "oa21cd": ["E00000001", "E00000001"]}
    )
    flagged, _ = flag_output_areas(postcodes, _DEFINITION)
    assert flagged["uk_ew_output_area_2021"] == []


def test_households_take_their_area_flag_and_northern_ireland_none() -> None:
    layer = load_rfdr_area_layer(json.dumps(_layer()))
    flags, counts = area_flags(
        layer=layer,
        area_system=[
            "uk_ew_output_area_2021",
            "uk_ew_output_area_2021",
            "uk_scotland_output_area_2022",
            "uk_ni_data_zone_2021",
        ],
        area_code=["E00000001", "E00000002", "S00135400", "95AA01S1"],
    )
    assert flags.tolist() == [False, True, True, False]
    assert counts == {
        "uk_ew_output_area_2021": 2,
        "uk_scotland_output_area_2022": 1,
        "uk_ni_data_zone_2021": 1,
        "flagged": 2,
    }
    with pytest.raises(ValueError, match="no support system"):
        area_flags(layer=layer, area_system=["legacy"], area_code=["X"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update(kind="other"), "Not a UK Rural Fuel Duty Relief"),
        (lambda p: p.update(method="other"), "method"),
        (lambda p: p["areas"].pop("uk_scotland_output_area_2022"), "exactly"),
        (
            lambda p: p["areas"].update(uk_ew_output_area_2021=["S00135400"]),
            "malformed",
        ),
        (
            lambda p: p["areas"].update(
                uk_ew_output_area_2021=["E00000002", "E00000002"]
            ),
            "repeats",
        ),
        (
            lambda p: p.update(
                areas={
                    "uk_ew_output_area_2021": [],
                    "uk_scotland_output_area_2022": [],
                }
            ),
            "flags no output area",
        ),
    ],
)
def test_a_malformed_layer_is_refused(mutate, match) -> None:
    payload = _layer()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        load_rfdr_area_layer(payload)


def test_flags_are_booleans_aligned_with_the_households() -> None:
    layer = load_rfdr_area_layer(_layer())
    flags, _ = area_flags(
        layer=layer,
        area_system=np.asarray(["uk_ew_output_area_2021"] * 3, dtype=object),
        area_code=np.asarray(["E00000002", "E00000003", "E00000002"], dtype=object),
    )
    assert flags.dtype == bool and flags.tolist() == [True, False, True]


@pytest.mark.parametrize("bad", [None, float("nan"), "", "e00000002", "E0000002"])
def test_a_malformed_great_britain_code_is_refused(bad) -> None:
    # A code that is missing, cased or shaped differently would silently miss
    # the layer (Anthony's review of microcosm#1420).
    layer = load_rfdr_area_layer(_layer())
    with pytest.raises(ValueError, match="well-formed"):
        area_flags(
            layer=layer,
            area_system=["uk_ew_output_area_2021"] * 2,
            area_code=["E00000002", bad],
        )


def test_a_system_whose_households_none_is_flagged_is_refused() -> None:
    layer = load_rfdr_area_layer(_layer())
    with pytest.raises(
        ValueError, match="none of the 1 uk_ew_output_area_2021 households"
    ):
        area_flags(
            layer=layer,
            area_system=["uk_ew_output_area_2021", "uk_scotland_output_area_2022"],
            area_code=["E00000003", "S00135401"],
        )
    # Anthony's re-review of microcosm#1420: each system on its own, so a
    # Scottish support on other codes refuses beside one flagged English
    # household instead of losing almost all of the relief.
    with pytest.raises(
        ValueError, match="none of the 2 uk_scotland_output_area_2022 households"
    ):
        area_flags(
            layer=layer,
            area_system=["uk_ew_output_area_2021"]
            + ["uk_scotland_output_area_2022"] * 2,
            area_code=["E00000002", "S00135401", "S00135402"],
        )
    # Northern Ireland alone has no area in the scheme, so nothing is flagged.
    flags, counts = area_flags(
        layer=layer, area_system=["uk_ni_data_zone_2021"], area_code=["95AA01S1"]
    )
    assert not flags.any() and counts["flagged"] == 0


def test_a_sampled_rung_may_flag_none_but_still_refuses_malformed_codes() -> None:
    layer = load_rfdr_area_layer(_layer())
    flags, counts = area_flags(
        layer=layer,
        area_system=["uk_ew_output_area_2021", "uk_scotland_output_area_2022"],
        area_code=["E00000003", "S00135401"],
        require_flagged=False,
    )
    assert not flags.any() and counts["flagged"] == 0
    with pytest.raises(ValueError, match="well-formed"):
        area_flags(
            layer=layer,
            area_system=["uk_ew_output_area_2021"],
            area_code=["e00000003"],
            require_flagged=False,
        )


def test_the_committed_layer_was_built_from_the_registered_nspl_zip() -> None:
    """Anthony's review of microcosm#1420, 7b: the committed layer comes from
    the NSPL zip Chronicle registers, not from a reduction of it, and its
    provenance carries the layer's own digest. The digest is held to the
    tool's pin and to the high value home stock's own record of the same
    registration, which the tool does not write (the independent review of
    microcosm#1420)."""

    import hashlib
    from importlib.resources import files

    uk = files("microcosm.build.uk")
    provenance = json.loads(
        uk.joinpath("rfdr_area_layer.provenance.json").read_text(encoding="utf-8")
    )
    stock = json.loads(
        uk.joinpath("high_value_home_stock.provenance.json").read_text(encoding="utf-8")
    )
    nspl = stock["inputs"]["registrations"]["nspl"]
    ((stock_file,),) = [list(nspl["files"].values())]
    pin = _load_tool().NSPL_REGISTRATION
    assert (nspl["chronicle_package_id"], stock_file["filename"]) == (
        pin["package_id"],
        pin["filename"],
    )
    # Both tools read the same member of the zip (the re-review of
    # microcosm#1420, I).
    stock_tool = _load_tool("tools/build_uk_high_value_home_stock.py")
    assert pin["member"] == stock_tool.NSPL_DATA_MEMBER
    registered = provenance["nspl_registration"]["zip_sha256"]
    assert registered == stock_file["sha256"] == pin["sha256"]
    assert provenance["nspl_registration"]["manifest"] == (
        "PolicyEngine/chronicle db/data/ons/nspl_august_2026/manifest.yaml"
    )
    assert provenance["postcode_source"] == {"kind": "nspl_zip", "sha256": registered}
    layer = uk.joinpath("rfdr_area_layer.json").read_bytes()
    assert provenance["layer_sha256"] == hashlib.sha256(layer).hexdigest()


def test_the_committed_layer_flags_the_scheme_s_output_areas() -> None:
    """The committed layer loads, and every part of the scheme is on it."""

    from importlib.resources import files

    payload = files("microcosm.build.uk").joinpath("rfdr_area_layer.json").read_bytes()
    layer = load_rfdr_area_layer(payload)
    england_wales = layer.areas["uk_ew_output_area_2021"]
    scotland = layer.areas["uk_scotland_output_area_2022"]
    # The five Hawes and High Abbotside output areas (microcosm#1418).
    assert {f"E0014133{d}" for d in "6789"} | {"E00141340"} <= england_wales
    # Scotland's islands and mainland districts hold most of the scheme.
    assert len(scotland) > 1000 > len(england_wales) > 30
    definition = json.loads(payload)["definition"]
    assert {"EX35", "HS1", "ZE1", "TR21", "KA27", "PA20", "PH41"} <= set(
        definition["districts"]
    )
    assert "IV45 8YD" in definition["excluded_units"]


def _load_tool(relative: str = "tools/build_uk_rfdr_area_layer.py"):
    path = _TEST_PATHS.repository / relative
    spec = importlib.util.spec_from_file_location(Path(relative).stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path, **overrides) -> Path:
    pin = _load_tool().NSPL_REGISTRATION
    fields = {
        "package_id": pin["package_id"],
        "filename": pin["filename"],
        "sha256": pin["sha256"],
        **overrides,
    }
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"package_id: {fields['package_id']}\nfiles:\n  2026:\n"
        f"    filename: {fields['filename']}\n    sha256: {fields['sha256']}\n"
        + (
            "  2025:\n    filename: other.zip\n    sha256: " + "1" * 64 + "\n"
            if fields.get("second")
            else ""
        )
    )
    return manifest


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"package_id": "ons-nspl-may-2026"}, "registers package"),
        ({"filename": "NSPL_MAY_2026.zip"}, "not the pinned"),
        ({"sha256": "0" * 64}, "not the pinned"),
        ({"second": True}, "registers 2 files"),
    ],
)
def test_the_tool_reads_only_the_pinned_nspl_registration(
    tmp_path, overrides, match
) -> None:
    """The independent review of microcosm#1420: any one-file manifest used to
    pass, so another lookup's registration could stand in for the pinned one."""

    tool = _load_tool()
    assert (
        tool._registered_sha256(_manifest(tmp_path))
        == (tool.NSPL_REGISTRATION["sha256"])
    )
    with pytest.raises(SystemExit, match=match):
        tool._registered_sha256(_manifest(tmp_path, **overrides))


def test_the_provenance_names_the_manifest_it_read(tmp_path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "chronicle/db/data/ons/nspl_august_2026/manifest.yaml"
    assert tool._manifest_label(manifest) == (
        "PolicyEngine/chronicle db/data/ons/nspl_august_2026/manifest.yaml"
    )
    elsewhere = tmp_path / "manifest.yaml"
    assert tool._manifest_label(elsewhere) == str(elsewhere)


def test_the_tool_refuses_a_lookup_without_the_output_area_column(tmp_path) -> None:
    # Anthony's review of microcosm#1420: a lookup that has the postcode but
    # not the 2021 output area refuses by name, not by a later KeyError.
    manifest = _manifest(tmp_path)
    lookup = tmp_path / "lookup.parquet"
    pd.DataFrame({"postcode": ["HS1 2AA"], "lsoa21cd": ["S01000001"]}).to_parquet(
        lookup
    )
    with pytest.raises(SystemExit, match="lacks postcode or oa21cd"):
        _load_tool().main(
            [
                "--postcode-oa-parquet",
                str(lookup),
                "--nspl-manifest",
                str(manifest),
                "--out",
                str(tmp_path / "layer.json"),
                "--provenance-json",
                str(tmp_path / "provenance.json"),
            ]
        )
    assert not (tmp_path / "layer.json").exists()


def test_check_mode_reports_a_missing_layer_as_a_difference(tmp_path) -> None:
    # The re-review of microcosm#1420, 15: --check against a file that is not
    # there reports a difference, not a FileNotFoundError.
    lookup = tmp_path / "lookup.parquet"
    pd.DataFrame({"postcode": ["HS1 2AA"], "oa21cd": ["S00000001"]}).to_parquet(lookup)
    code = _load_tool().main(
        [
            "--check",
            "--postcode-oa-parquet",
            str(lookup),
            "--nspl-manifest",
            str(_manifest(tmp_path)),
            "--out",
            str(tmp_path / "missing-layer.json"),
            "--provenance-json",
            str(tmp_path / "missing-provenance.json"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "missing-layer.json").exists()
