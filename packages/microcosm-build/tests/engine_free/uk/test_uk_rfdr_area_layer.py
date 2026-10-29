"""The Rural Fuel Duty Relief area layer and its household flags (microcosm#1418)."""

from __future__ import annotations

import importlib.util
import json

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


def test_great_britain_households_none_of_which_is_flagged_are_refused() -> None:
    layer = load_rfdr_area_layer(_layer())
    with pytest.raises(ValueError, match="none of the 2 Great Britain households"):
        area_flags(
            layer=layer,
            area_system=["uk_ew_output_area_2021", "uk_scotland_output_area_2022"],
            area_code=["E00000003", "S00135401"],
        )
    # Northern Ireland alone has no area in the scheme, so nothing is flagged.
    flags, counts = area_flags(
        layer=layer, area_system=["uk_ni_data_zone_2021"], area_code=["95AA01S1"]
    )
    assert not flags.any() and counts["flagged"] == 0


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


def _load_tool():
    path = _TEST_PATHS.repository / "tools/build_uk_rfdr_area_layer.py"
    spec = importlib.util.spec_from_file_location("build_uk_rfdr_area_layer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_tool_refuses_a_lookup_without_the_output_area_column(tmp_path) -> None:
    # Anthony's review of microcosm#1420: a lookup that has the postcode but
    # not the 2021 output area refuses by name, not by a later KeyError.
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("files:\n  nspl.zip:\n    sha256: " + "0" * 64 + "\n")
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
