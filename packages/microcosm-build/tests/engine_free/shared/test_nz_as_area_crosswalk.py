"""Offline spatial bridge and population-share contracts for MC9.

Small fixtures exercise geometry and population rules without source downloads.
The staged-byte differential is optional because the pinned ArcGIS pages are
distributed separately from the repository; set NZ_AS_AREA_SOURCES to replay it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from test_support.paths import paths_for
from tools import build_nz_as_area_crosswalk as crosswalk_builder
from tools.build_nz_as_area_crosswalk import (
    aggregate_rows,
    assign_points,
    concordance_ta_codes,
    exact_area_mapping,
    generate_crosswalk,
    interior_point,
    point_in_geometry,
    population_counts,
)

_BUILD = paths_for("microcosm-build")
_SPEC_PATH = _BUILD.package / "src/microcosm/build/nz/as_area_crosswalk.json"


def _spec() -> dict:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def _rectangle(x: float, y: float, width: float = 1.0) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [x, y],
                [x + width, y],
                [x + width, y + width],
                [x, y + width],
                [x, y],
            ]
        ],
    }


def _au(code: str, name: str, x: float = 0.0) -> dict:
    return {
        "type": "Feature",
        "properties": {"AU2017_V1_00": code, "AU2017_V1_00_NAME": name},
        "geometry": _rectangle(x, 0.0),
    }


def _ta(code: str, name: str, x: float = 0.0) -> dict:
    return {
        "type": "Feature",
        "properties": {"TA2025_V1_00": code, "TA2025_V1_00_NAME": name},
        "geometry": _rectangle(x, 0.0),
    }


def _sa1(code: str, count: object, older: object = 1) -> dict:
    return {
        "attributes": {
            "SA12023_V1_00": code,
            "VAR_1_2": older,
            "VAR_1_3": count,
        },
        "centroid": {"x": 0.5, "y": 0.5},
    }


def _meshblock(sa1_code: str, ta_code: str, ta_name: str, code: str = "mb") -> dict:
    return {
        "attributes": {
            "MB2026_V1_00": code,
            "SA12026_V1_00": sa1_code,
            "TA2026_V1_00": ta_code,
            "TA2026_V1_00_NAME": ta_name,
        }
    }


def _polygon_feature(sa1_code: str, geometry: dict) -> dict:
    return {
        "type": "Feature",
        "properties": {"SA12023_V1_00": sa1_code},
        "geometry": geometry,
    }


def _synthetic_spec(names: list[str]) -> dict:
    spec = _spec()
    spec["area_definitions"] = _definitions(names)
    spec["definition_name_aliases"] = []
    spec["resolved_definition_entries"] = []
    spec["heading_ta_expectations"] = {
        "Fixture location": ["Queenstown-Lakes District"]
    }
    return spec


def _definitions(names: list[str], area: int = 1) -> dict:
    return {
        "areas": {
            str(number): (
                {
                    "locations": [
                        {
                            "heading": "Fixture location",
                            "statistical_area_units_2017": names,
                        }
                    ]
                }
                if number == area
                else {"locations": []}
            )
            for number in (1, 2, 3)
        }
        | {"4": {"residual": "Anything unlisted is Area 4"}}
    }


def _shares(row: dict) -> dict[int, float]:
    return {item["as_area"]: item["population_share"] for item in row["shares"]}


@pytest.fixture(scope="module")
def staged_sources() -> Path:
    configured = os.environ.get("NZ_AS_AREA_SOURCES")
    directory = Path(configured) if configured else _BUILD.repository / ".hub-sources"
    if not directory.is_dir():
        pytest.skip(
            "Pinned ArcGIS bytes are distributed separately; set "
            "NZ_AS_AREA_SOURCES to their directory to replay the crosswalk"
        )
    provenance = _spec()["provenance"]
    names = [pin["file"] for pin in provenance["pages"]]
    names.extend(layer["metadata_file"] for layer in provenance["layers"])
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        pytest.skip(
            "Pinned ArcGIS source set is incomplete; set NZ_AS_AREA_SOURCES "
            f"to the directory containing all pinned bytes (missing {missing[0]})"
        )
    return directory


def test_point_in_polygon_hole_and_its_boundary() -> None:
    geometry = _rectangle(0.0, 0.0, 4.0)
    geometry["coordinates"].append(
        [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]]
    )
    assert point_in_geometry((0.5, 0.5), geometry)
    assert not point_in_geometry((2.0, 2.0), geometry)
    assert point_in_geometry((1.0, 2.0), geometry)


def test_point_in_multipolygon() -> None:
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            _rectangle(0.0, 0.0)["coordinates"],
            _rectangle(3.0, 0.0)["coordinates"],
        ],
    }
    assert point_in_geometry((0.5, 0.5), geometry)
    assert point_in_geometry((3.5, 0.5), geometry)
    assert not point_in_geometry((2.0, 0.5), geometry)


def test_nested_multipart_ring_is_a_hole_under_global_even_odd_rule() -> None:
    # The staged ArcGIS GeoJSON encodes some holes as separate multipart rings.
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            _rectangle(0.0, 0.0, 4.0)["coordinates"],
            _rectangle(1.0, 1.0, 2.0)["coordinates"],
        ],
    }
    assert point_in_geometry((0.5, 0.5), geometry)
    assert not point_in_geometry((2.0, 2.0), geometry)
    assert point_in_geometry((1.0, 2.0), geometry)


@pytest.mark.parametrize("point", [(0.0, 0.5), (1.0, 1.0), (0.5, 0.0)])
def test_point_on_edge_is_included(point: tuple[float, float]) -> None:
    assert point_in_geometry(point, _rectangle(0.0, 0.0))


def test_concave_polygon_uses_even_odd_rule() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [0, 0]]],
    }
    assert point_in_geometry((0.5, 2.5), geometry)
    assert point_in_geometry((2.5, 0.5), geometry)
    assert not point_in_geometry((2.5, 2.5), geometry)


@pytest.mark.parametrize(
    "shape", ["concave", "holed", "multipart", "nested_hole", "narrow"]
)
def test_interior_point_is_inside_and_deterministic(shape: str) -> None:
    if shape == "concave":
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [0, 0]]],
        }
    elif shape == "narrow":
        geometry = _rectangle(0.0, 0.0, 0.000001)
    elif shape == "holed":
        geometry = _rectangle(0.0, 0.0, 4.0)
        geometry["coordinates"].append(_rectangle(1.0, 1.0, 2.0)["coordinates"][0])
    else:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [
                _rectangle(0.0, 0.0, 4.0)["coordinates"],
                _rectangle(1.0, 1.0, 2.0)["coordinates"]
                if shape == "nested_hole"
                else _rectangle(6.0, 0.0, 2.0)["coordinates"],
            ],
        }
    original = copy.deepcopy(geometry)
    point = interior_point(geometry, tolerance=0.001)
    assert point_in_geometry(point, geometry)
    # A boundary point is included by point_in_geometry, but does not meet the
    # interior-point guarantee. Require positive clearance from every ring.
    assert crosswalk_builder._boundary_distance(np.asarray(point), geometry) > 0
    assert point == interior_point(copy.deepcopy(geometry), tolerance=0.001)
    assert geometry == original


@pytest.mark.parametrize("tolerance", [0, -1, math.inf, math.nan])
def test_interior_point_invalid_tolerance_refuses(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        interior_point(_rectangle(0.0, 0.0), tolerance=tolerance)


def test_interior_point_zero_area_diagonal_refuses() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 1], [2, 2], [0, 0]]],
    }
    with pytest.raises(ValueError, match="interior"):
        interior_point(geometry)


def test_concordance_repeats_meshblocks_but_returns_one_ta_per_sa1() -> None:
    features = [_sa1("a", 9), _sa1("b", 3)]
    authorities = [_ta("001", "First"), _ta("002", "Second", 2.0)]
    meshblocks = [
        _meshblock("b", "002", "Second", "1"),
        _meshblock("a", "001", "First", "2"),
        _meshblock("a", "001", "First", "3"),
    ]
    assert concordance_ta_codes(features, authorities, meshblocks) == ["001", "002"]
    assert concordance_ta_codes(features[::-1], authorities, meshblocks[::-1]) == [
        "002",
        "001",
    ]


def test_concordance_missing_sa1_refuses_with_code() -> None:
    with pytest.raises(ValueError, match="missing|Missing|coverage") as error:
        concordance_ta_codes(
            [_sa1("a", 9), _sa1("missing-sa1", 3)],
            [_ta("001", "Fixture")],
            [_meshblock("a", "001", "Fixture")],
        )
    assert "missing-sa1" in str(error.value)


def test_concordance_sa1_with_two_tas_refuses_with_code() -> None:
    with pytest.raises(ValueError, match="multiple|two|unique|exactly one") as error:
        concordance_ta_codes(
            [_sa1("a", 9)],
            [_ta("001", "First"), _ta("002", "Second", 2.0)],
            [
                _meshblock("a", "001", "First", "1"),
                _meshblock("a", "002", "Second", "2"),
            ],
        )
    assert "a" in str(error.value)


@pytest.mark.parametrize("mismatch", ["code", "name", "missing_ta"])
def test_concordance_ta_code_and_name_set_mismatch_refuses(mismatch: str) -> None:
    authorities = [_ta("001", "Fixture")]
    meshblocks = [_meshblock("a", "001", "Fixture")]
    if mismatch == "code":
        meshblocks[0]["attributes"]["TA2026_V1_00"] = "002"
    elif mismatch == "name":
        meshblocks[0]["attributes"]["TA2026_V1_00_NAME"] = "Wrong name"
    else:
        authorities.append(_ta("002", "Unrepresented", 2.0))
    with pytest.raises(ValueError, match="TA|territorial|authority"):
        concordance_ta_codes([_sa1("a", 9)], authorities, meshblocks)


def test_large_ring_latitude_bands_preserve_rectangle_membership() -> None:
    corners = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    # 640 edges and more than 32 points exercise the accelerated latitude bands.
    ring = [
        (first + fraction * (second - first)).tolist()
        for first, second in zip(corners[:-1], corners[1:], strict=True)
        for fraction in np.linspace(0, 1, 160, endpoint=False)
    ]
    ring.append(ring[0])
    points = np.asarray(
        [
            (x, y)
            for x in (-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1)
            for y in np.linspace(0, 1, 70)
        ]
        + [(0.5, -0.1), (0.5, 1.1)]
    )
    expected = np.asarray(
        [
            -1
            if x < 0 or x > 1 or y < 0 or y > 1
            else 0
            if x in (0, 1) or y in (0, 1)
            else 1
            for x, y in points
        ]
    )
    actual = crosswalk_builder._ring_status(points, ring)
    np.testing.assert_array_equal(actual, expected)
    # The small equivalent rectangle follows the ordinary edge scan.
    np.testing.assert_array_equal(
        actual,
        crosswalk_builder._ring_status(points, _rectangle(0, 0)["coordinates"][0]),
    )


def test_point_outside_everything_refuses_when_tolerance_is_zero() -> None:
    point = (5.0, 5.0)
    polygon = _au("001", "Fixture")
    assert not point_in_geometry(point, polygon["geometry"])
    with pytest.raises(ValueError):
        assign_points(np.asarray([point]), [polygon], "AU2017_V1_00", max_distance_m=0)


def test_no_hit_nearest_assignment_is_explicit_and_bounded() -> None:
    # About one metre east of the polygon; well within the stated 20 m rule.
    polygons = [_au("001", "Fixture")]
    point = np.asarray([[1.00001, 0.5]])
    codes, diagnostics = assign_points(
        point, polygons, "AU2017_V1_00", max_distance_m=20
    )
    assert codes == ["001"]
    assert len(diagnostics["no_polygon"]) == 1
    assert diagnostics["no_polygon"][0]["assigned_code"] == "001"
    assert 1 < diagnostics["no_polygon"][0]["distance_m"] < 2
    with pytest.raises(ValueError):
        assign_points(point, polygons, "AU2017_V1_00", max_distance_m=0.1)


def test_overlapping_polygon_interiors_refuse_first_hit() -> None:
    polygons = [_au("001", "First"), _au("002", "Second", 0.25)]
    for ordered in (polygons, polygons[::-1]):
        with pytest.raises(ValueError, match="multiple polygons"):
            assign_points(np.asarray([[0.5, 0.5]]), ordered, "AU2017_V1_00")


def test_bounded_overlap_uses_unique_deepest_candidate_and_reports_all_distances() -> (
    None
):
    polygons = [_au("001", "Shallow"), _au("002", "Deeper", 0.9999)]
    point = np.asarray([[0.99999, 0.5]])
    for ordered in (polygons, polygons[::-1]):
        with pytest.raises(ValueError, match="multiple polygons"):
            assign_points(point, ordered, "AU2017_V1_00")
        codes, diagnostics = assign_points(
            point, ordered, "AU2017_V1_00", max_overlap_m=20
        )
        assert codes == ["002"]
        assert diagnostics["no_polygon"] == []
        assert diagnostics["multiple_boundary_hits"] == []
        overlaps = diagnostics["multiple_interior_hits"]
        assert len(overlaps) == 1
        assert overlaps[0]["index"] == 0
        assert overlaps[0]["assigned_code"] == "002"
        distances = overlaps[0]["candidate_boundary_distances_m"]
        assert set(distances) == {"001", "002"}
        assert 1 < distances["001"] < 2
        assert 9 < distances["002"] < 11
        assert max(distances.values()) < 20


def test_bounded_overlap_refuses_equal_interior_distances() -> None:
    polygons = [_au("001", "Left"), _au("002", "Right", 0.9999)]
    with pytest.raises(ValueError, match="multiple polygons"):
        assign_points(
            np.asarray([[0.99995, 0.5]]),
            polygons,
            "AU2017_V1_00",
            max_overlap_m=20,
        )


def test_bounded_overlap_refuses_when_only_one_candidate_is_near_its_boundary() -> None:
    polygons = [_au("001", "Deep"), _au("002", "Shallow", 0.5)]
    with pytest.raises(ValueError, match="multiple polygons"):
        assign_points(
            np.asarray([[0.500001, 0.5]]),
            polygons,
            "AU2017_V1_00",
            max_overlap_m=20,
        )


def test_shared_boundary_reports_all_hits_and_uses_smallest_code() -> None:
    polygons = [_au("002", "Right", 1.0), _au("001", "Left")]
    for ordered in (polygons, polygons[::-1]):
        codes, diagnostics = assign_points(
            np.asarray([[1.0, 0.5]]), ordered, "AU2017_V1_00"
        )
        assert codes == ["001"]
        assert diagnostics["multiple_boundary_hits"] == [
            {"index": 0, "candidate_codes": ["001", "002"], "assigned_code": "001"}
        ]


def test_equal_nearest_boundaries_refuse_arbitrary_assignment() -> None:
    polygons = [_au("001", "Left"), _au("002", "Right", 2.0)]
    with pytest.raises(ValueError, match="tied nearest polygons"):
        assign_points(
            np.asarray([[1.5, 0.5]]), polygons, "AU2017_V1_00", max_distance_m=100000
        )


def test_distant_zero_population_assignment_is_explicit() -> None:
    polygons = [_au("001", "Fixture")]
    point = np.asarray([[5.0, 5.0]])
    codes, diagnostics = assign_points(
        point,
        polygons,
        "AU2017_V1_00",
        max_distance_m=20,
        populations=np.asarray([0]),
    )
    assert codes == ["001"]
    assert len(diagnostics["no_polygon"]) == 1
    assert diagnostics["no_polygon"][0]["distance_m"] > 20
    with pytest.raises(ValueError, match="beyond"):
        assign_points(
            point,
            polygons,
            "AU2017_V1_00",
            max_distance_m=20,
            populations=np.asarray([1]),
        )


def test_exact_name_matching_keeps_near_duplicate_in_area_four() -> None:
    polygons = [_au("001", "Silverdale Central"), _au("002", "Silverdale")]
    mapping, unmatched = exact_area_mapping(
        _definitions(["Silverdale Central"]), polygons
    )
    assert mapping == {"001": 1, "002": 4}
    assert unmatched == []


def test_ambiguous_definition_is_reported_with_candidate_population() -> None:
    polygons = [_au("001", "Shared"), _au("002", "Shared")]
    mapping, unmatched = exact_area_mapping(
        _definitions(["Shared"]), polygons, {"001": 11, "002": 23}
    )
    assert mapping == {"001": 4, "002": 4}
    assert len(unmatched) == 1
    report = json.dumps(unmatched)
    for expected in ("Shared", "001", "002", "11", "23"):
        assert expected in report


def test_queenstown_urban_area_is_unmatched_without_declared_resolution() -> None:
    polygons = [_au("001", "Queenstown Bay"), _au("002", "Frankton")]
    mapping, unmatched = exact_area_mapping(
        _definitions(["Queenstown urban area"]), polygons
    )
    assert mapping == {"001": 4, "002": 4}
    assert len(unmatched) == 1
    assert "Queenstown urban area" in json.dumps(unmatched)


def test_alias_maps_definition_by_declared_code() -> None:
    # A second unit with the same layer name must stay residual: an alias is a
    # reviewed code correspondence, rather than a new name-matching rule.
    polygons = [_au("540700", "Whaka"), _au("other", "Whaka", 2.0)]
    alias = {
        "definition_name": "Whakarewarewa",
        "au_code": "540700",
        "au_name": "Whaka",
        "evidence": "AU2013, AU2016 and AU2017 name Whaka; unique Rotorua entry",
    }
    mapping, unmatched = exact_area_mapping(
        _definitions(["Whakarewarewa"], area=3),
        polygons,
        definition_name_aliases=[alias],
    )
    assert mapping == {"540700": 3, "other": 4}
    assert unmatched == []


@pytest.mark.parametrize("wrong", ["code", "name"])
def test_alias_wrong_code_or_name_refuses(wrong: str) -> None:
    alias = {
        "definition_name": "Whakarewarewa",
        "au_code": "540700" if wrong == "name" else "missing-code",
        "au_name": "Whaka" if wrong == "code" else "Wrong name",
        "evidence": "fixture",
    }
    with pytest.raises(ValueError, match="alias|Alias"):
        exact_area_mapping(
            _definitions(["Whakarewarewa"], area=3),
            [_au("540700", "Whaka")],
            definition_name_aliases=[alias],
        )


def test_queenstown_urban_area_resolves_to_no_additional_unit() -> None:
    polygons = [_au("001", "Queenstown Bay"), _au("002", "Frankton East", 2.0)]
    definitions = _definitions(["Queenstown Bay", "Queenstown urban area"])
    definitions["areas"]["1"]["locations"][0]["heading"] = "Queenstown"
    resolved = {
        "name": "Queenstown urban area",
        "heading": "Queenstown",
        "as_area": 1,
        "au_codes": [],
        "resolution": "no additional area unit",
        "evidence": {"layer": "Urban_area_2014", "ua_code": "117"},
    }
    mapping, unmatched = exact_area_mapping(
        definitions, polygons, resolved_definition_entries=[resolved]
    )
    assert mapping == {"001": 1, "002": 4}
    assert unmatched == []


def test_sentinel_and_null_populations_are_never_summed_as_counts() -> None:
    features = [
        _sa1("a", 9),
        _sa1("b", -999),
        _sa1("c", -1),
        _sa1("d", None),
        _sa1("e", 0),
    ]
    counts, diagnostics = population_counts(features)
    assert counts.tolist() == [9, 0, 0, 0, 0]
    assert int(counts.sum()) == 9
    assert diagnostics
    row = aggregate_rows(
        [_ta("001", "Fixture")],
        features,
        ["au"] * len(features),
        ["001"] * len(features),
        {"au": 4},
    )[0]
    assert row["population_total"] == 9
    assert _shares(row)[4] == 1


@pytest.mark.parametrize("value", [True, 1.5, "12"])
def test_non_integer_populations_refuse(value: object) -> None:
    with pytest.raises(ValueError):
        population_counts([_sa1("a", value)])


def test_central_shares_use_2023_instead_of_2018_population() -> None:
    features = [_sa1("a", 90, older=10), _sa1("b", 10, older=90)]
    arguments = (
        [_ta("001", "Fixture")],
        features,
        ["listed", "residual"],
        ["001", "001"],
        {"listed": 1, "residual": 4},
    )
    central = aggregate_rows(*arguments)[0]
    older = aggregate_rows(*arguments, variable="VAR_1_2")[0]
    assert _shares(central) == {1: 0.9, 2: 0.0, 3: 0.0, 4: 0.1}
    assert _shares(older) == {1: 0.1, 2: 0.0, 3: 0.0, 4: 0.9}
    assert central["modal_area"] == 1
    assert older["modal_area"] == 4
    assert "VAR_1_3" in central["population_source"]
    assert "2023" in central["population_source"]


def test_unlisted_area_units_contribute_to_area_four_default() -> None:
    mapping, _ = exact_area_mapping(
        _definitions(["Listed"]), [_au("001", "Listed"), _au("002", "Rural")]
    )
    row = aggregate_rows(
        [_ta("001", "Fixture")],
        [_sa1("a", 10), _sa1("b", 90)],
        ["001", "002"],
        ["001", "001"],
        mapping,
    )[0]
    assert row["population_total"] == 100
    assert _shares(row) == {1: 0.1, 2: 0.0, 3: 0.0, 4: 0.9}
    assert row["modal_area"] == 4


def test_tied_modal_share_refuses() -> None:
    with pytest.raises(ValueError, match="tie"):
        aggregate_rows(
            [_ta("001", "Fixture")],
            [_sa1("a", 10), _sa1("b", 10)],
            ["one", "four"],
            ["001", "001"],
            {"one": 1, "four": 4},
        )


def test_zero_population_placeholder_is_retained_with_declared_area_four() -> None:
    rows = aggregate_rows(
        [
            _ta("001", "Populated"),
            _ta("999", "Area Outside Territorial Authority", 2.0),
        ],
        [_sa1("a", 12)],
        ["one"],
        ["001"],
        {"one": 1},
    )
    assert [row["ta_code"] for row in rows] == ["001", "999"]
    assert _shares(rows[1]) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}
    assert rows[1]["population_total"] == 0
    assert rows[1]["modal_area"] == 4


def test_unexpected_zero_population_authority_refuses() -> None:
    with pytest.raises(ValueError, match="zero usable population"):
        aggregate_rows(
            [_ta("001", "Empty authority")],
            [_sa1("a", 0)],
            ["one"],
            ["001"],
            {"one": 1},
        )


@settings(max_examples=40, deadline=None)
@given(
    populations=st.lists(st.integers(1, 20), min_size=4, max_size=4),
    order=st.permutations((0, 1, 2, 3)),
    polygon_order=st.permutations((0, 1, 2, 3)),
)
def test_shares_are_invariant_to_sa1_and_polygon_order(
    populations: list[int], order: list[int], polygon_order: list[int]
) -> None:
    # Ensure each TA has a unique modal area independently of the generated values.
    populations[0] += 100
    populations[2] += 100
    polygons = [_au(str(index), f"AU {index}", index * 2.0) for index in range(4)]
    points = np.asarray([[index * 2.0 + 0.5, 0.5] for index in range(4)])
    baseline, _ = assign_points(points, polygons, "AU2017_V1_00")
    reordered, _ = assign_points(
        points, [polygons[index] for index in polygon_order], "AU2017_V1_00"
    )
    assert reordered == baseline
    permuted_points, _ = assign_points(
        points[list(order)],
        [polygons[index] for index in polygon_order],
        "AU2017_V1_00",
    )
    assert permuted_points == [baseline[index] for index in order]
    features = [_sa1(str(index), value) for index, value in enumerate(populations)]
    tas = [_ta("001", "First"), _ta("002", "Second", 4.0)]
    ta_codes = ["001", "001", "002", "002"]
    area_by_au = {"0": 1, "1": 4, "2": 2, "3": 4}
    expected = aggregate_rows(tas, features, baseline, ta_codes, area_by_au)
    actual = aggregate_rows(
        tas[::-1],
        [features[index] for index in order],
        [baseline[index] for index in order],
        [ta_codes[index] for index in order],
        area_by_au,
    )
    assert actual == expected


@settings(max_examples=40, deadline=None)
@given(
    first=st.integers(100, 1000), second=st.integers(1, 99), scale=st.integers(1, 50)
)
def test_scaling_all_populations_leaves_shares_unchanged(
    first: int, second: int, scale: int
) -> None:
    tas = [_ta("001", "Fixture")]
    arguments = (["one", "four"], ["001", "001"], {"one": 1, "four": 4})
    baseline = aggregate_rows(tas, [_sa1("a", first), _sa1("b", second)], *arguments)[0]
    scaled = aggregate_rows(
        tas, [_sa1("a", first * scale), _sa1("b", second * scale)], *arguments
    )[0]
    assert _shares(scaled) == _shares(baseline)
    assert scaled["modal_area"] == baseline["modal_area"]


@settings(max_examples=40, deadline=None)
@given(moved=st.integers(1, 20), remaining=st.integers(100, 200))
def test_moving_one_sa1_changes_only_its_territorial_authority(
    moved: int, remaining: int
) -> None:
    tas = [_ta("001", "Affected"), _ta("002", "Unaffected", 4.0)]
    features = [_sa1("a", moved), _sa1("b", remaining), _sa1("c", 30)]
    assignments = ["001", "001", "002"]
    mapping = {"one": 1, "four": 4, "two": 2}
    baseline = aggregate_rows(
        tas, features, ["one", "four", "two"], assignments, mapping
    )
    moved_rows = aggregate_rows(
        tas, features, ["four", "four", "two"], assignments, mapping
    )
    assert _shares(moved_rows[0]) != _shares(baseline[0])
    assert moved_rows[1] == baseline[1]
    assert moved_rows[0]["population_total"] == baseline[0]["population_total"]


@pytest.mark.parametrize("corrupt_page", [0, 1])
def test_page_hash_mismatch_refuses_before_loading_data(
    tmp_path: Path, corrupt_page: int
) -> None:
    pins = []
    for index in range(2):
        page = tmp_path / f"sa1-2023-census-urp-p{index:03d}.json"
        page.write_bytes(b"tampered" if index == corrupt_page else b"{}")
        pins.append({"file": page.name, "sha256": hashlib.sha256(b"{}").hexdigest()})
    spec = _spec()
    spec["provenance"] = {"pages": pins, "layers": []}
    with pytest.raises(ValueError, match="SHA-256|sha256|hash"):
        generate_crosswalk(tmp_path, spec)


def test_generator_wires_population_names_audit_and_queenstown_sensitivity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the complete generator even when public source bytes are absent."""
    au_features = [
        _au("001", "Queenstown Bay", 0.0),
        _au("002", "Frankton East", 2.0),
        _au("003", "Rural", 4.0),
        _au("004", "Silverdale", 6.0),
        _au("005", "Frankton Other", 10.0),
        _au("006", "Silverdale Central", 12.0),
    ]
    ta_features = [
        _ta("001", "Queenstown-Lakes District"),
        _ta("002", "Other District", 10.0),
    ]
    ta_features[0]["geometry"] = _rectangle(0.0, 0.0, 8.0)
    ta_features[1]["geometry"] = _rectangle(10.0, 0.0, 4.0)
    sa1_features = [
        _sa1("a", 90, older=10),
        _sa1("b", 10, older=90),
        _sa1("c", 20, older=30),
        _sa1("d", 30, older=10),
        _sa1("e", 50, older=50),
        _sa1("f", 10, older=30),
    ]
    for feature, x in zip(sa1_features, (0.5, 2.5, 4.5, 6.5, 10.5, 12.5), strict=True):
        feature["centroid"]["x"] = x
    for code, count, older in (("g", -999, -997), ("h", None, None)):
        feature = _sa1(code, count, older)
        feature["centroid"]["x"] = 4.5
        sa1_features.append(feature)
    groups = {
        "sa1": sa1_features,
        "au": au_features,
        "ta": ta_features,
        "mbhg": [
            _meshblock(
                feature["attributes"]["SA12023_V1_00"],
                "001" if index < 4 or index > 5 else "002",
                "Queenstown-Lakes District"
                if index < 4 or index > 5
                else "Other District",
                str(index),
            )
            for index, feature in enumerate(sa1_features)
        ],
        "sa1poly": [],
    }
    aliases = {"VAR_1_3": "fixture Census 2023", "VAR_1_2": "fixture Census 2018"}
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), dict(aliases)),
    )
    spec = _spec()
    spec["definition_name_aliases"] = []
    spec["resolved_definition_entries"] = []
    definitions = _definitions(["Queenstown Bay", "Queenstown urban area"])
    definitions["areas"]["1"]["locations"][0]["heading"] = "Queenstown"
    definitions["areas"]["2"]["locations"] = [
        {
            "heading": "Wrong district heading",
            "statistical_area_units_2017": ["Silverdale Central"],
        }
    ]
    spec["area_definitions"] = definitions
    spec["heading_ta_expectations"] = {
        "Queenstown": ["Queenstown-Lakes District"],
        "Wrong district heading": ["Queenstown-Lakes District"],
    }
    original = copy.deepcopy(spec)
    report = {}
    generated = generate_crosswalk(tmp_path, spec, report=report)
    assert spec == original
    assert generated["area_definitions"] == definitions
    district, other = generated["rows"]
    assert district["population_total"] == 150
    assert _shares(district) == {1: 0.6, 2: 0.0, 3: 0.0, 4: 0.4}
    assert district["modal_area"] == 1
    assert "VAR_1_3" in district["population_source"]
    assert "2023-03-07" in district["population_source"]
    assert other["population_total"] == 60
    assert _shares(other) == {1: 0.0, 2: 1 / 6, 3: 0.0, 4: 5 / 6}
    assert generated["unmatched_definition_entries"] == [
        {
            "name": "Queenstown urban area",
            "heading": "Queenstown",
            "as_area": 1,
            "reason": "no exact match",
            "candidates": [],
        }
    ]
    assert "definition_heading_checks" not in generated
    assert "definition_heading_conflicts" not in generated
    conflicts = report["definition_heading_conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["name"] == "Silverdale Central"
    assert conflicts[0]["unexpected_ta_population"] == {"Other District": 10}
    sensitivity = generated["queenstown_sensitivity"]
    assert "sensitivity" in sensitivity["label"]
    assert sensitivity["area_units"] == [
        {"au_code": "002", "au_name": "Frankton East", "population": 10}
    ]
    assert sensitivity["moved_population"] == 10
    assert sensitivity["moved_population_share"] == 1 / 15
    assert _shares(sensitivity) == {1: 2 / 3, 2: 0.0, 3: 0.0, 4: 1 / 3}
    diagnostics = report["diagnostics"]
    assert diagnostics["national_layer_population"] == 210
    assert diagnostics["assigned_population"] == 210
    assert diagnostics["suppressed_values"]["count"] == 2
    assert diagnostics["queenstown_areas_1_and_4_only"] is True
    previous = diagnostics["stability_2018"]
    assert previous["national_layer_population"] == 220
    assert previous["suppressed_values"]["count"] == 2
    assert previous["rows"][0]["population_total"] == 140
    assert _shares(previous["rows"][0]) == {1: 1 / 14, 2: 0.0, 3: 0.0, 4: 13 / 14}
    assert previous["rows"][0]["modal_area"] == 4
    assert previous["largest_absolute_share_change"] == pytest.approx(37 / 70)
    assert previous["ta_code"] == "001"
    assert generated == generate_crosswalk(tmp_path, copy.deepcopy(generated))


def test_generator_uses_concordance_ta_and_reports_centroid_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sa1_features = [_sa1("moved", 27, 27), _sa1("land", 9, 9), _sa1("outside", 0, 0)]
    sa1_features[0]["centroid"]["x"] = 2.5
    sa1_features[2]["centroid"]["x"] = 2.75
    unit = _au("001", "Queenstown Bay")
    unit["geometry"] = _rectangle(0.0, 0.0, 4.0)
    groups = {
        "sa1": sa1_features,
        "au": [unit],
        "ta": [
            _ta("001", "Queenstown-Lakes District"),
            _ta("999", "Area Outside Territorial Authority", 2.0),
        ],
        "mbhg": [
            _meshblock("moved", "001", "Queenstown-Lakes District", "1"),
            _meshblock("land", "001", "Queenstown-Lakes District", "2"),
            _meshblock("outside", "999", "Area Outside Territorial Authority", "3"),
        ],
        "sa1poly": [],
    }
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), {}),
    )
    report = {}
    generated = generate_crosswalk(
        tmp_path, _synthetic_spec(["Queenstown Bay"]), report=report
    )
    district, outside = generated["rows"]
    assert district["population_total"] == 36
    assert outside["population_total"] == 0
    disagreements = report["ta_concordance_disagreements"]
    assert len(disagreements) == 1
    assert disagreements[0]["sa1_code"] == "moved"
    assert disagreements[0]["population"] == 27
    assert disagreements[0]["centroid_ta_code"] == "999"
    assert disagreements[0]["concordance_ta_code"] == "001"


def test_generator_applies_declared_whakarewarewa_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rotorua_sa1 = _sa1("rotorua", 12, 12)
    rotorua_sa1["centroid"]["x"] = 2.5
    groups = {
        "sa1": [_sa1("queenstown", 90, 90), rotorua_sa1],
        "au": [_au("001", "Queenstown Bay"), _au("540700", "Whaka", 2.0)],
        "ta": [
            _ta("001", "Queenstown-Lakes District"),
            _ta("024", "Rotorua District", 2.0),
        ],
        "mbhg": [
            _meshblock("queenstown", "001", "Queenstown-Lakes District", "1"),
            _meshblock("rotorua", "024", "Rotorua District", "2"),
        ],
        "sa1poly": [],
    }
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), {}),
    )
    spec = _synthetic_spec(["Queenstown Bay"])
    spec["area_definitions"]["areas"]["3"]["locations"] = [
        {"heading": "Rotorua", "statistical_area_units_2017": ["Whakarewarewa"]}
    ]
    spec["heading_ta_expectations"]["Rotorua"] = ["Rotorua District"]
    spec["definition_name_aliases"] = [
        {
            "definition_name": "Whakarewarewa",
            "au_code": "540700",
            "au_name": "Whaka",
            "evidence": "fixture correspondence",
        }
    ]
    report = {}
    generated = generate_crosswalk(tmp_path, spec, report=report)
    assert generated["unmatched_definition_entries"] == []
    assert _shares(generated["rows"][1]) == {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0}
    alias_check = next(
        check
        for check in report["definition_heading_checks"]
        if check["name"] == "Whakarewarewa"
    )
    assert alias_check["au_code"] == "540700"
    assert alias_check["observed_ta_population"] == {"Rotorua District": 12}


def _water_fixture(polygon: dict | None) -> dict:
    authority = _ta("001", "Queenstown-Lakes District")
    authority["geometry"] = _rectangle(0.0, 0.0, 4.0)
    features = [_sa1("land", 90, 90), _sa1("flagged", 10, 10)]
    features[1]["centroid"]["x"] = 2.5
    return {
        "sa1": features,
        "au": [_au("001", "Queenstown Bay"), _au("002", "Inlet fixture", 2.0)],
        "ta": [authority],
        "mbhg": [
            _meshblock("land", "001", "Queenstown-Lakes District", "1"),
            _meshblock("flagged", "001", "Queenstown-Lakes District", "2"),
        ],
        "sa1poly": [] if polygon is None else [_polygon_feature("flagged", polygon)],
    }


def test_generator_uses_flagged_interior_point_instead_of_water_centroid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    groups = _water_fixture(_rectangle(0.2, 0.2, 0.4))
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), {}),
    )
    report = {}
    generated = generate_crosswalk(
        tmp_path, _synthetic_spec(["Queenstown Bay"]), report=report
    )
    assert _shares(generated["rows"][0]) == {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}
    changes = report["flagged_sa1_changes"]
    assert len(changes) == 1
    assert changes[0]["sa1_code"] == "flagged"
    assert changes[0]["population"] == 10
    assert changes[0]["old_au_code"] == "002"
    assert changes[0]["new_au_code"] == "001"
    assert changes[0]["as_area_before"] == 4
    assert changes[0]["as_area_after"] == 1
    assert report["remaining_water_au_sa1s"] == []


@pytest.mark.parametrize("population", [0, 10])
def test_generator_unstaged_water_centroid_refuses_and_lists_sa1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, population: int
) -> None:
    groups = _water_fixture(None)
    groups["sa1"][1]["attributes"]["VAR_1_3"] = population
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), {}),
    )
    with pytest.raises(ValueError, match="water|Water") as error:
        generate_crosswalk(tmp_path, _synthetic_spec(["Queenstown Bay"]))
    assert "flagged" in str(error.value)


def test_generator_staged_interior_point_still_in_water_is_kept_and_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    groups = _water_fixture(_rectangle(2.2, 0.2, 0.4))
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), {}),
    )
    report = {}
    generated = generate_crosswalk(
        tmp_path, _synthetic_spec(["Queenstown Bay"]), report=report
    )
    assert _shares(generated["rows"][0]) == {1: 0.9, 2: 0.0, 3: 0.0, 4: 0.1}
    water = report["remaining_water_au_sa1s"]
    assert len(water) == 1
    assert water[0]["sa1_code"] == "flagged"
    assert water[0]["population"] == 10


def test_committed_spec_declares_alias_and_resolved_queenstown_without_bulk_audit() -> (
    None
):
    spec = _spec()
    assert "definition_heading_checks" not in spec
    assert "definition_heading_conflicts" not in spec
    assert "stability_2018" not in spec["diagnostics"]
    aliases = spec["definition_name_aliases"]
    assert len(aliases) == 1
    assert aliases[0]["definition_name"] == "Whakarewarewa"
    assert aliases[0]["au_code"] == "540700"
    assert aliases[0]["au_name"] == "Whaka"
    resolved = spec["resolved_definition_entries"]
    assert len(resolved) == 1
    assert resolved[0]["name"] == "Queenstown urban area"
    assert resolved[0]["au_codes"] == []
    assert resolved[0]["evidence"]["layer"] == "Urban_area_2014"
    assert resolved[0]["evidence"]["ua_code"] == "117"


def test_committed_rows_cover_68_tas_with_normalized_unique_modal_shares() -> None:
    spec = _spec()
    rows = spec["rows"]
    codes = [row["ta_code"] for row in rows]
    assert len(codes) == len(set(codes)) == 68
    assert spec["status"] != "requires_harvest"
    for row in rows:
        shares = _shares(row)
        assert set(shares) == {1, 2, 3, 4}
        assert len(row["shares"]) == 4
        assert all(0 <= share <= 1 for share in shares.values())
        assert abs(math.fsum(shares.values()) - 1) <= spec["method"]["share_tolerance"]
        largest = max(shares.values())
        winners = [area for area, share in shares.items() if share == largest]
        assert winners == [row["modal_area"]]


def test_every_ta_layer_code_appears_once(staged_sources: Path) -> None:
    codes = [
        feature["properties"]["TA2025_V1_00"]
        for page in sorted(staged_sources.glob("ta-2025-p*.geojson"))
        for feature in json.loads(page.read_text(encoding="utf-8"))["features"]
    ]
    committed = [row["ta_code"] for row in _spec()["rows"]]
    assert len(codes) == len(set(codes)) == 68
    assert sorted(committed) == sorted(codes)


def test_committed_rows_equal_fresh_pinned_generation(staged_sources: Path) -> None:
    spec = _spec()
    regenerated = generate_crosswalk(staged_sources, copy.deepcopy(spec))
    # Geometry-distance audit measurements live in the separate generated
    # report; the lean spec, including every count and share, replays exactly.
    assert regenerated["rows"] == spec["rows"]
    assert regenerated == spec


def test_population_variable_aliases_confirm_the_2023_and_2018_fields(
    staged_sources: Path,
) -> None:
    layer = json.loads(
        (
            staged_sources
            / "layer-2023_Census_totals_by_topic_for_individuals_by_SA1.json"
        ).read_text(encoding="utf-8")
    )
    aliases = {field["name"]: field["alias"] for field in layer["fields"]}
    for variable, year in (
        ("VAR_1_1", "2013"),
        ("VAR_1_2", "2018"),
        ("VAR_1_3", "2023"),
    ):
        assert year in aliases[variable]
        assert "usually resident population" in aliases[variable].lower()


def test_queenstown_sensitivity_excludes_ambiguous_but_listed_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    au_features = [
        _au("001", "Queenstown Bay", 0.0),
        _au("002", "Frankton", 2.0),
        _au("003", "Frankton", 4.0),
    ]
    authority = _ta("001", "Queenstown-Lakes District")
    authority["geometry"] = _rectangle(0.0, 0.0, 6.0)
    sa1_features = [
        _sa1("a", 90, older=90),
        _sa1("b", 10, older=10),
        _sa1("c", 10, older=10),
    ]
    for feature, x in zip(sa1_features, (0.5, 2.5, 4.5), strict=True):
        feature["centroid"]["x"] = x
    groups = {
        "sa1": sa1_features,
        "au": au_features,
        "ta": [authority],
        "mbhg": [
            _meshblock(
                feature["attributes"]["SA12023_V1_00"],
                "001",
                "Queenstown-Lakes District",
                str(index),
            )
            for index, feature in enumerate(sa1_features)
        ],
        "sa1poly": [],
    }
    monkeypatch.setattr(
        crosswalk_builder,
        "_load_sources",
        lambda source_dir, provenance: (copy.deepcopy(groups), {}),
    )
    spec = _synthetic_spec(["Queenstown Bay", "Frankton"])
    generated = generate_crosswalk(tmp_path, spec)
    assert generated["unmatched_definition_entries"] == [
        {
            "name": "Frankton",
            "heading": "Fixture location",
            "as_area": 1,
            "reason": "multiple exact matches",
            "candidates": [
                {"code": "002", "population": 10},
                {"code": "003", "population": 10},
            ],
        }
    ]
    central = generated["rows"][0]
    assert central["population_total"] == 110
    assert _shares(central) == {1: 9 / 11, 2: 0.0, 3: 0.0, 4: 2 / 11}
    sensitivity = generated["queenstown_sensitivity"]
    assert sensitivity["area_units"] == []
    assert sensitivity["moved_population"] == 0
    assert sensitivity["moved_population_share"] == 0
    assert sensitivity["shares"] == central["shares"]


@pytest.mark.parametrize("availability", ["unset", "missing_directory", "incomplete"])
def test_cli_skips_cleanly_when_pinned_sources_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, availability: str
) -> None:
    monkeypatch.delenv("NZ_AS_AREA_SOURCES", raising=False)
    output = tmp_path / "output.json"
    arguments = ["--out", str(output), "--spec", str(_SPEC_PATH)]
    if availability != "unset":
        source = (
            tmp_path / "absent" if availability == "missing_directory" else tmp_path
        )
        arguments.extend(["--sources", str(source)])
    crosswalk_builder.main(arguments)
    assert capsys.readouterr().out.startswith("SKIPPED:")
    assert not output.exists()
