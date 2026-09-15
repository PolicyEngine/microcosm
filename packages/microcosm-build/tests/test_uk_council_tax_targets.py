from __future__ import annotations

import copy
import json
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from tools.generate_uk_local_target_references import _area_signed_deferrals


def _crosswalk() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text()
    )


def _membership() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_target_reference_membership.json")
        .read_text()
    )


def _region_frame(rows: list[tuple[str, str, str]]) -> Frame:
    ids = np.arange(len(rows), dtype="int64")
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": ids, "person_benunit_id": ids, "person_household_id": ids}
            ),
            "benunit": pd.DataFrame({"benunit_id": ids}),
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "country": [row[0] for row in rows],
                    "region": [row[1] for row in rows],
                    "council_tax_band": [row[2] for row in rows],
                    "household_num_benunits": np.ones(len(rows)),
                }
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.ones(len(rows)), WeightKind.DESIGN)},
    )


def _registry_for(prefix: str) -> tuple[list, TargetRegistry]:
    refs = [
        r
        for r in load_country_spec("uk").target_references
        if r.name.startswith(prefix)
    ]
    registry = TargetRegistry(
        [
            TargetSpec(
                name=r.name,
                entity=r.entity,
                measure=r.measure,
                value=1,
                period=2025,
                source="synthetic",
                metadata=dict(r.metadata),
            )
            for r in refs
        ],
        country="uk",
    )
    return refs, registry


def test_national_voa_matrix_counts_only_its_region_for_every_band_and_total() -> None:
    """Each region cell must count only that region's English band households.

    The English VOA stock fans out over the nine English regions
    (microcosm#905); a Welsh, Scottish or Northern Irish household never
    enters any cell, and a Londoner never enters the North East's.
    """

    areas = (
        ("ENGLAND", "NORTH_EAST"),
        ("ENGLAND", "LONDON"),
        ("ENGLAND", "SOUTH_EAST"),
        ("WALES", "WALES"),
        ("SCOTLAND", "SCOTLAND"),
        ("NORTHERN_IRELAND", "NORTHERN_IRELAND"),
    )
    rows = [(country, region, band) for country, region in areas for band in "ABCDEFGH"]
    refs, registry = _registry_for("voa.council_tax_stock.")
    assert len(refs) == 81
    assert {r.ledger_selector["geography_level"] for r in refs} == {"region"}
    assert sorted({r.ledger_selector["geography_id"] for r in refs}) == [
        f"E1200000{index}" for index in range(1, 10)
    ]
    adapter = UKFrameTargetAdapter(_region_frame(rows))
    result = materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=registry
    )
    assert not result.skipped
    problem = build_constraint_matrix(
        adapter.to_frame(), registry.to_target_set(), weight_entity="household"
    )
    for name, values in zip(problem.names, problem.matrix.toarray(), strict=True):
        target_id, code = name.split("@")[:2]
        band = (
            target_id.rsplit("band_", 1)[-1].upper() if "band_" in target_id else None
        )
        region = UK_REGION_TIER_ENUM[code]
        expected = [
            country == "ENGLAND" and r == region and (band is None or b == band)
            for country, r, b in rows
        ]
        np.testing.assert_array_equal(values, expected, err_msg=name)


def test_national_ons_region_cells_count_only_their_region_at_person_grain() -> None:
    """The region predicate is a household fact projected to each person."""

    household_ids = np.arange(3, dtype="int64")
    person_ids = np.arange(4, dtype="int64")
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": person_ids,
                    "person_benunit_id": np.array([0, 0, 1, 2], dtype="int64"),
                    "person_household_id": np.array([0, 0, 1, 2], dtype="int64"),
                    "age": np.array([5.0, 40.0, 7.0, 3.0]),
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": household_ids}),
            "household": pd.DataFrame(
                {
                    "household_id": household_ids,
                    "region": ["LONDON", "WALES", "SCOTLAND"],
                    "household_num_benunits": np.ones(3),
                }
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.ones(3), WeightKind.DESIGN)},
    )
    refs, registry = _registry_for("ons.population.age_0_9_by_region@")
    assert len(refs) == 12
    adapter = UKFrameTargetAdapter(frame)
    result = materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=registry
    )
    assert not result.skipped
    problem = build_constraint_matrix(
        adapter.to_frame(), registry.to_target_set(), weight_entity="household"
    )
    # Matrix names carry the target period as a third "@" segment.
    by_name = {
        "@".join(name.split("@")[:2]): values
        for name, values in zip(problem.names, problem.matrix.toarray(), strict=True)
    }
    prefix = "ons.population.age_0_9_by_region@"
    # London: one child under ten; the 40-year-old is outside the band.
    assert by_name[f"{prefix}E12000007"].tolist() == [1.0, 0.0, 0.0]
    assert by_name[f"{prefix}W92000004"].tolist() == [0.0, 1.0, 0.0]
    assert by_name[f"{prefix}S92000003"].tolist() == [0.0, 0.0, 1.0]
    assert by_name[f"{prefix}N92000002"].tolist() == [0.0, 0.0, 0.0]
    assert by_name[f"{prefix}E12000001"].tolist() == [0.0, 0.0, 0.0]


def test_council_tax_band_cells_activate_and_defer_as_measured() -> None:
    membership = _membership()
    expected_active = {
        "a": 294,
        "b": 294,
        "c": 294,
        "d": 294,
        "e": 294,
        "f": 294,
        "g": 294,
        "h": 0,
    }
    for band, active_count in expected_active.items():
        target_id = f"voa.council_tax_stock.by_area.band_{band}"
        candidates = membership["targets"][target_id]["geography_levels"][
            "local_authority"
        ]["candidates"]
        assert len(candidates) == 361
        assert sum(row["status"] == "active" for row in candidates) == active_count


def test_council_tax_signed_deferrals_pin_exact_gaps() -> None:
    deferrals = _membership()["signed_deferrals"]
    by_reason: dict[str, list[dict]] = {}
    for row in deferrals:
        if row["reason_id"].startswith("council_tax_"):
            by_reason.setdefault(row["reason_id"], []).append(row)

    assert len(by_reason["council_tax_voa_scotland_absent"]) == 8
    assert {
        len(row["area_ids"]) for row in by_reason["council_tax_voa_scotland_absent"]
    } == {32}
    assert len(by_reason["council_tax_ni_domestic_rates"]) == 8
    assert {
        len(row["area_ids"]) for row in by_reason["council_tax_ni_domestic_rates"]
    } == {11}
    assert by_reason["council_tax_city_of_london_band_a_suppressed"][0]["area_ids"] == [
        "E09000001"
    ]
    wales = by_reason["council_tax_wales_country_control_absent"]
    expected_wales = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("W")
    )
    assert len(wales) == 8
    assert {row["target_id"] for row in wales} == {
        f"voa.council_tax_stock.by_area.band_{band}" for band in "abcdefgh"
    }
    assert {tuple(row["area_ids"]) for row in wales} == {expected_wales}
    assert len(expected_wales) == 22
    assert {row["defer_if_compiles"] for row in wales} == {True}
    assert {
        "no Wales country-level council-tax stock-by-band fact" in row["rationale"]
        for row in wales
    } == {True}
    band_h = by_reason["council_tax_band_h_spine_support_absent"]
    expected_english = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("E")
    )
    assert len(band_h) == 1
    assert tuple(band_h[0]["area_ids"]) == expected_english
    assert len(expected_english) == 296
    assert band_h[0]["defer_if_compiles"] is True
    assert "170 band-H households from 49 raw FRS households" in band_h[0]["rationale"]
    # K=15 is the ruled clone count; the K=10 figure rides as history.
    assert (
        "76 of the 296 authorities draw no band-H household" in band_h[0]["rationale"]
    )
    assert "84 of 296 at K=10" in band_h[0]["rationale"]
    assert "council_tax_wales_band_h_absent" not in by_reason


def test_council_tax_activation_adds_2058_references() -> None:
    membership = _membership()
    active = 0
    for band in "abcdefgh":
        target_id = f"voa.council_tax_stock.by_area.band_{band}"
        candidates = membership["targets"][target_id]["geography_levels"][
            "local_authority"
        ]["candidates"]
        active += sum(row["status"] == "active" for row in candidates)
    assert active == 2_058


def test_support_floor_deferrals_cover_the_two_authorities_remaining_cells() -> None:
    rows = [
        row
        for row in _membership()["signed_deferrals"]
        if row["reason_id"] == "local_authority_support_floor_excluded"
    ]
    assert len(rows) == 24
    assert sum(len(row["area_ids"]) for row in rows) == 43
    assert {
        area_id: sum(area_id in row["area_ids"] for row in rows)
        for area_id in ("E06000053", "E09000001")
    } == {"E06000053": 20, "E09000001": 23}
    assert all(row["defer_if_compiles"] is True for row in rows)
    assert all(not row["target_id"].endswith("band_h") for row in rows)
    assert {
        "their rows stay in the solve through the constituency families and "
        "the national rows" in row["rationale"]
        for row in rows
    } == {True}


def test_a14_deferral_declarations_cover_only_currently_active_cells() -> None:
    declarations = _area_signed_deferrals(load_uk_population_contract(), _crosswalk())
    band_h = [
        row
        for row in declarations
        if row.reason_id == "council_tax_band_h_spine_support_absent"
    ]
    expected_english = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("E")
    )
    assert len(band_h) == 1
    assert band_h[0].target_id == "voa.council_tax_stock.by_area.band_h"
    assert band_h[0].area_ids == expected_english
    assert len(expected_english) == 296
    assert band_h[0].defer_if_compiles is True

    support = [
        row
        for row in declarations
        if row.reason_id == "local_authority_support_floor_excluded"
    ]
    assert len(support) == 24
    assert sum(len(row.area_ids) for row in support) == 43
    by_area = {
        area_id: sum(area_id in row.area_ids for row in support)
        for area_id in ("E06000053", "E09000001")
    }
    assert by_area == {"E06000053": 20, "E09000001": 23}
    assert all(row.defer_if_compiles for row in support)
    assert all(not row.target_id.endswith("band_h") for row in support)


def test_declared_deferral_roster_matching_no_crosswalk_area_refuses() -> None:
    crosswalk = copy.deepcopy(_crosswalk())
    area_ids = crosswalk["levels"]["local_authority"]["area_ids"]
    area_ids.remove("E09000001")
    area_ids.append("E09999999")

    with pytest.raises(ValueError, match="unmatched area id.*E09000001"):
        _area_signed_deferrals(load_uk_population_contract(), crosswalk)


@pytest.mark.parametrize(
    ("prefix", "expected", "name"),
    [
        ("S", 32, "Scottish"),
        ("N", 11, "Northern Ireland"),
        ("W", 22, "Welsh"),
        ("E", 296, "English"),
    ],
)
def test_council_tax_country_masks_refuse_roster_count_drift(
    prefix: str, expected: int, name: str
) -> None:
    crosswalk = copy.deepcopy(_crosswalk())
    area_ids = crosswalk["levels"]["local_authority"]["area_ids"]
    area_ids.remove(next(area_id for area_id in area_ids if area_id.startswith(prefix)))

    with pytest.raises(
        ValueError,
        match=rf"{name}.*expected {expected}.*measured {expected - 1}",
    ):
        _area_signed_deferrals(load_uk_population_contract(), crosswalk)
