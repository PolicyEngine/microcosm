from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    compile_uk_target_registry,
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


def test_national_mhclg_region_cells_count_only_their_region_for_every_band() -> None:
    """Each composed region cell must count only that region's English band households.

    The English stock family binds the MHCLG taxbase authority rows composed
    per region (microcosm#929, on the microcosm#905 tier); a Welsh, Scottish
    or Northern Irish household never enters any cell, and a Londoner never
    enters the North East's.
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
    refs, registry = _registry_for("mhclg.council_tax_stock.")
    assert len(refs) == 81
    assert {r.metadata["geography_level"] for r in refs} == {"region"}
    assert sorted({r.metadata["geography_id"] for r in refs}) == [
        f"E1200000{index}" for index in range(1, 10)
    ]
    # A composed cell selects its authorities' rows and declares the count.
    assert {r.ledger_selector["geography_level"] for r in refs} == {"local_authority"}
    assert {r.metadata["composed_from_level"] for r in refs} == {"local_authority"}
    assert {r.value_operation for r in refs} == {"linear_combination"}
    assert sum(int(r.metadata["composed_member_count"]) for r in refs) == 9 * 296
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


def test_national_welsh_country_rows_count_only_welsh_households_by_band() -> None:
    """The StatsWales CT1 rows are one country control per band A-I."""

    areas = (("ENGLAND", "LONDON"), ("WALES", "WALES"), ("SCOTLAND", "SCOTLAND"))
    rows = [
        (country, region, band) for country, region in areas for band in "ABCDEFGHI"
    ]
    refs, registry = _registry_for("welshgov.council_tax_stock.")
    assert len(refs) == 10
    assert {r.ledger_selector["geography_id"] for r in refs} == {"W92000004"}
    assert {r.value_operation for r in refs} == {"linear_combination"}
    adapter = UKFrameTargetAdapter(_region_frame(rows))
    result = materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=registry
    )
    assert not result.skipped
    problem = build_constraint_matrix(
        adapter.to_frame(), registry.to_target_set(), weight_entity="household"
    )
    for name, values in zip(problem.names, problem.matrix.toarray(), strict=True):
        target_id = name.split("@")[0]
        band = (
            target_id.rsplit("band_", 1)[-1].upper() if "band_" in target_id else None
        )
        expected = [
            r == "WALES" and (band is None or b == band) for country, r, b in rows
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
    expected = {
        "mhclg": ("abcdefgh", 296, {"h": 0}, 294),
        "welshgov": ("abcdefghi", 22, {}, 22),
        # Shetland band H has no band-H clone at K=15 (support deferral).
        "scotgov": ("abcdefgh", 32, {"h": 31}, 32),
    }
    for source, (bands, roster, overrides, active_default) in expected.items():
        for band in bands:
            target_id = f"{source}.council_tax_stock.by_area.band_{band}"
            level = membership["targets"][target_id]["geography_levels"][
                "local_authority"
            ]
            candidates = level["candidates"]
            # Each family is scoped to its nation's roster (microcosm#929).
            assert len(candidates) == roster, target_id
            assert sum(row["status"] == "active" for row in candidates) == (
                overrides.get(band, active_default)
            ), target_id
    scopes = membership["area_scope_by_target_id"]
    assert (
        len(scopes["mhclg.council_tax_stock.by_area.band_a"]["local_authority"]) == 296
    )
    assert scopes["welshgov.council_tax_stock.by_area.band_i"]["local_authority"] == [
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("W")
    ]


def test_council_tax_signed_deferrals_pin_exact_gaps() -> None:
    deferrals = _membership()["signed_deferrals"]
    by_reason: dict[str, list[dict]] = {}
    for row in deferrals:
        if row["reason_id"].startswith("council_tax_"):
            by_reason.setdefault(row["reason_id"], []).append(row)

    # The Scottish, Northern Irish, Welsh and City-of-London masks retired with
    # the taxbase basis (microcosm#929): Scotland and Wales bind their own
    # returns, Northern Ireland is outside every family's roster, and MHCLG
    # publishes the City's band A.
    assert set(by_reason) == {"council_tax_band_h_spine_support_absent"}
    band_h = by_reason["council_tax_band_h_spine_support_absent"]
    expected_english = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("E")
    )
    # England is wholly deferred (A14); Scotland defers the one council the
    # rowwise support check refuses at K=15 (Shetland), the other 31 bind.
    assert [(row["target_id"], len(row["area_ids"])) for row in band_h] == [
        ("mhclg.council_tax_stock.by_area.band_h", 296),
        ("scotgov.council_tax_stock.by_area.band_h", 1),
    ]
    assert tuple(band_h[0]["area_ids"]) == expected_english
    assert len(expected_english) == 296
    assert all(row["defer_if_compiles"] is True for row in band_h)
    assert "170 band-H households from 49 raw FRS households" in band_h[0]["rationale"]
    assert band_h[1]["area_ids"] == ["S12000027"]
    assert "zero household support" in band_h[1]["rationale"]
    # K=15 is the ruled clone count; the K=10 figure rides as history.
    assert (
        "76 of the 296 authorities draw no band-H household" in band_h[0]["rationale"]
    )
    assert "84 of 296 at K=10" in band_h[0]["rationale"]


def test_council_tax_activation_binds_2511_references() -> None:
    membership = _membership()
    active = 0
    for target_id, payload in membership["targets"].items():
        if ".council_tax_stock.by_area." not in target_id:
            continue
        candidates = payload["geography_levels"]["local_authority"]["candidates"]
        active += sum(row["status"] == "active" for row in candidates)
    # 2,058 English A-G cells (microcosm#762) + 198 Welsh A-I + 255 Scottish
    # A-H (microcosm#929; Shetland band H is a support deferral); the 296
    # English band-H cells stay deferred.
    assert active == 2_511


def test_barnsley_and_sheffield_bind_their_2025_rows_through_the_code_aliases() -> None:
    membership = _membership()
    level = membership["targets"]["mhclg.council_tax_stock.by_area.band_a"][
        "geography_levels"
    ]["local_authority"]
    by_area = {row["geography_id"]: row for row in level["candidates"]}
    for area_id in ("E08000016", "E08000019"):
        assert by_area[area_id]["status"] == "active"
        assert by_area[area_id]["resolved_fact_period"] == "2025-10"
    holds = [
        row
        for row in membership["uprating_holds"]
        if row["target_id"].startswith("mhclg.council_tax_stock.by_area.")
        and row["geography_id"] in ("E08000016", "E08000019")
    ]
    assert {row["from"] for row in holds} == {"2025-10"}
    aliases = _crosswalk()["levels"]["local_authority"]["code_aliases"]
    assert aliases["E08000016"]["alias_codes"] == ["E08000038"]
    assert aliases["E08000019"]["alias_codes"] == ["E08000039"]


def test_support_floor_deferrals_cover_the_two_authorities_remaining_cells() -> None:
    rows = [
        row
        for row in _membership()["signed_deferrals"]
        if row["reason_id"] == "local_authority_support_floor_excluded"
    ]
    # 24 local targets before microcosm#929; the three council-tax families
    # add 25 by_area targets and retire eight, and the City's band A is no
    # longer a separate suppression.
    assert len(rows) == 41
    assert sum(len(row["area_ids"]) for row in rows) == 78
    assert {
        area_id: sum(area_id in row["area_ids"] for row in rows)
        for area_id in ("E06000053", "E09000001")
    } == {"E06000053": 37, "E09000001": 41}
    assert all(row["defer_if_compiles"] is True for row in rows)
    # The English band-H family is wholly deferred on spine support, so it
    # carries no support-floor row; the Welsh and Scottish band-H targets do
    # (the two English areas are outside their scope, so the rows are inert).
    assert "mhclg.council_tax_stock.by_area.band_h" not in {
        row["target_id"] for row in rows
    }
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
    assert [(row.target_id, row.area_ids) for row in band_h] == [
        ("mhclg.council_tax_stock.by_area.band_h", expected_english),
        ("scotgov.council_tax_stock.by_area.band_h", ("S12000027",)),
    ]
    assert len(expected_english) == 296
    assert all(row.defer_if_compiles for row in band_h)

    support = [
        row
        for row in declarations
        if row.reason_id == "local_authority_support_floor_excluded"
    ]
    assert len(support) == 41
    assert sum(len(row.area_ids) for row in support) == 78
    by_area = {
        area_id: sum(area_id in row.area_ids for row in support)
        for area_id in ("E06000053", "E09000001")
    }
    assert by_area == {"E06000053": 37, "E09000001": 41}
    assert all(row.defer_if_compiles for row in support)
    assert "mhclg.council_tax_stock.by_area.band_h" not in {
        row.target_id for row in support
    }


def test_declared_deferral_roster_matching_no_crosswalk_area_refuses() -> None:
    crosswalk = copy.deepcopy(_crosswalk())
    area_ids = crosswalk["levels"]["constituency"]["area_ids"]
    area_ids.remove("E14001416")
    area_ids.append("E14999999")

    with pytest.raises(ValueError, match="unmatched area id.*E14001416"):
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


# The untracked default location of the pinned consumer feed (the same one the
# regeneration test in test_uk_target_references.py reads).
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"


def _pinned_feed_facts() -> list[dict]:
    """The pinned Chronicle consumer facts, or skip (the artifact is untracked)."""

    root = Path(__file__).resolve().parents[3]
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else root / STABLE_UK_FACT_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    pin = load_uk_chronicle_feed()
    assert hashlib.sha256(feed.read_bytes()).hexdigest() == pin.facts_sha256
    with feed.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_composed_english_region_cells_sum_to_the_publisher_england_row() -> None:
    """The nine composed region cells are the publisher's England row, split.

    MHCLG prints no region row, so the region controls are consumer rollups
    of the 296 billing-authority facts (microcosm#929). The England row is
    published, and the same operands resolved at E92000001 must equal the
    sum of the nine cells to the unit, band by band, the check the binding
    notes cite. Feed-gated like the regeneration test.
    """

    facts = _pinned_feed_facts()
    spec = load_country_spec("uk")
    composed = {
        reference.name: reference
        for reference in spec.target_references
        if reference.name.startswith("mhclg.council_tax_stock.")
    }
    assert len(composed) == 81
    compiled = compile_uk_target_registry(facts, target_period=2025).registry
    values = {
        target.name.rsplit("@", 1)[0]
        if target.name.endswith("@2025")
        else target.name: (target.value)
        for target in compiled.specs
        if target.name.startswith("mhclg.council_tax_stock.")
    }
    assert len(values) == 81, sorted(values)[:3]

    suffixes = [f"band_{band}" for band in "abcdefgh"] + ["total"]
    england_references = []
    for suffix in suffixes:
        template = composed[f"mhclg.council_tax_stock.{suffix}@E12000001"]
        england_references.append(
            replace(
                template,
                name=f"mhclg.council_tax_stock.{suffix}@E92000001",
                ledger_selector={
                    **template.ledger_selector,
                    "geography_level": "country",
                    "geography_id": ["E92000001"],
                },
                value_operands=tuple(
                    {**operand, "expected_member_count": 1}
                    for operand in template.value_operands
                ),
                metadata={
                    "contract_target_id": f"mhclg.council_tax_stock.{suffix}",
                    "measure_kind": "prepared_column",
                    "geography_level": "country",
                    "geography_id": "E92000001",
                },
            )
        )
    registry = compile_ledger_target_references(facts, england_references, country="uk")
    published = {
        suffix: float(row.value)
        for suffix, row in zip(suffixes, registry.specs, strict=True)
    }
    for suffix in suffixes:
        region_sum = sum(
            values[f"mhclg.council_tax_stock.{suffix}@{code}"]
            for code in (f"E1200000{index}" for index in range(1, 10))
        )
        assert region_sum == pytest.approx(published[suffix], abs=0.5), suffix
    # The publisher's England row on the occupied-chargeable basis (CTB 2025:
    # line 7 plus A- for band A, minus lines 11 and 15).
    assert published["band_a"] == pytest.approx(5_590_029, abs=0.5)
    assert published["total"] == pytest.approx(24_246_267, abs=0.5)
    assert sum(published[f"band_{band}"] for band in "abcdefgh") == pytest.approx(
        published["total"], abs=0.5
    )
