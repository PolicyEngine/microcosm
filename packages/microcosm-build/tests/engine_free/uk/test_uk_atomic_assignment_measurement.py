"""The step-2 measurement harness on an invented pool: expectations, stability, disclosure.

No spine, publisher file or country engine: the toy supports and toy ladder
from the fixtures stand in for the pinned artifacts, and the harness's pure
measurement functions are exercised directly.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    SYSTEMS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.local_authority_input import (
    resolve_local_authority_engine_keys,
)
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_support_payloads,
)
from test_support.microcosm_build.uk_full_population_graph import SOURCE_VINTAGE
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)
from test_support.paths import paths_for

CLONE = ladder_clone_index_column("household")


@pytest.fixture(scope="module")
def harness():
    path = (
        paths_for("microcosm-build").repository
        / "tools/measure_uk_atomic_assignment.py"
    )
    spec = importlib.util.spec_from_file_location("uk_assignment_harness_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool(
    n_per_region: int, regions=("LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"), k=1
):
    rows = []
    household_id = 1
    for region in regions:
        for source in range(n_per_region):
            for clone in range(k):
                rows.append(
                    {
                        "household_id": household_id,
                        "source_household_id": source
                        + 1
                        + 1000 * regions.index(region),
                        "region": region,
                        CLONE: clone,
                        "household_weight": 1.0 / k,
                    }
                )
                household_id += 1
    table = pd.DataFrame(rows)
    table["region"] = table["region"].astype("string")
    table[IDENTITY_COLUMN] = pd.array(
        [
            household_draw_key(
                source="frs",
                source_vintage=SOURCE_VINTAGE,
                source_household_id=int(s),
                clone_path=(("geographic_support", int(c)),) if c else (),
            )
            for s, c in zip(table["source_household_id"], table[CLONE], strict=True)
        ],
        dtype="string",
    )
    return table


def _assigned(table, definition, supports):
    assigned = pd.concat(
        [table, geo.assign_atomic(table, definition, supports)], axis=1
    )
    derived = geo.derive_geography(assigned, definition, supports)
    # The engine input the UK-side node resolves after the shared derive.
    derived["local_authority"] = resolve_local_authority_engine_keys(
        derived["local_authority_code"]
    )
    return pd.concat([assigned, derived], axis=1)


def test_single_stage_expectation_matches_realized_draws_and_the_legacy_constituency_law(
    harness, toy_ladder
):
    ladder, _ = toy_ladder
    payloads = toy_support_payloads()
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    supports = {s: geo.decode_atomic_support(p) for s, p in payloads.items()}
    pool = _pool(400)
    household = _assigned(pool, definition, supports)
    expected = harness.expected_single_stage_area_support(household, ladder)
    legacy = harness.expected_area_support("legacy", household, ladder)
    # London's two output areas carry 40 census households each: half of the
    # 400 London rows are expected in each constituency under both laws.
    by_code = expected.set_index(["area_type", "area_code"])["expected_rows"]
    assert by_code[("constituency", "E14000001")] == pytest.approx(200.0)
    assert by_code[("constituency", "W07000041")] == pytest.approx(400.0)
    assert by_code[("constituency", "S14000001")] == pytest.approx(400.0)
    legacy_codes = legacy.set_index(["area_type", "area_code"])["expected_rows"]
    constituency = expected[expected["area_type"] == "constituency"]
    for _, row in constituency.iterrows():
        assert legacy_codes[("constituency", row["area_code"])] == pytest.approx(
            row["expected_rows"]
        )
    summaries = harness.realized_area_support(household, ladder)
    rows = harness._cell_rows(
        expected, summaries, household, harness._area_regions(ladder)
    )
    london = rows[
        (rows["level"] == "constituency") & (rows["area_code"].str.startswith("E14"))
    ]
    assert london["rows"].sum() == 400
    assert (london["rows"] - london["expected_rows"]).abs().max() / 200.0 < 0.15
    assert (rows.loc[rows["expected_rows"] >= 5, "z"].dropna().abs() < 4.5).all()
    summary = harness._level_summary(rows, "constituency")
    assert summary["areas_with_expected_ge_100"] == 5
    assert summary["max_relative_error_expected_ge_100"] < 0.15


def test_identity_stability_holds_on_reversed_rows_and_the_clone_zero_subset(
    harness, toy_ladder
):
    payloads = toy_support_payloads()
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    supports = {s: geo.decode_atomic_support(p) for s, p in payloads.items()}
    household = _assigned(_pool(30, k=3), definition, supports)
    stability = harness.identity_stability(household, definition, supports)
    assert stability == {
        "production_equals_in_process": True,
        "reversed_rows_equal": True,
        "clone_zero_subset_equal": True,
        "keys_unique": True,
    }
    tampered = household.copy()
    tampered.loc[tampered.index[0], "atomic_area_code"] = "E00000099"
    assert (
        harness.identity_stability(tampered, definition, supports)[
            "production_equals_in_process"
        ]
        is False
    )


def test_breach_tables_and_disclosure_suppression(harness, toy_ladder):
    ladder, _ = toy_ladder
    payloads = toy_support_payloads()
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    supports = {s: geo.decode_atomic_support(p) for s, p in payloads.items()}
    household = _assigned(_pool(2), definition, supports)
    summaries = harness.realized_area_support(household, ladder)
    breaches = harness.breach_tables(summaries, {})
    assert set(breaches["by_geography_level"]) == {"constituency", "la"}
    assert breaches["by_geography_level"]["la"]["breaches"]["any"] >= 1
    assert harness._suppress(2.0, 3) == "<3"
    assert harness._suppress(0.0, 3) == 0.0
    assert harness._suppress(3.0, 3) == 3.0
    # z is a function of the suppressed count, so it is blanked with it
    # (#932 round 1: the counts were recoverable from expected_rows and z).
    row = {"rows": 2.0, "ess": 1.5, "sources": 2.0, "z": 0.3, "expected_rows": 1.0}
    assert harness._suppressed_area(row, 3) == {
        **row,
        "rows": "<3",
        "ess": "<3",
        "sources": "<3",
        "z": None,
    }
    assert harness._suppressed_area({**row, "rows": 3.0}, 3)["z"] == 0.3
    assert harness._suppressed_area({**row, "rows": 0.0}, 3)["z"] == 0.3


def _level_areas(level, counts, expected):
    return [
        {
            "level": level,
            "area_code": f"{level}-{index:03d}",
            "region_code": "E12000001",
            "expected_rows": expected,
            "rows": float(count),
            "ess": float(count),
            "sources": float(count),
            "z": 0.1,
        }
        for index, count in enumerate(counts)
    ]


def test_complementary_suppression_leaves_no_suppressed_count_determined(harness):
    # Round 2 of microcosm#1057, item 8: with zeros published and the level total
    # known (the expected rows sum to it), two suppressed counts of 2 were pinned
    # by the residual. The complement adds an unknown of at least minimum + 1.
    counts = [2, 2, 0, 3, 7, 12, 30]
    areas = [
        harness._suppressed_area(row, 3)
        for row in _level_areas("constituency", counts, sum(counts) / len(counts))
    ]
    protected = harness._complement_suppressed(areas, 3)
    withheld = [area for area in protected if area["rows"] == harness.WITHHELD]
    assert len(withheld) == 1
    assert withheld[0]["z"] is None and withheld[0]["ess"] == harness.WITHHELD
    assert harness._complement_suppressed(protected, 3) == protected
    total = sum(area["expected_rows"] for area in protected)
    published = sum(a["rows"] for a in protected if not isinstance(a["rows"], str))
    suppressed = sum(1 for a in protected if a["rows"] == "<3")
    residual = total - published
    # The reader's feasible sums for the suppressed counts span more than one value.
    assert min(2 * suppressed, residual - 4) >= suppressed + 1


def test_resuppression_withholds_summaries_that_carry_a_suppressed_value(harness):
    counts = [1, 2, 9, 10, 11]
    areas = [
        harness._suppressed_area({**row, "z": float(index)}, 3)
        for index, row in enumerate(_level_areas("la", counts, 6.0))
    ]
    evidence = {
        "minimum_count": 3,
        "cells": [
            {
                "areas": areas,
                "levels": {
                    "la": {
                        "min_rows": 1.0,
                        "min_ess": 1.0,
                        "min_sources": 1.0,
                        "max_abs_z": 4.0,
                        "share_abs_z_gt_3": 0.2,
                    }
                },
                "breaches": {"la": {"rows": {"min": 1.0, "p10": 1.0, "median": 9.0}}},
            }
        ],
    }
    out = harness.resuppress_evidence(evidence)
    level = out["cells"][0]["levels"]["la"]
    assert level["min_rows"] is None and level["min_sources"] is None
    # The largest |z| belonged to a suppressed or withheld area; the summary is
    # over published areas only.
    published_z = [abs(a["z"]) for a in out["cells"][0]["areas"] if a["z"] is not None]
    assert level["max_abs_z"] == max(published_z)
    assert out["cells"][0]["breaches"]["la"]["rows"] == {
        "min": None,
        "p10": None,
        "median": 9.0,
    }
    assert harness.resuppress_evidence(out) == out


def test_committed_931_evidence_is_internally_consistent(harness):
    root = paths_for("microcosm-build").repository
    path = root / "docs/evidence/uk-931/atomic-assignment-cells.json"
    if not path.is_file():
        pytest.skip("committed #931 evidence is not present in this checkout")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["minimum_count"] >= 3
    assert set(evidence["supports"]) == set(SYSTEMS)
    assert evidence["pool"]["sample_fraction"] <= 0.01
    assert {(c["law"], c["n_clones"]) for c in evidence["cells"]} == {
        (law, k) for law in evidence["laws"] for k in evidence["n_clones"]
    }
    for cell in evidence["cells"]:
        assert cell["gate_outcome"] == "pass"
        for area in cell["areas"]:
            for field in ("rows", "ess", "sources"):
                value = area[field]
                if isinstance(value, str):
                    assert value in {f"<{evidence['minimum_count']}", harness.WITHHELD}
                else:
                    assert value == 0 or value >= evidence["minimum_count"]
            if isinstance(area["rows"], str):
                assert area["z"] is None
            assert "household_id" not in area and "source_household_id" not in area
        if cell["law"] == "keyed":
            assert all(cell["identity_stability"].values())
        minimum = evidence["minimum_count"]
        for level, summary in cell["levels"].items():
            areas = [area for area in cell["areas"] if area["level"] == level]
            suppressed = sum(1 for area in areas if area["rows"] == f"<{minimum}")
            if not suppressed:
                continue
            # One complement per level, and no summary carries a suppressed value.
            assert sum(1 for area in areas if area["rows"] == harness.WITHHELD) == 1
            assert summary["min_rows"] is None and summary["min_sources"] is None
            # A reader who knows the level total (the expected rows sum to it)
            # and the rule can bound the suppressed counts' sum, but not pin it.
            total = round(sum(area["expected_rows"] for area in areas))
            published = sum(
                area["rows"] for area in areas if not isinstance(area["rows"], str)
            )
            highest = min((minimum - 1) * suppressed, total - published - minimum - 1)
            assert highest >= suppressed + 1, (cell["law"], cell["n_clones"], level)
    # The committed file is a fixed point of the harness's disclosure pass.
    assert harness.resuppress_evidence(evidence) == evidence
    assert np.isfinite(sum(c["wall_seconds"] for c in evidence["cells"]))


def test_committed_931_receipts_render_the_level_summaries_from_the_evidence(harness):
    # Round 2 of the microcosm#1059 review, item 7: a hand-copied receipts table
    # printed the K=15 constituency minimum that the evidence withholds, which
    # pinned the suppressed counts again. The R1 table's summary cells must be the
    # harness's rendering of the committed evidence.
    root = paths_for("microcosm-build").repository
    evidence_path = root / "docs/evidence/uk-931/atomic-assignment-cells.json"
    receipts_path = root / "experiments/931-uk-atomic-assignment-receipts.md"
    if not evidence_path.is_file() or not receipts_path.is_file():
        pytest.skip("committed #931 evidence or receipts are not present")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    section = receipts_path.read_text(encoding="utf-8").split("## R1", 1)[1]
    section = section.split("## R2", 1)[0]
    table = [line for line in section.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in table[0].strip("|").split("|")]
    columns = {
        ("constituency", "min"): header.index("const. min rows / ESS / sources"),
        ("constituency", "z"): header.index(
            "const. max abs z / share abs z>3 (n published)"
        ),
        ("la", "min"): header.index("LA min rows / ESS / sources"),
        ("la", "z"): header.index("LA max abs z / share>3 (n published)"),
    }
    cells = {(cell["law"], cell["n_clones"]): cell for cell in evidence["cells"]}
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table[2:]]
    assert {(row[0], int(row[1])) for row in rows} == set(cells)
    for row in rows:
        cell = cells[(row[0], int(row[1]))]
        for level in ("constituency", "la"):
            minima, z = harness.level_summary_cells(cell, level)
            assert row[columns[(level, "min")]] == minima, (row[:2], level)
            assert row[columns[(level, "z")]] == z, (row[:2], level)
