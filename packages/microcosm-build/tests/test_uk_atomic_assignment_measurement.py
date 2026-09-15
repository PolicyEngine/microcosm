"""The step-2 measurement harness on an invented pool: expectations, stability, disclosure.

No spine, publisher file or country engine: the toy supports and toy ladder
from the fixtures stand in for the pinned artifacts, and the harness's pure
measurement functions are exercised directly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_uk_full_population_graph import SOURCE_VINTAGE
from test_uk_ladder_rowwise_clone import toy_ladder as toy_ladder
from uk_atomic_support_fixtures import toy_support_payloads

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    SYSTEMS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column

CLONE = ladder_clone_index_column("household")


@pytest.fixture(scope="module")
def harness():
    import microcosm.build

    path = (
        Path(microcosm.build.__file__).resolve().parents[5]
        / "tools/measure_uk_atomic_assignment.py"
    )
    if not path.is_file():
        path = (
            Path(__file__).resolve().parents[3]
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
    return pd.concat(
        [assigned, geo.derive_geography(assigned, definition, supports)], axis=1
    )


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


def test_committed_931_evidence_is_internally_consistent():
    import microcosm.build

    root = Path(microcosm.build.__file__).resolve().parents[5]
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
                    assert value == f"<{evidence['minimum_count']}"
                else:
                    assert value == 0 or value >= evidence["minimum_count"]
            assert "household_id" not in area and "source_household_id" not in area
        if cell["law"] == "keyed":
            assert all(cell["identity_stability"].values())
    assert np.isfinite(sum(c["wall_seconds"] for c in evidence["cells"]))
