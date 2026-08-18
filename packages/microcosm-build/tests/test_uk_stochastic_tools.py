from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import uk_national_frame

ROOT = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_childcare_draws_use_seeded_default_rng_not_global_seed(monkeypatch) -> None:
    tool = _load_tool("fit_uk_childcare_takeup")

    def forbidden_seed(*args, **kwargs):
        raise AssertionError("np.random.seed must not be used")

    monkeypatch.setattr(tool.np.random, "seed", forbidden_seed)
    first = tool.draw_childcare_inputs(5, tool.INITIAL_PARAMS, seed=42)
    second = tool.draw_childcare_inputs(5, tool.INITIAL_PARAMS, seed=42)

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_childcare_fit_receipt_is_deterministic_with_seeded_runner(
    tmp_path: Path,
) -> None:
    tool = _load_tool("fit_uk_childcare_takeup")
    input_h5 = tmp_path / "fixture.h5"
    input_h5.write_bytes(b"synthetic")

    def runner(path: Path, params: np.ndarray, seed: int):
        scale = float(params.sum()) + seed * 0.0 + len(path.read_bytes()) * 0.0
        spending = {
            "tfc": 0.63 + scale * 0.0,
            "extended": 2.5,
            "targeted": 0.6,
            "universal": 1.7,
        }
        caseload = {
            "tfc": 985,
            "extended": 740,
            "targeted": 130,
            "universal": 490,
        }
        return spending, caseload

    first = tool.fit_childcare_takeup(
        input_h5, seed=42, maxiter=1, runner=runner, generated_at="2026-08-17"
    )
    second = tool.fit_childcare_takeup(
        input_h5, seed=42, maxiter=1, runner=runner, generated_at="2026-08-17"
    )

    assert first == second
    assert first["input_sha256"]
    assert first["seed"] == 42
    assert first["target_citations"]["tfc"].startswith("HMRC")


def test_identity_stability_receipt_compares_by_entity_id() -> None:
    tool = _load_tool("verify_uk_identity_stability")
    frame = _frame()

    def transform(candidate):
        household = candidate.table("household").copy()
        household["property_purchased"] = household["household_id"] % 2 == 0
        return uk_national_frame(
            person=candidate.table("person").copy(),
            benunit=candidate.table("benunit").copy(),
            household=household,
            time_period="2023",
            household_weights=candidate.weights_for("household").values,
        )

    receipt = tool.identity_stability_receipt(
        frame,
        transform=transform,
        columns_by_entity={"household": ("property_purchased",)},
    )

    assert receipt["identical"] is True
    assert receipt["mismatches"] == {}


def test_e4_identity_receipt_survives_permutation_on_synthetic_frame() -> None:
    tool = _load_tool("verify_uk_identity_stability")

    class Contract:
        def rate(self, key: str) -> float:
            return 0.5 if not key.startswith("scp") else 0.9

        def continuous_entry(self, key: str):
            return {"mean": 15.019, "sd": 4.972, "lower": 0, "upper": 30}

    person = pd.DataFrame(
        {
            "person_id": [101, 102, 201, 301],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [1, 1, 1, 2],
            "age": [5, 6, 40, 70],
            "child_benefit_reported": [0.0, 10.0, 0.0, 0.0],
            "pension_credit_reported": [0.0, 0.0, 0.0, 5.0],
            "universal_credit_reported": [0.0, 0.0, 20.0, 0.0],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [10, 20, 30]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "SCOTLAND"],
                "household_weight": [2.0, 3.0],
            }
        ),
        time_period="2023",
    )
    count_resource = {
        "cells": {
            "LONDON": {"A": {"CENTRAL_LONDON": 3, "OUTER_LONDON": 1}},
            "SCOTLAND": {"A": {"LOTHIAN": 2}},
        }
    }

    receipt = tool.e4_identity_receipt(
        frame,
        contract=Contract(),
        count_resource=count_resource,
        lha_category=["A", "A", "A"],
        permutation_seed=7,
    )

    assert receipt["identical_under_permutation"] is True
    assert receipt["permutation_mismatches"] == {}
    # The synthetic frame carries no stored E4 columns; the receipt says so
    # explicitly instead of silently passing the stored comparison.
    assert receipt["matches_stored_columns"] is False
    assert set(receipt["stored_columns_missing"]) == {
        "person",
        "benunit",
        "household",
    }


def test_brma_cell_distribution_masks_small_counts_and_reports_z() -> None:
    tool = _load_tool("emit_uk_brma_distribution")
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20, 30, 40],
            "region": ["LONDON"] * 4,
            "LHA_category": ["A"] * 4,
            "brma": ["CENTRAL", "CENTRAL", "CENTRAL", "OUTER"],
        }
    )
    resource = {"cells": {"LONDON": {"A": {"CENTRAL": 1, "OUTER": 1}}}}

    payload = tool.brma_cell_distribution(
        benunit,
        count_resource=resource,
        minimum_count=3,
    )

    rows = {row["brma"]: row for row in payload["rows"]}
    assert rows["CENTRAL"]["built_count"] == 3
    assert rows["CENTRAL"]["cell_n"] == 4
    assert rows["CENTRAL"]["z"] is not None
    assert rows["OUTER"]["built_count"] == "<3"
    assert rows["OUTER"]["built_share"] is None
    assert rows["OUTER"]["z"] is None
    assert payload["max_abs_z"] > 0


def test_brma_cell_distribution_fails_closed_on_missing_cell() -> None:
    tool = _load_tool("emit_uk_brma_distribution")
    benunit = pd.DataFrame(
        {
            "benunit_id": [10],
            "region": ["LONDON"],
            "LHA_category": ["B"],
            "brma": ["CENTRAL"],
        }
    )
    resource = {"cells": {"LONDON": {"A": {"CENTRAL": 1}}}}

    with pytest.raises(KeyError, match="LHA_category"):
        tool.brma_cell_distribution(benunit, count_resource=resource)


def _frame():
    person = pd.DataFrame(
        {
            "person_id": [101, 102, 103],
            "person_benunit_id": [201, 202, 203],
            "person_household_id": [1, 2, 3],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [201, 202, 203]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [1.0, 1.0, 1.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )
