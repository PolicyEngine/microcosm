from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

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


def test_brma_distribution_masks_small_counts() -> None:
    tool = _load_tool("emit_uk_brma_distribution")
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["LONDON", "LONDON", "LONDON", "LONDON"],
            "brma": ["A", "A", "A", "B"],
        }
    )
    resource = {"cells": {"LONDON": {"A": {"A": 3, "B": 1}}}}

    payload = tool.brma_distribution(
        household,
        count_resource=resource,
        minimum_count=3,
    )

    rows = {row["brma"]: row for row in payload["rows"]}
    assert rows["A"]["built_count"] == 3
    assert rows["B"]["built_count"] == "<3"
    assert rows["B"]["built_share"] is None


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
