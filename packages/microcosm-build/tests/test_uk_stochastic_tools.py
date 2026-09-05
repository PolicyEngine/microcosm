from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract

ROOT = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_childcare_draws_are_identity_keyed_and_hold_hours_fixed(monkeypatch) -> None:
    tool = _load_tool("fit_uk_childcare_takeup")

    def forbidden_seed(*args, **kwargs):
        raise AssertionError("np.random.seed must not be used")

    monkeypatch.setattr(tool.np.random, "seed", forbidden_seed)
    contract = load_uk_take_up_contract()
    ids = np.array([5, 4, 3, 2, 1])
    first = tool.draw_childcare_inputs(
        ids, tool.INITIAL_PARAMS, seed=42, contract=contract
    )
    second = tool.draw_childcare_inputs(
        ids, tool.INITIAL_PARAMS, seed=42, contract=contract
    )

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_childcare_fit_receipt_is_deterministic_with_seeded_runner(
    tmp_path: Path,
) -> None:
    tool = _load_tool("fit_uk_childcare_takeup")
    input_h5 = tmp_path / "fixture.h5"
    input_h5.write_bytes(b"synthetic")

    targets = {name: float(index + 1) for index, name in enumerate(tool.TARGET_IDS)}

    def runner(path: Path, params: np.ndarray, seed: int, period: int, contract):
        scale = float(params.sum()) + seed * 0.0 + len(path.read_bytes()) * 0.0
        assert period == 2024
        assert contract.resource_sha256
        return {name: value + scale * 0.0 for name, value in targets.items()}

    first = tool.fit_childcare_takeup(
        input_h5,
        seed=42,
        maxiter=1,
        runner=runner,
        generated_at="2026-08-17",
        targets=targets,
    )
    second = tool.fit_childcare_takeup(
        input_h5,
        seed=42,
        maxiter=1,
        runner=runner,
        generated_at="2026-08-17",
        targets=targets,
    )

    assert first == second
    assert first["input_sha256"]
    assert first["seed"] == 42
    assert set(first["params"]) == set(tool.RATE_KEYS)
    assert len(first["params"]) == 4
    assert set(first["achieved"]) == set(tool.TARGET_IDS)


def test_childcare_expected_counts_are_bilinear_in_the_extended_rate() -> None:
    tool = _load_tool("fit_uk_childcare_takeup")
    # Two rows: the first family is extended-eligible (loses targeted and
    # universal when it claims extended), the second is not.
    with_ext = {
        "hmrc.tfc.government_top_up": np.array([100.0, 50.0]),
        "hmrc.tfc.children_with_used_accounts": np.array([1.0, 1.0]),
        "dfe.funded_childcare.working_parent_children_2_to_4": np.array([1.0, 0.0]),
        "dfe.funded_childcare.early_learning_2_year_olds": np.array([0.0, 1.0]),
        "dfe.funded_childcare.universal_only_children": np.array([0.0, 1.0]),
    }
    without_ext = {
        **with_ext,
        "dfe.funded_childcare.early_learning_2_year_olds": np.array([1.0, 1.0]),
        "dfe.funded_childcare.universal_only_children": np.array([1.0, 1.0]),
    }
    basis = tool.ChildcareExpectationBasis(
        with_extended=with_ext, without_extended=without_ext, rows=2
    )
    # RATE_KEYS order: tfc, extended, targeted, universal
    expected = tool.expected_childcare_counts(basis, np.array([0.5, 0.25, 1.0, 0.4]))

    assert expected["hmrc.tfc.government_top_up"] == pytest.approx(75.0)
    assert expected["hmrc.tfc.children_with_used_accounts"] == pytest.approx(1.0)
    assert expected["dfe.funded_childcare.working_parent_children_2_to_4"] == (
        pytest.approx(0.25)
    )
    # targeted: family 1 is eligible only when it does not claim extended.
    assert expected["dfe.funded_childcare.early_learning_2_year_olds"] == (
        pytest.approx(1.0 * (0.75 + 1.0))
    )
    assert expected["dfe.funded_childcare.universal_only_children"] == (
        pytest.approx(0.4 * (0.75 + 1.0))
    )
    ceiling = tool.expected_childcare_counts(basis, np.ones(4))
    assert ceiling["dfe.funded_childcare.early_learning_2_year_olds"] == 1.0


def test_childcare_fitter_family_allow_list_excludes_entitlement_spending() -> None:
    tool = _load_tool("fit_uk_childcare_takeup")

    assert tool.TARGET_IDS == (
        "hmrc.tfc.government_top_up",
        "hmrc.tfc.children_with_used_accounts",
        "dfe.funded_childcare.working_parent_children_2_to_4",
        "dfe.funded_childcare.early_learning_2_year_olds",
        "dfe.funded_childcare.universal_only_children",
    )
    assert not any("spending" in target_id for target_id in tool.TARGET_IDS[1:])


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
        build_year = 2024

        def rate(self, key: str, build_year: int | None = None) -> float:
            if key == "tax_free_childcare_spend_routed_share":
                return 0.593
            return 0.5 if not key.startswith("scp") else 0.9

        def entry(self, key: str):
            assert key == "tax_free_childcare_spend_routed_share"
            return SimpleNamespace(raw={"entity": "person"})

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
