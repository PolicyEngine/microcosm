"""UK spine graph declarations, kernels, and structural runtime contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import KernelResult, Node, StructuralDelta
from microcosm.graph.population import Population, PopulationError, patch


def _expand_population() -> Population:
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": pd.Series([1, 2], dtype="int64"),
                    "person_benunit_id": pd.Series([100, 200], dtype="int64"),
                    "person_household_id": pd.Series([10, 20], dtype="int64"),
                    "hidden_payload": pd.Series([1.25, 9.5], dtype="float64"),
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": pd.Series([100, 200], dtype="int64"),
                    "capital": pd.Series([4.0, 7.0], dtype="float64"),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": pd.Series([10, 20], dtype="int64"),
                    "region": pd.Series(["LONDON", "WALES"], dtype="string"),
                }
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {
            "household": Weights(
                np.array([1.0, 2.0], dtype=np.float64), WeightKind.DESIGN
            )
        },
        pd.Series(["base", "base"], dtype="string", name="stratum"),
        metadata={"time_period": "2024"},
    )
    return Population.from_frame(frame, "root")


def _expand_node() -> Node:
    return Node(
        id="clone",
        kernel="uk.stage.expand.test@1",
        structural=StructuralDelta.EXPAND,
        base="root",
        params={
            "expand_cells": (("household", "is_clone", "bool"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="conserve",
    )


def _expand_result(*, bad_source: bool = False) -> KernelResult:
    return KernelResult(
        columns={
            ("person", "person_id"): pd.Series(
                [1, 2, 99 if bad_source else 1],
                index=pd.Index([1, 2, 3], name="person_id"),
                dtype="int64",
            ),
            ("benunit", "benunit_id"): pd.Series(
                [100, 200, 100],
                index=pd.Index([100, 200, 300], name="benunit_id"),
                dtype="int64",
            ),
            ("household", "household_id"): pd.Series(
                [10, 20, 10],
                index=pd.Index([10, 20, 30], name="household_id"),
                dtype="int64",
            ),
            ("person", "person_benunit_id"): pd.Series(
                [100, 200, 300],
                index=pd.Index([1, 2, 3], name="person_id"),
                dtype="int64",
            ),
            ("person", "person_household_id"): pd.Series(
                [10, 20, 30],
                index=pd.Index([1, 2, 3], name="person_id"),
                dtype="int64",
            ),
            ("household", "is_clone"): pd.Series(
                [False, False, True],
                index=pd.Index([10, 20, 30], name="household_id"),
                dtype="bool",
            ),
        },
        weights=Weights(
            np.array([0.5, 2.0, 0.5], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
        receipt={
            "frame_mass_log_append": [
                {
                    "entity": "household",
                    "old_total": 3.0,
                    "new_total": 3.0,
                    "declared_factor": None,
                    "reason": "test clone mass is conserved",
                }
            ]
        },
    )


def test_uk_expand_contract_carries_cells_links_weights_and_design_lineage() -> None:
    expanded = patch(_expand_population(), _expand_node(), _expand_result())

    person = expanded.frame.table("person")
    assert person["person_id"].tolist() == [1, 2, 3]
    assert person["person_household_id"].tolist() == [10, 20, 30]
    assert person["hidden_payload"].tolist() == [1.25, 9.5, 1.25]
    assert expanded.frame.table("household")["is_clone"].tolist() == [
        False,
        False,
        True,
    ]
    assert expanded.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(
        expanded.frame.weights_for("household").values,
        np.array([0.5, 2.0, 0.5]),
    )
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0, 1.0])
    )
    assert expanded.frame.mass_log[-1].reason == "test clone mass is conserved"
    assert expanded.mass_ledger[-1].operation == "expand"


def test_uk_expand_contract_rejects_unknown_source_ids() -> None:
    with pytest.raises(PopulationError, match="unknown 'person' source ids"):
        patch(_expand_population(), _expand_node(), _expand_result(bad_source=True))
