"""UK spine graph declarations, kernels, and structural runtime contracts."""

# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import (
    UK_SPINE_EXCLUSIONS,
    UK_SPINE_STRUCTURAL_STAGES,
    uk_registry,
    uk_spine_graph,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Graph,
    KernelResult,
    Node,
    StructuralDelta,
    compile_graph,
    graph_from_json,
    graph_to_json,
)
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
        expand={
            "person": pd.Series(
                [99 if bad_source else 1],
                index=pd.Index([3], name="person_id"),
                dtype="int64",
            ),
            "benunit": pd.Series(
                [100],
                index=pd.Index([300], name="benunit_id"),
                dtype="int64",
            ),
            "household": pd.Series(
                [10],
                index=pd.Index([30], name="household_id"),
                dtype="int64",
            ),
        },
        columns={
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


def _mixed_size_population() -> Population:
    """Two households of different size: one person in 10, two in 20."""

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": pd.Series([1, 2, 3], dtype="int64"),
                    "person_benunit_id": pd.Series([100, 200, 200], dtype="int64"),
                    "person_household_id": pd.Series([10, 20, 20], dtype="int64"),
                }
            ),
            "benunit": pd.DataFrame(
                {"benunit_id": pd.Series([100, 200], dtype="int64")}
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
        pd.Series(["base", "base", "base"], dtype="string", name="stratum"),
        metadata={"time_period": "2024"},
    )
    return Population.from_frame(frame, "root")


def _mass_shifting_expand_result(*, declared: bool) -> KernelResult:
    """Clone the one-person household and move mass onto it from the larger one.

    Household mass is conserved (1 + 2 == 0.5 + 1.5 + 1.0) while person mass is
    not (1 + 2*2 = 5 against 0.5 + 1.5*2 + 1.0 = 4.5): the shape of the SPI
    support channel, whose prior-mass allocation moves half the household mass
    onto stacked households whose composition differs from the FRS households
    it is taken from.
    """

    receipt: dict[str, object] = {
        "frame_mass_log_append": [
            {
                "entity": "household",
                "old_total": 3.0,
                "new_total": 3.0,
                "declared_factor": None,
                "reason": "test stack conserves household mass, not person mass",
            }
        ]
    }
    if declared:
        receipt["mass"] = {
            "policy": "declared",
            "before": 5.0,
            "after": 4.5,
            "stratum_before": {"base": 5.0},
            "stratum_after": {"base": 4.5},
        }
    return KernelResult(
        expand={
            "person": pd.Series(
                [1], index=pd.Index([4], name="person_id"), dtype="int64"
            ),
            "benunit": pd.Series(
                [100], index=pd.Index([300], name="benunit_id"), dtype="int64"
            ),
            "household": pd.Series(
                [10], index=pd.Index([30], name="household_id"), dtype="int64"
            ),
        },
        columns={
            ("household", "is_clone"): pd.Series(
                [False, False, True],
                index=pd.Index([10, 20, 30], name="household_id"),
                dtype="bool",
            ),
        },
        weights=Weights(
            np.array([0.5, 1.5, 1.0], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
        receipt=receipt,
    )


def _zero_row_expand_node(node_id: str = "anchor") -> Node:
    return Node(
        id=node_id,
        kernel="uk.stage.expand.test@1",
        structural=StructuralDelta.EXPAND,
        base="root",
        params={
            "expand_cells": (),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="conserve",
    )


def _zero_row_expand_result(weights: list[float], *, total: float) -> KernelResult:
    def empty(id_column: str) -> pd.Series:
        return pd.Series(
            [],
            index=pd.Index([], name=id_column, dtype="int64"),
            dtype="int64",
            name=id_column,
        )

    return KernelResult(
        expand={
            "person": empty("person_id"),
            "benunit": empty("benunit_id"),
            "household": empty("household_id"),
        },
        columns={},
        weights=Weights(np.asarray(weights, dtype=np.float64), WeightKind.IMPORTANCE),
        receipt={
            "frame_mass_log_append": [
                {
                    "entity": "household",
                    "old_total": total,
                    "new_total": total,
                    "declared_factor": 1.0,
                    "reason": "test anchor moves mass between paired households",
                }
            ]
        },
    )


__all__ = [name for name in globals() if not name.startswith("__")]
