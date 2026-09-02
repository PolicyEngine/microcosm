"""Population patching, weight-lineage, and structural-version contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import (
    Node,
    Owned,
    Ownership,
    Slice,
    StructuralDelta,
    WeightTransition,
)
from microcosm.graph.kernel import KernelResult
from microcosm.graph.population import (
    Population,
    PopulationError,
    dtype_for_token,
    dtype_matches,
    owned_ids,
    patch,
    storage_equal,
    token_for_dtype,
    weight_cap_receipt,
)


def _frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3, 4], dtype=np.int64),
            "person_household_id": np.array([10, 10, 20, 30], dtype=np.int64),
            "keep": np.array([True, True, True, False], dtype=np.bool_),
            "owned": pd.Series([False, True, False, True], dtype="boolean"),
            "nullable": pd.Series([True, False, pd.NA, True], dtype="boolean"),
            "amount": np.array([-0.0, 1.0, -0.0, 2.0], dtype=np.float64),
        }
    )
    household = pd.DataFrame({"household_id": np.array([10, 20, 30], dtype=np.int64)})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b", "b"], name="stratum"),
    )


def _population() -> Population:
    return Population.from_frame(_frame(), "source")


def _replace_person_table(
    frame: Frame, person: pd.DataFrame, strata: pd.Series
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables[frame.schema.person_entity] = person
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(
        tables,
        frame.schema,
        weights,
        strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _mass_receipt(
    *,
    policy: str,
    before: float,
    after: float,
    stratum_before: dict[str, float],
    stratum_after: dict[str, float],
) -> dict[str, object]:
    return {
        "mass": {
            "policy": policy,
            "before": before,
            "after": after,
            "stratum_before": stratum_before,
            "stratum_after": stratum_after,
        }
    }


def test_dtype_helpers_distinguish_dense_and_nullable_types() -> None:
    assert dtype_for_token("boolean") == pd.BooleanDtype()
    assert dtype_for_token("bool") == np.dtype("bool")
    assert token_for_dtype(pd.Int64Dtype()) == "Int64"
    assert token_for_dtype(pd.StringDtype(storage="python")) == "string"
    assert dtype_matches(pd.Series([True, pd.NA], dtype="boolean"), "boolean")
    assert not dtype_matches(pd.Series([True, False], dtype=bool), "boolean")
    with pytest.raises(PopulationError, match="Unknown graph dtype"):
        dtype_for_token("category")


def test_population_from_frame_freezes_total_ownership_and_weight_kinds() -> None:
    population = _population()
    assert population.owners[("person", "amount")] == "source"
    assert population.weight_kind == {"household": WeightKind.DESIGN}
    with pytest.raises(TypeError):
        population.owners[("person", "amount")] = "other"  # type: ignore[index]


def test_owned_ids_follows_a_nullable_boolean_mask() -> None:
    population = _population()
    owned = Owned("person", "amount", "float64", rows="owned")
    assert owned_ids(population, owned).tolist() == [2, 4]
    with pytest.raises(PopulationError, match="contains nulls"):
        owned_ids(
            population,
            Owned("person", "amount", "float64", rows="nullable"),
        )


def test_masked_patch_preserves_nonowned_float_bits_including_negative_zero() -> None:
    population = _population()
    node = Node(
        "replace_amount",
        "test@1",
        inputs=(Slice("person", ("owned",)),),
        outputs=(Owned("person", "amount", "float64", rows="owned"),),
    )
    result = KernelResult(
        columns={
            ("person", "amount"): pd.Series(
                [7.0, 8.0], index=pd.Index([2, 4]), dtype="float64"
            )
        }
    )

    updated = patch(population, node, result)

    values = updated.frame.table("person")["amount"].to_numpy()
    assert values.tolist() == [-0.0, 7.0, -0.0, 8.0]
    assert np.signbit(values[[0, 2]]).all()
    assert storage_equal(
        population.frame.table("person")["amount"],
        updated.frame.table("person")["amount"],
        np.array([True, False, True, False]),
    )
    assert updated.owners[("person", "amount")] == "replace_amount"


def test_nullable_boolean_declaration_rejects_dense_bool_output() -> None:
    population = _population()
    node = Node(
        "replace_flag",
        "test@1",
        inputs=(Slice("person", ("owned",)),),
        outputs=(Owned("person", "nullable", "boolean", rows="owned"),),
    )
    result = KernelResult(
        columns={
            ("person", "nullable"): pd.Series([True, False], index=[2, 4], dtype=bool)
        }
    )
    with pytest.raises(PopulationError, match="requires 'boolean'"):
        patch(population, node, result)


def test_absent_ownership_requires_null_and_patches_only_owned_rows() -> None:
    population = _population()
    node = Node(
        "remove_flag",
        "test@1",
        inputs=(Slice("person", ("owned",)),),
        outputs=(
            Owned(
                "person",
                "nullable",
                "boolean",
                rows="owned",
                ownership=Ownership.ABSENT,
            ),
        ),
    )
    nonnull = KernelResult(
        columns={
            ("person", "nullable"): pd.Series(
                [True, pd.NA], index=[2, 4], dtype="boolean"
            )
        }
    )
    with pytest.raises(PopulationError, match="ABSENT"):
        patch(population, node, nonnull)

    absent = KernelResult(
        columns={
            ("person", "nullable"): pd.Series(
                [pd.NA, pd.NA], index=[2, 4], dtype="boolean"
            )
        }
    )
    updated = patch(population, node, absent)
    assert updated.frame.table("person")["nullable"].tolist() == [
        True,
        pd.NA,
        pd.NA,
        pd.NA,
    ]


def test_new_masked_dense_column_is_rejected_as_unrepresentable() -> None:
    population = _population()
    node = Node(
        "new_dense",
        "test@1",
        inputs=(Slice("person", ("owned",)),),
        outputs=(Owned("person", "dense", "bool", rows="owned"),),
    )
    result = KernelResult(
        columns={
            ("person", "dense"): pd.Series([True, False], index=[2, 4], dtype=bool)
        }
    )
    with pytest.raises(PopulationError, match="cannot be null outside"):
        patch(population, node, result)


def test_new_masked_float_column_uses_nan_for_unowned_rows() -> None:
    population = _population()
    node = Node(
        "new_float",
        "test@1",
        inputs=(Slice("person", ("owned",)),),
        outputs=(Owned("person", "dense_float", "float64", rows="owned"),),
    )
    result = KernelResult(
        columns={
            ("person", "dense_float"): pd.Series(
                [7.0, 8.0], index=[2, 4], dtype="float64"
            )
        }
    )

    updated = patch(population, node, result)

    values = updated.frame.table("person")["dense_float"]
    assert values.dtype == np.dtype("float64")
    assert values.iloc[[1, 3]].tolist() == [7.0, 8.0]
    assert values.iloc[[0, 2]].isna().all()


def test_weight_transition_is_immediate_explicit_and_records_mass() -> None:
    population = _population()
    node = Node(
        "importance",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    result = KernelResult(
        weights=Weights(np.array([2.0, 4.0, 6.0]), WeightKind.IMPORTANCE),
        receipt=_mass_receipt(
            policy="free",
            before=7.0,
            after=14.0,
            stratum_before={"a": 2.0, "b": 5.0},
            stratum_after={"a": 4.0, "b": 10.0},
        ),
    )

    updated = patch(population, node, result)

    assert updated.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    assert updated.weight_kind == {"household": WeightKind.IMPORTANCE}
    assert len(updated.mass_ledger) == 1
    record = updated.mass_ledger[0]
    assert (record.before_total, record.after_total) == (7.0, 14.0)
    assert dict(record.after_by_stratum) == {"a": 4.0, "b": 10.0}


def test_weight_transition_rejects_skips_and_inherited_weights() -> None:
    population = _population()
    # Forward moves are legal, including design straight to calibrated (the
    # Frame kernel's rule; interface amendment 6); backward moves are not.
    straight = Node(
        "straight",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
    )
    calibrated = patch(
        population,
        straight,
        KernelResult(
            weights=Weights(np.array([2.0, 4.0, 6.0]), WeightKind.CALIBRATED),
            receipt=_mass_receipt(
                policy="free",
                before=7.0,
                after=14.0,
                stratum_before={"a": 2.0, "b": 5.0},
                stratum_after={"a": 4.0, "b": 10.0},
            ),
        ),
    )
    assert calibrated.frame.weights_for("household").kind is WeightKind.CALIBRATED

    backward = Node(
        "backward",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="straight",
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    with pytest.raises(PopulationError, match="must move forward"):
        patch(
            calibrated,
            backward,
            KernelResult(
                weights=Weights(np.array([1.0, 2.0, 3.0]), WeightKind.IMPORTANCE)
            ),
        )

    # A weight transition on an ordinary node is refused at declaration.
    from microcosm.graph import GraphError

    with pytest.raises(GraphError, match="REWEIGHT node with a base"):
        Node(
            "ordinary_weights",
            "test@1",
            weights=WeightTransition("household", "importance", mass="free"),
        )

    inherited = Node(
        "person_importance",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("person", "importance", mass="free"),
        mass="free",
    )
    with pytest.raises(PopulationError, match="inherited weights"):
        patch(
            population,
            inherited,
            KernelResult(
                weights=Weights(np.ones(4, dtype=np.float64), WeightKind.IMPORTANCE)
            ),
        )


def test_conserve_checks_each_stratum_not_only_total() -> None:
    population = _population()
    node = Node(
        "importance",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="conserve"),
    )
    result = KernelResult(
        weights=Weights(np.array([2.0, 1.0, 3.0]), WeightKind.IMPORTANCE)
    )
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(population, node, result)


def test_declared_mass_validates_the_kernel_receipt() -> None:
    population = _population()
    node = Node(
        "importance",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="declared"),
        mass="declared",
    )
    result = KernelResult(
        weights=Weights(np.array([2.0, 4.0, 6.0]), WeightKind.IMPORTANCE),
        receipt=_mass_receipt(
            policy="declared",
            before=7.0,
            after=999.0,
            stratum_before={"a": 2.0, "b": 5.0},
            stratum_after={"a": 4.0, "b": 10.0},
        ),
    )
    with pytest.raises(PopulationError, match="computed value"):
        patch(population, node, result)


def test_filter_requires_subset_ids_and_records_free_mass() -> None:
    population = _population()
    filtered = population.frame.select(
        np.array([True, True, True, False], dtype=np.bool_)
    )
    node = Node(
        "filter",
        "test@1",
        inputs=(Slice("person", ("keep",)),),
        structural=StructuralDelta.FILTER,
        base="source",
        mass="free",
    )
    updated = patch(population, node, KernelResult(frame=filtered))
    assert updated.version == "filter"
    assert updated.frame.table("person")["person_id"].tolist() == [1, 2, 3]
    assert updated.mass_ledger[-1].operation == "filter"
    assert set(updated.owners.values()) == {"filter"}

    conserve = Node(
        "filter_conserve",
        "test@1",
        inputs=(Slice("person", ("keep",)),),
        structural=StructuralDelta.FILTER,
        base="source",
    )
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(population, conserve, KernelResult(frame=filtered))


def test_expand_must_retain_every_original_id() -> None:
    population = _population()
    filtered = population.frame.select(
        np.array([True, True, True, False], dtype=np.bool_)
    )
    node = Node(
        "expand",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        mass="free",
    )
    with pytest.raises(PopulationError, match="dropped original"):
        patch(population, node, KernelResult(frame=filtered))


def test_structural_nodes_cannot_rewrite_carried_cell_storage() -> None:
    population = _population()

    filtered = population.frame.select(
        np.array([True, True, True, False], dtype=np.bool_)
    )
    filtered_person = filtered.table("person").copy()
    filtered_person.loc[0, "amount"] = 99.0
    changed_filter = _replace_person_table(
        filtered, filtered_person, filtered.strata.copy()
    )
    filter_node = Node(
        "filter",
        "test@1",
        structural=StructuralDelta.FILTER,
        base="source",
        mass="free",
    )
    with pytest.raises(PopulationError, match="changed carried storage"):
        patch(population, filter_node, KernelResult(frame=changed_filter))

    before_person = population.frame.table("person")
    added = before_person.iloc[[0]].copy()
    added["person_id"] = np.asarray([5], dtype=np.int64)
    expanded_person = pd.concat([before_person, added], ignore_index=True)
    expanded_person.loc[0, "amount"] = 99.0
    expanded = _replace_person_table(
        population.frame,
        expanded_person,
        pd.concat(
            [population.frame.strata, pd.Series(["a"], name="stratum")],
            ignore_index=True,
        ),
    )
    expand_node = Node(
        "expand",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        mass="free",
    )
    with pytest.raises(PopulationError, match="changed carried storage"):
        patch(population, expand_node, KernelResult(frame=expanded))

    changed_person = before_person.copy()
    changed_person.loc[0, "amount"] = 99.0
    changed_reweight = _replace_person_table(
        population.frame, changed_person, population.frame.strata.copy()
    )
    reweight_node = Node(
        "reweight",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    with pytest.raises(PopulationError, match="changed carried storage"):
        patch(population, reweight_node, KernelResult(frame=changed_reweight))


def test_structural_nodes_cannot_smuggle_explicit_or_rewritten_weights() -> None:
    population = _population()
    frame = population.frame
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    expand = Node(
        "expand",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        mass="free",
    )

    added_explicit = Frame(
        tables,
        frame.schema,
        {
            "household": frame.weights_for("household"),
            "person": Weights(np.ones(frame.n("person")), WeightKind.DESIGN),
        },
        frame.strata,
    )
    with pytest.raises(PopulationError, match="explicit weighted entities"):
        patch(population, expand, KernelResult(frame=added_explicit))

    rewritten = Frame(
        tables,
        frame.schema,
        {"household": Weights(np.array([9.0, 8.0, 7.0]), WeightKind.DESIGN)},
        frame.strata,
    )
    with pytest.raises(PopulationError, match="changed carried weights"):
        patch(population, expand, KernelResult(frame=rewritten))


def test_calibration_cap_stays_anchored_to_original_design_after_filter() -> None:
    population = _population()
    importance_node = Node(
        "pool",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    importance = patch(
        population,
        importance_node,
        KernelResult(weights=Weights(np.array([2.0, 4.0, 6.0]), WeightKind.IMPORTANCE)),
    )
    filtered_frame = importance.frame.select(
        np.array([True, True, True, False], dtype=np.bool_)
    )
    filter_node = Node(
        "adults",
        "test@1",
        structural=StructuralDelta.FILTER,
        base="pool",
        mass="free",
    )
    filtered = patch(importance, filter_node, KernelResult(frame=filtered_frame))

    np.testing.assert_array_equal(
        filtered.frame.table("household")["household_id"], np.array([10, 20])
    )
    np.testing.assert_array_equal(
        filtered.design_weights["household"], np.array([1.0, 2.0])
    )
    assert not filtered.design_weights["household"].flags.writeable

    calibrated_node = Node(
        "calibrated",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="adults",
        params={"max_weight_ratio": 2.0, "weight_anchor": "design"},
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
    )
    within = patch(
        filtered,
        calibrated_node,
        KernelResult(
            weights=Weights(np.array([1.5, 3.0]), WeightKind.CALIBRATED),
            receipt={"weight_anchor": "incoming"},
        ),
    )
    assert weight_cap_receipt(within, calibrated_node) == {
        "weight_anchor": "design",
        "max_weight_ratio": 2.0,
        "realized_max_weight_ratio": 1.5,
    }

    with pytest.raises(PopulationError, match="calibrated.*original design"):
        patch(
            filtered,
            calibrated_node,
            KernelResult(weights=Weights(np.array([3.0, 6.0]), WeightKind.CALIBRATED)),
        )


def test_design_cap_fails_closed_when_source_has_no_design_lineage() -> None:
    frame = _frame()
    importance_frame = Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                frame.weights_for("household").values, WeightKind.IMPORTANCE
            )
        },
        frame.strata,
    )
    population = Population.from_frame(importance_frame, "importance_source")
    node = Node(
        "calibrated",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="importance_source",
        params={"max_weight_ratio": 2.0, "weight_anchor": "design"},
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
    )
    with pytest.raises(PopulationError, match="no original design-weight anchor"):
        patch(
            population,
            node,
            KernelResult(
                weights=Weights(np.array([1.0, 2.0, 3.0]), WeightKind.CALIBRATED)
            ),
        )


def test_reweight_can_synthesize_frame_but_must_not_change_ids() -> None:
    population = _population()
    node = Node(
        "reweight",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    result = KernelResult(
        weights=Weights(np.array([2.0, 4.0, 6.0]), WeightKind.IMPORTANCE),
        receipt=_mass_receipt(
            policy="free",
            before=7.0,
            after=14.0,
            stratum_before={"a": 2.0, "b": 5.0},
            stratum_after={"a": 4.0, "b": 10.0},
        ),
    )
    updated = patch(population, node, result)
    assert updated.version == "reweight"
    assert updated.frame.table("person")["person_id"].tolist() == [1, 2, 3, 4]

    tables = {
        entity: population.frame.table(entity).copy()
        for entity in population.frame.entities
    }
    tables["person"] = tables["person"].iloc[::-1]
    reordered = Frame(
        tables,
        population.frame.schema,
        {"household": population.frame.weights_for("household")},
        population.frame.strata.iloc[::-1],
    )
    # A REWEIGHT node without a transition is refused at declaration
    # (interface amendment 6), before any population is involved.
    from microcosm.graph import GraphError

    with pytest.raises(GraphError, match="declares its WeightTransition"):
        Node(
            "bad_reweight",
            "test@1",
            structural=StructuralDelta.REWEIGHT,
            base="source",
            mass="free",
        )
    with pytest.raises(PopulationError, match="changed 'person' ids"):
        patch(population, node, KernelResult(frame=reordered))
