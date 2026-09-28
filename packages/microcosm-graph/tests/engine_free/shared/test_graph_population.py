"""Population patching, weight-lineage, and structural-version contracts."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

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
    _storage_parts,
    dtype_for_token,
    dtype_matches,
    entrant_strata_receipt,
    expand_lineage_receipt,
    owned_ids,
    patch,
    restore_cached_expand,
    storage_equal,
    token_for_dtype,
    weight_cap_receipt,
)
from microcosm.graph.store import ContentStore, _encode_object_scalar


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


def test_mass_receipt_rejects_partition_when_graph_has_none() -> None:
    population = _population()
    node = Node(
        "importance",
        "test@1",
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="declared"),
        mass="declared",
    )
    receipt = _mass_receipt(
        policy="declared",
        before=7.0,
        after=14.0,
        stratum_before={"a": 2.0, "b": 5.0},
        stratum_after={"a": 4.0, "b": 10.0},
    )
    mass = receipt["mass"]
    assert isinstance(mass, dict)
    mass["partition"] = {}
    result = KernelResult(
        weights=Weights(np.array([2.0, 4.0, 6.0]), WeightKind.IMPORTANCE),
        receipt=receipt,
    )

    with pytest.raises(PopulationError, match="declares no mass partition"):
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


def test_filter_conserve_allows_removed_zero_mass_partition_support() -> None:
    base = _frame()
    person = base.table("person").copy()
    person["period"] = pd.Series(["zero", "zero", "kept", "kept"], dtype="string")
    frame = Frame(
        {"person": person, "household": base.table("household").copy()},
        base.schema,
        {
            "household": Weights(
                np.array([0.0, 2.0, 3.0], dtype=np.float64), WeightKind.DESIGN
            )
        },
        base.strata,
    )
    population = Population.from_frame(frame, "source")
    filtered = frame.select(np.array([False, False, True, True], dtype=np.bool_))
    node = Node(
        "filter_zero_support",
        "test@1",
        structural=StructuralDelta.FILTER,
        base="source",
        mass="conserve",
    )

    updated = patch(
        population,
        node,
        KernelResult(frame=filtered),
        mass_partition=("person", "period"),
    )

    record = updated.mass_ledger[-1]
    assert dict(record.before_by_stratum) == {"a": 0.0, "b": 5.0}
    assert dict(record.after_by_stratum) == {"b": 5.0}
    assert {
        partition: dict(strata)
        for partition, strata in record.before_partitions.items()
    } == {"kept": {"b": 5.0}, "zero": {"a": 0.0}}
    assert {
        partition: dict(strata) for partition, strata in record.after_partitions.items()
    } == {"kept": {"b": 5.0}}


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


def _lineage_expand_node() -> Node:
    return Node(
        "lineage_expand",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        params={
            "expand_cells": (),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="conserve",
    )


def _lineage_expand_result(*, bad_source: bool = False) -> KernelResult:
    return KernelResult(
        expand={
            "person": pd.Series(
                [999 if bad_source else 1, 2],
                index=pd.Index([5, 6], name="person_id"),
                dtype="int64",
            ),
            "household": pd.Series(
                [10],
                index=pd.Index([40], name="household_id"),
                dtype="int64",
            ),
        },
        weights=Weights(
            np.array([0.5, 2.0, 3.0, 0.5], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
    )


def _membership_overlay_result(target: int) -> KernelResult:
    return KernelResult(
        expand={
            "person": pd.Series(
                [1, 2, 3],
                index=pd.Index([5, 6, 7], name="person_id"),
                dtype="int64",
            ),
            "household": pd.Series(
                [10, 20],
                index=pd.Index([40, 50], name="household_id"),
                dtype="int64",
            ),
        },
        columns={
            ("person", "person_household_id"): pd.Series(
                [10, 10, 20, 30, target, 40, 50],
                index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="person_id"),
                dtype="int64",
            )
        },
        weights=Weights(
            np.array([0.5, 1.0, 3.0, 0.5, 1.0], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
    )


@pytest.mark.parametrize("cached", [False, True], ids=("cold", "cached"))
@pytest.mark.parametrize("target", [20, 50], ids=("incumbent", "copied-group"))
def test_expand_rejects_repointed_copied_membership(target: int, cached: bool) -> None:
    population = _population()
    node = Node(
        "membership_overlay",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        params={
            "expand_cells": (("person", "person_household_id", "int64"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="free",
    )
    result = _membership_overlay_result(target)
    if cached:
        legal = _membership_overlay_result(40)
        expanded = patch(population, node, legal)
        person = expanded.frame.table("person").copy()
        person.loc[person["person_id"] == 5, "person_household_id"] = target
        result = KernelResult(
            frame=_replace_person_table(expanded.frame, person, expanded.frame.strata),
            weights=legal.weights,
            receipt={"expand": expand_lineage_receipt(legal.expand)},
        )

    with pytest.raises(PopulationError) as error:
        if cached:
            restore_cached_expand(population, node, result)
        else:
            patch(population, node, result)

    message = str(error.value)
    assert "membership_overlay" in message
    assert "person.person_household_id" in message
    assert "person id 5" in message
    assert f"household id {target}" in message


def _entrant_to_copied_group_result(entrant_group: int) -> KernelResult:
    frame = _frame()
    person = frame.table("person")
    person_ids = pd.Index([1, 2, 3, 4, 5, 6, 7], name="person_id", dtype="int64")
    additions = {
        "person_household_id": (40, 40, entrant_group),
        "keep": (True, True, True),
        "owned": (False, True, False),
        "nullable": (True, False, pd.NA),
        "amount": (-0.0, 1.0, 3.0),
    }
    tokens = {column: token_for_dtype(person[column].dtype) for column in additions}
    return KernelResult(
        expand={
            "person": pd.Series(
                [1, 2, pd.NA],
                index=pd.Index([5, 6, 7], name="person_id", dtype="int64"),
                dtype="Int64",
            ),
            "household": pd.Series(
                [10],
                index=pd.Index([40], name="household_id", dtype="int64"),
                dtype="int64",
            ),
        },
        columns={
            ("person", column): pd.Series(
                pd.array([*person[column], *values], dtype=tokens[column]),
                index=person_ids,
            )
            for column, values in additions.items()
        },
        weights=Weights(
            np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64),
            WeightKind.DESIGN,
        ),
        strata=pd.Series(
            ["entrant"],
            index=pd.Index([7], name="person_id", dtype="int64"),
            dtype=object,
        ),
    )


@pytest.mark.parametrize("cached", [False, True], ids=("cold", "cached"))
def test_expand_rejects_entrant_membership_to_copied_group(cached: bool) -> None:
    population = _population()
    node = Node(
        "entrant_membership",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        params={
            "expand_cells": tuple(
                (
                    "person",
                    column,
                    token_for_dtype(_frame().table("person")[column].dtype),
                )
                for column in (
                    "person_household_id",
                    "keep",
                    "owned",
                    "nullable",
                    "amount",
                )
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )
    result = _entrant_to_copied_group_result(40)
    if cached:
        legal = _entrant_to_copied_group_result(10)
        expanded = patch(population, node, legal)
        person = expanded.frame.table("person").copy()
        person.loc[person["person_id"] == 7, "person_household_id"] = 40
        assert legal.expand is not None
        result = KernelResult(
            frame=_replace_person_table(expanded.frame, person, expanded.frame.strata),
            weights=legal.weights,
            receipt={
                "expand": expand_lineage_receipt(legal.expand),
                "entrant_strata": entrant_strata_receipt(
                    population.frame,
                    node,
                    legal.expand,
                    legal.strata,
                ),
            },
        )

    with pytest.raises(PopulationError) as error:
        if cached:
            restore_cached_expand(population, node, result)
        else:
            patch(population, node, result)

    message = str(error.value)
    assert "entrant_membership" in message
    assert "person.person_household_id" in message
    assert "entrant person id 7" in message
    assert "copied household id 40" in message


def _entrant_person_expand_node(*, membership_dtype: str = "int64") -> Node:
    return Node(
        "entrant_person",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        params={
            "expand_cells": (
                ("person", "person_household_id", membership_dtype),
                ("person", "keep", "bool"),
                ("person", "owned", "boolean"),
                ("person", "nullable", "boolean"),
                ("person", "amount", "float64"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )


def _entrant_person_expand_result(
    strata: object, *, frame: Frame | None = None
) -> KernelResult:
    frame = _frame() if frame is None else frame
    person = frame.table("person")
    person_id_dtype = person["person_id"].dtype
    household_id_dtype = frame.table("household")["household_id"].dtype
    entrant_id = int(person["person_id"].max()) + 1
    ids = pd.Index(
        pd.Series([*person["person_id"], entrant_id], dtype=person_id_dtype).array,
        name="person_id",
    )
    additions = {
        "person_household_id": 10,
        "keep": True,
        "owned": False,
        "nullable": pd.NA,
        "amount": 3.0,
    }
    tokens = {
        "person_household_id": token_for_dtype(person["person_household_id"].dtype),
        "keep": "bool",
        "owned": "boolean",
        "nullable": "boolean",
        "amount": "float64",
    }
    columns = {
        ("person", column): pd.Series(
            pd.array([*person[column], value], dtype=tokens[column]), index=ids
        )
        for column, value in additions.items()
    }
    return KernelResult(
        expand={
            "person": pd.Series(
                pd.array(
                    [pd.NA],
                    dtype=f"Int{np.dtype(person_id_dtype).itemsize * 8}",
                ),
                index=pd.Index(
                    pd.Series([entrant_id], dtype=person_id_dtype).array,
                    name="person_id",
                ),
            ),
            "household": pd.Series(
                [],
                index=pd.Index([], dtype=household_id_dtype, name="household_id"),
                dtype=household_id_dtype,
            ),
        },
        columns=columns,
        weights=frame.weights_for("household"),
        strata=strata,  # type: ignore[arg-type]
    )


def test_expand_lineage_carries_rows_remaps_memberships_and_restores_cache() -> None:
    population = _population()
    node = _lineage_expand_node()
    result = _lineage_expand_result()

    expanded = patch(population, node, result)

    person = expanded.frame.table("person")
    assert person["person_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert person["person_household_id"].tolist() == [10, 10, 20, 30, 40, 40]
    assert person["amount"].tolist() == [-0.0, 1.0, -0.0, 2.0, -0.0, 1.0]
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0, 3.0, 1.0])
    )
    assert expanded.mass_ledger[-1].operation == "expand"

    assert result.expand is not None
    receipt = {"expand": expand_lineage_receipt(result.expand)}
    cached = restore_cached_expand(
        population,
        node,
        KernelResult(
            frame=expanded.frame,
            weights=result.weights,
            receipt=receipt,
        ),
    )
    np.testing.assert_array_equal(
        cached.design_weights["household"], expanded.design_weights["household"]
    )
    assert cached.mass_ledger == expanded.mass_ledger

    mutated_person = expanded.frame.table("person").copy()
    mutated_person.loc[mutated_person["person_id"] == 5, "amount"] = 99.0
    mutated_frame = _replace_person_table(
        expanded.frame, mutated_person, expanded.frame.strata
    )
    with pytest.raises(
        PopulationError,
        match=r"person\.amount.*copied target/source ids.*\(5, 1\)",
    ):
        restore_cached_expand(
            population,
            node,
            KernelResult(
                frame=mutated_frame,
                weights=result.weights,
                receipt=receipt,
            ),
        )


def test_cached_expand_rejects_changed_incumbent_storage() -> None:
    population = _population()
    node = _lineage_expand_node()
    result = _lineage_expand_result()
    expanded = patch(population, node, result)
    person = expanded.frame.table("person").copy()
    person.loc[person["person_id"] == 1, "amount"] = 99.0
    mutated_frame = _replace_person_table(expanded.frame, person, expanded.frame.strata)
    assert result.expand is not None

    with pytest.raises(PopulationError, match="incumbent storage"):
        restore_cached_expand(
            population,
            node,
            KernelResult(
                frame=mutated_frame,
                weights=result.weights,
                receipt={"expand": expand_lineage_receipt(result.expand)},
            ),
        )


@pytest.mark.parametrize("mutation", ["extra", "missing"], ids=["extra", "missing"])
def test_cached_expand_requires_exact_declared_column_set(mutation: str) -> None:
    population = _population()
    node = Node(
        "column_set_expand",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        params={
            "expand_cells": (("person", "new_value", "float64"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="conserve",
    )
    lineage_result = _lineage_expand_result()
    result = KernelResult(
        expand=lineage_result.expand,
        columns={
            ("person", "new_value"): pd.Series(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                index=pd.Index([1, 2, 3, 4, 5, 6], name="person_id"),
                dtype="float64",
            )
        },
        weights=lineage_result.weights,
    )
    expanded = patch(population, node, result)
    person = expanded.frame.table("person").copy()
    if mutation == "extra":
        person["unexpected"] = np.arange(len(person), dtype=np.int64)
    else:
        person = person.drop(columns="new_value")
    mutated_frame = _replace_person_table(expanded.frame, person, expanded.frame.strata)
    assert result.expand is not None

    with pytest.raises(PopulationError, match="column set"):
        restore_cached_expand(
            population,
            node,
            KernelResult(
                frame=mutated_frame,
                weights=result.weights,
                receipt={"expand": expand_lineage_receipt(result.expand)},
            ),
        )


def test_cached_expand_requires_exact_lineage_id_sequence() -> None:
    population = _population()
    node = Node(
        "cached_midpoint",
        "test@1",
        structural=StructuralDelta.EXPAND,
        base="source",
        params={
            "expand_cells": (),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
    )
    before = population.frame
    person = before.table("person")
    household = before.table("household")
    added_person = person.iloc[[0]].copy()
    added_person["person_id"] = np.array([5], dtype=np.int64)
    added_person["person_household_id"] = np.array([15], dtype=np.int64)
    final_person = pd.concat([person, added_person], ignore_index=True)
    added_household = household.iloc[[0]].copy()
    added_household["household_id"] = np.array([15], dtype=np.int64)
    final_household = (
        pd.concat([household, added_household], ignore_index=True)
        .sort_values("household_id")
        .reset_index(drop=True)
    )
    final_weights = Weights(
        np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64), WeightKind.DESIGN
    )
    cached_frame = Frame(
        {"person": final_person, "household": final_household},
        before.schema,
        {"household": final_weights},
        pd.concat([before.strata, before.strata.iloc[[0]]], ignore_index=True),
    )
    lineage = {
        "person": pd.Series([1], index=pd.Index([5], name="person_id"), dtype="int64"),
        "household": pd.Series(
            [10], index=pd.Index([15], name="household_id"), dtype="int64"
        ),
    }

    with pytest.raises(PopulationError, match="final 'household' ids"):
        restore_cached_expand(
            population,
            node,
            KernelResult(
                frame=cached_frame,
                weights=final_weights,
                receipt={"expand": expand_lineage_receipt(lineage)},
            ),
        )


def test_entrant_person_strata_materialize_and_attest_cached_replay() -> None:
    population = _population()
    node = _entrant_person_expand_node()
    result = _entrant_person_expand_result(
        pd.Series(
            ["new"],
            index=pd.Index([5], dtype="int64", name="ignored"),
            dtype=object,
            name="ignored",
        )
    )

    expanded = patch(population, node, result)

    assert expanded.frame.table("person")["person_household_id"].tolist()[-1] == 10
    assert expanded.frame.strata.tolist() == ["a", "a", "b", "b", "new"]
    assert expanded.mass_ledger[-1].before_total == 7.0
    assert expanded.mass_ledger[-1].after_total == 8.0
    assert result.expand is not None
    entrant_receipt = entrant_strata_receipt(
        population.frame, node, result.expand, result.strata
    )
    receipt = {
        "expand": expand_lineage_receipt(result.expand),
        "entrant_strata": entrant_receipt,
    }
    cached = restore_cached_expand(
        population,
        node,
        KernelResult(
            frame=expanded.frame,
            weights=result.weights,
            receipt=receipt,
        ),
    )
    pd.testing.assert_series_equal(cached.frame.strata, expanded.frame.strata)
    assert cached.mass_ledger == expanded.mass_ledger

    with pytest.raises(PopulationError, match="entrant-strata receipt"):
        restore_cached_expand(
            population,
            node,
            KernelResult(
                frame=expanded.frame,
                weights=result.weights,
                receipt={"expand": expand_lineage_receipt(result.expand)},
            ),
        )


def test_cached_entrant_strata_rehydrate_the_base_id_dtype() -> None:
    source = _frame()
    person = source.table("person").copy()
    household = source.table("household").copy()
    for column in ("person_id", "person_household_id"):
        person[column] = person[column].astype("int32")
    household["household_id"] = household["household_id"].astype("int32")
    frame = Frame(
        {"person": person, "household": household},
        source.schema,
        {"household": source.weights_for("household")},
        source.strata.copy(),
    )
    population = Population.from_frame(frame, "source")
    node = _entrant_person_expand_node(membership_dtype="int32")
    result = _entrant_person_expand_result(
        pd.Series(["new"], index=pd.Index([5], dtype="int32"), dtype=object),
        frame=frame,
    )

    expanded = patch(population, node, result)
    assert result.expand is not None
    receipt = {
        "expand": expand_lineage_receipt(result.expand),
        "entrant_strata": entrant_strata_receipt(
            frame, node, result.expand, result.strata
        ),
    }
    cached = restore_cached_expand(
        population,
        node,
        KernelResult(
            frame=expanded.frame,
            weights=result.weights,
            receipt=receipt,
        ),
    )

    assert cached.frame.table("person")["person_id"].dtype == np.dtype("int32")
    pd.testing.assert_series_equal(cached.frame.strata, expanded.frame.strata)


@pytest.mark.parametrize(
    ("receipt_label", "changed_label"),
    [(1, True), (1, 1.0), (-0.0, 0.0)],
    ids=["bool", "float", "signed-zero"],
)
def test_cached_entrant_strata_preserve_label_scalar(
    receipt_label: object, changed_label: object
) -> None:
    population = _population()
    node = _entrant_person_expand_node()
    result = _entrant_person_expand_result(
        pd.Series([receipt_label], index=pd.Index([5], dtype="int64"), dtype=object)
    )
    expanded = patch(population, node, result)
    changed_strata = expanded.frame.strata.copy()
    changed_strata.iloc[-1] = changed_label
    changed_frame = _replace_person_table(
        expanded.frame,
        expanded.frame.table("person").copy(),
        changed_strata,
    )
    assert result.expand is not None
    receipt = {
        "expand": expand_lineage_receipt(result.expand),
        "entrant_strata": entrant_strata_receipt(
            population.frame, node, result.expand, result.strata
        ),
    }

    with pytest.raises(PopulationError, match="label"):
        restore_cached_expand(
            population,
            node,
            KernelResult(
                frame=changed_frame,
                weights=result.weights,
                receipt=receipt,
            ),
        )


def test_cached_entrant_strata_encode_bytes_labels() -> None:
    population = _population()
    node = _entrant_person_expand_node()
    result = _entrant_person_expand_result(
        pd.Series([b"new\x00stratum"], index=pd.Index([5], dtype="int64"), dtype=object)
    )
    expanded = patch(population, node, result)
    assert result.expand is not None
    receipt = {
        "expand": expand_lineage_receipt(result.expand),
        "entrant_strata": entrant_strata_receipt(
            population.frame, node, result.expand, result.strata
        ),
    }

    assert receipt["entrant_strata"] == [[5, {"bytes_hex": "6e6577007374726174756d"}]]
    cached = restore_cached_expand(
        population,
        node,
        KernelResult(
            frame=expanded.frame,
            weights=result.weights,
            receipt=receipt,
        ),
    )
    pd.testing.assert_series_equal(cached.frame.strata, expanded.frame.strata)


@pytest.mark.parametrize(
    "strata",
    [
        None,
        pd.Series(["new"], index=pd.Index([6], dtype="int64"), dtype=object),
        pd.Series(["old", "new"], index=pd.Index([1, 5], dtype="int64"), dtype=object),
        pd.Series(["new"], index=pd.Index([5], dtype="int32"), dtype=object),
        pd.Series([pd.NA], index=pd.Index([5], dtype="int64"), dtype=object),
        pd.Series([1], index=pd.Index([5], dtype="int64"), dtype="int64"),
    ],
    ids=[
        "missing",
        "unknown-id",
        "incumbent-id",
        "wrong-id-dtype",
        "missing-label",
        "wrong-label-dtype",
    ],
)
def test_entrant_person_strata_reject_malformed_exact_set(strata: object) -> None:
    with pytest.raises(PopulationError, match="strata"):
        patch(
            _population(),
            _entrant_person_expand_node(),
            _entrant_person_expand_result(strata),
        )


def test_strata_are_rejected_without_entrant_persons() -> None:
    result = _lineage_expand_result()
    with pytest.raises(PopulationError, match="without entrant persons"):
        patch(
            _population(),
            _lineage_expand_node(),
            KernelResult(
                expand=result.expand,
                weights=result.weights,
                strata=pd.Series([], dtype=object),
            ),
        )


def test_expand_lineage_rejects_an_unknown_source_id() -> None:
    with pytest.raises(PopulationError, match="unknown 'person' source ids"):
        patch(
            _population(),
            _lineage_expand_node(),
            _lineage_expand_result(bad_source=True),
        )


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


# ---------------------------------------------------------------------------
# Object-dtype storage hashing (issue #907)
# ---------------------------------------------------------------------------


def _fresh_str(value: str) -> str:
    """Return an equal ``str`` that is a distinct object from every literal."""

    return "".join(list(value))


def _object_series(values: list[object]) -> pd.Series:
    return pd.Series(values, dtype=object)


_ALL_ROWS = np.ones(3, dtype=np.bool_)


def test_object_storage_hashes_content_not_pyobject_pointers() -> None:
    left = _object_series([_fresh_str("alpha"), _fresh_str("beta"), None])
    right = _object_series([_fresh_str("alpha"), _fresh_str("beta"), None])

    assert left.dtype == object
    assert [id(value) for value in left.to_numpy()[:2]] != [
        id(value) for value in right.to_numpy()[:2]
    ]
    assert _storage_parts(left, _ALL_ROWS) == _storage_parts(right, _ALL_ROWS)
    assert storage_equal(left, right)


def test_object_storage_separates_differing_content() -> None:
    left = _object_series([_fresh_str("alpha"), _fresh_str("beta"), None])
    right = _object_series([_fresh_str("alpha"), _fresh_str("gamma"), None])

    assert _storage_parts(left, _ALL_ROWS) != _storage_parts(right, _ALL_ROWS)
    assert not storage_equal(left, right)


def test_object_storage_keeps_the_null_bitmap_separate_from_values() -> None:
    series = _object_series([_fresh_str("alpha"), None, _fresh_str("beta")])

    values, bitmap = _storage_parts(series, _ALL_ROWS)

    assert bitmap == np.array([False, True, False]).tobytes()
    assert (
        values
        == _storage_parts(
            _object_series([_fresh_str("alpha"), None, _fresh_str("beta")]), _ALL_ROWS
        )[0]
    )
    # A moved null is a different column even though the value bytes of the
    # surviving labels are unchanged, exactly as in the StringDtype branch.
    moved = _object_series([None, _fresh_str("alpha"), _fresh_str("beta")])
    assert _storage_parts(series, _ALL_ROWS) != _storage_parts(moved, _ALL_ROWS)


def test_object_storage_respects_the_selection_mask() -> None:
    left = _object_series([_fresh_str("alpha"), _fresh_str("beta"), None])
    right = _object_series([_fresh_str("alpha"), _fresh_str("zeta"), None])
    selected = np.array([True, False, True])

    assert _storage_parts(left, selected) == _storage_parts(right, selected)
    assert storage_equal(left, right, selected)
    assert not storage_equal(left, right)


@pytest.mark.parametrize(
    ("left_leaf", "right_leaf"),
    [
        ("1", 1),
        ("1", b"1"),
        (1, b"1"),
        (1, True),
        (1, 1.0),
        (True, 1.0),
        ("", None),
        (b"", None),
        (0.0, -0.0),
        ("a", "a\x00"),
        ("ab", "a"),
    ],
)
def test_object_storage_does_not_conflate_distinct_leaves(
    left_leaf: object, right_leaf: object
) -> None:
    left = _object_series([left_leaf, None, None])
    right = _object_series([right_leaf, None, None])

    assert _storage_parts(left, _ALL_ROWS) != _storage_parts(right, _ALL_ROWS)


@pytest.mark.parametrize(
    "leaf",
    [
        _fresh_str("label"),
        b"label",
        7,
        -(2**70),
        0,
        1.5,
        -0.0,
        True,
        False,
        None,
        pd.NA,
        pd.NaT,
        np.int64(3),
        np.bool_(True),
        np.float64(1.5),
        np.bytes_(b"label"),
        np.str_("label"),
    ],
)
def test_object_storage_accepts_every_supported_leaf(leaf: object) -> None:
    left = _object_series([leaf, None, _fresh_str("tail")])
    right = _object_series([leaf, None, _fresh_str("tail")])

    assert _storage_parts(left, _ALL_ROWS) == _storage_parts(right, _ALL_ROWS)


def test_object_storage_preserves_negative_zero_and_nan_payloads() -> None:
    nan_payload = np.frombuffer(
        np.uint64(0x7FF8_0000_0000_0001).tobytes(), dtype=np.float64
    )[0]
    quiet_nan = float("nan")

    assert _storage_parts(
        _object_series([-0.0, None, None]), _ALL_ROWS
    ) != _storage_parts(_object_series([0.0, None, None]), _ALL_ROWS)
    assert _storage_parts(
        _object_series([float(nan_payload), None, None]), _ALL_ROWS
    ) != _storage_parts(_object_series([quiet_nan, None, None]), _ALL_ROWS)


@pytest.mark.parametrize(
    "leaf",
    [
        object(),
        ("tuple",),
        ["list"],
        {"set"},
        {"dict": 1},
        bytearray(b"mutable"),
        complex(1, 2),
        np.datetime64("2020-01-01"),
        np.timedelta64(1, "D"),
        np.timedelta64(1, "ns"),
        np.timedelta64(1, "M"),
        np.timedelta64(1, "Y"),
        Decimal("1.5"),
        date(2020, 1, 1),
    ],
)
def test_object_storage_refuses_unsupported_leaves(leaf: object) -> None:
    series = _object_series([leaf, None, None])

    with pytest.raises(PopulationError, match="storage-object-leaf"):
        _storage_parts(series, _ALL_ROWS)


@pytest.mark.parametrize(
    "leaf",
    [
        np.timedelta64(1, "ns"),
        np.timedelta64(1, "M"),
        np.timedelta64(1, "Y"),
        np.datetime64("2020-01-01"),
    ],
)
def test_object_storage_refuses_numpy_datetimes_by_type_not_by_unit(
    leaf: object,
) -> None:
    """``np.timedelta64`` subclasses ``np.signedinteger`` at runtime.

    A unit ``int()`` converts (ns, M, Y) would otherwise reach the integer
    branch, drop the unit, and collide with the plain ``1``; the codec must
    refuse the type before that branch, so the refusal never depends on which
    unit happens to fail ``int()``.
    """

    from microcosm.graph.store import _encode_object_scalar

    assert _encode_object_scalar(1) == _encode_object_scalar(np.int64(1))
    with pytest.raises(TypeError, match="datetime64 or timedelta64"):
        _encode_object_scalar(leaf)
    series = _object_series([leaf, None, None])
    with pytest.raises(PopulationError, match="storage-object-leaf"):
        _storage_parts(series, _ALL_ROWS)


def test_object_storage_refuses_an_unencodable_string_leaf() -> None:
    """A lone surrogate makes the encoder raise UnicodeEncodeError, not TypeError."""

    series = _object_series(["\ud800", None, None])

    with pytest.raises(PopulationError, match="storage-object-leaf"):
        _storage_parts(series, _ALL_ROWS)


def test_object_storage_refuses_an_unsupported_leaf_compared_with_itself() -> None:
    """The one place the refusal is new rather than sharper.

    Positional copies preserve PyObject identity, so the pre-fix pointer
    comparison answered True for a column compared against itself no matter
    what it held. It now refuses, matching ContentStore, which will not
    persist such a column, and token_for_dtype, which will not declare it.
    """

    series = _object_series([date(2020, 1, 1), None, None])

    with pytest.raises(PopulationError, match="storage-object-leaf"):
        storage_equal(series, series)


def test_object_storage_refusal_message_excludes_the_value_repr() -> None:
    class _Loud:
        def __repr__(self) -> str:  # pragma: no cover - must never be called
            raise AssertionError("storage refusal must not repr the leaf")

    series = _object_series([_Loud(), None, None])

    with pytest.raises(PopulationError, match="storage-object-leaf") as excinfo:
        _storage_parts(series, _ALL_ROWS)
    assert "_Loud" in str(excinfo.value)


@pytest.mark.parametrize(
    ("numpy_leaf", "python_leaf"),
    [
        (np.int32(1), 1),
        (np.int64(1), 1),
        (np.uint64(2**64 - 1), 2**64 - 1),
        (np.float32(1.0), 1.0),
        (np.float64(1.5), 1.5),
        (np.bool_(True), True),
        (np.str_("a"), "a"),
        (np.bytes_(b"a"), b"a"),
    ],
)
def test_object_storage_normalizes_numpy_scalars_like_the_store_decoder(
    numpy_leaf: object, python_leaf: object
) -> None:
    """A deliberate widening, and the reason the store round trip is a fixed point.

    ``_decode_object_chunks`` hands back the Python form, so refusing to call
    these equal would mean a column could never equal its own reloaded self.
    """

    assert storage_equal(
        _object_series([numpy_leaf, None, None]),
        _object_series([python_leaf, None, None]),
    )


def test_object_storage_uses_the_content_store_leaf_encoding() -> None:
    """The object encoding is the ContentStore's, not a second definition."""

    leaves = [_fresh_str("alpha"), 7, None]
    series = _object_series(leaves)

    payload = bytearray()
    for leaf in leaves:
        body = _encode_object_scalar(leaf)
        payload.extend(len(body).to_bytes(8, "little"))
        payload.extend(body)

    assert _storage_parts(series, _ALL_ROWS)[0] == bytes(payload)


@pytest.mark.parametrize(
    ("left_leaf", "right_leaf"),
    [(None, pd.NA), (None, pd.NaT), (pd.NA, pd.NaT), (None, float("nan"))],
)
def test_object_storage_keeps_distinct_null_sentinels_apart(
    left_leaf: object, right_leaf: object
) -> None:
    """A null bitmap alone would collapse these; the value bytes must not."""

    left = _object_series([left_leaf, _fresh_str("tail"), None])
    right = _object_series([right_leaf, _fresh_str("tail"), None])

    assert left.isna().tolist() == right.isna().tolist()
    assert _storage_parts(left, _ALL_ROWS)[1] == _storage_parts(right, _ALL_ROWS)[1]
    assert _storage_parts(left, _ALL_ROWS) != _storage_parts(right, _ALL_ROWS)


def test_object_column_survives_a_content_store_round_trip(tmp_path: Path) -> None:
    """The point of the fix: a reloaded column equals the one that was stored."""

    schema = EntitySchema(group_entities=("household",))
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "tenure": pd.Series(
                [_fresh_str("OWNED"), None, _fresh_str("RENTED")], dtype=object
            ),
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([10, 20], dtype=np.int64)})
    frame = Frame(
        {"person": person, "household": household},
        schema,
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b"], name="stratum", dtype=object),
    )
    key = "b" * 64
    store = ContentStore(tmp_path / "store")
    store.put_frame(key, frame, node_key="c" * 64)
    reloaded = store.load_frame(key, node_key="c" * 64)

    original = frame.table("person")["tenure"]
    restored = reloaded.table("person")["tenure"]
    assert restored.dtype == object
    assert [id(value) for value in restored.to_numpy()] != [
        id(value) for value in original.to_numpy()
    ]
    assert storage_equal(original, restored)
    assert storage_equal(frame.strata, reloaded.strata)


def test_object_storage_flows_through_storage_equal_dtype_guard() -> None:
    objects = _object_series([_fresh_str("alpha"), _fresh_str("beta"), None])
    strings = pd.Series(
        ["alpha", "beta", None],
        dtype=pd.StringDtype(storage="python", na_value=pd.NA),
    )

    # Different dtypes short-circuit before any encoding is compared, so the
    # object encoding never has to agree byte-for-byte with the string branch.
    assert not storage_equal(objects, strings)


@pytest.mark.parametrize(
    ("dtype", "values", "expected_values", "expected_bitmap"),
    [
        (
            "Int64",
            [1, 2, 3, pd.NA],
            "010000000000000002000000000000000100000000000000",
            "000001",
        ),
        ("boolean", [True, False, pd.NA, True], "010001", "000000"),
        (
            "float64",
            [-0.0, 1.5, 2.0, float("nan")],
            "0000000000000080000000000000f83f000000000000f87f",
            "000001",
        ),
        (
            "int64",
            [1, 2, 3, 4],
            "010000000000000002000000000000000400000000000000",
            "000000",
        ),
        ("bool", [True, False, True, False], "010000", "000000"),
    ],
)
def test_masked_and_numeric_storage_encodings_stay_byte_identical(
    dtype: str, values: list[object], expected_values: str, expected_bitmap: str
) -> None:
    series = pd.Series(values, dtype=dtype)
    selected = np.array([True, True, False, True])

    payload, bitmap = _storage_parts(series, selected)

    assert payload.hex() == expected_values
    assert bitmap.hex() == expected_bitmap


@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("datetimetz", pd.to_datetime(["2020-01-01"] * 3, utc=True)),
        ("period", pd.period_range("2020-01", "2020-03", freq="M")),
        ("interval", pd.interval_range(0, 3)),
    ],
)
def test_object_backed_extension_dtypes_now_fail_closed(
    label: str, values: object
) -> None:
    """These materialize as object arrays, so they were pointer-hashed too.

    ``ContentStore`` already refuses to persist them (store.py rejects
    CategoricalDtype, DatetimeTZDtype and every other extension dtype), and
    ``token_for_dtype`` refuses to declare them, so an explicit refusal is the
    consistent outcome — silently comparing their addresses was not.
    """

    series = pd.Series(values)

    assert series.to_numpy(copy=False).dtype == object
    with pytest.raises(PopulationError, match="storage-object-leaf"):
        _storage_parts(series, _ALL_ROWS)


def test_categorical_storage_compares_category_values_not_addresses() -> None:
    left = pd.Series(pd.Categorical([_fresh_str("a"), _fresh_str("b"), None]))
    right = pd.Series(pd.Categorical([_fresh_str("a"), _fresh_str("b"), None]))

    assert left.dtype == right.dtype
    assert left.to_numpy(copy=False).dtype == object
    assert storage_equal(left, right)
    assert not storage_equal(
        left, pd.Series(pd.Categorical([_fresh_str("a"), _fresh_str("b"), "c"]))
    )


def test_masked_storage_still_compares_bytes_beneath_the_null_mask() -> None:
    """Pins the docstring claim that storage_equal stays an in-process seal."""

    direct = pd.Series([1, 2, pd.NA], dtype="Int64")
    masked = pd.Series([1, 2, 7], dtype="Int64").mask(pd.Series([False, False, True]))

    assert direct.tolist() == masked.tolist()
    assert direct.isna().tolist() == masked.isna().tolist()
    assert not storage_equal(direct, masked)


def test_string_storage_encoding_stays_byte_identical() -> None:
    series = pd.Series(
        ["a", "bb", "c", None],
        dtype=pd.StringDtype(storage="python", na_value=pd.NA),
    )
    selected = np.array([True, True, False, True])

    payload, bitmap = _storage_parts(series, selected)

    assert payload.hex() == ("020000000000000061030000000000000062620000000000000000")
    assert bitmap.hex() == "000001"


def test_new_dense_column_on_a_filtered_population_keeps_its_dtype() -> None:
    # A filtered population (a sampled spine rung) carries a non-contiguous
    # table index. The zero-filled placeholder for a new dense column is
    # built positionally; inserting it label-aligned NaN-filled the gaps and
    # silently widened person.int64 to float64 (found by the microcosm#791
    # sampled build). The placeholder must bind to the table's own index.
    frame = _frame()
    person = frame.table("person").copy()
    person.index = pd.Index([243, 244, 1099, 1250])
    strata = frame.strata.copy()
    strata.index = person.index
    population = Population.from_frame(
        _replace_person_table(frame, person, strata), "source"
    )
    node = Node(
        "new_dense_all_rows",
        "test@1",
        inputs=(Slice("person", ("amount",)),),
        outputs=(
            Owned("person", "count", "int64"),
            Owned("person", "flag", "bool"),
        ),
    )
    result = KernelResult(
        columns={
            ("person", "count"): pd.Series(
                [3, 1, 2, 4], index=[1, 2, 3, 4], dtype="int64"
            ),
            ("person", "flag"): pd.Series(
                [True, False, True, False], index=[1, 2, 3, 4], dtype=bool
            ),
        }
    )

    updated = patch(population, node, result)

    table = updated.frame.table("person")
    assert table["count"].dtype == np.dtype("int64")
    assert table["count"].tolist() == [3, 1, 2, 4]
    assert table["flag"].dtype == np.dtype("bool")
    assert table["flag"].tolist() == [True, False, True, False]
