"""Fixture-scale tests for the US Frame/spec-executor projection boundary."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from microcosm.build.spec_engine.compiler_ir import CompiledNode
from microcosm.build.spec_engine.model import FrozenMap, freeze_json
from microcosm.build.spec_engine.scope_algebra import ClosedScopeRegistry
from microcosm.build.us_runtime.pool_frame_projection import (
    FrameProjectionCodecError,
    classify_added_support_rows,
    frame_to_projection,
    legacy_result_to_patch,
    merge_projection_into_frame,
    projection_to_frame,
)
from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChangeRecord,
    WeightKind,
    Weights,
)

_SHA = "a" * 64


def _frozen(value: object) -> FrozenMap:
    frozen = freeze_json(value)
    assert isinstance(frozen, FrozenMap)
    return frozen


def _input(entity: str, column: str) -> dict[str, object]:
    return {
        "entity": entity,
        "column": column,
        "required_scope": "whole_pool",
        "alternatives": [],
        "tolerated_absence_receipts": [],
    }


def _output(entity: str, column: str) -> dict[str, object]:
    return {
        "entity": entity,
        "column": column,
        "coverage_scope": "whole_pool",
    }


def _node(
    *,
    registry: ClosedScopeRegistry,
    structural_delta: str = "join",
    inputs: tuple[dict[str, object], ...] = (),
    outputs: tuple[dict[str, object], ...] = (),
) -> CompiledNode:
    return CompiledNode(
        id="fixture_node",
        execution_rank=0,
        node_key=_SHA,
        node_slice_sha256=_SHA,
        kernel_ref="kernel:fixture",
        kernel_implementation_sha256=_SHA,
        depends_on=(),
        inputs=tuple(_frozen(value) for value in inputs),
        outputs=tuple(_frozen(value) for value in outputs),
        capabilities=_frozen({"structural_delta": structural_delta}),
        mutations=_frozen({}),
        write_scopes=tuple(
            _frozen(
                {
                    "entity": value["entity"],
                    "column": value["column"],
                }
            )
            for value in outputs
        ),
        scope_registry=_frozen(registry.to_wire()),
        row_classifier_ref=f"classifier:{registry.predicate_space}",
        row_classifier_implementation_sha256=_SHA,
        compiler_ir_abi=_frozen({}),
        seed_protocol_sha256=_SHA,
        seed_sites=(),
        seed_streams=(),
        resolved_params=(),
        transitive_nodes=(),
    )


@pytest.fixture
def registry() -> ClosedScopeRegistry:
    universe = (
        "origin:survey_alpha/clone:0",
        "origin:survey_alpha/clone:1",
        "origin:survey_beta/clone:0",
        "origin:survey_beta/clone:1",
        "receipt:virtual",
    )
    return ClosedScopeRegistry(
        "fixture_origin_clone_or_receipt",
        universe,
        {
            "native": universe[:1] + universe[2:3],
            "clones": universe[1:2] + universe[3:4],
            "receipt": ("receipt:virtual",),
            "whole_pool": universe,
        },
    )


@pytest.fixture
def join_node(registry: ClosedScopeRegistry) -> CompiledNode:
    return _node(
        registry=registry,
        inputs=(
            _input("person", "person_observed"),
            _input("frame", "@stage_receipt"),
            _input("person", "@resource_receipt"),
            _input("household", "@resolved_weight"),
        ),
        outputs=(
            _output("person", "person_prediction"),
            _output("frame", "@node_receipt"),
            _output("person", "@operator_receipt"),
        ),
    )


def _wide_frame(*, prediction: bool = False, node_receipt: bool = False) -> Frame:
    schema = EntitySchema(
        group_entities=("household",),
        links=(
            LinkSpec(
                name="person_household_edge",
                left_entity="person",
                right_entity="household",
            ),
        ),
    )
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [10, 20],
            "person_support_channel": ["survey_alpha", "survey_beta"],
            "person_support_clone_index": np.array([0, 0], dtype=np.int64),
            "person_source_id": [1, 2],
            "person_observed": np.array([11.0, 22.0], dtype=np.float64),
            "person_private": pd.Series(["left", "right"], dtype="string"),
        },
        index=pd.Index([4, 8], name="person_row"),
    )
    if prediction:
        person["person_prediction"] = np.array([12.5, 23.5], dtype=np.float64)
    person.attrs["table_contract"] = {"version": 1}
    household = pd.DataFrame(
        {
            "household_id": [10, 20],
            "household_support_channel": ["survey_alpha", "survey_beta"],
            "household_support_clone_index": np.array([0, 0], dtype=np.int64),
            "household_source_id": [10, 20],
            "household_private": np.array([101, 202], dtype=np.int64),
        }
    )
    edge = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 20],
            "edge_private": ["a", "b"],
        }
    )
    metadata: dict[str, object] = {"stage_receipt": {"version": 1}}
    if node_receipt:
        metadata["node_receipt"] = {"digest": "ok"}
    return Frame(
        {
            "person": person,
            "household": household,
            "person_household_edge": edge,
        },
        schema,
        {
            "household": Weights(
                np.array([1.25, 2.75]),
                WeightKind.IMPORTANCE,
            )
        },
        pd.Series(
            ["alpha", "beta"],
            index=person.index,
            name="stratum",
            dtype=object,
        ),
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=2.0,
                new_total=4.0,
                declared_factor=2.0,
                reason="fixture importance expansion",
            ),
        ),
        metadata=metadata,
    )


def test_frame_projection_is_node_bounded_and_covers_typed_surfaces(
    join_node: CompiledNode,
) -> None:
    frame = _wide_frame()
    resource_receipt = {"sha256": "b" * 64}

    projection = frame_to_projection(
        frame,
        node=join_node,
        available_inputs={"person.@resource_receipt": resource_receipt},
    )

    assert projection.entities == ("person", "household")
    assert projection.table("person").columns.tolist() == [
        "person_id",
        "person_household_id",
        "person_observed",
    ]
    assert projection.table("household").columns.tolist() == ["household_id"]
    assert projection.link("person_household_edge").columns.tolist() == [
        "person_id",
        "household_id",
    ]
    np.testing.assert_array_equal(
        projection.weights_for("household").values,
        np.array([1.25, 2.75]),
    )
    assert projection.weights_for("household").kind == "importance"
    pd.testing.assert_series_equal(projection.strata, frame.strata)
    assert projection.mass_history == frame.mass_log
    assert projection.row_atoms_for("person") == {
        1: frozenset({"origin:survey_alpha/clone:0"}),
        2: frozenset({"origin:survey_beta/clone:0"}),
    }

    narrow = projection_to_frame(projection, schema=frame.schema)
    assert narrow.table("person").columns.tolist() == [
        "person_id",
        "person_household_id",
        "person_observed",
    ]
    assert narrow.table("household").columns.tolist() == ["household_id"]
    pd.testing.assert_frame_equal(
        narrow.link("person_household_edge"),
        projection.link("person_household_edge"),
    )
    np.testing.assert_array_equal(
        narrow.weights_for("household").values,
        projection.weights_for("household").values,
    )
    assert narrow.weights_for("household").kind.value == "importance"
    assert narrow.metadata == frame.metadata
    assert narrow.mass_log == frame.mass_log
    assert projection.row_atoms_for("household") == {
        10: frozenset({"origin:survey_alpha/clone:0"}),
        20: frozenset({"origin:survey_beta/clone:0"}),
    }
    assert projection.virtual_receipts[("person", "@resource_receipt")] == (
        resource_receipt
    )
    assert ("frame", "@stage_receipt") in projection.virtual_receipts
    np.testing.assert_array_equal(
        projection.virtual_receipts[("household", "@resolved_weight")],
        np.array([1.25, 2.75]),
    )


def test_patch_and_merge_restore_only_proven_external_surfaces(
    join_node: CompiledNode,
) -> None:
    before = _wide_frame()
    result = _wide_frame(prediction=True, node_receipt=True)
    receipt = {"digest": "ok"}

    patch = legacy_result_to_patch(
        before,
        result,
        node=join_node,
        virtual_writes={
            ("frame", "@node_receipt"): receipt,
            ("person", "@operator_receipt"): {"rows": 2},
        },
    )

    assert patch.metadata is None
    assert patch.tables["person"].columns.tolist() == [
        "person_id",
        "person_household_id",
        "person_observed",
        "person_prediction",
    ]
    assert "person_private" not in patch.tables["person"]
    assert patch.virtual_writes[("frame", "@node_receipt")] == receipt

    validated_projection = frame_to_projection(result, node=join_node)
    merged = merge_projection_into_frame(
        validated_projection,
        before_frame=before,
        legacy_result_frame=result,
        node=join_node,
        validated_metadata=result.metadata,
    )
    for entity in result.entities:
        pd.testing.assert_frame_equal(
            merged.table(entity),
            result.table(entity),
            check_exact=True,
        )
    pd.testing.assert_frame_equal(
        merged.link("person_household_edge"),
        result.link("person_household_edge"),
        check_exact=True,
    )
    np.testing.assert_array_equal(
        merged.weights_for("household").values,
        result.weights_for("household").values,
    )
    assert dict(merged.metadata) == dict(result.metadata)


def test_merge_restores_full_frame_from_narrow_validated_result(
    join_node: CompiledNode,
) -> None:
    before = _wide_frame()
    expected = _wide_frame(prediction=True, node_receipt=True)
    projection = frame_to_projection(expected, node=join_node)
    narrow = projection_to_frame(projection, schema=expected.schema)

    merged = merge_projection_into_frame(
        projection,
        before_frame=before,
        legacy_result_frame=narrow,
        node=join_node,
        validated_metadata=expected.metadata,
    )

    for entity in expected.entities:
        pd.testing.assert_frame_equal(
            merged.table(entity),
            expected.table(entity),
            check_exact=True,
        )
    pd.testing.assert_frame_equal(
        merged.link("person_household_edge"),
        expected.link("person_household_edge"),
        check_exact=True,
    )


def test_patch_refuses_unprojected_data_or_unreceipted_metadata_changes(
    join_node: CompiledNode,
) -> None:
    before = _wide_frame()
    result = _wide_frame(prediction=True, node_receipt=True)
    result.table("person").loc[result.table("person").index[0], "person_private"] = (
        "tampered"
    )

    with pytest.raises(FrameProjectionCodecError, match="outside executor authority"):
        legacy_result_to_patch(
            before,
            result,
            node=join_node,
            virtual_writes={("frame", "@node_receipt"): {"digest": "ok"}},
        )

    untampered = _wide_frame(prediction=True, node_receipt=True)
    with pytest.raises(FrameProjectionCodecError, match="lacks an equal declared"):
        legacy_result_to_patch(before, untampered, node=join_node)


def test_expand_classifier_prefers_lineage_then_unique_remap_inference(
    registry: ClosedScopeRegistry,
) -> None:
    explicit = pd.DataFrame(
        {
            "person_id": [7, 107],
            "person_support_channel": ["survey_alpha", "survey_alpha"],
            "person_support_clone_index": np.array([0, 1], dtype=np.int64),
            "person_source_id": [7, 7],
        }
    )
    classified = classify_added_support_rows(
        "person",
        explicit,
        "person_id",
        frozenset({107}),
        registry,
    )
    assert classified[107].source_row_id == 7
    assert classified[107].atoms == frozenset(
        {"origin:survey_alpha/clone:1"}
    )

    inferred = classify_added_support_rows(
        "person",
        explicit.drop(columns=["person_source_id"]),
        "person_id",
        frozenset({107}),
        registry,
    )
    assert inferred[107].source_row_id == 7

    ambiguous = pd.DataFrame(
        {
            "person_id": [7, 97, 107],
            "person_support_channel": ["survey_alpha"] * 3,
            "person_support_clone_index": np.array([0, 0, 1], dtype=np.int64),
        }
    )
    with pytest.raises(FrameProjectionCodecError, match="exactly one native"):
        classify_added_support_rows(
            "person",
            ambiguous,
            "person_id",
            frozenset({107}),
            registry,
        )


def _expand_frame(*, expanded: bool, corrupt_private: bool = False) -> Frame:
    schema = EntitySchema(group_entities=("household",))
    person_private = ["native"] if not expanded else ["native", "native"]
    if corrupt_private:
        person_private[-1] = "forged"
    person = pd.DataFrame(
        {
            "person_id": [7] if not expanded else [7, 107],
            "person_household_id": [70] if not expanded else [70, 170],
            "person_support_channel": (
                ["survey_alpha"]
                if not expanded
                else ["survey_alpha", "survey_alpha"]
            ),
            "person_support_clone_index": np.array(
                [0] if not expanded else [0, 1], dtype=np.int64
            ),
            "person_source_id": [7] if not expanded else [7, 7],
            "person_private": person_private,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [70] if not expanded else [70, 170],
            "household_support_channel": (
                ["survey_alpha"]
                if not expanded
                else ["survey_alpha", "survey_alpha"]
            ),
            "household_support_clone_index": np.array(
                [0] if not expanded else [0, 1], dtype=np.int64
            ),
            "household_source_id": [70] if not expanded else [70, 70],
            "household_private": ["group"] if not expanded else ["group", "group"],
        }
    )
    return Frame(
        {"person": person, "household": household},
        schema,
        {
            "household": Weights(
                np.array([1.0]) if not expanded else np.array([0.5, 0.5]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(
            ["native"] if not expanded else ["native", "native"],
            index=person.index,
            name="stratum",
        ),
    )


def test_expand_patch_proves_external_columns_are_exact_source_copies(
    registry: ClosedScopeRegistry,
) -> None:
    lineage_inputs = tuple(
        _input(entity, column)
        for entity in ("person", "household")
        for column in (
            f"{entity}_support_channel",
            f"{entity}_support_clone_index",
            f"{entity}_source_id",
        )
    )
    node = _node(
        registry=registry,
        structural_delta="expand",
        inputs=lineage_inputs,
    )
    before = _expand_frame(expanded=False)
    result = _expand_frame(expanded=True)

    patch = legacy_result_to_patch(before, result, node=node)
    assert patch.structural_delta.value == "expand"
    assert "person_private" not in patch.tables["person"]
    assert patch.tables["person"]["person_id"].tolist() == [7, 107]

    corrupted = _expand_frame(expanded=True, corrupt_private=True)
    with pytest.raises(FrameProjectionCodecError, match="did not copy"):
        legacy_result_to_patch(before, corrupted, node=node)

    projection = frame_to_projection(result, node=node)
    narrow = projection_to_frame(projection, schema=result.schema)
    merged = merge_projection_into_frame(
        projection,
        before_frame=before,
        legacy_result_frame=narrow,
        node=node,
        validated_metadata=result.metadata,
    )
    for entity in result.entities:
        pd.testing.assert_frame_equal(
            merged.table(entity),
            result.table(entity),
            check_exact=True,
        )


def test_virtual_inputs_reject_receipts_outside_compiled_contract(
    join_node: CompiledNode,
) -> None:
    with pytest.raises(FrameProjectionCodecError, match="outside node"):
        frame_to_projection(
            _wide_frame(),
            node=join_node,
            available_inputs={("person", "@undeclared"): {"status": "forged"}},
        )


def test_merge_refuses_metadata_not_explicitly_validated(
    join_node: CompiledNode,
) -> None:
    before = _wide_frame()
    result = _wide_frame(prediction=True, node_receipt=True)
    projection = frame_to_projection(result, node=join_node)
    wrong_metadata: Mapping[str, object] = {"stage_receipt": {"version": 1}}

    with pytest.raises(FrameProjectionCodecError, match="validated metadata differs"):
        merge_projection_into_frame(
            projection,
            before_frame=before,
            legacy_result_frame=result,
            node=join_node,
            validated_metadata=wrong_metadata,
        )
