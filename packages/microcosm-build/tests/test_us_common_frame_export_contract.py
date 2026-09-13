"""Invented supplied-parent comparisons; no calibration or source issuer imitation."""

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build import frame_checkpoint
from microcosm.build.us_runtime import common_frame_export_contract as contract
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _parent():
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            "person_household_id": np.array([10, 10, 20, 20, 30, 30], dtype=np.int64),
            "person_tax_unit_id": np.array(
                [100, 100, 200, 201, 300, 300], dtype=np.int64
            ),
            "money": np.array([-0.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "hours": pd.Series([10, pd.NA, 30, 40, pd.NA, 60], dtype="Int32"),
            "reported": pd.Series(
                [True, pd.NA, False, True, pd.NA, False], dtype="boolean"
            ),
            "source_person_id": np.arange(101, 107, dtype=np.int64),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array([10, 20, 30], dtype=np.int64),
            "origin": pd.Series(["asec", "acs", "acs"], dtype="string"),
            "assigned_block": pd.Series(["001", "002", "003"], dtype="string"),
            "county": pd.Series(["A", "A", "B"], dtype="string"),
            "clone": np.array([0, 1, 0], dtype=np.int8),
        }
    )
    tax = pd.DataFrame(
        {
            "tax_unit_id": np.array([100, 200, 201, 300], dtype=np.int64),
            "deduction": np.array([1.0, 2.0, 3.0, 4.0]),
        }
    )
    return Frame(
        {"person": person, "household": household, "tax_unit": tax},
        EntitySchema(group_entities=("household", "tax_unit")),
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.IMPORTANCE)},
        pd.Series(["a", "a", "b", "b", "b", "b"], name="stratum"),
        metadata={"construction": "one-common-parent"},
    )


def _weighted(parent, person_mask, weights):
    selected = parent.select(np.asarray(person_mask, dtype=np.bool_))
    return Frame(
        {e: selected.table(e).copy(deep=True) for e in selected.entities},
        selected.schema,
        {
            "household": Weights(
                np.asarray(weights, dtype=np.float64), WeightKind.CALIBRATED
            )
        },
        selected.strata.copy(),
        metadata=selected.metadata,
    )


def _arguments(parent):
    return {
        "parent_reference": "supplied-parent-example-not-an-issued-graph-key",
        "ordered_household_ids": parent.table("household")
        .household_id.to_numpy()
        .copy(),
        "calibrated_weights": np.array([0.0, 9.0, 10.0]),
        "calibration_specification": b'{"targets":"invented","scope":"national"}',
    }


@pytest.mark.parametrize("view", ["full", "pruned", "local"])
def test_one_parent_full_pruned_and_local_preserve_all_inputs(view):
    parent = _parent()
    args = _arguments(parent)
    if view == "full":
        candidate = _weighted(parent, [True] * 6, [0, 9, 10])
        args["prune_zero_weight"] = False
    elif view == "pruned":
        candidate = _weighted(parent, [False, False, True, True, True, True], [9, 10])
    else:
        candidate = _weighted(parent, [False, False, False, False, True, True], [10])
        args["scope_household_ids"] = np.array([30], dtype=np.int64)
    binding = contract.verify_retained_frame_export(parent, candidate, **args)
    report = json.loads(binding)
    assert report["actual_graph_calibration_ancestry_verified"] is False
    assert report["release_eligible"] is False
    assert report["retained_entity_ids"]["person"]["rows"] == candidate.n("person")
    if view != "full":
        # A complete survey origin may legitimately have zero calibrated weight.
        assert candidate.table("household").origin.tolist() == ["acs"] * candidate.n(
            "household"
        )
    assert (
        contract.verify_retained_frame_export(
            parent, candidate, expected_binding=binding, **args
        )
        == binding
    )


def test_actual_checkpoint_readback_compares_retained_values_and_masks(tmp_path):
    parent = _parent()
    candidate = _weighted(parent, [False, False, True, True, True, True], [9, 10])
    args = _arguments(parent)
    before = contract.verify_retained_frame_export(parent, candidate, **args)
    path = tmp_path / "invented-export.h5"
    frame_checkpoint.write_frame_checkpoint(
        path, candidate, metadata={"comparison": before.decode()}
    )
    loaded = frame_checkpoint.load_frame_checkpoint(
        path, frame_metadata=parent.metadata
    )
    assert loaded.metadata["comparison"] == before.decode()
    assert (
        contract.verify_retained_frame_export(
            parent,
            loaded.frame,
            comparison="frame-checkpoint-readback",
            expected_binding=before,
            **args,
        )
        == before
    )
    loaded.frame.person.loc[loaded.frame.person.index[0], "money"] += 1
    with pytest.raises(
        contract.RetainedFrameExportError, match="RETAINED_INPUT:person.money"
    ):
        contract.verify_retained_frame_export(
            parent,
            loaded.frame,
            comparison="frame-checkpoint-readback",
            expected_binding=before,
            **args,
        )


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "mask",
        "membership",
        "id",
        "geography",
        "lineage",
        "weight",
        "strata",
        "column",
    ],
)
def test_retained_input_changes_refuse(change):
    parent = _parent()
    candidate = _weighted(parent, [False, False, True, True, True, True], [9, 10])
    if change == "value":
        candidate.person.loc[2, "money"] += 1
    elif change == "mask":
        candidate.person.loc[4, "hours"] = 0
    elif change == "membership":
        candidate.person.loc[2, "person_household_id"] = 30
    elif change == "id":
        candidate.person.loc[2, "person_id"] = 99
    elif change == "geography":
        candidate.table("household").loc[0, "assigned_block"] = "004"
    elif change == "lineage":
        candidate.person.loc[2, "source_person_id"] = 999
    elif change == "weight":
        candidate = _weighted(parent, [False, False, True, True, True, True], [8, 10])
    elif change == "strata":
        candidate.strata.iloc[0] = "changed"
    else:
        candidate.person.drop(columns="money", inplace=True)
    with pytest.raises(
        contract.RetainedFrameExportError, match="RETAINED_|COLUMN_ROSTER"
    ):
        contract.verify_retained_frame_export(parent, candidate, **_arguments(parent))


def test_partial_household_selection_refuses_even_with_valid_frame_links():
    parent = _parent()
    candidate = _weighted(parent, [False, False, True, False, True, True], [9, 10])
    with pytest.raises(
        contract.RetainedFrameExportError, match="RETAINED_ENTITY_IDS:person"
    ):
        contract.verify_retained_frame_export(parent, candidate, **_arguments(parent))


@pytest.mark.parametrize(
    "change",
    ["order", "dtype", "nan", "negative", "shape", "scope_order", "scope_unknown"],
)
def test_weight_and_scope_axes_are_explicit(change):
    parent = _parent()
    candidate = _weighted(parent, [False, False, True, True, True, True], [9, 10])
    args = _arguments(parent)
    if change == "order":
        args["ordered_household_ids"] = args["ordered_household_ids"][::-1]
    elif change == "dtype":
        args["ordered_household_ids"] = args["ordered_household_ids"].astype(float)
    elif change == "nan":
        args["calibrated_weights"][0] = np.nan
    elif change == "negative":
        args["calibrated_weights"][0] = -1
    elif change == "shape":
        args["calibrated_weights"] = args["calibrated_weights"][:2]
    elif change == "scope_order":
        args["scope_household_ids"] = np.array([30, 20], dtype=np.int64)
    else:
        args["scope_household_ids"] = np.array([99], dtype=np.int64)
    with pytest.raises(contract.RetainedFrameExportError):
        contract.verify_retained_frame_export(parent, candidate, **args)


@pytest.mark.parametrize(
    "change", ["specification", "parent_reference", "excluded_weight", "scope", "prune"]
)
def test_readback_binding_covers_even_inputs_outside_retained_rows(change):
    parent = _parent()
    candidate = _weighted(parent, [False, False, True, True, True, True], [9, 10])
    args = _arguments(parent)
    args["scope_household_ids"] = np.array([20, 30], dtype=np.int64)
    before = contract.verify_retained_frame_export(parent, candidate, **args)
    if change == "specification":
        args["calibration_specification"] = b'{"targets":"different"}'
    elif change == "parent_reference":
        args["parent_reference"] = "another-unverified-reference"
    elif change == "excluded_weight":
        args["calibrated_weights"][0] = 100
    elif change == "scope":
        args["scope_household_ids"] = None
    else:
        args["prune_zero_weight"] = False
    with pytest.raises(contract.RetainedFrameExportError, match="EXPORT_BINDING"):
        contract.verify_retained_frame_export(
            parent, candidate, expected_binding=before, **args
        )


@pytest.mark.parametrize("scope", ["empty", "all_zero"])
def test_empty_analysis_is_an_explicit_unsupported_frame_contract(scope):
    parent = _parent()
    candidate = _weighted(parent, [True] * 6, [0, 9, 10])
    args = _arguments(parent)
    if scope == "empty":
        args["scope_household_ids"] = np.array([], dtype=np.int64)
    else:
        args["calibrated_weights"] = np.zeros(3, dtype=np.float64)
    with pytest.raises(contract.RetainedFrameExportError, match="EMPTY_EXPORT_SUPPORT"):
        contract.verify_retained_frame_export(parent, candidate, **args)


def test_uint64_stable_ids_above_int64_are_not_coerced():
    ids = np.array([2**63 + 1, 2**63 + 7], dtype=np.uint64)
    parent = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.array([1, 2], dtype=np.int32),
                    "person_household_id": ids,
                }
            ),
            "household": pd.DataFrame({"household_id": ids}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(2), WeightKind.IMPORTANCE)},
    )
    candidate = _weighted(parent, [True, True], [2, 3])
    args = {
        "parent_reference": "large-ids",
        "ordered_household_ids": ids,
        "calibrated_weights": np.array([2.0, 3.0]),
        "calibration_specification": b"{}",
    }
    report = json.loads(
        contract.verify_retained_frame_export(parent, candidate, **args)
    )
    assert report["complete_ordered_household_ids"]["dtype"] == ids.dtype.str
    with pytest.raises(contract.RetainedFrameExportError, match="HOUSEHOLD_ID_VECTOR"):
        contract.verify_retained_frame_export(
            parent, candidate, **{**args, "ordered_household_ids": ids.astype(float)}
        )


def test_readback_normalization_is_closed_and_never_changes_knownness():
    parent = _parent()
    candidate = _weighted(parent, [False, False, True, True, True, True], [9, 10])
    args = _arguments(parent)
    before = contract.verify_retained_frame_export(parent, candidate, **args)
    values = candidate.person["hours"].array
    assert values._mask[2]
    values._data[2] ^= np.int32(1)
    with pytest.raises(
        contract.RetainedFrameExportError, match="RETAINED_INPUT:person.hours"
    ):
        contract.verify_retained_frame_export(
            parent, candidate, expected_binding=before, **args
        )
    assert (
        contract.verify_retained_frame_export(
            parent,
            candidate,
            comparison="frame-checkpoint-readback",
            expected_binding=before,
            **args,
        )
        == before
    )
    with pytest.raises(contract.RetainedFrameExportError, match="COMPARISON_MODE"):
        contract.verify_retained_frame_export(
            parent, candidate, comparison="approximate", **args
        )
    values._mask[2] = False
    with pytest.raises(
        contract.RetainedFrameExportError, match="RETAINED_INPUT:person.hours"
    ):
        contract.verify_retained_frame_export(
            parent, candidate, comparison="frame-checkpoint-readback", **args
        )


@pytest.mark.parametrize("prune_zero_weight", [True, False])
@pytest.mark.parametrize("scope", ["all", "zero_weight_household"])
def test_zero_weight_export_support_refuses(prune_zero_weight, scope):
    parent = _parent()
    # Keep a valid candidate so the export contract itself checks the supplied
    # weight/scope inputs before comparing retained rows or weight storage.
    candidate = _weighted(parent, [True] * 6, [1, 1, 1])
    args = _arguments(parent)
    args["calibrated_weights"] = np.zeros(3, dtype=np.float64)
    if scope == "zero_weight_household":
        args["calibrated_weights"][1:] = [9, 10]
        args["scope_household_ids"] = args["ordered_household_ids"][:1].copy()
    with pytest.raises(contract.RetainedFrameExportError, match="EMPTY_EXPORT_SUPPORT"):
        contract.verify_retained_frame_export(
            parent, candidate, prune_zero_weight=prune_zero_weight, **args
        )


@pytest.mark.parametrize("prune_zero_weight", [True, False])
def test_positive_calibrated_scope_can_have_zero_parent_weight(prune_zero_weight):
    baseline = _parent()
    parent = Frame(
        {e: baseline.table(e).copy(deep=True) for e in baseline.entities},
        baseline.schema,
        {
            "household": Weights(np.array([0.0, 2.0, 3.0]), WeightKind.IMPORTANCE),
            "tax_unit": Weights(np.array([7.0, 8.0, 9.0, 10.0]), WeightKind.DESIGN),
        },
        baseline.strata.copy(),
        metadata=baseline.metadata,
    )
    before_tables = {e: parent.table(e).copy(deep=True) for e in parent.entities}
    before_strata = parent.strata.copy(deep=True)
    before_metadata, before_mass_log = parent.metadata, parent.mass_log
    before_weights = {
        e: (parent.weights_for(e).values.copy(), parent.weights_for(e).kind)
        for e in parent.weighted_entities
    }
    retained = baseline.select(np.array([True, True, False, False, False, False]))

    def candidate(tax_weight):
        return Frame(
            {e: retained.table(e).copy(deep=True) for e in retained.entities},
            retained.schema,
            {
                "household": Weights(np.array([4.0]), WeightKind.CALIBRATED),
                "tax_unit": Weights(np.array([tax_weight]), WeightKind.DESIGN),
            },
            retained.strata.copy(),
            metadata=retained.metadata,
        )

    args = _arguments(parent)
    args["calibrated_weights"] = np.array([4.0, 0.0, 0.0])
    args["scope_household_ids"] = np.array([10], dtype=np.int64)
    report = json.loads(
        contract.verify_retained_frame_export(
            parent, candidate(7.0), prune_zero_weight=prune_zero_weight, **args
        )
    )
    assert report["actual_graph_calibration_ancestry_verified"] is False
    assert report["release_eligible"] is False
    assert report["complete_ordered_household_ids"]["rows"] == 3
    assert report["retained_entity_ids"]["household"]["rows"] == 1
    with pytest.raises(
        contract.RetainedFrameExportError, match="RETAINED_WEIGHTS:tax_unit"
    ):
        contract.verify_retained_frame_export(
            parent, candidate(8.0), prune_zero_weight=prune_zero_weight, **args
        )
    for entity, table in before_tables.items():
        pd.testing.assert_frame_equal(parent.table(entity), table, check_exact=True)
    pd.testing.assert_series_equal(parent.strata, before_strata, check_exact=True)
    assert parent.metadata == before_metadata and parent.mass_log == before_mass_log
    for entity, (values, kind) in before_weights.items():
        assert parent.weights_for(entity).kind is kind
        assert parent.weights_for(entity).values.tobytes() == values.tobytes()
