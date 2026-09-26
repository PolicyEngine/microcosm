"""Tiny maintained solves and native input attachment, with no model or issuer."""

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from test_us_common_frame_export_contract import _parent
from test_us_fiscal_refresh_builder import _load_builder_module
from test_us_fiscal_shared_solve import _args

from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import Frame, MassChange, Weights


class Leaves:
    def _engine_computed_columns(self, tables, *, period):
        return {"derived_target", "derived_person", "derived_group"} & {
            name for table in tables.values() for name in table
        }


def _case(dense=True):
    parent = _parent()
    target = Frame(
        {entity: parent.table(entity) for entity in parent.entities},
        parent.schema,
        {"household": parent.weights_for("household")},
        parent.strata,
    )
    target.table("household")["derived_target"] = [20.0, 50.0, 100.0]
    registry = TargetRegistry(
        (
            TargetSpec(
                name="invented.derived",
                entity="household",
                measure="derived_target",
                value=300.0,
                source="Invented tiny calibration fixture",
            ),
        ),
        country="us",
    )
    kwargs = dict(
        args=_args(dense),
        target_loss_weights=np.array([1.0]),
        formula_metadata=Leaves(),
        parent_reference="invented-supplied-parent",
        calibration_specification=b'{"invented":true}',
    )
    return parent, target, registry, kwargs


@pytest.mark.parametrize("dense", [True, False])
def test_maintained_solve_attaches_only_inputs_and_returns_detached_binding(dense):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case(dense)
    before = {e: parent.table(e).copy(deep=True) for e in parent.entities}
    initial = parent.weights_for("household").values.copy()
    attachment = builder._calibrate_native_input_frame(
        parent, target, registry, **kwargs
    )
    assert "derived_target" not in attachment.frame.table("household")
    assert "derived_target" in attachment.result.frame.table("household")
    assert attachment.frame.metadata == parent.metadata
    assert attachment.frame.weights_for("household").kind.value == "calibrated"
    assert attachment.full_parent_weights.dtype == np.dtype("float64")
    assert attachment.full_parent_weights.sum() == pytest.approx(initial.sum())
    binding = json.loads(attachment.binding)
    assert binding["actual_graph_calibration_ancestry_verified"] is False
    assert binding["release_eligible"] is False
    assert binding["prune_zero_weight"] is False
    for e in parent.entities:
        pd.testing.assert_frame_equal(parent.table(e), before[e])
    np.testing.assert_array_equal(parent.weights_for("household").values, initial)
    attachment.frame.person.loc[attachment.frame.person.index[0], "money"] = 987.0
    pd.testing.assert_frame_equal(parent.person, before["person"])
    assert target.person.money.iloc[0] == 0.0


@pytest.mark.parametrize(
    "change", ["household", "order", "membership", "input", "weight", "strata"]
)
def test_target_mismatch_refuses_before_solve(monkeypatch, change):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()
    if change == "household":
        target.table("household").loc[0, "household_id"] = 999
    elif change == "order":
        target.person.iloc[[0, 1]] = target.person.iloc[[1, 0]].to_numpy()
    elif change == "membership":
        target.person.loc[0, "person_household_id"] = 20
    elif change == "input":
        target.person.loc[0, "money"] = 999.0
    elif change == "weight":
        target._weights["household"] = Weights(
            np.array([1.0, 2.0, 4.0]), parent.weights_for("household").kind
        )
    else:
        target.strata.iloc[0] = "changed"
    monkeypatch.setattr(
        builder,
        "_calibrate_fiscal_support",
        lambda *a, **k: pytest.fail("solve reached"),
    )
    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_"):
        builder._calibrate_native_input_frame(parent, target, registry, **kwargs)


@pytest.mark.parametrize(
    "option", ["exact_k", "warm_start_weights", "target_frame_checkpoint_identity"]
)
def test_unsupported_solve_inputs_refuse_before_solve(monkeypatch, option):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()
    setattr(kwargs["args"], option, 1)
    monkeypatch.setattr(
        builder,
        "_calibrate_fiscal_support",
        lambda *a, **k: pytest.fail("solve reached"),
    )
    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_OPTIONS"):
        builder._calibrate_native_input_frame(parent, target, registry, **kwargs)


@pytest.mark.parametrize("which", ["input", "target"])
def test_progress_mutation_of_supplied_frames_refuses(which):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()

    def mutate(event):
        (parent if which == "input" else target).person.loc[0, "money"] = 444.0

    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_CHANGED"):
        builder._calibrate_native_input_frame(
            parent, target, registry, progress_callback=mutate, **kwargs
        )


def test_progress_changes_to_caller_options_and_registry_cannot_rewrite_solve():
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()
    registry_digest = hashlib.sha256(
        builder._strict_json_bytes(
            {"country": registry.country, "specs": [asdict(s) for s in registry]}
        )
    ).hexdigest()

    def mutate(event):
        kwargs["args"].epochs = 999
        kwargs["target_loss_weights"][:] = 999
        registry.specs[0].metadata["changed"] = "during solve"
        registry._specs = (replace(registry.specs[0], value=999.0),)

    attachment = builder._calibrate_native_input_frame(
        parent, target, registry, progress_callback=mutate, **kwargs
    )
    specification = json.loads(attachment.comparison_specification)
    assert specification["solver_options"]["epochs"] == 12
    assert specification["registry_sha256"] == registry_digest
    np.testing.assert_array_equal(attachment.result.target_loss_weights, [1.0])
    assert len(attachment.result.loss_trajectory) == 12


def test_added_person_and_group_fields_never_replace_input_context():
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()
    parent = parent.with_weights(
        "household",
        parent.weights_for("household"),
        mass=MassChange(factor=1.0, reason="invented input history"),
    )
    target = target.with_weights(
        "household",
        target.weights_for("household"),
        mass=MassChange(factor=1.0, reason="invented target history"),
    )
    target.person["derived_person"] = np.arange(6, dtype=np.float64)
    target.table("tax_unit")["derived_group"] = np.arange(4, dtype=np.float64)
    assert parent.metadata != target.metadata
    assert parent.mass_log != target.mass_log
    attachment = builder._calibrate_native_input_frame(
        parent, target, registry, **kwargs
    )
    for entity, field in (("person", "derived_person"), ("tax_unit", "derived_group")):
        assert field in attachment.result.frame.table(entity)
        assert field not in attachment.frame.table(entity)
    assert attachment.frame.metadata == parent.metadata
    assert attachment.frame.mass_log[: len(parent.mass_log)] == parent.mass_log
    specification = json.loads(attachment.comparison_specification)
    binding = json.loads(attachment.binding)
    assert (
        specification["supplied_specification_sha256"]
        == hashlib.sha256(kwargs["calibration_specification"]).hexdigest()
    )
    assert (
        binding["calibration_specification_sha256"]
        == hashlib.sha256(attachment.comparison_specification).hexdigest()
    )


@pytest.mark.parametrize("change", ["strata", "metadata", "mass_log", "weights"])
def test_progress_mutation_of_target_context_or_weights_refuses(change):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()

    def mutate(event):
        if change == "strata":
            target.strata.iloc[0] = "changed"
        elif change == "metadata":
            target._metadata = parent.metadata
        elif change == "mass_log":
            target._mass_log = parent.with_weights(
                "household",
                parent.weights_for("household"),
                mass=MassChange(factor=1.0, reason="callback replacement"),
            ).mass_log
        else:
            target._weights["household"] = Weights(
                np.array([2.0, 2.0, 2.0]), parent.weights_for("household").kind
            )

    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_CHANGED"):
        builder._calibrate_native_input_frame(
            parent, target, registry, progress_callback=mutate, **kwargs
        )


@pytest.mark.parametrize("option,value", [("epochs", 0), ("l2_lambda", -1.0)])
def test_maintained_solver_option_refusals_remain_refusals(option, value):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()
    setattr(kwargs["args"], option, value)
    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_STRUCTURE_OR_RESULT"):
        builder._calibrate_native_input_frame(parent, target, registry, **kwargs)


@pytest.mark.parametrize(
    "change",
    ["dtype", "negative", "mass", "initial", "entity", "support", "target_cell"],
)
def test_malformed_solver_results_refuse(monkeypatch, change):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case()
    solve = builder._calibrate_fiscal_support

    def corrupted(*args, **options):
        result, description = solve(*args, **options)
        if change in ("dtype", "negative", "mass"):
            weights = result.weights.copy()
            if change == "dtype":
                weights = weights.astype(np.float32)
            elif change == "negative":
                weights[0] = -1
            else:
                weights *= 2
            result = replace(result, weights=weights)
        elif change == "initial":
            result = replace(
                result, initial_weights=result.initial_weights[::-1].copy()
            )
        elif change == "entity":
            result = replace(result, weight_entity="person")
        elif change == "support":
            result.frame.table("household")["household_id"] = [30, 20, 10]
        else:
            result.frame.person.loc[0, "money"] = 919.0
        return result, description

    monkeypatch.setattr(builder, "_calibrate_fiscal_support", corrupted)
    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_"):
        builder._calibrate_native_input_frame(parent, target, registry, **kwargs)


def test_real_l0_sparse_vector_zeros_unselected_parent_positions():
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case(False)
    kwargs["args"].epochs = 100
    kwargs["args"].learning_rate = 0.1
    kwargs["args"].l0_refit_lambda_share = 0.3
    attachment = builder._calibrate_native_input_frame(
        parent, target, registry, **kwargs
    )
    selected = attachment.result.selected_mask
    assert selected.any() and not selected.all()
    np.testing.assert_array_equal(attachment.full_parent_weights[~selected], 0.0)
    np.testing.assert_array_equal(
        attachment.full_parent_weights[selected], attachment.result.weights
    )
    np.testing.assert_array_equal(
        attachment.frame.table("household").household_id,
        attachment.ordered_household_ids[selected],
    )


@pytest.mark.parametrize("change", ["dtype", "order", "mask", "initial"])
def test_l0_result_support_must_match_exact_parent_order(monkeypatch, change):
    builder = _load_builder_module()
    parent, target, registry, kwargs = _case(False)
    solve = builder._calibrate_fiscal_support

    def corrupted(*args, **options):
        result, description = solve(*args, **options)
        if change == "dtype":
            result = replace(
                result, selected_entity_ids=result.selected_entity_ids.astype(float)
            )
        elif change == "order":
            result = replace(
                result, selected_entity_ids=result.selected_entity_ids[::-1].copy()
            )
        elif change == "mask":
            result = replace(result, selected_mask=np.zeros(3, dtype=bool))
        else:
            result = replace(
                result, selection=replace(result.selection, initial_weights=np.ones(3))
            )
        return result, description

    monkeypatch.setattr(builder, "_calibrate_fiscal_support", corrupted)
    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_"):
        builder._calibrate_native_input_frame(parent, target, registry, **kwargs)
