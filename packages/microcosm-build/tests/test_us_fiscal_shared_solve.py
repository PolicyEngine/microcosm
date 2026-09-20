"""The shared release solve preserves the maintained calibrators and support."""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_fiscal_refresh_builder import _load_builder_module

from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind


def _args(dense):
    return argparse.Namespace(
        exact_k=None,
        dense_default_dataset=dense,
        epochs=12,
        learning_rate=0.03,
        max_weight_ratio=4.0,
        seed=17,
        l2_lambda=0.01,
        refit_l2_lambda=0.02,
        l0_refit_lambda_share=0.01,
    )


def _registry():
    return TargetRegistry(
        (
            TargetSpec(
                name="invented.income",
                entity="person",
                measure="income",
                value=750_000.0,
                source="Invented fixture",
            ),
        ),
        country="us",
    )


@pytest.mark.parametrize("dense", [True, False])
def test_shared_solve_matches_maintained_solver_and_preserves_parent(
    small_frame, dense
):
    builder = _load_builder_module()
    args = _args(dense)
    registry = _registry()
    before = {e: small_frame.table(e).copy(deep=True) for e in small_frame.entities}
    initial = small_frame.weights_for("household").values.copy()
    loss_weights = np.asarray([1.7])
    warm = np.asarray([1100.0, 1900.0]) if dense else None
    events = []
    result, metadata = builder._calibrate_fiscal_support(
        small_frame,
        registry,
        args=args,
        target_loss_weights=loss_weights,
        warm_start_weights=warm,
        progress_callback=events.append,
    )
    common = dict(
        epochs=12,
        learning_rate=0.03,
        max_weight_ratio=4.0,
        seed=17,
        mass="conserve",
        l2_lambda=0.01,
        target_loss_weights=loss_weights,
        target_loss_cap=builder.US_FISCAL_TARGET_LOSS_CAP,
        warm_start_weights=warm,
    )
    if dense:
        expected = builder.calibrate(small_frame, registry.to_target_set(), **common)
    else:
        expected = builder.calibrate_l0_refit(
            small_frame,
            registry.to_target_set(),
            refit_epochs=12,
            l0_lambda=0.005,
            refit_l2_lambda=0.02,
            **common,
        )
    np.testing.assert_array_equal(result.weights, expected.weights)
    assert result.final_loss == expected.final_loss
    assert events
    assert metadata["method"] == ("dense_no_l0" if dense else "l0_refit")
    assert metadata["n_candidate_households"] == 2
    json.dumps(metadata, allow_nan=False)
    for entity, table in before.items():
        pd.testing.assert_frame_equal(small_frame.table(entity), table)
    np.testing.assert_array_equal(small_frame.weights_for("household").values, initial)
    assert small_frame.weights_for("household").kind == WeightKind.DESIGN

    class LeafMetadata:
        def _engine_computed_columns(self, tables, *, period):
            return set()

    if dense:
        exported = builder._with_calibrated_weights(
            small_frame, result.weights, formula_metadata=LeafMetadata()
        )
        ids = small_frame.table("household")["household_id"].to_numpy()
    else:
        exported = builder._with_l0_refit_weights(
            small_frame, result, formula_metadata=LeafMetadata()
        )
        ids = result.selected_entity_ids
    np.testing.assert_array_equal(exported.table("household")["household_id"], ids)
    np.testing.assert_array_equal(
        exported.weights_for("household").values, result.weights
    )
    assert exported.weights_for("household").kind == WeightKind.CALIBRATED
    assert set(exported.table("person").columns) == set(before["person"].columns)
    assert set(exported.table("household").columns) == set(before["household"].columns)


def test_sparse_failure_losses_still_write_strict_diagnostics(monkeypatch, small_frame):
    builder = _load_builder_module()
    result = SimpleNamespace(
        selection=SimpleNamespace(n_nonzero=1, final_loss=float("inf")),
        frame=small_frame,
        l0_lambda=0.005,
        initial_loss=float("nan"),
        final_loss=float("inf"),
    )
    monkeypatch.setattr(builder, "calibrate_l0_refit", lambda *a, **kw: result)
    _, metadata = builder._calibrate_fiscal_support(
        small_frame,
        _registry(),
        args=_args(False),
        target_loss_weights=np.asarray([1.0]),
        warm_start_weights=None,
    )
    assert metadata["selection_final_loss"] is None
    assert metadata["refit_initial_loss"] is None
    assert metadata["refit_final_loss"] is None
    json.dumps(metadata, allow_nan=False)


def test_exact_k_cannot_silently_enter_ordinary_solve(monkeypatch, small_frame):
    builder = _load_builder_module()
    args = _args(False)
    args.exact_k = 1

    def forbidden(*a, **kw):
        raise AssertionError("ordinary solve must not run")

    monkeypatch.setattr(builder, "calibrate", forbidden)
    monkeypatch.setattr(builder, "calibrate_l0_refit", forbidden)
    with pytest.raises(ValueError, match="Exact-k"):
        builder._calibrate_fiscal_support(
            small_frame,
            _registry(),
            args=args,
            target_loss_weights=np.asarray([1.0]),
            warm_start_weights=None,
        )


def test_sparse_warm_start_keeps_existing_refusal(small_frame):
    builder = _load_builder_module()
    with pytest.raises(ValueError, match="warm_start_weights is not supported"):
        builder._calibrate_fiscal_support(
            small_frame,
            _registry(),
            args=_args(False),
            target_loss_weights=np.asarray([1.0]),
            warm_start_weights=np.asarray([1100.0, 1900.0]),
        )
