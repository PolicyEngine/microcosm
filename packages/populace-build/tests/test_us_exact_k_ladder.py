from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.exact_k_ladder import (
    calibrate_exact_k_ladder,
    exact_k_ladder_manifest_payload,
)
from populace.calibrate import Target, TargetSet
from populace.frame import EntitySchema, Frame, WeightKind, Weights


def _fixture_pool() -> tuple[Frame, TargetSet]:
    schema = EntitySchema(group_entities=("household",))
    weights = np.arange(1.0, 9.0)
    measure = np.asarray([0.0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 20.0])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": range(8),
                    "person_household_id": range(8),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": range(8),
                    "fixture_measure": measure,
                }
            ),
        },
        schema,
        {"household": Weights(weights, WeightKind.IMPORTANCE)},
    )
    targets = TargetSet(
        (
            Target(
                name="fixture_measure",
                entity="household",
                value=float(weights @ measure) * 0.9,
                measure="fixture_measure",
            ),
        )
    )
    return frame, targets


@pytest.mark.parametrize(
    ("ladder_point", "k", "expected_design"),
    (
        ("N", 8, "full-pool"),
        ("57,240 fixture analogue", 6, "sampford"),
        ("20,000 fixture analogue", 4, "sampford"),
    ),
)
def test_each_ladder_point_selects_refits_and_emits_round_trip_receipt(
    ladder_point: str,
    k: int,
    expected_design: str,
) -> None:
    frame, targets = _fixture_pool()

    outcome = calibrate_exact_k_ladder(
        frame,
        targets,
        k=k,
        pi_hi=1.0,
        seed=17,
        epochs=3,
        refit_epochs=3,
        learning_rate=0.02,
        max_weight_ratio=20.0,
        l0_lambda=1e-8,
    )

    assert ladder_point
    assert outcome.support.shape == (k,)
    assert outcome.result.frame.n("household") == k
    assert outcome.result.frame.weights_for("household").kind is WeightKind.CALIBRATED
    assert outcome.selection_receipt["design"] == expected_design
    assert outcome.selection_receipt["k"] == k
    assert set(outcome.selection_receipt) == {
        "k",
        "pi_hi",
        "seed",
        "certainty_count",
        "boundary_pool_size",
        "design",
    }
    assert json.loads(json.dumps(outcome.selection_receipt)) == (
        outcome.selection_receipt
    )
    assert outcome.refit_baseline_diagnostics["pool_weight_total"] == 36.0
    assert outcome.refit_baseline_diagnostics["refit_baseline_weight_total"] == 36.0

    original = frame.weights_for("household").values[outcome.support]
    q = outcome.selected_inclusion_probabilities
    expected_baseline = original / q
    expected_baseline *= 36.0 / expected_baseline.sum()
    np.testing.assert_allclose(
        outcome.result.initial_weights,
        expected_baseline,
        rtol=1e-12,
        atol=1e-12,
    )
    manifest_receipt = exact_k_ladder_manifest_payload(
        outcome,
        k=k,
        seed=17,
        pool={
            "release_id": "fixture-pool",
            "manifest_sha256": "a" * 64,
        },
        agreement_gate_reference={
            "passed": True,
            "diagnostics_sha256": "b" * 64,
        },
        frozen_target_register={
            "target_surface_sha256": "c" * 64,
            "incumbent_diagnostics_sha256": "d" * 64,
        },
    )
    round_trip = json.loads(json.dumps(manifest_receipt))
    assert round_trip["selection_receipt"] == outcome.selection_receipt
    assert round_trip["k"] == k
    assert round_trip["seed"] == 17
    assert round_trip["pool"]["manifest_sha256"] == "a" * 64
    assert round_trip["agreement_gate_reference"]["passed"] is True
    assert round_trip["frozen_target_register"]["target_surface_sha256"] == "c" * 64


def test_full_pool_support_is_identity_but_weights_are_refit() -> None:
    frame, targets = _fixture_pool()

    outcome = calibrate_exact_k_ladder(
        frame,
        targets,
        k=8,
        pi_hi=0.95,
        seed=99,
        epochs=3,
        l0_lambda=1e-8,
    )

    np.testing.assert_array_equal(outcome.support, np.arange(8))
    np.testing.assert_array_equal(
        outcome.result.frame.table("household")["household_id"],
        frame.table("household")["household_id"],
    )
    assert outcome.selection_receipt == {
        "k": 8,
        "pi_hi": 0.95,
        "seed": 99,
        "certainty_count": 8,
        "boundary_pool_size": 0,
        "design": "full-pool",
    }
    assert not np.array_equal(
        outcome.result.weights,
        frame.weights_for("household").values,
    )
    assert (
        outcome.refit_baseline_diagnostics["method"]
        == "full_pool_original_frame_weights"
    )


def test_manifest_refuses_requested_realized_count_mismatch() -> None:
    """The r1 k=8 receipt/frame.n=7 fault injection never reaches naming."""
    outcome = SimpleNamespace(
        result=SimpleNamespace(frame=SimpleNamespace(n=lambda entity: 7)),
        selection_receipt={
            "k": 8,
            "pi_hi": 0.95,
            "seed": 17,
            "certainty_count": 8,
            "boundary_pool_size": 0,
            "design": "full-pool",
        },
        refit_baseline_diagnostics={},
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "ExactKRealizedCountMismatchError: requested/realized household "
            "count mismatch: requested=8, realized=7"
        ),
    ):
        exact_k_ladder_manifest_payload(
            outcome,
            k=8,
            seed=17,
            pool={},
            agreement_gate_reference={},
            frozen_target_register={},
        )


def test_ladder_calibration_rejects_invalid_cardinality_before_selection() -> None:
    frame, targets = _fixture_pool()

    with pytest.raises(ValueError, match="k=9 exceeds the pool size 8"):
        calibrate_exact_k_ladder(
            frame,
            targets,
            k=9,
            pi_hi=0.95,
            seed=0,
            l0_lambda=1e-8,
        )

    with pytest.raises(ValueError, match="requires a positive finite l0_lambda"):
        calibrate_exact_k_ladder(
            frame,
            targets,
            k=4,
            pi_hi=0.95,
            seed=0,
            l0_lambda=0.0,
        )
