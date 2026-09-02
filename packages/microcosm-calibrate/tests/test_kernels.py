"""Byte-level parity for the legacy calibration graph kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import Target, TargetSet, calibrate, diagnostics_payload
from microcosm.calibrate.kernels import CALIBRATE_ADAM, CalibrateAdamKernel
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    Determinism,
    Kernel,
    KernelContext,
    Node,
    Numeric,
    SeedSource,
    Slice,
    StructuralDelta,
    WeightTransition,
)

TARGET_PARAMS = (
    ("income", "income", None, 420.0, 7.25),
    ("eligible_income", "income", "eligible", 220.0, 3),
)
SOLVER_PARAMS = {
    "targets": TARGET_PARAMS,
    "max_weight_ratio": 2.0,
    "epochs": 24,
    "learning_rate": 0.03,
    "mass": "conserve",
}


def _fixture_frame() -> Frame:
    income = np.asarray([10.0, 25.0, 40.0, 70.0, 90.0, 120.0])
    eligible = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    weights = np.asarray([1.0, 1.5, 0.75, 2.0, 1.25, 0.5])
    ids = np.arange(len(income), dtype=np.int64)
    return Frame(
        {
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "income": income,
                    "eligible": eligible,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.IMPORTANCE)},
    )


def _node(
    *,
    to_kind: str = "calibrated",
    transition_mass: str = "conserve",
    parameter_mass: str = "conserve",
) -> Node:
    return Node(
        id="calibrate",
        kernel="calibrate.adam@1",
        inputs=(Slice("household", ("income", "eligible")),),
        params={**SOLVER_PARAMS, "mass": parameter_mass},
        weights=WeightTransition(
            "household",
            to_kind,
            mass=transition_mass,
        ),
    )


def _context(frame: Frame, node: Node) -> KernelContext:
    table = frame.table("household")[["household_id", "income", "eligible"]]
    return KernelContext(
        node=node,
        tables={"household": table.copy(deep=True)},
        weights={"household": frame.resolve_weights("household")},
        strata=frame.strata.copy(deep=True),
        params=node.params,
        rng=np.random.default_rng(987654321),
    )


def _direct(frame: Frame):
    targets = TargetSet(
        Target(
            name=name,
            entity="household",
            measure=measure,
            filter=filter_column,
            value=value,
        )
        for name, measure, filter_column, value, _se in TARGET_PARAMS
    )
    return calibrate(
        frame,
        targets,
        weight_entity="household",
        method="adam",
        max_weight_ratio=SOLVER_PARAMS["max_weight_ratio"],
        epochs=SOLVER_PARAMS["epochs"],
        learning_rate=SOLVER_PARAMS["learning_rate"],
        mass=SOLVER_PARAMS["mass"],
        seed=0,
    )


def test_calibrate_adam_has_byte_parity_with_the_public_call() -> None:
    frame = _fixture_frame()
    context = _context(frame, _node())
    visible_before = context.tables["household"].copy(deep=True)
    direct = _direct(frame)

    wrapped = CALIBRATE_ADAM.run(context)

    assert wrapped.weights is not None
    assert wrapped.weights.kind is WeightKind.CALIBRATED
    assert wrapped.weights.values.tobytes() == direct.weights.tobytes()
    np.testing.assert_array_equal(wrapped.weights.values, direct.weights)
    assert wrapped.columns == {}
    assert wrapped.frame is None
    assert wrapped.artifacts == {}
    assert wrapped.receipt["declared_targets"] == TARGET_PARAMS
    assert wrapped.receipt["declared_targets"][1][4] == 3
    assert isinstance(wrapped.receipt["declared_targets"][1][4], int)
    assert wrapped.receipt["diagnostics"] == diagnostics_payload(direct)
    pd.testing.assert_frame_equal(context.tables["household"], visible_before)


def test_calibrate_adam_preserves_none_standard_error_unchanged() -> None:
    frame = _fixture_frame()
    node = _node()
    params = dict(node.params)
    params["targets"] = (TARGET_PARAMS[0], (*TARGET_PARAMS[1][:-1], None))
    node = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params=params,
        weights=node.weights,
    )

    result = CALIBRATE_ADAM.run(_context(frame, node))

    assert result.receipt["declared_targets"] == params["targets"]
    assert result.receipt["declared_targets"][1][4] is None


def test_calibrate_adam_declares_its_honest_capabilities() -> None:
    kernel = CalibrateAdamKernel()

    assert isinstance(kernel, Kernel)
    assert kernel.ref == "calibrate.adam@1"
    assert kernel.capabilities == Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )
    assert kernel.implementation_hash() == kernel.implementation_hash()


@pytest.mark.parametrize("to_kind", ["design", "importance"])
def test_calibrate_adam_requires_a_calibrated_weight_transition(
    to_kind: str,
) -> None:
    frame = _fixture_frame()
    context = _context(frame, _node(to_kind=to_kind))

    with pytest.raises(ValueError, match="to_kind is 'calibrated'"):
        CALIBRATE_ADAM.run(context)


def test_calibrate_adam_requires_mass_to_match_the_transition() -> None:
    frame = _fixture_frame()
    context = _context(
        frame,
        _node(transition_mass="free", parameter_mass="conserve"),
    )

    with pytest.raises(ValueError, match="mass parameter must match"):
        CALIBRATE_ADAM.run(context)


def test_calibrate_adam_rejects_unknown_params_and_invalid_standard_errors() -> None:
    frame = _fixture_frame()
    node = _node()
    unknown_params = {**node.params, "unused": 1}
    unknown_node = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params=unknown_params,
        weights=node.weights,
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        CALIBRATE_ADAM.run(_context(frame, unknown_node))

    invalid_params = dict(node.params)
    invalid_params["targets"] = ((*TARGET_PARAMS[0][:-1], 0.0), TARGET_PARAMS[1])
    invalid_node = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params=invalid_params,
        weights=node.weights,
    )
    with pytest.raises(ValueError, match="standard error must be positive"):
        CALIBRATE_ADAM.run(_context(frame, invalid_node))
