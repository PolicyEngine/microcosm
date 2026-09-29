"""Portable matrix/solution artifacts retain axes and reject corrupt inputs."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.calibrate import CalibrationProblem, Target, build_constraint_matrix
from microcosm.calibrate.artifacts import (
    decode_calibration_result,
    decode_problem,
    decode_solution,
    encode_calibration_result,
    encode_problem,
    encode_solution,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def problem():
    targets = (
        Target("count", "household", lambda f: np.ones(2), value=10, period=2024),
        Target("money", "household", "money", value=21, period="2025"),
    )
    return CalibrationProblem(
        sparse.csr_array([[1.0, 1.0], [2.0, 5.0]]),
        np.array([10.0, 21.0]),
        tuple(t.row_name for t in targets),
        Weights(np.array([2.0, 3.0]), WeightKind.IMPORTANCE),
        "household",
        targets,
    )


def test_problem_roundtrip_is_deterministic_and_recompiles_bound_rows():
    payload = encode_problem(
        problem(), entity_ids=(10, 20), bindings={"selector": "all"}
    )
    assert (
        encode_problem(problem(), entity_ids=(10, 20), bindings={"selector": "all"})
        == payload
    )
    restored = decode_problem(payload)
    assert restored.entity_ids == (10, 20)
    assert restored.problem.names == problem().names
    assert restored.bindings == {"selector": "all"}
    np.testing.assert_array_equal(
        restored.problem.matrix.toarray(), problem().matrix.toarray()
    )
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1, 2], "person_household_id": [10, 20]}
            ),
            "household": pd.DataFrame({"household_id": [10, 20]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": problem().initial_weights},
        pd.Series(["a", "a"]),
    )
    recompiled = build_constraint_matrix(
        frame, restored.to_target_set(), weight_entity="household"
    )
    np.testing.assert_array_equal(
        recompiled.matrix.toarray(), problem().matrix.toarray()
    )


def test_artifact_axes_and_solution_binding_are_strict():
    with pytest.raises(ValueError, match="unique"):
        encode_problem(problem(), entity_ids=(10, 10))
    with pytest.raises(ValueError, match="axis"):
        encode_problem(problem(), entity_ids=(10,))
    bound = decode_problem(encode_problem(problem(), entity_ids=(10, 20)))
    payload = encode_solution(
        [4.0, 6.0],
        entity_ids=(10, 20),
        problem_sha256=bound.sha256,
        diagnostics={"converged": True},
    )
    result = decode_solution(payload, problem_sha256=bound.sha256, entity_ids=(10, 20))
    np.testing.assert_array_equal(result.weights, [4.0, 6.0])
    with pytest.raises(ValueError, match="axis"):
        decode_solution(payload, entity_ids=(20, 10))
    with pytest.raises(ValueError, match="problem"):
        decode_solution(payload, problem_sha256="0" * 64)
    with pytest.raises(ValueError):
        decode_problem(payload)


def test_problem_target_definitions_do_not_densify_the_whole_sparse_system(monkeypatch):
    bound = decode_problem(encode_problem(problem(), entity_ids=(10, 20)))
    # Country matrices can have millions of households and thousands of rows.
    # Creating definitions must not materialize any dense contribution row.
    monkeypatch.setattr(
        sparse.csr_array, "toarray", lambda *a, **kw: pytest.fail("eager densification")
    )
    assert len(bound.to_target_set().targets) == 2


def test_complete_result_rebuild_preserves_dense_and_search_state(monkeypatch):
    import microcosm.calibrate.solve as solve
    from microcosm.calibrate import calibrate

    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1, 2], "person_household_id": [10, 20]}
            ),
            "household": pd.DataFrame({"household_id": [10, 20]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": problem().initial_weights},
        pd.Series(["a", "a"]),
    )
    bound = decode_problem(encode_problem(problem(), entity_ids=(10, 20)))
    for penalty in (0.0, 0.01):
        result = calibrate(
            frame,
            bound.to_target_set(),
            epochs=3,
            seed=17,
            mass="free",
            l0_lambda=penalty,
        )
        payload = encode_calibration_result(
            result, entity_ids=(10, 20), problem_sha256=bound.sha256
        )
        with monkeypatch.context() as context:
            context.setattr(
                solve, "_optimize", lambda *a, **kw: pytest.fail("replay optimized")
            )
            restored = decode_calibration_result(payload, frame=frame, problem=bound)
        np.testing.assert_array_equal(restored.weights, result.weights)
        np.testing.assert_array_equal(restored.loss_trajectory, result.loss_trajectory)
        assert restored.options == result.options
        if penalty:
            np.testing.assert_array_equal(
                restored.gate_open_probabilities, result.gate_open_probabilities
            )
