"""``calibrate.ordered_adam@1``: a differential against ``calibrate.adam@1``.

Both kernels call the same public :func:`microcosm.calibrate.calibrate`. The
ordered kernel receives its targets as compiled CSR rows inside a portable
``ordered-problem`` artifact; ``calibrate.adam@1`` receives the same targets as
node parameters and compiles them itself. Byte-identical weights hold because
both paths build the constraint matrix from the same
:class:`~microcosm.calibrate.TargetSet` through
:func:`microcosm.calibrate.matrix.build_constraint_matrix`: the problem
artifact is that matrix serialized, and ``to_target_set`` hands each row back
unchanged, so the solver sees one matrix either way.

Invariants (Hypothesis property test):

- differential: equal targets give byte-identical weights on both kernels,
  and where the shared solver refuses an input, both kernels refuse it with
  the same error;
- weights are finite and non-negative;
- determinism: two runs of one context give identical weight and artifact bytes;
- the solution and result artifacts bind the problem's SHA-256 and replay the
  installed weights without solving.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from microcosm.calibrate import Target, TargetSet
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    decode_calibration_result,
    decode_problem,
    decode_solution,
    encode_problem,
)
from microcosm.calibrate.kernels import CALIBRATE_ADAM, _frame_from_context
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.ordered_kernels import (
    CALIBRATE_ORDERED_ADAM,
    OUTPUT_TYPES,
    CalibrateOrderedAdamKernel,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    Kernel,
    KernelContext,
    Node,
    Numeric,
    NumericScope,
    SeedSource,
    Slice,
    StructuralDelta,
    WeightTransition,
)

TARGET_PARAMS = (
    ("income", "income", None, 420.0, 7.25),
    ("eligible_income", "income", "eligible", 220.0, 3),
)
SOLVE = {"max_weight_ratio": 2.0, "epochs": 24, "learning_rate": 0.03}
_KEY = "a" * 64
_PRODUCER = "b" * 64
_OUTPUTS = tuple(ArtifactOutput(name, kind) for name, kind in OUTPUT_TYPES.items())


def _frame(
    income=(10.0, 25.0, 40.0, 70.0, 90.0, 120.0),
    eligible=(1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    weights=(1.0, 1.5, 0.75, 2.0, 1.25, 0.5),
    kind=WeightKind.DESIGN,
) -> Frame:
    ids = np.arange(len(income), dtype=np.int64)
    return Frame(
        {
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "income": np.asarray(income, dtype=np.float64),
                    "eligible": np.asarray(eligible, dtype=np.float64),
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights, dtype=np.float64), kind)},
    )


def _target_set(target_params=TARGET_PARAMS) -> TargetSet:
    return TargetSet(
        Target(
            name=name,
            entity="household",
            measure=measure,
            filter=filter_column,
            value=value,
        )
        for name, measure, filter_column, value, _se in target_params
    )


def _problem_payload(frame: Frame, target_params=TARGET_PARAMS) -> bytes:
    problem = build_constraint_matrix(frame, _target_set(target_params), "household")
    return encode_problem(
        problem,
        entity_ids=frame.table("household")["household_id"].tolist(),
    )


def _adam_node(target_params=TARGET_PARAMS, mass="conserve", **solve) -> Node:
    return Node(
        id="calibrate",
        kernel="calibrate.adam@1",
        inputs=(Slice("household", ("income", "eligible")),),
        params={**SOLVE, **solve, "targets": target_params, "mass": mass},
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "calibrated", mass=mass),
        mass=mass,
    )


def _ordered_node(mass="conserve", *, weight_anchor="design", **solve) -> Node:
    params = {**SOLVE, **solve, "mass": mass}
    if weight_anchor is not None:
        params["weight_anchor"] = weight_anchor
    return Node(
        id="calibrate",
        kernel="calibrate.ordered_adam@1",
        inputs=(Slice("household", ("income",)),),
        params=params,
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "calibrated", mass=mass),
        mass=mass,
        artifact_outputs=_OUTPUTS,
    )


def _context(
    frame: Frame,
    node: Node,
    *,
    columns=("income", "eligible"),
    problem: bytes | None = None,
    problem_type: ArtifactType = PROBLEM_TYPE,
) -> KernelContext:
    table = frame.table("household")[["household_id", *columns]]
    artifacts = {}
    if problem is not None:
        artifacts["problem"] = ArtifactValue(
            payload=problem,
            type=problem_type,
            key=_KEY,
            producer_key=_PRODUCER,
            numerics=NumericScope(),
        )
    return KernelContext(
        node=node,
        tables={"household": table.copy(deep=True)},
        weights={"household": frame.resolve_weights("household")},
        strata=frame.strata.copy(deep=True),
        params=node.params,
        rng=np.random.default_rng(987654321),
        artifacts=artifacts,
    )


def _run_pair(frame: Frame, target_params=TARGET_PARAMS, mass="conserve", **solve):
    adam = CALIBRATE_ADAM.run(
        _context(frame, _adam_node(target_params, mass=mass, **solve))
    )
    ordered = CALIBRATE_ORDERED_ADAM.run(
        _context(
            frame,
            _ordered_node(mass=mass, **solve),
            columns=("income",),
            problem=_problem_payload(frame, target_params),
        )
    )
    return adam, ordered


def test_ordered_adam_is_byte_identical_to_adam_on_the_same_targets() -> None:
    frame = _frame()
    adam, ordered = _run_pair(frame)

    assert ordered.weights is not None and adam.weights is not None
    assert ordered.weights.kind is WeightKind.CALIBRATED
    assert ordered.weights.values.tobytes() == adam.weights.values.tobytes()
    assert set(ordered.artifacts) == {"solution", "result"}
    assert ordered.columns == {}
    assert ordered.frame is None and ordered.keep is None


def test_ordered_adam_artifacts_bind_the_problem_and_replay_without_solving() -> None:
    frame = _frame()
    payload = _problem_payload(frame)
    problem = decode_problem(payload)
    result = CALIBRATE_ORDERED_ADAM.run(
        _context(frame, _ordered_node(), columns=("income",), problem=payload)
    )

    solution = decode_solution(
        result.artifacts["solution"],
        problem_sha256=problem.sha256,
        entity_ids=problem.entity_ids,
    )
    assert solution.weights.tobytes() == result.weights.values.tobytes()
    replay_frame = _frame_from_context(
        _context(frame, _ordered_node(), columns=("income",), problem=payload),
        "household",
    )
    rebuilt = decode_calibration_result(
        result.artifacts["result"], frame=replay_frame, problem=problem
    )
    assert rebuilt.weights.tobytes() == result.weights.values.tobytes()
    assert result.receipt["problem_sha256"] == problem.sha256
    assert (
        result.receipt["solution_sha256"]
        == hashlib.sha256(result.artifacts["solution"]).hexdigest()
    )
    assert result.receipt["n_targets"] == 2
    assert result.receipt["n_records"] == 6


def test_ordered_adam_declares_its_honest_capabilities() -> None:
    kernel = CalibrateOrderedAdamKernel()

    assert isinstance(kernel, Kernel)
    assert kernel.ref == "calibrate.ordered_adam@1"
    assert kernel.capabilities == Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )
    assert kernel.implementation_hash() == kernel.implementation_hash()
    # The ordered kernel is a different computation from adam@1 and must not
    # share its code identity, even though it binds everything adam@1 binds.
    assert kernel.implementation_hash() != CALIBRATE_ADAM.implementation_hash()


def test_ordered_adam_refuses_a_problem_compiled_on_another_axis_or_weights() -> None:
    frame = _frame()
    payload = _problem_payload(frame)

    shifted = _frame(weights=(1.0, 1.5, 0.75, 2.0, 1.25, 0.6))
    with pytest.raises(ValueError, match="starting weights"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(shifted, _ordered_node(), columns=("income",), problem=payload)
        )

    importance = _frame(kind=WeightKind.IMPORTANCE)
    with pytest.raises(ValueError, match="starting weights"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(importance, _ordered_node(), columns=("income",), problem=payload)
        )

    reordered_ids = _frame()
    table = reordered_ids.table("household").iloc[::-1].reset_index(drop=True)
    context = _context(frame, _ordered_node(), columns=("income",), problem=payload)
    context = KernelContext(
        node=context.node,
        tables={"household": table[["household_id", "income"]]},
        weights=context.weights,
        strata=context.strata,
        params=context.params,
        rng=context.rng,
        artifacts=context.artifacts,
    )
    with pytest.raises(ValueError, match="ordered 'household' axis"):
        CALIBRATE_ORDERED_ADAM.run(context)


def test_ordered_adam_refuses_skipped_targets_and_bad_artifacts() -> None:
    frame = _frame()
    params = (*TARGET_PARAMS, ("missing", "no_such_column", None, 5.0, None))
    skipped = _problem_payload(frame, params)
    assert decode_problem(skipped).problem.skipped
    with pytest.raises(ValueError, match="skipped"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, _ordered_node(), columns=("income",), problem=skipped)
        )

    with pytest.raises(ValueError, match="ordered-problem artifact"):
        CALIBRATE_ORDERED_ADAM.run(_context(frame, _ordered_node(), problem=None))

    with pytest.raises(ValueError, match="must be"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(
                frame,
                _ordered_node(),
                problem=_problem_payload(frame),
                problem_type=ArtifactType("not.a.problem", 1),
            )
        )

    with pytest.raises(ValueError, match="not an ordered problem archive"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, _ordered_node(), problem=b"PK\x03\x04 not a problem")
        )


def test_ordered_adam_refuses_unknown_params_and_an_unanchored_cap() -> None:
    frame = _frame()
    payload = _problem_payload(frame)
    node = _ordered_node()
    unknown = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params={**node.params, "targets": ()},
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=node.weights,
        mass=node.mass,
        artifact_outputs=_OUTPUTS,
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, unknown, columns=("income",), problem=payload)
        )

    with pytest.raises(ValueError, match="weight_anchor='design'"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(
                frame,
                _ordered_node(weight_anchor=None),
                columns=("income",),
                problem=payload,
            )
        )
    with pytest.raises(ValueError, match="weight_anchor"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(
                frame,
                _ordered_node(weight_anchor="importance"),
                columns=("income",),
                problem=payload,
            )
        )

    mismatched = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params={**node.params, "mass": "free"},
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=node.weights,
        mass=node.mass,
        artifact_outputs=_OUTPUTS,
    )
    with pytest.raises(ValueError, match="mass parameter must match"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, mismatched, columns=("income",), problem=payload)
        )

    undeclared = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params=node.params,
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=node.weights,
        mass=node.mass,
    )
    with pytest.raises(ValueError, match="typed outputs"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, undeclared, columns=("income",), problem=payload)
        )


def test_ordered_adam_requires_a_calibrated_transition_on_the_problem_entity() -> None:
    frame = _frame()
    payload = _problem_payload(frame)
    node = _ordered_node()
    to_importance = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params=node.params,
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("household", "importance", mass="conserve"),
        mass="conserve",
        artifact_outputs=_OUTPUTS,
    )
    with pytest.raises(ValueError, match="to_kind is 'calibrated'"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, to_importance, columns=("income",), problem=payload)
        )

    on_person = Node(
        id=node.id,
        kernel=node.kernel,
        inputs=node.inputs,
        params=node.params,
        structural=StructuralDelta.REWEIGHT,
        base="source",
        weights=WeightTransition("person", "calibrated", mass="conserve"),
        mass="conserve",
        artifact_outputs=_OUTPUTS,
    )
    with pytest.raises(ValueError, match="calibrates 'household'"):
        CALIBRATE_ORDERED_ADAM.run(
            _context(frame, on_person, columns=("income",), problem=payload)
        )


_positive = st.floats(
    min_value=0.25, max_value=50.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _cases(draw):
    n = draw(st.integers(min_value=2, max_value=9))
    income = draw(st.lists(_positive, min_size=n, max_size=n))
    eligible = draw(st.lists(st.sampled_from((0.0, 1.0)), min_size=n, max_size=n))
    if not any(eligible):
        eligible[0] = 1.0
    weights = draw(st.lists(_positive, min_size=n, max_size=n))
    income_total = float(np.dot(income, weights))
    eligible_total = float(np.dot(np.multiply(income, eligible), weights))
    # Wide enough that a cap of 1.5 often binds.
    scale_a = draw(st.floats(min_value=0.4, max_value=2.5))
    scale_b = draw(st.floats(min_value=0.4, max_value=2.5))
    targets = (
        ("income", "income", None, income_total * scale_a, None),
        ("eligible_income", "income", "eligible", eligible_total * scale_b, None),
    )
    mass = draw(st.sampled_from(("conserve", "free")))
    cap = draw(st.sampled_from((None, 1.5, 3.0)))
    epochs, learning_rate = draw(st.sampled_from(((6, 0.05), (24, 0.1))))
    return income, eligible, weights, targets, mass, cap, epochs, learning_rate


def _outcome(frame, targets, mass, solve):
    """Both kernels' results, or the exception each raised."""

    def capture(kernel, context):
        try:
            return kernel.run(context), None
        except ValueError as error:
            return None, error

    adam = capture(
        CALIBRATE_ADAM, _context(frame, _adam_node(targets, mass=mass, **solve))
    )
    ordered = capture(
        CALIBRATE_ORDERED_ADAM,
        _context(
            frame,
            _ordered_node(mass=mass, **solve),
            columns=("income",),
            problem=_problem_payload(frame, targets),
        ),
    )
    return adam, ordered


#: Found by Hypothesis: a target already met, ``mass="conserve"`` and a cap.
#: The shared solver's conserved total then drifts past its 1e-9 tolerance
#: (``with_weights(..., mass='conserve') violates mass conservation``). It is a
#: solver property outside this kernel; both kernels must fail identically.
_MASS_DRIFT = (
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.4356505826601342],
    (("income", "income", None, 2.4356505826601342, None),),
    "conserve",
    1.5,
    6,
    0.05,
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
@given(_cases())
@example(_MASS_DRIFT)
def test_property_differential_nonnegative_and_deterministic(case) -> None:
    income, eligible, weights, targets, mass, cap, epochs, learning_rate = case
    frame = _frame(income=income, eligible=eligible, weights=weights)
    solve = {"epochs": epochs, "learning_rate": learning_rate, "max_weight_ratio": cap}
    if cap is None:
        solve["weight_anchor"] = None
    (adam, adam_error), (ordered, ordered_error) = _outcome(frame, targets, mass, solve)
    if adam_error is not None or ordered_error is not None:
        assert adam_error is not None and ordered_error is not None
        assert type(adam_error) is type(ordered_error)
        assert str(adam_error) == str(ordered_error)
        event("both kernels raise")
        return
    values = ordered.weights.values

    # Differential: one matrix, one solver, one answer.
    assert values.tobytes() == adam.weights.values.tobytes()
    # Weights are finite and non-negative.
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    # The solver's own cap is relative to the problem's starting weights.
    if cap is not None:
        start = np.asarray(weights, dtype=np.float64)
        limit = np.nextafter(cap * start, np.inf)
        assert (values <= limit).all()
        if np.any(values >= cap * start * (1 - 1e-6)):
            event("solver cap binds")
    # Determinism: the same context gives the same bytes, artifacts included.
    _, again = _run_pair(frame, targets, mass=mass, **solve)
    assert again.weights.values.tobytes() == values.tobytes()
    assert again.artifacts == ordered.artifacts


@settings(max_examples=25, deadline=None, suppress_health_check=(HealthCheck.too_slow,))
@given(
    n=st.integers(min_value=2, max_value=9),
    data=st.data(),
)
def test_property_differential_holds_where_the_cap_binds(n, data) -> None:
    """Targets far above what a 1.5x cap allows: the cap must bind, identically."""

    income = data.draw(st.lists(_positive, min_size=n, max_size=n))
    weights = data.draw(st.lists(_positive, min_size=n, max_size=n))
    scale = data.draw(st.floats(min_value=2.0, max_value=4.0))
    total = float(np.dot(income, weights)) * scale
    frame = _frame(income=income, eligible=[1.0] * n, weights=weights)
    targets = (("income", "income", None, total, None),)
    solve = {"epochs": 40, "learning_rate": 0.2, "max_weight_ratio": 1.5}
    adam, ordered = _run_pair(frame, targets, mass="free", **solve)
    values = ordered.weights.values
    start = np.asarray(weights, dtype=np.float64)

    assert values.tobytes() == adam.weights.values.tobytes()
    assert (values <= np.nextafter(1.5 * start, np.inf)).all()
    # The cap binds: some weight sits at its limit.
    assert np.any(values >= 1.5 * start * (1 - 1e-4))
