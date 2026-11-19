"""``measure.combine@1`` (G8-r2c hunk C).

The kernel sums declared signed integer-cent components on one entity. Every
column name, the floor and the scale here are synthetic declarations; the
kernel reads them from its declaration and holds none of them.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import fsum

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.reference_intervals import UNIVERSE_STATES
from microcosm.build.transport import compose
from microcosm.build.transport.measure_kernels import (
    MEASURE_COMBINE,
    MEASURE_COMBINE_REF,
    validate_measure_declaration,
)
from microcosm.build.transport.registry import register_transport_population_kernels
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import KernelRegistry, Node, Owned, Slice
from microcosm.graph.canonical import canonical_json
from microcosm.graph.executor import Population, _project_context
from test_support.microcosm_build.transport_population import digest

SCHEMA = EntitySchema(group_entities=("household",))
INT64_MAX = int(np.iinfo(np.int64).max)
INT64_MIN = int(np.iinfo(np.int64).min)
OUTPUTS = {
    "measure": "toy_measure_cents",
    "unit": "toy_measure_units",
    "universe_status": "toy_measure_universe_status",
}
PROPERTY = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)


def _declaration(components, add_backs=(), *, floor=0, scale=100, outputs=None):
    return {
        "entity": "person",
        "components": [{"column": c, "sign": s} for c, s in components],
        "add_backs": list(add_backs),
        "floor": floor,
        "scale": scale,
        "outputs": dict(outputs or OUTPUTS),
    }


def _frame(columns: dict[str, list], dtype: str, extra=None) -> Frame:
    size = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, size + 1, dtype=np.int64),
            "person_household_id": np.arange(1, size + 1, dtype=np.int64),
        }
    )
    for name, values in columns.items():
        person[name] = pd.array(values, dtype=dtype)
    for name, values in (extra or {}).items():
        person[name] = values
    household = pd.DataFrame({"household_id": np.arange(1, size + 1, dtype=np.int64)})
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"household": Weights(np.linspace(1, 2, size), WeightKind.DESIGN)},
    )


def _node(declaration, *, measure_dtype="Int64", measurement_id=None, params=None):
    text = canonical_json(declaration).decode()
    outputs = declaration["outputs"]
    node_params = {"measure": text, "measure_sha256": digest(declaration)}
    if measurement_id is not None:
        node_params["measurement_id"] = measurement_id
    node_params.update(params or {})
    return Node(
        "toy.measure",
        MEASURE_COMBINE_REF,
        inputs=(
            Slice(
                declaration["entity"],
                tuple(row["column"] for row in declaration["components"]),
            ),
        ),
        outputs=(
            Owned("person", outputs["measure"], measure_dtype),
            Owned("person", outputs["unit"], "float64"),
            Owned("person", outputs["universe_status"], "string"),
        ),
        params=node_params,
    )


def _run(columns, declaration, *, dtype="Int64", **node):
    frame = _frame(columns, dtype)
    context = _project_context(
        _node(declaration, **node),
        Population.from_frame(frame, "toy.population"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    result = MEASURE_COMBINE.run(context)
    out = {role: result.columns["person", name] for role, name in OUTPUTS.items()}
    return out, result.receipt


def _measure(out):
    return [None if pd.isna(v) else int(v) for v in out["measure"].tolist()]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

WAGES = "toy_wages_cents"
BUSINESS = "toy_business_cents"
RENT = "toy_rent_cents"
DEDUCTION = "toy_deduction_cents"


def test_losses_net_against_gains() -> None:
    out, receipt = _run(
        {WAGES: [10_000, 0, 500], BUSINESS: [-5_000, -200, 0]},
        _declaration([(WAGES, 1), (BUSINESS, 1)]),
    )
    assert _measure(out) == [5_000, -200, 500]
    assert out["universe_status"].tolist() == ["included", "excluded", "included"]
    assert receipt["components"][1]["unweighted_signed_total"] == -5_200


def test_a_negative_sign_subtracts() -> None:
    out, _ = _run(
        {WAGES: [10_000, 100], DEDUCTION: [2_500, 300]},
        _declaration([(WAGES, 1), (DEDUCTION, -1)]),
    )
    assert _measure(out) == [7_500, -200]


def test_the_add_back_ignores_an_add_back_components_losses() -> None:
    out, receipt = _run(
        {WAGES: [10_000, 10_000, 0], RENT: [-3_000, 3_000, -1]},
        _declaration([(WAGES, 1), (RENT, 1)], [RENT]),
    )
    assert _measure(out) == [10_000, 13_000, 0]
    assert receipt["add_backs"] == [{"column": RENT, "total": 3_001}]
    assert out["universe_status"].tolist() == ["included", "included", "excluded"]


def test_records_at_or_below_zero_sit_outside_the_positive_universe() -> None:
    out, receipt = _run(
        {WAGES: [0, -1, 1, -250]},
        _declaration([(WAGES, 1)]),
    )
    assert out["universe_status"].tolist() == [
        "excluded",
        "excluded",
        "included",
        "excluded",
    ]
    # The unit column is max(floor, x) / scale: excluded records sit at the floor.
    assert out["unit"].tolist() == [0.0, 0.0, 0.01, 0.0]
    assert receipt["below_zero"]["n"] == 2
    assert receipt["at_zero"]["n"] == 1


def test_the_receipt_weights_each_record_by_its_frame_weight() -> None:
    """The weighted totals and masses use each record's own weight."""

    out, receipt = _run(
        {
            WAGES: [10_000, 0, 500, 300, 0],
            BUSINESS: [-5_000, -200, 0, -300, 0],
        },
        _declaration([(WAGES, 1), (BUSINESS, 1)]),
    )
    assert _measure(out) == [5_000, -200, 500, 0, 0]
    # _frame weights its five households (one person each) 1, 1.25, 1.5, 1.75
    # and 2; every record here is resolved.
    w = [1.0, 1.25, 1.5, 1.75, 2.0]
    assert np.linspace(1, 2, 5).tolist() == w
    wages, business = receipt["components"]
    assert wages["weighted_signed_total"] == fsum(
        [w[0] * 10_000, w[1] * 0, w[2] * 500, w[3] * 300, w[4] * 0]
    )
    assert wages["weighted_signed_total"] == 11_275.0
    assert business["weighted_signed_total"] == fsum(
        [w[0] * -5_000, w[1] * -200, w[2] * 0, w[3] * -300, w[4] * 0]
    )
    assert business["weighted_signed_total"] == -5_775.0
    # x < 0 is record 1 alone; x = 0 is records 3 and 4.
    assert receipt["below_zero"] == {"n": 1, "weighted": w[1]}
    assert receipt["at_zero"] == {"n": 2, "weighted": fsum([w[3], w[4]])}
    assert receipt["at_zero"]["weighted"] == 3.75


def test_the_receipt_masses_count_resolved_records_only() -> None:
    """An unresolved record joins neither ``below_zero`` nor ``at_zero``."""

    out, receipt = _run({WAGES: [0, None, -5, None, 7]}, _declaration([(WAGES, 1)]))
    assert _measure(out) == [0, None, -5, None, 7]
    # _frame weights the five records 1, 1.25, 1.5, 1.75 and 2. Records 1 and 3
    # are unresolved, so x < 0 is record 2 alone and x = 0 is record 0 alone.
    assert receipt["below_zero"] == {"n": 1, "weighted": 1.5}
    assert receipt["at_zero"] == {"n": 1, "weighted": 1.0}


def test_an_int64_overflow_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="int64_overflow"):
        _run(
            {WAGES: [INT64_MAX, 1], BUSINESS: [1, 1]},
            _declaration([(WAGES, 1), (BUSINESS, 1)]),
        )
    with pytest.raises(ValueError, match="int64_overflow"):
        _run(
            {WAGES: [INT64_MIN, 1], BUSINESS: [1, 1]},
            _declaration([(WAGES, 1), (BUSINESS, -1)]),
        )


def test_a_null_component_leaves_the_record_unresolved_not_zero() -> None:
    out, receipt = _run(
        {WAGES: [100, None, 0], BUSINESS: [0, 50, None]},
        _declaration([(WAGES, 1), (BUSINESS, 1)]),
    )
    assert _measure(out) == [100, None, None]
    assert out["universe_status"].tolist() == ["included", "unresolved", "unresolved"]
    assert np.isnan(out["unit"].to_numpy()[1:]).all()
    assert receipt["universe_statuses"] == {
        "included": 1,
        "excluded": 0,
        "unresolved": 2,
    }
    assert [row["n_null"] for row in receipt["components"]] == [1, 1]
    with pytest.raises(ValueError, match="declare the measure Int64"):
        _run(
            {WAGES: [100, None]},
            _declaration([(WAGES, 1)]),
            measure_dtype="int64",
        )


def test_a_non_null_int64_measure_is_written_as_int64() -> None:
    out, _ = _run(
        {WAGES: [3, -4]},
        _declaration([(WAGES, 1)]),
        dtype="int64",
        measure_dtype="int64",
    )
    assert out["measure"].dtype == np.dtype("int64")
    assert out["measure"].tolist() == [3, -4]


def test_the_floor_and_scale_come_from_the_declaration() -> None:
    out, receipt = _run(
        {WAGES: [0, 1_000, 5_000, -7]},
        _declaration([(WAGES, 1)], floor=2_000, scale=1_000),
    )
    assert out["unit"].tolist() == [2.0, 2.0, 5.0, 2.0]
    assert (receipt["floor"], receipt["scale"]) == (2_000, 1_000)


def test_the_measurement_id_is_carried_not_used() -> None:
    columns, declaration = {WAGES: [5, -5]}, _declaration([(WAGES, 1)])
    out, receipt = _run(columns, declaration, measurement_id="toy_measurement_v1")
    bare, bare_receipt = _run(columns, declaration)
    assert receipt["measurement_id"] == "toy_measurement_v1"
    assert bare_receipt["measurement_id"] is None
    for role in OUTPUTS:
        pd.testing.assert_series_equal(out[role], bare[role])


@pytest.mark.parametrize(
    "node",
    [
        {"measurement_id": 123},
        {"measurement_id": ""},
        {"measurement_id": True},
        {"params": {"measurement_id": None}},
    ],
    ids=["integer", "empty", "boolean", "explicit_null"],
)
def test_a_measurement_id_that_is_not_a_non_empty_string_is_refused(node) -> None:
    with pytest.raises(TypeError) as refusal:
        _run({WAGES: [5, -5]}, _declaration([(WAGES, 1)]), **node)
    assert str(refusal.value) == (
        f"{MEASURE_COMBINE_REF} parameter 'measurement_id' must be a non-empty string."
    )


def test_the_receipt_names_the_declared_outputs() -> None:
    _, receipt = _run({WAGES: [1]}, _declaration([(WAGES, 1)]))
    assert receipt["outputs"] == OUTPUTS
    assert receipt["entity"] == "person"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"floor": 0.5}, "floor must be an integer"),
        ({"floor": True}, "floor must be an integer"),
        ({"floor": INT64_MAX + 1}, "floor must fit int64"),
        ({"floor": INT64_MIN - 1}, "floor must fit int64"),
        ({"scale": 0}, "scale must be a positive integer"),
        ({"scale": 1.0}, "scale must be a positive integer"),
        ({"add_backs": ["toy_other_cents"]}, "add_backs must name"),
        ({"components": []}, "components are a nonempty list"),
        ({"components": [{"column": WAGES, "sign": 2}]}, "sign"),
        (
            {"components": [{"column": WAGES, "sign": 1}] * 2},
            "must be distinct",
        ),
        ({"outputs": {**OUTPUTS, "unit": OUTPUTS["measure"]}}, "distinct measure"),
        ({"outputs": {**OUTPUTS, "unit": WAGES}}, "none of them a component"),
        ({"outputs": {"measure": "a", "unit": "b"}}, "distinct measure"),
    ],
)
def test_malformed_declarations_are_refused(change, message) -> None:
    declaration = {**_declaration([(WAGES, 1)]), **change}
    with pytest.raises(ValueError, match=message):
        validate_measure_declaration(declaration)
    missing = _declaration([(WAGES, 1)])
    del missing["floor"]
    with pytest.raises(ValueError, match="declares exactly"):
        validate_measure_declaration(missing)


def _run_node(node, frame) -> None:
    context = _project_context(
        node,
        Population.from_frame(frame, "toy.population"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    MEASURE_COMBINE.run(context)


def _reshaped(node, **change) -> Node:
    fields = {
        "inputs": node.inputs,
        "outputs": node.outputs,
        "params": node.params,
        **change,
    }
    return Node(node.id, node.kernel, **fields)


def test_node_shape_refusals() -> None:
    declaration = _declaration([(WAGES, 1)])
    with pytest.raises(ValueError, match="measure_sha256 differs"):
        _run({WAGES: [1]}, declaration, params={"measure_sha256": "0" * 64})
    with pytest.raises(ValueError, match="unknown parameter"):
        _run({WAGES: [1]}, declaration, params={"floor": 0})
    with pytest.raises(ValueError, match="is a new column of dtype"):
        _run({WAGES: [1]}, declaration, measure_dtype="float64")
    frame = _frame({WAGES: [1], BUSINESS: [2]}, "Int64")
    node = _node(declaration)
    wider = Node(
        node.id,
        node.kernel,
        inputs=(Slice("person", (WAGES, BUSINESS)),),
        outputs=node.outputs,
        params=node.params,
    )
    context = _project_context(
        wider,
        Population.from_frame(frame, "toy.population"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    with pytest.raises(ValueError, match="slice exactly the declared components"):
        MEASURE_COMBINE.run(context)

    # A row mask would combine a subset of the entity: whole rows only.
    frame = _frame({WAGES: [1, 2]}, "Int64", {"toy_mask": np.array([True, False])})
    masked = _reshaped(
        node, inputs=(Slice("person", (WAGES, "toy_mask"), rows="toy_mask"),)
    )
    with pytest.raises(ValueError, match="requires whole-row slices"):
        _run_node(masked, frame)

    # The outputs are exactly the declaration's three columns.
    frame = _frame({WAGES: [1, 2]}, "Int64")
    extra = _reshaped(
        node, outputs=(*node.outputs, Owned("person", "toy_extra_cents", "Int64"))
    )
    renamed = _reshaped(
        node,
        outputs=(
            node.outputs[0],
            Owned("person", "toy_other_units", "float64"),
            node.outputs[2],
        ),
    )
    for item in (extra, renamed):
        with pytest.raises(ValueError, match="outputs must equal the declaration"):
            _run_node(item, frame)

    # Each output is a new column; the stand-in never rewrites one.
    frame = _frame({WAGES: [1, 2]}, "Int64", {OUTPUTS["unit"]: np.zeros(2)})
    rewrite = _reshaped(
        node,
        outputs=(
            node.outputs[0],
            Owned("person", OUTPUTS["unit"], "float64", rewrite=True),
            node.outputs[2],
        ),
    )
    with pytest.raises(ValueError, match="is a new column of dtype"):
        _run_node(rewrite, frame)


def test_a_float_component_is_refused() -> None:
    with pytest.raises(ValueError, match="integer minor-unit column"):
        _run({WAGES: [1.0, 2.0]}, _declaration([(WAGES, 1)]), dtype="float64")


def test_measure_combine_is_registered_in_the_skeleton() -> None:
    registry = register_transport_population_kernels(KernelRegistry())
    assert registry.get(MEASURE_COMBINE_REF) is MEASURE_COMBINE
    assert MEASURE_COMBINE_REF in compose.SKELETON_KERNELS


# ---------------------------------------------------------------------------
# Properties and the differential
# ---------------------------------------------------------------------------

_CENTS = st.one_of(
    st.integers(min_value=-(10**12), max_value=10**12),
    st.integers(min_value=-(2**60), max_value=2**60),
    st.none(),
)


@st.composite
def measures(draw):
    width = draw(st.integers(min_value=1, max_value=6))
    size = draw(st.integers(min_value=1, max_value=12))
    names = [f"toy_component_{index}_cents" for index in range(width)]
    signs = [draw(st.sampled_from((1, -1))) for _ in names]
    backs = [name for name in names if draw(st.booleans())]
    columns = {name: [draw(_CENTS) for _ in range(size)] for name in names}
    floor = draw(st.integers(min_value=-(10**6), max_value=10**6))
    scale = draw(st.integers(min_value=1, max_value=10**4))
    return columns, list(zip(names, signs, strict=True)), backs, floor, scale


def _independent(columns, components, backs):
    """The measure by plain Python integers, independently of the kernel."""
    size = len(next(iter(columns.values())))
    result = []
    for index in range(size):
        cells = {name: columns[name][index] for name, _ in components}
        if any(value is None for value in cells.values()):
            result.append(None)
            continue
        total = 0
        for name, sign in components:
            total += sign * cells[name]
        for name, sign in components:
            if name in backs and sign * cells[name] < 0:
                total += -(sign * cells[name])
        result.append(total)
    return result


def _run_or_overflow(columns, components, backs, floor=0, scale=100):
    try:
        return _run(columns, _declaration(components, backs, floor=floor, scale=scale))
    except ValueError as error:
        assert "int64_overflow" in str(error)
        expected = _independent(columns, components, backs)
        assert any(v is not None and not INT64_MIN <= v <= INT64_MAX for v in expected)
        return None


@PROPERTY
@given(case=measures())
def test_property_differential_against_an_independent_integer_sum(case) -> None:
    columns, components, backs, _, _ = case
    result = _run_or_overflow(columns, components, backs)
    if result is None:
        return
    out, _ = result
    assert _measure(out) == _independent(columns, components, backs)


@PROPERTY
@given(case=measures())
def test_property_components_plus_add_back_equal_the_measure(case) -> None:
    columns, components, backs, _, _ = case
    result = _run_or_overflow(columns, components, backs)
    if result is None:
        return
    out, receipt = result
    measure = _measure(out)
    resolved = [m for m in measure if m is not None]
    assert receipt["sum_measure"] == sum(resolved)
    assert receipt["sum_measure"] == sum(
        row["unweighted_signed_total"] for row in receipt["components"]
    ) + sum(row["total"] for row in receipt["add_backs"])
    for index, value in enumerate(measure):
        if value is None:
            continue
        signed = [sign * columns[name][index] for name, sign in components]
        back = sum(
            max(0, -(sign * columns[name][index]))
            for name, sign in components
            if name in backs
        )
        assert sum(signed) + back == value


@PROPERTY
@given(case=measures())
def test_property_the_universe_statuses_partition_the_records(case) -> None:
    columns, components, backs, _, _ = case
    result = _run_or_overflow(columns, components, backs)
    if result is None:
        return
    out, receipt = result
    statuses = out["universe_status"].tolist()
    assert set(statuses) <= set(UNIVERSE_STATES)
    assert sum(receipt["universe_statuses"].values()) == len(statuses)
    for value, status in zip(_measure(out), statuses, strict=True):
        expected = (
            "unresolved" if value is None else "included" if value > 0 else "excluded"
        )
        assert status == expected


@PROPERTY
@given(case=measures(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_property_the_output_does_not_depend_on_component_order(case, seed) -> None:
    columns, components, backs, _, _ = case
    result = _run_or_overflow(columns, components, backs)
    if result is None:
        return
    shuffled = list(components)
    random.Random(seed).shuffle(shuffled)
    other, _ = _run(columns, _declaration(shuffled, list(reversed(backs))))
    out, _ = result
    for role in OUTPUTS:
        pd.testing.assert_series_equal(out[role], other[role])


@PROPERTY
@given(case=measures())
def test_property_the_unit_is_max_floor_x_over_scale(case) -> None:
    columns, components, backs, floor, scale = case
    result = _run_or_overflow(columns, components, backs, floor, scale)
    if result is None:
        return
    out, _ = result
    for value, unit in zip(_measure(out), out["unit"].tolist(), strict=True):
        if value is None:
            assert np.isnan(unit)
        else:
            assert unit == float(Fraction(max(floor, value), scale))
