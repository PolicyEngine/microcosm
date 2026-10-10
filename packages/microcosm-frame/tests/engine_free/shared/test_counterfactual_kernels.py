"""Engine-free bridge properties and differentials against direct engines."""

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateSolveZeroKernel,
)
from microcosm.graph import KernelRegistry, Node, Owned, Slice
from test_support.microcosm_frame.kernels import _context
from tools import transport_bridge_differential as differential_tool

SCHEMA = EntitySchema(group_entities=("family",))
RATE_REF = "toy-rates"
CREDIT_REF = "toy-credit"
POPULATIONS = st.lists(
    st.tuples(
        st.integers(0, 100),
        st.integers(1, 100),
        st.integers(1, 100),
        st.booleans(),
    ),
    min_size=1,
    max_size=5,
)


class RatesEngine:
    """Rates live in this toy engine; bridge code has no policy formula."""

    def __init__(self):
        self.calls = 0

    def variable_metadata(self, name):
        if name != "weekly_benefit":
            raise ValueError(name)
        return VariableMetadata(name, "person", "float", "year")

    def variables(self):
        return ("income", "rate", "reduction", "receives")

    def entity_schema(self):
        return SCHEMA

    def materialize(self, bundle, variables: Sequence[str], period):
        self.calls += 1
        person = bundle.table("person")
        values = np.maximum(
            person["rate"].to_numpy()
            - person["income"].to_numpy() * person["reduction"].to_numpy(),
            0,
        ).astype(np.float64)
        return {variable: values for variable in variables}

    def export_contract(self):
        return ExportContract.empty()

    def write_dataset(self, bundle, path: str | Path, period):
        return None


class CreditEngine(RatesEngine):
    def variable_metadata(self, name):
        if name != "eldest_credit":
            raise ValueError(name)
        return VariableMetadata(name, "family", "float", "year")

    def variables(self):
        return ("credit",)

    def materialize(self, bundle, variables, period):
        self.calls += 1
        return {
            variable: bundle.table("family")["credit"].to_numpy(dtype=np.float64)
            for variable in variables
        }


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _frame(population):
    persons, families = [], []
    for offset, (income, rate, reduction, receives) in enumerate(population):
        group = 100 + offset
        # Reverse the person ids within a group: "first" must use ids, not
        # incidental input row order. Group tables retain Frame's sorted ids.
        for member in (1, 0):
            persons.append(
                {
                    "person_id": 10 * group + member,
                    "person_family_id": group,
                    "income": float(income + member),
                    "rate": float(rate + member),
                    "reduction": float(reduction),
                    "receives": receives if member == 0 else False,
                }
            )
        families.append({"family_id": group, "credit": float(rate * 2)})
    return Frame(
        {"person": pd.DataFrame(persons), "family": pd.DataFrame(families)},
        SCHEMA,
        {"family": Weights(np.ones(len(families)), WeightKind.DESIGN)},
    )


def _node(kernel, **changes):
    params = {
        "engine_ref": RATE_REF,
        "period": "2026-27",
        "variables": ("weekly_benefit",),
        "input_overrides": _json(
            [{"entity": "person", "column": "income", "value": 0}]
        ),
        "output_coefficients": _json(
            [
                {
                    "variable": "weekly_benefit",
                    "coefficient": 1,
                    "entity": "family",
                    "column": "bridge",
                    "aggregation": "sum",
                }
            ]
        ),
        "resource_sha256": "a" * 64,
    }
    if isinstance(kernel, SimulateSolveZeroKernel):
        params.update(
            input_overrides="[]",
            solve_input=_json({"entity": "person", "column": "income"}),
            bracket=(0, 256),
            tolerance=1e-7,
            iterations=34,
        )
    params.update(changes)
    return Node(
        "bridge",
        kernel.ref,
        inputs=(
            Slice("person", ("income", "rate", "reduction", "receives")),
            Slice("family", ("credit",)),
        ),
        outputs=(Owned("family", "bridge", "float64"),),
        params=params,
    )


def _run(kernel, frame, node):
    return kernel.run(_context(frame, node, weighted_entities=("family",)))


@settings(max_examples=25, deadline=None)
@given(POPULATIONS, st.sampled_from(("sum", "max", "first")))
def test_counterfactual_matches_direct_engine_and_declared_reduction(
    population, aggregation
):
    frame = _frame(population)
    original = frame.table("person").copy(deep=True)
    engine = RatesEngine()
    kernel = SimulateCounterfactualKernel({RATE_REF: engine})
    term = {
        "variable": "weekly_benefit",
        "coefficient": 2,
        "entity": "family",
        "column": "bridge",
        "aggregation": aggregation,
    }
    node = _node(kernel, output_coefficients=_json([term]))
    result = _run(kernel, frame, node)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["income"] = 0
    direct = engine.materialize(
        Frame(tables, SCHEMA, {"family": frame.weights_for("family")}),
        ("weekly_benefit",),
        "2026-27",
    )["weekly_benefit"]
    expected = []
    for group in frame.table("family")["family_id"]:
        positions = np.flatnonzero(original["person_family_id"].to_numpy() == group)
        if aggregation == "first":
            position = min(positions, key=lambda item: original["person_id"].iloc[item])
            expected.append(direct[position] * 2)
        else:
            expected.append(getattr(direct[positions], aggregation)() * 2)
    series = result.columns[("family", "bridge")]
    np.testing.assert_array_equal(series.to_numpy(), expected)
    assert series.index.tolist() == frame.table("family")["family_id"].tolist()
    assert series.dtype == np.dtype("float64")
    pd.testing.assert_frame_equal(frame.table("person"), original)
    repeated = _run(kernel, frame, node)
    assert (
        repeated.columns[("family", "bridge")].to_numpy().tobytes()
        == series.to_numpy().tobytes()
    )


@settings(max_examples=20, deadline=None)
@given(POPULATIONS)
@example([(0, 50, 1, True)])
def test_multiple_engines_masks_and_group_input_broadcast_equal_direct_calls(
    population,
):
    frame = _frame(population)
    rates, credit = RatesEngine(), CreditEngine()
    kernel = SimulateCounterfactualKernel({RATE_REF: rates, CREDIT_REF: credit})
    primary_overrides = [
        {
            "entity": "person",
            "column": "rate",
            "source_entity": "family",
            "source_column": "credit",
            "mask": {"entity": "person", "column": "receives", "equals": True},
        }
    ]
    terms = [
        {
            "variable": "weekly_benefit",
            "coefficient": 1,
            "entity": "family",
            "column": "bridge",
            "aggregation": "sum",
            "mask": {"entity": "person", "column": "receives", "equals": True},
        },
        {
            "engine_ref": CREDIT_REF,
            "variable": "eldest_credit",
            "coefficient": 0.5,
            "entity": "family",
            "column": "bridge",
        },
    ]
    node = _node(
        kernel,
        input_overrides=_json(primary_overrides),
        output_coefficients=_json(terms),
        components=_json(
            [
                {
                    "engine_ref": CREDIT_REF,
                    "period": "2026-27",
                    "variables": ["eldest_credit"],
                    "input_overrides": [
                        {"entity": "family", "column": "credit", "value": 4}
                    ],
                }
            ]
        ),
    )
    result = _run(kernel, frame, node)
    person = frame.table("person").copy()
    family = frame.table("family").copy()
    lookup = family.set_index("family_id")["credit"]
    person.loc[person["receives"], "rate"] = (
        person.loc[person["receives"], "person_family_id"].map(lookup).to_numpy()
    )
    direct_frame = Frame(
        {"person": person, "family": family},
        SCHEMA,
        {"family": frame.weights_for("family")},
    )
    direct_rate = rates.materialize(direct_frame, ("weekly_benefit",), "2026-27")[
        "weekly_benefit"
    ]
    family["credit"] = 4
    direct_credit = credit.materialize(
        Frame(
            {"person": person, "family": family},
            SCHEMA,
            {"family": frame.weights_for("family")},
        ),
        ("eldest_credit",),
        "2026-27",
    )["eldest_credit"]
    expected = []
    for position, group in enumerate(family["family_id"]):
        mask = (person["person_family_id"].to_numpy() == group) & person[
            "receives"
        ].to_numpy()
        expected.append(direct_rate[mask].sum() + direct_credit[position] * 0.5)
    np.testing.assert_array_equal(result.columns[("family", "bridge")], expected)
    assert frame.table("family")["credit"].tolist() == [
        item[1] * 2 for item in population
    ]


@settings(max_examples=15, deadline=None)
@given(POPULATIONS)
def test_fixed_vector_bisection_matches_analytic_cutout_and_is_deterministic(
    population,
):
    frame = _frame(population)
    engine = RatesEngine()
    kernel = SimulateSolveZeroKernel({RATE_REF: engine})
    node = _node(kernel)
    result = _run(kernel, frame, node)
    cutouts = result.columns[("family", "bridge")].to_numpy()
    person = frame.table("person")
    direct = person["rate"].to_numpy() / person["reduction"].to_numpy()
    expected = np.asarray(
        [
            direct[person["person_family_id"].to_numpy() == group].max()
            for group in frame.table("family")["family_id"]
        ]
    )
    assert np.all(cutouts >= expected - np.spacing(expected))
    assert np.all(cutouts - expected <= node.params["tolerance"])
    assert engine.calls == node.params["iterations"] + 3
    assert result.receipt["evaluations"] == engine.calls
    repeated = _run(kernel, frame, node)
    assert (
        cutouts.tobytes() == repeated.columns[("family", "bridge")].to_numpy().tobytes()
    )


def test_cutout_override_uses_declared_reference_case_instead_of_current_benefit():
    frame = _frame([(10, 5, 1, True)])
    kernel = SimulateSolveZeroKernel({RATE_REF: RatesEngine()})
    node = _node(
        kernel,
        input_overrides=_json(
            [
                {"entity": "person", "column": "rate", "value": 100},
                {"entity": "person", "column": "reduction", "value": 2},
            ]
        ),
    )
    result = _run(kernel, frame, node)
    assert result.columns[("family", "bridge")].iloc[0] == 50


def test_nonpositive_lower_endpoint_is_returned_with_fixed_evaluation_count():
    frame = _frame([(0, 1, 1, False)])
    engine = RatesEngine()
    kernel = SimulateSolveZeroKernel({RATE_REF: engine})
    result = _run(kernel, frame, _node(kernel, bracket=(10, 256)))
    assert result.columns[("family", "bridge")].iloc[0] == 10
    assert engine.calls == 37
    assert result.receipt["evaluations"] == engine.calls


def test_solver_evaluates_final_nonzero_answer_on_zero_plateau():
    class RecordingEngine(RatesEngine):
        def materialize(self, bundle, variables, period):
            self.last_inputs = bundle.table("person")["income"].to_numpy().copy()
            self.last_residuals = super().materialize(bundle, variables, period)
            return self.last_residuals

    frame = _frame([(10, 5, 1, True)])
    engine = RecordingEngine()
    kernel = SimulateSolveZeroKernel({RATE_REF: engine})
    node = _node(
        kernel,
        input_overrides=_json(
            [
                {"entity": "person", "column": "rate", "value": 100},
                {"entity": "person", "column": "reduction", "value": 2},
            ]
        ),
    )
    result = _run(kernel, frame, node)
    answer = result.columns[("family", "bridge")].iloc[0]
    assert answer == 50
    np.testing.assert_array_equal(engine.last_inputs, np.full(2, answer))
    np.testing.assert_array_equal(engine.last_residuals["weekly_benefit"], [0, 0])
    assert engine.calls == node.params["iterations"] + 3
    assert result.receipt["evaluations"] == engine.calls


def test_solver_refuses_coupled_rows_with_positive_final_residual():
    class CoupledEngine(RatesEngine):
        def materialize(self, bundle, variables, period):
            self.calls += 1
            person = bundle.table("person")
            family = bundle.table("family")
            first_incomes = person.groupby("person_family_id")["income"].first()
            total_income = first_incomes.sum()
            residuals = pd.Series(
                np.maximum(np.asarray([100, 50]) - total_income, 0),
                index=family["family_id"],
            )
            values = person["person_family_id"].map(residuals).to_numpy()
            return {variable: values for variable in variables}

    frame = _frame([(0, 100, 1, True), (0, 50, 1, True)])
    engine = CoupledEngine()
    kernel = SimulateSolveZeroKernel({RATE_REF: engine})
    node = _node(kernel)
    with pytest.raises(ValueError, match="final answer.*positive residual"):
        _run(kernel, frame, node)
    assert engine.calls == node.params["iterations"] + 3


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"bracket": (0, 1)}, "positive residual"),
        ({"iterations": 1}, "tolerance"),
        ({"tolerance": 0}, "positive tolerance"),
        (
            {"solve_input": _json({"entity": "person", "column": "not_an_input"})},
            "engine input",
        ),
    ],
)
def test_solver_refuses_unbracketed_or_inadequately_declared_roots(changes, match):
    frame = _frame([(10, 50, 1, True)])
    kernel = SimulateSolveZeroKernel({RATE_REF: RatesEngine()})
    with pytest.raises(ValueError, match=match):
        _run(kernel, frame, _node(kernel, **changes))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"engine_ref": "missing"}, "no engine bound"),
        (
            {
                "input_overrides": _json(
                    [{"entity": "person", "column": "not_an_input", "value": 0}]
                )
            },
            "engine input",
        ),
        (
            {
                "output_coefficients": _json(
                    [
                        {
                            "variable": "weekly_benefit",
                            "coefficient": 1,
                            "entity": "family",
                            "column": "bridge",
                        }
                    ]
                )
            },
            "aggregation",
        ),
        ({"resource_sha256": "no digest"}, "SHA-256"),
        ({"components": "[{}]"}, "Each component"),
        ({"output_coefficients": "NaN"}, "finite JSON"),
    ],
)
def test_bridge_refuses_unbound_or_undeclared_inputs(changes, match):
    frame = _frame([(10, 50, 1, True)])
    kernel = SimulateCounterfactualKernel({RATE_REF: RatesEngine()})
    with pytest.raises(ValueError, match=match):
        _run(kernel, frame, _node(kernel, **changes))


def test_adding_same_adapter_binding_does_not_change_bridge_implementation_hash():
    for cls in (SimulateCounterfactualKernel, SimulateSolveZeroKernel):
        first = cls({RATE_REF: RatesEngine()})
        second = cls({RATE_REF: RatesEngine(), "unused": RatesEngine()})
        assert first.implementation_hash() == second.implementation_hash()


@pytest.mark.parametrize(
    "returned,match", [([1], "wrong shape"), ([np.nan, 1], "finite")]
)
def test_bridge_refuses_malformed_engine_outputs(returned, match):
    class MalformedEngine(RatesEngine):
        def materialize(self, bundle, variables, period):
            return {variable: np.asarray(returned) for variable in variables}

    frame = _frame([(10, 50, 1, True)])
    kernel = SimulateCounterfactualKernel({RATE_REF: MalformedEngine()})
    with pytest.raises(ValueError, match=match):
        _run(kernel, frame, _node(kernel))


@pytest.mark.parametrize(
    "entity,column",
    [
        ("person", "person_id"),
        ("person", "person_family_id"),
        ("family", "family_id"),
    ],
)
def test_solver_rejects_structural_trial_inputs_before_calling_engine(entity, column):
    class StructuralInputEngine(RatesEngine):
        def variables(self):
            return (*super().variables(), column)

    frame = _frame([(10, 50, 1, True)])
    original = {name: frame.table(name).copy() for name in frame.entities}
    engine = StructuralInputEngine()
    kernel = SimulateSolveZeroKernel({RATE_REF: engine})
    node = _node(kernel, solve_input=_json({"entity": entity, "column": column}))
    with pytest.raises(ValueError, match="solve_input cannot change structural"):
        _run(kernel, frame, node)
    assert engine.calls == 0
    for name in frame.entities:
        pd.testing.assert_frame_equal(frame.table(name), original[name])


@pytest.mark.parametrize("expected,status", [(101, "passed"), (100, "failed")])
def test_hub_differential_receipt_runs_kernel_and_compares_external_oracle(
    monkeypatch, expected, status
):
    frame = _frame([(10, 50, 1, True)])
    kernel = SimulateCounterfactualKernel({"prepared-reference": RatesEngine()})
    registry = KernelRegistry()
    registry.register(kernel)
    prepared = SimpleNamespace(
        schema=SCHEMA, kernels=registry, engine_refs={RATE_REF: "prepared-reference"}
    )
    monkeypatch.setattr(
        differential_tool.BenefitUnitRule, "from_dict", lambda data: object()
    )
    monkeypatch.setattr(
        differential_tool, "build_transport_registry", lambda *args, **kwargs: prepared
    )
    node = _node(kernel)
    case = {
        "unit_rule": {},
        "rules_bindings": {},
        "tables": {
            entity: frame.table(entity).to_dict("records") for entity in frame.entities
        },
        "weights": {"family": {"kind": "design", "values": [1]}},
        "calls": [
            {
                "id": node.id,
                "kernel": node.kernel,
                "inputs": {item.entity: list(item.columns) for item in node.inputs},
                "outputs": [
                    {"entity": "family", "column": "bridge", "dtype": "float64"}
                ],
                "params": dict(node.params),
            }
        ],
        "comparisons": [
            {
                "call": node.id,
                "entity": "family",
                "column": "bridge",
                "oracle_pointer": "/requests/0/expected/weekly",
                "tolerance": 0,
            }
        ],
    }
    receipt = differential_tool.differential(
        case, {"requests": [{"expected": {"weekly": expected}}]}, Path("unused")
    )
    assert receipt["status"] == status
    assert receipt["comparisons"][0]["actual"] == [101]
    assert receipt["comparisons"][0]["expected"] == [expected]


def test_hub_differential_skips_without_engine_before_reading_case(
    monkeypatch, tmp_path
):
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(
        differential_tool.importlib.util, "find_spec", lambda name: None
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "transport_bridge_differential.py",
            "--case",
            "missing-case.json",
            "--oracle",
            "missing-oracle.json",
            "--rulespec-root",
            "missing-root",
            "--out",
            str(receipt_path),
        ],
    )
    assert differential_tool.main() == 0
    assert json.loads(receipt_path.read_text())["status"] == "skipped"


def test_hub_binding_resolution_preserves_case_for_repeated_runs():
    params = {
        "engine_ref": "rates",
        "variables": ["benefit"],
        "components": [
            {
                "engine_ref": "credit",
                "variables": ["credit_amount"],
                "period": "2026-27",
                "input_overrides": [],
            }
        ],
    }
    original = _json(params)
    first = differential_tool._resolve_params(
        params, {"rates": "ref-a", "credit": "ref-b"}
    )
    second = differential_tool._resolve_params(
        params, {"rates": "ref-a", "credit": "ref-b"}
    )
    assert _json(params) == original
    assert first == second
    assert json.loads(first["components"])[0]["engine_ref"] == "ref-b"
