"""Affine residuals, certified quantization and exact-rational differentials."""

import json
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.exact_money import _half_even, _shortest_decimal
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateSolveZeroKernel,
)
from microcosm.graph import Node, Owned, Slice
from test_support.microcosm_frame.kernels import _context

SCHEMA = EntitySchema(group_entities=("family",))
REF = "fixture-tax"
Q = Fraction(1, 100)


def _tax(gross, schedule):
    """Fixture data only: marginal rates apply above their lower threshold."""
    total = Fraction(0)
    for offset, (lower, rate) in enumerate(schedule):
        upper = schedule[offset + 1][0] if offset + 1 < len(schedule) else gross
        total += max(Fraction(0), min(gross, upper) - lower) * rate
    return total


def exact_inverse(net, schedule):
    """Independent Fraction inverse for continuous piecewise-linear fixtures."""
    for offset, (lower, rate) in enumerate(schedule):
        candidate = (net + _tax(lower, schedule) - lower * rate) / (1 - rate)
        if offset + 1 == len(schedule) or candidate <= schedule[offset + 1][0]:
            return candidate
    raise AssertionError("Schedule must have a final unbounded bracket.")


class ScheduleEngine:
    """A tax double that refuses metadata requests for its root input."""

    def __init__(self, schedule=((Fraction(0), Fraction(1, 5)),)):
        self.schedule = schedule
        self.calls = []
        self.metadata_requests = []

    def variables(self):
        return ("income",)

    def variable_metadata(self, name):
        self.metadata_requests.append(name)
        if name != "tax":
            raise ValueError("Input variables have no output metadata.")
        return VariableMetadata(name, "person", "float", "year")

    def entity_schema(self):
        return SCHEMA

    def materialize(self, bundle, variables, period):
        trial = bundle.table("person")["income"].to_numpy(dtype=np.float64)
        self.calls.append(trial.copy())
        tax = np.array(
            [_tax(_shortest_decimal(x), self.schedule) for x in trial], dtype=object
        )
        return {variable: tax for variable in variables}

    def export_contract(self):
        return ExportContract.empty()

    def write_dataset(self, bundle, path, period):
        return None


def _frame(nets):
    count = len(nets)
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(11, 11 + count, dtype=np.int64),
                    "person_family_id": np.arange(1, count + 1, dtype=np.int64),
                    "income": np.zeros(count),
                    "net": np.asarray(nets, dtype=np.float64),
                }
            ),
            "family": pd.DataFrame(
                {
                    "family_id": np.arange(1, count + 1, dtype=np.int64),
                    "family_net": np.asarray(nets, dtype=np.float64),
                }
            ),
        },
        SCHEMA,
        {"family": Weights(np.ones(count), WeightKind.DESIGN)},
    )


def _terms(source="person", coefficient=1):
    return [
        {
            "source_entity": source,
            "source_column": "family_net" if source == "family" else "net",
            "coefficient": coefficient,
            "entity": "person",
            "column": "gross",
        },
        {"variable": "tax", "coefficient": 1, "entity": "person", "column": "gross"},
        {"trial": True, "coefficient": -1, "entity": "person", "column": "gross"},
    ]


def _node(kernel, *, terms=None, quantized=True, **params):
    parameters = {
        "engine_ref": REF,
        "period": "2026-27",
        "variables": ("tax",),
        "input_overrides": "[]",
        "output_coefficients": json.dumps(_terms() if terms is None else terms),
    }
    if isinstance(kernel, SimulateSolveZeroKernel):
        parameters.update(
            solve_input=json.dumps({"entity": "person", "column": "income"}),
            bracket=(0, 1048576),
            tolerance=1e-7,
            iterations=44,
        )
        if quantized:
            parameters.update(
                quantum="0.01",
                tie_rule="half_even",
                quantized_residual_bound="0.00500001",
                extra_evaluation_budget=9,
                certificate_margin="0",
            )
    parameters.update(params)
    return Node(
        "gross-up",
        kernel.ref,
        inputs=(Slice("person", ("income", "net")), Slice("family", ("family_net",))),
        outputs=(Owned("person", "gross", "float64"),),
        params=parameters,
    )


def _run(engine, nets, *, frame=None, **params):
    kernel = SimulateSolveZeroKernel({REF: engine})
    node = _node(kernel, **params)
    result = kernel.run(
        _context(
            _frame(nets) if frame is None else frame,
            node,
            weighted_entities=("family",),
        )
    )
    return result, node


def _run_masked(
    engine, mask_values, *, source="person", override=False, frame=None, **params
):
    frame = _frame(np.ones(len(mask_values))) if frame is None else frame
    frame.table(source)["eligible"] = mask_values
    mask = {"entity": source, "column": "eligible", "equals": True}
    terms = _terms()
    if override:
        params["input_overrides"] = json.dumps(
            [{"entity": "person", "column": "income", "value": 0, "mask": mask}]
        )
    else:
        terms[1]["mask"] = mask
    kernel = SimulateSolveZeroKernel({REF: engine})
    original = _node(kernel, terms=terms, **params)
    node = Node(
        original.id,
        original.kernel,
        inputs=(
            *original.inputs,
            Slice(source, ("eligible",)),
        ),
        outputs=original.outputs,
        params=original.params,
    )
    return kernel.run(_context(frame, node, weighted_entities=("family",)))


def test_affine_source_trial_and_tax_use_engine_without_trial_output_metadata():
    engine = ScheduleEngine()
    result, _ = _run(engine, [80, 160])
    assert result.columns["person", "gross"].tolist() == [100, 200]
    assert set(engine.metadata_requests) == {"tax"}
    assert result.receipt["max_quantized_residual"] == 0


def test_affine_source_term_is_used_and_group_column_broadcasts():
    result, _ = _run(ScheduleEngine(), [2, 3], terms=_terms("family", coefficient=52))
    assert result.columns["person", "gross"].tolist() == [130, 195]


@pytest.mark.parametrize("override", [False, True], ids=["term", "override"])
@pytest.mark.parametrize("source,row", [("person", 11), ("family", 1)])
@pytest.mark.parametrize(
    "mask_values",
    [
        pd.Series([pd.NA], dtype="boolean"),
        pd.Series([pd.NA, True], dtype="boolean"),
        pd.Series([None, True], dtype=object),
        pd.Series([np.nan, True], dtype=object),
    ],
    ids=["minimized-nullable", "selective-nullable", "none", "nan"],
)
def test_quantized_missing_masks_name_only_rows_in_the_source_entity(
    mask_values, source, row, override
):
    with pytest.raises(
        ValueError,
        match=rf"mask source {source}\.eligible.*missing.*rows \[{row}\]\.$",
    ):
        _run_masked(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            mask_values,
            source=source,
            override=override,
        )


@pytest.mark.parametrize("override", [False, True], ids=["term", "override"])
@pytest.mark.parametrize("source,row", [("person", 11), ("family", 1)])
@pytest.mark.parametrize(
    "invalid",
    [
        Decimal("sNaN"),
        np.array([True]),
        np.array([True, False]),
        pd.Series([True, False]),
    ],
    ids=["signalling-decimal", "length-one-array-cell", "array-cell", "series-cell"],
)
def test_quantized_nonscalar_masks_name_only_the_bad_row(
    invalid, source, row, override
):
    with pytest.raises(
        ValueError,
        match=rf"mask source {source}\.eligible.*scalar.*rows \[{row}\]\.$",
    ):
        _run_masked(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            pd.Series([invalid, True], dtype=object),
            source=source,
            override=override,
        )


@pytest.mark.parametrize(
    "invalid",
    [TypeError, OverflowError, Decimal("sNaN")],
    ids=["typeerror", "overflow", "signalling-decimal"],
)
def test_quantized_mask_comparison_exceptions_name_only_the_bad_row(invalid):
    if isinstance(invalid, type):

        class InvalidComparison:
            def __eq__(self, other):
                raise invalid("Invalid mask cell comparison.")

        value = InvalidComparison()
    else:
        value = invalid
    with pytest.raises(
        ValueError,
        match=r"mask source person\.eligible.*scalar.*rows \[11\]\.$",
    ):
        _run_masked(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            pd.Series([value, True], dtype=object),
        )


def test_quantized_engine_term_respects_a_false_tax_mask():
    result = _run_masked(ScheduleEngine(), pd.Series([False], dtype=bool))
    # The fixture computes exact G/5 tax, but this row declares no tax term.
    assert result.columns["person", "gross"].tolist() == [1.00]
    assert result.receipt["max_quantized_residual"] == 0


@pytest.mark.parametrize("override", [False, True], ids=["term", "override"])
def test_legacy_nullable_mask_retains_its_typeerror(override):
    with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
        _run_masked(
            ScheduleEngine(),
            pd.Series([pd.NA, True], dtype="boolean"),
            override=override,
            quantized=False,
        )


def test_legacy_nan_mask_still_compares_unequal():
    result = _run_masked(
        ScheduleEngine(),
        pd.Series([np.nan, True], dtype=object),
        quantized=False,
    )
    assert result.columns["person", "gross"].tolist() == pytest.approx(
        [1.00, 1.25], abs=1e-7
    )


def test_invalid_person_outputs_name_every_affected_family():
    class InvalidMembers(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            trial = bundle.table("person")["income"].to_numpy()
            self.calls.append(trial.copy())
            return {"tax": np.array([Fraction(0), np.nan, np.nan, Fraction(0)])}

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [11, 12, 21, 22],
                    "person_family_id": [1, 1, 2, 2],
                    "income": np.zeros(4),
                }
            ),
            "family": pd.DataFrame({"family_id": [1, 2], "family_net": [1, 1]}),
        },
        SCHEMA,
        {"family": Weights(np.ones(2), WeightKind.DESIGN)},
    )
    terms = [
        {
            "source_entity": "family",
            "source_column": "family_net",
            "coefficient": 1,
            "entity": "family",
            "column": "gross",
        },
        {
            "variable": "tax",
            "coefficient": 1,
            "entity": "family",
            "column": "gross",
            "aggregation": "sum",
        },
        {"trial": True, "coefficient": -1, "entity": "family", "column": "gross"},
    ]
    kernel = SimulateSolveZeroKernel({REF: InvalidMembers()})
    original = _node(kernel, terms=terms)
    node = Node(
        original.id,
        original.kernel,
        inputs=(Slice("person", ("income",)), Slice("family", ("family_net",))),
        outputs=(Owned("family", "gross", "float64"),),
        params=original.params,
    )
    with pytest.raises(ValueError, match=r"engine output 'tax'.*rows \[1, 2\]\.$"):
        kernel.run(_context(frame, node, weighted_entities=("family",)))


@pytest.mark.parametrize("quantized", [True, False], ids=["quantized", "legacy"])
@pytest.mark.parametrize(
    "selective_lookup,rows",
    [(False, r"\[11, 12\]"), (True, r"\[12\]")],
    ids=["both-miss", "selective-lookup"],
)
def test_mapping_distinguishes_boolean_group_ids_from_integer_membership(
    quantized, selective_lookup, rows, monkeypatch
):
    frame = _frame([1, 2])
    frame.table("family")["family_id"] = [False, True]
    frame.table("person")["person_family_id"] = np.asarray([0, 1], dtype=np.int64)
    if quantized:
        kernel = SimulateSolveZeroKernel({REF: ScheduleEngine()})
        node = _node(kernel, terms=_terms("family"))
        context = _context(frame, node, weighted_entities=("family",))
        lookups = []
        if selective_lookup:
            # Native bool ids and integer memberships miss both rows. Control
            # this lookup to exercise only the guard's selective row naming.
            def selective_indexer(index, memberships):
                assert index.dtype == np.dtype(bool)
                assert index.tolist() == [False, True]
                assert np.array_equal(memberships, [0, 1])
                lookups.append(index)
                return np.asarray([0, -1], dtype=np.intp)

            monkeypatch.setattr(pd.Index, "get_indexer", selective_indexer)
        with pytest.raises(
            ValueError,
            match=rf"membership identifiers must match group ids.*rows {rows}\.$",
        ):
            kernel._prepare(context, solver=True)
        if selective_lookup:
            assert len(lookups) == 1
    else:
        result, _ = _run(
            ScheduleEngine(), [], frame=frame, terms=_terms("family"), quantized=False
        )
        assert result.columns["person", "gross"].tolist() == pytest.approx(
            [2.5, 2.5], abs=1e-7
        )


def test_quantized_membership_check_covers_every_group_entity():
    schema = EntitySchema(group_entities=("family", "household"))

    class TwoGroupEngine(ScheduleEngine):
        def entity_schema(self):
            return schema

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [101, 102],
                    "person_family_id": [1, 1],
                    "person_household_id": [0, 1],
                    "income": [0.0, 0.0],
                }
            ),
            "family": pd.DataFrame({"family_id": [1]}),
            # A later group's bool ids miss these integer memberships.
            "household": pd.DataFrame(
                {"household_id": [False, True], "household_net": [3.0, 7.0]}
            ),
        },
        schema,
        {"family": Weights(np.ones(1), WeightKind.DESIGN)},
    )
    terms = _terms()
    terms[0] = {
        **terms[0],
        "source_entity": "household",
        "source_column": "household_net",
    }
    kernel = SimulateSolveZeroKernel(
        {REF: TwoGroupEngine(((Fraction(0), Fraction(0)),))}
    )
    original = _node(kernel, terms=terms)
    node = Node(
        original.id,
        original.kernel,
        inputs=(Slice("person", ("income",)), Slice("household", ("household_net",))),
        outputs=original.outputs,
        params=original.params,
    )
    with pytest.raises(
        ValueError,
        match=r"membership identifiers must match group ids.*rows \[101, 102\]\.$",
    ):
        kernel.run(_context(frame, node, weighted_entities=("family",)))


def test_trial_as_engine_output_refuses_the_old_route():
    engine = ScheduleEngine()
    terms = _terms()
    terms[2] = {
        "variable": "income",
        "coefficient": -1,
        "entity": "person",
        "column": "gross",
    }
    with pytest.raises(ValueError, match="requested component variable"):
        _run(engine, [80], terms=terms)
    assert engine.calls == []


@pytest.mark.parametrize("term", [_terms()[0], _terms()[2]], ids=["source", "trial"])
def test_counterfactual_refuses_affine_terms(term):
    engine = ScheduleEngine()
    kernel = SimulateCounterfactualKernel({REF: engine})
    with pytest.raises(ValueError, match="Invalid output coefficient fields"):
        kernel.run(
            _context(
                _frame([80]), _node(kernel, terms=[term]), weighted_entities=("family",)
            )
        )
    assert engine.calls == []


@pytest.mark.parametrize(
    "term,match",
    [
        ({**_terms()[0], "source_column": "missing"}, "existing source"),
        ({**_terms()[0], "source_column": "person_id"}, "structural"),
        ({**_terms()[0], "source_column": "person_family_id"}, "structural"),
        ({**_terms()[0], "aggregation": "sum"}, "Invalid source"),
        (
            {**_terms()[0], "source_entity": "family", "source_column": "family_id"},
            "structural",
        ),
        ({**_terms()[2], "trial": 1}, "trial: true"),
        ({**_terms()[2], "mask": {}}, "Invalid trial"),
        ({**_terms()[2], "coefficient": True}, "finite number"),
        ({**_terms()[2], "coefficient": float("inf")}, "finite JSON"),
    ],
)
def test_affine_terms_have_closed_fields_and_finite_coefficients(term, match):
    engine = ScheduleEngine()
    with pytest.raises(ValueError, match=match):
        _run(engine, [80], terms=[term])
    assert engine.calls == []


@pytest.mark.parametrize("net", [float("nan"), float("inf"), float("-inf"), "bad"])
def test_source_columns_refuse_nonfinite_or_nonnumeric_values(net):
    frame = _frame([80])
    frame.table("person")["net"] = [net]
    engine = ScheduleEngine()
    with pytest.raises(ValueError, match="finite numbers"):
        _run(engine, [], frame=frame)
    assert engine.calls == []


@pytest.mark.parametrize(
    "params,match",
    [
        ({"quantum": 0.01}, "decimal text"),
        ({"quantum": "NaN"}, "plain decimal"),
        ({"quantum": "0"}, "positive"),
        ({"quantum": "-0.01"}, "plain decimal"),
        ({"quantum": "1e-2"}, "plain decimal"),
        ({"tie_rule": "half_up"}, "half_even"),
        ({"quantized_residual_bound": 1}, "decimal text"),
        ({"quantized_residual_bound": "Infinity"}, "plain decimal"),
        ({"certificate_margin": 0}, "decimal text"),
        ({"certificate_margin": -1}, "decimal text"),
        ({"certificate_margin": "-0.001"}, "plain decimal"),
        ({"certificate_margin": "NaN"}, "plain decimal"),
        ({"certificate_margin": "Infinity"}, "plain decimal"),
        ({"certificate_margin": "1e-6"}, "plain decimal"),
        ({"extra_evaluation_budget": 2}, "odd integer"),
        ({"extra_evaluation_budget": True}, "odd integer"),
        (
            {"quantum": "0.0100000000000000000001"},
            r"boundaries must round-trip.*rows \[11\]",
        ),
        ({"quantum": "1" + "0" * 400}, r"boundaries exceed float64; rows \[11\]"),
    ],
)
def test_quantization_params_are_explicit_exact_and_closed(params, match):
    with pytest.raises(ValueError, match=match):
        _run(ScheduleEngine(), [80], **params)


@pytest.mark.parametrize(
    "missing",
    [
        "quantum",
        "tie_rule",
        "quantized_residual_bound",
        "extra_evaluation_budget",
        "certificate_margin",
    ],
)
def test_quantization_requires_all_five_parameters(missing):
    engine = ScheduleEngine()
    kernel = SimulateSolveZeroKernel({REF: engine})
    node = _node(kernel)
    node = Node(
        node.id,
        node.kernel,
        inputs=node.inputs,
        outputs=node.outputs,
        params={key: value for key, value in node.params.items() if key != missing},
    )
    with pytest.raises(ValueError, match="together"):
        kernel.run(_context(_frame([80]), node, weighted_entities=("family",)))
    assert engine.calls == []


@pytest.mark.parametrize(
    "root,expected", [(1.005, 1.00), (1.015, 1.02), (1.025, 1.02), (1.035, 1.04)]
)
def test_exact_half_cent_ties_round_half_even(root, expected):
    result, _ = _run(ScheduleEngine(((Fraction(0), Fraction(0)),)), [root])
    assert result.columns["person", "gross"].iloc[0] == expected


@pytest.mark.parametrize("net,expected", [(0.804, 1.00), (0.812, 1.02), (0.82, 1.02)])
def test_nonzero_tax_exact_half_cent_ties_preserve_affine_decimal_arithmetic(
    net, expected
):
    # Binary float addition gives .804 + .201 - 1.005 > 0; the exact affine
    # decimal residual is zero and the cent index 100 must win this tie.
    result, _ = _run(ScheduleEngine(), [net])
    assert result.columns["person", "gross"].iloc[0] == expected


def test_weekly_annualisation_is_exact_before_half_cent_tie_certification():
    # Float .405 * 52 is 21.060000000000002; the exact annual net is 21.06
    # and the exact annual gross 26.325 ties to 26.32, not 26.33.
    result, _ = _run(ScheduleEngine(), [0.405], terms=_terms(coefficient=52))
    assert result.columns["person", "gross"].tolist() == [26.32]


def test_exact_affine_residual_refuses_float_sign_underflow():
    with pytest.raises(
        ValueError, match=r"exact coefficient sum underflows.*rows \[11\]"
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [1e-200],
            terms=_terms(coefficient=1e-200),
        )


def test_subnormal_exact_source_underflow_names_only_affected_rows():
    frame = _frame([0, 1])
    frame.table("person")["net"] = pd.Series(
        [Fraction(1, 10**324), Fraction(1)], dtype=object
    )
    with pytest.raises(
        ValueError, match=r"exact coefficient sum underflows float64; rows \[11\]"
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [],
            frame=frame,
            bracket=(0, 2),
            iterations=25,
        )


def test_exact_coefficient_overflow_names_only_affected_rows():
    terms = _terms(coefficient=1e308)
    terms[-1]["coefficient"] = -1e308
    with pytest.raises(
        ValueError, match=r"bridge coefficient sum must be finite; rows \[11\]"
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [2, 1],
            terms=terms,
            bracket=(0, 4),
            iterations=26,
        )


@pytest.mark.parametrize("quantized", [True, False])
def test_nonfinite_decimal_source_names_only_affected_rows(quantized):
    frame = _frame([0, 1])
    frame.table("person")["net"] = pd.Series(
        [Decimal("NaN"), Fraction(1)], dtype=object
    )
    with pytest.raises(
        ValueError,
        match=r"source coefficient columns must contain finite numbers; rows \[11\]",
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [],
            frame=frame,
            bracket=(0, 2),
            iterations=25,
            quantized=quantized,
        )


@pytest.mark.parametrize(
    "root,quantum,params",
    [
        (
            "0.008824580881215078",
            "0.000000000000000002",
            {"bracket": (0, 0.02), "iterations": 54, "tolerance": 4e-18},
        ),
        (
            "80000000000000.04",
            "0.02",
            {
                "bracket": (80000000000000.03, 80000000000000.05),
                "iterations": 1,
                "tolerance": 0.02,
                "quantized_residual_bound": "0.02",
            },
        ),
    ],
    ids=["small-quantum-center", "large-center"],
)
def test_unrepresentable_final_center_refuses_even_when_boundaries_round_trip(
    root, quantum, params
):
    center, step = Fraction(root), Fraction(quantum)
    assert _shortest_decimal(float(center - step / 2)) == center - step / 2
    assert _shortest_decimal(float(center + step / 2)) == center + step / 2
    assert _shortest_decimal(float(center)) != center
    frame = _frame([0])
    frame.table("person")["net"] = pd.Series([center], dtype=object)
    with pytest.raises(
        ValueError,
        match=r"quantized values must round-trip exactly.*rows \[11\]",
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [],
            frame=frame,
            quantum=quantum,
            **params,
        )


def test_float64_bisection_tolerance_refusal_names_only_affected_rows():
    frame = _frame([0, 0])
    frame.table("person")["net"] = pd.Series(
        [Fraction(10000000000000001), Fraction(10000000000000000)], dtype=object
    )
    with pytest.raises(
        ValueError,
        match=r"float64 bisection cannot meet the declared tolerance; rows \[11\]",
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [],
            frame=frame,
            bracket=(10000000000000000, 10000000000000004),
            iterations=2,
            tolerance=1,
        )


def test_tie_selected_final_center_overflow_names_only_affected_rows():
    # One boundary pair certifies k=1's upper tie, so half-even selects k=2.
    # The evaluated boundaries remain finite, but the chosen center overflows.
    with pytest.raises(
        ValueError,
        match=r"quantized values must round-trip exactly.*rows \[11\]",
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [1.5e308, 1e308],
            bracket=(0, 1.6e308),
            iterations=1,
            tolerance=1e308,
            quantum="1" + "0" * 308,
            extra_evaluation_budget=3,
        )


def test_boundary_overflow_refusal_names_only_affected_rows():
    with pytest.raises(
        ValueError, match=r"quantization boundaries exceed float64; rows \[11\]"
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [1.6e308, 0],
            bracket=(0, 1.7e308),
            iterations=2,
            tolerance=1e308,
            quantum="18" + "0" * 307,
        )


def test_integer_units_range_refusal_names_only_affected_rows(monkeypatch):
    from types import SimpleNamespace

    from microcosm.frame import rules_kernels

    original_quantize = SimulateSolveZeroKernel._quantize
    original_iinfo = np.iinfo

    def quantize_with_small_range(self, *args, **kwargs):
        # Distinct exact shortest-decimal boundaries imply |k| < 10**18, so
        # both coordinate checks together preclude a real int64 overflow.
        # Narrow the range to exercise only the guard's selective row naming.
        with monkeypatch.context() as patch:
            patch.setattr(
                rules_kernels.np,
                "iinfo",
                lambda dtype: (
                    SimpleNamespace(min=-150, max=150)
                    if dtype is np.int64
                    else original_iinfo(dtype)
                ),
            )
            return original_quantize(self, *args, **kwargs)

    monkeypatch.setattr(SimulateSolveZeroKernel, "_quantize", quantize_with_small_range)
    with pytest.raises(
        ValueError, match=r"quantization integer units exceed int64; rows \[11\]"
    ):
        _run(ScheduleEngine(((Fraction(0), Fraction(0)),)), [2, 1])


def test_distinct_boundaries_above_int64_refuse_their_decimal_roundtrip(monkeypatch):
    from decimal import localcontext

    center = Fraction(2**52) + Fraction(1, 2)
    index = 2**64
    quantum = center / index
    with localcontext() as context:
        context.prec = 200
        quantum_text = format(
            Decimal(quantum.numerator) / Decimal(quantum.denominator), "f"
        )
    assert Fraction(quantum_text) == quantum
    original_quantize = SimulateSolveZeroKernel._quantize
    inspected = []

    def inspect_candidate(self, context, frame, key, lo, hi, residual):
        candidate = _half_even(
            (_shortest_decimal(lo[0]) + _shortest_decimal(hi[0])) / (2 * quantum)
        )
        left = (Fraction(candidate) - Fraction(1, 2)) * quantum
        right = (Fraction(candidate) + Fraction(1, 2)) * quantum
        assert candidate == index > np.iinfo(np.int64).max
        assert float(left) < float(right)
        assert _shortest_decimal(float(left)) != left
        assert _shortest_decimal(float(right)) != right
        inspected.append(candidate)
        return original_quantize(self, context, frame, key, lo, hi, residual)

    monkeypatch.setattr(SimulateSolveZeroKernel, "_quantize", inspect_candidate)
    frame = _frame([0])
    frame.table("person")["net"] = pd.Series([center], dtype=object)
    with pytest.raises(
        ValueError, match=r"boundaries must round-trip exactly.*rows \[11\]\.$"
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [],
            frame=frame,
            bracket=(2**52, 2**52 + 1),
            tolerance=1,
            iterations=1,
            quantum=quantum_text,
            quantized_residual_bound="1",
            extra_evaluation_budget=3,
        )
    assert inspected == [index]


def test_nearest_quantum_is_not_first_nonpositive_grid_point():
    result, _ = _run(ScheduleEngine(((Fraction(0), Fraction(0)),)), [1.001])
    assert result.columns["person", "gross"].iloc[0] == 1
    assert result.receipt["max_quantized_residual"] == pytest.approx(0.001)


def test_half_cent_crossing_is_certified_instead_of_rounding_upper_endpoint():
    engine = ScheduleEngine(((Fraction(0), Fraction(0)),))
    result, node = _run(engine, [1.0049], bracket=(0, 2), tolerance=0.01, iterations=8)
    # The old upper endpoint is 1.0078125: blindly rounding it gives 1.01.
    assert result.columns["person", "gross"].iloc[0] == 1.00
    assert any(np.array_equal(call, [1.005]) for call in engine.calls)
    assert len(engine.calls) == node.params["iterations"] + 3 + 9


def test_wide_interval_uses_further_boundary_evaluations_until_one_quantum_is_pinned():
    engine = ScheduleEngine(((Fraction(0), Fraction(0)),))
    result, _ = _run(
        engine,
        [1.7749],
        bracket=(0, 4),
        tolerance=2,
        iterations=1,
        extra_evaluation_budget=17,
    )
    assert result.columns["person", "gross"].iloc[0] == 1.77
    # The first two candidate bins lie below the root. Their boundary signs
    # must narrow the interval before the final 1.775 boundary can certify it.
    for boundary in (1.005, 1.505, 1.775):
        assert any(np.array_equal(call, [boundary]) for call in engine.calls)
    assert len(engine.calls) == 21


def test_uncertified_interval_refuses_and_names_rows():
    with pytest.raises(
        ValueError, match=r"certification budget exhausted; rows \[11\]"
    ):
        _run(
            ScheduleEngine(),
            [1],
            bracket=(0, 100),
            tolerance=50,
            iterations=1,
            extra_evaluation_budget=3,
        )


def test_quantized_residual_is_reevaluated_and_bound_refuses_with_rows():
    engine = ScheduleEngine(((Fraction(0), Fraction(0)),))
    with pytest.raises(
        ValueError, match=r"residual exceeds declared bound; rows \[11\]"
    ):
        _run(engine, [1.004], quantized_residual_bound="0.001")
    assert engine.calls[-1].tolist() == [1]


def test_exact_zero_residual_bound_is_admitted_without_decimal_context_arithmetic():
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 1
        result, _ = _run(ScheduleEngine(), [80], quantized_residual_bound="0")
    assert result.columns["person", "gross"].tolist() == [100]
    assert result.receipt["max_quantized_residual"] == 0


def test_positive_upper_bracket_is_still_refused():
    with pytest.raises(ValueError, match="upper endpoint has a positive residual"):
        _run(ScheduleEngine(), [80], bracket=(0, 90))


def test_zero_plateau_cannot_certify_a_unique_rounding_bin():
    class Plateau(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            self.calls.append(bundle.table("person")["income"].to_numpy().copy())
            return {"tax": np.zeros(bundle.n("person"))}

    # Source 0 and no trial term leaves every boundary at zero.
    with pytest.raises(ValueError, match=r"ambiguous.*rows \[11\]"):
        _run(Plateau(), [0], terms=_terms()[:-1])


class RoundedScheduleEngine(ScheduleEngine):
    """Model a rules engine that rounds each exact tax result to float64."""

    def materialize(self, bundle, variables, period):
        result = super().materialize(bundle, variables, period)
        return {
            variable: values.astype(np.float64) for variable, values in result.items()
        }


@pytest.mark.parametrize(
    "rate,net,margin,params",
    [
        (Fraction(1, 5), 0.812, "0.000001", {}),
        (
            Fraction(729, 1000),
            17286951182.993874,
            "0.0001",
            {"bracket": (0, 100000000000), "tolerance": 0.001, "iterations": 50},
        ),
        (
            Fraction(454, 1000),
            93147957362.88861,
            "0.0001",
            {
                "bracket": (170600654510, 170600654512),
                "tolerance": 0.0001,
                "iterations": 20,
            },
        ),
        (
            Fraction(44577160543827161, 50750000000000000),
            0.12345678912345678,
            "0.000001",
            {},
        ),
    ],
    ids=["float-02-tie", "large-0729", "large-0454", "fraction-rate-tie"],
)
def test_rounded_engine_boundary_reproductions_refuse_ambiguous_rows(
    rate, net, margin, params
):
    class FloatMultiply(RoundedScheduleEngine):
        def materialize(self, bundle, variables, period):
            if rate != Fraction(1, 5):
                return super().materialize(bundle, variables, period)
            trial = bundle.table("person")["income"].to_numpy()
            self.calls.append(trial.copy())
            return {"tax": trial * 0.2}

    with pytest.raises(
        ValueError,
        match=r"ambiguous.*largest \|F\(b\)\| [0-9/]+; rows \[11\]",
    ):
        _run(
            FloatMultiply(((Fraction(0), rate),)),
            [net],
            certificate_margin=margin,
            **params,
        )


def test_positive_margin_never_resolves_even_an_exact_half_cent_tie():
    with pytest.raises(ValueError, match=r"ambiguous.*rows \[11\]"):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [1.015],
            certificate_margin="0.000001",
            quantized_residual_bound="0.006",
        )


def test_margin_ambiguity_names_all_affected_rows_and_largest_absolute_residual():
    with pytest.raises(
        ValueError,
        match=r"ambiguous.*largest \|F\(b\)\| 1/2000000; rows \[11, 12\]",
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [1.0049995, 2.00500025, 3],
            certificate_margin="0.000001",
        )


def test_certificate_margin_applies_to_the_final_exact_residual_bound():
    with pytest.raises(
        ValueError, match=r"residual exceeds declared bound; rows \[11\]"
    ):
        _run(
            ScheduleEngine(((Fraction(0), Fraction(0)),)),
            [1.003],
            certificate_margin="0.001",
            quantized_residual_bound="0.0035",
        )


def test_exact_affine_residual_bound_never_rounds_down_through_float64():
    terms = _terms()
    terms.insert(1, {**terms[0], "coefficient": 1e-19})
    residual = Fraction("0.8032") * (1 + Fraction("0.0000000000000000001"))
    residual += Fraction("0.2") - 1
    assert residual > Fraction("0.0032")
    assert float(residual) == float("0.0032")
    with pytest.raises(
        ValueError, match=r"residual exceeds declared bound; rows \[11\]"
    ):
        _run(
            ScheduleEngine(),
            [0.8032],
            terms=terms,
            quantized_residual_bound="0.0032",
        )


def test_integer_affine_source_enters_exactly_before_small_coefficient():
    frame = _frame([0])
    frame.table("person")["net"] = np.asarray([9007199254740993], dtype=np.int64)
    frame.table("family")["family_net"] = [0.1142800745259007]
    terms = _terms(coefficient=1e-16)
    terms.insert(1, _terms("family")[0])
    exact_root = Fraction(9007199254740993) * Fraction("0.0000000000000001")
    exact_root += Fraction("0.1142800745259007")
    assert exact_root == Fraction("1.015")
    result, _ = _run(
        ScheduleEngine(((Fraction(0), Fraction(0)),)),
        [],
        frame=frame,
        terms=terms,
        bracket=(0, 2),
        tolerance=1e-7,
        iterations=25,
    )
    assert result.columns["person", "gross"].tolist() == [1.02]


def test_later_boundary_pairs_reveal_plateau_even_after_initial_certification():
    class Floored(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            trial = bundle.table("person")["income"].to_numpy()
            self.calls.append(trial.copy())
            return {"tax": np.maximum(1.006 - trial, 0)}

    engine = Floored()
    terms = [
        {"variable": "tax", "coefficient": 1, "entity": "person", "column": "gross"}
    ]
    with pytest.raises(ValueError, match=r"ambiguous.*rows \[11\]"):
        _run(engine, [0], terms=terms, bracket=(0, 2), iterations=25, tolerance=1e-7)
    assert any(np.array_equal(call, [1.015]) for call in engine.calls)
    assert any(np.array_equal(call, [1.025]) for call in engine.calls)


def test_later_boundary_pairs_reject_observed_increase_after_initial_certification():
    class IncreasingAtLaterBoundary(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            trial = bundle.table("person")["income"].to_numpy()
            self.calls.append(trial.copy())
            return {
                "tax": np.asarray(
                    [
                        Fraction(-1)
                        if value == 0.995
                        else Fraction("1.005") - _shortest_decimal(value)
                        for value in trial
                    ],
                    dtype=object,
                )
            }

    terms = [
        {"variable": "tax", "coefficient": 1, "entity": "person", "column": "gross"}
    ]
    with pytest.raises(ValueError, match=r"non-increasing.*rows \[11\]"):
        _run(
            IncreasingAtLaterBoundary(),
            [0],
            terms=terms,
            bracket=(0, 2),
            iterations=25,
            tolerance=1e-7,
        )


@pytest.mark.parametrize("point", [1.005, 1.0], ids=["boundary", "bound"])
def test_nonfinite_certificate_or_bound_outputs_name_affected_rows(point):
    class NonfiniteAtPoint(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            trial = bundle.table("person")["income"].to_numpy()
            self.calls.append(trial.copy())
            return {"tax": np.where(trial == point, np.nan, np.zeros(len(trial)))}

    with pytest.raises(ValueError, match=r"engine output 'tax'.*rows \[11\]"):
        _run(
            NonfiniteAtPoint(),
            [1.004, 2.004],
            bracket=(0, 3),
            iterations=25,
            tolerance=1e-7,
        )


@st.composite
def schedules_and_nets(draw):
    thresholds = sorted(
        draw(st.lists(st.integers(1, 1000), min_size=0, max_size=3, unique=True))
    )
    rates = draw(
        st.lists(
            st.integers(0, 899),
            min_size=len(thresholds) + 1,
            max_size=len(thresholds) + 1,
        )
    )
    schedule = tuple(
        (Fraction(threshold), Fraction(rate, 1000))
        for threshold, rate in zip([0, *thresholds], rates, strict=True)
    )
    cents = draw(st.lists(st.integers(0, 100000), min_size=1, max_size=4))
    return schedule, [Fraction(net, 100) for net in sorted(cents)]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(schedules_and_nets())
def test_piecewise_quantized_inverse_matches_fraction_distance_roundtrip_and_monotonicity(
    case,
):
    schedule, nets = case
    engine = ScheduleEngine(schedule)
    result, _ = _run(engine, [float(net) for net in nets], terms=_terms(coefficient=52))
    gross = result.columns["person", "gross"].to_numpy()
    for net, observed in zip(nets, gross, strict=True):
        root = exact_inverse(52 * net, schedule)
        expected = Q * _half_even(root / Q)
        assert _shortest_decimal(observed) == expected
        assert abs(root - expected) <= Q / 2
        assert abs(52 * net - (expected - _tax(expected, schedule))) <= Fraction(
            "0.00500001"
        )
    assert np.all(np.diff(gross) >= 0)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    schedules_and_nets(),
    st.sampled_from(["0", "0.000001"]),
    st.lists(st.integers(-1000, 1000), min_size=1, max_size=20),
)
def test_bounded_engine_error_never_certifies_a_wrong_quantum_and_refusals_name_rows(
    case, margin_text, errors
):
    schedule, nets = case
    margin = Fraction(margin_text)

    class BoundedErrorEngine(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            output = super().materialize(bundle, variables, period)
            # Each engine term carries e <= m/3; the source and trial terms
            # carry zero error. Fraction outputs keep that promise exact.
            offset = (len(self.calls) - 1) * len(nets)
            noisy = [
                value + margin * Fraction(errors[(offset + row) % len(errors)], 3000)
                for row, value in enumerate(output["tax"])
            ]
            return {"tax": np.asarray(noisy, dtype=object)}

    engine = BoundedErrorEngine(schedule)
    try:
        result, node = _run(
            engine, [float(net) for net in nets], certificate_margin=margin_text
        )
    except ValueError as error:
        message = str(error)
        assert "rows [" in message
        assert any(str(row) in message for row in range(11, 11 + len(nets)))
        return
    for net, observed in zip(nets, result.columns["person", "gross"], strict=True):
        root = exact_inverse(net, schedule)
        exact_observed = _shortest_decimal(observed)
        assert abs(exact_observed - root) <= Q / 2
        if margin == 0:
            assert exact_observed == Q * _half_even(root / Q)
    expected_calls = (
        node.params["iterations"] + 3 + node.params["extra_evaluation_budget"]
    )
    assert len(engine.calls) == result.receipt["evaluations"] == expected_calls
    assert result.receipt["certificate_margin"] == margin_text
    assert result.receipt["certificate_contract"] == (
        "At every evaluated point, the computed residual differs from "
        "the exact residual of the declared terms by at most certificate_margin."
    )


@settings(max_examples=35, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(schedules_and_nets())
def test_quantized_determinism_and_evaluation_count_are_value_independent(case):
    schedule, nets = case
    engine = ScheduleEngine(schedule)
    result, node = _run(engine, [float(net) for net in nets])
    calls = node.params["iterations"] + 3 + node.params["extra_evaluation_budget"]
    assert len(engine.calls) == result.receipt["evaluations"] == calls
    assert result.receipt["extra_evaluations"] == 9
    assert (
        result.receipt["evaluation_count_formula"]
        == "iterations + 3 + extra_evaluation_budget"
    )
    repeated, _ = _run(engine, [float(net) for net in nets])
    assert (
        result.columns["person", "gross"].to_numpy().tobytes()
        == repeated.columns["person", "gross"].to_numpy().tobytes()
    )
    assert dict(result.receipt) == dict(repeated.receipt)
    assert len(engine.calls) == 2 * calls


@pytest.mark.parametrize(
    "kernel_type,expected",
    [
        (
            SimulateSolveZeroKernel,
            "simulate.solve_zero@1 parameters require ['bracket', 'engine_ref', "
            "'input_overrides', 'iterations', 'output_coefficients', 'period', "
            "'solve_input', 'tolerance', 'variables'] and allow only ['bracket', "
            "'certificate_margin', 'components', 'engine_ref', "
            "'extra_evaluation_budget', 'input_overrides', 'iterations', "
            "'output_coefficients', 'period', 'quantized_residual_bound', "
            "'quantum', 'resource_sha256', 'solve_input', 'tie_rule', "
            "'tolerance', 'variables'].",
        ),
        (
            SimulateCounterfactualKernel,
            "simulate.counterfactual@1 parameters require ['engine_ref', "
            "'input_overrides', 'output_coefficients', 'period', 'variables'] "
            "and allow only ['components', 'engine_ref', 'input_overrides', "
            "'output_coefficients', 'period', 'resource_sha256', 'variables'].",
        ),
    ],
    ids=["solver-lists-quantization", "counterfactual-keeps-legacy-list"],
)
def test_legacy_invalid_parameter_diagnostics_pin_both_permitted_lists(
    kernel_type, expected
):
    kernel = kernel_type({REF: ScheduleEngine()})
    terms = [_terms()[1]]
    node = _node(kernel, terms=terms, quantized=False, garbage=1)
    with pytest.raises(ValueError) as error:
        kernel.run(_context(_frame([1]), node, weighted_entities=("family",)))
    assert str(error.value) == expected


def test_legacy_solve_zero_without_quantum_preserves_output_and_receipt_bytes():
    class Legacy(ScheduleEngine):
        def materialize(self, bundle, variables, period):
            trial = bundle.table("person")["income"].to_numpy()
            self.calls.append(trial.copy())
            return {"tax": np.maximum(10 - trial, 0)}

    engine = Legacy()
    terms = [
        {"variable": "tax", "coefficient": 1, "entity": "person", "column": "gross"}
    ]
    result, _ = _run(
        engine,
        [80],
        terms=terms,
        quantized=False,
        bracket=(0, 256),
        tolerance=1e-7,
        iterations=34,
    )
    expected_receipt = {
        "engine_ref": REF,
        "period": "2026-27",
        "variables": ("tax",),
        "components": "[]",
        "bracket": (0, 256),
        "tolerance": 1e-7,
        "iterations": 34,
        "evaluations": 37,
    }
    assert (
        result.columns["person", "gross"].to_numpy().tobytes()
        == np.array([10.0]).tobytes()
    )
    assert (
        json.dumps(dict(result.receipt)).encode()
        == json.dumps(expected_receipt).encode()
    )
    assert len(engine.calls) == 37


def test_legacy_counterfactual_engine_terms_preserve_output_and_receipt_bytes():
    engine = ScheduleEngine()
    kernel = SimulateCounterfactualKernel({REF: engine})
    terms = [
        {"variable": "tax", "coefficient": 1, "entity": "person", "column": "gross"}
    ]
    frame = _frame([80])
    frame.table("person")["income"] = [5.0]
    result = kernel.run(
        _context(frame, _node(kernel, terms=terms), weighted_entities=("family",))
    )
    assert (
        result.columns["person", "gross"].to_numpy().tobytes()
        == np.array([1.0]).tobytes()
    )
    expected = {
        "engine_ref": REF,
        "period": "2026-27",
        "variables": ("tax",),
        "components": "[]",
    }
    assert json.dumps(dict(result.receipt)).encode() == json.dumps(expected).encode()
    assert len(engine.calls) == 1
