"""NZ's engine gross-up declarations, composed through the real S5 guard.

The packaged population graph is a later wiring task. The composition check
installs the exact reusable NZ nodes in the existing small transport fixture;
its CREATE inventory supplies the absent model inputs without executing a
population build or claiming a real-engine result.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.transport.compose import (
    compose_transport_graph,
    load_transport_spec,
)
from microcosm.build.transport.money_kernels import MONEY_MINOR_UNITS
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA, axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import load_axiom_input_surface
from microcosm.frame.input_closure import ClosureClass, EncodedBy, InputClosure
from microcosm.frame.rules_kernels import SimulateSolveZeroKernel
from microcosm.graph import Owned, StructuralDelta
from test_support.microcosm_build.nz_msd_gross_up import (
    ScheduleFixtureEngine,
    _node,
    exact_inverse,
    exact_tax,
    schedule_fixture,
)
from test_support.microcosm_build.transport_composed import make_composed_fixture
from test_support.microcosm_frame.kernels import _context
from test_support.paths import paths_for
from tools.build_nz_benefit_gross_up import (
    DISCLOSURE,
    NET_ROUNDING,
    PROGRAMMES,
    documents,
)

_BUILD = paths_for("microcosm-build")
_NZ = _BUILD.package / "src/microcosm/build/nz"


def _load(name):
    return json.loads((_NZ / f"{name}.json").read_text())


def test_generated_resources_match_and_are_packaged():
    resources = load_transport_spec("nz")["resources"]
    for name, document in documents().items():
        assert _load(name.removesuffix(".json")) == document
        assert resources[name.removesuffix(".json")] == document


def test_stage_citations_and_new_binding_notes_name_public_references():
    document = _load("axiom_rules_bindings")
    bindings = {row["id"]: row for row in document["bindings"]}
    rulespec = document["rulespec"]
    rulespec_url = (
        f"https://github.com/{rulespec['repository']}/blob/{rulespec['commit']}"
    )
    msd_url = (
        "https://www.workandincome.govt.nz/products/benefit-rates/"
        "benefit-rates-april-2026.html"
    )
    references = [
        (stage["id"], stage["citation"], bindings[stage["rules_binding"]])
        for stage in _load("precalibration_stages")["stages"]
    ]
    # Only Schedule 1 adds a binding note in this change; older notes retain
    # the provenance conventions that the existing package already carries.
    binding = bindings["income_tax_schedule_1"]
    references.append((binding["id"], binding["notes"], binding))
    for name, citation, binding in references:
        pinned_module = f"{rulespec_url}/{binding['rulespec_path']}"
        assert (
            pinned_module in citation and binding["sha256"] in citation
        ) or msd_url in citation, name
        assert re.search(r"H\d+|G8-|D1", citation) is None, name


@pytest.mark.parametrize(
    "name",
    [
        "benefit_gross_up",
        "precalibration_stages",
        "msd_benefit_rates_20260401",
        "axiom_input_closure",
    ],
)
def test_new_resource_text_carries_no_private_hub_labels(name):
    document = _load(name)
    # The three new resources contain only this change's text. The closure
    # predates this change, so inspect only its rewritten description.
    text = (
        document["description"]
        if name == "axiom_input_closure"
        else json.dumps(document)
    )
    assert re.search(r"H\d+|G8-|D1", text) is None


def test_schedule_binding_is_compiled_and_taxable_income_is_solver_owned():
    bindings = {row["id"]: row for row in _load("axiom_rules_bindings")["bindings"]}
    tax = bindings["income_tax_schedule_1"]
    assert tax["role"] == "benefit_gross_up_tax_schedule"
    assert tax["variables"] == ["individual_income_tax_before_credits"]
    assert (tax["entity"], tax["engine_entity"], tax["period"]) == (
        "person",
        "Person",
        "2026-27",
    )
    surface = load_axiom_input_surface(
        paths_for("microcosm-frame").tests / "fixtures/axiom_input_surfaces/nz.json"
    )
    assert surface.modules[tax["rulespec_path"]].status == "compiled"
    assert surface.modules[tax["rulespec_path"]].sha256 == tax["sha256"]
    closure = InputClosure.from_dict(_load("axiom_input_closure"))
    assert closure.check(axiom_concept_mapping("nz"), surface) == ()
    (entry,) = closure.entries_for(tax["rulespec_path"])
    assert entry.input == "taxable_income"
    assert entry.closure_class is ClosureClass.ENCODED
    assert entry.encoded_by is EncodedBy.GRAPH_NODE
    assert entry.value is None
    assert closure.defaults(tax["rulespec_path"], "Person") == {}
    assert not any(
        binding.ref == entry.ref for binding in axiom_concept_mapping("nz").bindings
    )
    (ownership,) = _load("axiom_input_closure")["solver_owned_inputs"]
    assert ownership["binding"] == "income_tax_schedule_1"
    assert ownership["data_input"] is ownership["defaulted"] is False
    assert ownership["node_ids"] == [f"nz.gross_up.{p}" for p in PROGRAMMES]
    for node in _load("benefit_gross_up")["nodes"]:
        assert all("taxable_income" not in row["columns"] for row in node["inputs"])


def test_affine_residual_uses_whole_cent_nets_and_engine_tax_only():
    document = _load("benefit_gross_up")
    assert document["disclosure"] == DISCLOSURE
    assert document["net_rounding"] == NET_ROUNDING
    solves = [
        node for node in document["nodes"] if node["kernel"] == "simulate.solve_zero@1"
    ]
    assert len(solves) == 3
    for programme, node in zip(PROGRAMMES, solves, strict=True):
        params = node["params"]
        cents = f"{programme}_net_weekly_payment_cents"
        gross = f"{programme}_annual_gross"
        assert node["inputs"] == [{"entity": "person", "columns": [cents]}]
        assert params["engine_ref"] == {
            "binding": "income_tax_schedule_1",
            "field": "engine_ref",
        }
        assert params["period"] == {
            "binding": "income_tax_schedule_1",
            "field": "period",
        }
        terms = json.loads(params["output_coefficients"])
        assert terms == [
            {
                "source_entity": "person",
                "source_column": cents,
                "coefficient": 0.52,
                "entity": "person",
                "column": gross,
            },
            {
                "variable": "individual_income_tax_before_credits",
                "coefficient": 1,
                "entity": "person",
                "column": gross,
            },
            {"trial": True, "coefficient": -1, "entity": "person", "column": gross},
        ]
        # 0.52 enters the quantized solve as its shortest decimal, exactly 52/100.
        assert Fraction(str(terms[0]["coefficient"])) == Fraction(52, 100)
        assert params["variables"] == ["individual_income_tax_before_credits"]
        assert json.loads(params["solve_input"]) == {
            "entity": "person",
            "column": "taxable_income",
        }
        assert json.loads(params["input_overrides"]) == []
        assert (params["quantum"], params["tie_rule"]) == ("0.01", "half_even")
        assert params["quantized_residual_bound"] == "0.0050001"
        assert params["certificate_margin"] == "0.000001"
        assert params["extra_evaluation_budget"] == 9
        assert "acc" not in json.dumps(terms)
    rates = document["nodes"][0]
    assert rates["params"]["variables"] == [
        f"{p}_net_weekly_payment" for p in PROGRAMMES
    ]
    assert not any(
        "gross_weekly_rate" in output["column"] for output in rates["outputs"]
    )


def test_net_cents_node_rounds_half_even_ahead_of_every_gross_up():
    nodes = _load("benefit_gross_up")["nodes"]
    ids = [node["id"] for node in nodes]
    cents_node = nodes[ids.index("nz.benefits.net_cents")]
    assert ids.index("nz.benefits.rates") < ids.index("nz.benefits.net_cents")
    assert all(
        ids.index("nz.benefits.net_cents") < ids.index(f"nz.gross_up.{p}")
        for p in PROGRAMMES
    )
    assert cents_node["kernel"] == "money.minor_units@1"
    assert cents_node["params"]["engine_ref"] == {
        "binding": "main_benefits_rates",
        "field": "engine_ref",
    }
    declaration = json.loads(cents_node["params"]["declaration"])
    assert declaration == {
        "entity": "person",
        "columns": [
            {
                "source_column": f"{p}_net_weekly_payment",
                "output_column": f"{p}_net_weekly_payment_cents",
                "quantum": "0.01",
                "rounding": "half_even",
            }
            for p in PROGRAMMES
        ],
    }
    assert {row["dtype"] for row in cents_node["outputs"]} == {"int64"}
    (rates,) = [
        stage
        for stage in _load("precalibration_stages")["stages"]
        if stage["rules_binding"] == "main_benefits_rates"
    ]
    assert [row["column"] for row in rates["derived_outputs"]] == [
        f"{p}_net_weekly_payment_cents" for p in PROGRAMMES
    ]


def test_direct_annual_cents_exposure_and_super_before_tax_are_explicit():
    document = _load("benefit_gross_up")
    nodes = {node["id"]: node for node in document["nodes"]}
    annual = json.loads(nodes["nz.gross_up.annual_cents"]["params"]["declaration"])
    assert [column["source_column"] for column in annual["columns"]] == [
        f"{p}_annual_gross" for p in PROGRAMMES
    ]
    assert all(column["rounding"] == "exact" for column in annual["columns"])
    super_node = nodes["nz.gross_up.super_cents"]
    assert "engine_ref" not in super_node["params"]
    (super_column,) = json.loads(super_node["params"]["declaration"])["columns"]
    assert super_column["source_column"] == "nz_super_weekly_amount_before_tax"
    assert super_column["source_tax_basis"] == "before_tax"
    transfer = json.loads(
        nodes["nz.benefits.annual_transfers"]["params"]["declaration"]
    )
    assert transfer["exposure"] == {
        "central": 1,
        "convention": "full_year_stock_equivalent",
    }
    assert (
        transfer["weeks_per_year"]
        == _load("scenarios")["gap_conventions"]["annualisation"]["weeks_per_year"]
    )
    assert [row["source_column"] for row in transfer["transfers"][:3]] == [
        f"{p}_annual_gross_cents" for p in PROGRAMMES
    ]
    assert all(
        row["source_basis"] == "annual_gross_cents" for row in transfer["transfers"][:3]
    )
    assert transfer["transfers"][-1]["source_basis"] == "weekly_before_tax_cents"
    assert document["weekly_gross_reporting"] == {
        "source": "annual_gross_cents",
        "divisor": 52,
        "tie_rule": "half_even",
        "reporting_only": True,
        "persist": False,
    }
    contract = _load("export_contract")
    persisted = set(contract["required"]) | set(contract["optional"])
    assert not persisted.intersection(
        output["column"] for node in nodes.values() for output in node["outputs"]
    )


@pytest.fixture
def composed_gross_up(tmp_path):
    fixture = make_composed_fixture(tmp_path / "gross-up-composition")
    spec = copy.deepcopy(fixture.spec)
    nodes = _load("benefit_gross_up")["nodes"]
    wanted = {"main_benefits_rates", "income_tax_schedule_1"}
    bindings = [
        row for row in _load("axiom_rules_bindings")["bindings"] if row["id"] in wanted
    ]
    spec["resources"]["axiom_rules_bindings"]["bindings"].extend(bindings)
    spec["resources"]["precalibration_stages"] = _load("precalibration_stages")
    rows = spec["resources"]["transport_graph"]["nodes"]
    rows.append(
        {
            "id": "nz.central",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "nz.open",
            "inputs": [{"entity": "person", "columns": ["age"]}],
        }
    )
    for row in rows:
        if row["id"] == "nz.calibrate":
            row["base"] = "nz.central"
    rows.extend(nodes)
    produced = {output["column"] for node in nodes for output in node["outputs"]}
    required = {
        column for node in nodes for row in node["inputs"] for column in row["columns"]
    }
    inventory = {
        owned.column
        for owned in fixture.config.create_outputs
        if owned.entity == "person"
    }
    missing = sorted(required - produced - inventory)
    create_outputs = (
        *fixture.config.create_outputs,
        *(
            Owned("person", name, "bool" if name.startswith("receives_") else "float64")
            for name in missing
        ),
    )
    config = replace(
        fixture.config,
        create_outputs=create_outputs,
        engine_refs={
            **fixture.config.engine_refs,
            **{binding: f"prepared:{binding}" for binding in wanted},
        },
    )
    problem = next(row for row in rows if row["id"] == "nz.targets.problem")
    problem["population"] = "nz.central"
    problem["inputs"].append(
        {
            "entity": "person",
            "columns": [
                f"{p}_annual_gross_transfer_cents" for p in (*PROGRAMMES, "nz_super")
            ],
        }
    )
    return spec, config


def test_real_gross_up_declarations_compose_with_all_stage_nodes_nonstructural(
    composed_gross_up,
):
    spec, config = composed_gross_up
    graph = compose_transport_graph(spec, config)
    stage_refs = {
        config.engine_refs[name]
        for name in ("main_benefits_rates", "income_tax_schedule_1")
    }
    found = []
    for node in graph.nodes:
        if any(ref in json.dumps(dict(node.params)) for ref in stage_refs):
            found.append(node.id)
            assert node.structural is StructuralDelta.NONE
    # The package's prepared-reference inventory also carries these refs;
    # the same non-structural assertion above covers that metadata node.
    assert {
        "nz.benefits.rates",
        "nz.benefits.net_cents",
        "nz.gross_up.annual_cents",
        *(f"nz.gross_up.{p}" for p in PROGRAMMES),
    } <= set(found)
    assert not any(
        output.rows != "all"
        for node in graph.nodes
        if node.id in found
        for output in node.outputs
    )


def test_private_tax_output_cannot_leak_to_calibration(composed_gross_up):
    spec, config = composed_gross_up
    stage = spec["resources"]["precalibration_stages"]["stages"][1]
    stage["derived_outputs"] = [
        row
        for row in stage["derived_outputs"]
        if row["column"] != "jobseeker_support_annual_gross_cents"
    ]
    with pytest.raises(ValueError, match="undeclared pre-calibration stage output"):
        compose_transport_graph(spec, config)


def _solve_weekly_nets(nets, ids):
    """Run the packaged net-cents node, then the packaged JS gross-up solve."""
    spec = load_transport_spec("nz")
    rows = {row["id"]: row for row in spec["resources"]["benefit_gross_up"]["nodes"]}
    engine_ref = "fractional-net-schedule-fixture"
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": ids,
                    "person_household_id": ids,
                    "person_family_id": ids,
                    **{
                        f"{p}_net_weekly_payment": [
                            float(net) if p == "jobseeker_support" else 0.0
                            for net in nets
                        ]
                        for p in PROGRAMMES
                    },
                }
            ),
            "household": pd.DataFrame({"household_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
        },
        NZ_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )
    cents = _node(rows["nz.benefits.net_cents"], spec, engine_ref)
    result = MONEY_MINOR_UNITS.run(
        _context(frame, cents, weighted_entities=("household",))
    )
    for (entity, column), values in result.columns.items():
        frame.table(entity)[column] = values.to_numpy()
    node = _node(rows["nz.gross_up.jobseeker_support"], spec, engine_ref)
    kernel = SimulateSolveZeroKernel({engine_ref: ScheduleFixtureEngine()})
    solved = kernel.run(_context(frame, node, weighted_entities=("household",)))
    return frame, solved


def _rounded_net_oracle(net, brackets):
    """The half-even cent net and the half-even cent inverse of its annual net."""
    rounded = Fraction(round(net * 100), 100)
    return rounded, Fraction(round(exact_inverse(rounded * 52, brackets) * 100), 100)


# 801 and 802 carry fractional-cent nets. 901 and 902 put the exact root of
# their unrounded nets on a half-cent, which refused before nets entered as
# whole cents; rounded first, both solve.
_FRACTIONAL_NETS = (
    (801, Fraction("123.4567")),
    (802, Fraction("123.4567") - Fraction("12.3456")),
    (901, Fraction("0.00111875")),
    (902, Fraction("0.01111875") - Fraction("0.01")),
)


def test_fractional_cent_nets_solve_against_the_exact_inverse_of_the_rounded_net():
    ids = [row for row, _ in _FRACTIONAL_NETS]
    nets = [net for _, net in _FRACTIONAL_NETS]
    assert all((net * 100).denominator != 1 for net in nets)
    brackets = schedule_fixture()
    # The unrounded 901 and 902 nets tie exactly on the 0.065 half-cent.
    boundary = Fraction("0.065")
    assert all(
        net * 52 + exact_tax(boundary, brackets) - boundary == 0 for net in nets[2:]
    )
    frame, result = _solve_weekly_nets(nets, ids)
    cents = frame.table("person")["jobseeker_support_net_weekly_payment_cents"]
    actual = result.columns["person", "jobseeker_support_annual_gross"]
    for net, cent, gross in zip(nets, cents.tolist(), actual.tolist(), strict=True):
        rounded, oracle = _rounded_net_oracle(net, brackets)
        assert Fraction(cent, 100) == rounded
        assert Fraction(str(gross)) == oracle
        residual = rounded * 52 + exact_tax(oracle, brackets) - oracle
        assert abs(residual) + Fraction("0.000001") <= Fraction("0.0050001")


def test_gross_up_refuses_an_unrounded_net_tie_when_it_reads_the_net_directly():
    """The declared margin still refuses an exact tie: only rounding removes it."""
    spec = load_transport_spec("nz")
    row = copy.deepcopy(
        next(
            row
            for row in spec["resources"]["benefit_gross_up"]["nodes"]
            if row["id"] == "nz.gross_up.jobseeker_support"
        )
    )
    terms = json.loads(row["params"]["output_coefficients"])
    terms[0].update(
        source_column="jobseeker_support_net_weekly_payment", coefficient=52
    )
    row["params"]["output_coefficients"] = json.dumps(terms)
    row["inputs"] = [
        {"entity": "person", "columns": ["jobseeker_support_net_weekly_payment"]}
    ]
    ids = [901, 902]
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": ids,
                    "person_household_id": ids,
                    "person_family_id": ids,
                    "jobseeker_support_net_weekly_payment": [
                        float(net) for _, net in _FRACTIONAL_NETS[2:]
                    ],
                }
            ),
            "household": pd.DataFrame({"household_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
        },
        NZ_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )
    engine_ref = "fractional-net-schedule-fixture"
    kernel = SimulateSolveZeroKernel({engine_ref: ScheduleFixtureEngine()})
    with pytest.raises(
        ValueError,
        match=r"ambiguous quantization certificate within margin.*rows \[901, 902\]",
    ):
        kernel.run(
            _context(
                frame, _node(row, spec, engine_ref), weighted_entities=("household",)
            )
        )


def _bracket_q(gross, brackets):
    """q in r = 1 - p/q for the bracket holding gross, in lowest terms."""
    rate = Fraction(brackets[0]["rate"])
    for bracket in brackets:
        if gross >= Fraction(bracket["lower"]):
            rate = Fraction(bracket["rate"])
    return (1 - rate).denominator


@settings(max_examples=300, deadline=None, derandomize=True)
@given(st.integers(min_value=0, max_value=1_500_000))
def test_cent_exact_nets_keep_every_half_cent_boundary_clear_of_the_margin(cents):
    """H13 item 5: |F(b)| >= 1/(200q) >= 2.5e-5 at the boundaries around the root.

    F is monotone non-increasing, so the two half-cent boundaries next to the
    exact root hold its smallest absolute residuals.
    """
    brackets = schedule_fixture()
    net_annual = Fraction(cents, 100) * 52
    root = exact_inverse(net_annual, brackets)
    below = (Fraction(math.floor(root * 100 - Fraction(1, 2))) + Fraction(1, 2)) / 100
    for boundary in (below, below + Fraction(1, 100)):
        if boundary < 0:
            continue
        residual = net_annual + exact_tax(boundary, brackets) - boundary
        bound = Fraction(1, 200 * _bracket_q(boundary, brackets))
        assert abs(residual) >= bound >= Fraction("0.000025")
        assert abs(residual) >= 25 * Fraction("0.000001")


@settings(max_examples=10, deadline=None, derandomize=True)
@given(
    st.lists(
        st.integers(min_value=0, max_value=200_000_000).map(
            lambda units: Fraction(units, 10_000)
        ),
        min_size=1,
        max_size=6,
    )
)
def test_any_four_decimal_net_solves_against_its_rounded_exact_inverse(nets):
    ids = list(range(1, len(nets) + 1))
    brackets = schedule_fixture()
    _, result = _solve_weekly_nets(nets, ids)
    actual = result.columns["person", "jobseeker_support_annual_gross"]
    for net, gross in zip(nets, actual.tolist(), strict=True):
        assert Fraction(str(gross)) == _rounded_net_oracle(net, brackets)[1]
