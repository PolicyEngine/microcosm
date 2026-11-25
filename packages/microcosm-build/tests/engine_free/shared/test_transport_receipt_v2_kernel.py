"""Receipt surface validation and the reserved declared-unit dispatch seam."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.column_kernels import TAKEUP_ASSIGN
from microcosm.build.transport.receipt_contract import (
    ReceiptAssignmentRefused,
    receipt_dispatch_tables,
)
from microcosm.build.transport.target_kernels import compile_target_surface
from microcosm.graph import ArtifactValue, KernelContext, Node, NumericScope, Owned
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import (
    canonical_text,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import digest


def _context(
    tmp_path,
    *,
    filter_name="adult",
    entity="person",
    measure="receives_alpha",
    output="receives_alpha",
    unit="count",
):
    contract = {
        "contract_version": 2,
        "seed_column": "seed",
        "default_assignment_profile": "central",
        "assignment_knob": "assignment",
        "assignment_profiles": {"central": {"draw_key_suffix": ""}},
        "programs": [
            {
                "program": "alpha",
                "output": "receives_alpha",
                "judgment_columns": ["eligible"],
                "judgment_reduction": "any",
                "payment_column": "payment",
                "strata": [{"target": "alpha_target", "filter": "adult"}],
                "outside_strata": "not_assigned",
            }
        ],
        "exclusion_groups": [],
        "unmodelled_programs": ["unsupported"],
    }
    bindings = {
        "schema_version": 1,
        "required_inputs": {"person": ["age"], "household": ["size"]},
        "input_domains": {
            "person": {"age": {"kind": "nonnegative_integer"}},
            "household": {"size": {"kind": "nonnegative_integer"}},
        },
        "columns": [
            {
                "entity": "person",
                "name": "adult",
                "predicates": [{"column": "age", "operator": "ge", "value": 18}],
            }
        ],
    }
    document = reference_document(
        [
            reference_row(
                "alpha_target", entity=entity, measure=measure, filter_=filter_name
            )
        ]
    )
    fact = toy_fact("alpha_target", 1, entity=entity)
    fact["observed_measure"]["unit"] = unit
    facts = write_facts(tmp_path / "facts.jsonl", [fact])
    surface, _ = compile_target_surface(
        facts, document, country="xx", references_sha256=digest(document)
    )
    params = {
        "receipt_contract": canonical_text(contract),
        "receipt_contract_sha256": digest(contract),
        "calibration_bindings": canonical_text(bindings),
        "calibration_bindings_sha256": digest(bindings),
        "profile_selection": canonical_text({"assignment": "central"}),
    }
    node = Node(
        "receipt",
        TAKEUP_ASSIGN.ref,
        population="open",
        params=params,
        outputs=(Owned("person", output, "bool"),),
    )
    return KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "age": [20],
                    "seed": [0.2],
                    "eligible": [1],
                    "payment": [1],
                }
            ),
            "household": pd.DataFrame({"household_id": [1], "size": [1]}),
        },
        weights={"person": SimpleNamespace(values=np.ones(1))},
        strata=pd.DataFrame(),
        params=node.params,
        rng=np.random.default_rng(1),
        artifacts={
            "surface": ArtifactValue(
                payload=surface,
                type=TARGET_SURFACE_TYPE,
                key="0" * 64,
                producer_key="0" * 64,
                numerics=NumericScope(),
            )
        },
    )


def test_v2_kernel_checks_declared_stratum_filter(tmp_path):
    with pytest.raises(ValueError, match="declared stratum filter"):
        TAKEUP_ASSIGN.run(_context(tmp_path, filter_name="different"))


@pytest.mark.parametrize(
    "field,value", [("entity", "household"), ("measure", "other"), ("unit", "currency")]
)
def test_v2_kernel_requires_declared_count_surface(tmp_path, field, value):
    with pytest.raises(ValueError, match="person count"):
        TAKEUP_ASSIGN.run(_context(tmp_path, **{field: value}))


@pytest.mark.parametrize("output", ["unsupported", "receives_unknown"])
def test_unmodelled_or_unknown_output_never_defaults_false(tmp_path, output):
    with pytest.raises(ValueError, match="unmodelled or unknown receipt outputs"):
        TAKEUP_ASSIGN.run(_context(tmp_path, output=output))


def test_v2_kernel_validates_full_bindings_before_preparation(tmp_path):
    context = _context(tmp_path)
    result = TAKEUP_ASSIGN.run(context)
    assert result.columns["person", "receives_alpha"].tolist() == [True]
    assert result.receipt["programs"]["receives_alpha"]["strata"][0]["numerical_check"][
        "passed"
    ]
    context.tables["household"].loc[0, "size"] = -1
    with pytest.raises(ValueError, match="Calibration input household.size"):
        TAKEUP_ASSIGN.run(context)


@pytest.mark.parametrize(
    "field", ["entitlement_entity", "payee", "draw_unit", "projection"]
)
def test_dispatch_reads_each_reserved_field_with_registered_fake_entity(field):
    person, fake = object(), object()
    row = dict.fromkeys(
        ["entitlement_entity", "payee", "draw_unit", "projection"], "person"
    )
    row[field] = "registered_fake"
    result = receipt_dispatch_tables({"person": person, "registered_fake": fake}, row)
    assert result[field] is fake
    assert all(value is person for name, value in result.items() if name != field)


def test_v2_kernel_resolves_profile_from_declared_knob(tmp_path):
    context = _context(tmp_path)
    params = copy.deepcopy(dict(context.params))
    params["profile_selection"] = json.dumps(
        {"unrelated": "central"}, sort_keys=True, separators=(",", ":")
    )
    context = KernelContext(
        node=context.node,
        tables=context.tables,
        weights=context.weights,
        strata=context.strata,
        params=params,
        rng=context.rng,
        artifacts=context.artifacts,
    )
    with pytest.raises(ValueError, match="assignment_knob"):
        TAKEUP_ASSIGN.run(context)


@pytest.mark.parametrize(
    "missing", ["calibration_bindings", "calibration_bindings_sha256"]
)
def test_v2_kernel_requires_bindings_and_sha_together(tmp_path, missing):
    # No strata need bindings here: a lone SHA must refuse on its own, rather
    # than falling through to the separate missing-stratum-bindings refusal.
    context = _context(tmp_path, filter_name=None)
    params = dict(context.params)
    contract = json.loads(params["receipt_contract"])
    row = contract["programs"][0]
    del row["strata"]
    del row["outside_strata"]
    row["target"] = "alpha_target"
    params["receipt_contract"] = canonical_text(contract)
    params["receipt_contract_sha256"] = digest(contract)
    del params[missing]
    with pytest.raises(ValueError, match="supplied together"):
        TAKEUP_ASSIGN.run(replace(context, params=params))


@pytest.mark.parametrize("weight", [0.5, 5e-324], ids=["half", "tiny"])
@pytest.mark.parametrize("declared", [False, True], ids=["plain", "declared"])
def test_v2_kernel_carries_the_mass_rule_through_the_contract_document(
    tmp_path, declared, weight
):
    context = _context(tmp_path)
    params = dict(context.params)
    contract = json.loads(params["receipt_contract"])
    if declared:
        contract["target_exceeds_eligible_mass"] = "assign_all_eligible"
    params["receipt_contract"] = canonical_text(contract)
    params["receipt_contract_sha256"] = digest(contract)
    # One adult of this weight against the surface's target of 1. The tiny
    # weight overflows every target-to-mass ratio.
    weights = {"person": SimpleNamespace(values=np.array([weight]))}
    context = replace(context, params=params, weights=weights)
    if not declared:
        with pytest.raises(
            ReceiptAssignmentRefused, match="exceeds eligible weighted mass"
        ):
            TAKEUP_ASSIGN.run(context)
        return
    result = TAKEUP_ASSIGN.run(context)
    assert result.columns["person", "receives_alpha"].tolist() == [True]
    program = result.receipt["programs"]["receives_alpha"]
    row = program["strata"][0]
    assert (
        row["target"],
        row["eligible_mass"],
        row["rate"],
        row["shortfall"],
        row["realized_mass"],
        row["n_receipts"],
    ) == (1.0, weight, 1.0, 1.0 - weight, weight, 1)
    ratio = {0.5: 2.0, 5e-324: None}[weight]
    assert [
        audit[field]
        for audit in (program, row)
        for field in (
            "target_to_eligible_mass_before_exclusions",
            "target_to_remaining_eligible_mass",
        )
    ] == [ratio] * 4
    json.dumps(result.receipt, allow_nan=False)
    canonical_json(result.receipt)


def _huge_target_context(tmp_path, case, declared):
    """Two count targets of 1e308 from the real target compiler.

    "strata" gives one program two strata with one unit-weight adult each.
    "group" puts two programs in one exclusion group, both eligible for both
    adults. Each target fits a float; the program total or the Hall target
    does not.
    """
    if case == "strata":
        strata = {
            "alpha": [
                {"target": "alpha_young", "filter": "young"},
                {"target": "alpha_older", "filter": "older"},
            ]
        }
        groups = []
    else:
        strata = {
            name: [{"target": f"{name}_target", "filter": "adult"}]
            for name in ("alpha", "beta")
        }
        groups = [["receives_alpha", "receives_beta"]]
    contract = {
        "contract_version": 2,
        "seed_column": "seed",
        "default_assignment_profile": "central",
        "assignment_knob": "assignment",
        "assignment_profiles": {"central": {"draw_key_suffix": ""}},
        "programs": [
            {
                "program": name,
                "output": f"receives_{name}",
                "judgment_columns": ["eligible"],
                "judgment_reduction": "any",
                "payment_column": "payment",
                "strata": rows,
                "outside_strata": "not_assigned",
            }
            for name, rows in strata.items()
        ],
        "exclusion_groups": groups,
    }
    if declared:
        contract["target_exceeds_eligible_mass"] = "assign_all_eligible"
    adult = {"column": "age", "operator": "ge", "value": 18}
    bindings = {
        "schema_version": 1,
        "required_inputs": {"person": ["age"]},
        "input_domains": {"person": {"age": {"kind": "nonnegative_integer"}}},
        "columns": [
            {"entity": "person", "name": "adult", "predicates": [adult]},
            {
                "entity": "person",
                "name": "young",
                "predicates": [adult, {"column": "age", "operator": "lt", "value": 30}],
            },
            {
                "entity": "person",
                "name": "older",
                "predicates": [{"column": "age", "operator": "ge", "value": 30}],
            },
        ],
    }
    references = [
        (row["target"], f"receives_{name}", row["filter"])
        for name, rows in strata.items()
        for row in rows
    ]
    document = reference_document(
        [
            reference_row(name, entity="person", measure=measure, filter_=filter_name)
            for name, measure, filter_name in references
        ]
    )
    facts = []
    for name, _, _ in references:
        fact = toy_fact(name, 1e308, entity="person")
        fact["observed_measure"]["unit"] = "count"
        facts.append(fact)
    surface, _ = compile_target_surface(
        write_facts(tmp_path / "facts.jsonl", facts),
        document,
        country="xx",
        references_sha256=digest(document),
    )
    params = {
        "receipt_contract": canonical_text(contract),
        "receipt_contract_sha256": digest(contract),
        "calibration_bindings": canonical_text(bindings),
        "calibration_bindings_sha256": digest(bindings),
        "profile_selection": canonical_text({"assignment": "central"}),
    }
    node = Node(
        "receipt",
        TAKEUP_ASSIGN.ref,
        population="open",
        params=params,
        outputs=tuple(Owned("person", f"receives_{name}", "bool") for name in strata),
    )
    return KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "age": [20, 40],
                    "seed": [0.2, 0.8],
                    "eligible": [1, 1],
                    "payment": [1, 1],
                }
            )
        },
        weights={"person": SimpleNamespace(values=np.ones(2))},
        strata=pd.DataFrame(),
        params=node.params,
        rng=np.random.default_rng(1),
        artifacts={
            "surface": ArtifactValue(
                payload=surface,
                type=TARGET_SURFACE_TYPE,
                key="0" * 64,
                producer_key="0" * 64,
                numerics=NumericScope(),
            )
        },
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("strata", "intermediate overflow in fsum"),
        ("group", "integer division result too large for a float"),
    ],
)
def test_v2_kernel_records_a_target_total_too_large_for_a_float_as_null(
    tmp_path, case, message
):
    with pytest.raises(OverflowError, match=message):
        TAKEUP_ASSIGN.run(_huge_target_context(tmp_path / "plain", case, False))
    result = TAKEUP_ASSIGN.run(_huge_target_context(tmp_path / "declared", case, True))
    canonical_json(result.receipt)
    json.dumps(result.receipt, allow_nan=False)
    assert result.receipt["status"] == "completed"
    programs = result.receipt["programs"]
    if case == "strata":
        assert result.columns["person", "receives_alpha"].tolist() == [True, True]
        program = programs["receives_alpha"]
        assert [
            program[field]
            for field in (
                "target",
                "target_to_eligible_mass_before_exclusions",
                "target_to_remaining_eligible_mass",
            )
        ] == [None, None, None]
        assert [
            (row["target"], row["rate"], row["shortfall"], row["n_receipts"])
            for row in program["strata"]
        ] == [(1e308, 1.0, 1e308 - 1.0, 1)] * 2
        return
    assert result.columns["person", "receives_alpha"].tolist() == [True, True]
    assert result.columns["person", "receives_beta"].tolist() == [False, False]
    (group,) = result.receipt["exclusion_groups"]
    assert group["status"] == "completed"
    assert group["feasibility"]["violations"] == [
        {
            "programs": ["receives_alpha", "receives_beta"],
            "filter": "adult",
            "filters": ["adult", "adult"],
            "target": None,
            "eligible_mass": 2.0,
        }
    ]
    assert [
        (row["shortfall"], row["n_receipts"])
        for name in ("receives_alpha", "receives_beta")
        for row in programs[name]["strata"]
    ] == [(1e308 - 2.0, 2), (1e308, 0)]
