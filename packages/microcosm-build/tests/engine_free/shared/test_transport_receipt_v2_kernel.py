"""Receipt surface validation and the reserved declared-unit dispatch seam."""

import copy
import json
import math
import sys
from dataclasses import replace
from fractions import Fraction
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
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactValue,
    KernelContext,
    Node,
    NumericScope,
    Owned,
    Population,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.executor import _validate_result
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


def _huge_target_context(tmp_path, case, declared, *, target=1e308):
    """Two count targets, 1e308 unless given, from the real target compiler.

    "strata" gives one program two strata with one unit-weight adult each.
    "group" puts two programs in one exclusion group, both eligible for both
    adults. Each target fits a float. For the default target and the largest
    float, the program total or the Hall target does not.
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
        fact = toy_fact(name, target, entity="person")
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


# Eight weights from an independent review. The frame and the population accept
# them, and fsum of all eight is the largest float. The seven adults' exact sum,
# 0.36 ULP above it, rounds to it too, yet fsum raises on the adults: a partial
# sum overflows on the way.
_NEAR_MAX_WEIGHTS = [
    float.fromhex(value)
    for value in (
        "0x1.5b6b7a2f4dddbp+997",
        "0x1.1455647cb7a32p+984",
        "0x1.7f841fa0aca2ep+984",
        "0x1.c8aa803d47c02p+1005",
        "0x1.85fa197ec05e2p+1018",
        "0x1.9e7572c2e19f0p+1023",
        "0x1.69226dbae147bp+961",
        "0x1.556927bea121dp+1021",
    )
]
# Six young adults, one child outside both strata, then one older adult.
_NEAR_MAX_AGES = [20, 20, 20, 20, 20, 20, 15, 40]


def _exact(values):
    return float(sum(map(Fraction, values), Fraction()))


def _near_max_frame():
    count = len(_NEAR_MAX_AGES)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1),
            "person_household_id": np.ones(count, dtype=np.int64),
            "age": _NEAR_MAX_AGES,
            "seed": np.arange(1, count + 1) / 10,
            "eligible": np.ones(count, dtype=np.int64),
            "payment": np.ones(count, dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, "household": pd.DataFrame({"household_id": [1]})},
        EntitySchema(group_entities=("household",)),
        {"person": Weights(np.array(_NEAR_MAX_WEIGHTS), WeightKind.DESIGN)},
    )


def _near_max_audit(target):
    """The audit under the rule, with every mass the exact sum rounded."""
    largest = sys.float_info.max
    young, older = _exact(_NEAR_MAX_WEIGHTS[:6]), _NEAR_MAX_WEIGHTS[7]
    # The largest target caps both strata; 1e300 sits within both masses.
    capped = target > young
    strata = []
    for name, filter_name, mass, count in (
        ("alpha_young", "young", young, 6),
        ("alpha_older", "older", older, 1),
    ):
        ratio = target / mass
        row = {
            "target_name": name,
            "target": target,
            "filter": filter_name,
            "eligible_mass": mass,
            "eligible_mass_before_exclusions": mass,
            "target_to_eligible_mass_before_exclusions": ratio,
            "target_to_remaining_eligible_mass": ratio,
            "rate": 1.0 if capped else ratio,
        }
        if capped:
            row["shortfall"] = target - mass
        row.update(
            {
                "realized_mass": mass if capped else 0.0,
                "n_receipts": count if capped else 0,
                "numerical_check": {
                    "passed": True,
                    "relative_tolerance": 1e-12,
                    "expected_mass": mass if capped else ratio * mass,
                },
            }
        )
        strata.append(row)
    # Two largest targets total more than a float holds, so None.
    total = None if capped else 2 * target
    ratio = None if capped else total / largest
    program = {
        "program": "alpha",
        "target": total,
        "eligible_mass_before_exclusions": largest,
        "eligible_mass": largest,
        "target_to_eligible_mass_before_exclusions": ratio,
        "target_to_remaining_eligible_mass": ratio,
        "strata": strata,
        "outside_strata": {"n_eligible": 1, "eligible_mass": _NEAR_MAX_WEIGHTS[6]},
        "realized_mass": largest if capped else 0.0,
        "n_receipts": 7 if capped else 0,
    }
    return {
        "assignment_profile": "central",
        "exclusion_order": "declared",
        "status": "completed",
        "programs": {"receives_alpha": program},
        "exclusion_groups": [],
    }


def _near_max_group_audit():
    """The group's audit under the rule for the largest target.

    The adults' mass rounds to the largest float, the target, so alpha's rate
    is 1 and it takes every adult. Beta, left with none, caps, and the pair's
    Hall target overflows.
    """
    largest, order = sys.float_info.max, ["receives_alpha", "receives_beta"]
    programs = {}
    for name, mass, ratio, count in (
        ("alpha", largest, 1.0, 7),
        ("beta", 0.0, None, 0),
    ):
        row = {
            "target_name": f"{name}_target",
            "target": largest,
            "filter": "adult",
            "eligible_mass": mass,
            "eligible_mass_before_exclusions": largest,
            "target_to_eligible_mass_before_exclusions": 1.0,
            "target_to_remaining_eligible_mass": ratio,
            "rate": 1.0,
        }
        if not mass:
            row["shortfall"] = largest
        row.update(
            {
                "realized_mass": mass,
                "n_receipts": count,
                "numerical_check": {
                    "passed": True,
                    "relative_tolerance": 1e-12,
                    "expected_mass": mass,
                },
            }
        )
        programs[f"receives_{name}"] = {
            "program": name,
            "target": largest,
            "eligible_mass_before_exclusions": largest,
            "eligible_mass": mass,
            "target_to_eligible_mass_before_exclusions": 1.0,
            "target_to_remaining_eligible_mass": ratio,
            "strata": [row],
            "outside_strata": {"n_eligible": 1, "eligible_mass": _NEAR_MAX_WEIGHTS[6]},
            "realized_mass": mass,
            "n_receipts": count,
        }
    # Everyone is eligible for both programs; the adults' overlap mass is the
    # one fsum raises on.
    overlaps = [
        {
            "programs": pair,
            "filter": filter_name,
            "n_eligible": count,
            "eligible_mass": largest,
        }
        for pair in (order, order[::-1])
        for filter_name, count in ((None, 8), ("adult", 7))
    ]
    violation = {
        "programs": order,
        "filter": "adult",
        "filters": ["adult", "adult"],
        "target": None,
        "eligible_mass": largest,
    }
    return {
        "assignment_profile": "central",
        "exclusion_order": "declared",
        "status": "completed",
        "programs": programs,
        "exclusion_groups": [
            {
                "order": order,
                "status": "completed",
                "overlaps": overlaps,
                "feasibility": {
                    "violations": [violation],
                    "feasible": False,
                    "relative_tolerance": 1e-12,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("case", "target"),
    [
        ("strata", sys.float_info.max),
        ("strata", 1e300),
        ("group", sys.float_info.max),
    ],
    ids=["capped", "within", "group"],
)
def test_v2_kernel_masses_adults_whose_fsum_overflows_under_the_mass_rule(
    tmp_path, case, target
):
    weights = np.array(_NEAR_MAX_WEIGHTS)
    adults = np.array(_NEAR_MAX_AGES) >= 18
    # Keep the case honest: every weight passes fsum, the adults do not, and
    # their exact sum still rounds to the largest float.
    assert math.fsum(weights.tolist()) == sys.float_info.max
    with pytest.raises(OverflowError, match="intermediate overflow in fsum"):
        math.fsum(weights[adults].tolist())
    assert _exact(weights[adults].tolist()) == sys.float_info.max
    frame = _near_max_frame()
    population = Population.from_frame(frame, "open")

    def context(declared):
        base = _huge_target_context(
            tmp_path / str(declared), case, declared, target=target
        )
        return replace(
            base,
            tables={"person": frame.table("person")},
            weights={"person": frame.weights_for("person")},
        )

    # Without the rule, the OverflowError base 9189f745 raises on these inputs:
    # from the program's target total when capped, from the adults' mass when
    # within, and from the adults' overlap mass in the group.
    with pytest.raises(OverflowError) as raised:
        TAKEUP_ASSIGN.run(context(False))
    assert str(raised.value) == "intermediate overflow in fsum"
    declared = context(True)
    result = TAKEUP_ASSIGN.run(declared)
    # The largest target gives alpha every adult; in the group, beta gets none.
    capped = target == sys.float_info.max
    flags = {"receives_alpha": adults.tolist() if capped else [False] * len(adults)}
    expected = _near_max_audit(target)
    if case == "group":
        flags["receives_beta"] = [False] * len(adults)
        expected = _near_max_group_audit()
    assert {name: result.columns["person", name].tolist() for name in flags} == flags
    canonical_json(result.receipt)
    assert json.dumps(result.receipt, allow_nan=False) == json.dumps(expected)
    _validate_result(declared.node, TAKEUP_ASSIGN.capabilities, result, population)
