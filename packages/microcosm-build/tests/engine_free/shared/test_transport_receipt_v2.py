"""Shared receipt contracts on synthetic people, never country declarations.

The v1 oracle below freezes the assignment operation at 4f333f2c; the toy
contract is loaded from the existing composed-graph fixture. V2 deliberately
requires disjoint masks on the actual people rather than overlap precedence.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.transport import receipt_contract as receipt_module
from microcosm.build.transport.column_kernels import (
    assign_receipts,
    validate_receipt_contract,
)
from microcosm.build.transport.receipt_contract import (
    ReceiptAssignmentRefused,
    receipt_draw_key,
    reversed_receipt_groups,
)
from microcosm.frame.concepts import derive_take_up_draws
from microcosm.frame.input_closure import resolve_judgments
from test_support.microcosm_build.transport_composed import make_composed_fixture

PROPERTY = settings(
    max_examples=18,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _program(name="alpha", *, stratified=False):
    row = {
        "program": name,
        "output": f"receives_{name}",
        "judgment_columns": ["eligible"],
        "judgment_reduction": "any",
        "payment_column": "payment",
    }
    if stratified:
        row.update(
            strata=[
                {"target": f"{name}_young", "filter": "young"},
                {"target": f"{name}_older", "filter": "older"},
            ],
            outside_strata="not_assigned",
        )
    else:
        row["target"] = f"{name}_total"
    return row


def _contract(*programs, groups=None):
    return {
        "contract_version": 2,
        "seed_column": "take_up_seed",
        "default_assignment_profile": "central",
        "assignment_knob": "assignment",
        "assignment_profiles": {
            "central": {"draw_key_suffix": ""},
            "variant": {"draw_key_suffix": "variant"},
            "reversed": {"draw_key_suffix": "", "exclusion_order": "reversed"},
        },
        "programs": list(programs) or [_program()],
        "exclusion_groups": [] if groups is None else groups,
        "unmodelled_programs": ["unsupported"],
    }


def _people(ages=(19, 23, 31, 45, 61, 72)):
    count = len(ages)
    return pd.DataFrame(
        {
            "person_id": np.arange(count, dtype=np.int64) + 1,
            "age": list(ages),
            "take_up_seed": np.linspace(0.01, 0.99, count),
            "eligible": np.ones(count, dtype=np.int64),
            "alternate": np.ones(count, dtype=np.int64),
            "payment": np.ones(count),
        }
    )


def _bindings():
    return {
        "schema_version": 1,
        "required_inputs": {"person": ["age"]},
        "input_domains": {"person": {"age": {"kind": "nonnegative_integer"}}},
        "columns": [
            {
                "entity": "person",
                "name": "young",
                "predicates": [
                    {"column": "age", "operator": "ge", "value": 18},
                    {"column": "age", "operator": "lt", "value": 30},
                ],
            },
            {
                "entity": "person",
                "name": "older",
                "predicates": [
                    {"column": "age", "operator": "ge", "value": 30},
                    {"column": "age", "operator": "lt", "value": 65},
                ],
            },
        ],
    }


def _assign(
    person,
    contract=None,
    *,
    weights=None,
    targets=None,
    bindings=None,
    profile="central",
):
    return assign_receipts(
        person,
        _contract() if contract is None else contract,
        weights=np.ones(len(person)) if weights is None else np.asarray(weights),
        targets={"alpha_total": 0} if targets is None else targets,
        bindings=_bindings() if bindings is None else bindings,
        knobs={"assignment": profile},
    )


@pytest.mark.parametrize("location", ["contract", "program", "stratum", "profile"])
def test_v2_closed_key_sets_refuse_unknown_fields(location):
    contract = _contract(_program(stratified=True))
    destination = {
        "contract": contract,
        "program": contract["programs"][0],
        "stratum": contract["programs"][0]["strata"][0],
        "profile": contract["assignment_profiles"]["central"],
    }[location]
    destination["unknown_field"] = "typo"
    with pytest.raises(ValueError, match="unknown|fields|declares"):
        validate_receipt_contract(contract)


@pytest.mark.parametrize("version", [1, 3, True, "2", None])
def test_v2_refuses_unknown_or_noninteger_contract_version(version):
    contract = _contract()
    contract["contract_version"] = version
    with pytest.raises(ValueError, match="contract_version|version"):
        validate_receipt_contract(contract)


@pytest.mark.parametrize(
    "columns", [[], ["eligible", "eligible"], [""], [None], "eligible"]
)
def test_v2_judgment_columns_are_nonempty_unique_names(columns):
    contract = _contract()
    contract["programs"][0]["judgment_columns"] = columns
    with pytest.raises(ValueError, match="judgment|non-empty|distinct|unique"):
        validate_receipt_contract(contract)


@pytest.mark.parametrize("reduction", ["some", "majority", "ANY", None])
def test_v2_refuses_unsupported_judgment_reduction(reduction):
    contract = _contract()
    contract["programs"][0]["judgment_reduction"] = reduction
    with pytest.raises(ValueError, match="reduction"):
        validate_receipt_contract(contract)


@pytest.mark.parametrize(("reduction", "determined"), [("any", 1), ("all", -1)])
def test_undetermined_judgment_refuses_before_reduction(reduction, determined):
    person = _people((20,))
    person["eligible"] = determined
    person["alternate"] = 0
    program = _program()
    program.update(
        judgment_columns=["eligible", "alternate"], judgment_reduction=reduction
    )
    with pytest.raises(ValueError, match="[Uu]ndetermined"):
        _assign(person, _contract(program))


@pytest.mark.parametrize(
    ("reduction", "expected"), [("any", [True, True]), ("all", [False, True])]
)
def test_v2_reduces_only_determined_judgments(reduction, expected):
    person = _people((20, 21))
    person["eligible"] = [1, 1]
    person["alternate"] = [-1, 1]
    program = _program()
    program.update(
        judgment_columns=["eligible", "alternate"], judgment_reduction=reduction
    )
    out, _ = _assign(person, _contract(program), targets={"alpha_total": sum(expected)})
    assert out["receives_alpha"].tolist() == expected


@pytest.mark.parametrize("field", ["program", "output", "target"])
def test_v2_contract_identities_are_unique(field):
    first, second = _program("alpha"), _program("beta")
    second[field] = first[field]
    with pytest.raises(ValueError, match="distinct|unique|repeat"):
        validate_receipt_contract(_contract(first, second))


def test_v2_targets_are_unique_across_strata_and_programs():
    first, second = _program("alpha", stratified=True), _program("beta")
    second["target"] = first["strata"][0]["target"]
    with pytest.raises(
        ValueError, match="target.*distinct|target.*unique|target.*repeat"
    ):
        validate_receipt_contract(_contract(first, second))


@pytest.mark.parametrize("defect", ["both", "neither"])
def test_v2_program_declares_target_xor_strata(defect):
    program = _program(stratified=True)
    if defect == "both":
        program["target"] = "total"
    else:
        del program["strata"]
    with pytest.raises(ValueError, match="target.*strata|strata.*target"):
        validate_receipt_contract(_contract(program))


@pytest.mark.parametrize("outside", [None, "assign", "false"])
def test_v2_outside_strata_policy_is_explicit(outside):
    program = _program(stratified=True)
    if outside is None:
        del program["outside_strata"]
    else:
        program["outside_strata"] = outside
    with pytest.raises(ValueError, match="outside_strata|not_assigned"):
        validate_receipt_contract(_contract(program))


@pytest.mark.parametrize(
    "field", ["entitlement_entity", "payee", "draw_unit", "projection"]
)
def test_reserved_person_defaults_refuse_household(field):
    contract = _contract()
    _, rows, _ = validate_receipt_contract(contract)
    assert rows[0][field] == "person"
    contract["programs"][0][field] = "household"
    with pytest.raises(
        ValueError, match="non-person.*reserved for the declared-unit receipt extension"
    ):
        validate_receipt_contract(contract)


@pytest.mark.parametrize("profile", ["missing", None])
def test_v2_resolves_only_declared_assignment_profiles(profile):
    with pytest.raises(ValueError, match="profile|assignment"):
        _assign(_people(), profile=profile)


def test_v2_assignment_knob_must_be_present():
    with pytest.raises(ValueError, match="knob|assignment"):
        assign_receipts(
            _people(),
            _contract(),
            weights=np.ones(6),
            targets={"alpha_total": 0},
            knobs={},
        )


@pytest.mark.parametrize("order", ["random", None, True])
def test_v2_exclusion_order_is_closed(order):
    contract = _contract()
    contract["assignment_profiles"]["central"]["exclusion_order"] = order
    with pytest.raises(ValueError, match="exclusion_order|declared.*reversed"):
        validate_receipt_contract(contract)


def test_overlapping_strata_refuse_and_name_both_masks():
    bindings = _bindings()
    bindings["columns"][1]["predicates"] = [
        {"column": "age", "operator": "ge", "value": 18}
    ]
    with pytest.raises(
        ValueError, match="[Oo]verlap.*young.*older|young.*older.*overlap"
    ):
        _assign(
            _people(),
            _contract(_program(stratified=True)),
            targets={"alpha_young": 0, "alpha_older": 0},
            bindings=bindings,
        )


@pytest.mark.parametrize("invalid_age", [None, -1, 20.5, "unknown"])
def test_receipt_predicate_inputs_are_validated_before_masking(invalid_age):
    person = _people((20,))
    person["age"] = [invalid_age]
    with pytest.raises(
        ValueError,
        match="Calibration input person.age.*missing|Calibration input person.age.*domain",
    ):
        _assign(
            person,
            _contract(_program(stratified=True)),
            targets={"alpha_young": 0, "alpha_older": 0},
        )


def test_receipt_predicates_require_declared_validated_inputs():
    bindings = _bindings()
    bindings["columns"][0]["predicates"][0]["column"] = "payment"
    with pytest.raises(ValueError, match="validated person input"):
        _assign(
            _people(),
            _contract(_program(stratified=True)),
            targets={"alpha_young": 0, "alpha_older": 0},
            bindings=bindings,
        )


def test_stratum_mask_uses_its_declared_predicate():
    person = _people((20, 40))
    out, audit = _assign(
        person,
        _contract(_program(stratified=True)),
        weights=[2, 3],
        targets={"alpha_young": 2, "alpha_older": 0},
    )
    assert out["receives_alpha"].tolist() == [True, False]
    assert [
        row["eligible_mass"] for row in audit["programs"]["receives_alpha"]["strata"]
    ] == [2, 3]


def test_each_stratum_has_its_own_rate_and_realized_mass():
    person = _people((20, 21, 40, 41))
    weights = np.array([2, 2, 1, 1])
    out, audit = _assign(
        person,
        _contract(_program(stratified=True)),
        weights=weights,
        targets={"alpha_young": 1, "alpha_older": 1},
    )
    rows = audit["programs"]["receives_alpha"]["strata"]
    assert [row["filter"] for row in rows] == ["young", "older"]
    assert [row["rate"] for row in rows] == [0.25, 0.5]
    draws = derive_take_up_draws(
        person["take_up_seed"].to_numpy(), receipt_draw_key("alpha", "")
    )
    expected = draws < np.array([0.25, 0.25, 0.5, 0.5])
    assert out["receives_alpha"].tolist() == expected.tolist()
    for row, mask in zip(
        rows,
        [np.array([True, True, False, False]), np.array([False, False, True, True])],
        strict=True,
    ):
        assert row["realized_mass"] == math.fsum(weights[mask & expected].tolist())
        assert row["numerical_check"]["passed"] is True
        assert row["numerical_check"]["relative_tolerance"] > 0


def test_rate_uses_mass_remaining_after_earlier_exclusion():
    person = _people((20, 21))
    person["alternate"] = [1, -1]
    first, second = _program("alpha"), _program("beta")
    first["judgment_columns"] = ["alternate"]
    contract = _contract(first, second, groups=[["receives_alpha", "receives_beta"]])
    out, audit = _assign(person, contract, targets={"alpha_total": 1, "beta_total": 1})
    assert out["receives_alpha"].tolist() == [True, False]
    assert out["receives_beta"].tolist() == [False, True]
    row = audit["programs"]["receives_beta"]["strata"][0]
    assert row["eligible_mass"] == row["rate"] == 1
    assert row["target_to_eligible_mass_before_exclusions"] == 0.5
    assert row["target_to_remaining_eligible_mass"] == 1
    program = audit["programs"]["receives_beta"]
    assert program["target_to_eligible_mass_before_exclusions"] == 0.5
    assert program["target_to_remaining_eligible_mass"] == 1


def test_v2_existing_exclusion_columns_must_be_distinct():
    program = _program()
    program["exclude_columns"] = ["already_receives", "already_receives"]
    with pytest.raises(
        ValueError, match=r"^Receipt exclude_columns must be distinct columns\.$"
    ):
        validate_receipt_contract(_contract(program))


@pytest.mark.parametrize(
    "values",
    [pd.Series([1, 0]), pd.Series([True, None], dtype="boolean")],
    ids=["non_boolean", "null"],
)
def test_v2_existing_exclusion_columns_must_be_nonnull_booleans(values):
    person = _people((20, 21))
    person["already_receives"] = values
    program = _program()
    program["exclude_columns"] = ["already_receives"]
    with pytest.raises(
        ValueError, match=r"^Existing receipt exclusions must be non-null booleans\.$"
    ):
        _assign(person, _contract(program))


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_rate_uses_mass_remaining_after_existing_exclusion(dtype):
    person = _people((20, 21, 40))
    person["already_receives"] = pd.Series([True, False, False], dtype=dtype)
    person["another_receipt"] = pd.Series([False, True, False], dtype=dtype)
    program = _program()
    program["exclude_columns"] = ["already_receives", "another_receipt"]
    out, audit = _assign(
        person, _contract(program), weights=[2, 3, 5], targets={"alpha_total": 5}
    )
    assert out["receives_alpha"].tolist() == [False, False, True]
    detail = audit["programs"]["receives_alpha"]["strata"][0]
    assert detail["eligible_mass_before_exclusions"] == 10
    assert detail["eligible_mass"] == 5
    assert detail["rate"] == 1
    assert detail["target_to_eligible_mass_before_exclusions"] == 0.5
    assert detail["realized_mass"] == 5
    assert person["already_receives"].tolist() == [True, False, False]
    assert person["another_receipt"].tolist() == [False, True, False]


def test_programme_ratios_sum_strata_and_exclude_the_outside_remainder():
    _, audit = _assign(
        _people((20, 40, 70)),
        _contract(_program(stratified=True)),
        weights=[2, 3, 5],
        targets={"alpha_young": 1, "alpha_older": 1},
    )
    program = audit["programs"]["receives_alpha"]
    assert (
        program["target"] == math.fsum(row["target"] for row in program["strata"]) == 2
    )
    assert (
        program["eligible_mass"]
        == math.fsum(row["eligible_mass"] for row in program["strata"])
        == 5
    )
    assert program["target_to_eligible_mass_before_exclusions"] == 0.4
    assert program["target_to_remaining_eligible_mass"] == 0.4
    assert program["outside_strata"]["eligible_mass"] == 5
    assert program["realized_mass"] == math.fsum(
        row["realized_mass"] for row in program["strata"]
    )


def test_positive_payment_screen_precedes_target_mass():
    person = _people((20, 21))
    person["payment"] = [0, 1]
    with pytest.raises(ValueError, match="exceeds eligible weighted mass"):
        _assign(person, targets={"alpha_total": 2})


def test_target_above_stratum_mass_refuses_even_when_total_fits():
    with pytest.raises(ValueError, match="alpha_young.*exceeds eligible weighted mass"):
        _assign(
            _people((20, 40)),
            _contract(_program(stratified=True)),
            targets={"alpha_young": 2, "alpha_older": 0},
        )


def test_outside_strata_remains_unassigned_and_is_disclosed():
    out, audit = _assign(
        _people((10, 20, 70)),
        _contract(_program(stratified=True)),
        weights=[2, 3, 5],
        targets={"alpha_young": 3, "alpha_older": 0},
    )
    assert out["receives_alpha"].tolist() == [False, True, False]
    assert audit["programs"]["receives_alpha"]["outside_strata"] == {
        "n_eligible": 2,
        "eligible_mass": 7,
    }


def test_overlap_audit_records_overall_and_stratum_counts_before_assignment():
    first, second = (
        _program("alpha", stratified=True),
        _program("beta", stratified=True),
    )
    contract = _contract(first, second, groups=[["receives_alpha", "receives_beta"]])
    _, audit = _assign(
        _people((20, 40, 70)),
        contract,
        weights=[2, 3, 5],
        targets={"alpha_young": 0, "alpha_older": 0, "beta_young": 0, "beta_older": 0},
    )
    group = audit["exclusion_groups"][0]
    overlaps = {
        row["filter"]: row
        for row in group["overlaps"]
        if row["programs"] == ["receives_alpha", "receives_beta"]
    }
    assert (overlaps[None]["n_eligible"], overlaps[None]["eligible_mass"]) == (3, 10)
    assert (overlaps["young"]["n_eligible"], overlaps["young"]["eligible_mass"]) == (
        1,
        2,
    )
    assert (overlaps["older"]["n_eligible"], overlaps["older"]["eligible_mass"]) == (
        1,
        3,
    )
    assert group["status"] == audit["status"] == "completed"


def test_hall_diagnostic_checks_pairs_when_each_singleton_fits():
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    with pytest.raises(ReceiptAssignmentRefused) as refused:
        _assign(
            _people((20, 21)), contract, targets={"alpha_total": 2, "beta_total": 1}
        )
    violations = refused.value.audit["exclusion_groups"][0]["feasibility"]["violations"]
    assert len(violations) == 1
    assert violations[0]["programs"] == ["receives_alpha", "receives_beta"]
    assert violations[0]["target"] == 3
    assert violations[0]["eligible_mass"] == 2


def test_hall_violation_is_a_diagnostic_and_does_not_gate_assignment():
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    candidates = np.linspace(0, np.nextafter(1.0, 0), 1000)
    alpha = derive_take_up_draws(candidates, receipt_draw_key("alpha", ""))
    beta = derive_take_up_draws(candidates, receipt_draw_key("beta", ""))
    person = _people((20, 21))
    person["take_up_seed"] = candidates[(alpha >= 0.75) & (beta >= 0.75)][:2]
    out, audit = _assign(
        person, contract, targets={"alpha_total": 1.5, "beta_total": 1.5}
    )
    assert not out[["receives_alpha", "receives_beta"]].to_numpy().any()
    assert audit["status"] == "completed"
    group = audit["exclusion_groups"][0]
    assert group["status"] == "completed"
    assert group["feasibility"]["feasible"] is False
    (violation,) = group["feasibility"]["violations"]
    assert violation["programs"] == ["receives_alpha", "receives_beta"]
    assert violation["target"] == 3
    assert violation["eligible_mass"] == 2


def test_reversed_profile_refusal_is_raised_with_its_audit_result():
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    with pytest.raises(
        ReceiptAssignmentRefused, match="exceeds eligible weighted mass"
    ) as refused:
        _assign(
            _people((20, 21)),
            contract,
            targets={"alpha_total": 1, "beta_total": 2},
            profile="reversed",
        )
    audit = refused.value.audit
    assert audit["assignment_profile"] == "reversed"
    assert audit["exclusion_order"] == "reversed"
    assert audit["status"] == "refused"
    assert audit["exclusion_groups"][0]["order"] == ["receives_beta", "receives_alpha"]
    assert audit["exclusion_groups"][0]["status"] == "refused"


@pytest.mark.parametrize("first_refuses", [True, False])
def test_refusal_never_marks_an_unfinished_group_completed(first_refuses):
    names = ["alpha", "beta", "gamma", "delta"]
    contract = _contract(
        *[_program(name) for name in names],
        groups=[
            ["receives_alpha", "receives_beta"],
            ["receives_gamma", "receives_delta"],
        ],
    )
    targets = {f"{name}_total": 0 for name in names}
    failed_pair = names[:2] if first_refuses else names[2:]
    targets[f"{failed_pair[0]}_total"] = 2
    targets[f"{failed_pair[1]}_total"] = 1
    with pytest.raises(ReceiptAssignmentRefused) as refused:
        _assign(_people((20, 21)), contract, targets=targets)
    statuses = [group["status"] for group in refused.value.audit["exclusion_groups"]]
    assert statuses == (
        ["refused", "refused"] if first_refuses else ["completed", "refused"]
    )


def test_reversed_profile_reverses_assignment_priority_only():
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    candidates = np.linspace(0, np.nextafter(1.0, 0), 1000)
    alpha = derive_take_up_draws(candidates, receipt_draw_key("alpha", ""))
    beta = derive_take_up_draws(candidates, receipt_draw_key("beta", ""))
    person = _people((20, 21))
    person["take_up_seed"] = [
        candidates[(alpha < 0.5) & (beta < 0.5)][0],
        candidates[(alpha >= 0.5) & (beta >= 0.5)][0],
    ]
    central, _ = _assign(person, contract, targets={"alpha_total": 1, "beta_total": 1})
    reversed_out, audit = _assign(
        person,
        contract,
        targets={"alpha_total": 1, "beta_total": 1},
        profile="reversed",
    )
    assert (
        central["receives_alpha"].tolist()
        == reversed_out["receives_beta"].tolist()
        == [True, False]
    )
    assert (
        central["receives_beta"].tolist()
        == reversed_out["receives_alpha"].tolist()
        == [False, True]
    )
    assert audit["exclusion_groups"][0]["order"] == ["receives_beta", "receives_alpha"]


def test_profile_suffix_changes_the_declared_draw_stream():
    person = _people(tuple(range(20, 40)))
    out, audit = _assign(person, targets={"alpha_total": 10}, profile="variant")
    expected = (
        derive_take_up_draws(
            person["take_up_seed"].to_numpy(), receipt_draw_key("alpha", "variant")
        )
        < 0.5
    )
    assert out["receives_alpha"].tolist() == expected.tolist()
    assert audit["assignment_profile"] == "variant"


def test_draw_keys_distinguish_adversarial_program_suffix_pairs():
    pairs = [
        ("a", "bc"),
        ("ab", "c"),
        ("a:b", "c"),
        ("a", "b:c"),
        ("a\0b", "c"),
        ("a", "b\0c"),
        ('["a",', '"b"]'),
        ("", "a"),
    ]
    assert len({receipt_draw_key(*pair) for pair in pairs}) == len(pairs)


def test_contract_refuses_a_draw_key_collision(monkeypatch):
    monkeypatch.setattr(
        receipt_module, "receipt_draw_key", lambda program, suffix: "collision"
    )
    with pytest.raises(ValueError, match="[Dd]raw.*collision|collision.*draw"):
        validate_receipt_contract(_contract())


@PROPERTY
@given(
    st.lists(
        st.tuples(
            st.text(alphabet='a:/|"[]\\', max_size=12),
            st.text(alphabet='a:/|"[]\\', max_size=12),
        ),
        min_size=1,
        max_size=15,
        unique=True,
    )
)
@example(pairs=[("a:b", "c"), ("a", "b:c"), ("", ":"), (":", "")])
def test_property_draw_key_is_injective(pairs):
    assert len({receipt_draw_key(*pair) for pair in pairs}) == len(pairs)


@PROPERTY
@given(
    st.lists(
        st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=5, unique=True),
        min_size=0,
        max_size=3,
    )
)
def test_property_reversing_exclusion_order_twice_is_declared(groups):
    reversed_once = reversed_receipt_groups(groups, "reversed")
    assert reversed_receipt_groups(
        reversed_once, "reversed"
    ) == reversed_receipt_groups(groups, "declared")


@PROPERTY
@given(
    st.lists(st.integers(min_value=1, max_value=1000), min_size=2, max_size=12),
    st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_property_per_stratum_numerical_check(weights, fraction):
    ages = [20 if index % 2 else 40 for index in range(len(weights))]
    person = _people(tuple(ages))
    masses = {
        "alpha_young": math.fsum(
            weight for age, weight in zip(ages, weights, strict=True) if age < 30
        ),
        "alpha_older": math.fsum(
            weight for age, weight in zip(ages, weights, strict=True) if age >= 30
        ),
    }
    _, audit = _assign(
        person,
        _contract(_program(stratified=True)),
        weights=weights,
        targets={name: mass * fraction for name, mass in masses.items()},
    )
    for row in audit["programs"]["receives_alpha"]["strata"]:
        check = row["numerical_check"]
        assert check["passed"] is True
        assert math.isclose(
            math.fsum([row["rate"] * row["eligible_mass"]]),
            row["target"],
            rel_tol=check["relative_tolerance"],
            abs_tol=0,
        )


@PROPERTY
@given(
    st.lists(st.integers(min_value=0, max_value=90), min_size=1, max_size=15), st.data()
)
def test_property_disjoint_strata_and_outside_partition_eligible_mass(ages, data):
    weights = data.draw(
        st.lists(st.integers(0, 100), min_size=len(ages), max_size=len(ages))
    )
    eligible = data.draw(
        st.lists(st.booleans(), min_size=len(ages), max_size=len(ages))
    )
    person = _people(tuple(ages))
    person["eligible"] = np.where(eligible, 1, -1)
    _, audit = _assign(
        person,
        _contract(_program(stratified=True)),
        weights=weights,
        targets={"alpha_young": 0, "alpha_older": 0},
    )
    program = audit["programs"]["receives_alpha"]
    partition = math.fsum(
        [
            *(row["eligible_mass"] for row in program["strata"]),
            program["outside_strata"]["eligible_mass"],
        ]
    )
    assert partition == math.fsum(
        weight for weight, permitted in zip(weights, eligible, strict=True) if permitted
    )
    assert program["eligible_mass"] == math.fsum(
        row["eligible_mass"] for row in program["strata"]
    )


@settings(max_examples=18, deadline=None, database=None, derandomize=True)
@given(st.integers(1, 15), st.floats(min_value=0, max_value=0.25, allow_nan=False))
@example(count=15, fraction=0.25)
def test_property_exclusion_group_never_assigns_two_flags(count, fraction):
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    try:
        out, _ = _assign(
            _people(tuple(range(20, 20 + count))),
            contract,
            targets={"alpha_total": count * fraction, "beta_total": count * fraction},
        )
    except ReceiptAssignmentRefused as refused:
        assert refused.audit["status"] == "refused"
    else:
        assert (out[["receives_alpha", "receives_beta"]].sum(axis=1) <= 1).all()


def _v1_before_change(person, contract, *, weights=None, targets=None):
    """Freeze the valid-input v1 assignment loop from HEAD 4f333f2c, independently.

    This intentionally does not call the evolving contract validator or receipt
    implementation; generated contracts below are all valid v1 declarations.
    """
    out = person.loc[:, ["person_id"]].copy()
    audit = {}
    for row in contract["programs"]:
        output = row["output"]
        eligible = resolve_judgments(
            person[row["judgment_column"]].to_numpy(), "refuse"
        )
        if row.get("payment_column") is not None:
            eligible &= person[row["payment_column"]].to_numpy(dtype=np.float64) > 0
        for column in row.get("exclude_columns", []):
            eligible &= ~person[column].to_numpy(dtype=bool)
        for group in contract["exclusion_groups"]:
            if output in group:
                for previous in group:
                    if previous in out:
                        eligible &= ~out[previous].to_numpy()
        target = None
        mass = None if weights is None else math.fsum(weights[eligible].tolist())
        if "target" in row:
            target = targets[row["target"]]
            rate = target / mass if mass else target
        else:
            rate = row["rate"]
        draws = derive_take_up_draws(
            person[contract["seed_column"]].to_numpy(), row["program"]
        )
        assigned = eligible & (draws < rate)
        out[output] = assigned
        audit[output] = {
            "program": row["program"],
            "rate": rate,
            "target": target,
            "eligible_mass": mass,
            "n_receipts": int(assigned.sum()),
            "realized_mass": None
            if weights is None
            else math.fsum(weights[assigned].tolist()),
        }
    return out, audit


def _assert_v1_bytes(person, contract, *, weights=None, targets=None):
    before, before_audit = _v1_before_change(
        person, contract, weights=weights, targets=targets
    )
    after, after_audit = assign_receipts(
        person, contract, weights=weights, targets=targets
    )
    assert (
        before.to_records(index=False).tobytes()
        == after.to_records(index=False).tobytes()
    )
    assert (
        json.dumps(before_audit, sort_keys=True, separators=(",", ":")).encode()
        == json.dumps(after_audit, sort_keys=True, separators=(",", ":")).encode()
    )


def test_v1_toy_flags_and_audit_bytes_match_frozen_before_change(tmp_path):
    fixture = make_composed_fixture(tmp_path / "toy")
    contract = fixture.spec["resources"]["receipt_contract"]["receipts"]
    person = _people()
    person["toy_main_eligible"] = [1, -1, 1, 1, -1, 1]
    person["toy_pension_eligible"] = [1, 1, 1, -1, 1, 1]
    person["toy_main_payment"] = [1, 1, 0, 2, -1, 1]
    person["toy_pension_payment"] = [1, 0, 1, 1, 1, 1]
    _assert_v1_bytes(person, contract, weights=np.array([1, 2, 3, 4, 5, 6]))


@PROPERTY
@given(
    st.lists(st.sampled_from([-1, 1]), min_size=1, max_size=15),
    st.floats(min_value=0, max_value=1, allow_nan=False),
    st.data(),
)
def test_property_v1_flags_and_audit_bytes_match_before_change(judgments, rate, data):
    count = len(judgments)
    person = _people(tuple(range(20, 20 + count)))
    person["eligible"] = judgments
    person["existing"] = data.draw(
        st.lists(st.booleans(), min_size=count, max_size=count)
    )
    person["payment"] = data.draw(
        st.lists(st.integers(-2, 2), min_size=count, max_size=count)
    )
    weights = np.asarray(
        data.draw(st.lists(st.integers(0, 100), min_size=count, max_size=count)),
        dtype=np.float64,
    )
    contract = {
        "seed_column": "take_up_seed",
        "programs": [
            {
                "program": "toy.alpha",
                "output": "receives_alpha",
                "judgment_column": "eligible",
                "payment_column": "payment",
                "exclude_columns": ["existing"],
                "rate": rate,
            },
            {
                "program": "toy.beta",
                "output": "receives_beta",
                "judgment_column": "eligible",
                "rate": rate,
            },
        ],
        "exclusion_groups": [["receives_alpha", "receives_beta"]],
    }
    _assert_v1_bytes(person, contract, weights=weights)
    target_contract = copy.deepcopy(contract)
    target_contract["programs"] = [target_contract["programs"][0]]
    target_contract["exclusion_groups"] = []
    del target_contract["programs"][0]["rate"]
    target_contract["programs"][0]["target"] = "toy_target"
    permitted = (
        (person["eligible"].to_numpy() == 1)
        & (person["payment"].to_numpy() > 0)
        & ~person["existing"].to_numpy()
    )
    target = math.fsum(weights[permitted].tolist()) * rate
    _assert_v1_bytes(
        person, target_contract, weights=weights, targets={"toy_target": target}
    )


def test_hall_diagnostic_checks_mixed_unstratified_and_stratified_subsets():
    people = _people((19, 23, 31))
    people["eligible"] = [1, 1, -1]
    alpha, beta = _program(), _program("beta", stratified=True)
    beta["judgment_columns"] = ["alternate"]
    contract = _contract(alpha, beta, groups=[["receives_alpha", "receives_beta"]])
    # The two young people cannot supply alpha's 2 and beta.young's 1.
    # beta's separate older pool makes programme-total and individual-filter
    # checks pass, so only a mixed-filter demand subset exposes infeasibility.
    with pytest.raises(ReceiptAssignmentRefused) as refused:
        _assign(
            people,
            contract,
            weights=[1, 1, 100],
            targets={"alpha_total": 2, "beta_young": 1, "beta_older": 0},
        )
    feasibility = refused.value.audit["exclusion_groups"][0]["feasibility"]
    assert feasibility["feasible"] is False
    assert any(
        violation["programs"] == ["receives_alpha", "receives_beta"]
        and violation["filters"] == [None, "young"]
        and violation["target"] == 3
        and violation["eligible_mass"] == 2
        for violation in feasibility["violations"]
    )


@st.composite
def _hall_cases(draw):
    count = draw(st.integers(0, 12))
    size = draw(st.integers(0, 8))
    # Binary fractions give varied capacities; the extremes (subnormal, 1/3,
    # 1e300) stress the shared integer denominator. The rational reference
    # below is exact for every float.
    capacity = st.one_of(
        st.integers(0, 32).map(lambda value: value / 8),
        st.sampled_from([5e-324, 2.2250738585072014e-308, 1e-300, 1 / 3, 1e300]),
    )
    weights = np.asarray(draw(st.lists(capacity, min_size=count, max_size=count)))
    demands = [
        (
            f"p{index}",
            f"f{index % 3}",
            draw(capacity),
            np.asarray(
                draw(st.lists(st.booleans(), min_size=count, max_size=count)),
                dtype=bool,
            ),
        )
        for index in range(size)
    ]
    return demands, weights


@settings(max_examples=120, deadline=None, database=None, derandomize=True)
@given(_hall_cases())
def test_property_hall_flow_matches_exact_bruteforce(case):
    demands, weights = case
    scale = 1 - Fraction(receipt_module.RECEIPT_RELATIVE_TOLERANCE)

    def deficiency(indices):
        total = sum((Fraction(demands[index][2]) for index in indices), Fraction())
        mask = np.zeros(len(weights), dtype=bool)
        for index in indices:
            mask |= demands[index][3]
        mass = sum((Fraction(float(value)) for value in weights[mask]), Fraction())
        return scale * total - mass

    maximum = max(
        (
            deficiency([i for i in range(len(demands)) if subset & (1 << i)])
            for subset in range(1, 1 << len(demands))
        ),
        default=Fraction(),
    )
    result = receipt_module.hall_diagnostics(demands, weights)
    assert result["feasible"] == (maximum <= 0)
    if result["feasible"]:
        assert result["violations"] == []
    else:
        (violation,) = result["violations"]
        indices = [
            i for i, demand in enumerate(demands) if demand[0] in violation["programs"]
        ]
        assert deficiency(indices) == maximum
        # The reported set is the union of every maximising set.
        maximisers = [
            members
            for subset in range(1, 1 << len(demands))
            for members in [[i for i in range(len(demands)) if subset >> i & 1]]
            if deficiency(members) == maximum
        ]
        assert indices == sorted(set().union(*maximisers))
        assert violation["target"] == float(
            sum((Fraction(demands[i][2]) for i in indices), Fraction())
        )
        union = np.logical_or.reduce([demands[i][3] for i in indices])
        assert violation["eligible_mass"] == float(
            sum((Fraction(float(value)) for value in weights[union]), Fraction())
        )
        assert violation["filters"] == [demands[i][1] for i in indices]


def test_hall_exact_relative_tolerance_boundary(monkeypatch):
    monkeypatch.setattr(receipt_module, "RECEIPT_RELATIVE_TOLERANCE", 0.125)
    demands = [("alpha", None, 8.0, np.ones(1, dtype=bool))]
    assert receipt_module.hall_diagnostics(demands, np.array([7.0]))["feasible"]
    assert not receipt_module.hall_diagnostics(
        demands, np.array([np.nextafter(7.0, 0)])
    )["feasible"]


def test_hall_24_demands_collapses_supplies_without_enumerating_subsets(monkeypatch):
    def no_subsets(*args, **kwargs):
        raise AssertionError("Hall diagnostics must not enumerate demand subsets")

    monkeypatch.setattr(receipt_module.itertools, "combinations", no_subsets)
    people = np.arange(20_000)
    demands = [
        (f"p{program}", f"s{stratum}", 1.0, people % 6 == stratum)
        for program in range(4)
        for stratum in range(6)
    ]
    result = receipt_module.hall_diagnostics(demands, np.ones(len(people)))
    assert result["feasible"]
    assert result["violations"] == []


@pytest.mark.parametrize(("share", "feasible"), [(0.066, True), (1.0, False)])
def test_hall_distinct_patterns_at_scale(share, feasible):
    # Independent masks give almost every person a pattern of their own: about
    # 20,000 supplies, a shape Edmonds-Karp over Fractions did not finish.
    rng = np.random.default_rng(0)
    masks = rng.random((20_000, 24)) < 0.5
    weights = rng.uniform(1, 100, 20_000)
    demands = [
        (f"p{index}", None, share * float(weights[masks[:, index]].sum()), mask)
        for index, mask in enumerate(masks.T)
    ]
    result = receipt_module.hall_diagnostics(demands, weights)
    assert result["feasible"] is feasible
    if not feasible:
        (violation,) = result["violations"]
        assert violation["programs"] == [f"p{index}" for index in range(24)]


@pytest.mark.parametrize(
    "target", [4, 4.0, Fraction(4), np.int64(4), np.int32(4), np.float64(4)]
)
def test_hall_targets_of_every_real_type_agree(target):
    weights = np.array([0.1, 3.3])

    def diagnose(value):
        return receipt_module.hall_diagnostics(
            [
                ("alpha", None, value, np.array([True, True])),
                ("beta", "adult", 0.5, np.array([True, False])),
            ],
            weights,
        )

    assert diagnose(target) == diagnose(4)
    assert not diagnose(4)["feasible"]


def test_hall_rational_targets_are_exact():
    scale = 1 - Fraction(receipt_module.RECEIPT_RELATIVE_TOLERANCE)
    # A target whose denominator is not a power of two, at the exact boundary.
    target = Fraction(7) / scale
    assert target.denominator & (target.denominator - 1)
    demands = [("alpha", None, target, np.ones(1, dtype=bool))]
    assert receipt_module.hall_diagnostics(demands, np.array([7.0]))["feasible"]
    assert not receipt_module.hall_diagnostics(
        demands, np.array([np.nextafter(7.0, 0)])
    )["feasible"]
    # Capacities of 4/3 and 1/5 need the lcm of 3, 5 and the weights' 4. Alpha
    # is short by 1/12, more than the pair's 1/30, so it alone is reported.
    demands = [
        ("alpha", None, Fraction(4, 3) / scale, np.array([True, False])),
        ("beta", None, Fraction(1, 5) / scale, np.array([False, True])),
    ]
    result = receipt_module.hall_diagnostics(demands, np.array([1.25, 0.25]))
    assert not result["feasible"]
    assert result["violations"][0]["programs"] == ["alpha"]


def test_hall_refuses_weights_that_do_not_align_with_masks():
    demands = [("alpha", None, 1.0, np.ones(2, dtype=bool))]
    with pytest.raises(ValueError, match="align with the demand masks"):
        receipt_module.hall_diagnostics(demands, np.ones(3))


def test_hall_keeps_targets_exact_below_float_resolution():
    scale = 1 - Fraction(receipt_module.RECEIPT_RELATIVE_TOLERANCE)
    # Both targets round to the same float but sit either side of the boundary.
    below = 1 / scale - Fraction(1, 2**200)
    above = 1 / scale + Fraction(1, 2**200)
    assert float(below) == float(above)

    def feasible(target):
        demands = [("alpha", None, target, np.ones(1, dtype=bool))]
        return receipt_module.hall_diagnostics(demands, np.ones(1))["feasible"]

    assert feasible(below)
    assert not feasible(above)


def test_hall_reroutes_flow_along_reverse_arcs():
    # The first blocking flow sends alpha through the first person and strands
    # beta; only undoing that flow along a reverse arc finds the allocation.
    demands = [
        ("alpha", None, 1.0, np.array([True, True])),
        ("beta", None, 1.0, np.array([True, False])),
        ("gamma", None, 0.0, np.array([False, True])),
    ]
    assert receipt_module.hall_diagnostics(demands, np.ones(2))["feasible"]


@pytest.mark.parametrize(("target", "feasible"), [(1.25, True), (2.0, False)])
def test_hall_sums_weights_across_chunks(target, feasible):
    people = receipt_module._HALL_WEIGHT_CHUNK + 1
    weights = np.zeros(people)
    # One weight in each chunk, each with its own power-of-two denominator.
    weights[0], weights[-1] = 0.5, 1.0
    demands = [("alpha", None, target, np.ones(people, dtype=bool))]
    result = receipt_module.hall_diagnostics(demands, weights)
    assert result["feasible"] is feasible
    if not feasible:
        assert result["violations"][0]["eligible_mass"] == 1.5


def test_hall_keeps_pattern_indices_aligned_across_chunks():
    people = receipt_module._HALL_WEIGHT_CHUNK + 1
    weights = np.zeros(people)
    weights[0], weights[-1] = 1.0, 1.0
    # Only the last person, alone in the second chunk, is eligible for beta.
    beta = np.zeros(people, dtype=bool)
    beta[-1] = True
    demands = [("alpha", None, 1.0, ~beta), ("beta", None, 1.0, beta)]
    assert receipt_module.hall_diagnostics(demands, weights)["feasible"]


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf])
def test_hall_ignores_weights_of_people_eligible_for_nothing(weight):
    demands = [("alpha", None, 1.0, np.array([False, True]))]
    assert receipt_module.hall_diagnostics(
        demands, np.array([weight, 1.0])
    ) == receipt_module.hall_diagnostics(demands, np.array([0.0, 1.0]))
    # The same weight on an eligible person still refuses.
    eligible = [("alpha", None, 1.0, np.array([True, True]))]
    error = ValueError if math.isnan(weight) else OverflowError
    with pytest.raises(error, match="cannot convert"):
        receipt_module.hall_diagnostics(eligible, np.array([weight, 1.0]))


def test_numerical_check_underflow_refuses_with_audit():
    with pytest.raises(
        ReceiptAssignmentRefused, match="numerical expected-mass check"
    ) as refused:
        _assign(
            _people((20, 21, 22)), weights=[1e200] * 3, targets={"alpha_total": 1e-200}
        )
    audit = refused.value.audit
    assert audit["status"] == "refused"
    detail = audit["programs"]["receives_alpha"]["strata"][0]
    assert detail["numerical_check"]["passed"] is False
    assert detail["numerical_check"]["expected_mass"] == 0
    assert detail["target"] == 1e-200


@pytest.mark.parametrize("weight", [math.inf, -math.inf, math.nan])
def test_v2_nonfinite_weights_refuse(weight):
    with pytest.raises(ValueError, match="finite"):
        _assign(_people((20,)), weights=[weight])


def test_v2_overflowing_weight_sum_refuses_as_value_error():
    with pytest.raises(ValueError, match="finite sum"):
        _assign(_people((20, 21, 22)), weights=[1e308] * 3)


def test_v2_cyclic_exclusion_groups_refuse_at_validation():
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[
            ["receives_alpha", "receives_beta"],
            ["receives_beta", "receives_alpha"],
        ],
    )
    with pytest.raises(ValueError, match="cyclic order"):
        validate_receipt_contract(contract)


@pytest.mark.parametrize("output", ["person_id", "person_household_id"])
def test_v2_output_structural_column_refuses_in_pure_function(output):
    row = _program()
    row["output"] = output
    with pytest.raises(ValueError, match="structural column"):
        _assign(_people((20,)), _contract(row))


@pytest.mark.parametrize("mismatch", ["length", "ids", "order", "missing"])
def test_v2_person_table_must_match_argument(mismatch):
    person = _people((20, 21))
    table = person.copy()
    if mismatch == "length":
        table = table.iloc[:1]
    elif mismatch == "ids":
        table["person_id"] += 10
    else:
        table = table.iloc[::-1]
    with pytest.raises(ValueError, match="align with person length and ids"):
        receipt_module.assign_receipts_v2(
            person,
            _contract(),
            weights=np.ones(2),
            targets={"alpha_total": 0},
            knobs={"assignment": "central"},
            tables={} if mismatch == "missing" else {"person": table},
        )


@pytest.mark.parametrize("design", ["view", "seam"])
def test_declared_unit_extension_only_widens_values(monkeypatch, design):
    """Widening a reserved unit, as BE's extension will, needs no rewrite here.

    ``payee`` becomes household after validation. Either a person-aligned
    household view, or a native household table broadcast at the dispatch seam,
    must assign, matching ids through ``person_household_id``.
    """
    validate = receipt_module.validate_receipt_v2
    dispatch = receipt_module.receipt_dispatch_tables
    person = _people((20, 21, 22)).drop(columns="payment")
    person["person_household_id"] = [10, 10, 20]
    household = pd.DataFrame({"household_id": [10, 20], "payment": [1.0, 1.0]})

    def to_people(table):
        members = person["person_household_id"].to_numpy()
        return table.set_index("household_id").loc[members].reset_index()

    def widened(contract):
        seed, rows, groups = validate(contract)
        return seed, [{**row, "payee": "household"} for row in rows], groups

    def broadcast(tables, row):
        return {**dispatch(tables, row), "payee": to_people(tables["household"])}

    monkeypatch.setattr(receipt_module, "validate_receipt_v2", widened)
    if design == "seam":
        monkeypatch.setattr(receipt_module, "receipt_dispatch_tables", broadcast)
        tables = {"person": person, "household": household}
    else:
        tables = {"person": person, "household": to_people(household)}
    out, _ = receipt_module.assign_receipts_v2(
        person,
        _contract(),
        weights=np.ones(3),
        targets={"alpha_total": 3},
        knobs={"assignment": "central"},
        tables=tables,
    )
    assert out["receives_alpha"].tolist() == [True, True, True]


@pytest.mark.parametrize("target", [np.int64(1), np.float32(1), np.float64(1)])
def test_v2_numpy_real_targets_are_accepted(target):
    out, audit = _assign(_people((20,)), targets={"alpha_total": target})
    assert out["receives_alpha"].tolist() == [True]
    assert type(audit["programs"]["receives_alpha"]["strata"][0]["target"]) is float


@pytest.mark.parametrize("target", [True, np.bool_(True)])
def test_v2_boolean_targets_refuse(target):
    with pytest.raises(ValueError, match="target must be finite"):
        _assign(_people((20,)), targets={"alpha_total": target})


# Literal audits from main, pinned before the target-above-mass rule lands.


def _check(expected_mass):
    return {
        "passed": True,
        "relative_tolerance": 1e-12,
        "expected_mass": expected_mass,
    }


def _row(name, target, filter_name, mass, before, before_ratio, ratio, rate):
    return {
        "target_name": name,
        "target": target,
        "filter": filter_name,
        "eligible_mass": mass,
        "eligible_mass_before_exclusions": before,
        "target_to_eligible_mass_before_exclusions": before_ratio,
        "target_to_remaining_eligible_mass": ratio,
        "rate": rate,
    }


def _done(row, realized, count, expected):
    return {
        **row,
        "realized_mass": realized,
        "n_receipts": count,
        "numerical_check": _check(expected),
    }


def _program_audit(name, target, before, mass, before_ratio, ratio, strata, outside):
    return {
        "program": name,
        "target": target,
        "eligible_mass_before_exclusions": before,
        "eligible_mass": mass,
        "target_to_eligible_mass_before_exclusions": before_ratio,
        "target_to_remaining_eligible_mass": ratio,
        "strata": strata,
        "outside_strata": {"n_eligible": outside[0], "eligible_mass": outside[1]},
    }


def _finished(program_audit, realized, count):
    return {**program_audit, "realized_mass": realized, "n_receipts": count}


def _audit(programs, groups=(), status="completed"):
    return {
        "assignment_profile": "central",
        "exclusion_order": "declared",
        "status": status,
        "programs": programs,
        "exclusion_groups": list(groups),
    }


def _pinned(case):
    """Literal audits traced from main, without the mass rule.

    Returns person, contract, weights, targets, flags, audit and the refusal
    message (None when the run completes). Fix a literal only from a run of
    the unchanged base, never from the changed code.
    """
    if case == "completed":
        young = _row("alpha_young", 2.0, "young", 2.0, 2.0, 1.0, 1.0, 1.0)
        older = _row("alpha_older", 3.0, "older", 3.0, 3.0, 1.0, 1.0, 1.0)
        program_audit = _program_audit(
            "alpha",
            5.0,
            5.0,
            5.0,
            1.0,
            1.0,
            [_done(young, 2.0, 1, 2.0), _done(older, 3.0, 1, 3.0)],
            (1, 5.0),
        )
        return (
            _people((20, 40, 70)),
            _contract(_program(stratified=True)),
            [2, 3, 5],
            {"alpha_young": 2, "alpha_older": 3},
            {"receives_alpha": [True, True, False]},
            _audit({"receives_alpha": _finished(program_audit, 5.0, 2)}),
            None,
        )
    if case == "refused":
        row = _row("alpha_total", 3.0, None, 2.0, 2.0, 1.5, 1.5, 1.5)
        program_audit = _program_audit(
            "alpha", 3.0, 2.0, 2.0, 1.5, 1.5, [row], (0, 0.0)
        )
        return (
            _people((20, 21)),
            _contract(),
            [1, 1],
            {"alpha_total": 3},
            None,
            _audit({"receives_alpha": program_audit}, status="refused"),
            "Receipt target 'alpha_total' exceeds eligible weighted mass in declared order.",
        )
    if case not in ("group", "group_refused"):
        raise AssertionError(f"unknown case {case!r}")
    order = ["receives_alpha", "receives_beta"]
    alpha = _row("alpha_total", 2.0, None, 2.0, 2.0, 1.0, 1.0, 1.0)
    programs = {
        "receives_alpha": _finished(
            _program_audit(
                "alpha", 2.0, 2.0, 2.0, 1.0, 1.0, [_done(alpha, 2.0, 2, 2.0)], (0, 0.0)
            ),
            2.0,
            2,
        ),
    }
    overlaps = [
        {
            "programs": pair,
            "filter": None,
            "n_eligible": 2,
            "eligible_mass": 2.0,
        }
        for pair in (order, order[::-1])
    ]
    if case == "group_refused":
        # Alpha takes beta's whole pool, so beta's remaining ratio is None and
        # beta refuses. Targets of 3.0 over a mass of 2.0 fail the Hall check.
        beta = _row("beta_total", 1.0, None, 0.0, 2.0, 0.5, None, None)
        programs["receives_beta"] = _program_audit(
            "beta", 1.0, 2.0, 0.0, 0.5, None, [beta], (0, 0.0)
        )
        violation = {
            "programs": order,
            "filter": None,
            "filters": [None, None],
            "target": 3.0,
            "eligible_mass": 2.0,
        }
        group = {
            "order": order,
            "status": "refused",
            "overlaps": overlaps,
            "feasibility": {
                "violations": [violation],
                "feasible": False,
                "relative_tolerance": 1e-12,
            },
        }
        return (
            _people((20, 21)),
            _contract(_program("alpha"), _program("beta"), groups=[order]),
            [1, 1],
            {"alpha_total": 2, "beta_total": 1},
            None,
            _audit(programs, [group], status="refused"),
            "Receipt target 'beta_total' exceeds eligible weighted mass in declared order.",
        )
    # A zero target over zero remaining mass gives the integer ratio 0.
    beta = _row("beta_total", 0.0, None, 0.0, 2.0, 0.0, 0, 0)
    programs["receives_beta"] = _finished(
        _program_audit(
            "beta", 0.0, 2.0, 0.0, 0.0, 0, [_done(beta, 0.0, 0, 0.0)], (0, 0.0)
        ),
        0.0,
        0,
    )
    group = {
        "order": order,
        "status": "completed",
        "overlaps": overlaps,
        "feasibility": {
            "violations": [],
            "feasible": True,
            "relative_tolerance": 1e-12,
        },
    }
    return (
        _people((20, 21)),
        _contract(_program("alpha"), _program("beta"), groups=[order]),
        [1, 1],
        {"alpha_total": 2, "beta_total": 0},
        {"receives_alpha": [True, True], "receives_beta": [False, False]},
        _audit(programs, [group]),
        None,
    )


@pytest.mark.parametrize("case", ["completed", "refused", "group", "group_refused"])
def test_v2_audit_bytes_without_the_mass_rule_are_pinned(case):
    person, contract, weights, targets, flags, audit, message = _pinned(case)
    if message is not None:
        with pytest.raises(ReceiptAssignmentRefused) as refused:
            _assign(person, contract, weights=weights, targets=targets)
        assert str(refused.value) == message
        assert json.dumps(refused.value.audit) == json.dumps(audit)
        return
    out, result = _assign(person, contract, weights=weights, targets=targets)
    assert list(out.columns) == ["person_id", *flags]
    assert {name: out[name].tolist() for name in flags} == flags
    assert json.dumps(result) == json.dumps(audit)


# Opt-in rule for targets above eligible weighted mass.


def _declared(contract):
    contract["target_exceeds_eligible_mass"] = "assign_all_eligible"
    return contract


class _Text(str):
    """Equal to the literal, but not the exact str the validator requires."""


def test_mass_rule_assigns_the_whole_pool_and_records_the_shortfall():
    person, contract, weights, targets, *_ = _pinned("refused")
    out, audit = _assign(person, _declared(contract), weights=weights, targets=targets)
    assert out["receives_alpha"].tolist() == [True, True]
    row = {**_row("alpha_total", 3.0, None, 2.0, 2.0, 1.5, 1.5, 1.0), "shortfall": 1.0}
    program_audit = _program_audit(
        "alpha", 3.0, 2.0, 2.0, 1.5, 1.5, [_done(row, 2.0, 2, 2.0)], (0, 0.0)
    )
    expected = _audit({"receives_alpha": _finished(program_audit, 2.0, 2)})
    assert json.dumps(audit) == json.dumps(expected)


@pytest.mark.parametrize("case", ["completed", "group", "draws"])
def test_mass_rule_changes_nothing_until_a_target_exceeds_its_mass(case):
    if case == "draws":
        person, contract = _people(tuple(range(20, 40))), _contract()
        weights, targets = None, {"alpha_total": 10}
    else:
        person, contract, weights, targets, *_ = _pinned(case)
    plain, plain_audit = _assign(
        person, copy.deepcopy(contract), weights=weights, targets=targets
    )
    out, audit = _assign(person, _declared(contract), weights=weights, targets=targets)
    assert (
        out.to_records(index=False).tobytes() == plain.to_records(index=False).tobytes()
    )
    assert json.dumps(audit) == json.dumps(plain_audit)


def test_v2_mass_rule_is_one_optional_literal_and_keeps_the_validator_shape():
    plain = validate_receipt_contract(_contract())
    assert len(plain) == 3
    assert validate_receipt_contract(_declared(_contract())) == plain
    assert receipt_module.validate_receipt_v2(_declared(_contract())) == plain


@pytest.mark.parametrize(
    "value",
    [
        "refuse",
        "assign_all",
        "ASSIGN_ALL_ELIGIBLE",
        "",
        None,
        True,
        1,
        ["assign_all_eligible"],
        {"assign_all_eligible": True},
        np.array(["assign_all_eligible"]),
        _Text("assign_all_eligible"),
    ],
)
def test_v2_mass_rule_refuses_any_other_value(value):
    contract = _contract()
    contract["target_exceeds_eligible_mass"] = value
    with pytest.raises(
        ValueError,
        match=r"^Receipt target_exceeds_eligible_mass must be declared as assign_all_eligible\.$",
    ):
        validate_receipt_contract(contract)


@pytest.mark.parametrize("location", ["program", "stratum", "profile", "v1"])
def test_mass_rule_is_read_only_at_the_v2_contract_top_level(location):
    contract = _contract(_program(stratified=True))
    match = "missing or unknown fields"
    if location == "v1":
        contract = {
            "seed_column": "take_up_seed",
            "programs": [
                {
                    "program": "toy.alpha",
                    "output": "receives_alpha",
                    "judgment_column": "eligible",
                    "rate": 0.5,
                }
            ],
            "exclusion_groups": [],
        }
        destination = contract
        match = (
            r"^Receipt contract declares seed_column, programs and exclusion_groups\.$"
        )
    else:
        destination = {
            "program": contract["programs"][0],
            "stratum": contract["programs"][0]["strata"][0],
            "profile": contract["assignment_profiles"]["central"],
        }[location]
    destination["target_exceeds_eligible_mass"] = "assign_all_eligible"
    with pytest.raises(ValueError, match=match):
        validate_receipt_contract(contract)


@pytest.mark.parametrize("case", ["screened", "weightless", "zero_target"])
def test_mass_rule_at_zero_eligible_mass(case):
    person = _people((20, 21))
    weights, targets = [1, 1], {"alpha_total": 0 if case == "zero_target" else 1}
    if case == "screened":
        person["payment"] = [0.0, 0.0]
    else:
        weights = [0, 0]
    out, audit = _assign(
        person, _declared(_contract()), weights=weights, targets=targets
    )
    assert audit["status"] == "completed"
    program_audit = audit["programs"]["receives_alpha"]
    (row,) = program_audit["strata"]
    if case == "zero_target":
        # Not capped: the integer rate 0 assigns nobody, exactly as before.
        _, plain_audit = _assign(person, _contract(), weights=weights, targets=targets)
        assert json.dumps(audit) == json.dumps(plain_audit)
        assert out["receives_alpha"].tolist() == [False, False]
        assert "shortfall" not in row
        assert row["rate"] == 0
        return
    with pytest.raises(
        ReceiptAssignmentRefused, match="exceeds eligible weighted mass"
    ):
        _assign(person, _contract(), weights=weights, targets=targets)
    # Every pool member receives, the zero-weight people included.
    flags = [case == "weightless"] * 2
    assert out["receives_alpha"].tolist() == flags
    capped = {
        **_row("alpha_total", 1.0, None, 0.0, 0.0, None, None, 1.0),
        "shortfall": 1.0,
    }
    assert json.dumps(row) == json.dumps(_done(capped, 0.0, sum(flags), 0.0))
    assert (program_audit["realized_mass"], program_audit["n_receipts"]) == (
        0.0,
        sum(flags),
    )


def test_mass_rule_caps_each_stratum_on_its_own():
    person = _people((20, 21, 40, 41))
    weights = [2, 2, 1, 1]
    contract = _contract(_program(stratified=True))
    targets = {"alpha_young": 5, "alpha_older": 1}
    with pytest.raises(
        ReceiptAssignmentRefused, match="'alpha_young' exceeds eligible weighted mass"
    ):
        _assign(person, copy.deepcopy(contract), weights=weights, targets=targets)
    out, audit = _assign(
        person, _declared(copy.deepcopy(contract)), weights=weights, targets=targets
    )
    # Capping a stratum with mass assigns as a target equal to that mass would.
    plain, plain_audit = _assign(
        person, contract, weights=weights, targets={"alpha_young": 4, "alpha_older": 1}
    )
    draws = derive_take_up_draws(
        person["take_up_seed"].to_numpy(), receipt_draw_key("alpha", "")
    )
    assert out["receives_alpha"].tolist() == [True, True, *(draws[2:] < 0.5).tolist()]
    assert (
        out.to_records(index=False).tobytes() == plain.to_records(index=False).tobytes()
    )
    young, older = audit["programs"]["receives_alpha"]["strata"]
    plain_young, plain_older = plain_audit["programs"]["receives_alpha"]["strata"]
    assert list(young) == [*list(plain_young)[:8], "shortfall", *list(plain_young)[8:]]
    assert young == {
        **plain_young,
        "target": 5.0,
        "target_to_eligible_mass_before_exclusions": 1.25,
        "target_to_remaining_eligible_mass": 1.25,
        "shortfall": 1.0,
    }
    assert json.dumps(older) == json.dumps(plain_older)


@pytest.mark.parametrize(
    ("ages", "weights", "targets", "flags", "rates", "shortfalls", "program_values"),
    [
        (
            (20, 40),
            [1, 1],
            {"alpha_young": 2, "alpha_older": 0},
            [True, False],
            [1.0, 0.0],
            [1.0, None],
            (2.0, 2.0, 1.0, 1.0, 1, {"n_eligible": 0, "eligible_mass": 0.0}),
        ),
        (
            (20, 40, 70),
            [2, 3, 5],
            {"alpha_young": 4, "alpha_older": 3},
            [True, True, False],
            [1.0, 1.0],
            [2.0, None],
            (7.0, 5.0, 1.4, 5.0, 2, {"n_eligible": 1, "eligible_mass": 5.0}),
        ),
    ],
    ids=["sibling", "outside"],
)
def test_mass_rule_never_moves_a_shortfall_to_a_sibling_or_outside_strata(
    ages, weights, targets, flags, rates, shortfalls, program_values
):
    out, audit = _assign(
        _people(ages),
        _declared(_contract(_program(stratified=True))),
        weights=weights,
        targets=targets,
    )
    assert out["receives_alpha"].tolist() == flags
    result = audit["programs"]["receives_alpha"]
    assert [row["rate"] for row in result["strata"]] == rates
    assert [row.get("shortfall") for row in result["strata"]] == shortfalls
    assert (
        result["target"],
        result["eligible_mass"],
        result["target_to_remaining_eligible_mass"],
        result["realized_mass"],
        result["n_receipts"],
        result["outside_strata"],
    ) == program_values
    assert "shortfall" not in result


@pytest.mark.parametrize("profile", ["central", "reversed"])
def test_mass_rule_follows_the_profile_exclusion_order(profile):
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    targets = {"alpha_total": 3, "beta_total": 2}
    with pytest.raises(
        ReceiptAssignmentRefused, match="exceeds eligible weighted mass"
    ):
        _assign(
            _people((20, 21)), copy.deepcopy(contract), targets=targets, profile=profile
        )
    out, audit = _assign(
        _people((20, 21)), _declared(contract), targets=targets, profile=profile
    )
    first, second = ("alpha", "beta") if profile == "central" else ("beta", "alpha")
    assert out[f"receives_{first}"].tolist() == [True, True]
    assert out[f"receives_{second}"].tolist() == [False, False]
    rows = {
        name: audit["programs"][f"receives_{name}"]["strata"][0]
        for name in ("alpha", "beta")
    }
    assert {name: row.get("shortfall") for name, row in rows.items()} == {
        "central": {"alpha": 1.0, "beta": 2.0},
        "reversed": {"alpha": 3.0, "beta": None},
    }[profile]
    assert all(row["rate"] == 1.0 for row in rows.values())
    assert rows[second]["eligible_mass"] == 0.0
    assert rows[second]["eligible_mass_before_exclusions"] == 2.0
    assert rows[second]["target_to_remaining_eligible_mass"] is None
    assert rows[second]["numerical_check"] == _check(0.0)
    (group,) = audit["exclusion_groups"]
    assert audit["status"] == group["status"] == "completed"
    assert group["order"] == [f"receives_{first}", f"receives_{second}"]
    assert group["feasibility"]["feasible"] is False
    (violation,) = group["feasibility"]["violations"]
    assert violation["programs"] == group["order"]
    assert (violation["target"], violation["eligible_mass"]) == (5.0, 2.0)


@pytest.mark.parametrize(
    ("profile", "targets", "short"),
    [
        ("central", {"alpha_total": 2, "beta_total": 1}, "beta"),
        ("reversed", {"alpha_total": 1, "beta_total": 2}, "alpha"),
    ],
)
def test_mass_rule_completes_where_the_hall_diagnostic_fails(profile, targets, short):
    contract = _contract(
        _program("alpha"),
        _program("beta"),
        groups=[["receives_alpha", "receives_beta"]],
    )
    out, audit = _assign(
        _people((20, 21)), _declared(contract), targets=targets, profile=profile
    )
    full = "alpha" if short == "beta" else "beta"
    assert out[f"receives_{full}"].tolist() == [True, True]
    assert out[f"receives_{short}"].tolist() == [False, False]
    assert "shortfall" not in audit["programs"][f"receives_{full}"]["strata"][0]
    row = audit["programs"][f"receives_{short}"]["strata"][0]
    capped = {
        **_row(f"{short}_total", 1.0, None, 0.0, 2.0, 0.5, None, 1.0),
        "shortfall": 1.0,
    }
    assert json.dumps(row) == json.dumps(_done(capped, 0.0, 0, 0.0))
    (group,) = audit["exclusion_groups"]
    assert audit["status"] == group["status"] == "completed"
    assert group["feasibility"]["feasible"] is False
    (violation,) = group["feasibility"]["violations"]
    assert violation["programs"] == group["order"]
    assert (violation["target"], violation["eligible_mass"]) == (3.0, 2.0)


def test_mass_rule_keeps_the_numerical_check_on_strata_within_their_mass():
    with pytest.raises(
        ReceiptAssignmentRefused, match="numerical expected-mass check"
    ) as refused:
        _assign(
            _people((20, 21, 22)),
            _declared(_contract()),
            weights=[1e200] * 3,
            targets={"alpha_total": 1e-200},
        )
    audit = refused.value.audit
    assert audit["status"] == "refused"
    detail = audit["programs"]["receives_alpha"]["strata"][0]
    assert "shortfall" not in detail
    assert detail["numerical_check"]["passed"] is False
    assert detail["numerical_check"]["expected_mass"] == 0


def test_mass_rule_keeps_an_earlier_shortfall_when_a_later_check_refuses():
    with pytest.raises(
        ReceiptAssignmentRefused,
        match="'alpha_older' failed the numerical expected-mass check",
    ) as refused:
        _assign(
            _people((20, 21, 40, 41)),
            _declared(_contract(_program(stratified=True))),
            weights=[1, 1, 1e200, 1e200],
            # The young target exceeds its mass of 2; the older rate, 1e-200
            # over 2e200, underflows to 0.0.
            targets={"alpha_young": 3, "alpha_older": 1e-200},
        )
    audit = refused.value.audit
    assert audit["status"] == "refused"
    young, older = audit["programs"]["receives_alpha"]["strata"]
    assert (young["rate"], young["shortfall"]) == (1.0, 1.0)
    assert young["numerical_check"] == _check(2.0)
    assert (older["rate"], older["numerical_check"]["passed"]) == (0.0, False)
    assert "shortfall" not in older


def test_mass_rule_caps_a_pool_that_exclude_columns_shrank():
    person = _people((20, 21))
    person["excluded"] = np.array([True, False])
    program = _program()
    program["exclude_columns"] = ["excluded"]
    contract = _contract(program)
    targets = {"alpha_total": 2}
    with pytest.raises(
        ReceiptAssignmentRefused, match="exceeds eligible weighted mass"
    ):
        _assign(person, copy.deepcopy(contract), weights=[1, 1], targets=targets)
    out, audit = _assign(person, _declared(contract), weights=[1, 1], targets=targets)
    assert out["receives_alpha"].tolist() == [False, True]
    (row,) = audit["programs"]["receives_alpha"]["strata"]
    assert (
        row["eligible_mass"],
        row["eligible_mass_before_exclusions"],
        row["rate"],
        row["shortfall"],
        row["realized_mass"],
        row["n_receipts"],
    ) == (1.0, 2.0, 1.0, 1.0, 1.0, 1)


def test_mass_rule_assigns_the_capped_pool_even_when_every_draw_is_nan(monkeypatch):
    keys = []

    def nan_draws(seeds, key):
        keys.append(key)
        return np.full(len(seeds), np.nan)

    monkeypatch.setattr(receipt_module, "derive_take_up_draws", nan_draws)
    out, audit = _assign(
        _people((20, 21, 22)), _declared(_contract()), targets={"alpha_total": 4}
    )
    # NaN < rate is False, so a pool chosen by its draws would assign nobody.
    assert out["receives_alpha"].tolist() == [True, True, True]
    assert keys == [receipt_draw_key("alpha", "")]
    (row,) = audit["programs"]["receives_alpha"]["strata"]
    assert (row["rate"], row["shortfall"], row["n_receipts"]) == (1.0, 1.0, 3)


@pytest.mark.parametrize("profile", ["central", "variant"])
def test_mass_rule_assigns_the_capped_pool_and_leaves_other_programs_draws(profile):
    person = _people(tuple(range(20, 40)))
    contract = _declared(_contract(_program("alpha"), _program("beta")))
    suffix = "" if profile == "central" else "variant"
    seeds = person["take_up_seed"].to_numpy()

    def drawn(program):
        draws = derive_take_up_draws(seeds, receipt_draw_key(program, suffix))
        return (draws < 0.5).tolist()

    capped, audit = _assign(
        person, contract, targets={"alpha_total": 25, "beta_total": 10}, profile=profile
    )
    plain, _ = _assign(
        person, contract, targets={"alpha_total": 10, "beta_total": 10}, profile=profile
    )
    assert capped["receives_alpha"].all()
    assert audit["programs"]["receives_alpha"]["strata"][0]["shortfall"] == 5.0
    assert plain["receives_alpha"].tolist() == drawn("alpha")
    assert capped["receives_beta"].tolist() == plain["receives_beta"].tolist()
    assert capped["receives_beta"].tolist() == drawn("beta")


MASS_RULE_PROPERTY = settings(
    max_examples=40,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)


@MASS_RULE_PROPERTY
@given(
    st.lists(st.integers(min_value=0, max_value=90), min_size=1, max_size=12),
    st.sampled_from(["central", "variant"]),
    st.data(),
)
def test_property_mass_rule_changes_only_strata_whose_target_exceeds_mass(
    ages, profile, data
):
    count = len(ages)
    weights = data.draw(st.lists(st.integers(0, 100), min_size=count, max_size=count))
    eligible = np.asarray(
        data.draw(st.lists(st.booleans(), min_size=count, max_size=count)), dtype=bool
    )
    eighths = data.draw(st.lists(st.integers(0, 16), min_size=2, max_size=2))
    person = _people(tuple(ages))
    person["eligible"] = np.where(eligible, 1, -1)
    age = np.asarray(ages)
    pools = {
        "alpha_young": eligible & (age >= 18) & (age < 30),
        "alpha_older": eligible & (age >= 30) & (age < 65),
    }
    values = np.asarray(weights, dtype=np.float64)
    masses = {name: math.fsum(values[pool].tolist()) for name, pool in pools.items()}
    # Eighths keep rate * mass exact; a zero mass still gets a positive target.
    targets = {
        name: (masses[name] or 1.0) * share / 8
        for name, share in zip(masses, eighths, strict=True)
    }
    exceeded = {name for name in targets if targets[name] > masses[name]}
    contract = _contract(_program(stratified=True))
    out, audit = _assign(
        person,
        _declared(copy.deepcopy(contract)),
        weights=weights,
        targets=targets,
        profile=profile,
    )
    assert audit["status"] == "completed"
    json.dumps(audit, allow_nan=False)
    if exceeded:
        with pytest.raises(
            ReceiptAssignmentRefused, match="exceeds eligible weighted mass"
        ):
            _assign(person, contract, weights=weights, targets=targets, profile=profile)
    else:
        plain, plain_audit = _assign(
            person, contract, weights=weights, targets=targets, profile=profile
        )
        assert (
            out.to_records(index=False).tobytes()
            == plain.to_records(index=False).tobytes()
        )
        assert json.dumps(audit) == json.dumps(plain_audit)
    flags = out["receives_alpha"].to_numpy()
    for row in audit["programs"]["receives_alpha"]["strata"]:
        name = row["target_name"]
        assert row["eligible_mass"] == masses[name]
        assert 0 <= row["rate"] <= 1
        assert row["numerical_check"]["passed"] is True
        if name in exceeded:
            assert flags[pools[name]].all()
            assert row["n_receipts"] == int(pools[name].sum())
            assert row["rate"] == 1.0
            assert row["realized_mass"] == masses[name]
            assert row["numerical_check"]["expected_mass"] == masses[name]
            assert row["shortfall"] == targets[name] - masses[name] > 0
        else:
            assert "shortfall" not in row


@MASS_RULE_PROPERTY
@given(
    st.integers(1, 15),
    st.integers(0, 12),
    st.integers(0, 12),
    st.sampled_from(["central", "reversed"]),
)
@example(count=2, alpha=12, beta=8, profile="central")
def test_property_mass_rule_group_never_refuses_or_assigns_two_flags(
    count, alpha, beta, profile
):
    contract = _declared(
        _contract(
            _program("alpha"),
            _program("beta"),
            groups=[["receives_alpha", "receives_beta"]],
        )
    )
    out, audit = _assign(
        _people(tuple(range(20, 20 + count))),
        contract,
        targets={"alpha_total": count * alpha / 8, "beta_total": count * beta / 8},
        profile=profile,
    )
    assert audit["status"] == audit["exclusion_groups"][0]["status"] == "completed"
    json.dumps(audit, allow_nan=False)
    assert (out[["receives_alpha", "receives_beta"]].sum(axis=1) <= 1).all()
    first, second = ("alpha", "beta") if profile == "central" else ("beta", "alpha")
    if {"alpha": alpha, "beta": beta}[first] > 8:
        assert out[f"receives_{first}"].all()
        assert not out[f"receives_{second}"].any()
    for name in ("alpha", "beta"):
        row = audit["programs"][f"receives_{name}"]["strata"][0]
        assert row["realized_mass"] <= row["eligible_mass"]
        assert ("shortfall" in row) == (row["target"] > row["eligible_mass"])
        if "shortfall" in row:
            assert row["n_receipts"] == row["eligible_mass"]


# A target over a tiny mass overflows its ratio to infinity, which canonical
# JSON refuses. Each case gives ages, exclusions, weights, the masses before
# and after exclusions, the flags under the rule, then (before, remaining)
# ratios without the rule and with it.
_TINY = 5e-324
_OVERFLOW = {
    # One tiny pool overflows both ratios.
    "tiny": (
        (20,),
        [False],
        [_TINY],
        _TINY,
        _TINY,
        [True],
        (math.inf, math.inf),
        (None, None),
    ),
    # The exclusion empties the pool, so its zero mass gives no remaining
    # ratio and only the ratio before exclusions overflows.
    "excluded": (
        (20,),
        [True],
        [_TINY],
        _TINY,
        0.0,
        [False],
        (math.inf, None),
        (None, None),
    ),
    # The exclusion leaves a tiny pool, so only the remaining ratio overflows.
    "remaining": (
        (20, 21),
        [True, False],
        [1.0, _TINY],
        1.0,
        _TINY,
        [False, True],
        (1.0, math.inf),
        (1.0, None),
    ),
}


def _overflow_inputs(ages, excluded):
    person = _people(ages)
    person["excluded"] = np.array(excluded)
    program = _program()
    program["exclude_columns"] = ["excluded"]
    return person, _contract(program)


@pytest.mark.parametrize("case", list(_OVERFLOW))
def test_mass_rule_records_an_overflowing_ratio_as_none(case):
    ages, excluded, weights, before, mass, flags, _, ratios = _OVERFLOW[case]
    person, contract = _overflow_inputs(ages, excluded)
    out, audit = _assign(
        person, _declared(contract), weights=weights, targets={"alpha_total": 1.0}
    )
    assert out["receives_alpha"].tolist() == flags
    row = {
        **_row("alpha_total", 1.0, None, mass, before, *ratios, 1.0),
        "shortfall": 1.0,
    }
    program_audit = _program_audit(
        "alpha",
        1.0,
        before,
        mass,
        *ratios,
        [_done(row, mass, sum(flags), mass)],
        (0, 0.0),
    )
    expected = _audit({"receives_alpha": _finished(program_audit, mass, sum(flags))})
    assert json.dumps(audit, allow_nan=False) == json.dumps(expected)


@pytest.mark.parametrize("case", list(_OVERFLOW))
def test_overflowing_ratios_without_the_mass_rule_refuse_as_before(case):
    ages, excluded, weights, before, mass, _, ratios, _ = _OVERFLOW[case]
    person, contract = _overflow_inputs(ages, excluded)
    with pytest.raises(ReceiptAssignmentRefused) as refused:
        _assign(person, contract, weights=weights, targets={"alpha_total": 1.0})
    assert str(refused.value) == (
        "Receipt target 'alpha_total' exceeds eligible weighted mass in declared order."
    )
    # Traced from a run of base 9189f745: the refusal audit keeps its infinities.
    row = _row("alpha_total", 1.0, None, mass, before, *ratios, ratios[1])
    program_audit = _program_audit("alpha", 1.0, before, mass, *ratios, [row], (0, 0.0))
    expected = _audit({"receives_alpha": program_audit}, status="refused")
    assert json.dumps(refused.value.audit) == json.dumps(expected)


# Each target is finite, but two of them sum past the largest float. Under the
# rule the audit records such a total as None; without it the OverflowError
# propagates, as on base 9189f745.
_HUGE = 1e308
_HUGE_ORDER = ["receives_alpha", "receives_beta"]


def _huge_inputs(case):
    if case == "strata":
        # One program whose two strata each hold one unit-weight adult.
        targets = {"alpha_young": _HUGE, "alpha_older": _HUGE}
        return _people((20, 40)), _contract(_program(stratified=True)), targets
    # Two programs in one exclusion group, both eligible for both adults.
    contract = _contract(_program("alpha"), _program("beta"), groups=[_HUGE_ORDER])
    targets = {"alpha_total": _HUGE, "beta_total": _HUGE}
    return _people((20, 21)), contract, targets


def _huge_expected(case):
    if case == "strata":
        strata = [
            _done(
                {
                    **_row(name, _HUGE, filter_name, 1.0, 1.0, _HUGE, _HUGE, 1.0),
                    "shortfall": _HUGE - 1.0,
                },
                1.0,
                1,
                1.0,
            )
            for name, filter_name in (
                ("alpha_young", "young"),
                ("alpha_older", "older"),
            )
        ]
        # The program total overflows, so it and both of its ratios are None.
        program = _program_audit("alpha", None, 2.0, 2.0, None, None, strata, (0, 0.0))
        flags = {"receives_alpha": [True, True]}
        return flags, _audit({"receives_alpha": _finished(program, 2.0, 2)})
    alpha = {
        **_row("alpha_total", _HUGE, None, 2.0, 2.0, _HUGE / 2, _HUGE / 2, 1.0),
        "shortfall": _HUGE - 2.0,
    }
    beta = {
        **_row("beta_total", _HUGE, None, 0.0, 2.0, _HUGE / 2, None, 1.0),
        "shortfall": _HUGE,
    }
    programs = {
        "receives_alpha": _finished(
            _program_audit(
                "alpha",
                _HUGE,
                2.0,
                2.0,
                _HUGE / 2,
                _HUGE / 2,
                [_done(alpha, 2.0, 2, 2.0)],
                (0, 0.0),
            ),
            2.0,
            2,
        ),
        "receives_beta": _finished(
            _program_audit(
                "beta",
                _HUGE,
                2.0,
                0.0,
                _HUGE / 2,
                None,
                [_done(beta, 0.0, 0, 0.0)],
                (0, 0.0),
            ),
            0.0,
            0,
        ),
    }
    # The pair's exact target overflows, so the Hall violation records None.
    group = {
        "order": _HUGE_ORDER,
        "status": "completed",
        "overlaps": [
            {"programs": pair, "filter": None, "n_eligible": 2, "eligible_mass": 2.0}
            for pair in (_HUGE_ORDER, _HUGE_ORDER[::-1])
        ],
        "feasibility": {
            "violations": [
                {
                    "programs": _HUGE_ORDER,
                    "filter": None,
                    "filters": [None, None],
                    "target": None,
                    "eligible_mass": 2.0,
                }
            ],
            "feasible": False,
            "relative_tolerance": 1e-12,
        },
    }
    flags = {"receives_alpha": [True, True], "receives_beta": [False, False]}
    return flags, _audit(programs, [group])


_HUGE_ERRORS = {
    "strata": "intermediate overflow in fsum",
    "group": "integer division result too large for a float",
}


@pytest.mark.parametrize("case", list(_HUGE_ERRORS))
def test_mass_rule_records_a_target_total_too_large_for_a_float_as_none(case):
    person, contract, targets = _huge_inputs(case)
    out, audit = _assign(person, _declared(contract), targets=targets)
    flags, expected = _huge_expected(case)
    assert {name: out[name].tolist() for name in flags} == flags
    assert json.dumps(audit, allow_nan=False) == json.dumps(expected)


@pytest.mark.parametrize("case", list(_HUGE_ERRORS))
def test_target_total_too_large_for_a_float_raises_without_the_mass_rule(case):
    person, contract, targets = _huge_inputs(case)
    with pytest.raises(OverflowError, match=_HUGE_ERRORS[case]):
        _assign(person, contract, targets=targets)


def test_hall_overflowing_target_raises_unless_recorded_as_none():
    demands = [(name, None, _HUGE, np.ones(2, dtype=bool)) for name in _HUGE_ORDER]
    weights = np.ones(2)
    with pytest.raises(OverflowError, match=_HUGE_ERRORS["group"]):
        receipt_module.hall_diagnostics(demands, weights)
    with pytest.raises(TypeError):
        receipt_module.hall_diagnostics(demands, weights, True)
    assert receipt_module.hall_diagnostics(demands, weights, overflow_as_none=True) == {
        "violations": [
            {
                "programs": _HUGE_ORDER,
                "filter": None,
                "filters": [None, None],
                "target": None,
                "eligible_mass": 2.0,
            }
        ],
        "feasible": False,
        "relative_tolerance": receipt_module.RECEIPT_RELATIVE_TOLERANCE,
    }
    # A total a float can hold reports the same either way.
    demands = [(name, None, _HUGE / 4, np.ones(2, dtype=bool)) for name in _HUGE_ORDER]
    plain = receipt_module.hall_diagnostics(demands, weights)
    assert plain["violations"][0]["target"] == _HUGE / 2
    assert (
        receipt_module.hall_diagnostics(demands, weights, overflow_as_none=True)
        == plain
    )


# fsum raises on these four finite targets although their exact sum, rounded,
# is the largest float.
_NEAR_MAX_TARGETS = [
    float.fromhex(value)
    for value in (
        "0x1.fffffffffffe8p+1020",
        "0x1.0000000000002p+1023",
        "0x1.fffffffffffffp+1020",
        "0x1.0000000000001p+1022",
    )
]


def test_target_total_falls_back_to_the_exact_sum_under_the_mass_rule():
    exact = float(sum(map(Fraction, _NEAR_MAX_TARGETS), Fraction()))
    assert math.isfinite(exact)
    for assign_all in (False, True):
        assert receipt_module._target_total([1.0, 2.5], assign_all) == 3.5
    with pytest.raises(OverflowError, match="intermediate overflow in fsum"):
        receipt_module._target_total(_NEAR_MAX_TARGETS, False)
    assert receipt_module._target_total(_NEAR_MAX_TARGETS, True) == exact
    with pytest.raises(OverflowError, match="intermediate overflow in fsum"):
        receipt_module._target_total([_HUGE, _HUGE], False)
    assert receipt_module._target_total([_HUGE, _HUGE], True) is None


# Eight weights from an independent review: fsum of all eight is the largest
# float. The seven adults' exact sum, 0.36 ULP above it, rounds to it too, yet
# fsum raises on the adults: a partial sum overflows on the way.
_NEAR_MAX_WEIGHTS = np.array(
    [
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
)
_ADULTS = np.array([True] * 6 + [False, True])
_LARGEST = sys.float_info.max


def _exact_or_none(values):
    """The exact sum rounded to a float, or None when it rounds past them all."""
    try:
        return float(sum(map(Fraction, values), Fraction()))
    except OverflowError:
        return None


def test_mass_falls_back_to_the_exact_sum_only_under_the_mass_rule():
    weights = _NEAR_MAX_WEIGHTS
    assert math.fsum(weights.tolist()) == _LARGEST
    assert _exact_or_none(weights[_ADULTS].tolist()) == _LARGEST
    with pytest.raises(OverflowError) as raised:
        receipt_module._mass(weights, _ADULTS, assign_all=False)
    assert str(raised.value) == "intermediate overflow in fsum"
    assert receipt_module._mass(weights, _ADULTS, assign_all=True) == _LARGEST
    # A sum fsum holds is the same either way.
    for assign_all in (False, True):
        assert (
            receipt_module._mass(weights, ~_ADULTS, assign_all=assign_all)
            == (weights[6])
        )
    # The rule must be named at every call.
    with pytest.raises(TypeError):
        receipt_module._mass(weights, _ADULTS)
    with pytest.raises(TypeError):
        receipt_module._mass(weights, _ADULTS, True)


def test_weight_total_falls_back_to_the_exact_sum_only_under_the_mass_rule():
    # The adults alone: fsum raises, but their exact sum rounds to a float.
    weights = _NEAR_MAX_WEIGHTS[_ADULTS]
    person = _people(tuple(range(20, 20 + len(weights))))
    with pytest.raises(ValueError) as refused:
        _assign(person, weights=weights)
    assert str(refused.value) == "Receipt weights must have a finite sum."
    assert type(refused.value.__cause__) is OverflowError
    assert str(refused.value.__cause__) == "intermediate overflow in fsum"
    out, audit = _assign(person, _declared(_contract()), weights=weights)
    assert not out["receives_alpha"].any()
    program = audit["programs"]["receives_alpha"]
    assert program["eligible_mass"] == program["eligible_mass_before_exclusions"]
    assert program["eligible_mass"] == _LARGEST
    # An exact sum past the largest float still refuses under the rule.
    with pytest.raises(ValueError) as refused:
        _assign(_people((20, 21, 22)), _declared(_contract()), weights=[1e308] * 3)
    assert str(refused.value) == "Receipt weights must have a finite sum."
    assert str(refused.value.__cause__) == (
        "integer division result too large for a float"
    )


def test_outside_strata_mass_falls_back_to_the_exact_sum_only_under_the_mass_rule():
    # The seven people at _ADULTS, aged 70, sit outside both strata: fsum
    # raises on their mass. The small weight is the one young person.
    weights, small = _NEAR_MAX_WEIGHTS, float(_NEAR_MAX_WEIGHTS[6])
    person = _people(tuple(np.where(_ADULTS, 70, 20).tolist()))
    contract = _contract(_program(stratified=True))
    targets = {"alpha_young": 1.0, "alpha_older": 1.0}
    # Without the rule, the OverflowError base 9189f745 raises, from the mass
    # outside the strata.
    with pytest.raises(OverflowError) as raised:
        _assign(person, copy.deepcopy(contract), weights=weights, targets=targets)
    assert str(raised.value) == "intermediate overflow in fsum"
    out, audit = _assign(person, _declared(contract), weights=weights, targets=targets)
    assert not out["receives_alpha"].any()
    program = audit["programs"]["receives_alpha"]
    assert program["outside_strata"] == {"n_eligible": 7, "eligible_mass": _LARGEST}
    rate = 1.0 / small
    young = _row("alpha_young", 1.0, "young", small, small, rate, rate, rate)
    older = {
        **_row("alpha_older", 1.0, "older", 0.0, 0.0, None, None, 1.0),
        "shortfall": 1.0,
    }
    program_audit = _program_audit(
        "alpha",
        2.0,
        small,
        small,
        2.0 / small,
        2.0 / small,
        [_done(young, 0.0, 0, rate * small), _done(older, 0.0, 0, 0.0)],
        (7, _LARGEST),
    )
    expected = _audit({"receives_alpha": _finished(program_audit, 0.0, 0)})
    assert json.dumps(audit, allow_nan=False) == json.dumps(expected)


# Finite, non-negative weights, often near the largest float.
_PROPERTY_WEIGHT = st.one_of(
    st.floats(min_value=0, max_value=_LARGEST),
    st.floats(min_value=2.0**1020, max_value=_LARGEST),
    st.sampled_from(_NEAR_MAX_WEIGHTS.tolist()),
)


@MASS_RULE_PROPERTY
@given(st.lists(st.tuples(_PROPERTY_WEIGHT, st.booleans()), min_size=1, max_size=10))
@example(rows=list(zip(_NEAR_MAX_WEIGHTS.tolist(), _ADULTS.tolist(), strict=True)))
@example(rows=[(weight, True) for weight in _NEAR_MAX_WEIGHTS[_ADULTS].tolist()])
# The largest float plus half a ULP ties and rounds past it; just under rounds
# to it.
@example(rows=[(_LARGEST, True), (2.0**970, True)])
@example(rows=[(_LARGEST, True), (2.0**970 - 2.0**917, True), (2.0**916, False)])
def test_property_mass_is_the_exact_sum_rounded_under_the_mass_rule(rows):
    weights = np.array([weight for weight, _ in rows])
    mask = np.array([chosen for _, chosen in rows])
    total = _exact_or_none(weights.tolist())
    masked = _exact_or_none(weights[mask].tolist())
    # Without the rule, _mass is fsum: the exact sum rounded when it returns,
    # else fsum's own OverflowError.
    try:
        expected = math.fsum(weights[mask].tolist())
    except OverflowError as error:
        with pytest.raises(OverflowError) as raised:
            receipt_module._mass(weights, mask, assign_all=False)
        assert str(raised.value) == str(error)
    else:
        assert receipt_module._mass(weights, mask, assign_all=False) == expected
        assert expected == masked
    # Under the rule, the weights pass validation exactly when their exact
    # total rounds to a float, and then every mass is its exact sum rounded.
    person = _people(tuple(range(20, 20 + len(rows))))
    contract = _declared(_contract())
    if total is None:
        with pytest.raises(ValueError, match="finite sum"):
            _assign(person, contract, weights=weights)
        return
    _, audit = _assign(person, contract, weights=weights)
    assert audit["programs"]["receives_alpha"]["eligible_mass"] == total
    assert receipt_module._mass(weights, mask, assign_all=True) == masked
    assert masked is not None
