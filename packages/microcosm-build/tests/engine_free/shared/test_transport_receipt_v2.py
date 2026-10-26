"""Shared receipt contracts on synthetic people, never country declarations.

The v1 oracle below freezes the assignment operation at 4f333f2c; the toy
contract is loaded from the existing composed-graph fixture. V2 deliberately
requires disjoint masks on the actual people rather than overlap precedence.
"""

from __future__ import annotations

import copy
import json
import math
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
    # Binary fractions retain exact, varied float capacities without rounding
    # away the comparison against the independent rational reference below.
    capacity = st.integers(0, 32).map(lambda value: value / 8)
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


@pytest.mark.parametrize("target", [np.int64(1), np.float32(1), np.float64(1)])
def test_v2_numpy_real_targets_are_accepted(target):
    out, audit = _assign(_people((20,)), targets={"alpha_total": target})
    assert out["receives_alpha"].tolist() == [True]
    assert type(audit["programs"]["receives_alpha"]["strata"][0]["target"]) is float


@pytest.mark.parametrize("target", [True, np.bool_(True)])
def test_v2_boolean_targets_refuse(target):
    with pytest.raises(ValueError, match="target must be finite"):
        _assign(_people((20,)), targets={"alpha_total": target})
