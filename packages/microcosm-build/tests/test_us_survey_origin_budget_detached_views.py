"""Detached value checks over an actual invented atomic prefix, without fitting."""

import hashlib
import json
import sys

import numpy as np
import pytest
from test_us_survey_origin_budget import _advance
from test_us_survey_origin_budget_atomic_geography import (  # noqa: F401
    atomic_budget,
    known_atomic_run,
)

from microcosm.build.us_runtime import survey_origin_budget as owner


def _forge_detached_values(outer, replacement):
    # Only fresh detached locals are changed. The actual budget, source,
    # Populations and original immutable payload are never modified.
    document = outer.f_locals.get("document")
    if isinstance(document, dict):
        document.clear()
        document["forged"] = True
    constraint = outer.f_locals.get("constraint")
    if constraint is not None:
        object.__setattr__(
            constraint, "absolute_bounds", constraint.absolute_bounds * 100
        )
        object.__setattr__(constraint, "constraint_digest", "0" * 64)
    result = outer.f_locals.get("result")
    if result is not None:
        for name, value in (
            ("payload", b"{}"),
            ("digest", "0" * 64),
            ("document", {"forged": True}),
            ("initial_population", replacement),
            ("current", replacement),
            ("previous_binding", object()),
        ):
            if hasattr(result, name):
                object.__setattr__(result, name, value)


def _borrow_during_last_support_return(handle, replacement):
    checked_code = type(handle).checked_view.__code__
    fired = []

    def profile(frame, event, argument):
        caller = frame.f_back
        if (
            event != "return"
            or frame.f_code is not owner.geography._read_support.__code__
            or caller is None
            or caller.f_code is not owner._final_budget_state.__code__
            or fired
        ):
            return
        outer = caller.f_back
        if type(handle) is owner.SamplingOriginSuccessor:
            if (
                outer is None
                or outer.f_code is not owner._final_successor_state.__code__
            ):
                return
            outer = outer.f_back
        if outer is not None and outer.f_code is checked_code:
            fired.append(True)
            _forge_detached_values(outer, replacement)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = handle.checked_view()
    finally:
        sys.setprofile(previous)
    assert fired == [True]
    return result


def _assert_decoded_document_cannot_be_forged(handle, expected_code):
    checked_code = type(handle).checked_view.__code__
    fired = []

    def profile(frame, event, result):
        if (
            event == "return"
            and frame.f_code is json.loads.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is checked_code
        ):
            assert type(result) is dict
            fired.append(True)
            result["release_eligible"] = True

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(owner.SurveyOriginBudgetError, match=expected_code):
            handle.checked_view()
    finally:
        sys.setprofile(previous)
    assert fired == [True]


def test_original_budget_returns_fresh_bound_values_after_final_borrow(request):
    case, budget = request.getfixturevalue("atomic_budget")
    issued_payload = budget.payload
    expected_document = json.loads(issued_payload)
    view = _borrow_during_last_support_return(budget, case.cold.allocated_population)
    assert view.payload == issued_payload == budget.payload
    assert view.digest == hashlib.sha256(issued_payload).hexdigest()
    assert view.document == expected_document
    assert view.preparation is case.cold.preparation
    assert view.allocated_population is case.cold.allocated_population
    assert view.initial_population is case.cold.clone_population
    bounds = view.grouped_bounds
    assert tuple(bounds.household_ids) == tuple(expected_document["household_ids"])
    assert bounds.group_indices.tolist() == expected_document["group_indices"]
    assert [float(value).hex() for value in bounds.absolute_bounds] == [
        row["upper_float64_hex"] for row in expected_document["origins"]
    ]
    expected = owner.group_bounds.GroupedUpperBounds(
        expected_document["household_ids"],
        expected_document["group_indices"],
        [
            float.fromhex(row["upper_float64_hex"])
            for row in expected_document["origins"]
        ],
    )
    assert bounds.digest == expected.digest
    weights = case.cold.clone_population.frame.weights_for("household").values
    np.testing.assert_array_equal(bounds.check(weights), expected.check(weights))
    with pytest.raises(ValueError, match="group weights exceed frozen absolute bounds"):
        bounds.check(weights * 5)
    _assert_decoded_document_cannot_be_forged(budget, "FINAL_BUDGET_VIEW_DOCUMENT")


def test_weight_successor_returns_fresh_values_after_final_borrow(request):
    case, budget = request.getfixturevalue("atomic_budget")
    initial = case.cold.clone_population
    current = _advance(
        initial,
        initial.frame.weights_for("household").values * 0.75,
        "invented.detached_successor_view",
    )
    binding = owner.admit_survey_weight_only_population(
        budget, previous=initial, current=current
    )
    issued_payload = binding.payload
    expected_document = json.loads(issued_payload)
    view = _borrow_during_last_support_return(binding, initial)
    assert view.payload == issued_payload == binding.payload
    assert view.digest == hashlib.sha256(issued_payload).hexdigest()
    assert view.document == expected_document
    assert view.budget is budget
    assert view.previous is initial and view.current is current
    assert view.previous_binding is None
    assert view.document["budget_sha256"] == hashlib.sha256(budget.payload).hexdigest()
    assert view.document["previous_binding_sha256"] is None
    _assert_decoded_document_cannot_be_forged(binding, "FINAL_SUCCESSOR_VIEW_DOCUMENT")


def test_original_budget_refuses_constructor_return_constraint_mutation(request):
    _, budget = request.getfixturevalue("atomic_budget")
    checked_code = owner.SamplingOriginBudget.checked_view.__code__
    constructor_code = owner.group_bounds.GroupedUpperBounds.__post_init__.__code__
    # These are independent mutations: the member case still has immutable
    # storage, the digest case leaves every array intact, and the storage case
    # retains every numerical value and the correct digest.
    for mutation in ("members", "digest", "storage"):
        fired = []

        def profile(frame, event, argument, *, _mutation=mutation, _fired=fired):
            if event != "return" or frame.f_code is not constructor_code or _fired:
                return
            constructor = frame.f_back
            outer = None if constructor is None else constructor.f_back
            if outer is None or outer.f_code is not checked_code:
                return
            constraint = frame.f_locals["self"]
            if _mutation == "members":
                object.__setattr__(
                    constraint,
                    "_members",
                    tuple(np.frombuffer(b"", dtype="<i8") for _ in constraint._members),
                )
            elif _mutation == "digest":
                object.__setattr__(constraint, "constraint_digest", "0" * 64)
            else:
                object.__setattr__(
                    constraint,
                    "_members",
                    tuple(member.copy() for member in constraint._members),
                )
            _fired.append(True)

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            with pytest.raises(
                owner.SurveyOriginBudgetError, match="FINAL_BUDGET_VIEW_CONSTRAINT"
            ):
                budget.checked_view()
        finally:
            sys.setprofile(previous)
        assert fired == [True], mutation
