"""Financial ancestry admission over actual invented cold/required graph runs."""

import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_us_graph_atomic_survey_financial import known_financial_run  # noqa: F401
from test_us_survey_origin_budget import _advance

from microcosm.build.us_runtime import graph_atomic_survey_financial as runner
from microcosm.build.us_runtime import survey_financial_successor as financial
from microcosm.build.us_runtime import survey_origin_budget as budgets


def _budget(run, *, candidate=None):
    prefix = run.prefix
    return budgets.freeze_survey_origin_budget(
        prefix.preparation,
        allocated_population=prefix.allocated_population,
        clone_population=prefix.clone_population,
        geography_config=prefix.geography_config,
        candidate=candidate,
    )


@pytest.fixture(scope="module")
def admitted_financial(request):
    case = request.getfixturevalue("known_financial_run")
    budget = _budget(case.cold)
    binding = financial.admit_survey_financial_population(
        budget, financial_run=case.cold
    )
    yield SimpleNamespace(case=case, budget=budget, binding=binding)
    runner._pure_run(case.cold, runner._run_entry(case.cold))


def test_financial_candidate_replay_and_weight_only_successor(admitted_financial):
    case, budget, binding = (
        admitted_financial.case,
        admitted_financial.budget,
        admitted_financial.binding,
    )
    original = case.cold.prefix.clone_population
    enriched = case.cold.financial_population
    budget_view = budget.checked_view()
    assert budget_view.initial_population is original
    financial_view = binding.checked_view()
    assert financial_view.previous is original
    assert financial_view.current is enriched
    warm_budget = _budget(case.warm, candidate=budget.payload)
    warm = financial.admit_survey_financial_population(
        warm_budget, financial_run=case.warm, candidate=binding.payload
    )
    assert warm.payload == binding.payload
    assert json.loads(binding.payload)["sampling_bounds_changed"] is False
    assert json.loads(binding.payload)["release_eligible"] is False
    incoming = enriched.frame.weights_for("household").values
    current = _advance(enriched, incoming * 0.75, "invented.financial_calibrated")
    admitted = budgets.admit_survey_weight_only_population(
        budget, previous=enriched, current=current, previous_binding=binding
    )
    checked = admitted.checked_view()
    assert checked.previous is enriched and checked.current is current
    assert checked.previous_binding is binding
    assert (
        checked.document["previous_binding_sha256"]
        == hashlib.sha256(binding.payload).hexdigest()
    )
    assert checked.document["constraint_digest"] == budget_view.grouped_bounds.digest
    np.testing.assert_array_equal(
        current.design_weights["household"], original.design_weights["household"]
    )
    # The old initial path still applies exactly to its original clone.
    legacy_current = _advance(original, incoming * 0.75, "invented.original_calibrated")
    legacy = budgets.admit_survey_weight_only_population(
        budget, previous=original, current=legacy_current
    )
    assert json.loads(legacy.payload)["previous_binding_sha256"] is None
    assert budget.checked_view().initial_population is original


def test_unissued_copy_and_wrong_original_budget_refuse(admitted_financial):
    case = admitted_financial.case
    forged = replace(case.cold)
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        financial.admit_survey_financial_population(
            admitted_financial.budget, financial_run=forged
        )
    with pytest.raises(financial.SurveyFinancialSuccessorError, match="NO_PUBLIC"):
        financial.SamplingOriginFinancialSuccessor(admitted_financial.binding.payload)
    # Same numerical budget bytes cannot substitute for the retained owner.
    other = _budget(case.warm, candidate=admitted_financial.budget.payload)
    with pytest.raises(ValueError, match="ORIGINAL_BUDGET_RUN_ANCESTRY"):
        financial.admit_survey_financial_population(other, financial_run=case.cold)


def test_arbitrary_nonweight_changes_still_refuse(admitted_financial):
    run = admitted_financial.case.cold
    previous = run.financial_population
    incoming = previous.frame.weights_for("household").values
    current = _advance(previous, incoming * 0.75, "invented.requires_financial_binding")
    with pytest.raises(ValueError, match="INITIAL_PREVIOUS_REQUIRED"):
        budgets.admit_survey_weight_only_population(
            admitted_financial.budget, previous=previous, current=current
        )
    for column in ("age", runner.values.OUTPUTS[0]):
        changed = _advance(previous, incoming * 0.75, "invented.changed_" + column)
        changed.frame.person.loc[changed.frame.person.index[0], column] += 1
        with pytest.raises(ValueError, match="SUCCESSOR_NONWEIGHT_TABLE"):
            budgets.admit_survey_weight_only_population(
                admitted_financial.budget,
                previous=previous,
                current=changed,
                previous_binding=admitted_financial.binding,
            )


def test_rehashed_graph_artifact_cannot_replace_issued_draw(admitted_financial):
    run = admitted_financial.case.cold
    node = run.manifest.node(runner.financial.APPLY_PREFIX + ".000")
    key = node.opaque_artifacts["raw_draw"]
    directory = run.store.object_path(key)
    payload_path, meta_path = directory / "payload.bin", directory / "meta.json"
    payload, metadata = payload_path.read_bytes(), meta_path.read_bytes()
    changed = (
        payload[:-8]
        + (np.frombuffer(payload[-8:], dtype="<f8") + 1).astype("<f8").tobytes()
    )
    document = json.loads(metadata)
    document["payloads"]["payload.bin"] = {
        "sha256": hashlib.sha256(changed).hexdigest(),
        "size": len(changed),
    }
    try:
        payload_path.write_bytes(changed)
        meta_path.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"))
        )
        assert run.store.load_bytes(key) == changed
        with pytest.raises(ValueError, match="FINANCIAL_RUN_ARTIFACT_CHANGED"):
            admitted_financial.binding.checked_view()
    finally:
        payload_path.write_bytes(payload)
        meta_path.write_bytes(metadata)


def test_final_budget_support_borrow_cannot_mutate_financial_run(admitted_financial):
    binding, fired = admitted_financial.binding, []
    table = admitted_financial.case.cold.financial_population.frame.table("household")
    index = table.index[0]
    original = table.loc[index, "survey_observed_state"]

    def profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code is budgets.geography._read_support.__code__
            and caller is not None
            and caller.f_code is budgets._final_budget_state.__code__
            and caller.f_back is not None
            and caller.f_back.f_code is financial._final_state.__code__
            and not fired
        ):
            fired.append(True)
            table.loc[index, "survey_observed_state"] = "99"

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            ValueError, match="FINANCIAL_RUN_(POPULATION|ATTACHED_POPULATION)_CHANGED"
        ):
            binding.checked_view()
    finally:
        sys.setprofile(previous)
        table.loc[index, "survey_observed_state"] = original
    assert fired == [True]


def test_final_support_borrow_cannot_forge_returned_view(admitted_financial):
    binding, fired = admitted_financial.binding, []
    run = admitted_financial.case.cold

    def profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code is budgets.geography._read_support.__code__
            and caller is not None
            and caller.f_code is budgets._final_budget_state.__code__
            and caller.f_back is not None
            and caller.f_back.f_code is financial._final_state.__code__
            and not fired
        ):
            outer = caller.f_back.f_back
            assert (
                outer.f_code
                is financial.SamplingOriginFinancialSuccessor.checked_view.__code__
            )
            fired.append(True)
            # The regressed implementation exposed its detached result before
            # this last I/O; changing it left every actual owner untouched.
            result = outer.f_locals.get("result")
            if result is not None:
                object.__setattr__(result, "payload", b"{}")
                object.__setattr__(result, "digest", "0" * 64)
                object.__setattr__(result, "current", run.prefix.clone_population)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        view = binding.checked_view()
    finally:
        sys.setprofile(previous)
    assert fired == [True]
    assert view.payload == binding.payload
    assert view.digest == hashlib.sha256(binding.payload).hexdigest()
    assert view.current is run.financial_population
    assert view.previous is run.prefix.clone_population
    assert view.budget is admitted_financial.budget


def test_financial_borrow_cannot_relax_detached_budget_bounds(admitted_financial):
    previous = admitted_financial.case.cold.financial_population
    incoming = previous.frame.weights_for("household").values
    # With this fixture's p=1, five times incoming is below the generic
    # eight-times-design row guard but above the actual four-times-incoming U.
    current = _advance(previous, incoming * 5, "invented.over_origin_budget")
    fired = []

    def profile(frame, event, arg):
        if (
            event != "return"
            or frame.f_code is not budgets.geography._read_support.__code__
        ):
            return
        outer = frame.f_back
        while outer is not None:
            if outer.f_code is budgets._successor_document.__code__:
                view = outer.f_locals["budget_view"]
                if not any(view is value for value in fired):
                    fired.append(view)
                    constraint = view.grouped_bounds
                    object.__setattr__(
                        view,
                        "grouped_bounds",
                        budgets.group_bounds.GroupedUpperBounds(
                            constraint.household_ids,
                            constraint.group_indices,
                            constraint.absolute_bounds * 10,
                        ),
                    )
                    object.__setattr__(view, "digest", "0" * 64)
                    for row in view.document["origins"]:
                        row["design_bound_float64_hex"] = (1e30).hex()
                return
            outer = outer.f_back

    old_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            ValueError, match="group weights exceed frozen absolute bounds"
        ):
            budgets.admit_survey_weight_only_population(
                admitted_financial.budget,
                previous=previous,
                current=current,
                previous_binding=admitted_financial.binding,
            )
    finally:
        sys.setprofile(old_profile)
    assert fired


def test_run_borrow_cannot_forge_original_budget_digest(admitted_financial):
    fired = []

    def profile(frame, event, arg):
        if (
            event != "return"
            or frame.f_code is not budgets.geography._read_support.__code__
        ):
            return
        outer = frame.f_back
        while outer is not None:
            if outer.f_code is financial.admit_survey_financial_population.__code__:
                view = outer.f_locals.get("budget_view")
                if view is not None and not fired:
                    fired.append(True)
                    object.__setattr__(view, "digest", "0" * 64)
                return
            outer = outer.f_back

    old_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="ORIGINAL_BUDGET_RUN_ANCESTRY"):
            financial.admit_survey_financial_population(
                admitted_financial.budget,
                financial_run=admitted_financial.case.cold,
            )
    finally:
        sys.setprofile(old_profile)
    assert fired == [True]
