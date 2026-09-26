"""Invented allocation contracts; neither source qualification nor engine runs."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from microcosm.build.us_runtime import survey_social_security as reports
from microcosm.build.us_runtime import survey_social_security_beneficiaries as ss


def lot(**changes):
    return replace(
        ss.ReportLot(
            survey="asec",
            source_vintage="invented-2025",
            reporter_id="original-1",
            amount=1200.0,
            receipt_code=1,
            reporter_age=70,
            reason_codes=(1, 0),
            publisher_allocation="publisher_no_allocation",
            definition=ss.ASEC_OASDI_DEFINITION,
            period_basis="calendar_year",
            income_year=2024,
        ),
        **changes,
    )


def roster(person_id="original-1", **changes):
    return replace(
        ss.RosterEvidence((person_id,), complete=True, housing_unit=True),
        **changes,
    )


def resolved(report=None, household=None, **assumptions):
    return ss.resolve_singleton_asec_retirement(
        report or lot(),
        household or roster(),
        closed_benefit_unit=assumptions.get("closed_benefit_unit", True),
        stable_roster_over_income_year=assumptions.get(
            "stable_roster_over_income_year", True
        ),
    )


def consume(allocations, *, ids=("original-1",), year=2024):
    return ss.validate_development_consumer_subset(
        allocations, beneficiary_ids=ids, income_year=year
    )


def test_opt_in_singleton_convention_has_exact_amount_and_convention_zeros():
    report, household = lot(amount=0.1), roster()
    allocation = resolved(report, household)
    output = consume((allocation,))
    assert allocation.report is report and allocation.roster is household
    assert allocation.assignment_resolution == "resolved_by_convention"
    assert allocation.accepted_assumptions == ss.SINGLETON_ASSUMPTIONS
    assert allocation.unresolved_amount == 0.0
    assert allocation.definition_resolution == "asec_oasdi_by_convention"
    assert allocation.period_resolution == "source_calendar_year"
    assert output.rows[0].values == (0.1, 0.0, 0.0, 0.0)
    assert output.rows[0].value_origin == "singleton_retirement_convention"
    assert output.report_keys == (report.key,)
    assert output.development_only
    assert not output.source_admission_issued
    assert not output.observed_beneficiary_income_claim
    assert not output.engine_compatibility_qualified
    assert not output.release_eligible


@pytest.mark.parametrize("age", [15, 20, 61, 70])
def test_interview_age_does_not_replace_report_category_with_eligibility(age):
    assert consume((resolved(lot(reporter_age=age)),)).rows[0].values[0] == 1200


@pytest.mark.parametrize("reasons", [(1, 0), (0, 1), (1, 1)])
def test_retirement_reasons_have_no_first_reason_priority(reasons):
    assert resolved(lot(reason_codes=reasons)).assignments[0].amount == 1200


@pytest.mark.parametrize(
    "changes",
    [
        {"closed_benefit_unit": False},
        {"stable_roster_over_income_year": False},
        {"closed_benefit_unit": 1},
        {"stable_roster_over_income_year": None},
    ],
)
def test_both_assumptions_must_be_explicit_true(changes):
    with pytest.raises(ValueError, match="ASSUMPTIONS"):
        resolved(**changes)


def test_assumptions_are_required_keyword_arguments():
    with pytest.raises(TypeError):
        ss.resolve_singleton_asec_retirement(lot(), roster())


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"survey": "acs"}, "ASEC_ONLY"),
        ({"amount": None}, "POSITIVE_REPORT"),
        ({"amount": 0.0, "receipt_code": 2}, "POSITIVE_REPORT"),
        ({"receipt_code": 2}, "REPORT_RECEIPT"),
        ({"receipt_code": None}, "REPORT_RECEIPT"),
        ({"reporter_age": 14}, "REPORT_UNIVERSE"),
        ({"reporter_age": None}, "REPORT_UNIVERSE"),
        ({"reason_codes": (7, 0)}, "RETIREMENT_ONLY"),
        ({"reason_codes": (8, 0)}, "RETIREMENT_ONLY"),
        ({"reason_codes": (1, 2)}, "RETIREMENT_ONLY"),
        ({"reason_codes": (0, 0)}, "RETIREMENT_ONLY"),
        ({"reason_codes": (1, None)}, "RETIREMENT_ONLY"),
        ({"publisher_allocation": "publisher_allocated"}, "ALLOCATION_ORIGIN"),
        ({"publisher_allocation": "unresolved"}, "ALLOCATION_ORIGIN"),
        ({"definition": "social_security_and_railroad_retirement"}, "DEFINITION"),
        ({"definition": "unknown"}, "DEFINITION"),
        ({"period_basis": "rolling_12_months"}, "CALENDAR_YEAR"),
        ({"period_basis": "unknown", "income_year": None}, "CALENDAR_YEAR"),
    ],
)
def test_unsupported_reports_never_use_singleton_shortcut(changes, error):
    with pytest.raises(ValueError, match=error):
        resolved(lot(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"complete": None},
        {"complete": False},
        {"housing_unit": None},
        {"housing_unit": False},
        {"person_ids": ("original-1", "child")},
        {"person_ids": ("different-person",)},
        {"person_ids": ()},
    ],
)
def test_unknown_nonhousing_or_nonsingleton_roster_refuses(changes):
    with pytest.raises(ValueError, match="SINGLETON_ROSTER"):
        resolved(household=roster(**changes))


@pytest.mark.parametrize("amount", [None, 0.0, 1200.0])
def test_unresolved_amount_and_source_zero_never_become_beneficiary_zeros(amount):
    report = lot(
        amount=amount,
        receipt_code=2 if amount == 0 else 1,
        reason_codes=(0, 0) if amount == 0 else (1, 0),
    )
    allocation = ss.unresolved_report_lot(report)
    assert allocation.unresolved_amount is amount
    assert allocation.assignments == ()
    assert allocation.assignment_resolution == "unresolved"
    assert allocation.definition_resolution == "unresolved"
    assert allocation.period_resolution == "unresolved"
    with pytest.raises(ValueError, match="UNRESOLVED_ASSIGNMENT"):
        consume((allocation,))


def test_below15_child_can_remain_candidate_for_code7_without_any_amount():
    report = lot(reason_codes=(7, 0))
    child = ss.BeneficiaryCandidate(
        "child-age-10",
        tuple(reports.COMPONENTS[i] for i in reports.REASON_COMPONENTS[7]),
        "invented parent pointer; age10 does not exclude beneficiary status",
    )
    allocation = ss.unresolved_report_lot(report, candidates=(child,))
    assert allocation.candidates == (child,)
    assert allocation.unresolved_amount == 1200.0
    assert not allocation.assignments
    with pytest.raises(ValueError, match="UNRESOLVED_ASSIGNMENT"):
        consume((allocation,), ids=("child-age-10",))


def test_two_child_candidates_do_not_imply_equal_shares():
    candidates = tuple(
        ss.BeneficiaryCandidate(child, (reports.COMPONENTS[2],), "parent pointer")
        for child in ("child-1", "child-2")
    )
    allocation = ss.unresolved_report_lot(
        lot(reason_codes=(6, 0)), candidates=candidates
    )
    assert allocation.unresolved_amount == 1200.0
    assert allocation.assignments == ()


def test_positive_classifier_scores_are_report_evidence_not_four_benefits():
    report_values = reports.complete_positive_basis(
        np.array([1200.0]),
        np.full((1, 4), np.nan),
        np.ones((1, 4), dtype=bool),
        np.array([[0.1, 0.2, 0.3, 0.4]]),
    )
    assert (report_values > 0).all()
    allocation = resolved()
    manufactured = replace(
        allocation,
        assignments=tuple(
            ss.BeneficiaryAssignment("original-1", component, float(value))
            for component, value in zip(
                reports.COMPONENTS, report_values[0], strict=True
            )
        ),
    )
    with pytest.raises(ValueError, match="ASSIGNMENT_CONVENTION"):
        consume((manufactured,))
    assert consume((allocation,)).rows[0].values[1:] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"assignment_resolution": "observed"}, "RESOLUTION"),
        ({"definition_resolution": "unresolved"}, "DEFINITION_RESOLUTION"),
        ({"period_resolution": "unresolved"}, "PERIOD_RESOLUTION"),
        ({"accepted_assumptions": ()}, "ASSUMPTIONS"),
        ({"convention": "arbitrary_model"}, "CONVENTION"),
        ({"unresolved_amount": None}, "UNRESOLVED_AMOUNT"),
        ({"unresolved_amount": 1.0}, "UNRESOLVED_AMOUNT"),
        ({"assignments": ()}, "ASSIGNMENT_CONVENTION"),
        ({"candidates": ()}, "CANDIDATE_CONVENTION"),
        ({"roster": None}, "ROSTER_TYPE"),
    ],
)
def test_consumer_revalidates_constructible_descriptive_artifact(changes, error):
    with pytest.raises(ValueError, match=error):
        consume((replace(resolved(), **changes),))


@pytest.mark.parametrize("amount", [1199.0, 1201.0, float("nan"), float("inf"), -1.0])
def test_no_residual_budget_or_nonfinite_assignment_is_accepted(amount):
    allocation = resolved()
    changed = replace(allocation.assignments[0], amount=amount)
    with pytest.raises(ValueError, match="ASSIGNMENT_CONVENTION"):
        consume((replace(allocation, assignments=(changed,)),))


def test_duplicate_original_report_identity_refuses_even_with_changed_payload():
    first = resolved()
    second = resolved(lot(amount=2400.0))
    assert first.report.key == second.report.key
    with pytest.raises(ValueError, match="DUPLICATE_REPORT_LOT"):
        consume((first, second))


def test_distinct_reports_for_same_beneficiary_require_overlap_resolution():
    first = resolved()
    second = resolved(lot(source_vintage="another-invented-vintage"))
    with pytest.raises(ValueError, match="OVERLAPPING_BENEFICIARY_LOTS"):
        consume((first, second))


def test_parent_combined_plus_child_report_cannot_silently_add_or_subtract():
    parent = ss.unresolved_report_lot(lot(reason_codes=(7, 0)))
    child = resolved(lot(reporter_id="child", reporter_age=15), roster("child"))
    with pytest.raises(ValueError, match="UNRESOLVED_ASSIGNMENT"):
        consume((parent, child), ids=("original-1", "child"))


@pytest.mark.parametrize(
    "ids", [(), ("other",), ("original-1", "missing"), ("original-1", "original-1")]
)
def test_consumer_requires_exact_explicit_subset(ids):
    with pytest.raises(ValueError, match="SUBSET"):
        consume((resolved(),), ids=ids)


def test_consumer_rejects_wrong_calendar_year():
    with pytest.raises(ValueError, match="CONSUMER_PERIOD"):
        consume((resolved(),), year=2025)


def test_subset_order_is_requested_order_without_population_completion_claim():
    first = resolved()
    second = resolved(lot(reporter_id="original-2"), roster("original-2"))
    result = consume((first, second), ids=("original-2", "original-1"))
    assert tuple(row.beneficiary_id for row in result.rows) == (
        "original-2",
        "original-1",
    )
    assert len(result.rows) == 2
    assert result.report_keys == (second.report.key, first.report.key)


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf"), True, 1200])
def test_report_amount_requires_finite_nonnegative_float_or_explicit_unknown(amount):
    with pytest.raises(ValueError, match="REPORT_AMOUNT"):
        ss.unresolved_report_lot(lot(amount=amount))


def test_artifacts_retain_original_report_and_are_immutable():
    report = lot()
    allocation = resolved(report)
    assert allocation.report == report
    with pytest.raises(FrozenInstanceError):
        report.amount = 1.0
    with pytest.raises(FrozenInstanceError):
        allocation.unresolved_amount = 100.0
