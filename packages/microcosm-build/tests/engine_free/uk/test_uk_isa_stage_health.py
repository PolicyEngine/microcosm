"""Stage-health checks of the ISA subscriptions and Lifetime ISA anchor (microcosm#1415)."""

from __future__ import annotations

import copy

import pytest

from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

PARAMETERS = {
    "maximum_relative_total_deviation": 1e-6,
    "maximum_household_sum_error_gbp": 0.01,
}
PASSING = {
    "stage": "isa_subscriptions",
    "balance_split": {"max_abs_household_sum_error_gbp": 1e-9},
    "checks": {
        "violations": {
            "over_overall_limit": 0,
            "lifetime_over_annual_limit": 0,
            "subscriber_under_minimum_age": 0,
            "cash_without_cash_holding": 0,
            "stocks_without_stocks_holding": 0,
            "both_class_missing_a_part": 0,
            "negative_amounts": 0,
        }
    },
    "growth": {
        "cash_target_gbp": 100.0,
        "cash_achieved_gbp": 100.0,
        "stocks_and_shares_target_gbp": 50.0,
        "stocks_and_shares_achieved_gbp": 50.0,
    },
    "both_type_split": {
        "saturated": None,
        "stocks_and_shares_target_gbp": 10.0,
        "stocks_and_shares_achieved_gbp": 10.0,
    },
    "lifetime_isa": {"saturated": None, "target_gbp": 5.0, "achieved_gbp": 5.0},
    "selection": {"cash_only": {"base": {"shortfall_weight_unmet": 12.0}}},
}


def _gate(evidence, check="isa_subscriptions", parameters=PARAMETERS):
    return uk_stage_health_gate(
        evidence=evidence, stage=evidence["stage"], check=check, parameters=parameters
    )


def test_a_coherent_receipt_passes_and_reports_shortfalls():
    result = _gate(PASSING)
    assert result.passed, result.failures
    assert (
        result.details["selection"]["cash_only"]["base"]["shortfall_weight_unmet"]
        == 12.0
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("checks", "violations", "over_overall_limit"), 1),
        (("balance_split", "max_abs_household_sum_error_gbp"), 5.0),
        (("growth", "cash_achieved_gbp"), 90.0),
        (("both_type_split", "stocks_and_shares_achieved_gbp"), 9.0),
        (("lifetime_isa", "achieved_gbp"), 4.0),
    ],
)
def test_broken_invariants_fail(path, value):
    evidence = copy.deepcopy(PASSING)
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not _gate(evidence).passed


def test_saturation_is_named_not_failed():
    evidence = copy.deepcopy(PASSING)
    evidence["lifetime_isa"] = {
        "saturated": "target_above_reach",
        "target_gbp": 5.0,
        "achieved_gbp": 1.0,
    }
    result = _gate(evidence)
    assert result.passed
    assert result.details["saturations_named"] == ["lifetime_isa:target_above_reach"]


ANCHOR_PARAMETERS = {"maximum_relative_deviation": 0.01}


def _anchor_evidence(achieved=1004.0, **anchor):
    return {
        "stage": "was_lisa",
        "ownership_model": {
            "anchor": {
                "target_holders": 1000.0,
                "logit_shift": 1.2,
                "saturated": None,
                "removed_holders": 0,
                **anchor,
            }
        },
        "anchored_holders_after_cap": achieved,
    }


@pytest.mark.parametrize(
    ("evidence", "passes"),
    [
        (_anchor_evidence(), True),
        (_anchor_evidence(900.0), False),
        (_anchor_evidence(1000.0, removed_holders=3), False),
        # Holders drawn at or above the payers are left as drawn.
        (_anchor_evidence(1300.0, logit_shift=0.0), True),
        (_anchor_evidence(900.0, logit_shift=0.0), False),
        (_anchor_evidence(1000.0, logit_shift=-0.5), False),
        (
            _anchor_evidence(400.0, logit_shift=30.0, saturated="target_above_reach"),
            True,
        ),
    ],
)
def test_the_lifetime_isa_anchor_check(evidence, passes):
    result = _gate(evidence, "was_lisa_anchor", ANCHOR_PARAMETERS)
    assert result.passed is passes, result.failures


def test_an_unreachable_anchor_is_named():
    result = _gate(
        _anchor_evidence(400.0, logit_shift=30.0, saturated="target_above_reach"),
        "was_lisa_anchor",
        ANCHOR_PARAMETERS,
    )
    assert result.details["saturations_named"] == ["anchor:target_above_reach"]
