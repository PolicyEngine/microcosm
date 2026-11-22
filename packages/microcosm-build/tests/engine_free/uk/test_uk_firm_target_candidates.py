"""The M2 UK firm target candidates: declarations only, none in the contract."""

from __future__ import annotations

import json
from dataclasses import asdict

from microcosm.build.uk_runtime.firm_generation import (
    UK_FIRM_TARGET_IDS,
    load_uk_firms_contract,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
)
from microcosm.build.uk_runtime.firm_target_candidates import (
    FIXTURE_AVAILABLE,
    PENDING_LEDGER_SOURCE,
    UK_FIRM_TARGET_CANDIDATE_STATUSES,
    UK_FIRM_TARGET_CANDIDATES,
    uk_firm_target_candidates_feeding,
)
from test_support.microcosm_build.uk_firm_target_keys import carried_forbidden_keys
from test_support.paths import paths_for

SNAPSHOT = (
    paths_for("microcosm-build").tests
    / "fixtures"
    / "uk"
    / "policyengine_uk_firm_inputs.json"
)

#: The ONS UK Business 2025 tables Microcosm holds as fixtures.
_FIXTURE_TABLES = {"Table 8", "Table 13", "Table 15"}

#: The tables pending candidates name: ONS UK Business Table 3 (no regional
#: extract in Microcosm) and Corporation Tax statistics Table 12A.
_PENDING_TABLES = {"Table 3", "Table 12A"}

#: Every family that must declare at least one pending candidate.
_PENDING_FAMILIES = {
    "ons_uk_business",
    "dbt_bpe",
    "ons_abs",
    "hmrc_corporation_tax",
    "hmrc_capital_allowances",
    "hmrc_rd_tax_relief",
    "hmrc_paye_rti",
    "mhclg_nndr",
    "voa_non_domestic_rating",
}


def _snapshot_inputs() -> set[str]:
    return set(json.loads(SNAPSHOT.read_text(encoding="utf-8"))["firm_input_names"])


def test_candidates_are_value_free_and_hook_free() -> None:
    carried = carried_forbidden_keys(
        [asdict(candidate) for candidate in UK_FIRM_TARGET_CANDIDATES]
    )

    assert not carried, sorted(carried)


def test_candidate_ids_are_unique_and_outside_the_contract() -> None:
    ids = [candidate.target_id for candidate in UK_FIRM_TARGET_CANDIDATES]

    assert len(ids) == len(set(ids))
    assert set(ids).isdisjoint(UK_FIRM_TARGET_IDS.values())


def test_candidates_carry_no_ledger_record_set_id() -> None:
    assert all(
        candidate.ledger_record_set_id is None
        for candidate in UK_FIRM_TARGET_CANDIDATES
    )


def test_committed_contract_keeps_its_eight_targets() -> None:
    contract = load_uk_firms_contract()

    assert len(contract["targets"]) == 8
    assert [target["target_id"] for target in contract["targets"]] == list(
        UK_FIRM_TARGET_IDS.values()
    )


def test_candidates_feed_policyengine_uk_firm_inputs() -> None:
    inputs = _snapshot_inputs()

    for candidate in UK_FIRM_TARGET_CANDIDATES:
        assert candidate.feeds_inputs, candidate.target_id
        assert set(candidate.feeds_inputs) <= inputs, (
            candidate.target_id,
            sorted(set(candidate.feeds_inputs) - inputs),
        )


def test_candidate_statuses_families_and_fixture_tables() -> None:
    for candidate in UK_FIRM_TARGET_CANDIDATES:
        assert candidate.status in UK_FIRM_TARGET_CANDIDATE_STATUSES
        assert candidate.entity == "firm"
        assert candidate.publication
        assert candidate.concept.startswith(("uk.firm.", "uk.tax."))
        assert all(
            dimension.startswith(("uk.firm.", "uk.tax.", "uk.property."))
            for dimension in candidate.dimensions
        ), candidate.target_id
        assert candidate.target_id.startswith(f"{candidate.source_name}."), (
            candidate.target_id
        )
    fixture = [c for c in UK_FIRM_TARGET_CANDIDATES if c.status == FIXTURE_AVAILABLE]
    pending = [
        c for c in UK_FIRM_TARGET_CANDIDATES if c.status == PENDING_LEDGER_SOURCE
    ]

    assert {c.source_table for c in fixture} == _FIXTURE_TABLES
    assert all(c.family == "ons_uk_business" for c in fixture)
    assert {c.family for c in pending} == _PENDING_FAMILIES
    assert {c.source_table for c in pending} - {None} == _PENDING_TABLES


#: not_set policyengine-uk inputs that no candidate informs yet: no source is
#: declared for them.
_NOT_SET_WITHOUT_CANDIDATE = {
    "firm_apprenticeship_levy_allowance_share",
    "firm_claims_employment_allowance",
    "firm_deductions_allowance_share",
    "firm_exempt_distributions",
    "firm_in_freeport",
    "firm_is_limited_cost_trader",
    "firm_is_single_director_company",
    "firm_reduced_rated_supplies",
    "firm_uses_flat_rate_scheme",
    "firm_zero_rated_class_1_earnings",
    "firm_zero_rated_supplies",
    # Pay-distribution and minimum-wage inputs (policyengine-uk ca037fad2).
    "firm_employees_above_auto_enrolment_upper_limit",
    "firm_employees_below_auto_enrolment_earnings_trigger",
    "firm_employees_below_secondary_threshold",
    "firm_low_paid_hours",
    "firm_low_paid_paybill",
    "firm_paybill_above_auto_enrolment_upper_limit",
    "firm_paybill_below_auto_enrolment_earnings_trigger",
    "firm_paybill_below_secondary_threshold",
}


def test_not_set_inputs_without_a_candidate_are_pinned() -> None:
    not_set = {
        name
        for name, status in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.items()
        if status == "not_set"
    }
    without = {name for name in not_set if not uk_firm_target_candidates_feeding(name)}

    assert without == _NOT_SET_WITHOUT_CANDIDATE


def test_candidates_feeding_names_the_candidates_for_an_input() -> None:
    feeding = uk_firm_target_candidates_feeding("firm_paybill")

    assert feeding
    assert all(
        "firm_paybill" in candidate.feeds_inputs
        for candidate in UK_FIRM_TARGET_CANDIDATES
        if candidate.target_id in feeding
    )
    assert uk_firm_target_candidates_feeding("not_a_firm_input") == ()
