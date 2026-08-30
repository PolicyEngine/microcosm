"""Neutral immigration facts execute the authority-layered Axiom SNAP slice.

The integration tests need the real native Axiom engine and a rulespec-us
checkout at ``MICROCOSM_RULESPEC_US``. The contract tests run without either.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.axiom_immigration import (
    US_IMMIGRATION_AXIOM_BINDING_CONTRACT,
    US_IMMIGRATION_AXIOM_MODULE,
    us_immigration_axiom_engine,
)
from microcosm.build.us_runtime.immigration import US_IMMIGRATION_FACT_COLUMNS
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_RULESPEC_US = os.environ.get("MICROCOSM_RULESPEC_US")
_ENGINE_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _ENGINE_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _DENSE_AVAILABLE = NativeCompiledDenseProgram is not None
else:
    _DENSE_AVAILABLE = False

_NEEDS_AXIOM = pytest.mark.skipif(
    not (_DENSE_AVAILABLE and _RULESPEC_US),
    reason="real Axiom dense engine and MICROCOSM_RULESPEC_US are required",
)

# These are explicit fixture facts for people with no military, tribal,
# disability-benefit, work-quarter, or grandfathered exception, and no abuse,
# petition, T-status, or agency findings under 8 USC 1641(c). They are NOT
# adapter defaults. An additional RuleSpec root fails the fixture surface
# assertion instead of silently acquiring a false/zero value.
_OTHER_LEGAL_INPUTS = {
    "alien_actively_participated_in_child_battery_or_cruelty": False,
    "alien_battered_or_subjected_to_extreme_cruelty_in_us_by_spouse_or_parent": False,
    "alien_battered_or_subjected_to_extreme_cruelty_in_us_by_spouse_or_parent_family_member": False,
    "alien_child_battered_or_subjected_to_extreme_cruelty_in_us_by_alien_spouse_or_parent": False,
    "alien_child_battered_or_subjected_to_extreme_cruelty_in_us_by_alien_spouse_or_parent_family_member": False,
    "alien_child_family_perpetrator_resides_in_alien_household": False,
    "alien_child_resides_with_battered_parent": False,
    "alien_granted_ina_101_a_15_T_nonimmigrant_status": False,
    "alien_has_approved_former_ina_244_a_3_petition": False,
    "alien_has_approved_ina_204_a_1_A_i_petition": False,
    "alien_has_approved_ina_204_a_1_A_ii_iii_or_iv_petition": False,
    "alien_has_approved_ina_204_a_1_B_i_petition": False,
    "alien_has_approved_ina_204_a_1_B_ii_or_iii_petition": False,
    "alien_has_approved_ina_240A_b_2_petition": False,
    "alien_has_pending_prima_facie_former_ina_244_a_3_petition": False,
    "alien_has_pending_prima_facie_ina_204_a_1_A_i_petition": False,
    "alien_has_pending_prima_facie_ina_204_a_1_A_ii_iii_or_iv_petition": False,
    "alien_has_pending_prima_facie_ina_204_a_1_B_i_petition": False,
    "alien_has_pending_prima_facie_ina_204_a_1_B_ii_or_iii_petition": False,
    "alien_has_pending_prima_facie_ina_240A_b_2_petition": False,
    "alien_is_child": False,
    "alien_parent_battered_or_subjected_to_extreme_cruelty_in_us_by_parent_spouse": False,
    "alien_parent_battered_or_subjected_to_extreme_cruelty_in_us_by_parent_spouse_family_member": False,
    "alien_pending_ina_101_a_15_T_application_sets_forth_prima_facie_case": False,
    "alien_spouse_or_parent_consented_or_acquiesced_to_child_battery_or_cruelty": False,
    "any_counted_qualifying_quarter_begins_after_december_31_1996": False,
    "battery_or_cruelty_perpetrator_is_in_victim_family_eligibility_unit": False,
    "battery_or_cruelty_perpetrator_resides_with_victim_in_same_household": False,
    "benefits_agency_finds_substantial_connection_to_need_from_alien_battery_or_cruelty": False,
    "benefits_agency_finds_substantial_connection_to_need_from_child_battery_or_cruelty": False,
    "benefits_agency_finds_substantial_connection_to_need_from_parent_battery_or_cruelty": False,
    "connected_person_active_duty_is_for_training": False,
    "connected_person_discharge_is_honorable": False,
    "connected_person_discharge_was_not_on_account_of_alienage": False,
    "connected_person_fulfills_title_38_section_5303A_d_minimum_active_duty_service_requirements": False,
    "connected_person_is_lawfully_residing_in_united_states": False,
    "connected_person_is_on_active_duty_in_united_states_armed_forces": False,
    "connected_person_is_veteran_as_defined_or_described_in_referenced_title_38_sections": False,
    "connected_person_was_hmong_or_highland_laotian_tribal_member_when_tribe_assisted_united_states_personnel_in_vietnam_era_operation": False,
    "individual_active_duty_is_for_training": False,
    "individual_age_on_august_22_1996": 5,
    "individual_discharge_is_honorable": False,
    "individual_discharge_was_not_on_account_of_alienage": False,
    "individual_fulfills_title_38_section_5303A_d_minimum_active_duty_service_requirements": False,
    "individual_is_american_indian_born_in_canada_to_whom_ina_section_289_applies": False,
    "individual_is_lawfully_residing_in_any_state": True,
    "individual_is_lawfully_residing_in_united_states": True,
    "individual_is_member_of_indian_tribe_as_defined_in_title_25_section_5304_e": False,
    "individual_is_on_active_duty_in_united_states_armed_forces": False,
    "individual_is_spouse_of_connected_hmong_or_highland_laotian_tribal_member": False,
    "individual_is_spouse_of_connected_person": False,
    "individual_is_unmarried_dependent_child_of_connected_hmong_or_highland_laotian_tribal_member": False,
    "individual_is_unmarried_dependent_child_of_connected_person": False,
    "individual_is_unremarried_surviving_spouse_of_deceased_connected_hmong_or_highland_laotian_tribal_member": False,
    "individual_is_unremarried_surviving_spouse_of_deceased_connected_person": False,
    "individual_is_veteran_as_defined_or_described_in_referenced_title_38_sections": False,
    "individual_receives_benefits_or_assistance_for_blindness_or_disability_within_food_stamp_act_section_3_j": False,
    "individual_was_hmong_or_highland_laotian_tribal_member_when_tribe_assisted_united_states_personnel_in_vietnam_era_operation": False,
    "individual_was_lawfully_residing_in_united_states_on_august_22_1996": False,
    "parent_spouse_consented_or_acquiesced_to_parent_battery_or_cruelty": False,
    "parent_spouse_family_perpetrator_resides_in_parent_household": False,
    "person_applies_for_federal_public_benefit": True,
    "person_attempts_to_receive_federal_public_benefit": False,
    "person_is_member_of_household_otherwise_eligible_to_participate_under_this_section": True,
    "person_receives_federal_public_benefit": False,
    "qualifying_quarters_of_coverage_worked_or_creditable_under_section_1645": 0,
    "received_federal_means_tested_public_benefit_during_any_counted_post_1996_quarter": False,
    "spouse_or_parent_consented_or_acquiesced_to_battery_or_cruelty": False,
    "spouse_or_parent_family_perpetrator_resides_in_alien_household": False,
    "surviving_spouse_marriage_fulfills_title_38_section_1304_requirements": False,
}


def test_binding_contract_uses_neutral_facts_and_requested_period() -> None:
    contract = US_IMMIGRATION_AXIOM_BINDING_CONTRACT
    assert contract.version == "us-immigration-snap-2025-12-09-v1"
    assert contract.bindings["benefit_evaluation_month_index"] == "@period.month_index"
    assert set(contract.bindings.values()) <= {
        *US_IMMIGRATION_FACT_COLUMNS,
        "age",
        "@period.month_index",
    }
    assert not {"immigration_status_str", "ssn_card_type"} & set(
        contract.bindings.values()
    )
    assert len(contract.sha256) == 64


def test_engine_constructor_preserves_binding_receipt(tmp_path: Path) -> None:
    engine = us_immigration_axiom_engine(tmp_path)
    assert engine.entity_schema() == US_SCHEMA
    assert engine.input_binding_receipt() == (
        US_IMMIGRATION_AXIOM_BINDING_CONTRACT.receipt()
    )
    assert US_IMMIGRATION_AXIOM_MODULE.endswith("obbb-alien-eligibility.yaml")


def _fixture_frame(engine) -> Frame:
    sources = set(US_IMMIGRATION_AXIOM_BINDING_CONTRACT.bindings.values()) - {
        "@period.month_index"
    }
    assert set(engine.variables()) == sources | set(_OTHER_LEGAL_INPUTS)
    ids = np.arange(1, 5)
    values = {source: np.zeros(4, dtype=bool) for source in sources}
    for source in sources:
        if source.endswith("_month_index") or source in {
            "age",
            "immigration_parole_period_years",
        }:
            values[source] = np.zeros(4, dtype=np.int64)
    values.update(
        {
            name: np.full(4, value, dtype=bool if isinstance(value, bool) else np.int64)
            for name, value in _OTHER_LEGAL_INPUTS.items()
        }
    )
    values.update(
        {
            "person_id": ids,
            **{
                US_SCHEMA.membership_column(entity): ids
                for entity in US_SCHEMA.group_entities
            },
            "age": np.full(4, 35, dtype=np.int64),
            "immigration_is_alien": np.ones(4, dtype=bool),
            "immigration_is_us_resident": np.ones(4, dtype=bool),
            "immigration_is_lawful_permanent_resident": np.array(
                [False, True, True, True]
            ),
            "immigration_was_admitted_as_refugee": np.array([True, True, False, False]),
            "immigration_refugee_admission_month_index": np.array(
                [2024 * 12, 2025 * 12, 0, 0], dtype=np.int64
            ),
            "immigration_first_qualified_status_start_month_index": np.array(
                [2024 * 12, 2025 * 12, 2021 * 12 + 8, 2021 * 12 + 7],
                dtype=np.int64,
            ),
            "immigration_status_start_month_is_imputed": np.ones(4, dtype=bool),
            "immigration_first_qualified_status_start_month_is_imputed": np.ones(
                4, dtype=bool
            ),
        }
    )
    return Frame(
        {
            "person": pd.DataFrame(values),
            **{
                entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
                for entity in US_SCHEMA.group_entities
            },
        },
        US_SCHEMA,
        {"household": Weights(values=np.ones(4), kind=WeightKind.DESIGN)},
    )


@_NEEDS_AXIOM
def test_real_axiom_preserves_history_and_evaluates_requested_month() -> None:
    engine = us_immigration_axiom_engine(_RULESPEC_US)
    frame = _fixture_frame(engine)
    output = "snap_obbb_status_and_waiting_period_eligible"

    august = engine.materialize(frame, [output], period="2026-08")[output]
    september = engine.materialize(frame, [output], period="2026-09")[output]

    # Refugee alone is not a listed current status. An LPR with refugee
    # history retains the no-new-wait exception. Ordinary LPRs cross their
    # own 60-month boundary when the evaluation month changes, without any
    # dataset regeneration or handwritten eligibility logic in the adapter.
    np.testing.assert_array_equal(august, [-1, 1, -1, 1])
    np.testing.assert_array_equal(september, [-1, 1, 1, 1])


@_NEEDS_AXIOM
def test_battered_classification_does_not_replace_statutory_petition_and_finding() -> (
    None
):
    engine = us_immigration_axiom_engine(_RULESPEC_US)
    frame = _fixture_frame(engine)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    person.loc[0, "immigration_was_admitted_as_refugee"] = False
    person.loc[0, "immigration_is_battered_alien_or_qualifying_relative"] = True
    person.loc[0, "immigration_first_qualified_status_start_month_index"] = 2020 * 12
    output = "snap_five_year_qualified_alien_residence_exception_applies"

    def evaluate() -> np.ndarray:
        return engine.materialize(
            Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")}),
            [output],
            period="2026-08",
        )[output]

    # The imputed classification does not prove the petition route or the
    # benefits agency's program-specific determination required by 1641(c).
    assert evaluate()[0] == -1
    person.loc[
        0, "alien_battered_or_subjected_to_extreme_cruelty_in_us_by_spouse_or_parent"
    ] = True
    person.loc[0, "alien_has_approved_ina_204_a_1_A_ii_iii_or_iv_petition"] = True
    assert evaluate()[0] == -1
    person.loc[
        0,
        "benefits_agency_finds_substantial_connection_to_need_from_alien_battery_or_cruelty",
    ] = True
    assert evaluate()[0] == 1


@_NEEDS_AXIOM
@pytest.mark.parametrize(
    "missing",
    [
        "qualifying_quarters_of_coverage_worked_or_creditable_under_section_1645",
        "benefits_agency_finds_substantial_connection_to_need_from_alien_battery_or_cruelty",
    ],
)
def test_missing_unbound_legal_fact_is_not_defaulted_false(missing: str) -> None:
    engine = us_immigration_axiom_engine(_RULESPEC_US)
    frame = _fixture_frame(engine)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[missing])
    incomplete = Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    with pytest.raises(RuntimeError, match=missing):
        engine.materialize(
            incomplete,
            ["snap_obbb_status_and_waiting_period_eligible"],
            period="2026-08",
        )
