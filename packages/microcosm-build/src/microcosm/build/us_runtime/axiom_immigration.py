"""Bind neutral immigration facts to the separately encoded Axiom authorities.

This is a rename-only contract, not another implementation of eligibility.
Status combinations, waiting periods, exceptions, and authority precedence
remain in RuleSpec and execute in the real Axiom runtime. The requested
materialization month supplies the evaluation clock; the population dataset
stores event months, never a frozen number of elapsed years.

The SNAP composition is scoped to the December 9, 2025 guidance layer. Its
RuleSpec metadata records the remaining historical-as-known and exact-day
runtime gates. This helper is not a complete SNAP simulation: unbound inputs
such as military-family connections, qualifying work quarters, disability-
benefit receipt, household applicability, and subsection 1641(c)'s petition
and agency determinations must be supplied separately. A generic imputed
battered-alien fact cannot replace those statutory predicates. Missing
inputs are never silently defaulted to false here.
"""

from pathlib import Path

from microcosm.frame.adapters.axiom import AxiomEngine, AxiomInputBindingContract
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_IMMIGRATION_AXIOM_BINDING_CONTRACT",
    "US_IMMIGRATION_AXIOM_MODULE",
    "us_immigration_axiom_engine",
]

US_IMMIGRATION_AXIOM_MODULE = "us/policies/usda/snap/obbb-alien-eligibility.yaml"

# Several authorities refer to the same underlying fact using different
# authored names. Binding those names to one column preserves the separate
# authority encodings without duplicating or reconciling population facts.
US_IMMIGRATION_AXIOM_BINDING_CONTRACT = AxiomInputBindingContract(
    version="us-immigration-snap-2025-12-09-v1",
    bindings={
        "benefit_evaluation_month_index": "@period.month_index",
        "individual_age": "age",
        "person_is_alien": "immigration_is_alien",
        "person_is_citizen_of_united_states": "immigration_is_us_citizen",
        "person_is_national_of_united_states": "immigration_is_us_national",
        "person_is_resident_of_united_states": "immigration_is_us_resident",
        "alien_lawfully_admitted_for_permanent_residence_under_immigration_and_nationality_act": "immigration_is_lawful_permanent_resident",
        "refugee_admitted_to_united_states_under_section_207_of_immigration_and_nationality_act": "immigration_was_admitted_as_refugee",
        "member_has_refugee_admission_or_status": "immigration_was_admitted_as_refugee",
        "alien_granted_asylum_under_section_208_of_immigration_and_nationality_act": "immigration_has_been_granted_asylum",
        "member_has_asylum_grant": "immigration_has_been_granted_asylum",
        "alien_paroled_into_united_states_under_section_212_d_5_of_immigration_and_nationality_act": "immigration_is_currently_paroled",
        "member_has_parole_under_ina_212_d_5": "immigration_is_currently_paroled",
        "alien_deportation_or_removal_withheld_under_listed_immigration_and_nationality_act_provisions": "immigration_has_deportation_or_removal_withheld",
        "member_has_deportation_or_removal_withheld": "immigration_has_deportation_or_removal_withheld",
        "alien_granted_conditional_entry_under_section_203_a_7_as_in_effect_prior_to_april_1_1980": "immigration_was_granted_conditional_entry",
        "member_has_conditional_entrant_status": "immigration_was_granted_conditional_entry",
        "alien_is_cuban_and_haitian_entrant_as_defined_in_refugee_education_assistance_act_section_501_e": "immigration_is_current_cuban_haitian_entrant",
        "individual_lawfully_resides_in_united_states_in_accordance_with_compact_of_free_association_referred_to_in_section_1612_b_2_G": "immigration_is_current_cofa_resident",
        "alien_admitted_to_united_states_as_amerasian_immigrant_described_in_section_1612_a_2_A_i_V": "immigration_is_amerasian_immigrant",
        "member_has_temporary_protected_status": "immigration_is_current_tps",
        "member_has_deferred_action_for_childhood_arrivals": "immigration_is_current_daca",
        "member_has_battered_alien_status": "immigration_is_battered_alien_or_qualifying_relative",
        "member_has_severe_trafficking_victim_or_derivative_status": "immigration_is_severe_trafficking_victim_or_qualifying_relative",
        "member_has_iraqi_or_afghan_special_immigrant_history": "immigration_has_iraqi_or_afghan_special_immigrant_history",
        "member_has_qualifying_afghan_humanitarian_parole_history": "immigration_has_qualifying_afghan_humanitarian_parole_history",
        "member_has_qualifying_ukrainian_humanitarian_parole_history": "immigration_has_qualifying_ukrainian_humanitarian_parole_history",
        "member_adjusted_to_lpr_from_cofa_status": "immigration_adjusted_to_lpr_from_cofa_status",
        "member_adjusted_to_lpr_from_conditional_entrant_status": "immigration_adjusted_to_lpr_from_conditional_entry_status",
        "member_adjusted_to_lpr_from_iraqi_or_afghan_special_immigrant_status": "immigration_adjusted_to_lpr_from_iraqi_afghan_special_immigrant_status",
        "member_adjusted_to_lpr_from_qualifying_afghan_parole": "immigration_adjusted_to_lpr_from_qualifying_afghan_parole",
        "member_adjusted_to_lpr_from_qualifying_ukrainian_parole": "immigration_adjusted_to_lpr_from_qualifying_ukrainian_parole",
        "member_adjusted_to_lpr_from_battered_alien_status": "immigration_adjusted_to_lpr_from_battered_alien_status",
        "member_adjusted_to_lpr_from_severe_trafficking_victim_status": "immigration_adjusted_to_lpr_from_severe_trafficking_victim_status",
        "member_adjusted_to_lpr_from_parole_for_at_least_one_year": "immigration_adjusted_to_lpr_from_parole_for_at_least_one_year",
        "parole_period_years": "immigration_parole_period_years",
        "refugee_admission_month_index": "immigration_refugee_admission_month_index",
        "asylum_grant_month_index": "immigration_asylum_grant_month_index",
        "deportation_or_removal_withholding_month_index": "immigration_withholding_start_month_index",
        "cuban_and_haitian_entrant_status_grant_month_index": "immigration_cuban_haitian_entrant_start_month_index",
        "amerasian_immigrant_admission_month_index": "immigration_amerasian_admission_month_index",
        "first_qualified_alien_status_start_month_index": "immigration_first_qualified_status_start_month_index",
        "first_qualified_alien_status_start_month_is_imputed": "immigration_first_qualified_status_start_month_is_imputed",
        # CPS/ACS observe none of these status-event months. The common flag
        # states that every emitted event month is imputed; Axiom conditions
        # each event-specific diagnostic on its corresponding status fact.
        "refugee_admission_month_is_imputed": "immigration_status_start_month_is_imputed",
        "asylum_grant_month_is_imputed": "immigration_status_start_month_is_imputed",
        "deportation_or_removal_withholding_month_is_imputed": "immigration_status_start_month_is_imputed",
        "cuban_and_haitian_entrant_status_grant_month_is_imputed": "immigration_status_start_month_is_imputed",
        "amerasian_immigrant_admission_month_is_imputed": "immigration_status_start_month_is_imputed",
    },
)


def us_immigration_axiom_engine(rulespec_us: str | Path) -> AxiomEngine:
    """Create the real Axiom adapter over a caller-selected RuleSpec checkout.

    Release callers must pin the RuleSpec artifact independently of this
    versioned binding contract. No unbound legal-input defaults or
    PolicyEngine enum reverse mappings are introduced by this constructor.
    """

    return AxiomEngine(
        Path(rulespec_us) / US_IMMIGRATION_AXIOM_MODULE,
        schema=US_SCHEMA,
        input_bindings=US_IMMIGRATION_AXIOM_BINDING_CONTRACT,
    )
