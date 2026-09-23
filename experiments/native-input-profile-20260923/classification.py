"""Reviewed classification of every input main's release requires.

One row per name in main's ``release_input_coverage_manifest.json`` (174 at
``4305a7d34``). Citations are ``(ref, path, token)`` triples that
``build_inventory.py`` resolves to ``path:line`` at the pinned commit and
refuses when the token is absent, so no line number here is hand-typed.

Primary classes (the lane brief's three, plus declared scope):

* ``native_successor_exists`` - a native graph owner produces or attaches the
  canonical name at the integration commit or a 2026-09-21 descendant.
  ``state`` says whether the current development request enables it, whether
  it is opt-in, partial, or descendant-only.
* ``port_needed`` - no native canonical owner; ``port_from`` names the main
  module(s) holding the maintained logic, ``native_source`` any native source
  projection that already reads the needed literals.
* ``source_absent`` - no locked source carries the observation; the evidence
  is cited. Main's reviewed-exclusion reasons that depend only on the pooled
  H5 files are *not* accepted as native source absence, because the native
  line reads the pinned Census members, whose headers do carry those fields
  (``source_header_observations.json``).
* ``declared_scope_exclusion`` - excluded by the native profile's declared
  scope (engine omissions or the prior-year family), not missing data.

These are source-code judgements. They are not actual-data missingness counts,
source-signal verification or scientific qualification of any value.
"""

from __future__ import annotations

INTEGRATION = "integration"  # 47960af43
MAIN = "main"  # 4305a7d34
SS_CORE = "native-ss-beneficiary-core-20260921"
DISABILITY = "native-other-disability-context-replay-20260921"
AGI_TAIL = "native-agi-tail-matching-20260921"
LANE = "lane"  # this branch's working tree


def native(path, token, ref=INTEGRATION):
    return {"ref": ref, "path": path, "token": token}


def main_ref(path, token):
    return {"ref": MAIN, "path": path, "token": token}


R = "packages/microcosm-build/src/microcosm/build/us_runtime/"
B = "packages/microcosm-build/src/microcosm/build/"
MANIFEST = B + "us/release_input_coverage_manifest.json"
REQUEST = (
    "~/PolicyEngine/_recovered/scratch-backup/893/current-native-composition-20260921/"
    "current_request.py (enrichment_arguments: groups=('unemployment','health_costs'), "
    "immigration_transfer=None, health_completion=False, demographic_inputs=False, "
    "race_hispanic_inputs=False, full_original_amount_donors=False; financial: "
    "person_status, household_roles, source_qualified_development_inputs=True)"
)

ROWS: list[dict] = []


def row(
    names, cls, state, *, native=(), port_from=(), native_source=(), evidence=(), note
):
    for name in names.split():
        ROWS.append(
            {
                "name": name,
                "class": cls,
                "state": state,
                "native": list(native),
                "port_from": list(port_from),
                "native_source": list(native_source),
                "evidence": list(evidence),
                "note": note,
            }
        )


# --- declared scope -------------------------------------------------------
row(
    "block_geoid tract_geoid",
    "declared_scope_exclusion",
    "engine_omission",
    native=[
        native(R + "input_coverage_profile.py", "NATIONAL_CD_ENGINE_OMISSIONS ="),
        native(
            R + "input_coverage_profile.py",
            'ASSIGNED_BLOCK_COLUMN = "census_block_geoid"',
        ),
    ],
    note="National/CD profiles omit block/tract engine inputs; build-owned census_block_geoid stays independent build state.",
)
row(
    "previous_year_income_available self_employment_income_last_year",
    "declared_scope_exclusion",
    "prior_year_family_excluded",
    native=[
        native(R + "input_coverage_profile.py", "def scope_excluded_us_inputs("),
        native(
            R + "native_survey_handoff.py",
            "excluded = name in US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS",
        ),
    ],
    note="The whole prior-year income family is outside native scope (Max's prior-wages removal); the handoff inventory labels it excluded_native_scope.",
)

# --- identity, weights, geography, demographics --------------------------
row(
    "age",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(R + "survey_observed_age.py", '"""'),
        native(R + "survey_population_preparation.py", '"age"'),
    ],
    note="Observed common source age (ASEC A_AGE, ACS AGEP); topcoded, no cohort advance.",
)
row(
    "household_weight",
    "native_successor_exists",
    "attached_by_default_request",
    native=[native(R + "native_survey_handoff.py", 'name == "household_weight"')],
    note="Typed household weight of the native population; calibration is separate.",
)
row(
    "is_household_head",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(
            R + "current_survey_household_roles.py",
            'CANONICAL_COLUMN = "is_household_head"',
        ),
        native(R + "graph_current_survey_household_roles.py", "roles.CANONICAL_COLUMN"),
    ],
    note="Household roles enabled in the current financial request.",
)
row(
    "state_fips",
    "native_successor_exists",
    "attached_by_default_request",
    native=[native(R + "graph_current_survey_state.py", 'OUTPUT = "state_fips"')],
    note="canonical_state_input=True in the current request; bound after enrichment with an explicit population version.",
)
row(
    "county_fips congressional_district_geoid",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(R + "atomic_block_support.py", '"congressional_district_geoid"'),
        native(R + "survey_atomic_geography.py", "def "),
    ],
    note="Atomic block geography ladder; source-signal qualification of the ladder is still an open release qualification.",
)
row(
    "is_female",
    "native_successor_exists",
    "opt_in_not_in_current_request",
    native=[
        native(R + "current_survey_sex_source.py", 'OUTPUT = "is_female"'),
        native(
            R + "graph_current_survey_sex.py", 'BIND_NODE = "survey_sex.bind_is_female"'
        ),
        native(R + "graph_us_survey_enrichment.py", "demographic_inputs=False"),
    ],
    note="Canonical binding on originals and clones exists (5cb3b54c3) behind demographic_inputs, which the current request leaves False; financial conditioning's private survey_predictor_is_female is not the canonical input.",
)
row(
    "cps_race is_hispanic",
    "native_successor_exists",
    "opt_in_not_in_current_request",
    native=[
        native(
            R + "current_survey_race_hispanic_source.py",
            'OUTPUTS = ("cps_race", "is_hispanic")',
        ),
        native(R + "graph_us_survey_enrichment.py", "race_hispanic_inputs=False"),
    ],
    note="b64a7c1d9 binds exact survey race/Hispanic origin on originals and clones behind race_hispanic_inputs (False in the current request). Unsupported ACS race categories and unknown allocation stay explicit.",
)
row(
    "immigration_status_str ssn_card_type",
    "native_successor_exists",
    "opt_in_not_in_current_request",
    native=[
        native(
            R + "current_survey_immigration_transfer.py",
            "OUTPUTS = rules.US_IMMIGRATION_OUTPUT_COLUMNS",
        ),
        native(
            R + "graph_current_survey_immigration.py",
            'ATTACH_NODE = "survey_immigration.attach"',
        ),
        native(R + "graph_us_survey_enrichment.py", "immigration_transfer=None,"),
        native(
            R + "graph_current_survey_immigration.py",
            '"national_stock_alignment_qualified": False',
        ),
    ],
    note="Immigration clone attachment is integrated (d33d71bee, a4ed33447, 15b7befff; all patch-equivalent to the 09-20 clone-successor branches). The host takes it through the immigration_transfer option, which the current request sets to None. The fragment records graph_fit_artifact_qualified=False and national_stock_alignment_qualified=False; enabling it is a request/qualification decision, not a missing port.",
)
row(
    "is_spm_independent_minor_role",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(
            R + "current_survey_spm_source.py",
            'values["_role"] = asec_roles.independent_minor_role(values)',
        ),
        native(R + "current_survey_spm_projection.py", "roles.to_frame(ROLE_INPUT)"),
        native(
            B + "spm_input_contract.py", 'ROLE_INPUT = "is_spm_independent_minor_role"'
        ),
    ],
    port_from=[
        main_ref(R + "spm_independence_role.py", "def with_us_spm_independence_role("),
        main_ref(R + "spm_role_source.py", "def independent_minor_role("),
    ],
    note="Main's #959 stage (merged, 4305a7d34) derives the role through spm_role_source.independent_minor_role, whose body is unchanged versus 47960af43 (the spm_role_source diff is zip-archive I/O only). The native SPM owner calls the same function on its own pinned member read and projects the nullable role; OUTSIDE units use the declared placeholder, UNRESOLVED stay unresolved. No port needed.",
)

# --- health ---------------------------------------------------------------
row(
    "has_esi has_indian_health_service_coverage_at_interview",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(
            R + "current_survey_health_coverage.py",
            'CoverageField("has_esi", "NOW_GRP", "HINS1")',
        )
    ],
    note="Basic source recoding is on even with health_completion=False.",
)
row(
    "has_champva_health_coverage_at_interview has_marketplace_health_coverage_at_interview "
    "has_medicaid_health_coverage_at_interview has_non_marketplace_direct_purchase_health_coverage_at_interview "
    "has_other_means_tested_health_coverage_at_interview has_tricare_health_coverage_at_interview "
    "has_va_health_coverage_at_interview",
    "native_successor_exists",
    "partial_acs_semantic_gap",
    native=[
        native(R + "current_survey_health_coverage.py", "FIELDS = ("),
        native(R + "current_survey_health_completion.py", '"""'),
        native(R + "graph_us_survey_enrichment.py", "health_completion=False"),
    ],
    note="ASEC originals are recoded from the pinned member (NOW_* read directly, so #720's H5 gap does not apply). The ACS categories are explicit semantic gaps; the opt-in completion transports 2025 ASEC coverage to 2024 ACS recipients with __known=False/__imputed and no temporal-equivalence claim.",
)

# --- default and opt-in amount groups ------------------------------------
row(
    "unemployment_compensation health_insurance_premiums_without_medicare_part_b other_medical_expenses over_the_counter_health_expenses",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(
            R + "current_survey_amounts.py",
            'AmountGroup("unemployment", (("UC_VAL", "unemployment_compensation"),)),',
        )
    ],
    note="Amount groups unemployment and health_costs are selected by the current request.",
)
row(
    "workers_compensation child_support_received veterans_benefits",
    "native_successor_exists",
    "opt_in_not_in_current_request",
    native=[
        native(
            R + "current_survey_amounts.py",
            'AmountGroup("workers_compensation", (("WC_VAL", "workers_compensation"),)),',
        ),
        native(
            R + "current_survey_amounts.py",
            'AmountGroup("child_support", (("CSP_VAL", "child_support_received"),)),',
        ),
        native(R + "current_asec_veterans_source.py", 'OUTPUT = "veterans_benefits"'),
    ],
    note="Opt-in amount groups (460addba2, 0042e38e3, 450316171) not selected by the current request; child support and veterans need full-original donors, which the request leaves off.",
)
row(
    "child_support_expense",
    "port_needed",
    "native_source_observation_only",
    native_source=[native(R + "current_asec_child_support_source.py", "CHSP_VAL")],
    port_from=[main_ref(R + "child_support.py", '"child_support_expense"')],
    note="The native child-support source reads CHSP_VAL, but paid support stays observed-only (obligation does not establish payment incidence); no canonical attachment or ACS completion.",
)
row(
    "disability_benefits",
    "native_successor_exists",
    "descendant_only_opt_in_partial",
    native=[
        native(
            R + "current_asec_other_disability_source.py",
            'OUTPUT = _ARCHIVED["output"]',
        ),
        native(
            R + "graph_current_survey_other_disability_completion.py",
            '"""',
            ref=DISABILITY,
        ),
        native(R + "graph_us_other_disability_host.py", '"""', ref=DISABILITY),
    ],
    note="The source adapter is at 47960af43 (fe4bf263c); completion and opt-in host continuation are only on 09-21 descendants (a5fc3e9f8, 162348849, aa2eaab11). Bounded invented acceptance only; under-15 remains unresolved (no manufactured zero).",
)

# --- Social Security ------------------------------------------------------
row(
    "social_security_retirement social_security_disability social_security_survivors social_security_dependents",
    "port_needed",
    "report_basis_only_no_canonical_attachment",
    native_source=[
        native(
            R + "current_social_security_source.py",
            '"individual_beneficiary_assignment_claim": False',
        ),
        native(R + "survey_social_security.py", "def asec_reason_basis("),
        native(R + "graph_current_survey_ss_completion.py", "No beneficiary columns."),
        native(R + "cps_carried_current.py", "def _social_security("),
        native(
            R + "survey_social_security_beneficiaries.py",
            'SINGLETON_CONVENTION = "development_asec_singleton_retirement_v1"',
            ref=SS_CORE,
        ),
    ],
    port_from=[main_ref(R + "cps_carried.py", "def _fill_social_security_leaves(")],
    note="The survey line has a report-grain basis and an unwired report-completion fragment, not beneficiary inputs; the enrichment host names no SS node. The ASEC-only five-node slice (cps_carried_current) reuses main's reason-priority/age-62 rule, and the 09-21 core adds a singleton-retirement convention. This lane adds survey_social_security_canonical.py: a pure canonical contract with explicit, recorded conventions and exact two-clone fan-out; host wiring remains.",
)

# --- hours, housing, tenure ----------------------------------------------
row(
    "weekly_hours_worked_before_lsr",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(
            R + "current_survey_hours.py", 'TARGET = "weekly_hours_worked_before_lsr"'
        )
    ],
    note="Source-qualified usual hours; source-signal qualification of usual hours is still listed as an outstanding release qualification.",
)
row(
    "receives_housing_assistance takes_up_housing_assistance_if_eligible",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(
            R + "current_survey_housing.py",
            'SPM_OUTPUTS = ("receives_housing_assistance", "takes_up_housing_assistance_if_eligible")',
        )
    ],
    note="SPM-unit housing receipt/take-up from source literals.",
)
row(
    "tenure_type spm_unit_tenure_type",
    "port_needed",
    "source_literal_read_not_bound",
    native_source=[
        native(
            R + "current_survey_housing.py",
            'tenure = _integer(raw["H_TENURE"], minimum=1, maximum=3)',
        )
    ],
    port_from=[
        main_ref(
            R + "housing_inputs.py",
            'US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS: tuple[str, ...] = ("tenure_type",)',
        ),
        main_ref(R + "acs_inputs.py", '"spm_unit_tenure_type",'),
    ],
    note="The housing owner reads H_TENURE (ASEC) and TEN (ACS) to classify ownership but emits no canonical tenure enum; a small binding is needed (fef2cb56a audit: source_projection_only).",
)

# --- financial / PUF ------------------------------------------------------
_FIN = (
    "employment_income_before_lsr self_employment_income_before_lsr taxable_interest_income "
    "tax_exempt_interest_income qualified_dividend_income non_qualified_dividend_income "
    "short_term_capital_gains long_term_capital_gains_before_response"
)
row(
    _FIN,
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(R + "current_survey_predictors.py", "OUTPUTS = ("),
        native(
            R + "puf55_survey_observed.py",
            "FINANCIAL_TARGETS = recipients.financial.values.OUTPUTS",
        ),
    ],
    note="Financial graph outputs on originals, fixed survey-arm inputs for the PUF55 arm; the 09-21 AGI-tail branch adds an invented tail-clone contract (not integrated in the development profile).",
)
row(
    "taxable_private_pension_income taxable_ira_distributions rental_income farm_operations_income",
    "native_successor_exists",
    "attached_by_default_request",
    native=[
        native(R + "puf55_survey_observed.py", "DEVELOPMENT_TARGETS = ("),
        native(
            R + "graph_current_asec_development_inputs.py",
            'ATTACH_NODE = "survey_development_inputs.attach"',
        ),
    ],
    note="Source-qualified development mappings on originals (c8652fbdc; source_qualified_development_inputs=True) plus PUF55 clone-one values; the pension/IRA mappings are the maintained cps_carried_current rules, labelled qualified values rather than observed taxable amounts.",
)
_PUF44 = (
    "alimony_expense alimony_income business_is_sstb casualty_loss charitable_cash_donations "
    "charitable_non_cash_donations domestic_production_ald educator_expense estate_income "
    "estate_income_would_be_qualified farm_income farm_operations_income_would_be_qualified "
    "farm_rent_income farm_rent_income_would_be_qualified health_savings_account_ald "
    "home_mortgage_interest investment_income_elected_form_4952 investment_interest_expense "
    "long_term_capital_gains_on_collectibles miscellaneous_income non_sch_d_capital_gains "
    "partnership_s_corp_income_would_be_qualified qualified_bdc_income qualified_reit_and_ptp_income "
    "qualified_tuition_expenses real_estate_taxes rental_income_would_be_qualified salt_refund_income "
    "self_employed_pension_contributions_desired self_employment_income_would_be_qualified "
    "sstb_self_employment_income_before_lsr sstb_self_employment_income_would_be_qualified "
    "sstb_unadjusted_basis_qualified_property sstb_w2_wages_from_qualified_business "
    "student_loan_interest traditional_ira_contributions_desired unadjusted_basis_qualified_property "
    "unrecaptured_section_1250_gain unreimbursed_business_employee_expenses w2_wages_from_qualified_business"
)
row(
    _PUF44,
    "native_successor_exists",
    "puf_clone_one_arm_zero_partial",
    native=[
        native(R + "full_puf_enrichment.py", "PUF55_SURVEY_SS_PERSON_OUTPUTS = tuple("),
        native(R + "puf55_original_placement.py", "SINGLETON_OUTPUTS = ("),
    ],
    note="PUF55 clone-one (PUF arm) outputs; the original (arm-zero) placement is conservative (3 unit + 5 singleton outputs on fully-known units, 35 person outputs deferred) and mixed-member finalization is the puf-finalization lane.",
)

# --- ports: education, ORG wages, SCF, SIPP, retirement, take-up ----------
row(
    "attends_eligible_educational_institution_for_american_opportunity_credit "
    "has_american_opportunity_credit_1098_t_or_exception has_american_opportunity_credit_institution_ein "
    "is_enrolled_at_least_half_time_for_american_opportunity_credit "
    "is_pursuing_credential_for_american_opportunity_credit",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "education_inputs.py", "def ")],
    note="AOC education inputs; main derives them in education_inputs (ASEC enrollment plus modeled attributes).",
)
row(
    "educational_assistance",
    "port_needed",
    "no_native_owner",
    port_from=[
        main_ref(R + "education_assistance_source.py", "ED_VAL"),
        main_ref(R + "education_inputs.py", '"educational_assistance"'),
    ],
    evidence=[
        "ED_VAL is a current-money field (asec_current_money.FIELDS) in the native money recipe, but no native canonical owner maps it."
    ],
    note="Source amount available natively; canonical attachment and ACS completion missing.",
)
row(
    "detailed_occupation_recode fsla_overtime_premium has_never_worked hourly_wage hours_worked_last_week "
    "is_computer_scientist is_executive_administrative_professional is_farmer_fisher is_military "
    "is_paid_hourly is_union_member_or_covered",
    "port_needed",
    "no_native_owner",
    port_from=[
        main_ref(R + "org_wages.py", "def "),
        main_ref(R + "hours_worked.py", '"hours_worked_last_week"'),
    ],
    evidence=[
        "Pinned ASEC person members carry A_DTOCC, A_HRLYWK, A_UNMEM, A_HRS1, PEMLR, A_CLSWKR (source_header_observations.json / header read this session); no native module reads them."
    ],
    note="Main's CPS-ORG wage/occupation donor stage (org_wages) and hours_worked; the native survey line has no successor.",
)
row(
    "bank_account_assets bond_assets stock_assets net_worth",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "scf_wealth.py", "def ")],
    note="SCF wealth donor stage; SSI countable-resource assets are required with no exclusion on main (#368).",
)
row(
    "auto_loan_balance auto_loan_interest qualified_passenger_vehicle_loan_interest",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "scf_auto_loans.py", "def ")],
    note="SCF auto-loan donor stage.",
)
row(
    "first_home_mortgage_balance first_home_mortgage_interest first_home_mortgage_origination_year",
    "port_needed",
    "reserved_for_downstream_scf_producer",
    native=[native(R + "full_puf_enrichment.py", "SCF_MORTGAGE_OUTPUTS = (")],
    port_from=[main_ref(R + "puf_support.py", '"first_home_mortgage_interest"')],
    note="The native PUF enrichment explicitly reserves these tax-unit fields for an independent downstream SCF producer that does not exist natively.",
)
row(
    "household_vehicles_owned household_vehicles_value",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "sipp_vehicles.py", "def ")],
    note="SIPP vehicle donor stage.",
)
row(
    "tip_income treasury_tipped_occupation_code",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "sipp_tips.py", "def ")],
    note="SIPP tips donor stage.",
)
row(
    "keogh_distributions tax_exempt_ira_distributions tax_exempt_private_pension_income "
    "taxable_401k_distributions taxable_403b_distributions taxable_sep_distributions",
    "port_needed",
    "native_source_observation_only",
    native_source=[
        native(
            R + "current_asec_retirement_detail_source.py",
            "def qualify_current_asec_retirement_detail(",
        ),
        native(R + "current_asec_retirement_basis.py", "DISTRIBUTION_SLOTS = ("),
        native(R + "cps_carried_current.py", '"tax_exempt_private_pension_income"'),
    ],
    port_from=[main_ref(R + "retirement_distributions.py", "def ")],
    note="Native retirement-detail source qualification and candidate basis exist without canonical distribution attachment; the ASEC-only five-node slice carries only the pension split.",
)
row(
    "roth_401k_contributions_desired roth_ira_contributions_desired traditional_401k_contributions_desired",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "retirement_contributions.py", "def ")],
    evidence=[
        "RETCB_VAL is a native current-money field but no native owner splits it into desired contributions."
    ],
    note="traditional_ira_contributions_desired and self_employed_pension_contributions_desired come from the PUF arm; these three need the maintained contribution stage.",
)
row(
    "takes_up_aca_if_eligible takes_up_eitc takes_up_head_start_if_eligible takes_up_medicaid_if_eligible "
    "takes_up_medicare_if_eligible takes_up_snap_if_eligible takes_up_ssi_if_eligible "
    "takes_up_tanf_if_eligible takes_up_wic_if_eligible would_file_taxes_voluntarily",
    "port_needed",
    "no_native_owner",
    port_from=[
        main_ref(R + "take_up.py", "def "),
        main_ref(R + "medicaid_take_up.py", "def "),
        main_ref(R + "medicare_take_up.py", "def "),
        main_ref(R + "snap_take_up.py", "def "),
        main_ref(R + "ssi_take_up.py", "def "),
        main_ref(R + "wic_claim.py", "def "),
        main_ref(R + "sipp_head_start.py", "def "),
        main_ref(R + "voluntary_filing.py", "def "),
    ],
    note="Take-up and voluntary-filing stages; ACA take-up is set in the release tool on main. None has a native successor.",
)
row(
    "is_blind is_disabled is_full_time_college_student meets_ssi_disability_criteria",
    "port_needed",
    "descriptive_status_not_canonical",
    native_source=[native(R + "current_survey_person_status.py", '"""')],
    port_from=[
        main_ref(R + "eligibility_inputs.py", "def "),
        main_ref(R + "ssi_disability_criteria.py", "def "),
    ],
    note="The native person-status owner reports survey_status_canonical_eligibility_assigned=False by design; six difficulty items and the student reference period do not establish these consumer-period statuses.",
)
row(
    "is_separated is_surviving_spouse own_children_in_household",
    "port_needed",
    "no_native_owner",
    native_source=[native(R + "current_survey_primary_family.py", "A_MARITL")],
    port_from=[
        main_ref(
            R + "relationship_inputs.py", 'result["is_separated"] = marital_status == 6'
        ),
        main_ref(R + "eligibility_inputs.py", "def _own_children_in_household("),
    ],
    note="Marital/relationship derivations; the native primary-family owner already reads A_MARITL, and roles/primary-family sources can supply own-children counts.",
)
row(
    "is_self_employed",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "other_health_insurance.py", '"is_self_employed"')],
    note="Main materializes it in the other-health-insurance stage (output_self_employed_flag in us/source_stages.json and us/spec/sources.yaml).",
)
row(
    "is_pregnant",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "pregnancy.py", "def ")],
    note="Pregnancy stage.",
)
row(
    "is_incapable_of_self_care pre_subsidy_care_expenses",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "adult_care.py", "def ")],
    note="Adult-care stage.",
)
row(
    "spm_unit_pre_subsidy_childcare_expenses spm_unit_energy_subsidy",
    "port_needed",
    "descriptive_spm_amount_only",
    native_source=[
        native(R + "current_survey_spm_amount_source.py", '"""'),
        native(R + "cps_carried_current.py", "CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES"),
    ],
    port_from=[
        main_ref(R + "childcare.py", "def "),
        main_ref(R + "energy_subsidy.py", "def "),
    ],
    note="Native SPM_CHILDCAREXPNS/SPM_ENGVAL handling is expressly descriptive at unit grain; ACS completion/applicability undefined. The ASEC-only five-node slice carries the childcare leaf only.",
)
row(
    "health_insurance_premiums other_health_insurance_premiums",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "other_health_insurance.py", "def ")],
    note="Other health insurance premiums stage.",
)
row(
    "pre_subsidy_rent",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "housing_inputs.py", '"pre_subsidy_rent"')],
    native_source=[native(R + "acs_inputs.py", '("GRNTP", "acs_monthly_gross_rent"),')],
    note="ACS GRNTP/RNTP are read as rent predictors by the maintained ACS reader at 47960af43, but no native canonical pre_subsidy_rent is emitted.",
)
row(
    "weeks_unemployed",
    "port_needed",
    "no_native_owner",
    port_from=[main_ref(R + "weeks_unemployed.py", "def ")],
    evidence=[
        "LKWEEKS is in the pinned members (header read this session); no native survey module reads it."
    ],
    note="Weeks-unemployed stage.",
)
row(
    "schedule_d_capital_gain_distributions",
    "port_needed",
    "no_native_owner",
    port_from=[
        main_ref(R + "multispine_pool.py", '"schedule_d_capital_gain_distributions"')
    ],
    note="#282 route leg (#462); PUF-derived on main.",
)
row(
    "selected_marketplace_plan_benchmark_ratio",
    "port_needed",
    "no_native_owner",
    port_from=[
        main_ref(
            "tools/build_us_fiscal_refresh_release.py",
            "selected_marketplace_plan_benchmark_ratio",
        )
    ],
    note="Set by main's release tool; the native release entry must own it or a reviewed successor.",
)
row(
    "receives_wic receives_snap receives_tanf",
    "port_needed",
    "source_literals_available_not_read",
    port_from=[
        main_ref(R + "cps_carried.py", "def reported_wic_receipt_carrier("),
        main_ref(R + "cps_carried.py", "def reported_snap_receipt_by_spm_unit("),
        main_ref(R + "cps_carried.py", "def reported_tanf_enrollment_by_spm_unit("),
    ],
    evidence=[
        "Pinned ASEC person members carry WICYN, SPM_SNAPSUB, SPM_WICVAL, PAW_TYP, PAW_YN (header read this session). The pinned 2024 ACS PUMS headers carry household FS (SNAP) and person PAP (all public assistance, not TANF-specific) and no WIC item (header read this session)."
    ],
    note="New on main (#978 option 1, ACS-transfer donor inputs). Native ASEC originals can take the maintained reported-receipt rules directly; ACS originals need an explicit completion or applicability decision.",
)

# --- main reviewed exclusions ----------------------------------------------
row(
    "is_wic_at_nutritional_risk takes_up_dc_ptc takes_up_early_head_start_if_eligible",
    "source_absent",
    "main_reviewed_exclusion_concept_absent",
    evidence=[main_ref(MANIFEST, '"is_wic_at_nutritional_risk"')],
    note="Main's exclusion reasons rest on the concept being absent from every locked source (no individual nutritional-risk assessment, no DC Schedule H claim indicator, no Early Head Start enrollment item), not on H5 pooling; they hold for the native Census members too.",
)
_H5_ONLY = (
    "Main excludes it because the three pooled 2022-2024 ASEC H5 files omit the field. "
    "The pinned Census person members the native line reads carry it (pppub23/24/25 "
    "headers read this session; source_header_observations.json), so the exclusion "
    "does not transfer. Port source: the archived policyengine-us-data derivation that "
    "main's exclusion cites (commit 42ed5d45c, datasets/cps/cps.py), plus an explicit "
    "ACS completion or applicability decision for ACS originals."
)
row(
    "survivor_benefits",
    "port_needed",
    "main_exclusion_h5_specific_source_in_native_member",
    native_source=[
        native(
            R + "current_asec_retirement_detail_source.py",
            '("SRVS_VAL", 6, 628, 49, "6C-28", 999999, "SUR_YN = 1"),',
        )
    ],
    port_from=[main_ref(MANIFEST, '"survivor_benefits": {')],
    evidence=[_H5_ONLY],
    note="The native retirement-detail owner already qualifies SRVS_VAL and SUR_VAL1/2 literals under SUR_YN; only the canonical person binding and ACS treatment are missing.",
)
row(
    "financial_assistance",
    "port_needed",
    "main_exclusion_h5_specific_source_in_native_member",
    port_from=[main_ref(MANIFEST, '"financial_assistance": {')],
    evidence=[_H5_ONLY],
    note="FIN_VAL/FIN_YN/I_FINVAL are in the member but not in the native current-money FIELDS registry; a port must extend that pinned registry or add a dedicated source owner.",
)
row(
    "employer_sponsored_insurance_premiums",
    "port_needed",
    "main_exclusion_h5_specific_source_in_native_member",
    native_source=[
        native(
            R + "current_survey_amounts.py",
            '("PHIP_VAL", "health_insurance_premiums_without_medicare_part_b"),',
        )
    ],
    port_from=[main_ref(MANIFEST, '"employer_sponsored_insurance_premiums": {')],
    evidence=[_H5_ONLY],
    note="NOW_OWNGRP/NOW_HIPAID/NOW_GRPFTYP are in the member; PHIP_VAL is already a native amount-group field. has_esi is not a substitute.",
)
row(
    "is_unmarried_partner_of_household_head",
    "port_needed",
    "main_exclusion_h5_specific_source_in_native_member",
    port_from=[main_ref(MANIFEST, '"is_unmarried_partner_of_household_head": {')],
    evidence=[_H5_ONLY],
    native_source=[
        native(
            R + "current_survey_primary_family.py",
            'ACS_PERSON_FIELDS = ("SERIALNO", "SPORDER", "RELSHIPP", "AGEP", "MAR")',
        )
    ],
    note="PERRP and PECOHAB are in the 2025 member the native current source reads (header read this session); the native primary-family owner already reads ACS RELSHIPP. Which codes identify an unmarried partner must be taken from the 2025 ASEC and 2024 PUMS dictionaries during the port; this inventory has not verified those code lists.",
)
