"""Reviewed reads of the PolicyEngine-US helpers defined outside ``variables/``.

The static consumer index interprets formula modules only.  A formula that
calls a package helper (``tax_unit_non_dep_add`` in ``tools/general.py``, the
section 911 tests in ``tools/section_911.py``, the HUD loaders under
``parameters/``) reads whatever that helper reads.  Each helper the locked
engine's formulas call is reviewed here: which call arguments name variables
the helper reads, and which variables its body reads on its own.  The index
records both as ``helper_call`` receipts of the calling formula.

Each review pins a digest of the helper's code, of every package function it
uses and of the module constants they read (docstrings excluded).  A formula
calling a helper with no review, or one whose code has changed, stops the
index until the helper is reviewed again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class HelperReview:
    """What one PolicyEngine-US helper reads, reviewed at one digest."""

    #: SHA-256 from ``_helper_digest`` for the reviewed code.
    digest: str
    #: Call arguments naming variables the helper reads, as ``(position,
    #: keyword)``: a variable name or a list of them.
    arguments: tuple[tuple[int, str], ...] = ()
    #: Variables the helper's body reads whatever it is passed.
    reads: tuple[str, ...] = ()
    #: Variables read only when the argument at ``(position, keyword)`` is
    #: omitted or ``None``, as ``(position, keyword, variables)``.
    reads_when_omitted: tuple[tuple[int, str, tuple[str, ...]], ...] = ()
    #: Variables read for each exact value of a string argument, as
    #: ``(position, keyword, ((value, variables), ...))``: a helper that
    #: builds names from the argument (``f"social_security_{component}_..."``).
    #: A call passing a value not listed, or one the index cannot read as an
    #: exact string, stops the index.
    reads_by_argument: tuple[
        tuple[int, str, tuple[tuple[str, tuple[str, ...]], ...]], ...
    ] = ()
    #: What was checked, in a sentence or two.
    rationale: str = ""


# ``tools/section_911.py``'s SECTION_911_LEAF_INPUTS: the Form 2555 leaves.
_SECTION_911_LEAVES = (
    "foreign_earned_income_exclusion_amount",
    "foreign_housing_exclusion",
    "foreign_earned_income_exclusion_allocable_deductions",
    "foreign_housing_deduction",
    "foreign_earned_income_exclusion_disallowed_deductions",
)
_SECTION_911_ELECTION = (
    "foreign_earned_income_exclusion",
    "section_911_excluded_income",
    *_SECTION_911_LEAVES,
)
_EITC_FILING = (
    "tax_unit_is_required_to_file",
    "would_file_taxes_voluntarily",
    "would_file_if_eligible_for_refundable_credit",
)
_READ_FREE = (
    "Reads no variable: it works only on the arrays, parameters or "
    "simulation objects it is passed, or on package data files."
)

_EITC_SEPARATION = (
    "is_tax_unit_head_or_spouse",
    "is_separated",
    "is_separated_under_decree",
    "is_eitc_qualifying_child",
    "eitc_child_lived_with_taxpayer_more_than_half_year",
    "eitc_spouses_lived_apart_last_six_months",
    "eitc_legal_separation_and_apart_at_year_end",
)
# ``tools/mt_separate_return.py``: the net long-term capital gain and the
# tax unit's status for spouses filing separately on one return.
_MT_CAPITAL_GAINS = (
    "long_term_capital_gains",
    "short_term_capital_gains",
    "state_filing_status_if_married_filing_separately_on_same_return",
)
_MT_DEPENDENT_EXEMPTIONS = (
    "mt_dependent_exemptions_person",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
)

# The Social Security helpers build variable names from a component.
_SS_COMPONENTS = ("retirement", "disability", "survivors", "dependents")


def _ss(component: str, *fields: str) -> tuple[str, ...]:
    return tuple(f"social_security_{component}_{field}" for field in fields)


def _calculator(*fields: str) -> tuple[str, ...]:
    return tuple(f"ss_calculator_{field}" for field in fields)


# The ss_calculator inputs a SuppliedRecord and a SurvivorBasis take.
_CALCULATOR_RECORD = _calculator(
    "worker_id",
    "birth_date",
    "entitlement_month",
    "insured",
    "eligibility_year",
    "source_basis",
)
_CALCULATOR_SURVIVOR_BASIS = _calculator(
    "prior_disability_eligibility",
    "same_year_death_before_62",
    "worker_birth_year",
    "death_year",
    "basis_provenance",
    "worker_birth_date",
    "death_date",
    "authoritative_year",
    "ordinary_non_windex",
    "year_only_not_january_1",
)
# The further ss_calculator inputs the benefit calculator reads.
_CALCULATOR_PAYMENT = _calculator(
    "monthly_amount",
    "valuation_month",
    "composite_ratio",
    "ratio_description",
    "nonbinding_rib_lim",
    "nonbinding_family_maximum",
    "no_dual_entitlement",
    "no_earnings_test_at_valuation",
    "supplied_aime",
    "supplied_pia",
    "claiming_factor",
    "no_special_monthly_test",
    "no_prior_withholding_adjustment",
    "earnings_test_no_trigger",
    "earnings_history",
    "donut_variant",
    "earnings_test_settlement_month",
    "annual_earnings",
    "pre_fra_earnings",
    "withhold_cola",
    "immediate_earnings_test_payment",
)


def _calculator_basis_reads(component: str) -> tuple[str, ...]:
    """``means_test.calculator_component_basis``: calculated sources only."""

    if component not in ("retirement", "survivors"):
        return ()
    return (
        *_ss(component, "source"),
        "ss_calculator_source",
        *_CALCULATOR_RECORD,
        *_CALCULATOR_SURVIVOR_BASIS,
    )


def _means_test_reads(component: str) -> tuple[str, ...]:
    """``means_test.component_withholding`` and the dates it resolves."""

    return (
        *_ss(
            component,
            "cola_withheld",
            "cola_override",
            "worker_eligibility_year",
            "worker_id",
            "entitlement_year",
            "entitlement_month",
        ),
        "social_security_worker_computation_year",
        "age",
        "social_security_cola_means_test_withheld",
        *_calculator_basis_reads(component),
    )


def _joint_cola_reads(component: str) -> tuple[str, ...]:
    return (
        *_ss(
            component,
            "source",
            "first_cola_year",
            "first_cola_month",
            "entitlement_month",
            "worker_eligibility_year",
            "worker_id",
            "entitlement_year",
            "cola_withheld",
        ),
        "social_security_worker_computation_year",
        "age",
        *_means_test_reads(component),
    )


def _selected_benefit_reads(component: str) -> tuple[str, ...]:
    calculated = (
        ("ss_calculator_source", f"ss_calculated_{component}")
        if component in ("retirement", "survivors")
        else ()
    )
    return (*_ss(component, "source", "reported"), *calculated)


def _pia_ratio_reads(component: str) -> tuple[str, ...]:
    """``population.reported_pia_ratio_output`` through ``_record_basis``."""

    reads = [
        *_ss(
            component,
            "reported",
            "source",
            "worker_eligibility_year",
            "worker_id",
            "entitlement_year",
        ),
        "age",
        "birth_year",
        "social_security_worker_computation_year",
    ]
    if component == "retirement":
        for other in ("survivors", "dependents", "disability"):
            reads.extend(_ss(other, "reported"))
        reads.extend(
            (
                "social_security_retirement_claim_age_months",
                "social_security_birth_month",
                "social_security_birth_day",
                "ss_calculator_birth_date",
                *_ss(component, "entitlement_month"),
            )
        )
    if component in ("survivors", "dependents"):
        reads.extend(_ss(component, "worker_benefit_ratio", "worker_ratio_identified"))
        reads.append("social_security_retirement_worker_eligibility_year")
    if component == "survivors":
        reads.extend(_CALCULATOR_SURVIVOR_BASIS)
    return tuple(reads)


#: Reviews of every helper policyengine-us 2.145.3's formulas call.
POLICYENGINE_US_HELPER_REVIEWS: Mapping[str, HelperReview] = MappingProxyType(
    {
        "policyengine_us.model_api.all_of_variables": HelperReview(
            digest="a98ff1663acb94be709ec337409b7ef6568333f467ea4bb90f47a70e117c2079",
            arguments=((0, "variables"),),
            rationale=(
                "Builds a formula that adds each listed variable over the "
                "entity and requires every sum to be positive."
            ),
        ),
        "policyengine_us.model_api.allocate_joint_amount_to_minimize_combined_tax": HelperReview(
            digest=("8afa7ae71ce9806261cf1e4f698fa5fa591c9502ea5ea27083f6a098dda6b5eb"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.model_api.move_dependent_amounts_to_filer": HelperReview(
            digest="66c1eca4acd7471bdd5b56757b96ecd7f8749c88b06aa625a22e04487b74e8b3",
            reads=("is_tax_unit_dependent", "is_tax_unit_head", "is_tax_unit_spouse"),
            rationale=(
                "Moves the passed person amounts of tax unit dependents onto the "
                "head's or spouse's column, by the tax unit roles."
            ),
        ),
        "policyengine_us.parameters.gov.hud.fmr.nearest_fmr_year": HelperReview(
            digest="f93ec524019d3517cfe1c86e88ca43e848e2495786aefdd0e61d0bc4e12e4618",
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.fmr.nearest_safmr_year": HelperReview(
            digest="8c42a8e4a82dd91b351c5aa869bcd0cb2298a8f73eba3694e88a361ac9d6a396",
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.income_limits.lookup_income_limit": HelperReview(
            digest=("96ddc1c7280222e74881df5e056b6da114cf0b2180c11e9bb19072e1321069da"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.income_limits.lookup_sized_income_limit": HelperReview(
            digest=("4cbe425b3791e506c3768ac95e9afdbe67c3e3d6caec09349bf263cb998eaa24"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.payment_standards.nearest_payment_standard_year": HelperReview(
            digest=("9c1c3f1cb0e6992700ec5b3eacdf3476506a94da174897d428ed7a39371ff461"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.utility_allowance.utility_allowance_schedule": HelperReview(
            digest=("b6d0e07f37586325e0993737d940030f662e424770c3fa3c9db2874c0fbe9676"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.states.fl.doe.sr.rates.reimbursement.get_reimbursement_rates": HelperReview(
            digest=("e375122c4fb90ff4d90c9acad374f63bf747923fd8e1ac3af28b1eb2280ab62a"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.states.tx.twc.ccs.payment.rates_expanded_age_groups.get_rates": HelperReview(
            digest=("5da20dd4745dafcf5189b49bba20e946277e32bc26b5302712462d74fd5ed119"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.reforms.congress.s770.section_8.active_residual": HelperReview(
            digest="894b460a37181ec07cf0b5f0adfff905e829ae06479169ae39f72e2fea7eb4ba",
            rationale=_READ_FREE,
        ),
        "policyengine_us.reforms.congress.s770.section_8.allocate_bases": HelperReview(
            digest="8afbd009829379a97168a5bd18156a0ab8163aa3df02fe95f0f991bee7474557",
            rationale=_READ_FREE,
        ),
        "policyengine_us.reforms.congress.s770.section_8.enabled": HelperReview(
            digest="99f288e2328421dfbf635f618139f06813efeeea0690e4c02fdb6eb52c0c57f0",
            rationale=_READ_FREE,
        ),
        "policyengine_us.reforms.congress.s770.section_8.revenue_shares": HelperReview(
            digest="13a07c63cccd8505c958d1a455fc45c91320164b45b689f52451bcf6ca3685eb",
            rationale=_READ_FREE,
        ),
        "policyengine_us.spm.clone_spm_system": HelperReview(
            digest="6f533e496c631134d32d852effabf6beff8e8b7da150a042f2ebd884a5cd9607",
            rationale=_READ_FREE,
        ),
        "policyengine_us.spm.masked_policyengine_amount": HelperReview(
            digest="9ed181def9984ed10aaf40879f62cd5585e296ec0f203b528608948301bdefc8",
            reads=("spm_unit_tenure_type", "county_fips"),
            rationale=(
                "Reads the SPM unit's tenure and, when the provider is "
                "county-based, the household county, for the selected units. "
                "It also reads spm_measurement_adults and "
                "spm_measurement_children, which spm-calculator defines at run "
                "time outside this source tree, so no receipt can name them."
            ),
        ),
        "policyengine_us.tools.general.earliest_input_year": HelperReview(
            digest="20bf73d19ef9bc496303bb03386d16eb5d780df079f2e06125cfb39b1df6b918",
            rationale=(
                "Scans which periods every supplied input covers, whatever "
                "the variable, leaving out the Social Security calculator "
                "records and COLA overrides. No one variable's value is read, "
                "so it adds no receipt; the IRMAA lookback that calls it reads "
                "its income variables directly."
            ),
        ),
        "policyengine_us.tools.general.filer_share": HelperReview(
            digest="d4d54b9d564d108ba717e899a91a0dd62774350213941e1c8e7d8a6283477a34",
            reads=("is_tax_unit_dependent",),
            rationale="Splits passed amounts among each tax unit's non-dependents.",
        ),
        "policyengine_us.tools.general.get_itemization_branch": HelperReview(
            digest="eaa3820c76e1b027eda56d6598d4ca7eaf9398c9737f16d4665553a7616a58f3",
            reads=(
                "claim_of_right_section_1341_eligible",
                "claim_of_right_credit_applies",
            ),
            rationale=(
                "Returns the branch that sets tax_unit_itemizes, and also "
                "claim_of_right_credit_applies when any tax unit qualifies under "
                "section 1341, so it reads both. The overrides are written, not "
                "read; the stored-value check on them only decides whether "
                "inherited values are dropped."
            ),
        ),
        "policyengine_us.tools.general.get_previous_threshold": HelperReview(
            digest="fe2deb6c514f6c48a131536270b29d254ccc18231a992510168f2acffd630c38",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.general.has_input_for_period": HelperReview(
            digest="026021ea7dba312a0803b808a2e6946493c50cda63925dc2fa01277ec8a8b040",
            arguments=((1, "variable"),),
            rationale="Tests whether the named variable was supplied for the period.",
        ),
        "policyengine_us.tools.general.in_state": HelperReview(
            digest="6e4def773609523032dec0787f2744955486cae5674b32fc7edaddb82266bd5a",
            reads=("state_code_str",),
            rationale="Builds a formula comparing state_code_str with a state.",
        ),
        "policyengine_us.tools.general.person_non_dep_add": HelperReview(
            digest="524378701c59f49193e0e871feec0fac306b30a4e996c1f787e7402c24874d36",
            arguments=((2, "variables"),),
            reads=("is_tax_unit_dependent",),
            rationale=(
                "Adds each listed person variable for each person, leaving out "
                "dependents' own amounts. include_dependents only names listed "
                "variables kept for dependents, so it names no further read."
            ),
        ),
        "policyengine_us.tools.general.person_share_of_tax_unit_amount": HelperReview(
            digest="1154dacef5680704b9a883766421892d5866acbf369bb17afa0b67cf79b9f0e3",
            arguments=((2, "tax_unit_variable"), (3, "person_variable")),
            reads=(
                "is_tax_unit_dependent",
                "is_tax_unit_head",
                "is_tax_unit_head_or_spouse",
            ),
            rationale=(
                "Reads the named tax-unit amount and person amounts, and "
                "attributes the amount to the head and spouse; with "
                "split_between_spouses it reads is_tax_unit_head_or_spouse."
            ),
        ),
        "policyengine_us.tools.general.spouse": HelperReview(
            digest="fac41183a12c8e73a6ec7060ec38ff9c8031b4a4ca8d1fa690f4ed672a42436f",
            arguments=((2, "variable"),),
            rationale="Reads the named person variable for the marital-unit partner.",
        ),
        "policyengine_us.tools.general.sum_among_non_dependents": HelperReview(
            digest="d2f6ecaa8ca742a50fe93fe48b78d935fa43531526a9b13f9e01023357f4ebc3",
            arguments=((0, "variable"),),
            reads=("is_tax_unit_dependent",),
            rationale=(
                "Builds a formula summing the named person variable over "
                "non-dependents (tax_unit_non_dep_sum)."
            ),
        ),
        "policyengine_us.tools.general.sum_contained_tax_units": HelperReview(
            digest="6a2e6bd0ed3c76f01525c92571abd94655a8247589c46d08f179f7afe4eca722",
            arguments=((0, "var"),),
            reads=("is_tax_unit_head",),
            rationale=(
                "Sums the named tax-unit variable over the tax units whose "
                "heads the group contains."
            ),
        ),
        "policyengine_us.tools.general.tax_unit_non_dep_add": HelperReview(
            digest="d4c3a56e340a5442ac1864769979bf1f421cf8eb833f1a8caa35aba3e508a3ca",
            arguments=((2, "variables"),),
            reads=("is_tax_unit_dependent",),
            rationale=(
                "Adds each listed variable over the tax unit, leaving out "
                "dependents' person amounts. include_dependents only changes "
                "how a listed variable is summed, so it names no further read."
            ),
        ),
        "policyengine_us.tools.general.tax_unit_non_dep_sum": HelperReview(
            digest="69538fd4f1ceb072421c5dea5fbf0b595ac0e9ba6c5762cc42c9ab14ca3df9c3",
            arguments=((0, "var"),),
            reads=("is_tax_unit_dependent",),
            rationale="Sums the named person variable over non-dependents.",
        ),
        "policyengine_us.tools.geography.county_helpers.county_has_state_suffix": HelperReview(
            digest="85c7a06f4e739c20bb74f2d7a4cd8794b719ce8527f37f8c6097f0d00ff3fa6f",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.geography.county_helpers.load_county_fips_dataset": HelperReview(
            digest=("6aa0734d69ad2ae78207cb463166e4069942487061a70970784b28bff9c6ebb2"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.geography.county_helpers.map_county_string_to_enum": HelperReview(
            digest=("ac55be83f3c3e124e85c9383de959dd799c2854bd6d748d4444c35f7ea8e8401"),
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.local_sales_tax_geography.sales_tax_location": HelperReview(
            digest="ba5d9465d9774de9a6eb1f56902b2ab91794f5a28af2c8c625c64b19b96c278e",
            reads=(
                "state_code_str",
                "county_fips",
                "block_geoid",
                "county",
                "place_fips",
                "in_nyc",
                "in_yonkers",
            ),
            rationale=(
                "Locates each household for local sales tax from its state, county "
                "FIPS, block and place codes, and, where those leave the county or "
                "place unknown, from a supplied county, in_nyc or in_yonkers."
            ),
        ),
        "policyengine_us.tools.local_sales_tax_rates.locality_sales_tax_rates": HelperReview(
            digest="d208872dece196b533f66e34198b155a871aad07882b355e5e3ad2d3642de176",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.local_sales_tax_rates.state_general_sales_tax_rates": HelperReview(
            digest="b5649421e505852bb894d1c4c0e1201e92f6341a716f044de3c046b5fe4ef8fb",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.mt_separate_return.mt_allocate_dependent_exemptions": HelperReview(
            digest="af0cba2dc21ec08fc65c72e72625085f8a0e73c6e64f6a153ab69d3bc7b55b2b",
            reads=_MT_DEPENDENT_EXEMPTIONS,
            rationale=(
                "Allocates the dependent exemptions between the head's and "
                "spouse's columns of the passed income."
            ),
        ),
        "policyengine_us.tools.mt_separate_return.mt_capital_gains_tax": HelperReview(
            digest="dded334e4b8182cd92e53d6804f3a60eff8cdb45367245aae953357b42a89757",
            reads=_MT_CAPITAL_GAINS,
            rationale=(
                "Taxes each column's net long-term capital gain at the filing "
                "status's lower and higher capital gains rates."
            ),
        ),
        "policyengine_us.tools.mt_separate_return.mt_capital_gains_threshold": HelperReview(
            digest="b376784bc351e9828fc18389889c38a6943b936196c37e3d4bbe2f2c7337799d",
            reads=_MT_CAPITAL_GAINS,
            rationale=(
                "Finds the part of the lower capital gains band left after each "
                "column's ordinary income, by filing status."
            ),
        ),
        "policyengine_us.tools.mt_separate_return.mt_medical_deduction": HelperReview(
            digest="5542022186cd97199a320768c51e96b6271c8d0e23304b90dfff519bcc5bd3ac",
            reads=(
                "mt_agi_indiv",
                "mt_eligible_medical_insurance_premiums",
                "mt_eligible_long_term_care_insurance_premiums",
                "is_tax_unit_head_or_spouse",
            ),
            rationale=(
                "Floors the passed line 1a expenses at 7.5% of each column's AGI "
                "and adds the medical (and, 2021-2023, long-term care) premiums."
            ),
        ),
        "policyengine_us.tools.mt_separate_return.mt_medical_floor_room": HelperReview(
            digest="0eed71a61db099b67427e90cb36f85c983b1936a8789acc9c50c28a15c43219d",
            reads=(
                "is_tax_unit_head_or_spouse",
                "other_medical_expenses",
                "mt_eligible_medical_insurance_premiums",
                "mt_agi_indiv",
            ),
            rationale=(
                "Measures how far each column's medical expenses (and, from 2024, "
                "premiums) sit below its 7.5% AGI floor."
            ),
        ),
        "policyengine_us.tools.mt_separate_return.mt_regular_tax": HelperReview(
            digest="90b2f4390479963e7c017bb23b1ab43a577f2201709c8f677f719611767f5e2e",
            reads=_MT_CAPITAL_GAINS,
            rationale=(
                "Applies the filing status's rate schedule to the passed taxable "
                "income, less net long-term capital gains while those are taxed "
                "apart."
            ),
        ),
        "policyengine_us.tools.mt_separate_return.mt_separate_tax_before_refundable_credits": HelperReview(
            digest="3f009d58826ee48e49da90d27fa241bf41970f1a7ba1e07d5e727ae85992f8c7",
            reads=(
                "mt_agi_indiv",
                "mt_personal_exemptions_indiv",
                "mt_non_refundable_credits",
                "is_tax_unit_head_or_spouse",
                *_MT_DEPENDENT_EXEMPTIONS,
                *_MT_CAPITAL_GAINS,
            ),
            rationale=(
                "Computes the couple's combined separate-column tax for the passed "
                "deductions: AGI less exemptions, the dependent exemption split, "
                "the rate schedule, capital gains tax and non-refundable credits."
            ),
        ),
        "policyengine_us.tools.ny_ccap.ny_ccap_monthly_rate": HelperReview(
            digest="a30988e30ff823be0c97840d5b83bc8e15ad4ede41e2ad9edd2c7fcd36e4540c",
            reads=(
                "ny_ccap_duration_of_care",
                "childcare_hours_per_day",
                "childcare_days_per_week",
            ),
            rationale=(
                "Prices a month of care from the passed weekly, daily and part-day "
                "rates by the child's duration of care, hours and days."
            ),
        ),
        "policyengine_us.tools.period_branch.drop_inherited_values": HelperReview(
            digest="7e9d081792f54f60217b30fc862b4f4297b31a17b975553c8baa1ecf9f8480de",
            rationale="Deletes a branch's calculated arrays; reads no value.",
        ),
        "policyengine_us.tools.period_branch.drop_recomputed_values": HelperReview(
            digest="1c2eb13cfae72ed5f3c2a9965c3f94525a22cc24f349f7103f3b9138fade00cf",
            rationale=(
                "Deletes a branch's values of the formula variables whose names "
                "contain a passed part; reads no value."
            ),
        ),
        "policyengine_us.tools.period_branch.get_branch_for_period": HelperReview(
            digest="5cea518a7ce51e7e0c108b67c26c497b71bedaf079f123890196ea537bd392ad",
            rationale=(
                "Returns a per-period branch with no overrides; the caller's "
                "reads on it are recorded at the caller."
            ),
        ),
        "policyengine_us.tools.period_branch.get_override_branch": HelperReview(
            digest="63b5ee0e86b58117b63368507f8c534e335287500eedbcc3d9acba8bb34551fd",
            rationale=(
                "Returns a branch with the passed inputs set. Overrides are "
                "written, not read; the stored-value check on them only "
                "decides whether inherited values are dropped."
            ),
        ),
        "policyengine_us.tools.period_branch.has_input": HelperReview(
            digest="a71c921e175fa96284890198d0fbb5f42c1b55a5ae9234c501a66fa749e7a895",
            arguments=((1, "variable"),),
            rationale="Tests whether the named variable was supplied for the period.",
        ),
        "policyengine_us.tools.pinned_tbs.get_2020_irc_tbs": HelperReview(
            digest="79235eb5080ee25ed04085ebc63bd1c12cb240f0562f5c988ce7cb85b67cc6f4",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.pinned_tbs.get_pre_arpa_eitc_tbs": HelperReview(
            digest="05a6e52240c3acbb262fc88903a8d205564bb45ecb53eb61b44ccafd9befc04a",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.pinned_tbs.get_pre_tcja_ctc_tbs": HelperReview(
            digest="847d72a8c02a7bfb08fedf6fe8727bf9083cfcebd0a70d1597c5b34f2a1d7b58",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.section_911.elects_section_911_exclusion": HelperReview(
            digest=("607ded7ddab79c8612b2d1b74288bc3f301a6d89f5560c93af407dbda6f799fd"),
            reads=_SECTION_911_ELECTION,
            rationale=(
                "Tests the section 911 aggregates and every Form 2555 leaf "
                "for a positive amount."
            ),
        ),
        "policyengine_us.tools.section_911.excludes_amount_under_section_911": HelperReview(
            digest="c74893b970e058754c0dd5b920e89dfb5d1d567e6c86d9354f81c089055d468d",
            reads=_SECTION_911_ELECTION,
            rationale=(
                "Tests the Form 2555 exclusion leaves when any leaf was supplied, "
                "else the legacy section 911 aggregates, for a positive amount."
            ),
        ),
        "policyengine_us.tools.section_911.has_section_911_leaf_inputs": HelperReview(
            digest=("cd9018c420a718e3f4fed001a4611d66af4ade8aeac8cee169b2d108ff1db9f8"),
            reads=_SECTION_911_LEAVES,
            rationale="Tests whether any Form 2555 leaf was supplied.",
        ),
        "policyengine_us.tools.section_911.section_911_gross_exclusion": HelperReview(
            digest=("6f203a5f296db8900707a89bac362c3603c4375228868bc83aed3283c7e2e43b"),
            reads=(
                "foreign_earned_income_exclusion_gross",
                "foreign_earned_income_exclusion",
                *_SECTION_911_LEAVES,
            ),
            rationale=(
                "Reads the Form 2555 gross exclusion when any leaf was "
                "supplied, else the legacy aggregate."
            ),
        ),
        "policyengine_us.tools.social_security.earnings.highest_earnings": HelperReview(
            digest="d6dd276333c0bcbe3ce750a96dbc09ec634445915ad2223824c0acec764109d7",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.social_security.earnings.index_covered_earnings": HelperReview(
            digest="9a119ba9a58a6bf7b205ecd31abbdfdd77be56241b6f0df6474b5d864c91fa36",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.social_security.engine.calculator_output": HelperReview(
            digest="7070a8fb5ddafbddd2b5ab51c9a577dcca4a09d02b77449f2c20a0c0a343b19c",
            reads=(
                "ss_calculator_source",
                "ss_calculator_cola_override",
                *_CALCULATOR_RECORD,
                *_CALCULATOR_SURVIVOR_BASIS,
                *_CALCULATOR_PAYMENT,
                *_ss("retirement", "cola_withheld", "cola_override"),
                *_ss("survivors", "cola_withheld", "cola_override"),
                "social_security_cola_means_test_withheld",
            ),
            rationale=(
                "Runs the supplied-record benefit calculator for each person whose "
                "ss_calculator_source is set: it reads the ss_calculator record, "
                "survivor basis, earnings and COLA inputs, and, while the COLA "
                "means test applies, the retirement or survivors COLA override "
                "and withholding and the tax unit's means-test decision. The "
                "output argument only selects which result is returned."
            ),
        ),
        "policyengine_us.tools.social_security.means_test.component_withholding": HelperReview(
            digest="24041f9a9e03cd4b1f3d67e42c38c559d205f0fe493ff76d489afb1e5a535fc5",
            reads_by_argument=(
                (
                    3,
                    "component",
                    tuple(
                        (component, _means_test_reads(component))
                        for component in _SS_COMPONENTS
                    ),
                ),
            ),
            rationale=(
                "Decides whether one component's COLA is withheld: its legacy "
                "withheld input when the means test is off, else its override, "
                "eligibility and first-receipt dates (from the ss_calculator "
                "record for retirement and survivors) and the tax unit's "
                "means-test decision."
            ),
        ),
        "policyengine_us.tools.social_security.means_test.tax_unit_withholding": HelperReview(
            digest="17a4f12e75e9d447d144702cf15c6c4e0ff920036d1e50879f7233ad766d2ddd",
            reads=("social_security_cola_means_test_magi", "filing_status"),
            rationale=(
                "Compares the prior year's means-test MAGI with the indexed "
                "threshold for the prior year's filing status."
            ),
        ),
        "policyengine_us.tools.social_security.population.reported_pia_ratio_output": HelperReview(
            digest="e624811091b5e33f46f0778a7f4fa5ad0ba22bb8fb37d72e2c757eb6aa42b2df",
            reads_by_argument=(
                (
                    3,
                    "component",
                    tuple(
                        (component, _pia_ratio_reads(component))
                        for component in _SS_COMPONENTS
                    ),
                ),
            ),
            rationale=(
                "Infers a component's PIA ratio and diagnostics from its reported "
                "amount, source, worker links and entitlement dates, with age and "
                "birth year; retirement also reads the other components' amounts, "
                "claim age and birth date, survivors and dependents the worker "
                "benefit ratio, and survivors the ss_calculator survivor basis."
            ),
        ),
        "policyengine_us.tools.social_security_benefit_levers.reported_joint_cola_factor": HelperReview(
            digest="708092d4e7a3f6c153300de84b3f8250f0d3f48efa05e10ac2a01682650dff32",
            reads_by_argument=(
                (
                    3,
                    "component",
                    tuple(
                        (component, _joint_cola_reads(component))
                        for component in _SS_COMPONENTS
                    ),
                ),
            ),
            rationale=(
                "Builds a reported component's COLA factor from its source, first "
                "COLA and entitlement dates, worker links and age, its withheld "
                "COLAs, and, once the means test has applied, the means-test "
                "withholding inputs."
            ),
        ),
        "policyengine_us.tools.social_security_benefit_levers.selected_reported_benefit": HelperReview(
            digest="f8b073a574f0977c35adb6d2a12e4375defbd0b537e3aea920664888dc21786b",
            reads_by_argument=(
                (
                    2,
                    "component",
                    tuple(
                        (component, _selected_benefit_reads(component))
                        for component in _SS_COMPONENTS
                    ),
                ),
            ),
            rationale=(
                "Returns a component's reported amount, or for a calculated "
                "retirement or survivors source the calculator's amount."
            ),
        ),
        "policyengine_us.tools.state_eitc_helpers.calculate_eitc_demographic_eligibility": HelperReview(
            digest="a48a167c24a380f51aefd2ebab692da805b4545389697a00b623bb29cf96b3dc",
            reads=("age", "is_full_time_student", "is_tax_unit_dependent"),
            reads_when_omitted=((3, "child_count", ("eitc_child_count",)),),
            rationale=(
                "Applies the EITC child-count test and the childless age test to "
                "the filer and spouse."
            ),
        ),
        "policyengine_us.tools.state_eitc_helpers.calculate_eitc_like_amount": HelperReview(
            digest="389ce1c330e65b5db60c83ff0514143dd74a7ca857df3a769775688a4ea07834",
            reads=(
                "takes_up_eitc",
                "eitc_earned_income",
                "adjusted_gross_income",
                "tax_unit_is_joint",
                "filing_status",
                *_EITC_SEPARATION,
                *_EITC_FILING,
                *_SECTION_911_ELECTION,
            ),
            reads_when_omitted=(
                (
                    8,
                    "investment_income_eligible",
                    ("eitc_relevant_investment_income",),
                ),
            ),
            rationale=(
                "Computes a federal-style EITC: earnings and AGI phase-in "
                "and phase-out, the filing, filing-status (with the separated "
                "spouse exception), take-up and section 911 tests."
            ),
        ),
        "policyengine_us.tools.state_eitc_helpers.calculate_eitc_max_agi_limit": HelperReview(
            digest=("dcc6efa060f8b7d0f61f1785225cceb7e425b1137bce1e8a518441c189c86ff1"),
            reads=("tax_unit_is_joint",),
            rationale="Computes the EITC phase-out end, with the joint bonus.",
        ),
        "policyengine_us.tools.state_eitc_helpers.eitc_filing_requirement_met": HelperReview(
            digest=("ecb333c8100d731e0a1d3c60ca43ae8e14dfdf52a9a0590f50add285ae7d2d84"),
            reads=_EITC_FILING,
            rationale="Mirrors the federal EITC filing condition.",
        ),
        "policyengine_us.tools.state_eitc_helpers.eitc_filing_status_eligible": HelperReview(
            digest="490028b6333667c4c54581167ac6ee8b9c0bad105276c0fde88e477c5e37a191",
            reads=("filing_status", *_EITC_SEPARATION),
            rationale=(
                "Applies the EITC separate-filer rule and, where the switch allows "
                "it, the separated spouse exception."
            ),
        ),
        "policyengine_us.tools.state_eitc_helpers.eitc_section_911_eligible": HelperReview(
            digest=("aaed8c181d4a7f367ed3daee8a0040d039aced4d7655e331199c23a9257c19ff"),
            reads=_SECTION_911_ELECTION,
            rationale="Negates the section 911 election test.",
        ),
        "policyengine_us.tools.tob.income_tax_with_social_security_override": HelperReview(
            digest="8cdd90a8b28701875815425f9b1753d74319f4bbd0cdf7da0139cb62229f76c4",
            reads=("income_tax_chapter_1",),
            rationale=(
                "Reads chapter 1 income tax on a short-lived branch where taxable "
                "Social Security is neutralized or replaced by the passed array, "
                "which it writes, not reads."
            ),
        ),
    }
)

__all__ = ["POLICYENGINE_US_HELPER_REVIEWS", "HelperReview"]
