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

#: Reviews of every helper policyengine-us 2.60.4's formulas call.
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
        "policyengine_us.model_api.allocate_joint_amount_to_minimize_combined_tax": (
            HelperReview(
                digest=(
                    "8afa7ae71ce9806261cf1e4f698fa5fa591c9502ea5ea27083f6a098dda6b5eb"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.parameters.gov.hud.fmr.nearest_fmr_year": HelperReview(
            digest="f93ec524019d3517cfe1c86e88ca43e848e2495786aefdd0e61d0bc4e12e4618",
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.fmr.nearest_safmr_year": HelperReview(
            digest="8c42a8e4a82dd91b351c5aa869bcd0cb2298a8f73eba3694e88a361ac9d6a396",
            rationale=_READ_FREE,
        ),
        "policyengine_us.parameters.gov.hud.income_limits.lookup_income_limit": (
            HelperReview(
                digest=(
                    "96ddc1c7280222e74881df5e056b6da114cf0b2180c11e9bb19072e1321069da"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.parameters.gov.hud.income_limits.lookup_sized_income_limit": (
            HelperReview(
                digest=(
                    "4cbe425b3791e506c3768ac95e9afdbe67c3e3d6caec09349bf263cb998eaa24"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.parameters.gov.hud.payment_standards.nearest_payment_standard_year": (
            HelperReview(
                digest=(
                    "9c1c3f1cb0e6992700ec5b3eacdf3476506a94da174897d428ed7a39371ff461"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.parameters.gov.hud.utility_allowance.utility_allowance_schedule": (
            HelperReview(
                digest=(
                    "b6d0e07f37586325e0993737d940030f662e424770c3fa3c9db2874c0fbe9676"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.parameters.gov.states.fl.doe.sr.rates.reimbursement.get_reimbursement_rates": (
            HelperReview(
                digest=(
                    "e375122c4fb90ff4d90c9acad374f63bf747923fd8e1ac3af28b1eb2280ab62a"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.parameters.gov.states.tx.twc.ccs.payment.rates_expanded_age_groups.get_rates": (
            HelperReview(
                digest=(
                    "5da20dd4745dafcf5189b49bba20e946277e32bc26b5302712462d74fd5ed119"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.spm.clone_spm_system": HelperReview(
            digest="210e5ee39dd383801dd4841f53c6361f845aa231d85ccbac1e6c5dc6132d672c",
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
            digest="79baffc8cd475fcc2bd875b50c5e4b70369ae66353fe6793132ba612dd2fab6f",
            rationale=(
                "Scans which periods every supplied input covers, whatever "
                "the variable. No one variable's value is read, so it adds "
                "no receipt; the IRMAA lookback that calls it reads its "
                "income variables directly."
            ),
        ),
        "policyengine_us.tools.general.filer_share": HelperReview(
            digest="d4d54b9d564d108ba717e899a91a0dd62774350213941e1c8e7d8a6283477a34",
            reads=("is_tax_unit_dependent",),
            rationale="Splits passed amounts among each tax unit's non-dependents.",
        ),
        "policyengine_us.tools.general.get_itemization_branch": HelperReview(
            digest="6e3d1af70f756b56b4971d6b24623e8ee34932b0f8ed920dc0304036009f8c0c",
            rationale=(
                "Returns the branch that sets tax_unit_itemizes. The override "
                "is written, not read; the stored-value check on it only "
                "decides whether inherited values are dropped."
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
        "policyengine_us.tools.general.person_share_of_tax_unit_amount": (
            HelperReview(
                digest=(
                    "c6059e2f44902e4e2703187ce995729f6c597cb0590a2053d7b0aa8a05e0e19b"
                ),
                arguments=((2, "tax_unit_variable"), (3, "person_variable")),
                reads=("is_tax_unit_dependent", "is_tax_unit_head"),
                rationale=(
                    "Reads the named tax-unit amount and person amounts, and "
                    "attributes the amount to the head and spouse."
                ),
            )
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
        "policyengine_us.tools.geography.county_helpers.load_county_fips_dataset": (
            HelperReview(
                digest=(
                    "6aa0734d69ad2ae78207cb463166e4069942487061a70970784b28bff9c6ebb2"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.tools.geography.county_helpers.map_county_string_to_enum": (
            HelperReview(
                digest=(
                    "ac55be83f3c3e124e85c9383de959dd799c2854bd6d748d4444c35f7ea8e8401"
                ),
                rationale=_READ_FREE,
            )
        ),
        "policyengine_us.tools.period_branch.drop_inherited_values": HelperReview(
            digest="f0bea20da35ce50adfd17aa73945282ddbfa965515a419f3d9538a4e6a7fbc4d",
            rationale="Deletes a branch's calculated arrays; reads no value.",
        ),
        "policyengine_us.tools.period_branch.get_branch_for_period": HelperReview(
            digest="27733309daad8fa4aada3fa514c4b82d18a42045e6ba952391ad1a86f9ecb6ef",
            rationale=(
                "Returns a per-period branch with no overrides; the caller's "
                "reads on it are recorded at the caller."
            ),
        ),
        "policyengine_us.tools.period_branch.get_override_branch": HelperReview(
            digest="28b176988c91d1feee4394ea274253872f870857593c5740b34b893c4e0f28fc",
            rationale=(
                "Returns a branch with the passed inputs set. Overrides are "
                "written, not read; the stored-value check on them only "
                "decides whether inherited values are dropped."
            ),
        ),
        "policyengine_us.tools.pinned_tbs.get_2020_irc_tbs": HelperReview(
            digest="975bacc84e897855862ef6a584e712dd81127017c24805570130dcaad002777a",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.pinned_tbs.get_pre_arpa_eitc_tbs": HelperReview(
            digest="e9cc06584eb80e0ced9dfaccfc25ff7e3af5f5ceb5e807dc3a38a740d10b73d6",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.pinned_tbs.get_pre_tcja_ctc_tbs": HelperReview(
            digest="245d9cd291e2970a4cb112276ae387bf2360d561b437cd4da8cb84abb3245dde",
            rationale=_READ_FREE,
        ),
        "policyengine_us.tools.section_911.elects_section_911_exclusion": (
            HelperReview(
                digest=(
                    "607ded7ddab79c8612b2d1b74288bc3f301a6d89f5560c93af407dbda6f799fd"
                ),
                reads=_SECTION_911_ELECTION,
                rationale=(
                    "Tests the section 911 aggregates and every Form 2555 leaf "
                    "for a positive amount."
                ),
            )
        ),
        "policyengine_us.tools.section_911.has_section_911_leaf_inputs": (
            HelperReview(
                digest=(
                    "cd9018c420a718e3f4fed001a4611d66af4ade8aeac8cee169b2d108ff1db9f8"
                ),
                reads=_SECTION_911_LEAVES,
                rationale="Tests whether any Form 2555 leaf was supplied.",
            )
        ),
        "policyengine_us.tools.section_911.section_911_gross_exclusion": (
            HelperReview(
                digest=(
                    "6f203a5f296db8900707a89bac362c3603c4375228868bc83aed3283c7e2e43b"
                ),
                reads=(
                    "foreign_earned_income_exclusion_gross",
                    "foreign_earned_income_exclusion",
                    *_SECTION_911_LEAVES,
                ),
                rationale=(
                    "Reads the Form 2555 gross exclusion when any leaf was "
                    "supplied, else the legacy aggregate."
                ),
            )
        ),
        "policyengine_us.tools.state_eitc_helpers.calculate_eitc_demographic_eligibility": (
            HelperReview(
                digest=(
                    "dcd9e3a0ffcaca29a17296b372581d20d66fcde163baa47b41bfe257bb3484cf"
                ),
                reads=("age", "is_full_time_student"),
                reads_when_omitted=((3, "child_count", ("eitc_child_count",)),),
                rationale="Applies the EITC child-count and age tests.",
            )
        ),
        "policyengine_us.tools.state_eitc_helpers.calculate_eitc_like_amount": (
            HelperReview(
                digest=(
                    "387ae56d12c4b1e6ed465a2cd7728799516a6c3cd405a88c2ef2dac1607dad48"
                ),
                reads=(
                    "takes_up_eitc",
                    "filer_adjusted_earnings",
                    "adjusted_gross_income",
                    "tax_unit_is_joint",
                    "filing_status",
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
                    "and phase-out, the filing, filing-status, take-up and "
                    "section 911 tests."
                ),
            )
        ),
        "policyengine_us.tools.state_eitc_helpers.calculate_eitc_max_agi_limit": (
            HelperReview(
                digest=(
                    "dcc6efa060f8b7d0f61f1785225cceb7e425b1137bce1e8a518441c189c86ff1"
                ),
                reads=("tax_unit_is_joint",),
                rationale="Computes the EITC phase-out end, with the joint bonus.",
            )
        ),
        "policyengine_us.tools.state_eitc_helpers.eitc_filing_requirement_met": (
            HelperReview(
                digest=(
                    "ecb333c8100d731e0a1d3c60ca43ae8e14dfdf52a9a0590f50add285ae7d2d84"
                ),
                reads=_EITC_FILING,
                rationale="Mirrors the federal EITC filing condition.",
            )
        ),
        "policyengine_us.tools.state_eitc_helpers.eitc_filing_status_eligible": (
            HelperReview(
                digest=(
                    "aef122a0a7edd73ad01621297a04c7d81aeb6754b4568e95413961b5fa60f1fa"
                ),
                reads=("filing_status",),
                rationale="Applies the EITC separate-filer rule.",
            )
        ),
        "policyengine_us.tools.state_eitc_helpers.eitc_section_911_eligible": (
            HelperReview(
                digest=(
                    "aaed8c181d4a7f367ed3daee8a0040d039aced4d7655e331199c23a9257c19ff"
                ),
                reads=_SECTION_911_ELECTION,
                rationale="Negates the section 911 election test.",
            )
        ),
    }
)

__all__ = ["POLICYENGINE_US_HELPER_REVIEWS", "HelperReview"]
