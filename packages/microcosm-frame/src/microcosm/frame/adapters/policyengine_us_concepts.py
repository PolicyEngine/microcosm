"""The policyengine-us concept mapping: which engine inputs each concept feeds.

Kept apart from the adapter module so that editing evidence notes never
changes the adapter's source hash, which the graph's simulate kernel folds
into its implementation hash. :class:`~microcosm.frame.adapters.policyengine_us.
PolicyEngineUSEngine` returns :data:`POLICYENGINE_US_CONCEPT_MAPPING` from
``concept_mapping()``.
"""

from microcosm.frame.concept_mapping import (
    AllocateToReferencePerson,
    ConceptMapping,
    CoresidentChildCount,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    Positive,
    Predicate,
    Recode,
    RelationshipRole,
    Role,
    Share,
    Sum,
    TakeUpThreshold,
    bind,
)
from microcosm.frame.concepts import AlignmentRelation

__all__ = ["POLICYENGINE_US_CONCEPT_MAPPING"]

# ---------------------------------------------------------------------------
# Concept mapping
# ---------------------------------------------------------------------------
#
# Which PolicyEngine-US inputs each engine-neutral concept feeds
# (microcosm.frame.concepts), reviewed against policyengine-us 2.2.1. Every
# target is a pure input (in ``PolicyEngineUSVariableMetadataIndex.variables``),
# never a formula-owned aggregate: totals such as ``interest_income`` or
# ``employment_income`` are fed through their input leaves. Notes quote the
# engine's own label or documentation, or say how Microcosm's US builds
# populate the input today.


_US_PERSON_TAKE_UP = (
    "basic_health_program",
    "chip",
    "early_head_start",
    "head_start",
    "medicaid",
    "medicare",
    "ssi",
    "wic",
)
_US_TAX_UNIT_TAKE_UP = (
    "aca",
    "ca_premium_subsidy",
    "co_premium_assistance",
    "nm_premium_assistance",
)
_US_SPM_UNIT_TAKE_UP = ("housing_assistance", "snap", "tanf")


def _us_take_up(program: str, entity: str, engine_input: str) -> InputBinding:
    person = entity == "person"
    return bind(
        engine_input,
        entity,
        "fact:person.take_up_seed",
        TakeUpThreshold(program=f"us.{program}"),
        AlignmentRelation.APPROXIMATE,
        (
            "A data-seeded take-up flag (engine default True): true when the "
            "program's draw from the persistent seed falls below the rate "
            "aligned to its caseload."
            + ("" if person else " The unit takes its reference member's draw.")
            + (
                " The engine reads it monthly; the flag is the same each month."
                if program == "wic"
                else ""
            )
        ),
        group_rule=None if person else GroupRule.REFERENCE_MEMBER,
    )


POLICYENGINE_US_CONCEPT_MAPPING = ConceptMapping(
    engine="policyengine-us",
    engine_version="2.2.1",
    entity_correspondence={"person": "person", "household": "household"},
    input_declaration=InputDeclaration.ENGINE_TYPED,
    bindings=(
        # --- Demography and relationships ---------------------------------
        bind(
            "age",
            "person",
            "fact:person.age",
            Identity(),
            AlignmentRelation.EXACT,
            "Engine label 'age' (float years); builds copy CPS A_AGE.",
        ),
        bind(
            "is_female",
            "person",
            "fact:person.sex",
            Recode(pairs=(("female", True), ("male", False))),
            AlignmentRelation.EXACT,
            "Engine label 'Is female'; is_male is formula-owned as its negation.",
        ),
        bind(
            "is_separated",
            "person",
            "fact:person.legal_marital_status",
            Predicate(clauses=(("fact:person.legal_marital_status", ("separated",)),)),
            AlignmentRelation.EXACT,
            (
                "Engine label 'Separated', documentation 'Whether the person is "
                "separated from a partner.', referencing 26 USC 7703; consumers "
                "read it as a married person living apart (filing_status maps it "
                "to SEPARATE). Builds set it from CPS A_MARITL == 6 "
                "(relationship_inputs.py)."
            ),
        ),
        bind(
            "is_surviving_spouse",
            "person",
            "fact:person.legal_marital_status",
            Predicate(clauses=(("fact:person.legal_marital_status", ("widowed",)),)),
            AlignmentRelation.EXACT,
            (
                "Engine label 'surviving spouse', documentation 'Whether the "
                "person is surviving spouse.': a person-level widowhood input. "
                "The IRC 2(a) filing status is the separate formula "
                "surviving_spouse_eligible, which adds tax-unit headship, child "
                "dependents and not being married. Builds set it from CPS "
                "A_MARITL == 4 (relationship_inputs.py)."
            ),
        ),
        bind(
            "is_household_head",
            "person",
            "fact:household.reference_person_id",
            RelationshipRole(role=Role.REFERENCE_PERSON),
            AlignmentRelation.EXACT,
            (
                "Engine label 'is head of this household'; builds set it from "
                "CPS P_SEQ == 1, the reference person's line."
            ),
        ),
        bind(
            "is_household_spouse",
            "person",
            ("fact:household.reference_person_id", "fact:person.partner_person_id"),
            RelationshipRole(role=Role.REFERENCE_PERSON_PARTNER),
            AlignmentRelation.APPROXIMATE,
            (
                "spm-calculator label 'Explicit household spouse role'; it does "
                "not say whether an unmarried partner counts, while the concept "
                "includes one."
            ),
        ),
        bind(
            "is_unmarried_partner_of_household_head",
            "person",
            (
                "fact:household.reference_person_id",
                "fact:person.partner_person_id",
                "fact:person.legal_marital_status",
            ),
            RelationshipRole(role=Role.UNMARRIED_PARTNER_OF_REFERENCE_PERSON),
            AlignmentRelation.EXACT,
            "Engine label 'is unmarried partner of household head'.",
        ),
        bind(
            "own_children_in_household",
            "person",
            ("fact:person.parent_1_person_id", "fact:person.parent_2_person_id"),
            CoresidentChildCount(),
            AlignmentRelation.APPROXIMATE,
            (
                'Engine label "Count of one\'s own children in the household" '
                "gives no age or marital limit; the concept counts every "
                "co-resident person naming this person as a parent."
            ),
        ),
        # --- Labour income ------------------------------------------------
        bind(
            "employment_income_before_lsr",
            "person",
            "fact:person.employment_income",
            Identity(),
            AlignmentRelation.EXACT,
            (
                "Engine label 'employment income before labor supply "
                "responses'; employment_income is formula-owned from it. "
                "Builds copy CPS WSAL_VAL."
            ),
        ),
        bind(
            "self_employment_income_before_lsr",
            "person",
            "fact:person.nonfarm_self_employment_income",
            Share(parameter="us.non_sstb_self_employment_share"),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine label 'self-employment income before labor supply "
                "responses': the non-farm income of businesses that are not "
                "specified service trades (SSTB, 26 USC 199A(d)(2)); "
                "farm_operations_income is documented as excluded from it. "
                "Builds copy CPS SEMP_VAL (cps_carried.py)."
            ),
        ),
        bind(
            "sstb_self_employment_income_before_lsr",
            "person",
            "fact:person.nonfarm_self_employment_income",
            Share(parameter="us.non_sstb_self_employment_share", complement=True),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'SSTB self-employment non-farm income "
                "before labor supply responses.' The complement leaf."
            ),
        ),
        bind(
            "farm_operations_income",
            "person",
            "fact:person.farm_self_employment_income",
            Identity(),
            AlignmentRelation.EXACT,
            (
                "Engine documentation: 'Income from active farming operations. "
                "Schedule F. Do not include this income in self-employment "
                "income.' Builds copy CPS FRSE_VAL."
            ),
        ),
        # --- Capital income -----------------------------------------------
        bind(
            "taxable_interest_income",
            "person",
            "fact:person.interest_income",
            Share(parameter="us.taxable_interest_share"),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine label 'taxable interest income': the taxable part of "
                "all interest; interest_income is formula-owned as the sum of "
                "the taxable and tax-exempt leaves."
            ),
        ),
        bind(
            "tax_exempt_interest_income",
            "person",
            "fact:person.interest_income",
            Share(parameter="us.taxable_interest_share", complement=True),
            AlignmentRelation.NARROW_MATCH,
            "Engine label 'tax-exempt interest income': the complement leaf.",
        ),
        bind(
            "qualified_dividend_income",
            "person",
            "fact:person.dividend_income",
            Share(parameter="us.qualified_dividend_share"),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine label 'qualified dividend income'; dividend_income and "
                "ordinary_dividend_income are formula-owned from the two "
                "leaves."
            ),
        ),
        bind(
            "non_qualified_dividend_income",
            "person",
            "fact:person.dividend_income",
            Share(parameter="us.qualified_dividend_share", complement=True),
            AlignmentRelation.NARROW_MATCH,
            "Engine label 'non-qualified dividend income': the complement leaf.",
        ),
        bind(
            "rental_income",
            "person",
            "fact:person.rental_income",
            Identity(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation 'Income from rental of property' does not "
                "say whether subletting part of the own dwelling counts; the "
                "concept excludes it. Builds copy CPS RNT_VAL."
            ),
        ),
        bind(
            "long_term_capital_gains_before_response",
            "person",
            "fact:person.realized_capital_gains",
            Share(parameter="us.long_term_capital_gains_share"),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine label 'capital gains before responses', the long-term "
                "leaf; long_term_capital_gains adds the behavioral response."
            ),
        ),
        bind(
            "short_term_capital_gains",
            "person",
            "fact:person.realized_capital_gains",
            Share(parameter="us.long_term_capital_gains_share", complement=True),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'Net gains made from sales of assets held "
                "for one year or less (losses are expressed as negative gains).'"
            ),
        ),
        # --- Pensions -----------------------------------------------------
        bind(
            "taxable_private_pension_income",
            "person",
            "fact:person.private_pension_income",
            Share(parameter="us.taxable_private_pension_share"),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'Taxable income from non-government "
                "employee pensions.' The concept also covers government "
                "employee and personal pensions, which this mapping places on "
                "the private leaves (the public-pension leaves stay "
                "uncovered)."
            ),
        ),
        bind(
            "tax_exempt_private_pension_income",
            "person",
            "fact:person.private_pension_income",
            Share(parameter="us.taxable_private_pension_share", complement=True),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'Tax-exempt income from non-government "
                "employee pensions.'"
            ),
        ),
        bind(
            "social_security_retirement",
            "person",
            "fact:person.public_pension_income",
            Identity(),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine label 'Social Security retirement benefits', one of the "
                "four leaves social_security adds. Benefits paid on a living "
                "spouse's record (dependents leaf) and a deceased spouse's "
                "record (survivors leaf) stay uncovered. Builds split CPS SS_VAL "
                "across the leaves by RESNSS1/2 (cps_carried.py)."
            ),
        ),
        # --- Work intensity -----------------------------------------------
        bind(
            "weekly_hours_worked_before_lsr",
            "person",
            "fact:person.usual_weekly_hours",
            Identity(),
            AlignmentRelation.EXACT,
            (
                "Engine label 'average weekly hours worked (before labor supply "
                "responses)', documentation 'Usual weekly hours worked, before "
                "labor supply responses. Datasets populate this from survey "
                "data; ...'. Builds copy CPS HRSWK, usual hours (hours_worked.py)."
            ),
        ),
        bind(
            "worked_last_year",
            "person",
            "fact:person.weeks_worked",
            Positive(),
            AlignmentRelation.EXACT,
            (
                "Engine label 'worked at any time in the previous year', "
                "documentation 'the WKSWORK variable in the Current Population "
                "Survey exceeding zero'. WKSWORK counts weeks worked in the "
                "survey's income reference year, the concept's reference year."
            ),
        ),
        # --- Disability and education -------------------------------------
        bind(
            "is_disabled",
            "person",
            "fact:person.has_disability",
            Identity(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Is disabled' carries no definition. Builds set it "
                "when any of the six CPS ASEC functional-difficulty items "
                "(PEDIS*) is 1, the functional measure this concept names, and "
                "also for reported SSI recipients under 65, a program criterion "
                "(eligibility_inputs.py)."
            ),
        ),
        bind(
            "is_full_time_college_student",
            "person",
            ("fact:person.education_enrollment", "fact:person.enrolled_full_time"),
            Predicate(
                clauses=(
                    ("fact:person.education_enrollment", ("tertiary",)),
                    ("fact:person.enrolled_full_time", (True,)),
                )
            ),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Is a full time college student'; college "
                "enrollment is read as ISCED 5-8 enrollment."
            ),
        ),
        bind(
            "is_part_time_college_student",
            "person",
            ("fact:person.education_enrollment", "fact:person.enrolled_full_time"),
            Predicate(
                clauses=(
                    ("fact:person.education_enrollment", ("tertiary",)),
                    ("fact:person.enrolled_full_time", (False,)),
                )
            ),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'Enrolled at least half-time but less "
                "than full-time in an institution of higher education', "
                "narrower than any part-time tertiary enrollment."
            ),
        ),
        bind(
            "is_in_secondary_school",
            "person",
            "fact:person.education_enrollment",
            Predicate(
                clauses=(
                    (
                        "fact:person.education_enrollment",
                        ("lower_secondary", "upper_secondary"),
                    ),
                )
            ),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Is in secondary school (or in an equivalent "
                "level of training)'; the US grade span of secondary school "
                "differs from ISCED levels 2-3 at the margins."
            ),
        ),
        bind(
            "has_completed_first_four_years_of_postsecondary_education",
            "person",
            "fact:person.educational_attainment",
            Predicate(
                clauses=(
                    (
                        "fact:person.educational_attainment",
                        (
                            "bachelor_or_equivalent",
                            "master_or_equivalent",
                            "doctoral_or_equivalent",
                        ),
                    ),
                )
            ),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'Whether the student completed the first "
                "four years of postsecondary education before the beginning of "
                "the tax year' (26 USC 25A). Read as ISCED 6-8 attainment; four "
                "years of study without a degree is missed."
            ),
        ),
        # --- Take-up ------------------------------------------------------
        *(
            _us_take_up(program, "person", f"takes_up_{program}_if_eligible")
            for program in _US_PERSON_TAKE_UP
        ),
        *(
            _us_take_up(program, "tax_unit", f"takes_up_{program}_if_eligible")
            for program in _US_TAX_UNIT_TAKE_UP
        ),
        _us_take_up("dc_ptc", "tax_unit", "takes_up_dc_ptc"),
        _us_take_up("eitc", "tax_unit", "takes_up_eitc"),
        *(
            _us_take_up(program, "spm_unit", f"takes_up_{program}_if_eligible")
            for program in _US_SPM_UNIT_TAKE_UP
        ),
        # --- Housing ------------------------------------------------------
        bind(
            "tenure_type",
            "household",
            "fact:household.tenure",
            Recode(
                pairs=(
                    ("owned_outright", "OWNED_OUTRIGHT"),
                    ("owned_with_mortgage", "OWNED_WITH_MORTGAGE"),
                    ("rented_public_authority", "RENTED"),
                    ("rented_nonprofit_social", "RENTED"),
                    ("rented_private", "RENTED"),
                    ("rent_free", "NONE"),
                )
            ),
            AlignmentRelation.BROAD_MATCH,
            (
                "Engine enum TenureType has one RENTED category for all three "
                "rented categories; NONE takes occupiers without rent, as the "
                "ACS TEN = 4 build map does."
            ),
        ),
        bind(
            "spm_unit_tenure_type",
            "spm_unit",
            "fact:household.tenure",
            Recode(
                pairs=(
                    ("owned_outright", "OWNER_WITHOUT_MORTGAGE"),
                    ("owned_with_mortgage", "OWNER_WITH_MORTGAGE"),
                    ("rented_public_authority", "RENTER"),
                    ("rented_nonprofit_social", "RENTER"),
                    ("rented_private", "RENTER"),
                    ("rent_free", "RENTER"),
                )
            ),
            AlignmentRelation.BROAD_MATCH,
            (
                "Engine enum SPMUnitTenureType has three categories; occupiers "
                "without rent count as renters, as the ACS TEN = 4 build map "
                "does."
            ),
            group_rule=GroupRule.HOUSEHOLD_VALUE,
        ),
        bind(
            "is_in_public_housing",
            "household",
            "fact:household.tenure",
            Predicate(
                clauses=(("fact:household.tenure", ("rented_public_authority",)),)
            ),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine flag for public housing; the concept's public-authority "
                "landlord category is its closest primitive."
            ),
        ),
        bind(
            "pre_subsidy_rent",
            "person",
            "fact:household.rent",
            AllocateToReferencePerson(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Pre subsidy rent' (a person input; rent is "
                "formula-owned from it after housing assistance); it does not "
                "say whether separately billed utilities count, and the concept "
                "excludes them."
            ),
        ),
        bind(
            "home_mortgage_interest",
            "person",
            "fact:household.mortgage_interest",
            AllocateToReferencePerson(),
            AlignmentRelation.BROAD_MATCH,
            (
                "Engine documentation: 'Home mortgage interest, including both "
                "reported and not reported on federal Form 1098', which also "
                "covers second homes; the concept is the own dwelling only."
            ),
        ),
        bind(
            "real_estate_taxes",
            "person",
            "fact:household.property_tax",
            AllocateToReferencePerson(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Real estate taxes', with no definition. The "
                "engine reads it both as tax on the home (SPM housing_cost, "
                "homestead credits) and in the SALT deduction, which covers all "
                "real property. ACS builds place TAXAMT on the reference person "
                "(acs_inputs.py)."
            ),
        ),
        bind(
            "mortgage_payments",
            "spm_unit",
            ("fact:household.mortgage_interest", "fact:household.mortgage_principal"),
            Sum(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Mortgage payments' (SPM housing cost); read as "
                "interest plus principal. Whether escrowed taxes and insurance "
                "count is not stated."
            ),
            group_rule=GroupRule.ALLOCATE_TO_REFERENCE_UNIT,
        ),
        bind(
            "would_file_if_eligible_for_refundable_credit",
            "tax_unit",
            "fact:person.take_up_seed",
            TakeUpThreshold(program="us.refundable_credit_filing"),
            AlignmentRelation.APPROXIMATE,
            (
                "A filing-behaviour flag (engine default True): take-up of "
                "refundable credits through filing. The unit takes its "
                "reference member's draw."
            ),
            group_rule=GroupRule.REFERENCE_MEMBER,
        ),
        bind(
            "would_file_taxes_voluntarily",
            "tax_unit",
            "fact:person.take_up_seed",
            TakeUpThreshold(program="us.voluntary_tax_filing"),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'Whether this tax unit would file taxes "
                "even when not required and not seeking a refund from "
                "refundable credits' (default False). A filing propensity "
                "rather than program take-up, drawn the same way; the unit "
                "takes its reference member's draw."
            ),
            group_rule=GroupRule.REFERENCE_MEMBER,
        ),
    ),
    unmapped={},
    structural_inputs=(
        "household_id",
        "tax_unit_id",
        "spm_unit_id",
        "family_id",
        "marital_unit_id",
        "person_household_id",
        "person_tax_unit_id",
        "person_spm_unit_id",
        "person_family_id",
        "person_marital_unit_id",
        "household_weight",
        "family_weight",
    ),
)
