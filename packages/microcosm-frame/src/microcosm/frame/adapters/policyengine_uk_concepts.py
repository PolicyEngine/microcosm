"""The policyengine-uk concept mapping: which engine inputs each concept feeds.

Kept apart from the adapter module so that editing evidence notes never
changes the adapter's source hash, which the graph's simulate kernel folds
into its implementation hash. :class:`~microcosm.frame.adapters.policyengine_uk.
PolicyEngineUKEngine` returns :data:`POLICYENGINE_UK_CONCEPT_MAPPING` from
``concept_mapping()``.
"""

from microcosm.frame.concept_mapping import (
    ConceptMapping,
    Fraction,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    Predicate,
    Product,
    Recode,
    RelationshipRole,
    Role,
    Sum,
    TakeUpThreshold,
    bind,
)
from microcosm.frame.concepts import AlignmentRelation

__all__ = ["POLICYENGINE_UK_CONCEPT_MAPPING"]

# ---------------------------------------------------------------------------
# Concept mapping
# ---------------------------------------------------------------------------
#
# Which PolicyEngine-UK inputs each engine-neutral concept feeds
# (microcosm.frame.concepts), reviewed against policyengine-uk 2.100.0. Every
# target is a pure input (in ``PolicyEngineUKEngine.variables``). The UK
# loader also accepts some formula-owned variables as overrides
# (``employment_income``, ``state_pension_reported``, ``is_household_head``);
# the mapping never targets those, and says so where a concept has no other
# route. Notes quote the engine's own label or documentation.


_UK_PERSON_TAKE_UP = (
    "adult_dependants_grant",
    "bursary_fund_16_to_19",
    "carers_allowance",
    "childcare_grant",
    "disabled_students_allowance",
    "marriage_allowance",
    "parents_learning_allowance",
    "scp",
    "travel_grant",
)
_UK_BENUNIT_TAKE_UP = (
    "care_to_learn",
    "child_benefit",
    "extended_childcare",
    "pc",
    "targeted_childcare",
    "tfc",
    "uc",
    "uc_childcare",
    "universal_childcare",
)


#: Each take-up flag's own engine documentation, and how the UK build fills
#: it, so every note quotes its own evidence.
_UK_TAKE_UP_EVIDENCE: dict[str, tuple[str, str]] = {
    "adult_dependants_grant": (
        "Whether this person would claim Adult Dependants' Grant if eligible.",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "bursary_fund_16_to_19": (
        "Whether this person would claim 16 to 19 Bursary Fund support if eligible.",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "carers_allowance": (
        "Whether this person would claim Carer's Allowance (or the Scottish "
        "Carer Support Payment) if entitled. Generated in a dataset from "
        "reported receipt ...",
        "The UK build sets it from reported receipt (frs_spine.py). No concept "
        "carries that receipt, so a seed threshold stands in for it.",
    ),
    "childcare_grant": (
        "Whether this person would claim Childcare Grant if eligible.",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "disabled_students_allowance": (
        "Whether this person would claim Disabled Students' Allowance if eligible.",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "marriage_allowance": (
        "Whether this person would claim Marriage Allowance if eligible. "
        "Generated stochastically in the dataset using take-up rates.",
        "The UK build draws it from one rate (frs_person_draws.py).",
    ),
    "parents_learning_allowance": (
        "Whether this person would claim Parents' Learning Allowance if eligible.",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "scp": (
        "Whether this child would be claimed for under Scottish Child Payment. "
        "Generated stochastically in the dataset using age-based take-up "
        "rates ...",
        "The UK build draws it with separate rates under and over 6 "
        "(frs_person_draws.py); one rate per program key replaces them.",
    ),
    "travel_grant": (
        "Whether this person would claim Travel Grant if eligible.",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "care_to_learn": (
        "Whether this BenUnit would claim Care to Learn if eligible",
        "The UK build does not set it, so the engine default (True) applies.",
    ),
    "child_benefit": (
        "Whether this benefit unit would claim Child Benefit if eligible. "
        "Generated stochastically in the dataset using take-up rates.",
        "The UK build forces reported recipients to take up and draws the rest "
        "(frs_take_up.py); a threshold alone has no such anchor.",
    ),
    "extended_childcare": (
        "Whether this family would claim extended childcare entitlement if eligible",
        "The UK build draws it from one rate (frs_take_up.py).",
    ),
    "pc": (
        "Whether this benefit unit would claim Pension Credit if eligible. "
        "Generated stochastically in the dataset using take-up rates.",
        "The UK build forces reported recipients to take up and draws the rest "
        "(frs_take_up.py); a threshold alone has no such anchor.",
    ),
    "targeted_childcare": (
        "Whether this family would claim targeted childcare entitlement if eligible",
        "The UK build draws it from one rate (frs_take_up.py).",
    ),
    "tfc": (
        "Whether this family would claim Tax-Free Childcare if eligible",
        "The UK build draws it from one rate (frs_take_up.py).",
    ),
    "uc": (
        "Whether this family would claim the Universal Credit if eligible. "
        "Generated stochastically in the dataset using take-up rates.",
        "The UK build forces reported recipients to take up and draws the rest "
        "only among benefit units with an adult under State Pension age, the "
        "units Universal Credit can reach (frs_take_up.py, "
        "stochastic_assignment.py); a threshold alone has neither the anchor "
        "nor the population.",
    ),
    "uc_childcare": (
        "Whether this family would claim the Universal Credit childcare element "
        "if entitled. Generated stochastically in a dataset from take-up rates "
        "...",
        "The UK build draws it with separate couple and single rates "
        "(frs_take_up.py); one rate per program key replaces them.",
    ),
    "universal_childcare": (
        "Whether this BenUnit would claim universal childcare entitlement if eligible",
        "The UK build draws it from one rate (frs_take_up.py).",
    ),
}


def _uk_take_up(program: str, entity: str) -> InputBinding:
    person = entity == "person"
    documentation, build = _UK_TAKE_UP_EVIDENCE[program]
    return bind(
        f"would_claim_{program}",
        entity,
        "fact:person.take_up_seed",
        TakeUpThreshold(program=f"uk.{program}"),
        AlignmentRelation.APPROXIMATE,
        (
            f"Engine documentation: '{documentation}' (default True). {build} "
            "Here the flag is true when the program's draw from the persistent "
            "seed falls below its rate."
            + ("" if person else " The unit takes its reference member's draw.")
        ),
        group_rule=None if person else GroupRule.REFERENCE_MEMBER,
    )


POLICYENGINE_UK_CONCEPT_MAPPING = ConceptMapping(
    engine="policyengine-uk",
    engine_version="2.100.0",
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
            "Engine label 'age', documentation 'Age in years' (float).",
        ),
        bind(
            "gender",
            "person",
            "fact:person.sex",
            Recode(pairs=(("female", "FEMALE"), ("male", "MALE"))),
            AlignmentRelation.EXACT,
            "Engine enum Gender (MALE, FEMALE), label 'Gender of the person'.",
        ),
        bind(
            "is_parent",
            "person",
            (
                "fact:person.parent_1_person_id",
                "fact:person.parent_2_person_id",
                "fact:person.age",
            ),
            RelationshipRole(role=Role.PARENT_OF_CORESIDENT_CHILD, max_child_age=19),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Whether this person is a parent in their benefit "
                "unit'. Read as the parent of a co-resident child aged 19 or "
                "under; the benefit-unit definition of a dependent child (under "
                "16, or under 20 in qualifying education) is narrower, and a "
                "stepchild outside the benefit unit may differ."
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
                "Engine label 'employment income before labour supply "
                "responses'; employment_income is formula-owned from it."
            ),
        ),
        bind(
            "self_employment_income",
            "person",
            (
                "fact:person.nonfarm_self_employment_income",
                "fact:person.farm_self_employment_income",
            ),
            Sum(),
            AlignmentRelation.EXACT,
            (
                "Engine documentation: 'Income from self-employment profits, "
                "including gig work ... net of self-employment expenses'; it "
                "has no farm split, so the two concepts sum into it."
            ),
        ),
        # --- Capital income -----------------------------------------------
        bind(
            "savings_interest_income",
            "person",
            "fact:person.interest_income",
            Identity(),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'Income from interest on savings, gross "
                "of tax', ISA interest included (taxable savings interest "
                "subtracts tax_free_savings_income from it). Securities and "
                "National Savings interest belong to other_investment_income, "
                "which stays uncovered."
            ),
        ),
        bind(
            "individual_savings_account_interest_income",
            "person",
            "fact:person.interest_income",
            Fraction(parameter="uk.isa_interest_fraction"),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine label 'Amount received in interest from Individual "
                "Savings Accounts': a tax-free part of the interest "
                "savings_interest_income already holds in full, read through "
                "tax_free_savings_income."
            ),
        ),
        bind(
            "dividend_income",
            "person",
            "fact:person.dividend_income",
            Identity(),
            AlignmentRelation.EXACT,
            "Engine documentation: 'Total income from dividends, gross of tax'.",
        ),
        bind(
            "property_income",
            "person",
            "fact:person.rental_income",
            Identity(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'rental income', documentation 'Income from rental "
                "of property'; whether expenses are netted is not stated, and "
                "the concept is net."
            ),
        ),
        bind(
            "capital_gains_before_response",
            "person",
            "fact:person.realized_capital_gains",
            Identity(),
            AlignmentRelation.EXACT,
            (
                "Engine label 'capital gains before responses'; capital_gains "
                "adds the behavioural response. The residential-property, BADR "
                "and carried-interest leaves stay uncovered."
            ),
        ),
        # --- Pensions -----------------------------------------------------
        bind(
            "private_pension_income",
            "person",
            "fact:person.private_pension_income",
            Identity(),
            AlignmentRelation.EXACT,
            (
                "Engine documentation: 'Income from private or occupational "
                "pensions (not including the State Pension)'."
            ),
        ),
        # --- Work intensity -----------------------------------------------
        bind(
            "hours_worked",
            "person",
            ("fact:person.usual_weekly_hours", "fact:person.weeks_worked"),
            Product(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine label 'Annual hours worked'; weekly_hours is "
                "formula-owned as hours_worked / 52, so usual hours times weeks "
                "worked reads back as usual hours only for a full-year worker."
            ),
        ),
        # --- Disability and education -------------------------------------
        bind(
            "is_disabled_for_benefits",
            "person",
            "fact:person.has_disability",
            Identity(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'Whether this person is disabled for "
                "benefits purposes. In dataset mode, determined by reported "
                "DLA/PIP claims.' A benefit status standing in for the "
                "functional concept; the two differ."
            ),
        ),
        bind(
            "highest_education",
            "person",
            "fact:person.educational_attainment",
            Recode(
                pairs=(
                    ("less_than_primary", "NOT_COMPLETED_PRIMARY"),
                    ("primary", "PRIMARY"),
                    ("lower_secondary", "LOWER_SECONDARY"),
                    ("upper_secondary", "UPPER_SECONDARY"),
                    ("post_secondary_non_tertiary", "POST_SECONDARY"),
                    ("short_cycle_tertiary", "POST_SECONDARY"),
                    ("bachelor_or_equivalent", "TERTIARY"),
                    ("master_or_equivalent", "TERTIARY"),
                    ("doctoral_or_equivalent", "TERTIARY"),
                )
            ),
            AlignmentRelation.BROAD_MATCH,
            (
                "Engine enum EducationType, label 'Highest status education "
                "completed'. The enum states no ISCED correspondence, and the "
                "engine reads TERTIARY as a graduate (student-loan plan "
                "assignment, data/economic_assumptions.py). The mapping "
                "follows the UK build's EDUCQUAL_MAP (frs_education.py): "
                "degree and above (ISCED 6-8) is TERTIARY, and higher "
                "education below degree (ISCED 5) joins post-secondary "
                "non-tertiary (ISCED 4) as POST_SECONDARY."
            ),
        ),
        bind(
            "childcare_grant_full_time_student",
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
                "Engine documentation: 'Whether the person is studying full-time "
                "for Childcare Grant purposes. This must currently be set "
                "explicitly ...' (default False). Read as full-time ISCED 5-8 "
                "enrollment; the grant's own course rules are narrower."
            ),
        ),
        bind(
            "in_HE",
            "person",
            "fact:person.education_enrollment",
            Predicate(clauses=(("fact:person.education_enrollment", ("tertiary",)),)),
            AlignmentRelation.APPROXIMATE,
            "Engine label 'In higher education', read as ISCED 5-8 enrollment.",
        ),
        # --- Take-up ------------------------------------------------------
        *(_uk_take_up(program, "person") for program in _UK_PERSON_TAKE_UP),
        *(_uk_take_up(program, "benunit") for program in _UK_BENUNIT_TAKE_UP),
        bind(
            "child_benefit_opts_out",
            "benunit",
            "fact:person.take_up_seed",
            TakeUpThreshold(program="uk.child_benefit_opt_out"),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'Whether this family would opt out of "
                "receiving Child Benefit payments. Generated stochastically in "
                "the dataset using opt-out rates.' (default False). True when "
                "the opt-out draw falls below the opt-out rate; the unit takes "
                "its reference member's draw."
            ),
            group_rule=GroupRule.REFERENCE_MEMBER,
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
                    ("rented_public_authority", "RENT_FROM_COUNCIL"),
                    ("rented_nonprofit_social", "RENT_FROM_HA"),
                    ("rented_private", "RENT_PRIVATELY"),
                    ("rent_free", "RENT_PRIVATELY"),
                )
            ),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine enum TenureType matches five categories one to one; it "
                "has no rent-free category, and the UK build also defaults "
                "unmapped FRS tenure codes to RENT_PRIVATELY."
            ),
        ),
        bind(
            "rent",
            "household",
            "fact:household.rent",
            Identity(),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'The total amount of rent paid by the "
                "household in the year'; it does not say whether housing "
                "benefit or separately billed charges are netted."
            ),
        ),
        bind(
            "mortgage_interest_repayment",
            "household",
            "fact:household.mortgage_interest",
            Identity(),
            AlignmentRelation.EXACT,
            (
                "Engine documentation: 'Total amount spent on mortgage interest "
                "repayments'."
            ),
        ),
        bind(
            "mortgage_capital_repayment",
            "household",
            "fact:household.mortgage_principal",
            Identity(),
            AlignmentRelation.EXACT,
            "Engine label 'mortgage capital repayments'.",
        ),
        bind(
            "council_tax",
            "household",
            "fact:household.property_tax",
            Identity(),
            AlignmentRelation.NARROW_MATCH,
            (
                "Engine documentation: 'Gross annual Council Tax liability before "
                "Council Tax Reduction', supplied by the dataset. Great Britain "
                "only: Northern Ireland's domestic_rates stays uncovered."
            ),
        ),
    ),
    unmapped={
        "fact:person.legal_marital_status": (
            "policyengine-uk derives marital status from benefit-unit "
            "composition; marital_status and is_married are formula-owned."
        ),
        "fact:person.partner_person_id": (
            "Partnerships reach policyengine-uk only through benefit-unit "
            "membership, a structure the bundle must already carry (the "
            "adapter builds no units); is_benunit_head and is_married are "
            "formula-owned."
        ),
        "fact:household.reference_person_id": (
            "is_household_head is formula-owned (the oldest member). The UK "
            "build overrides it with the FRS household reference person "
            "(frs_spine.py), an override the loader accepts, but it is not an "
            "input, so the mapping does not target it. Every housing input is "
            "household-level, so nothing is allocated to the reference person."
        ),
        "fact:person.liquid_financial_assets": (
            "policyengine-uk holds financial wealth on the household: savings "
            "('Household liquid savings'), gross_financial_wealth and "
            "net_financial_wealth are household inputs, and its one "
            "person-level stock input is student_loan_balance, a debt. A "
            "person concept can feed only a person input or, through a group "
            "rule, a group-entity input. uc_reported_capital (benefit unit; "
            "'Claimant-level capital for Universal Credit when household-level "
            "asset data cannot be attributed across multiple benefit units', "
            "default -1) is Universal Credit capital: when it is zero or more, "
            "uc_assessable_capital takes it in place of the household-capital "
            "proxy it otherwise apportions by benefit-unit adults, so feeding "
            "it would be a modelling choice for a UK build, not a mapping of "
            "this concept."
        ),
        "fact:person.public_pension_income": (
            "The State Pension is computed from state_pension_reported, which "
            "is formula-owned: the loader accepts it as an override, but it is "
            "not an input."
        ),
    },
    structural_inputs=(
        "person_id",
        "benunit_id",
        "household_id",
        "person_benunit_id",
        "person_household_id",
        "household_weight",
        "raw_person_weight",
        "original_weight",
        "people",
        "families",
        "households",
    ),
)
