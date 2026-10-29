"""The policyengine-uk concept mapping: which engine inputs each concept feeds.

Kept apart from the adapter module so that editing evidence notes never
changes the adapter's source hash, which the graph's simulate kernel folds
into its implementation hash. :class:`~microcosm.frame.adapters.policyengine_uk.
PolicyEngineUKEngine` returns :data:`POLICYENGINE_UK_CONCEPT_MAPPING` from
``concept_mapping()``.
"""

from microcosm.frame.concept_mapping import (
    ConceptMapping,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    Predicate,
    Product,
    Recode,
    RelationshipRole,
    Role,
    SignedPart,
    TakeUpThreshold,
    bind,
)
from microcosm.frame.concepts import AlignmentRelation

__all__ = [
    "POLICYENGINE_UK_CONCEPT_MAPPING",
    "UK_ENGINE_DEFAULT_INPUTS",
    "UK_ENGINE_DEFAULT_TAKE_UP_INPUTS",
]

# ---------------------------------------------------------------------------
# Concept mapping
# ---------------------------------------------------------------------------
#
# Which PolicyEngine-UK inputs each engine-neutral concept feeds
# (microcosm.frame.concepts), reviewed against policyengine-uk 2.162.14. Every
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
    "uc_at_legacy_closure",
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
        "The UK build sets it from reported receipt (frs_spine.py) and opens it "
        "at one rate to carers the earnings limit alone keeps from the allowance "
        "(carers_allowance_take_up.py). No concept carries that receipt, so a "
        "seed threshold stands in for it.",
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
        "Whether this benefit unit would claim Child Benefit if eligible, "
        "including registered claimants who opt out of payment. ... Datasets "
        "must export claims rather than claims excluding opt-outs.",
        "The UK build forces reported recipients to take up and draws the rest "
        "(frs_take_up.py), and on an engine with opt-outs the Child Benefit "
        "redraw exports registered claims, opt-outs included "
        "(child_benefit_take_up.py); a threshold alone has no such anchor.",
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
    "uc_at_legacy_closure": (
        "Whether this family would claim Universal Credit once a legacy "
        "means-tested benefit it receives closes (legacy_benefits_closed). "
        "Datasets draw it at DWP's Move to Universal Credit claim rates by "
        "legacy benefit combination. ...",
        "The UK build draws it at DWP's Move to Universal Credit claim rates "
        "for each combination of legacy benefits, net of the units that "
        "would claim anyway, among units reporting a legacy benefit and not "
        "Universal Credit (legacy_closure_take_up.py, microcosm#1095); a "
        "threshold alone has neither the combinations nor the population.",
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


#: Take-up inputs of the engine the build leaves at the engine's default,
#: with why. ``would_claim_uc_at_legacy_closure`` (policyengine-uk 2.128.0,
#: pe-uk#2119) left this register once DWP's Move to UC claim rates became a
#: Chronicle fact (microcosm#1095).
UK_ENGINE_DEFAULT_TAKE_UP_INPUTS: dict[str, str] = {
    "would_claim_wales_childcare_offer": (
        "Whether a family would claim the Childcare Offer for Wales if eligible "
        "(policyengine-uk 2.155.0, pe-uk#2535); the engine's default is true, so "
        "every eligible family in Wales claims it. The Welsh Government publishes "
        "the Offer's take-up, but the pinned Chronicle feed carries no fact for "
        "it, so the build leaves the default: the Offer's cost, and the "
        "childcare support of Welsh families, are overstated by the eligible "
        "families who do not claim. A Chronicle request is drafted, and the flag "
        "is drawn at the published rate once the fact lands (María's ruling of "
        "2026-10-25, microcosm#1095)."
    ),
}


_UK_OWNED_COMPANY_STOCKS_AT_DEFAULT = (
    "The FRS records whether an adult runs their own company (the job record's "
    "director answers, from which the root stage sets "
    "stands_as_sole_owner_or_partner_of_company and five more owner-manager "
    "inputs) but no company balance sheet. So the three regulation 77 stocks, "
    "owned_company_capital, owned_company_holding_value and "
    "owned_company_trade_assets, stay at zero, and Universal Credit makes no "
    "company substitution (UC Regs 2013 reg. 77(2), (3)(a)): an owner-manager's "
    "capital is the unit's recorded capital alone. It is too high by their own "
    "company's shares where the recorded capital includes them (the FRS "
    "stocks-and-shares questions do not exclude them), which the regulation "
    "replaces by the company's capital, and too low by the company's capital "
    "beyond the assets used wholly and exclusively in its trade, which the "
    "regulation counts. Only the capital test moves: tariff income above "
    "GBP 6,000 and the GBP 16,000 limit (microcosm#1095; for María's signature)."
)
_UK_OWNED_COMPANY_INTERMEDIARIES_AT_DEFAULT = (
    "The FRS does not record whether the intermediaries or managed service "
    "company rules (ITEPA 2003 Part 2, Chapters 8 to 10) make an owner-manager's "
    "company income employed earnings, so owned_company_intermediary_earnings_"
    "chapter stays NONE and owned_company_intermediary_earnings_from_main_"
    "employment stays False. uc_company_intermediary_exclusion_applies is then "
    "false for everyone, and Universal Credit's company owner rule applies to "
    "every owner-manager the root stage flags, including a contractor caught by "
    "those rules in their main employment, whom UC Regs 2013 reg. 77(5) takes "
    "out of it. Such a contractor's award is assessed on the company's income "
    "and capital instead of as an employee's, which can move it either way "
    "(microcosm#1095; for María's signature)."
)
_UK_CHILD_LIMIT_EXCEPTIONS_AT_DEFAULT = (
    "The exceptions to the Universal Credit and Child Tax Credit two-child limit "
    "(UC Regs 2013 Sch 12; CTC Regs 2002 regs 9 to 14; policyengine-uk 2.161.0): "
    "multiple births, adoption, non-parental caring arrangements, non-consensual "
    "conception and the continuation of an earlier exception. The FRS records "
    "none of them as such, so each stays false, and a third or later child who "
    "qualifies for one gets no child element: Universal Credit is understated "
    "for those families until the limit ends on 6 April 2026. Multiple births "
    "(siblings of the same age) and non-parental care (a child whose benefit "
    "unit holds no parent) could be read from the FRS household grid after this "
    "lock (microcosm#1095; for María's signature)."
)


#: Other engine inputs the build leaves at the engine's default, each signed as
#: an approximation with the error it can make (microcosm#1095).
UK_ENGINE_DEFAULT_INPUTS: dict[str, str] = {
    "self_employment_gross_receipts": (
        "No source of the self-employed's turnover is read yet (the FRS job "
        "record and the SPI tape are the candidates, as #1145 found landlords' "
        "receipts). Since policyengine-uk 2.108.0 the trading allowance replaces "
        "expenses and does not stack on them, so without receipts "
        "trading_allowance_deduction relieves a profit of GBP 1,000 or less in "
        "full and gives a larger profit no allowance. Taxable profit is off by at "
        "most GBP 1,000 a person: too low where receipts above GBP 1,000 leave a "
        "small profit, too high just above GBP 1,000 where the allowance would "
        "beat the expenses. Signed as an approximation with the plan for the "
        "next policyengine-uk bump, approved on 2026-10-23."
    ),
    "housing_benefit_residential_capital_exception": (
        "The working-age residential capital tariff exception (Housing Benefit "
        "Regulations 2006 reg. 52(4)-(5), (8)-(9); NI reg. 49) covers residents "
        "of the residential accommodation and saved circumstances it lists, and "
        "policyengine-uk 2.148.4 asks for it pre-assessed. The FRS samples "
        "private households, so it holds almost no such resident, and it records "
        "no assessment of the circumstances, so the input stays False. A "
        "qualifying working-age resident takes tariff income on capital above "
        "GBP 6,000 instead of GBP 10,000, so Housing Benefit is understated for "
        "such residents alone; eligibility, capital and the upper limit do not "
        "move (Vahid's review of #1383, item 4)."
    ),
    "uc_receives_new_claimant_health_element": (
        "From 6 April 2026 the LCWRA element of a claim made after that date "
        "is paid at the lower new-claimant rate (UC Regs 2013 reg. 36). Every "
        "FRS 2024-25 recipient claimed before it, and the survey cannot say who "
        "would claim after it, so the build supplies no value and "
        "policyengine-uk draws one: for a simulation built from data it gives "
        "the new-claimant rate to a share of the benefit units "
        "(gov.dwp.universal_credit.rebalancing.new_claimant_share, WPI "
        "Economics' estimates for 2026-27 to 2029-30) on a seeded draw in "
        "benefit-unit order. The share is zero in 2025, the calibration year. "
        "From 2026 the share of health-element units paid the lower rate "
        "follows the parameter on average, but which units are paid it is "
        "random with respect to everything the file records (microcosm#1095; "
        "for María's signature)."
    ),
    "owned_company_capital": _UK_OWNED_COMPANY_STOCKS_AT_DEFAULT,
    "owned_company_holding_value": _UK_OWNED_COMPANY_STOCKS_AT_DEFAULT,
    "owned_company_trade_assets": _UK_OWNED_COMPANY_STOCKS_AT_DEFAULT,
    "owned_company_intermediary_earnings_chapter": (
        _UK_OWNED_COMPANY_INTERMEDIARIES_AT_DEFAULT
    ),
    "owned_company_intermediary_earnings_from_main_employment": (
        _UK_OWNED_COMPANY_INTERMEDIARIES_AT_DEFAULT
    ),
    "benunit_reported_capital": (
        "The build records each benefit unit's capital as the complete "
        "overrides uc_reported_capital and pension_credit_reported_capital "
        "(uc_capital_coherence), which both means tests read in place of this "
        "financial-only record. Where a unit's recorded capital is unavailable "
        "(-1, with no TOTCAPB4), it falls back to the household capital proxy, "
        "as the engine documents for this input at -1. Setting it from TOTCAPB4 "
        "as well would change no assessment: where TOTCAPB4 is recorded the "
        "overrides take precedence, and where it is not the input would also "
        "be -1 (microcosm#1095; for María's signature)."
    ),
    "housing_benefit_has_main_phase_esa": (
        "Whether the Housing Benefit claimant is personally entitled to "
        "main-phase ESA (policyengine-uk 2.152.0), which gives a working-age "
        "claimant the higher personal allowance (HB Regs 2006 Sch 3 para 1A; NI "
        "Sch 4 para 1A). The build does not set it, so it stays false and the "
        "engine keeps its age-based allowance: Housing Benefit is understated "
        "for a claimant under 25 on main-phase ESA. The FRS records ESA receipt "
        "and its group, from which a mapping could follow after this lock "
        "(microcosm#1095; for María's signature)."
    ),
    "severe_disability_premium_joint_occupier_exclusion_applies": (
        "A caller's assertion that every other rent-liable claimant or partner "
        "in the household is a joint occupier, or has a separate liability, "
        "that the legacy severe disability premium's non-dependant test "
        "excludes (policyengine-uk 2.159.0). The FRS records the conditions "
        "only in part, so the input stays false, as the engine asks when the "
        "facts are unknown: the premium can be denied to a severely disabled "
        "claimant who shares a liability on those terms (microcosm#1095; for "
        "María's signature)."
    ),
    "housing_benefit_family_premium_category": (
        "Housing Benefit's protected family premium (policyengine-uk 2.160.0): "
        "a working-age claim that kept it from before its closure to new "
        "claims (1 May 2016 in Great Britain, 5 September 2016 in Northern "
        "Ireland), or the pre-April 1998 lone-parent rate. The FRS records no "
        "claim history, so every unit takes the default, NONE: Housing Benefit "
        "is understated for any such protected claim still in payment "
        "(microcosm#1095; for María's signature)."
    ),
    **dict.fromkeys(
        (
            "child_limit_adoption_exception",
            "child_limit_continuation_exception",
            "child_limit_multiple_birth_exception",
            "child_limit_non_consensual_conception_exception",
            "child_limit_non_parental_caring_exception",
        ),
        _UK_CHILD_LIMIT_EXCEPTIONS_AT_DEFAULT,
    ),
    **dict.fromkeys(
        (
            "electricity_consumption_kwh",
            "gas_consumption_kwh",
            "ni_epg_full_rate_applies",
        ),
        "Northern Ireland's Energy Price Guarantee discounts (policyengine-uk "
        "2.161.1) read each month's kWh and an assertion that the full rates "
        "apply. The frame stores annual spend, not monthly kWh, so the kWh stay "
        "unknown (-1), the full rates are not asserted and the discounts are not "
        "calculated. The guarantee applied in 2022-23 only, so the calibration "
        "year is unaffected (microcosm#1095; for María's signature).",
    ),
    **dict.fromkeys(
        (
            "individual_savings_account_dividend_income",
            "non_cash_isa_cash_interest",
        ),
        "ISA dividends and the cash interest held in non-cash ISAs "
        "(policyengine-uk 2.162.2). The FRS records ISA interest as one total, "
        "which the frame supplies in individual_savings_account_interest_income "
        "(microcosm#1360), so the of-which detail stays zero. Until 2027-28 that changes nothing; from 2027-28 the "
        "detail sets isa_cash_interest_flat_rate_charge, the final charge on cash "
        "interest in non-cash ISAs, so projections from that year show no such "
        "charge. It records no ISA dividends "
        "apart from other dividends, so ISA dividends stay zero: any the "
        "survey's dividend answer holds are taxed as ordinary dividends "
        "(microcosm#1095; for María's signature).",
    ),
    **dict.fromkeys(
        (
            "lifetime_isa_withdrawals",
            "lifetime_isa_withdrawals_towards_first_home",
            "lifetime_isa_first_time_buyer",
            "lifetime_isa_terminally_ill",
            "lifetime_isa_months_since_first_payment",
        ),
        "The Lifetime ISA's withdrawals and the conditions on them "
        "(policyengine-uk 2.162.2). Neither the FRS nor the WAS records "
        "withdrawals from a Lifetime ISA (the was_lisa stage draws holdings and "
        "isa_subscriptions the payments, microcosm#1415), so withdrawals stay "
        "zero and the withdrawal charge is zero for every account "
        "(microcosm#1095; for María's signature).",
    ),
    **dict.fromkeys(
        (
            "main_residential_property_purchased_is_in_uk",
            "main_residential_property_purchased_with_mortgage",
        ),
        "Two flags on a main-home purchase (policyengine-uk 2.162.2) that only "
        "the Lifetime ISA withdrawal charge's first-home exemption reads. With no "
        "Lifetime ISA withdrawals they change nothing, so they stay false "
        "(microcosm#1095; for María's signature).",
    ),
    "is_in_first_year_of_apprenticeship": (
        "Whether an apprentice is in the first 12 months of the apprenticeship "
        "(policyengine-uk 2.162.8, policyengine-uk#2683): the apprentice "
        "minimum wage applies under 19 or in that first year. The build flags "
        "apprentices aged 21 to 24 only (frs_person_draws) and records no "
        "apprenticeship start, so the input stays false and every flagged "
        "apprentice takes the age-related minimum wage. A first-year apprentice "
        "aged 21 to 24 is then held to the higher minimum income in the "
        "Tax-Free Childcare, extended childcare and Childcare Offer for Wales "
        "income tests, so those tests are slightly harder for them to pass; "
        "employer NICs do not read it (microcosm#1373; for María's signature)."
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
    engine_version="2.162.14",
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
            SignedPart("positive"),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'Income from self-employment profits, "
                "including gig work ... net of self-employment expenses', "
                "never negative from 2.131.0, the year's losses going to "
                "trading_loss (pe-uk#2085). It has no farm split, so the two "
                "concepts sum and the profit is the positive part of the sum. "
                "The concepts net each person's trades, so a profit beside "
                "another trade's loss arrives net of that loss."
            ),
        ),
        bind(
            "trading_loss",
            "person",
            (
                "fact:person.nonfarm_self_employment_income",
                "fact:person.farm_self_employment_income",
            ),
            SignedPart("negative"),
            AlignmentRelation.APPROXIMATE,
            (
                "Engine documentation: 'Losses made in the tax year by the "
                "person's trades, professions or vocations, as a positive "
                "amount' (pe-uk#2085, from 2.131.0): the negative part of the "
                "two concepts' sum, so it and self_employment_income never "
                "both hold an amount. The concepts net each person's trades, "
                "so a loss beside another trade's profit arrives net of it."
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
                "Engine label from 2.162.2: 'Savings interest outside tax-free "
                "accounts'. The frame stores it so, under the additive ISA "
                "income contract it declares (microcosm#1360): ISA interest is "
                "stored apart in individual_savings_account_interest_income, a "
                "total the engine's formula takes first, and "
                "tax_free_savings_income adds it. A legacy gross file (the "
                "enhanced FRS) stores interest gross with the ISA part in "
                "tax_free_savings_income, which policyengine-uk subtracts when "
                "it loads the file. Securities and National Savings interest "
                "belong to other_investment_income, which stays uncovered."
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
                "Engine label 'property income', documentation: profits of UK and "
                "overseas property businesses, receipts less allowable expenses, "
                "'before the property allowance and before any costs of "
                "dwelling-related loans', which are relieved as a tax reduction "
                "instead. The concept nets every expense, mortgage interest "
                "included, and excludes sub-letting part of the own dwelling, so "
                "the engine's profit sits above it for mortgaged landlords."
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
                "Engine documentation: 'Whether a family would opt out of Child "
                "Benefit payments because of the High Income Child Benefit "
                "Charge. This is independent of would_claim_child_benefit and "
                "never identifies a claim by itself. ...' (default False). True "
                "when the opt-out draw falls below the opt-out rate; the unit "
                "takes its reference member's draw."
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
        "fact:household.source_household_id": (
            "Original donor household lineage is build provenance and weight "
            "diagnostic support, not a policyengine-uk rules input."
        ),
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
            "('Household liquid savings'), corporate_wealth ('Wealth held in "
            "corporations directly or through investment funds'), "
            "gross_financial_wealth and net_financial_wealth are household "
            "inputs, and no transform puts a person amount on a household "
            "input. Its person-level GBP stock inputs are student_loan_balance, "
            "a debt; car_list_price, the price vehicle excise duty's "
            "expensive-car supplement tests; and lifetime_isa_balance "
            "('Current value of the person's Lifetime ISAs, cash and stocks "
            "and shares together'), one savings wrapper, which the engine "
            "documents as already inside gross_financial_wealth and "
            "net_financial_wealth. Since policyengine-uk 2.130.0 three more "
            "describe a company the person owns, for Universal Credit's "
            "regulation 77 (owned_company_capital, owned_company_holding_value "
            "and owned_company_trade_assets); every one defaults to no such "
            "company, and the build sets none. savings and share_like_capital are two "
            "of Universal Credit's capital sources "
            "(gov.dwp.universal_credit.means_test.capital.sources); since "
            "policyengine-uk 2.147.0 share_like_capital is corporate_wealth unless "
            "its components, the household inputs directly_held_shares and "
            "unit_and_investment_trusts, add up to more; the WAS stage writes "
            "both on its corporate_wealth less the stocks-and-shares ISA, so "
            "they and the ISA sum to it (microcosm#1095). Universal Credit's "
            "one person-level source is "
            "lifetime_isa_countable_capital, the Lifetime ISA at its surrender "
            "value (capital.person_sources). "
            "uc_reported_capital (benefit unit, default -1) is Universal "
            "Credit capital; since policyengine-uk 2.136.1 a value of zero or "
            "more set directly is the unit's complete countable capital, in "
            "place of every source, property included. benunit_reported_capital, "
            "a benefit-unit input ('Savings and investments of the benefit "
            "unit's adults (the claimant and any partner) as recorded'), "
            "replaces the financial sources only: the household pool it "
            "otherwise sums from the household sources, net of what other "
            "benefit units report or override, and apportions by claimant and "
            "partner count, and the person-level sources of the unit's "
            "claimant and partner. Property is valued separately "
            "(uc_property_capital) and shared across every claimant and "
            "partner in the household. Feeding either would be a modelling "
            "choice for a UK build, not a mapping of this concept."
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
