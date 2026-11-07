"""policyengine-uk's whole-population formulas, declared for engine blocks.

policyengine-uk lists the formulas that read beyond their own households in
``utils/household_subset.py`` (``WHOLE_POPULATION_VARIABLES``): a weighted
total, a weighted rank or median, or records spread by weight. Over an engine
block or batch that holds part of the pool they read the part, not the pool
(microcosm#931). :data:`UK_ENGINE_POPULATION` classifies each one for the
shared machinery in :mod:`microcosm.frame.adapters.engine_population`,
reviewed against policyengine-uk 2.174.1:

- corrected: ``shareholding``, ``corporate_land_value`` and
  ``consumption_shareholding`` allocate by ``x / sum(x * w)``, so a block's
  own value times its pool-mass factor and key sums is the pool's;
- stored: ``months_since_last_birthday`` and ``attends_private_school``
  are inputs the spine stores;
- zero at policy: the labour supply response, the employer NI incidence
  responses, the carbon tax and the four contributed budget changes are zero
  while their parameters are (checked at each run);
- inert: before the spine stores attendance, a batched call may compute it
  only where it reaches household totals through
  ``private_school_vat_change``, which is zero while the private school VAT
  rate equals its baseline's;
- refused: the household income and wealth deciles and the relative poverty
  flags rank or take a median over the simulated population, which no factor
  corrects;
- restated: attendance ranks household net income over the simulated
  population, so a stage whose call runs in batches may compute it over the
  pool from household-local engine values instead; every batch's engine must
  then hold the attendance rates and the population factor the formula reads
  as the adapter's own tree does.

The window: policyengine-uk's constructor draws which Universal Credit
claimants are new claimants by benefit-unit position for the rebalancing
years (``scenarios/uc_reform.py``, ``REBALANCING_YEARS``), so blocks and
batches refuse any work from 2026 on.

Kept apart from the adapter module, as the concept mapping is, so that
editing the declaration never moves the adapter's source hash. Nothing here
imports policyengine-uk when the module loads; :func:`uk_engine_population`
imports it, validates the declaration against the installed engine and
fails closed.
"""

from __future__ import annotations

from typing import Any

from microcosm.frame.adapters.engine_population import (
    InertRule,
    ParameterTest,
    PopulationDeclaration,
    PositionWindow,
    all_of,
    any_of,
    validate_declaration,
)

__all__ = [
    "UK_ENGINE_POPULATION",
    "UK_SPINE_ENGINE_BATCH_COLUMN",
    "uk_engine_population",
]

#: The household column spine engine calls batch by: the clone index the
#: ladder frame carries (``ladder_clone_index_column("household")``), so a
#: batch is one clone's households.
UK_SPINE_ENGINE_BATCH_COLUMN = "household_clone_index"

_COICOP_CONSUMPTION = (
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
)

_EMPLOYER_NI = "gov.contrib.policyengine.employer_ni"
_BUDGET = "gov.contrib.policyengine.budget"


def _zero(path: str) -> ParameterTest:
    return ParameterTest(path, "eq", 0)


def _as_baseline(path: str) -> ParameterTest:
    return ParameterTest(path, "eq_path", f"baseline.{path}")


UK_ENGINE_POPULATION = PopulationDeclaration(
    engine="policyengine_uk",
    distribution="policyengine-uk",
    reviewed_version="2.174.1",
    list_source="policyengine_uk.utils.household_subset:WHOLE_POPULATION_VARIABLES",
    weight_variable="household_weight",
    corrected={
        # shareholding.py and corporate_land_value.py: corporate_sector_wealth
        # over its weighted total (times the aggregate corporate land value).
        "shareholding": "corporate_sector_wealth",
        "corporate_land_value": "corporate_sector_wealth",
        # consumption_share.py: consumption over its weighted total.
        "consumption_shareholding": "consumption",
    },
    key_adds={
        "corporate_sector_wealth": ("corporate_wealth", "private_pension_wealth"),
        "consumption": _COICOP_CONSUMPTION,
    },
    stored=("months_since_last_birthday", "attends_private_school"),
    zero_at_policy={
        # Zero without both elasticities (its formula's first test).
        "employment_income_behavioral_response": all_of(
            _zero("gov.simulation.labour_supply_responses.income_elasticity"),
            _zero("gov.simulation.labour_supply_responses.substitution_elasticity"),
        ),
        # Zero without the channel's incidence, or with all of it on employees.
        "employer_ni_response_capital_incidence": any_of(
            _zero(f"{_EMPLOYER_NI}.capital_incidence"),
            ParameterTest(f"{_EMPLOYER_NI}.employee_incidence", "eq", 1),
        ),
        "employer_ni_response_consumer_incidence": any_of(
            _zero(f"{_EMPLOYER_NI}.consumer_incidence"),
            ParameterTest(f"{_EMPLOYER_NI}.employee_incidence", "eq", 1),
        ),
        # Both incidence terms carry the rate.
        "carbon_tax": _zero("gov.contrib.ubi_center.carbon_tax.rate"),
        "consumer_incident_tax_revenue_change": _zero(
            f"{_BUDGET}.consumer_incident_tax_change"
        ),
        "education_budget_change": _zero(f"{_BUDGET}.education"),
        "nhs_budget_change": _zero(f"{_BUDGET}.nhs"),
        "other_public_spending_budget_change": _zero(
            f"{_BUDGET}.other_public_spending"
        ),
    },
    inert={
        # Attendance reaches household taxes only through
        # private_school_vat_change (household_tax.py), pupils times fees
        # times the difference between private_school_vat_rate under the
        # policy and under its baseline (private_school_vat.py); the rate
        # reads these three parameters. gov_tax adds the level.
        "attends_private_school": InertRule(
            sink="private_school_vat_change",
            levels=("private_school_vat", "baseline_private_school_vat"),
            carriers=("gov_tax",),
            predicate=all_of(
                _as_baseline("gov.hmrc.vat.private_school_fees_exempt"),
                _as_baseline("gov.hmrc.vat.standard_rate"),
                _as_baseline("gov.contrib.labour.private_school_vat"),
            ),
        ),
    },
    refused=(
        "household_income_decile",
        "household_income_decile_ahc",
        "household_wealth_decile",
        "in_relative_poverty_ahc",
        "in_relative_poverty_bhc",
        "pre_budget_change_ons_household_income_decile",
    ),
    window=PositionWindow(
        first_year=2026,
        source="policyengine_uk.scenarios.uc_reform:REBALANCING_YEARS",
        reason=(
            "policyengine-uk's constructor draws which Universal Credit "
            "claimants are new claimants (the health element they receive) "
            "with one random number per benefit unit in the order the "
            "simulation holds them, for each rebalancing year from 2026, so "
            "a block or batch draws other statuses than the pool"
        ),
    ),
    # Recorded for the receipts: what reads the corrected shares.
    observed=("LVT", "business_rates", "corporate_sdlt"),
    # create_structural_reforms_from_parameters (reforms/reforms.py) applies
    # these in the constructor, before any guard is armed; adjust_budgets
    # would rescale spending by a population total.
    constructor_factories=(
        "policyengine_uk.reforms.cps.marriage_tax_reforms:create_marriage_tax_reform",
        "policyengine_uk.reforms.conservatives.household_based_hitc:"
        "create_household_based_hitc_reform",
        "policyengine_uk.reforms.policyengine.disable_simulated_benefits:"
        "disable_simulated_benefits",
        "policyengine_uk.reforms.policyengine.adjust_budgets:adjust_budgets",
        "policyengine_uk.reforms.scotland.scottish_child_payment_reform:"
        "create_scottish_child_payment_reform",
    ),
    # attends_private_school.py reads these two under
    # gov.simulation.private_school_vat: the rates at the 21 percentile
    # breakpoints and the population adjustment factor.
    restated={
        "attends_private_school": (
            "gov.simulation.private_school_vat.private_school_attendance_rate",
            "gov.simulation.private_school_vat.private_school_factor",
        ),
    },
    digests={
        "policyengine_uk/variables/household/wealth/shareholding.py": (
            "95fae830757b8db478cab62c1e84b093af268abd8d0ed21a518c65b0d635a52b"
        ),
        "policyengine_uk/variables/household/wealth/corporate_land_value.py": (
            "2082f7890365a40c20c68a7e2cbc7d9cccfeacb13ce0684670bb960e138bcae2"
        ),
        "policyengine_uk/variables/household/consumption/consumption_share.py": (
            "b2e59e6f3a32f33482630938e385c07762265da2cf9f0848cd011a6166ad30cf"
        ),
        "policyengine_uk/variables/contrib/ubi_center/carbon_tax.py": (
            "ce9ce9c9fe346c9b6fcaeec6cd03fed602c1168bbd691a0fca24c797f3235627"
        ),
        "policyengine_uk/variables/contrib/policyengine/"
        "consumer_incident_tax_revenue_change.py": (
            "47dfa0d09317dd652bbbcecd46a76466d72eade2abaaf9a0a3bd7e5df9cf4304"
        ),
        "policyengine_uk/variables/contrib/policyengine/education_budget_change.py": (
            "4f1723792c35d41f3a13d1d790825bff830f5332fb2e134202ef8f266978d469"
        ),
        "policyengine_uk/variables/contrib/policyengine/nhs_budget_change.py": (
            "667a919cd07f0ad61c860473268770e7e76e77202e9f595c0e6e88711b5b6c48"
        ),
        "policyengine_uk/variables/contrib/policyengine/"
        "other_public_spending_budget_change.py": (
            "3ce3ac97d8a473f60732fcefdf4e4e6430958493f7ae132f14555a641a5488f1"
        ),
        "policyengine_uk/variables/contrib/policyengine/employer_ni/"
        "employer_ni_response_capital_incidence.py": (
            "6cb5bdd539d22342f0c614c8561ad0b86d45fdd545c3e814c819cbeff64512bd"
        ),
        "policyengine_uk/variables/contrib/policyengine/employer_ni/"
        "employer_ni_response_consumer_incidence.py": (
            "abfef8b2f57cc489282b09634e1d96f7464a64040e1ed701f01cdc8d30268058"
        ),
        "policyengine_uk/variables/gov/simulation/labour_supply_response/"
        "employment_income_behavioral_response.py": (
            "aeac3d6e861b742c2d260fe513ad430e4411cbe9804451076391fda9e4b64e07"
        ),
        "policyengine_uk/variables/contrib/labour/attends_private_school.py": (
            "1c0dc90e234cba789f31f94bdc7b619c882b155d63858939beda4ef8a7334c21"
        ),
        "policyengine_uk/variables/gov/hmrc/private_school_vat.py": (
            "389885eaead3ccc87971218fe5f6d5e8e50cdbb835b255096f0f82569e75cb30"
        ),
        "policyengine_uk/variables/gov/hmrc/baseline_private_school_vat.py": (
            "3ee5f683b7b9461203d166e3b7f0ccfa305b14819da3a474163a7d0060988269"
        ),
        "policyengine_uk/variables/gov/hmrc/private_school_vat_change.py": (
            "9d373dcd40dcb74747afd43d5637d7b650e65916992d31b34d08a7cba0b347ee"
        ),
        "policyengine_uk/reforms/policyengine/adjust_budgets.py": (
            "4aabd28d8bfdd617e4f29f5150e3ddc6cadd97ee5270b77455d0dd71c1acd23a"
        ),
        "policyengine_uk/reforms/reforms.py": (
            "51341a9f1494944eeea945bcc614f047de568bc455a00936f794d4b5984e42eb"
        ),
        "policyengine_uk.scenarios.uc_reform:add_universal_credit_reform": (
            "ca13079bb2043c8965dd2cc9397c13c58b7f98f8e58b2c678ab7602f72a55935"
        ),
    },
)


def uk_engine_population(system: Any | None = None) -> PopulationDeclaration:
    """The declaration, validated against the installed policyengine-uk.

    Pass the tax-benefit system the caller's simulations use; without one a
    fresh ``CountryTaxBenefitSystem`` is built. Raises
    :class:`~microcosm.frame.adapters.engine_population.PopulationDeclarationError`
    when the engine has moved from what was reviewed, so per-clone blocks
    and batches never run on an unreviewed engine. Only callers that block
    or batch call this; a single-engine run never needs it.
    """

    try:
        import policyengine_uk
    except ImportError as exc:
        raise ImportError(
            "The UK whole-population declaration requires the 'policyengine-uk' "
            "package. Install it with 'microcosm-frame[uk]'."
        ) from exc
    if system is None:
        system = policyengine_uk.CountryTaxBenefitSystem()
    validate_declaration(UK_ENGINE_POPULATION, system)
    return UK_ENGINE_POPULATION
