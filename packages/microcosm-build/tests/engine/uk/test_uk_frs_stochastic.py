"""Tests split from packages/microcosm-build/tests/test_uk_frs_stochastic.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_stochastic import *


def _engine_uc_eligible(people: dict, benunits: dict) -> list[bool]:
    """policyengine-uk's is_uc_eligible at 2025, with zero reported capital."""

    import importlib

    policyengine_uk = importlib.import_module("policyengine_uk")
    situation = {
        "people": people,
        "benunits": {
            name: {**unit, "uc_reported_capital": {2025: 0}}
            for name, unit in benunits.items()
        },
        "households": {
            f"h_{name}": {"members": unit["members"]} for name, unit in benunits.items()
        },
    }
    sim = policyengine_uk.Simulation(situation=situation)
    return np.asarray(sim.calculate("is_uc_eligible", 2025), dtype=bool).tolist()


def test_uc_take_up_population_mirrors_the_engines_eligibility_rule() -> None:
    """The population is policyengine-uk's is_uc_eligible, capital aside.

    The take-up population holds a unit with a UC claimant or partner aged 18
    to under Pension Credit qualifying age. One-person units aged 14 to 90
    agree with the engine; the two cases below are engine conditions the
    population does not read, named so that an engine change to either fails
    here.
    """

    policy = uk_take_up_population_policy(2025)
    assert policy == UKTakeUpPopulationPolicy(
        adult_age=18,
        state_pension_age=66,
        instant="2025-01-01",
        source="policyengine-uk parameters " + metadata.version("policyengine-uk"),
    )

    ages = list(range(14, 91))
    engine = _engine_uc_eligible(
        {f"p{age}": {"age": {2025: age}} for age in ages},
        {f"b{age}": {"members": [f"p{age}"]} for age in ages},
    )
    # A one-person unit's member is its claimant.
    assert engine == policy.working_age(np.array(ages)).tolist()

    # Named exception (ii): a 16- or 17-year-old claimant under the UC Regs
    # 2013 reg 8 exceptions (here a carer) meets UC's reduced minimum age.
    # The population leaves them out.
    assert _engine_uc_eligible(
        {"carer": {"age": {2025: 17}, "care_hours": {2025: 40}}},
        {"b_carer": {"members": ["carer"]}},
    ) == [True]
    assert not policy.working_age(np.array([17])).any()

    # Named exception (iii): a mixed-age couple keeping the SI 2019/37 saving
    # takes the Pension Credit route, which the population does not read: its
    # younger partner is a claimant or partner of working age.
    couple = {"older": {"age": {2025: 70}}, "younger": {"age": {2025: 60}}}
    assert _engine_uc_eligible(
        couple,
        {
            "b_couple": {
                "members": ["older", "younger"],
                "has_mixed_age_couple_pension_credit_saving": {2025: True},
            }
        },
    ) == [False]
    assert policy.working_age(np.array([70, 60])).tolist() == [False, True]
