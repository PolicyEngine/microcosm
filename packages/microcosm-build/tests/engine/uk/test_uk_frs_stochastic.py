"""Tests split from packages/microcosm-build/tests/test_uk_frs_stochastic.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_stochastic import *


def test_uc_take_up_population_policy_matches_the_engine_working_age_test() -> None:
    """The policy read from the engine reproduces is_WA_adult for every age."""

    import importlib

    policyengine_uk = importlib.import_module("policyengine_uk")
    policy = uk_take_up_population_policy(2025)
    assert policy == UKTakeUpPopulationPolicy(
        adult_age=18,
        state_pension_age=66,
        instant="2025-01-01",
        source="policyengine-uk parameters " + metadata.version("policyengine-uk"),
    )

    ages = list(range(0, 101))
    people = {f"p{age}": {"age": {2025: age}} for age in ages}
    sim = policyengine_uk.Simulation(
        situation={
            "people": people,
            "benunits": {f"b{age}": {"members": [f"p{age}"]} for age in ages},
            "households": {f"h{age}": {"members": [f"p{age}"]} for age in ages},
        }
    )
    is_wa_adult = np.asarray(sim.calculate("is_WA_adult", 2025), dtype=bool)

    assert is_wa_adult.tolist() == policy.working_age(np.array(ages)).tolist()
