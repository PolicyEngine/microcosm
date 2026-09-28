"""Tests split from packages/microcosm-build/tests/test_uk_uc_relationships.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_relationships import *


@pytest.mark.parametrize("route", ["adapter", "direct_h5"])
def test_released_engine_retains_source_claimants_and_uc_awards(tmp_path, route):
    """Explicit FRS roles survive both model-loading paths without fallback.

    This tests recorded claim membership, including a 17-year-old partner;
    it does not add or establish eligibility for under-18 claimants.
    2025 monthly standard allowances are 400.14 single old, 497.55 couple
    young and 628.10 couple old; an eldest child born before 2017 adds 339.
    """
    from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        write_uk_national_frame,
    )
    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    person = pd.DataFrame(
        {
            "person_id": range(1, 10),
            "person_benunit_id": [10, 10, 20, 20, 30, 30, 40, 40, 40],
            "person_household_id": [1, 1, 2, 2, 3, 3, 4, 4, 4],
            "age": [40, 19, 24, 17, 40, 18, 40, 39, 19],
            "is_benunit_head": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
            ],
            "is_parent": [True, False, False, False, True, False, True, True, False],
            "is_in_non_advanced_education": [
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
            ],
            "age_started_or_accepted_current_education_or_training": [18] * 9,
            "is_before_universal_credit_qualifying_young_person_terminal_date": [True]
            * 9,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20, 30, 40],
            "dependent_children": [1, 0, 1, 1],
            "would_claim_uc": [True] * 4,
            "uc_reported_capital": [0.0] * 4,
            "uc_latent_deduction_rate": [0.0] * 4,
            "uc_LCWRA_element": [0.0] * 4,
            "benefit_cap_reduction": [0.0] * 4,
        }
    )
    claimant = frs_uc_claimant_mask(person, benunit)
    person["is_uc_claimant"] = claimant
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=pd.DataFrame(
            {
                "household_id": [1, 2, 3, 4],
                "region": ["LONDON"] * 4,
                "council_tax": [0.0] * 4,
                "rent": [0.0] * 4,
                "tenure_type": ["OWNED_OUTRIGHT"] * 4,
            }
        ),
        time_period=2025,
        household_weights=np.ones(4),
    )
    variables = [
        "is_uc_claimant",
        "uc_standard_allowance",
        "uc_child_element",
        "universal_credit",
    ]
    if route == "adapter":
        values = PolicyEngineUKEngine().materialize(frame, variables, 2025)
    else:
        path = tmp_path / "claimants.h5"
        write_uk_national_frame(frame, path)
        resolver = UKMeasureResolver(
            simulation_source=path, scratch_dir=tmp_path, year=2025, frame=frame
        )
        assert "is_uc_claimant" in resolver.simulation.input_variables
        values = {
            name: np.asarray(resolver.simulation.calculate(name, 2025))
            for name in variables
        }
    np.testing.assert_array_equal(values["is_uc_claimant"], claimant)
    assert values["is_uc_claimant"].dtype.kind == "b"
    np.testing.assert_allclose(
        values["uc_standard_allowance"],
        [4801.68, 5970.60, 4801.68, 7537.20],
        atol=0.01,
        rtol=0,
    )
    np.testing.assert_allclose(
        values["uc_child_element"], [0, 0, 4068, 4068], atol=0.01, rtol=0
    )
    np.testing.assert_allclose(
        values["universal_credit"],
        [4801.68, 5970.60, 8869.68, 11605.20],
        atol=0.01,
        rtol=0,
    )
