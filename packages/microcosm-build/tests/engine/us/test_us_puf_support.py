"""Tests split from packages/microcosm-build/tests/test_us_puf_support.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_puf_support import *


@pytest.mark.parametrize("period", ["2024-01", "2024-12"])
def test_policyengine_broadcasts_annual_reported_enrollment_to_each_month(
    period: str,
) -> None:
    try:
        from policyengine_us import CountryTaxBenefitSystem, Simulation
    except ImportError:
        pytest.skip("requires the policyengine-us [us] extra")

    derived = derive_us_cps_carried_inputs(_raw_asec_predictor_frame())
    spm_flags = derived.table("spm_unit").set_index("spm_unit_id")
    person_flags = derived.table("person").set_index("person_id")
    variables = CountryTaxBenefitSystem().variables
    expected_entities = {
        "receives_tanf": "spm_unit",
        "receives_snap": "spm_unit",
        "receives_wic": "person",
    }
    for name, entity in expected_entities.items():
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == entity
        assert variable.value_type is bool
        assert str(variable.definition_period).lower() == "month"

    situation = {
        "people": {
            f"person_{person_id}": {
                "age": {"2024": int(person_flags.loc[person_id, "age"])},
                "receives_wic": {
                    "2024": bool(person_flags.loc[person_id, "receives_wic"])
                },
            }
            for person_id in person_flags.index
        },
        "spm_units": {
            "unit_100": {
                "members": ["person_1", "person_2"],
                "receives_tanf": {"2024": bool(spm_flags.loc[100, "receives_tanf"])},
                "receives_snap": {"2024": bool(spm_flags.loc[100, "receives_snap"])},
            },
            "unit_200": {
                "members": ["person_3"],
                "receives_tanf": {"2024": bool(spm_flags.loc[200, "receives_tanf"])},
                "receives_snap": {"2024": bool(spm_flags.loc[200, "receives_snap"])},
            },
        },
        "households": {
            "household": {
                "members": ["person_1", "person_2", "person_3"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    simulation = Simulation(situation=situation)

    np.testing.assert_array_equal(
        simulation.calculate("receives_tanf", period),
        spm_flags["receives_tanf"].to_numpy(dtype=bool),
    )
    np.testing.assert_array_equal(
        simulation.calculate("receives_snap", period),
        spm_flags["receives_snap"].to_numpy(dtype=bool),
    )
    np.testing.assert_array_equal(
        simulation.calculate("receives_wic", period),
        person_flags["receives_wic"].to_numpy(dtype=bool),
    )
