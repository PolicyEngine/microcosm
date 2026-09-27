"""Tests split from packages/microcosm-build/tests/test_uk_uc_family.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_family import *


@pytest.mark.parametrize("year", [2024, 2025])
def test_released_engine_single_allowance_is_not_misclassified_by_float_precision(year):
    """Use the engine's processed year rates and real float32 allowance output."""
    from policyengine_uk import Simulation

    from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask

    frame, _ = _fixture()
    person, benunit = frame.table("person"), frame.table("benunit")
    claimants = frs_uc_claimant_mask(person, benunit)
    people = {
        str(int(row.person_id)): {
            "age": {str(year): int(row.age)},
            "is_uc_claimant": {str(year): bool(claimants[i])},
        }
        for i, row in enumerate(person.itertuples(index=False))
    }
    units = {
        str(int(bu)): {
            "members": [
                str(int(pid))
                for pid in person.loc[person.person_benunit_id == bu, "person_id"]
            ]
        }
        for bu in benunit.benunit_id
    }
    sim = Simulation(
        situation={
            "people": people,
            "benunits": units,
            "households": {
                key: {"members": unit["members"]} for key, unit in units.items()
            },
        }
    )
    values, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_administrative_family_type", year
    )
    assert values.tolist() == [
        "LONE_PARENT",
        "COUPLE_NO_CHILDREN",
        "LONE_PARENT",
        "SINGLE",
        "COUPLE_WITH_CHILDREN",
    ]
