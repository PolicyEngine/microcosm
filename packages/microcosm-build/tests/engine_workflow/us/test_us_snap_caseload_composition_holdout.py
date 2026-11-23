"""The SNAP caseload composition holdout through the release scorer (microcosm#647).

The report takes each SPM unit's state and weight from the engine: the unit's
own ``spm_unit_state_fips`` and the weights on each scored result. This test
writes a release H5 whose first household holds two SPM units of different
sizes and reads the report the release tool writes from it, on the locked
policyengine-us.
"""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_post_export_scoring import *

# (person, household, tax unit, SPM unit, family, age, wages, weekly hours).
# Household 1, in Texas (48), holds a one-person unit and a three-person unit.
# Household 2, in California (06), is one person far above the income limits.
_PEOPLE = (
    (1, 1, 10, 100, 1000, 70, 0.0, 0.0),
    (2, 1, 11, 101, 1001, 35, 12_000.0, 25.0),
    (3, 1, 11, 101, 1001, 8, 0.0, 0.0),
    (4, 1, 11, 101, 1001, 5, 0.0, 0.0),
    (5, 2, 20, 200, 2000, 40, 150_000.0, 40.0),
)
_HOUSEHOLD_WEIGHTS = (1_000.0, 2_000.0)


def _write_two_unit_household_h5(builder, directory: Path) -> Path:
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    person = pd.DataFrame(
        [
            {
                "person_id": person_id,
                "person_household_id": household_id,
                "person_tax_unit_id": tax_unit_id,
                "person_spm_unit_id": spm_unit_id,
                "person_family_id": family_id,
                "person_marital_unit_id": person_id * 10_000,
                "age": age,
                "employment_income_before_lsr": wages,
                "weekly_hours_worked_before_lsr": hours,
            }
            for (
                person_id,
                household_id,
                tax_unit_id,
                spm_unit_id,
                family_id,
                age,
                wages,
                hours,
            ) in _PEOPLE
        ]
    )
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "state_fips": [48, 6],
                    "county_fips": ["48201", "06037"],
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": [10, 11, 20]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": [100, 101, 200]}),
            "family": pd.DataFrame({"family_id": [1000, 1001, 2000]}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray(_HOUSEHOLD_WEIGHTS), WeightKind.CALIBRATED)},
    )
    path = directory / "populace_us_2024.h5"
    PolicyEngineUSEngine().write_dataset(frame, path, period=builder.PERIOD)
    return path


def test_each_unit_of_a_shared_household_carries_its_state_and_weight(
    builder, tmp_path
) -> None:
    """Both Texas units read Texas at the household's weight (microcosm#647).

    policyengine-core maps a household value to SPM units by sharing it out by
    head count: asked for ``state_fips`` or ``household_weight`` that way, it
    gives the one-person unit a quarter of the Texas household's value and the
    three-person unit three quarters, so codes 12 and 36 and weights 250 and
    750. The report would then show no Texas row, 1,000 eligible units and
    2,500 eligible persons. It asks for the unit's own state and reads the
    weights on the result instead, which this test pins on the locked engine.
    """
    path = _write_two_unit_household_h5(builder, tmp_path)
    scorer = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=1
    )
    try:
        plan = builder._record_post_export_baseline_plan(
            builder._snap_caseload_composition_holdout_consumer
        )
        simulation = scorer.open_consumer(
            "snap_caseload_composition_holdout", plan
        ).simulate(None)
        state = simulation.calculate("spm_unit_state_fips", builder.PERIOD)
        size = simulation.calculate("snap_unit_size", builder.PERIOD)
        np.testing.assert_array_equal(np.asarray(state), [48, 48, 6])
        np.testing.assert_array_equal(np.asarray(size), [1, 3, 1])
        for result in (state, size):
            np.testing.assert_array_equal(result.weights, [1_000, 1_000, 2_000])
    finally:
        scorer.close()

    builder._write_snap_caseload_composition_holdout(
        release_dir=tmp_path,
        dataset_path=path,
        release_id="fixture",
        maximum_microsim_batch_size=1,
    )
    payload = json.loads(
        (tmp_path / "us_snap_caseload_composition_holdout.json").read_text()
    )

    assert payload["status"] == "available", payload.get("error")
    states = {row["state_fips"]: row for row in payload["states"]}
    assert list(states) == ["06", "48"]
    texas = states["48"]["take_up"]
    assert texas["eligible_records"] == 2
    assert texas["eligible_weighted_units"] == 2_000
    assert texas["eligible_weighted_persons"] == 4_000
    assert states["06"]["take_up"]["eligible_weighted_units"] == 0
    assert payload["national"]["take_up"]["eligible_weighted_units"] == 2_000
