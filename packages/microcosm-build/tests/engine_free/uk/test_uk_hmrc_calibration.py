"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_calibration.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_calibration import *


def test_materialization_requires_strictly_positive_household_prior() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    household = dataset.table("household").copy()
    household["household_weight"] = dataset.weights_for("household").values
    household.loc[0, "household_weight"] = 0.0
    broken = _with(dataset, household=household)

    with pytest.raises(ValueError, match="every household prior weight"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )


def test_materialization_rejects_wrong_mapped_period() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    broken = _with(dataset, time_period="2024")

    with pytest.raises(ValueError, match="does not match"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )
