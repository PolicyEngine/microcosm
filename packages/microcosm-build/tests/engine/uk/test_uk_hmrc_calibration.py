"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_calibration.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_calibration import *


def test_materializes_all_hmrc_targets_with_mapped_taxpayer_semantics() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    _FakeSimulation.calls = []

    materialized = materialize_uk_hmrc_calibration_frame(
        dataset,
        targets,
        simulation_factory=_simulation_factory,
    )

    assert len(materialized.registry) == HMRC_SPI_TARGET_RECORD_COUNT == 208
    assert materialized.frame.resolve_weights("household").kind is (
        WeightKind.IMPORTANCE
    )
    assert materialized.taxpayer_rows == len(dataset.person)
    assert materialized.minimum_positive_support_rows == 1
    assert _FakeSimulation.calls == [
        {
            "variables": [
                "person_id",
                "income_tax",
            ],
            "period": "2023",
            "map_to": "person",
            "use_weights": True,
        }
    ]

    specs = materialized.registry.specs
    assert {spec.metadata["taxpayer_mask"] for spec in specs} == {HMRC_TAXPAYER_COLUMN}
    assert {spec.metadata["band_measure"] for spec in specs} == {
        HMRC_ASSESSABLE_INCOME_COLUMN
    }
    problem = materialized.registry.to_target_set()
    assert len(problem) == 208


def test_state_pension_targets_use_the_same_srp_auxiliary_as_band_income() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    person = dataset.person.copy()
    person["state_pension_reported"] = 9_999_999.0
    dataset = _with(dataset, person=person)

    materialized = materialize_uk_hmrc_calibration_frame(
        dataset,
        targets,
        simulation_factory=_simulation_factory,
    )

    target = next(
        spec
        for spec in materialized.registry.specs
        if spec.metadata["component"] == "state_pension"
        and spec.metadata["measure"] == "amount"
        and spec.metadata["income_lower_bound"] == "12570"
    )
    measure = materialized.frame.table("person")[target.measure]
    assert float(measure.sum()) == 12_571.0


def test_exact_surface_calibrates_with_conserved_positive_weights() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    materialized = materialize_uk_hmrc_calibration_frame(
        dataset,
        targets,
        simulation_factory=_simulation_factory,
    )

    calibrated = calibrate_uk_hmrc_income(
        materialized,
        epochs=2,
        learning_rate=0.01,
        max_weight_ratio=5.0,
        maximum_abs_relative_error=1e-6,
        seed=3,
    )

    assert calibrated.result.frame.resolve_weights("household").kind is (
        WeightKind.CALIBRATED
    )
    assert calibrated.result.skipped == ()
    assert calibrated.maximum_abs_relative_error <= 1e-6
    assert calibrated.result.weights.sum() == pytest.approx(
        calibrated.result.initial_weights.sum()
    )
    assert (calibrated.result.weights > 0.0).all()


def test_materialization_fails_closed_when_one_component_has_no_band_support() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    person = dataset.person.copy()
    mask = person["other_investment_income"] > 0
    person.loc[mask, "other_investment_income"] = 0.0
    broken = _with(dataset, person=person)

    with pytest.raises(ValueError, match="no strictly positive-mass support"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )


def test_materialization_rejects_tax_free_interest_above_gross() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    person = dataset.person.copy()
    person.loc[0, "tax_free_savings_income"] = 1.0
    broken = _with(dataset, person=person)

    with pytest.raises(ValueError, match="gross savings_interest_income"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )
