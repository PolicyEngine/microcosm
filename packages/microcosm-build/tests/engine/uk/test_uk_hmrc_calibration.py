"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_calibration.py."""

# ruff: noqa: F403, F405
from microcosm.build.uk_runtime.hmrc_calibration import (
    HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN,
)
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


def test_materialization_rejects_negative_interest_outside_isas() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    person = dataset.person.copy()
    person.loc[0, "savings_interest_income"] = -1.0
    broken = _with(dataset, person=person)

    with pytest.raises(ValueError, match="negative"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )


def test_materialization_counts_interest_outside_isas_once_in_either_layout() -> None:
    """Table 3.7 and its bands read non-ISA interest once (microcosm#1360).

    The producer stores it; a legacy gross frame stores gross interest with the
    ISA part in tax_free_savings_income, taken off once.
    """

    from microcosm.frame.adapters.policyengine_uk import (
        UK_LEGACY_ISA_INCOME_CONTRACT,
    )

    dataset, targets = _feasible_dataset_and_targets()
    person = dataset.person.copy()
    person["individual_savings_account_interest_income"] = 25.0
    additive = _with(dataset, person=person)
    legacy_person = person.drop(columns="individual_savings_account_interest_income")
    legacy_person["savings_interest_income"] = person["savings_interest_income"] + 25.0
    legacy_person["tax_free_savings_income"] = 25.0
    legacy = _with(
        dataset,
        person=legacy_person,
        isa_income_contract=UK_LEGACY_ISA_INCOME_CONTRACT,
    )

    readings = [
        materialize_uk_hmrc_calibration_frame(
            frame, targets, simulation_factory=_simulation_factory
        )
        for frame in (additive, legacy)
    ]
    columns = (HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN, "hmrc_spi_assessable_income")
    tables = [reading.frame.table("person") for reading in readings]
    for column in columns:
        np.testing.assert_allclose(
            tables[0][column].to_numpy(dtype=float),
            tables[1][column].to_numpy(dtype=float),
        )
    assert readings[0].taxpayer_rows == readings[1].taxpayer_rows

    broken_person = legacy_person.copy()
    broken_person.loc[0, "tax_free_savings_income"] = (
        broken_person.loc[0, "savings_interest_income"] + 1.0
    )
    broken = _with(
        dataset,
        person=broken_person,
        isa_income_contract=UK_LEGACY_ISA_INCOME_CONTRACT,
    )
    with pytest.raises(ValueError, match="negative"):
        materialize_uk_hmrc_calibration_frame(
            broken, targets, simulation_factory=_simulation_factory
        )
