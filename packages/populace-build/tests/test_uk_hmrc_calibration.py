from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime.hmrc_calibration import (
    HMRC_ASSESSABLE_INCOME_COLUMN,
    HMRC_TAXPAYER_COLUMN,
    calibrate_uk_hmrc_income,
    materialize_uk_hmrc_calibration_frame,
)
from populace.build.uk_runtime.hmrc_income import (
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
)
from populace.build.uk_runtime.national_build import UKNationalDataset
from populace.frame import MassChangeRecord, WeightKind


def _name(
    component: str,
    measure: str,
    lower: int,
    upper: int | None,
) -> str:
    return (
        f"hmrc/{component}_{measure}_income_band_{lower}_to_"
        f"{'inf' if upper is None else upper}"
    )


def _feasible_dataset_and_targets() -> tuple[UKNationalDataset, HMRCIncomeTargetSet]:
    rows: list[dict[str, object]] = []
    records: list[HMRCIncomeBandTargetRecord] = []
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    row_id = 1
    for lower, upper in zip(
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
        upper_bounds,
        strict=True,
    ):
        value = float(lower + 1)
        for component in HMRC_SPI_INCOME_COMPONENTS:
            row: dict[str, object] = {
                "person_id": row_id,
                "person_household_id": row_id,
                "person_benunit_id": row_id,
                "state_pension_reported": 0.0,
            }
            for income_component in HMRC_SPI_INCOME_COMPONENTS:
                if income_component != "state_pension":
                    row[income_component] = 0.0
            if component == "state_pension":
                row["state_pension_reported"] = value
            else:
                row[component] = value
            rows.append(row)
            for measure, target_value, unit in (
                ("count", 1.0, "people"),
                ("amount", value, "GBP"),
            ):
                records.append(
                    HMRCIncomeBandTargetRecord(
                        name=_name(component, measure, lower, upper),
                        component=component,
                        measure=measure,
                        unit=unit,
                        value=target_value,
                        period="2023",
                        total_income_lower_bound=lower,
                        total_income_upper_bound=upper,
                    )
                )
            row_id += 1

    person = pd.DataFrame(rows)
    ids = np.arange(1, len(person) + 1, dtype="int64")
    dataset = UKNationalDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": np.ones(len(ids)),
            }
        ),
        time_period="2023",
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=float(len(ids)),
                new_total=float(len(ids)),
                declared_factor=1.0,
                reason="test reviewed SPI prior",
            ),
        ),
    )
    source = HMRCIncomeSourceProvenance(
        local_path=Path("/tmp/test-hmrc.ods"),
        sha256="a" * 64,
        publication_url="https://www.gov.uk/test",
        ods_url="https://assets.publishing.service.gov.uk/test.ods",
        source_vintage="2023-24",
        source_tax_year="2023-24",
        source_tax_year_start=2023,
        build_period="2023",
        table_names=("Table_3_6", "Table_3_7"),
    )
    return dataset, HMRCIncomeTargetSet(source=source, targets=tuple(records))


class _FakeSimulation:
    calls: list[dict[str, object]] = []

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def calculate_dataframe(
        self,
        variables: list[str],
        *,
        period: str,
        map_to: str,
        use_weights: bool,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "variables": variables,
                "period": period,
                "map_to": map_to,
                "use_weights": use_weights,
            }
        )
        person = self.dataset.person
        return pd.DataFrame(
            {
                "person_id": person["person_id"],
                "income_tax": np.ones(len(person)),
                "state_pension": person["state_pension_reported"],
            }
        )


def _simulation_factory(dataset) -> _FakeSimulation:
    return _FakeSimulation(dataset)


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
            "variables": ["person_id", "income_tax", "state_pension"],
            "period": "2023",
            "map_to": "person",
            "use_weights": True,
        }
    ]

    specs = materialized.registry.specs
    assert {spec.metadata["taxpayer_mask"] for spec in specs} == {
        HMRC_TAXPAYER_COLUMN
    }
    assert {spec.metadata["band_measure"] for spec in specs} == {
        HMRC_ASSESSABLE_INCOME_COLUMN
    }
    problem = materialized.registry.to_target_set()
    assert len(problem) == 208


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
    broken = dataset.with_tables(person=person)

    with pytest.raises(ValueError, match="no strictly positive-mass support"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )


def test_materialization_requires_strictly_positive_household_prior() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    household = dataset.household.copy()
    household.loc[0, "household_weight"] = 0.0
    broken = dataset.with_tables(household=household)

    with pytest.raises(ValueError, match="every household prior weight"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )


def test_materialization_rejects_wrong_mapped_period() -> None:
    dataset, targets = _feasible_dataset_and_targets()
    broken = dataset.with_tables(time_period="2024")

    with pytest.raises(ValueError, match="does not match"):
        materialize_uk_hmrc_calibration_frame(
            broken,
            targets,
            simulation_factory=_simulation_factory,
        )
