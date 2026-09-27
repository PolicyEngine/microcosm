# ruff: noqa: F401
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.hmrc_calibration import (
    HMRC_ASSESSABLE_INCOME_COLUMN,
    HMRC_TAXPAYER_COLUMN,
    calibrate_uk_hmrc_income,
    materialize_uk_hmrc_calibration_frame,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.spi_support import (
    SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind


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


def _feasible_dataset_and_targets() -> tuple[Frame, HMRCIncomeTargetSet]:
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
                "tax_free_savings_income": 0.0,
                SPI_HMRC_OTHER_INCOME_COLUMN: 0.0,
                SPI_HMRC_STATE_PENSION_INCOME_COLUMN: 0.0,
            }
            for leaf in SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS:
                row[leaf] = 0.0
            for income_component in HMRC_SPI_INCOME_COMPONENTS:
                if income_component != "state_pension":
                    row[income_component] = 0.0
            if component == "state_pension":
                row[SPI_HMRC_STATE_PENSION_INCOME_COLUMN] = value
            elif component == "employment_income":
                row[SPI_HMRC_PAY_COLUMN] = value
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
    dataset = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": np.ones(len(ids)),
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
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


def _with(
    frame: Frame,
    *,
    person: pd.DataFrame | None = None,
    household: pd.DataFrame | None = None,
    time_period: str | None = None,
) -> Frame:
    """Rebuild the national frame with selected tables replaced."""

    return uk_national_frame(
        person=frame.table("person") if person is None else person,
        benunit=frame.table("benunit"),
        household=frame.table("household") if household is None else household,
        time_period=uk_time_period(frame) if time_period is None else time_period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=(
            None
            if household is not None and "household_weight" in household
            else frame.weights_for("household").values
        ),
        mass_log=frame.mass_log,
    )


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
            }
        )


def _simulation_factory(dataset) -> _FakeSimulation:
    return _FakeSimulation(dataset)


__all__ = [name for name in globals() if not name.startswith("__")]
