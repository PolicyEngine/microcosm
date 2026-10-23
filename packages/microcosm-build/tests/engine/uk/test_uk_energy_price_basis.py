"""policyengine-uk reads the energy price period the UK writer declares.

From policyengine-uk 2.148.1 (PolicyEngine/policyengine-uk#2571) the engine
projects stored domestic energy spending from the price period a dataset
declares in its root ``energy_price_basis`` attribute, and keeps spending
without one at its stored prices in every later year
(PolicyEngine/policyengine-uk#2526).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import (
    UK_ENERGY_PRICE_BASIS_COLUMNS,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.frame import MassChangeRecord, WeightKind


def _frame():
    household = pd.DataFrame(
        {
            "household_id": np.asarray([11, 12], dtype="int64"),
            "household_weight": [10.0, 20.0],
            "electricity_consumption": [900.0, 1_100.0],
            "gas_consumption": [700.0, 0.0],
            "domestic_energy_consumption": [1_600.0, 1_100.0],
        }
    )
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": np.asarray([1, 2], dtype="int64"),
                "person_benunit_id": np.asarray([101, 201], dtype="int64"),
                "person_household_id": np.asarray([11, 12], dtype="int64"),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": np.asarray([101, 201], dtype="int64")}),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=30.0,
                new_total=30.0,
                declared_factor=1.0,
                reason="Toy reviewed record.",
            ),
        ),
    )


def test_the_declared_columns_are_the_ones_the_engine_projects_from_a_basis() -> None:
    from policyengine_uk.data.economic_assumptions import (
        DATASET_UPRATING_EXCLUSIONS,
    )

    assert set(UK_ENERGY_PRICE_BASIS_COLUMNS) == set(DATASET_UPRATING_EXCLUSIONS)


def test_policyengine_uk_reads_the_written_declaration(tmp_path: Path) -> None:
    from policyengine_uk.data.energy_price_basis import (
        EnergyPriceBasis,
        read_energy_price_basis,
    )

    path = write_uk_national_frame(_frame(), tmp_path / "energy.h5")
    assert read_energy_price_basis(path) == EnergyPriceBasis("calendar_year", 2024)
