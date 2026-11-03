"""The CT-C tool reads the engine's council tax beside the stored bill (microcosm#1095)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.frame import MassChangeRecord, WeightKind
from tools.compare_uk_council_tax_liability import household_table


def _frame(**area_columns):
    household = pd.DataFrame(
        {
            "household_id": np.asarray([11, 12], dtype="int64"),
            "household_weight": [10.0, 20.0],
            # The household inputs the engine's yearly uprating reads.
            "region": ["SOUTH_EAST", "WALES"],
            "tenure_type": ["OWNED_OUTRIGHT", "RENT_PRIVATELY"],
            "rent": [0.0, 0.0],
            "council_tax_band": ["D", "B"],
            "council_tax": [2_100.0, 1_400.0],
            **area_columns,
        }
    )
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": np.asarray([1, 2], dtype="int64"),
                "person_benunit_id": np.asarray([101, 201], dtype="int64"),
                "person_household_id": np.asarray([11, 12], dtype="int64"),
                "age": [40.0, 40.0],
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


def test_the_table_holds_the_engine_liabilities_beside_the_stored_bill(
    tmp_path: Path,
) -> None:
    from policyengine_uk import Microsimulation

    path = write_uk_national_frame(
        _frame(
            local_authority=["MAIDSTONE", "CARDIFF"],
            la_code_oa=["E07000110", "W06000015"],
            household_source_id=np.asarray([7, 8], dtype="int64"),
        ),
        tmp_path / "dense.h5",
    )

    table = household_table(str(path), 2024).set_index("household_id")

    band_d = Microsimulation(dataset=str(path)).calculate(
        "council_tax_band_d_amount", 2024
    )
    # Maidstone's band D household pays its authority's Band D amount.
    assert band_d.values[0] > 0
    assert table.loc[11, "simulated_council_tax"] == pytest.approx(band_d.values[0])
    assert table.loc[11, "council_tax"] == pytest.approx(2_100.0)
    assert table.loc[12, "council_tax_band"] == "B"
    assert table.loc[12, "country"] == "WALES"
    assert table["la_code_oa"].tolist() == ["E07000110", "W06000015"]
    assert table["household_source_id"].tolist() == [7, 8]


def test_a_file_without_authorities_is_refused(tmp_path: Path) -> None:
    path = write_uk_national_frame(_frame(), tmp_path / "national.h5")

    with pytest.raises(ValueError, match="CT-C needs a dense release"):
        household_table(str(path), 2024)
