from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.provenance_flags import (
    UK_HOUSEHOLD_PROVENANCE_FLAG_PREFIX,
    discover_household_provenance_flags,
    household_provenance_flag_summary,
)


def _frame(flag_values=(True, False, True)):
    person = pd.DataFrame(
        {
            "person_id": [101, 102, 103],
            "person_benunit_id": [201, 202, 203],
            "person_household_id": [1, 2, 3],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [201, 202, 203]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [1.0, 2.0, 3.0],
            "household_is_spi_synthetic": list(flag_values),
            "household_is_capital_gains_clone": [False, True, False],
            "other": [True, True, True],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def test_prefix_discovery_returns_sorted_household_flags() -> None:
    assert UK_HOUSEHOLD_PROVENANCE_FLAG_PREFIX == "household_is_"

    assert discover_household_provenance_flags(
        ["other", "household_is_z", "household_is_a"]
    ) == ("household_is_a", "household_is_z")


def test_weighted_summary_uses_household_weights() -> None:
    summary = household_provenance_flag_summary(_frame())["flags"]

    assert summary["household_is_spi_synthetic"]["count"] == 2
    assert summary["household_is_spi_synthetic"]["share"] == pytest.approx(2 / 3)
    assert summary["household_is_spi_synthetic"]["weighted_share"] == pytest.approx(
        4 / 6
    )


def test_non_bool_flag_is_refused() -> None:
    frame = _frame(flag_values=(1, 0, 1))

    with pytest.raises(ValueError, match="must be boolean"):
        household_provenance_flag_summary(frame)
