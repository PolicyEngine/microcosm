from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_employment import derive_frs_employment


def test_employment_maps_status_sector_and_sic() -> None:
    person = pd.DataFrame({"person_id": [1, 2, 3, 4]})
    adult = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "empstati": [0, 10, 11, np.nan],
            "mjobsect": [0, 1, 2, np.nan],
            "sic": [-5, 84.9, np.nan, 7],
        }
    )

    result = derive_frs_employment(person, adult)

    assert result["employment_status"].tolist() == [
        "CHILD",
        "SHORT_TERM_DISABLED",
        "LONG_TERM_DISABLED",
        "CHILD",
    ]
    assert result["employment_sector"].tolist() == [
        "NOT_EMPLOYED",
        "PRIVATE",
        "PUBLIC",
        "NOT_EMPLOYED",
    ]
    assert result["sic_industry_division"].tolist() == [0, 84, 0, 7]


@pytest.mark.parametrize("missing", ["mjobsect", "sic"])
def test_employment_missing_fail_loud_columns_raise(missing: str) -> None:
    person = pd.DataFrame({"person_id": [1]})
    adult = pd.DataFrame(
        {"person_id": [1], "empstati": [1], "mjobsect": [1], "sic": [10]}
    ).drop(columns=[missing])

    with pytest.raises(KeyError):
        derive_frs_employment(person, adult)
