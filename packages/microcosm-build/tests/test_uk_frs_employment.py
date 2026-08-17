from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_employment import derive_frs_employment


def test_employment_maps_status_sector_and_sic() -> None:
    person = pd.DataFrame({"person_id": [1, 2, 3, 4, 5]})
    adult = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "empstati": [0, 10, 11, np.nan, 12],
            "mjobsect": [0, 1, 2, np.nan, 1],
            "sic": [-5, 84.9, np.nan, 7, 20],
        }
    )

    result = derive_frs_employment(person, adult)

    assert result["employment_status"].tolist() == [
        "CHILD",
        "SHORT_TERM_DISABLED",
        "LONG_TERM_DISABLED",
        "CHILD",
        # Beyond-domain codes take the incumbent's post-map fillna, not the
        # CHILD default reserved for 0/NaN rows.
        "LONG_TERM_DISABLED",
    ]
    assert result["employment_sector"].tolist() == [
        "NOT_EMPLOYED",
        "PRIVATE",
        "PUBLIC",
        "NOT_EMPLOYED",
        "PRIVATE",
    ]
    assert result["sic_industry_division"].tolist() == [0, 84, 0, 7, 20]


@pytest.mark.parametrize("missing", ["empstati", "mjobsect", "sic"])
def test_employment_missing_fail_loud_columns_raise(missing: str) -> None:
    person = pd.DataFrame({"person_id": [1]})
    adult = pd.DataFrame(
        {"person_id": [1], "empstati": [1], "mjobsect": [1], "sic": [10]}
    ).drop(columns=[missing])

    with pytest.raises(KeyError):
        derive_frs_employment(person, adult)
