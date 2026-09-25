from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_council_tax import derive_council_tax
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def test_council_tax_imputes_raw_missing_from_raw_cells() -> None:
    household = pd.DataFrame({"household_id": [1, 2, 3, 4, 5, 6]})
    raw = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5, 6],
            "gvtregno": [12, 12, 1, 1, 2, 2],
            "ctband": [1, 1, np.nan, np.nan, 2, 2],
            "adulth": [1, 1, 1, 1, 2, 2],
            "ctannual": [1000.0, -1.0, 600.0, np.nan, -1.0, 0.0],
            # FRS 2024-25 shape: CSEWAMT is retired and empty; the Scottish
            # charge comes from CWATAMTD plus CSEWAMT1 at the household's own
            # discount factor CWATAMTD/CWATAMT1.
            "csewamt": [np.nan] * 6,
            "cwatamtd": [3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            "cwatamt1": [4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            "csewamt1": [5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = derive_council_tax(household, raw)

    scottish_tax = 1000.0 - (3.0 + 5.0 * 0.75) * WEEKS_IN_YEAR
    assert np.isclose(result.loc[1], scottish_tax)
    assert np.isclose(result.loc[2], scottish_tax)
    assert result.loc[4] == 600.0
    assert result.loc[5] == 0.0
    assert result.loc[6] == 0.0
