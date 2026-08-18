from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime.regional_uprating import (
    uprate_household_property_by_region,
)


def test_regional_property_uprating_scales_owners_only_and_skips_missing_regions():
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "region": ["LONDON", "LONDON", "SCOTLAND", "SCOTLAND", "NORTHERN_IRELAND"],
            "main_residence_value": [100.0, 0.0, 200.0, 400.0, 500.0],
            "property_wealth": [150.0, 10.0, 300.0, 600.0, 700.0],
        }
    )
    resource = {
        "values": [
            {"region": "LONDON", "avg_house_price": 200.0, "dwellings": 1},
            {"region": "SCOTLAND", "avg_house_price": 600.0, "dwellings": 1},
        ]
    }

    uprated = uprate_household_property_by_region(household, resource)

    assert uprated.loc[0, "main_residence_value"] == pytest.approx(200.0)
    assert uprated.loc[0, "property_wealth"] == pytest.approx(300.0)
    assert uprated.loc[1, "main_residence_value"] == 0.0
    assert uprated.loc[1, "property_wealth"] == 10.0
    assert uprated.loc[2, "main_residence_value"] == pytest.approx(400.0)
    assert uprated.loc[3, "main_residence_value"] == pytest.approx(800.0)
    assert uprated.loc[4, "main_residence_value"] == 500.0


def test_regional_property_uprating_requires_columns() -> None:
    with pytest.raises(KeyError, match="property-uprating"):
        uprate_household_property_by_region(pd.DataFrame({"region": ["LONDON"]}), {})
