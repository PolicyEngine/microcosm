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


def test_stage_transform_rewrites_property_and_records_its_receipt() -> None:
    import numpy as np

    from microcosm.build.source_manifest import SourceStageSpec
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.build.uk_runtime.regional_uprating import (
        UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON,
        UKRegionalPropertyUpratingStageTransform,
    )
    from microcosm.frame import WeightKind

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "SCOTLAND"],
                "main_residence_value": [100.0, 200.0],
                "property_wealth": [150.0, 300.0],
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.array([10.0, 30.0]),
    )
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "regional_property_uprating",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [{"kind": "uprate_to_regional_reference"}],
            "outputs": [],
            "rewrites": ["main_residence_value", "property_wealth"],
        }
    )
    transform = UKRegionalPropertyUpratingStageTransform(
        stage=stage,
        resource={
            "values": [
                {"region": "LONDON", "avg_house_price": 200.0, "dwellings": 1},
                {"region": "SCOTLAND", "avg_house_price": 600.0, "dwellings": 1},
            ]
        },
    )

    result = transform(frame)

    household = result.table("household")
    assert household["main_residence_value"].tolist() == pytest.approx([200.0, 600.0])
    assert result.weights_for("household").values.tolist() == [10.0, 30.0]
    receipt = result.mass_log[-1]
    assert receipt.reason == UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON
    assert receipt.old_total == receipt.new_total == 40.0
    assert receipt.declared_factor == 1.0
