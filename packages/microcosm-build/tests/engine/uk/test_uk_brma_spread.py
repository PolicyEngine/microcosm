"""Small UK-engine contract for BRMA-spread rate ordering."""

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.brma_spread import lha_rate_keys
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def test_engine_materializes_household_mean_lha_rates_for_each_regional_brma():
    # Two differently sized units in the London household make aggregation
    # across benefit units observable; Scotland has only one BRMA slot.
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 20, 20, 30],
            "person_household_id": [1, 1, 1, 2],
            "age": [40, 40, 39, 40],
            "is_benunit_head": [True, True, False, True],
            "is_parent": True,
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [10, 20, 30], "LHA_category": ["B", "C", "B"]}
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "SCOTLAND"],
            "brma": ["CENTRAL_LONDON", "LOTHIAN"],
            "tenure_type": "RENT_PRIVATELY",
            "rent": 12_000.0,
        }
    )

    def frame_for(table):
        return uk_national_frame(
            person=person,
            benunit=benunit,
            household=table,
            household_weights=[2.0, 3.0],
            time_period=2025,
        )

    engine = PolicyEngineUKEngine()
    frame = frame_for(household)
    original = frame.table("household").copy()
    brmas = {
        "LONDON": ["CENTRAL_LONDON", "INNER_EAST_LONDON"],
        "SCOTLAND": ["LOTHIAN"],
    }
    keys = lha_rate_keys(frame, engine=engine, brmas=brmas)
    variable = "uncapped_BRMA_LHA_rate"
    try:
        engine.variable_metadata(variable)
    except ValueError:
        variable = "BRMA_LHA_rate"
    assert variable not in engine.variables()
    for slot, london_brma in enumerate(brmas["LONDON"]):
        moved = household.assign(brma=[london_brma, "LOTHIAN"])
        rate = engine.materialize(frame_for(moved), [variable], 2025)[variable]
        assert np.isfinite(rate).all()
        assert rate[0] != rate[1]
        np.testing.assert_allclose(keys[0, slot], rate[:2].mean())
        if slot == 0:
            np.testing.assert_allclose(keys[1, slot], rate[2])
    assert keys[1, 1] == np.inf
    pd.testing.assert_frame_equal(frame.table("household"), original)
