"""uc_take_up's LCWRA fit reads the element policyengine-uk pays (microcosm#1095, B3).

The stage's one materialization, on a copy where every unit claims, returns
``uc_LCWRA_element`` with the award; a unit is in the LCWRA fit where it is
positive. The engine pays the element for a claimant's limited capability for
work and work-related activity only: a child with the flag gives the award the
disabled child addition instead, so that unit is not an LCWRA unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_take_up import UC_TAKE_UP_ENTITLEMENT_VARIABLES
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def _frame():
    """Three claiming units with no income: a single adult with the LCWRA
    flag, one without it, and a lone parent whose child has it."""

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [1, 2, 3, 3],
            "person_household_id": [1, 2, 3, 3],
            "age": [40.0, 40.0, 35.0, 10.0],
            "uc_limited_capability_for_WRA": [True, False, False, True],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1, 2, 3], "would_claim_uc": [True] * 3})
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "region": ["WALES"] * 3,
            "council_tax": [0.0] * 3,
            "tenure_type": ["OWNED_OUTRIGHT"] * 3,
            "rent": [0.0] * 3,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(3),
        weight_kind=WeightKind.DESIGN,
        time_period="2025",
    )


def test_the_lcwra_units_are_those_the_engine_pays_the_element() -> None:
    materialized = PolicyEngineUKEngine().materialize(
        _frame(), list(UC_TAKE_UP_ENTITLEMENT_VARIABLES), 2025
    )

    assert set(UC_TAKE_UP_ENTITLEMENT_VARIABLES) <= set(materialized)
    award = np.asarray(materialized["universal_credit"], dtype=float)
    element = np.asarray(materialized["uc_LCWRA_element"], dtype=float)
    assert (award > 0).all()
    assert (element > 0).tolist() == [True, False, False]
    # The element is part of the award the unit would be paid.
    assert award[0] > award[1]
