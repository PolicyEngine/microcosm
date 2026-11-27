"""policyengine-uk reads the flag the uc_housing_costs_claim stage stores (B1 of microcosm#1710).

The stage's rule is tested engine-free in
tests/engine_free/uk/test_uk_uc_housing_costs_claim.py. Here, on the locked
engine (2.180.0), the stage asks the engine for the housing costs element and
stores the flag; the element follows DWP's statement on a linked claim the
engine pays, and every other unit keeps it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import FRS_UC_HOUSING_ELEMENT_COLUMN
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_housing_costs_claim import (
    UC_HOUSING_COSTS_CLAIM,
    UC_HOUSING_COSTS_CLAIM_STAGE_NAME,
    UC_HOUSING_COSTS_ELEMENT,
    UKUCHousingCostsClaimStageTransform,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def _frame():
    # Four single adults, each their own benefit unit and household:
    # 1. a private renter whose linked statement carries no housing element;
    # 2. a private renter whose linked statement carries one;
    # 3. an owner whose linked statement carries none (the engine pays none);
    # 4. an unlinked private renter: it keeps the default.
    person = pd.DataFrame(
        {
            "person_id": [1001, 2001, 3001, 4001],
            "person_benunit_id": [101, 201, 301, 401],
            "person_household_id": [1, 2, 3, 4],
            "age": [40, 40, 40, 40],
            "is_benunit_head": [True] * 4,
            "is_household_head": [True] * 4,
            FRS_UC_HOUSING_ELEMENT_COLUMN: ["NONE", "PAID", "NONE", "NOT_LINKED"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["NORTH_WEST"] * 4,
            "household_source_id": [1, 2, 3, 4],
            # The true source each floor counts (microcosm#931).
            "source_household_id": [1, 2, 3, 4],
            "household_is_spi_synthetic": [False] * 4,
            # The engine adapter's dataset reads these on every household.
            "council_tax": [0.0] * 4,
            "tenure_type": [
                "RENT_PRIVATELY",
                "RENT_PRIVATELY",
                "OWNED_OUTRIGHT",
                "RENT_PRIVATELY",
            ],
            "rent": [8_000.0, 8_000.0, 0.0, 8_000.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [101, 201, 301, 401]}),
        household=household,
        household_weights=np.ones(4),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_the_engine_reads_the_stored_flag() -> None:
    engine = PolicyEngineUKEngine()
    stage = load_country_spec("uk").sources.stage_map()[
        UC_HOUSING_COSTS_CLAIM_STAGE_NAME
    ]
    frame = _frame()
    before = engine.materialize(frame, [UC_HOUSING_COSTS_ELEMENT], 2024)
    assert (np.asarray(before[UC_HOUSING_COSTS_ELEMENT]) > 0).tolist() == [
        True,
        True,
        False,
        True,
    ]

    stored = UKUCHousingCostsClaimStageTransform(stage=stage, engine=engine)(frame)
    assert stored.table("benunit")[UC_HOUSING_COSTS_CLAIM].tolist() == [
        False,
        True,
        True,
        True,
    ]

    after = engine.materialize(stored, [UC_HOUSING_COSTS_ELEMENT], 2024)
    assert (np.asarray(after[UC_HOUSING_COSTS_ELEMENT]) > 0).tolist() == [
        False,
        True,
        False,
        True,
    ]
    # Every unit the flag keeps is paid exactly what it was.
    kept = np.asarray([1, 2, 3])
    assert np.asarray(after[UC_HOUSING_COSTS_ELEMENT])[kept].tolist() == (
        np.asarray(before[UC_HOUSING_COSTS_ELEMENT])[kept].tolist()
    )
