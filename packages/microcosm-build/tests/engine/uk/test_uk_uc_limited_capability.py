"""policyengine-uk reads the flags the uc_limited_capability stage stores (microcosm#1095, Y1).

The stage's rule is tested engine-free in
tests/engine_free/uk/test_uk_uc_limited_capability.py. Here, on the locked
engine (2.177.4), the stage asks the engine for its proxy and stores both
flags; unlinked people keep the engine's own answer exactly, and the LCWRA
element follows DWP's status on linked claims, in the survey year and in the
calibration year the engine projects the stored values to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import FRS_UC_LCW_STATUS_COLUMN
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_limited_capability import (
    UC_LCW,
    UC_LCWRA,
    UC_LIMITED_CAPABILITY_STAGE_NAME,
    UKUCLimitedCapabilityStageTransform,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def _frame():
    # Three single adults, each their own benefit unit and household:
    # 1. a linked LCWRA claim with no disability benefit (the proxy misses it);
    # 2. a linked NONE claim with a disability benefit (the proxy flags it);
    # 3. an unlinked claim with a disability benefit (the proxy stands).
    person = pd.DataFrame(
        {
            "person_id": [1001, 2001, 3001],
            "person_benunit_id": [101, 201, 301],
            "person_household_id": [1, 2, 3],
            "age": [40, 40, 40],
            "is_benunit_head": [True, True, True],
            "is_household_head": [True, True, True],
            "is_disabled_for_benefits": [False, True, True],
            FRS_UC_LCW_STATUS_COLUMN: ["LCWRA", "NONE", "NOT_LINKED"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "region": ["WALES", "WALES", "WALES"],
            "household_source_id": [1, 2, 3],
            "household_is_spi_synthetic": [False, False, False],
            # The engine adapter's dataset reads these on every household.
            "council_tax": [0.0, 0.0, 0.0],
            "tenure_type": ["RENT_PRIVATELY"] * 3,
            "rent": [0.0, 0.0, 0.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [101, 201, 301]}),
        household=household,
        household_weights=np.ones(3),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_the_engine_reads_the_stored_flags_and_the_proxy_stands_elsewhere() -> None:
    engine = PolicyEngineUKEngine()
    stage = load_country_spec("uk").sources.stage_map()[
        UC_LIMITED_CAPABILITY_STAGE_NAME
    ]
    frame = _frame()
    proxy = engine.materialize(frame, [UC_LCWRA, UC_LCW], 2024)

    stored = UKUCLimitedCapabilityStageTransform(stage=stage, engine=engine)(frame)
    person = stored.table("person")

    # The unlinked claim keeps the engine's own answer exactly.
    assert bool(person[UC_LCWRA].iloc[2]) == bool(proxy[UC_LCWRA][2])
    assert bool(person[UC_LCW].iloc[2]) == bool(proxy[UC_LCW][2])
    # The linked claims take DWP's status.
    assert person[UC_LCWRA].tolist()[:2] == [True, False]
    assert person[UC_LCW].tolist()[:2] == [True, False]

    before = engine.materialize(frame, ["uc_LCWRA_element"], 2024)
    after = engine.materialize(stored, ["uc_LCWRA_element"], 2024)
    assert (np.asarray(before["uc_LCWRA_element"]) > 0).tolist() == [False, True, True]
    assert (np.asarray(after["uc_LCWRA_element"]) > 0).tolist() == [True, False, True]


def test_the_stored_flags_hold_in_the_calibration_year() -> None:
    # The release stores the flags, which policyengine-uk carries into every
    # projected year as it does any stored column; a dataset holding them for
    # 2024 and 2025 gives the element by DWP's status in both.
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKMultiYearDataset, UKSingleYearDataset

    engine = PolicyEngineUKEngine()
    stage = load_country_spec("uk").sources.stage_map()[
        UC_LIMITED_CAPABILITY_STAGE_NAME
    ]
    stored = UKUCLimitedCapabilityStageTransform(stage=stage, engine=engine)(_frame())
    person = stored.table("person")
    released = person[
        [
            "person_id",
            "person_benunit_id",
            "person_household_id",
            "age",
            "is_benunit_head",
            "is_household_head",
            "is_disabled_for_benefits",
            UC_LCWRA,
            UC_LCW,
        ]
    ]

    def single(year: int) -> UKSingleYearDataset:
        return UKSingleYearDataset(
            person=released.copy(),
            benunit=pd.DataFrame({"benunit_id": released["person_benunit_id"]}),
            household=pd.DataFrame(
                {
                    "household_id": released["person_household_id"],
                    "household_weight": np.ones(len(released)),
                }
            ),
            fiscal_year=year,
        )

    simulation = Microsimulation(
        dataset=UKMultiYearDataset(datasets=[single(2024), single(2025)])
    )
    for year in (2024, 2025):
        element = np.asarray(simulation.calculate("uc_LCWRA_element", year).values)
        assert (element > 0).tolist() == [True, False, True], year
