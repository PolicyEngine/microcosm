"""policyengine-uk reads the new-claimant draw the uc_new_claimant_draw stage stores (microcosm#1095).

The stage's rules are tested engine-free in
tests/engine_free/uk/test_uk_uc_new_claimant_draw.py. Here, on the locked
engine (2.179.2), the stage reads the award with its LCWRA element at the
calibration year, orders the draw by claim recency, and the engine gives the
new-claimant rate in each rebalancing year to the units whose stored
survey-year draw falls below that year's share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import (
    FRS_UC_CLAIM_MONTHS_COLUMN,
    FRS_UC_CLAIM_MONTHS_NOT_LINKED,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_new_claimant_draw import (
    UC_NEW_CLAIMANT_DRAW_OUTPUT,
    UC_NEW_CLAIMANT_DRAW_STAGE_NAME,
    UKUCNewClaimantDrawStageTransform,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

#: Ten single renters on Universal Credit with the LCWRA element, and one
#: who claims nothing. Claim months: the most recent claim first.
_MONTHS = [0.0, 2.0, 5.0, 9.0, 14.0, 20.0, 30.0, 48.0, 70.0, 100.0]
_STATUS = "uc_receives_new_claimant_health_element"


def _frame() -> object:
    count = len(_MONTHS) + 1
    ids = np.arange(1, count + 1)
    person = pd.DataFrame(
        {
            "person_id": ids * 1000 + 1,
            "person_benunit_id": ids * 100 + 1,
            "person_household_id": ids,
            "age": [30] * count,
            "is_benunit_head": [True] * count,
            "is_household_head": [True] * count,
            "is_claimant_or_partner": [True] * count,
            "uc_limited_capability_for_WRA": [True] * count,
            "uc_limited_capability_for_work": [True] * count,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": ids * 100 + 1,
            "would_claim_uc": [True] * (count - 1) + [False],
            FRS_UC_CLAIM_MONTHS_COLUMN: [*_MONTHS, FRS_UC_CLAIM_MONTHS_NOT_LINKED],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": ["WALES"] * count,
            "household_source_id": ids,
            # The true source each floor counts (microcosm#931).
            "source_household_id": ids,
            "household_is_spi_synthetic": [False] * count,
            # The engine adapter's dataset reads these on every household.
            "council_tax": [0.0] * count,
            "tenure_type": ["RENT_PRIVATELY"] * count,
            "rent": [6_000.0] * count,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(count),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_the_engine_gives_the_lower_rate_to_the_most_recent_claims() -> None:
    from policyengine_uk.scenarios.uc_reform import REBALANCING_YEARS

    # The stage's population is read at the calibration year, before the
    # first rebalancing year, whose share is the first above zero.
    assert REBALANCING_YEARS[0] == 2026
    engine = PolicyEngineUKEngine()
    stage = load_country_spec("uk").sources.stage_map()[UC_NEW_CLAIMANT_DRAW_STAGE_NAME]
    stored = UKUCNewClaimantDrawStageTransform(stage=stage, engine=engine)(_frame())
    draw = stored.table("benunit")[UC_NEW_CLAIMANT_DRAW_OUTPUT].to_numpy()

    # The ten claimants take the midpoints of tenths, most recent first.
    np.testing.assert_allclose(draw[:10], np.arange(10) * 0.1 + 0.05)

    parameters = engine._tax_benefit_system().parameters
    share = parameters.gov.dwp.universal_credit.rebalancing.new_claimant_share
    for year in (2026, 2027, 2029):
        status = engine.materialize(stored, [_STATUS], year, dataset_period=2024)[
            _STATUS
        ]
        below = draw[:10] < float(share(year))
        assert below.any()
        # The units below the year's share, the most recent claims, get the
        # new-claimant rate, and the others the protected one.
        assert np.asarray(status)[:10].tolist() == below.tolist()


def test_without_the_draw_the_engine_keeps_its_own_seeded_draw() -> None:
    engine = PolicyEngineUKEngine()
    frame = _frame()
    stage = load_country_spec("uk").sources.stage_map()[UC_NEW_CLAIMANT_DRAW_STAGE_NAME]
    stored = UKUCNewClaimantDrawStageTransform(stage=stage, engine=engine)(frame)

    own = engine.materialize(frame, [_STATUS], 2029, dataset_period=2024)[_STATUS]
    ours = engine.materialize(stored, [_STATUS], 2029, dataset_period=2024)[_STATUS]

    # The engine's seeded draw (default_rng(43) in benefit-unit order) picks
    # by position, not by claim recency: at 2029's share it picks the second
    # and third units, and the stored draw the two most recent claims.
    assert np.asarray(own).tolist() != np.asarray(ours).tolist()
