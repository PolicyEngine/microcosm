"""The HMRC self-employment rows read the profit after losses brought forward.

The SPI channel keeps the tape's losses brought forward in ``loss_relief``
(microcosm#1095, uk-data#528), so the year's profit sits in
``self_employment_income``. HMRC's rows count the profit after that relief and
band on total income with it; the engine's ``total_income`` adds the profit
before the relief, so the band measure swaps the one for the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.measure_simulation import (
    UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF,
    UK_TOTAL_INCOME_AFTER_LOSS_RELIEF,
    UKMeasureResolver,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind


def _frame():
    ids = np.arange(3, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [45, 45, 45],
                "employment_income": [20_000.0, 0.0, 0.0],
                # Person 0 keeps a profit after its losses, person 1's losses
                # exceed its profit, person 2 has none.
                "self_employment_income": [10_000.0, 2_000.0, 5_000.0],
                "loss_relief": [3_000.0, 5_000.0, 0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [1.0, 1.0, 1.0],
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2025",
        weight_kind=WeightKind.DESIGN,
    )


def test_the_hmrc_measures_read_the_profit_after_loss_relief(tmp_path) -> None:
    resolver = UKMeasureResolver(
        simulation_source=None, frame=_frame(), scratch_dir=tmp_path, year=2025
    )
    for name in (
        UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF,
        UK_TOTAL_INCOME_AFTER_LOSS_RELIEF,
    ):
        assert resolver.knows("person", name)
        assert not resolver.knows("household", name)
        assert resolver.entity_for(name) == "person"

    after, route = resolver.compute(
        "person", UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF
    )
    banded, _ = resolver.compute("person", UK_TOTAL_INCOME_AFTER_LOSS_RELIEF)
    profit, _ = resolver.compute("person", "self_employment_income")
    total, _ = resolver.compute("person", "total_income")

    # At the frame's own period the profit after the relief is read from the
    # stored columns (microcosm#1095).
    np.testing.assert_allclose(after, [7_000.0, 0.0, 5_000.0])
    assert route.startswith("frame_columns:")
    # At any other period the engine computes both components there, as it
    # does every stored value (microcosm#1324).
    later = UKMeasureResolver(
        simulation_source=None, frame=_frame(), scratch_dir=tmp_path, year=2026
    )
    projected, route = later.compute(
        "person", UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF
    )
    assert route.startswith("engine_components:")
    later_profit, _ = later.compute("person", "self_employment_income")
    later_relief, _ = later.compute("person", "loss_relief")
    np.testing.assert_allclose(
        projected,
        np.maximum(np.asarray(later_profit) - np.asarray(later_relief), 0.0),
    )
    # The band reads total income with the profit after the relief in place of
    # the profit before it.
    np.testing.assert_allclose(banded, np.asarray(total) - np.asarray(profit) + after)
    np.testing.assert_allclose(np.asarray(total) - banded, [3_000.0, 2_000.0, 0.0])
