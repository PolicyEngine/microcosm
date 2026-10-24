"""The engine's tax_band is what the marginal-rate targets measure (#1123).

The national ITL Table 2.2 rows and the SPI area cells count taxpayers by
the highest rate they pay; their bindings filter on policyengine-uk's
``tax_band``. These pin what that variable returns for the cases the targets
depend on, on the installed engine at the 2025 calibration period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_annual_income import (
    ANNUAL_SHARE_COLUMNS,
    OTHER_SPELLS_COLUMNS,
)
from microcosm.build.uk_runtime.local_targets import (
    HIGHER_RATE_TAXPAYERS_METRIC,
    compute_household_metrics,
)
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

PERIOD = 2025


def _simulation(people: dict[str, dict[str, float | str]], country: str):
    from policyengine_uk import Simulation

    members = list(people)
    return Simulation(
        situation={
            "people": {
                name: {key: {str(PERIOD): value} for key, value in fields.items()}
                for name, fields in people.items()
            },
            "benunits": {"benunit": {"members": members}},
            "households": {
                "household": {
                    "members": members,
                    "country": {str(PERIOD): country},
                }
            },
        }
    )


def _bands(people: dict[str, dict[str, float | str]], country: str) -> list[str]:
    result = _simulation(people, country).calculate("tax_band", PERIOD)
    decoded = result.decode_to_str() if hasattr(result, "decode_to_str") else result
    return [str(value) for value in decoded]


@pytest.mark.parametrize(
    ("employment_income", "band"),
    [
        (30_000.0, "BASIC"),
        (60_000.0, "HIGHER"),
        (200_000.0, "ADDITIONAL"),
    ],
)
def test_a_rest_of_uk_taxpayer_is_classified_on_the_uk_thresholds(
    employment_income, band
):
    people = {"adult": {"age": 40, "employment_income": employment_income}}
    assert _bands(people, "ENGLAND") == [band]


@pytest.mark.parametrize(
    ("employment_income", "band"),
    [
        # Scottish higher rate from 43,663; advanced from 75,001; top from
        # 125,141. HMRC's Table 2.2 puts advanced and top under "additional"
        # from 2024-25, as the engine does.
        (50_000.0, "HIGHER"),
        (80_000.0, "ADDITIONAL"),
        (200_000.0, "ADDITIONAL"),
    ],
)
def test_a_scottish_taxpayer_is_classified_on_the_scottish_thresholds(
    employment_income, band
):
    people = {"adult": {"age": 40, "employment_income": employment_income}}
    assert _bands(people, "SCOTLAND") == [band]


def test_scottish_savings_and_dividends_count_against_the_scottish_thresholds():
    """A known gap against HMRC's classification, pinned so a fix shows.

    Scottish rates apply to non-savings, non-dividend income only: earnings of
    40,000 sit in the intermediate band, and 5,000 of dividends on top are
    taxed at the UK ordinary rate (the total is under the UK higher-rate
    threshold of 50,270), so HMRC counts no higher rate taxpayer. The engine
    compares total income with the Scottish thresholds and returns HIGHER.
    The same person in England is BASIC. Only Scotland's two national rows
    (hmrc.itl.higher_rate_taxpayers and additional) carry this; the Scottish
    area cells are unbound.
    """

    people = {
        "adult": {"age": 40, "employment_income": 40_000.0, "dividend_income": 5_000.0}
    }
    assert _bands(people, "SCOTLAND") == ["HIGHER"]
    assert _bands(people, "ENGLAND") == ["BASIC"]


def test_the_area_metric_counts_higher_rate_taxpayers_only(tmp_path):
    # A basic, a higher and an additional rate taxpayer sharing one English
    # household, each a single adult's benefit unit.
    ids = np.arange(1, 4)
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": [1, 1, 1],
                "age": [40.0, 41.0, 42.0],
                "employment_income": [30_000.0, 60_000.0, 200_000.0],
                "is_benunit_head": [True, True, True],
                "is_parent": [False, False, False],
                # Each paid all year at today's rate: the rows read the tax
                # band of each person's year, here the current band.
                **{column: [1.0] * 3 for column in ANNUAL_SHARE_COLUMNS},
                **{column: [0.0] * 3 for column in OTHER_SPELLS_COLUMNS},
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids, "dependent_children": [0, 0, 0]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [1.0],
                "region": ["NORTH_WEST"],
                "tenure_type": ["OWNED_OUTRIGHT"],
                "main_residence_value": [250_000.0],
                "council_tax": [0.0],
                "rent": [0.0],
            }
        ),
        time_period=str(PERIOD),
        weight_kind=WeightKind.DESIGN,
    )
    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=PERIOD
    )
    for area_type in ("constituency", "la"):
        metrics = compute_household_metrics(resolver, area_type, period=PERIOD)
        assert metrics[HIGHER_RATE_TAXPAYERS_METRIC].tolist() == [1.0]
