"""Concepts round-trip through a real policyengine-us Microsimulation.

The encoded person and household inputs load into an actual engine
simulation, are read back, and decode to the original concepts. The engine
stores amounts as float32, so amounts survive to about seven significant
digits.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame import US_SCHEMA
from microcosm.frame.adapters.policyengine_us import (
    POLICYENGINE_US_CONCEPT_MAPPING,
    PolicyEngineUSEngine,
)
from test_support.microcosm_frame.concept_engine_frames import (
    assert_engine_round_trip,
)
from test_support.microcosm_frame.concept_frames import concept_frames, shares

MAPPING = POLICYENGINE_US_CONCEPT_MAPPING


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine(spm={"geography_kind": "national"})


@settings(
    max_examples=4,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(tables=concept_frames(min_households=2, max_households=4), data=st.data())
def test_concepts_round_trip_through_the_engine(engine, tables, data) -> None:
    assert_engine_round_trip(
        MAPPING,
        engine,
        tables,
        US_SCHEMA,
        2024,
        shares={name: data.draw(shares) for name in MAPPING.share_parameters()},
        take_up_rates={name: data.draw(shares) for name in MAPPING.take_up_programs()},
    )
