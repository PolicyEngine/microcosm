"""UK-engine coverage for the CGT projection path."""

import pytest

from microcosm.build.uk_runtime.cgt_projection import (
    uk_cgt_projection,
    uk_cgt_projection_read_from_engine,
    uk_engine_installed,
)
from test_support.microcosm_build.uk_cgt_projection import manifest_entry


def test_engine_is_reported_available_with_the_uk_extra() -> None:
    assert uk_engine_installed()


def test_engine_projection_matches_the_published_growth_path() -> None:
    projection = uk_cgt_projection(2024, 2030)

    assert projection.engine.startswith("policyengine-uk==")
    assert uk_cgt_projection_read_from_engine(projection.engine)
    assert set(projection.exempt_amount_by_year.values()) == {3_000.0}
    # The OBR's October 2026 path (policyengine-uk 2.164.1, microcosm#1415).
    assert list(projection.yoy_growth_by_year.values()) == pytest.approx(
        [0.0462, 0.04, 0.031, 0.0344, 0.036, 0.0343], abs=5e-4
    )
    assert projection.cumulative_gains_factor_by_year["2030"] == pytest.approx(
        1.2434, abs=5e-4
    )
    pinned = manifest_entry().parameters
    for year, rate in projection.yoy_growth_by_year.items():
        assert rate == pytest.approx(
            float(pinned["expected_yoy_growth_by_year"][year]),
            abs=float(pinned["maximum_growth_drift"]),
        )
    # The path runs one year past the pinned horizon, to 2031, and later years
    # repeat the 2031 rate.
    beyond = uk_cgt_projection(2024, 2033)
    assert beyond.yoy_growth_by_year["2031"] != projection.yoy_growth_by_year["2030"]
    assert beyond.yoy_growth_by_year["2033"] == beyond.yoy_growth_by_year["2031"]
