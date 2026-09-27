"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_uprating import *


def test_pinned_engine_factors_are_the_ones_the_assessment_measured() -> None:
    def factor(path: str) -> float:
        return engine_parameter_value(path, "2025-01-01") / engine_parameter_value(
            path, "2023-01-01"
        )

    assert factor(
        "gov.economic_assumptions.indices.obr.average_earnings"
    ) == pytest.approx(1.1067, abs=2e-3)
    assert factor(
        "gov.economic_assumptions.indices.obr.per_capita.mixed_income"
    ) == pytest.approx(1.0344, abs=2e-3)
    assert factor("gov.dwp.state_pension.new_state_pension.amount") == pytest.approx(
        230.25 / 203.85, abs=1e-6
    )
