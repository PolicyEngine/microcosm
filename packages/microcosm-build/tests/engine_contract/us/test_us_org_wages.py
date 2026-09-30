"""Tests split from packages/microcosm-build/tests/test_us_org_wages.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_org_wages import *


def test_live_2024_flsa_policy_values_when_us_extra_is_installed() -> None:
    __import__("policyengine_us")
    assert module._flsa_policy(2024) == pytest.approx(
        (107_432.0, 35_568.0, 57_470.4, 40.0, 1.5)
    )
