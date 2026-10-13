"""Tests split from packages/microcosm-build/tests/test_uk_take_up_contract.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_take_up_contract import *


def test_engine_version_is_read_dynamically_and_tie_break_absent() -> None:
    import policyengine_uk

    package_version = metadata.version("policyengine-uk")
    assert package_version
    assert getattr(policyengine_uk, "__version__", None) is None
    assert (
        "higher_earner_tie_break"
        not in policyengine_uk.CountryTaxBenefitSystem().variables
    )
