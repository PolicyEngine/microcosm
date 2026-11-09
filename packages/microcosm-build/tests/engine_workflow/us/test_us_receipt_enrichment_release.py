"""Tests split from packages/microcosm-build/tests/test_us_receipt_enrichment_release.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_receipt_enrichment_release import *


def test_the_country_core_honours_all_three_supplied_receipts(lineage) -> None:
    __import__("policyengine_us")
    contract._check_native_input_precedence(
        _country(lineage), native_inputs=contract.RECEIPT_COLUMNS
    )
