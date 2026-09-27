"""Tests split from packages/microcosm-build/tests/test_uk_frs_disability.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_disability import *


def test_dwp_readers_share_fiscal_converted_tree() -> None:
    # The readers construct the real engine's parameter tree; the wheel gate
    # and the us-extra CI lane run without policyengine-uk, so skip there.
    from microcosm.build.uk_runtime.frs_disability import (
        uk_dwp_disability_flag_rates,
    )

    category_2024 = uk_dwp_disability_category_rates(2024)
    flags_2024 = uk_dwp_disability_flag_rates(2024)
    category_2023 = uk_dwp_disability_category_rates(2023)
    flags_2023 = uk_dwp_disability_flag_rates(2023)

    assert category_2024.instant == flags_2024.instant == "2024-01-01"
    assert np.isfinite(category_2024.aa_lower)
    assert category_2024.aa_higher == flags_2024.aa_higher == pytest.approx(108.55)
    assert category_2023.aa_higher == flags_2023.aa_higher == pytest.approx(101.75)
