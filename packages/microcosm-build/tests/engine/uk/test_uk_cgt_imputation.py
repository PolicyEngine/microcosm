"""Tests split from packages/microcosm-build/tests/test_uk_cgt_imputation.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_cgt_imputation import *


class TestPolicyParameters:
    def test_reads_the_2023_values_from_the_parameter_tree(self) -> None:
        from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_policy_parameters

        parameters = uk_cgt_policy_parameters(2023)

        assert parameters.personal_allowance == 12_570.0
        assert parameters.personal_allowance_taper_threshold == 100_000.0
        assert parameters.personal_allowance_taper_rate == 0.5
        assert parameters.annual_exempt_amount == 6_000.0
        assert parameters.instant == "2023-06-01"
