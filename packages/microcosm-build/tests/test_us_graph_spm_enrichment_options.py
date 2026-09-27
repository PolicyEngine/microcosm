"""Fail-closed SPM host configuration without running an enrichment fit."""

import json
from dataclasses import replace

import pytest

from microcosm.build.us_runtime import current_survey_spm_source as source
from microcosm.build.us_runtime import graph_us_survey_enrichment as graph


def profile():
    return source.acs.ACSAnalysisProfile(2024, "acs_2024_1yr", True, True)


def test_spm_disabled_requires_all_options_absent():
    assert graph._spm_configuration(None, None, None) is None


@pytest.mark.parametrize("placeholder", [False, True])
@pytest.mark.parametrize("policy", [None, source.ASEC_2025_INCOME_2024_SPM_POLICY])
def test_spm_configuration_preserves_explicit_scope_and_representation(
    placeholder, policy
):
    configured = json.loads(graph._spm_configuration(profile(), policy, placeholder))
    assert configured["outside_role_placeholder"] is placeholder
    assert configured["acs_profile"]["year"] == 2024
    assert (configured["asec_scope_policy"] is None) == (policy is None)


@pytest.mark.parametrize(
    "options",
    [
        (None, None, False),
        (None, source.ASEC_2025_INCOME_2024_SPM_POLICY, None),
        (None, source.ASEC_2025_INCOME_2024_SPM_POLICY, False),
        (profile(), None, None),
        (profile(), None, 0),
        (profile(), None, 1),
        ({"year": 2024}, None, False),
        (profile(), "include_all", False),
        (profile(), replace(source.ASEC_2025_INCOME_2024_SPM_POLICY), False),
    ],
)
def test_spm_partial_or_untyped_configuration_refuses(options):
    with pytest.raises(ValueError, match="SPM_"):
        graph._spm_configuration(*options)


def test_spm_configuration_rechecks_mutated_annual_profile():
    changed = profile()
    object.__setattr__(changed, "year", 2025)
    with pytest.raises(ValueError, match="ANNUAL_YEAR"):
        graph._spm_configuration(changed, None, False)
