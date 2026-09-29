"""Country-owned labels for programmatically declared providers."""

from __future__ import annotations

import pytest

from microcosm.calibrate import (
    CALIBRATION_PROVIDER_LABELS_BY_COUNTRY,
    US_CALIBRATION_PROVIDER_LABELS,
    calibration_provider_label,
)


def test_current_us_providers_have_labels() -> None:
    current_source_ids = {
        "bea",
        "bea_nipa",
        "cbo",
        "census_acs",
        "census_pep",
        "census_stc",
        "cms_aca",
        "cms_medicaid",
        "cms_medicare",
        "cms_nhe",
        "federal_reserve",
        "federal_reserve_z1",
        "hhs_acf_liheap",
        "hhs_acf_tanf",
        "irs_soi",
        "jct",
        "ssa",
        "ssa_ssi_monthly",
        "ssa_supplement",
        "unspecified",
        "usda_snap",
    }

    assert current_source_ids <= US_CALIBRATION_PROVIDER_LABELS.keys()
    assert all(
        calibration_provider_label("us", provider_id)
        for provider_id in current_source_ids
    )


def test_every_programmatic_provider_has_a_non_empty_label() -> None:
    labels = CALIBRATION_PROVIDER_LABELS_BY_COUNTRY["us"]

    assert labels
    assert all(provider_id.strip() for provider_id in labels)
    assert all(label.strip() for label in labels.values())
    assert all(
        calibration_provider_label("us", provider_id) == label
        for provider_id, label in labels.items()
    )


def test_provider_label_resolution_is_country_specific() -> None:
    assert calibration_provider_label("us", "irs_soi") == ("IRS Statistics of Income")
    assert calibration_provider_label("UK", "obr") is None
    assert calibration_provider_label("be", "statbel") is None
    assert calibration_provider_label("uk", "irs_soi") is None


def test_provider_label_registries_are_immutable() -> None:
    with pytest.raises(TypeError):
        CALIBRATION_PROVIDER_LABELS_BY_COUNTRY["be"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        US_CALIBRATION_PROVIDER_LABELS["new"] = "New"  # type: ignore[index]
