"""Country-owned provider labels used by schema-7 diagnostics."""

from __future__ import annotations

import pytest

from microcosm.calibrate import (
    CALIBRATION_PROVIDER_LABELS_BY_COUNTRY,
    UK_CALIBRATION_PROVIDER_LABELS,
    US_CALIBRATION_PROVIDER_LABELS,
    calibration_provider_label,
)


def test_current_us_schema_7_sources_have_provider_labels() -> None:
    current_source_ids = {
        "bea_nipa",
        "cbo",
        "census_pep",
        "census_stc",
        "cms_aca",
        "cms_medicaid",
        "cms_medicare",
        "federal_reserve_z1",
        "hhs_acf_liheap",
        "hhs_acf_tanf",
        "irs_soi",
        "jct",
        "ssa_ssi_monthly",
        "ssa_supplement",
        "unspecified",
        "usda_snap",
    }

    assert current_source_ids <= US_CALIBRATION_PROVIDER_LABELS.keys()


def test_current_uk_schema_7_sources_have_provider_labels() -> None:
    current_source_ids = {
        "dwp",
        "hmrc",
        "isc",
        "obr",
        "ons",
        "scotgov",
        "slc",
        "voa",
    }

    assert current_source_ids == UK_CALIBRATION_PROVIDER_LABELS.keys()


def test_provider_label_resolution_is_country_specific() -> None:
    assert calibration_provider_label("us", "irs_soi") == ("IRS Statistics of Income")
    assert calibration_provider_label("UK", "obr") == (
        "Office for Budget Responsibility"
    )
    assert calibration_provider_label("be", "statbel") is None
    assert calibration_provider_label("uk", "irs_soi") is None


def test_provider_label_registries_are_immutable() -> None:
    with pytest.raises(TypeError):
        CALIBRATION_PROVIDER_LABELS_BY_COUNTRY["be"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        UK_CALIBRATION_PROVIDER_LABELS["new"] = "New"  # type: ignore[index]
