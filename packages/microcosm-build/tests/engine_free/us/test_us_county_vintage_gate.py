"""Stored 2023 county metadata must remain complete and geographically valid."""

import numpy as np
import pytest

from microcosm.build.us_runtime.county_vintage import (
    county_fips_2023_assignment_failures,
)


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"blocks": ["090010701001000", "360610001001000"]},
        {"tracts": ["09001070100", "36061000100"]},
    ],
)
def test_valid_assignment_passes_with_or_without_source_geography(source):
    assert (
        county_fips_2023_assignment_failures(
            ["09120", np.str_("36061")], ["09001", "36061"], **source
        )
        == ()
    )


@pytest.mark.parametrize(
    "invalid", [None, np.nan, 9120, b"09120", "", "0912", "invalid"]
)
def test_gate_refuses_absent_or_non_text_counties(invalid):
    failures = county_fips_2023_assignment_failures([invalid], ["09001"])
    assert any("five-digit text" in failure for failure in failures)


@pytest.mark.parametrize("invalid", ["09001", "09100", "09200", "36061"])
def test_ct_rows_must_name_a_current_planning_region(invalid):
    failures = county_fips_2023_assignment_failures([invalid], ["09001"])
    assert any("planning region" in failure for failure in failures)


def test_non_ct_rows_must_preserve_the_legacy_county():
    failures = county_fips_2023_assignment_failures(["36001"], ["36061"])
    assert any("non-CT" in failure for failure in failures)


@pytest.mark.parametrize(
    "source", [{"blocks": ["090010701001000"]}, {"tracts": ["09001070100"]}]
)
def test_a_valid_but_wrong_ct_region_fails_the_source_join(source):
    failures = county_fips_2023_assignment_failures(["09190"], ["09001"], **source)
    assert any("disagree" in failure for failure in failures)


@pytest.mark.parametrize(
    "source", [{"blocks": ["090019999991000"]}, {"tracts": ["09001999999"]}]
)
def test_unknown_source_geography_becomes_a_gate_failure(source):
    failures = county_fips_2023_assignment_failures(["09120"], ["09001"], **source)
    assert any("cannot resolve" in failure for failure in failures)


@pytest.mark.parametrize("counties", ["09120", None, [["09120"]]])
def test_gate_refuses_non_vector_metadata(counties):
    assert county_fips_2023_assignment_failures(counties, ["09001"])


def test_gate_refuses_mismatched_vector_lengths():
    assert county_fips_2023_assignment_failures([], ["09001"])
    assert county_fips_2023_assignment_failures(["09120"], ["09001"], tracts=[])


def test_empty_assignment_passes():
    assert county_fips_2023_assignment_failures([], [], blocks=[]) == ()
