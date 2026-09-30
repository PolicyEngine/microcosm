"""The UK country adapter uses raw FRS sources and the canonical full graph."""

import shutil

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.country_adapter import (
    validate_uk_country_source_projection,
)
from test_support.paths import paths_for


def test_country_raw_source_projection_is_current():
    spec = load_country_spec("uk")
    validate_uk_country_source_projection(spec)
    sources = spec.resolved_spec.resource("sources").domain.to_wire()["sources"]
    frs = [row for row in sources if row["role"] == "frs_raw_table"]
    assert all(row["loader"] == "kernel:build_uk_frs_spine" for row in frs)
    assert {"frs_adult", "frs_benefits", "frs_child", "frs_househol"} <= {
        row["id"] for row in frs
    }
    # The three atomic-area supports (microcosm#932) sit beside the FRS tables.
    supports = [row for row in sources if row["role"] != "frs_raw_table"]
    assert {row["id"] for row in supports} == {
        "uk_ew_output_area_2021_support",
        "uk_scotland_output_area_2022_support",
        "uk_ni_data_zone_2021_support",
    }
    assert all(
        row["role"] == "uk_atomic_area_support"
        and row["loader"] == "kernel:load_uk_atomic_area_support"
        for row in supports
    )
    assert "uk_national_candidate_2023" not in str(sources)


def test_country_source_projection_refuses_divergent_header_pin(tmp_path):
    source = paths_for("microcosm-build").package / "src/microcosm/build/uk"
    destination = tmp_path / "uk"
    shutil.copytree(source, destination)
    path = destination / "spec/sources.yaml"
    text = path.read_text()
    first_sha = text.split("  sha256: ", 1)[1].splitlines()[0]
    path.write_text(text.replace(first_sha, "f" * 64, 1))
    spec = load_country_spec(destination)
    with pytest.raises(ValueError, match="raw-source pins differ"):
        validate_uk_country_source_projection(spec)
