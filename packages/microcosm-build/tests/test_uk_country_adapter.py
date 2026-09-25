"""The UK country adapter uses raw FRS sources and the canonical full graph."""

import shutil
from pathlib import Path

import pytest
from uk_atomic_support_fixtures import toy_support_payloads

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.country_adapter import (
    build_uk_country_graph,
    validate_uk_country_source_projection,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig


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
    source = Path(__file__).parents[1] / "src/microcosm/build/uk"
    destination = tmp_path / "uk"
    shutil.copytree(source, destination)
    path = destination / "spec/sources.yaml"
    text = path.read_text()
    first_sha = text.split("  sha256: ", 1)[1].splitlines()[0]
    path.write_text(text.replace(first_sha, "f" * 64, 1))
    spec = load_country_spec(destination)
    with pytest.raises(ValueError, match="raw-source pins differ"):
        validate_uk_country_source_projection(spec)


@pytest.mark.requires_uk
def test_country_adapter_compiles_the_same_full_graph_with_all_targets_default():
    definition = uk_atomic_assignment_definition(
        toy_support_payloads(), seed=UKFullBuildConfig.seed
    )
    built = build_uk_country_graph(atomic_geography_definition=definition)
    release = load_uk_frs_release()
    assert built.config.geography_levels is None
    assert built.config.calibration_year == release.calibration_year
    assert built.config.source_year == release.survey_year
    kernels = {node.kernel for node in built.graph.nodes}
    assert {"uk.create@1", "uk.full.target_compilation@1", "uk.full.dense@1"} <= kernels
    assert not any(
        "national_candidate" in source.name for source in built.graph.sources
    )
