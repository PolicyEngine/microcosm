"""The UK country adapter uses raw FRS sources and the canonical full graph."""

import shutil
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.country_adapter import (
    build_uk_country_graph,
    validate_uk_country_source_projection,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release


def test_country_raw_source_projection_is_current():
    spec = load_country_spec("uk")
    validate_uk_country_source_projection(spec)
    sources = spec.resolved_spec.resource("sources").domain.to_wire()["sources"]
    assert all(row["role"] == "frs_raw_table" for row in sources)
    assert all(row["loader"] == "kernel:build_uk_frs_spine" for row in sources)
    assert {"frs_adult", "frs_benefits", "frs_child", "frs_househol"} <= {
        row["id"] for row in sources
    }
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
    built = build_uk_country_graph()
    release = load_uk_frs_release()
    assert built.config.geography_levels is None
    assert built.config.calibration_year == release.calibration_year
    assert built.config.source_year == release.survey_year
    kernels = {node.kernel for node in built.graph.nodes}
    assert {"uk.create@1", "uk.full.target_compilation@1", "uk.full.dense@1"} <= kernels
    assert not any(
        "national_candidate" in source.name for source in built.graph.sources
    )
