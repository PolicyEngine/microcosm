"""The UK country adapter compiles the canonical full graph on the engine."""

from microcosm.build.uk_runtime.country_adapter import build_uk_country_graph
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release


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
