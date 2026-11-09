"""The UK country adapter compiles the canonical full graph on the engine."""

from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.country_adapter import build_uk_country_graph
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_support_payloads,
)


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
