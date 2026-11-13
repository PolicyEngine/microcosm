"""The UK country adapter compiles the canonical full graph on the engine."""

from microcosm.build.uk_runtime.atomic_area_support import CONDITIONED_SOURCES
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
    # The dense line runs on the declared spine's version as it stands
    # (microcosm#931): its clone count is the spine's, and it composes no
    # area draw and binds no atomic-area support of its own. The graph's only
    # support sources are the spine's three conditioned supports, which the
    # spine's area stages alone read.
    assert built.config.n_clones == 1
    assert not any(kernel.startswith("geography.") for kernel in kernels)
    supports = {
        source.name
        for source in built.graph.sources
        if source.name.endswith("_support")
    }
    assert supports == set(CONDITIONED_SOURCES.values())
    readers = {node.id for node in built.graph.nodes if set(node.sources) & supports}
    assert readers == {"atomic_area_assign", "rfdr", "spi_copy_areas"}
