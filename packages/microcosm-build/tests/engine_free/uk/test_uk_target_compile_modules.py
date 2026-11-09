"""Both UK target kernels hash every module that decides a target value.

Two checks pin the closure, as for the transport compile kernels:

- the shared list's own: every first-party import of a module in
  ``UK_TARGET_COMPILE_MODULE_NAMES`` is listed, or excluded with a reason;
- each kernel's: the walk from every module a target or problem kernel
  hashes, its own module included, meets only modules the kernel hashes, the
  list's exclusions, or the imports recorded in ``UNCLASSIFIED_IMPORTS`` for
  that kernel's group, which no review has classified yet (microcosm#1497).
  That mapping may only shrink.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType

import pytest

import microcosm.calibrate.geography_constants as geography_constants_module
import microcosm.calibrate.solve as calibrate_solve
from microcosm.build import ledger_artifact
from microcosm.build import ledger_targets as shared_ledger_targets
from microcosm.build.uk_runtime import (
    graph_national,
    graph_targets,
    hmrc_uprating,
    measure_simulation,
    target_weights,
)
from microcosm.build.uk_runtime.target_compile_modules import (
    EXCLUDED_DIRECT_IMPORTS,
    UK_TARGET_COMPILE_MODULE_NAMES,
    uk_target_compile_modules,
)
from test_support.microcosm_build.first_party_imports import (
    HashInput,
    assert_an_edit_rekeys,
    first_party_imports,
    hashed_module_names,
    recorded_import_mismatches,
    unlisted_imports,
)


def test_every_direct_import_of_a_listed_module_is_listed_or_excluded():
    """Every first-party import, not only microcosm.build: the target
    hierarchies take their geography labels from
    microcosm.calibrate.geography_constants, which a scan of microcosm.build
    imports never saw (microcosm#1492)."""
    uncovered = unlisted_imports(
        UK_TARGET_COMPILE_MODULE_NAMES, frozenset(EXCLUDED_DIRECT_IMPORTS)
    )
    assert not uncovered, sorted(uncovered)


def test_exclusions_are_live_and_never_listed():
    imported = set().union(*map(first_party_imports, UK_TARGET_COMPILE_MODULE_NAMES))
    assert set(EXCLUDED_DIRECT_IMPORTS) <= imported
    assert not set(EXCLUDED_DIRECT_IMPORTS) & set(UK_TARGET_COMPILE_MODULE_NAMES)


def _hashed_by(kernel, monkeypatch, module) -> list:
    hashed: list = []
    monkeypatch.setattr(
        module, "source_hash", lambda *objects: hashed.extend(objects) or "test"
    )
    kernel.implementation_hash()
    return hashed


def test_full_target_kernel_hashes_the_shared_resolver_and_the_appliers(
    monkeypatch,
):
    hashed = _hashed_by(
        graph_targets.UKFullTargetCompilationKernel(), monkeypatch, graph_targets
    )
    assert shared_ledger_targets in hashed
    assert hmrc_uprating in hashed
    assert set(uk_target_compile_modules()) <= set(hashed)


def test_full_target_kernels_hash_the_measure_simulation(monkeypatch):
    """full_measure and full_targets resolve their measures in
    measure_simulation, the high-value homes region rows among them
    (microcosm#1301), so an edit there alone re-keys the measures node."""
    for kernel in (
        graph_targets.UKFullTargetCompilationKernel(),
        graph_targets.UKFullMeasureKernel(),
    ):
        assert measure_simulation in _hashed_by(kernel, monkeypatch, graph_targets)


def test_full_target_kernels_hash_the_fact_decoder(monkeypatch):
    """full_targets loads the ledger facts inside run through
    ledger_artifact's decoder, so an edit there alone re-keys every full-graph
    kernel built on this hash: compilation, selection and measure share it,
    and the problem kernel embeds it (microcosm#1382). The pin holds even if
    no listed module imports ledger_artifact any more."""
    for kernel in (
        graph_targets.UKFullTargetCompilationKernel(),
        graph_targets.UKFullTargetSelectionKernel(),
        graph_targets.UKFullMeasureKernel(),
        graph_targets.UKFullProblemKernel(),
    ):
        assert ledger_artifact in _hashed_by(kernel, monkeypatch, graph_targets)


def test_national_target_kernel_hashes_the_same_compile_modules(monkeypatch):
    hashed = _hashed_by(
        graph_national.UKNationalTargetKernel(), monkeypatch, graph_national
    )
    assert set(uk_target_compile_modules()) <= set(hashed)


def test_national_target_kernels_hash_the_measure_simulation(monkeypatch):
    """The national targets apply their measure exclusions, and the national
    problem resolves its measures, in measure_simulation, as the dense line's
    do (microcosm#1301), so an edit there alone re-keys both nodes."""
    for kernel in (
        graph_national.UKNationalTargetKernel(),
        graph_national.UKNationalProblemKernel(resolver_factory=None),
    ):
        assert measure_simulation in _hashed_by(kernel, monkeypatch, graph_national)


def test_national_target_kernels_hash_the_fact_decoder(monkeypatch):
    """The national targets load the ledger facts through full_targets and
    ledger_artifact's decoder too, so an edit there alone re-keys both
    kernels that share this hash (microcosm#1382)."""
    for kernel in (
        graph_national.UKNationalTargetKernel(),
        graph_national.UKNationalProblemKernel(resolver_factory=None),
    ):
        assert ledger_artifact in _hashed_by(kernel, monkeypatch, graph_national)


#: The kernels whose hash includes the UK target compile modules, with the
#: module whose ``source_hash`` each calls: the full graph's target,
#: selection, measures and problem kernels and the national graph's target
#: and problem kernels.
_TARGET_AND_PROBLEM_KERNELS = (
    (graph_targets.UKFullTargetCompilationKernel(), graph_targets),
    (graph_targets.UKFullTargetSelectionKernel(), graph_targets),
    (graph_targets.UKFullMeasureKernel(), graph_targets),
    (graph_targets.UKFullProblemKernel(), graph_targets),
    (graph_national.UKNationalTargetKernel(), graph_national),
    (graph_national.UKNationalProblemKernel(resolver_factory=None), graph_national),
)
_KERNELS_BY_REF = {
    kernel.ref: (kernel, module) for kernel, module in _TARGET_AND_PROBLEM_KERNELS
}
_FULL_GRAPH = tuple(
    ref for ref in _KERNELS_BY_REF if not ref.startswith("uk.full.national")
)
_NATIONAL_GRAPH = tuple(
    ref for ref in _KERNELS_BY_REF if ref.startswith("uk.full.national")
)

#: Code the target kernels ran without hashing it, found by the review of
#: microcosm#1492, and the kernels that now hash it. All four full-graph
#: kernels share one hash, so what one of them runs re-keys all four.
_RUN_MODULES = {
    # uk.full.target_selection@1 calls select_targets.
    "microcosm.calibrate.target_selection": _FULL_GRAPH,
    # The measures and problem kernels build and encode constraint matrices.
    "microcosm.calibrate.matrix": _FULL_GRAPH + _NATIONAL_GRAPH,
    "microcosm.calibrate.artifacts": _FULL_GRAPH + _NATIONAL_GRAPH,
    # The full problem's loss scales (default_target_loss_scales). The
    # national kernels call nothing in it (see the test below).
    "microcosm.calibrate.solve": ("uk.full.problem@1",),
    # full_targets' exclusion clock (exclusion_evaluation_date).
    "microcosm.build.uk_runtime.weighted_integrity": _FULL_GRAPH + _NATIONAL_GRAPH,
    # full_targets' surface provenance (_ledger_provenance) and band-edge
    # roster check (_validate_band_edge_registry), moved out of
    # calibration_run so that the runner itself stays unhashed.
    "microcosm.build.uk_runtime.ledger_surface": _FULL_GRAPH + _NATIONAL_GRAPH,
}

#: The kernels grouped by shared hash: the full graph's target, selection and
#: measures kernels share _TargetKernel's; the problem adds its loss-weight
#: hash; the two national kernels share _NationalKernel's.
FULL_TARGETS = "uk.full.target_compilation@1, target_selection@1 and measures@1"
FULL_PROBLEM = "uk.full.problem@1"
NATIONAL = "uk.full.national_targets@1 and national_problem@1"
KERNEL_GROUPS = MappingProxyType(
    {
        FULL_TARGETS: tuple(
            _KERNELS_BY_REF[ref]
            for ref in (
                "uk.full.target_compilation@1",
                "uk.full.target_selection@1",
                "uk.full.measures@1",
            )
        ),
        FULL_PROBLEM: (_KERNELS_BY_REF["uk.full.problem@1"],),
        NATIONAL: tuple(_KERNELS_BY_REF[ref] for ref in _NATIONAL_GRAPH),
    }
)

#: Imports a kernel's module makes only to build the hash, and the importers
#: the reason covers. They stay unhashed on purpose, so an edit to the list
#: re-keys only the kernels whose hashed modules it adds or removes.
HASH_INPUTS = MappingProxyType(
    {
        "microcosm.build.uk_runtime.target_compile_modules": HashInput(
            importers=(
                "microcosm.build.uk_runtime.graph_national",
                "microcosm.build.uk_runtime.graph_targets",
            ),
            reason=(
                "names the modules the target kernels hash; graph_targets and "
                "graph_national read it only to build their hashes"
            ),
        ),
        "microcosm.build.uk_runtime.solve_scope": HashInput(
            importers=("microcosm.build.uk_runtime.graph_targets",),
            reason=(
                "computes uk_solve_fingerprint, which graph_targets puts in the "
                "full graph's target hashes and in the graph's contract-identity "
                "parameter; no kernel's run reads it"
            ),
        ),
        # graph_national imports these only inside the national gate and
        # readback kernels' implementation hashes, which bind them there
        # (microcosm#931); the national target and problem kernels never run
        # those methods.
        **{
            module: HashInput(
                importers=("microcosm.build.uk_runtime.graph_national",),
                reason=(
                    "imported only in uk.full.national-gates@1's or "
                    "uk.full.national-readback@1's implementation hash, to "
                    "hash it there"
                ),
            )
            for module in (
                "microcosm.build.gates",
                "microcosm.build.trace",
                "microcosm.build.uk_runtime.donor_uprating",
                "microcosm.build.uk_runtime.etb_services",
                "microcosm.calibrate.monetary_binding",
                "microcosm.frame.adapters.policyengine_uk",
                "microcosm.frame.materialize",
            )
        },
    }
)

#: Every ``importer -> imported`` edge the walk from a kernel group's hashed
#: modules meets that the group does not hash and no review has classified,
#: with the groups that leave it unhashed. Recorded as found when the
#: per-kernel check was added; microcosm#1497 classifies them. Hash an entry
#: in the kernels that run it, move what they run into a module they hash, or
#: record why they need not hash it, then delete it here. The check fails on
#: an edge a group meets that is not recorded for it and on an edge recorded
#: for a group that no longer meets it, so the mapping can only shrink, and
#: removing a module from one group's hash fails even where another group
#: records the same edge.
UNCLASSIFIED_IMPORTS = MappingProxyType(
    {
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.artifact_files": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.country_spec": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.country_spec": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.gate_battery": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.fiscal_targets -> microcosm.build.gates": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.weighted_integrity -> microcosm.build.gates": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.build.holdout": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.weighted_integrity -> microcosm.build.input_mass": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.logbook_adoption": (
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.plan": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.source_manifest": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.frs_relationships -> microcosm.build.source_manifest": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.canonical": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.engine_abi": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.loader": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.model": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.schemas": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.take_up_semantics": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.spec_engine.yaml12": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.staging_v2": (
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.build.trace": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.atomic_area_support": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.battery_bindings": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.calibration_run": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.cgt_projection": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.content_identity": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.build.uk_runtime.dataset_size": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.dfe_spring_term": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.diagnostics": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.frs_annual_income": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.frs_relationships -> microcosm.build.uk_runtime.frs_spine": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.frs_spine": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.full_gates": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_authority_input -> microcosm.build.uk_runtime.geography_sources": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.graph_build": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.graph_calibration": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.graph_population": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.build.uk_runtime.graph_population": (
            FULL_TARGETS,
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.graph_targets": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.graph_terminal": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.hmrc_replay -> microcosm.build.uk_runtime.hmrc_income": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.hmrc_income": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.weighted_integrity -> microcosm.build.uk_runtime.hmrc_source_contract": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.income_tax_rate_classes": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.lcfs_consumption": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.build.uk_runtime.local_geography": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.full_measure -> microcosm.build.uk_runtime.local_rowwise": (
            FULL_TARGETS,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.full_problem -> microcosm.build.uk_runtime.local_rowwise": (
            FULL_TARGETS,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.local_rowwise": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.build.uk_runtime.local_rowwise": (
            FULL_TARGETS,
        ),
        "microcosm.build.uk_runtime.full_measure -> microcosm.build.uk_runtime.national_calibration": (
            FULL_TARGETS,
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.build.uk_runtime.national_doctrine": (
            FULL_TARGETS,
        ),
        "microcosm.build.uk_runtime.local_doctrine -> microcosm.build.uk_runtime.national_doctrine": (
            FULL_TARGETS,
        ),
        "microcosm.build.uk_runtime.frs_relationships -> microcosm.build.uk_runtime.national_frame": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.full_measure -> microcosm.build.uk_runtime.national_frame": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.national_frame": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.build.uk_runtime.national_frame": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.national_frame": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.weighted_integrity -> microcosm.build.uk_runtime.national_frame": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.full_measure -> microcosm.build.uk_runtime.rowwise_dataset": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.rowwise_posture": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.build.uk_runtime.size_checkpoint": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.build.uk_runtime.solve_progress": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.build.uk_runtime.target_weights": (
            FULL_TARGETS,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.build.uk_runtime.uc_target_measurements": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.build.uk_runtime.weight_year": (
            NATIONAL,
        ),
        "microcosm.build.gates -> microcosm.build.us_runtime.stacked_spine": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.solve -> microcosm.calibrate.exact_k": (FULL_PROBLEM,),
        "microcosm.calibrate.solve -> microcosm.calibrate.gates": (FULL_PROBLEM,),
        "microcosm.calibrate.solve -> microcosm.calibrate.initialization": (
            FULL_PROBLEM,
        ),
        "microcosm.calibrate.matrix -> microcosm.calibrate.monetary_binding": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.gates -> microcosm.calibrate.solve": (FULL_TARGETS, NATIONAL),
        # Reviewed (round 2 of #1493): only national_calibration_manifest calls
        # it, in uk.full.national-gates@1, which reads the solve's result.
        "microcosm.build.uk_runtime.graph_national -> microcosm.calibrate.solve": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.calibrate.solve": (
            FULL_TARGETS,
        ),
        "microcosm.calibrate.artifacts -> microcosm.calibrate.solve": (
            FULL_TARGETS,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.frame.adapters.policyengine_uk": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.national_frame -> microcosm.frame.adapters.policyengine_uk": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.country_spec -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.frs_relationships -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.full_measure -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.frame.bundle": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.frame.bundle": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.national_calibration -> microcosm.frame.bundle": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.national_frame -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.weighted_integrity -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.artifacts -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.matrix -> microcosm.frame.bundle": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.solve -> microcosm.frame.bundle": (FULL_PROBLEM,),
        "microcosm.build.uk_runtime.national_frame -> microcosm.frame.materialize": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.national_frame -> microcosm.frame.schema": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.frame.bundle -> microcosm.frame.schema": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.gates -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.full_measure -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.frame.weights": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.local_rowwise -> microcosm.frame.weights": (
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.measure_simulation -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.national_calibration -> microcosm.frame.weights": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.national_frame -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.artifacts -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.matrix -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.calibrate.solve -> microcosm.frame.weights": (FULL_PROBLEM,),
        "microcosm.frame.bundle -> microcosm.frame.weights": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.graph.canonical": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.graph.canonical": (
            FULL_TARGETS,
            FULL_PROBLEM,
        ),
        "microcosm.calibrate.artifacts -> microcosm.graph.canonical": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.graph.codecs": (
            FULL_TARGETS,
            FULL_PROBLEM,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.graph.decl": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.graph.decl": (
            FULL_TARGETS,
            FULL_PROBLEM,
        ),
        "microcosm.calibrate.artifacts -> microcosm.graph.decl": (
            FULL_TARGETS,
            FULL_PROBLEM,
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_national -> microcosm.graph.kernel": (
            NATIONAL,
        ),
        "microcosm.build.uk_runtime.graph_targets -> microcosm.graph.kernel": (
            FULL_TARGETS,
            FULL_PROBLEM,
        ),
    }
)


@pytest.mark.parametrize(
    ("kernel", "module"),
    _TARGET_AND_PROBLEM_KERNELS,
    ids=[kernel.ref for kernel, _ in _TARGET_AND_PROBLEM_KERNELS],
)
def test_an_edit_to_the_geography_labels_rekeys_the_kernel(
    tmp_path, monkeypatch, kernel, module
):
    """The shared and UK ledger targets label a target's geography from
    UK_GEOGRAPHY_ID_TO_LABEL, so an edit there must give these kernels new
    keys instead of reusing a surface with the old label (microcosm#1492)."""
    assert_an_edit_rekeys(kernel, geography_constants_module, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("module_name", "ref"),
    [(name, ref) for name, refs in _RUN_MODULES.items() for ref in refs],
    ids=[
        f"{name.rsplit('.', 1)[1]}-{ref}"
        for name, refs in _RUN_MODULES.items()
        for ref in refs
    ],
)
def test_an_edit_to_code_the_kernels_run_rekeys_them(
    tmp_path, monkeypatch, module_name, ref
):
    kernel, _ = _KERNELS_BY_REF[ref]
    module = importlib.import_module(module_name)
    assert_an_edit_rekeys(kernel, module, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("kernel", "module"),
    _TARGET_AND_PROBLEM_KERNELS,
    ids=[kernel.ref for kernel, _ in _TARGET_AND_PROBLEM_KERNELS],
)
def test_an_edit_to_calibration_run_leaves_the_kernels_keys(
    tmp_path, monkeypatch, kernel, module
):
    """The target kernels run only the two functions now in ledger_surface.
    An edit elsewhere in calibration_run, the national calibration runner,
    re-runs no target compilation and, in builds that construct the spine in
    the graph, no solve (microcosm#1492)."""
    from microcosm.build.uk_runtime import calibration_run

    before = kernel.implementation_hash()
    source = Path(calibration_run.__file__)
    edited = tmp_path / source.name
    edited.write_bytes(source.read_bytes() + b"\n# an edited line\n")
    monkeypatch.setattr(calibration_run, "__file__", str(edited))
    assert kernel.implementation_hash() == before


def test_full_problem_loss_weights_hash_what_target_weights_reads(monkeypatch):
    """target_weights takes UK_WIDE_GEOGRAPHY_IDS from geography_constants
    and the loss scales from calibrate.solve for the weights the problem
    binds, so the problem's weight hash covers both modules itself, not only
    through the target modules' hash."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        graph_targets,
        "source_hash",
        lambda *objects, **_dependencies: calls.append(objects) or "test",
    )
    graph_targets.UKFullProblemKernel().implementation_hash()
    (weights,) = [objects for objects in calls if target_weights in objects]
    assert geography_constants_module in weights
    assert calibrate_solve in weights


def test_the_national_kernels_leave_calibrate_solve_unhashed():
    """Neither the national target nor the problem path calls calibrate.solve:
    its one caller in graph_national, national_calibration_manifest, runs in
    uk.full.national-gates@1, whose node reads the solve's result. Hashing it
    here would re-run national target compilation and the problem's engine
    measures on every solver edit, with no coverage gain (round 2 of #1493)."""
    for ref in _NATIONAL_GRAPH:
        kernel, module = _KERNELS_BY_REF[ref]
        assert calibrate_solve.__name__ not in hashed_module_names(kernel, module)


def test_every_import_a_target_kernel_reaches_is_hashed_or_recorded():
    """The walk covers each kernel's own module and every module hashed
    beside the shared list, not only the list (microcosm#1492 review), and
    the record is kept per group of kernels sharing one hash."""
    assert {kernel for group in KERNEL_GROUPS.values() for kernel, _ in group} == {
        kernel for kernel, _ in _TARGET_AND_PROBLEM_KERNELS
    }
    mismatches = recorded_import_mismatches(
        KERNEL_GROUPS,
        listed=UK_TARGET_COMPILE_MODULE_NAMES,
        excluded=frozenset(EXCLUDED_DIRECT_IMPORTS),
        hash_inputs=HASH_INPUTS,
        recorded=UNCLASSIFIED_IMPORTS,
    )
    assert not mismatches, mismatches
