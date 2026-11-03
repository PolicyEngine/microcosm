"""Both UK target kernels hash every module that decides a target value.

Two checks pin the closure, as for the transport compile kernels:

- the shared list's own: every first-party import of a module in
  ``UK_TARGET_COMPILE_MODULE_NAMES`` is listed, or excluded with a reason;
- each kernel's: the walk from every module a target or problem kernel
  hashes, its own module included, meets only modules the kernel hashes, the
  list's exclusions, or the imports recorded in ``UNCLASSIFIED_IMPORTS``,
  which no review has classified yet (microcosm#1497). That mapping may only
  shrink.
"""

from __future__ import annotations

import importlib
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
    assert_an_edit_rekeys,
    edges,
    first_party_imports,
    hashed_module_names,
    unhashed_imports,
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
    # The full problem's loss scales (default_target_loss_scales) and the
    # national problem's effective sample size.
    "microcosm.calibrate.solve": ("uk.full.problem@1", *_NATIONAL_GRAPH),
    # full_targets' exclusion clock (exclusion_evaluation_date).
    "microcosm.build.uk_runtime.weighted_integrity": _FULL_GRAPH + _NATIONAL_GRAPH,
    # full_targets' surface provenance and band-edge roster check, moved out
    # of calibration_run, which no target kernel hashes.
    "microcosm.build.uk_runtime.ledger_surface": _FULL_GRAPH + _NATIONAL_GRAPH,
}

#: Imports the kernels' modules make only to build the hash. They stay
#: unhashed on purpose, so an edit to the list re-keys only the kernels whose
#: hashed modules it adds or removes.
HASH_INPUTS = MappingProxyType(
    {
        "microcosm.build.uk_runtime.target_compile_modules": (
            "names the modules the target kernels hash; graph_targets and "
            "graph_national read it only to build their hashes"
        ),
        "microcosm.build.uk_runtime.solve_scope": (
            "computes uk_solve_fingerprint, which graph_targets puts in the "
            "full graph's target hashes and in the graph's contract-identity "
            "parameter; no kernel's run reads it"
        ),
    }
)

#: Every first-party module the walk from a target or problem kernel's
#: hashed modules meets that the kernel does not hash and no review has
#: classified, with the hashed (or excluded) modules that import it. Recorded
#: as found when the per-kernel check was added; microcosm#1497 classifies
#: them. Hash an entry in the kernels that run it, move what they run into a
#: module they hash, or record why they need not hash it, then delete it
#: here. The check fails on an import missing from this mapping and on an
#: entry the walk no longer meets, so the mapping can only shrink.
UNCLASSIFIED_IMPORTS = MappingProxyType(
    {
        "microcosm.build.artifact_files": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.country_spec": (
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.gate_battery": ("microcosm.build.uk_runtime.graph_national",),
        "microcosm.build.gates": (
            "microcosm.build.uk_runtime.fiscal_targets",
            "microcosm.build.uk_runtime.weighted_integrity",
        ),
        "microcosm.build.holdout": ("microcosm.build.uk_runtime.local_rowwise",),
        "microcosm.build.input_mass": (
            "microcosm.build.uk_runtime.weighted_integrity",
        ),
        "microcosm.build.logbook_adoption": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.plan": ("microcosm.build.country_spec",),
        "microcosm.build.source_manifest": (
            "microcosm.build.country_spec",
            "microcosm.build.uk_runtime.frs_relationships",
        ),
        "microcosm.build.spec_engine.canonical": ("microcosm.build.country_spec",),
        "microcosm.build.spec_engine.engine_abi": ("microcosm.build.country_spec",),
        "microcosm.build.spec_engine.loader": ("microcosm.build.country_spec",),
        "microcosm.build.spec_engine.model": ("microcosm.build.country_spec",),
        "microcosm.build.spec_engine.schemas": ("microcosm.build.country_spec",),
        "microcosm.build.spec_engine.take_up_semantics": (
            "microcosm.build.country_spec",
        ),
        "microcosm.build.spec_engine.yaml12": ("microcosm.build.country_spec",),
        "microcosm.build.staging_v2": ("microcosm.build.uk_runtime.graph_national",),
        "microcosm.build.trace": ("microcosm.build.country_spec",),
        "microcosm.build.uk_runtime.atomic_area_support": (
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.battery_bindings": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.calibration_run": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.cgt_projection": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.content_identity": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.dataset_size": (
            "microcosm.build.uk_runtime.local_rowwise",
        ),
        "microcosm.build.uk_runtime.dfe_spring_term": (
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.diagnostics": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.frs_annual_income": (
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.frs_spine": (
            "microcosm.build.uk_runtime.frs_relationships",
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.full_gates": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.geography_sources": (
            "microcosm.build.uk_runtime.local_authority_input",
        ),
        "microcosm.build.uk_runtime.graph_build": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.graph_calibration": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.graph_population": (
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.graph_targets",
        ),
        "microcosm.build.uk_runtime.graph_targets": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.graph_terminal": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.hmrc_income": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.hmrc_source_contract": (
            "microcosm.build.uk_runtime.weighted_integrity",
        ),
        "microcosm.build.uk_runtime.income_tax_rate_classes": (
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.lcfs_consumption": (
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.local_geography": (
            "microcosm.build.uk_runtime.local_rowwise",
        ),
        "microcosm.build.uk_runtime.local_rowwise": (
            "microcosm.build.uk_runtime.full_measure",
            "microcosm.build.uk_runtime.full_problem",
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.graph_targets",
        ),
        "microcosm.build.uk_runtime.national_calibration": (
            "microcosm.build.uk_runtime.full_measure",
        ),
        "microcosm.build.uk_runtime.national_doctrine": (
            "microcosm.build.uk_runtime.graph_targets",
            "microcosm.build.uk_runtime.local_doctrine",
        ),
        "microcosm.build.uk_runtime.national_frame": (
            "microcosm.build.uk_runtime.frs_relationships",
            "microcosm.build.uk_runtime.full_measure",
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.local_rowwise",
            "microcosm.build.uk_runtime.measure_simulation",
            "microcosm.build.uk_runtime.weighted_integrity",
        ),
        "microcosm.build.uk_runtime.rowwise_dataset": (
            "microcosm.build.uk_runtime.full_measure",
        ),
        "microcosm.build.uk_runtime.rowwise_posture": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.uk_runtime.size_checkpoint": (
            "microcosm.build.uk_runtime.local_rowwise",
        ),
        "microcosm.build.uk_runtime.solve_progress": (
            "microcosm.build.uk_runtime.local_rowwise",
        ),
        "microcosm.build.uk_runtime.target_weights": (
            "microcosm.build.uk_runtime.graph_targets",
        ),
        "microcosm.build.uk_runtime.uc_target_measurements": (
            "microcosm.build.uk_runtime.measure_simulation",
        ),
        "microcosm.build.uk_runtime.weight_year": (
            "microcosm.build.uk_runtime.graph_national",
        ),
        "microcosm.build.us_runtime.stacked_spine": ("microcosm.build.gates",),
        "microcosm.calibrate.exact_k": ("microcosm.calibrate.solve",),
        "microcosm.calibrate.gates": ("microcosm.calibrate.solve",),
        "microcosm.calibrate.initialization": ("microcosm.calibrate.solve",),
        "microcosm.calibrate.monetary_binding": ("microcosm.calibrate.matrix",),
        "microcosm.calibrate.solve": (
            "microcosm.build.gates",
            "microcosm.build.uk_runtime.graph_targets",
            "microcosm.calibrate.artifacts",
        ),
        "microcosm.frame.adapters.policyengine_uk": (
            "microcosm.build.uk_runtime.measure_simulation",
            "microcosm.build.uk_runtime.national_frame",
        ),
        "microcosm.frame.bundle": (
            "microcosm.build.country_spec",
            "microcosm.build.uk_runtime.frs_relationships",
            "microcosm.build.uk_runtime.full_measure",
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.local_rowwise",
            "microcosm.build.uk_runtime.measure_simulation",
            "microcosm.build.uk_runtime.national_calibration",
            "microcosm.build.uk_runtime.national_frame",
            "microcosm.build.uk_runtime.weighted_integrity",
            "microcosm.calibrate.artifacts",
            "microcosm.calibrate.matrix",
            "microcosm.calibrate.solve",
        ),
        "microcosm.frame.materialize": ("microcosm.build.uk_runtime.national_frame",),
        "microcosm.frame.schema": (
            "microcosm.build.uk_runtime.national_frame",
            "microcosm.frame.bundle",
        ),
        "microcosm.frame.weights": (
            "microcosm.build.gates",
            "microcosm.build.uk_runtime.full_measure",
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.local_rowwise",
            "microcosm.build.uk_runtime.measure_simulation",
            "microcosm.build.uk_runtime.national_calibration",
            "microcosm.build.uk_runtime.national_frame",
            "microcosm.calibrate.artifacts",
            "microcosm.calibrate.matrix",
            "microcosm.calibrate.solve",
            "microcosm.frame.bundle",
        ),
        "microcosm.graph.canonical": (
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.graph_targets",
            "microcosm.calibrate.artifacts",
        ),
        "microcosm.graph.codecs": ("microcosm.build.uk_runtime.graph_targets",),
        "microcosm.graph.decl": (
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.graph_targets",
            "microcosm.calibrate.artifacts",
        ),
        "microcosm.graph.kernel": (
            "microcosm.build.uk_runtime.graph_national",
            "microcosm.build.uk_runtime.graph_targets",
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


def test_every_import_a_target_kernel_reaches_is_hashed_or_classified():
    """The walk covers each kernel's own module and every module hashed
    beside the shared list, not only the list (microcosm#1492 review)."""
    met: set[str] = set()
    for kernel, module in _TARGET_AND_PROBLEM_KERNELS:
        hashed = hashed_module_names(kernel, module)
        assert not set(HASH_INPUTS) & hashed, kernel.ref
        met |= unhashed_imports(
            hashed,
            listed=UK_TARGET_COMPILE_MODULE_NAMES,
            excluded=frozenset(EXCLUDED_DIRECT_IMPORTS),
        )
    met = {edge for edge in met if edge.split(" -> ")[1] not in HASH_INPUTS}
    recorded = edges(UNCLASSIFIED_IMPORTS)
    assert met == recorded, {
        "unhashed and not classified": sorted(met - recorded),
        "recorded but no longer met (delete them)": sorted(recorded - met),
    }
