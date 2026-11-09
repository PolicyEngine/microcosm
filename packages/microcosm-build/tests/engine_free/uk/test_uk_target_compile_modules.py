"""Both UK target kernels hash every module that decides a target value.

Two checks pin the closure:

- the shared list's own, as for the transport compile kernels: every
  first-party import of a module in ``UK_TARGET_COMPILE_MODULE_NAMES`` is
  listed, or excluded with a reason;
- each kernel's: the derived binding rule in test_uk_graph.py holds what each
  target and problem kernel's hash covers to what its code computes through
  (``test_every_uk_graph_kernel_hashes_each_helper_it_computes_through``). It
  replaced microcosm#1497's per-kernel import record for these kernels
  (microcosm#931 ruling 6), so this file keeps only what that rule does not
  cover: the modules a kernel's file imports only to build its hash
  (``HASH_INPUTS``) stay out of the hashes.
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
#: microcosm#1492, and the kernels that now hash it. These modules are in the
#: hash all four full-graph kernels share, so what one of them runs re-keys
#: all four.
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

#: Imports a kernel's module makes only to build an implementation hash, and
#: the importers the reason covers. The derived binding rule in test_uk_graph.py
#: leaves every implementation hash out of what a kernel's code computes
#: through, so such an import binds nothing. A target or problem kernel hashes
#: one only where another module it hashes imports it too, and none hashes the
#: first two, which stay unhashed on purpose: an edit to the list re-keys only
#: the kernels whose hashed modules it adds or removes
#: (test_a_target_kernel_hashes_a_hash_input_only_where_another_module_imports_it).
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
        # graph_national imports these only inside implementation hashes,
        # which bind them there (microcosm#931): the national gate and
        # readback kernels', and for the last three the national problem's
        # too, whose code computes through them. The national target and
        # problem kernels never run the gate and readback kernels' methods.
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
            )
        },
        **{
            module: HashInput(
                importers=("microcosm.build.uk_runtime.graph_national",),
                reason=(
                    "imported only in implementation hashes, to hash it there: "
                    "uk.full.national-gates@1's or uk.full.national-readback@1's, "
                    "and uk.full.national_problem@1's, whose code computes "
                    "through it"
                ),
            )
            for module in (
                "microcosm.calibrate.monetary_binding",
                "microcosm.frame.adapters.policyengine_uk",
                "microcosm.frame.materialize",
            )
        },
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


def test_a_target_kernel_hashes_a_hash_input_only_where_another_module_imports_it():
    """What the derived binding rule does not cover (microcosm#931 ruling 6).

    The rule in test_uk_graph.py holds each target and problem kernel's hash
    to what its code computes through, in place of microcosm#1497's per-kernel
    import record, and leaves every implementation hash out of that code, so
    an import made only to build a hash binds nothing. It does not check the
    other side, that such a module stays out of the kernels' hashes. Each hash
    input is still imported by every importer listed for it, and a target or
    problem kernel hashes one only where a module it hashes, other than those
    importers, imports it too, as measure_simulation, which the measures
    compute through, imports the engine adapter. So the compile list and the
    solve fingerprint's module stay unhashed, and an edit to a name or a
    reason in either re-keys none of these kernels."""
    for name, hash_input in HASH_INPUTS.items():
        for importer in hash_input.importers:
            assert name in first_party_imports(importer), (importer, name)
    for kernel, module in _TARGET_AND_PROBLEM_KERNELS:
        hashed = hashed_module_names(kernel, module)
        for name, hash_input in HASH_INPUTS.items():
            if name not in hashed:
                continue
            beside = {
                other
                for other in hashed - set(hash_input.importers)
                if name in first_party_imports(other)
            }
            assert beside, (kernel.ref, name)
    hashed = set().union(
        *(hashed_module_names(*pair) for pair in _TARGET_AND_PROBLEM_KERNELS)
    )
    assert not hashed & {
        "microcosm.build.uk_runtime.target_compile_modules",
        "microcosm.build.uk_runtime.solve_scope",
    }
