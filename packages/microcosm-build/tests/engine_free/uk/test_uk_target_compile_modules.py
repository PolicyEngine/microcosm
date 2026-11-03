"""Both UK target kernels hash every module that decides a target value."""

from __future__ import annotations

from pathlib import Path

import pytest

import microcosm.calibrate.geography_constants as geography_constants_module
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
    first_party_imports,
    uncovered_imports,
)


def test_every_direct_import_of_a_listed_module_is_listed_or_excluded():
    """Every first-party import, not only microcosm.build: the target
    hierarchies take their geography labels from
    microcosm.calibrate.geography_constants, which a scan of microcosm.build
    imports never saw (microcosm#1492)."""
    uncovered = uncovered_imports(
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


#: The kernels whose hash includes the UK target compile modules: the full
#: graph's target, selection, measures and problem kernels and the national
#: graph's target and problem kernels.
_TARGET_AND_PROBLEM_KERNELS = (
    graph_targets.UKFullTargetCompilationKernel(),
    graph_targets.UKFullTargetSelectionKernel(),
    graph_targets.UKFullMeasureKernel(),
    graph_targets.UKFullProblemKernel(),
    graph_national.UKNationalTargetKernel(),
    graph_national.UKNationalProblemKernel(resolver_factory=None),
)


@pytest.mark.parametrize(
    "kernel", _TARGET_AND_PROBLEM_KERNELS, ids=lambda kernel: kernel.ref
)
def test_an_edit_to_the_geography_labels_rekeys_the_kernel(
    tmp_path, monkeypatch, kernel
):
    """The shared and UK ledger targets label a target's geography from
    UK_GEOGRAPHY_ID_TO_LABEL, so an edit there must give these kernels new
    keys instead of reusing a surface with the old label (microcosm#1492)."""
    before = kernel.implementation_hash()
    source = Path(geography_constants_module.__file__)
    edited = tmp_path / source.name
    edited.write_bytes(source.read_bytes() + b"\n# an edited geography label\n")
    monkeypatch.setattr(geography_constants_module, "__file__", str(edited))
    assert kernel.implementation_hash() != before


def test_full_problem_loss_weights_hash_the_uk_wide_geographies(monkeypatch):
    """target_weights takes UK_WIDE_GEOGRAPHY_IDS from geography_constants for
    the loss weights the problem binds, so the problem's weight hash covers
    that module itself, not only through the target compile modules."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        graph_targets,
        "source_hash",
        lambda *objects, **_dependencies: calls.append(objects) or "test",
    )
    graph_targets.UKFullProblemKernel().implementation_hash()
    (weights,) = [objects for objects in calls if target_weights in objects]
    assert geography_constants_module in weights
