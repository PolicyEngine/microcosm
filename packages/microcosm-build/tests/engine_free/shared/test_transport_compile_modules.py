"""The transport compile kernels hash every module target compilation reads.

``targets.compile@1`` and every other transport kernel built on target
compilation hash ``target_kernels._COMPILE_MODULES``, the modules named in
``transport.compile_modules``. ``source_hash`` hashes the files it is given
and does not follow imports, so a module that a hashed module imports, and
that the kernel does not hash, can change a kernel's output without changing
a node key. ``ledger_targets`` takes a target's geography label from
``microcosm.calibrate.geography_constants`` that way; before microcosm#1492
no compile kernel hashed it.

Two checks pin the closure:

- the shared list's own: every first-party import of a listed module is
  listed, or excluded with a reason in ``compile_modules``;
- each kernel's: the walk from every module the kernel hashes, its own
  module included, meets only modules the kernel hashes, the list's
  exclusions, or the imports recorded in ``UNCLASSIFIED_IMPORTS``, which no
  review has classified yet (microcosm#1497). That mapping may only shrink.
"""

from __future__ import annotations

import importlib
from types import MappingProxyType, ModuleType

import pytest

import microcosm.calibrate.geography_constants as geography_constants_module
from microcosm.build.transport import (
    population_kernels,
    quantile_kernels,
    takeup_kernels,
    target_kernels,
)
from test_support.microcosm_build.first_party_imports import (
    assert_an_edit_rekeys,
    edges,
    first_party_imports,
    hashed_module_names,
    unhashed_imports,
    unlisted_imports,
)

#: Every kernel whose implementation hash includes ``_COMPILE_MODULES``, with
#: the module whose ``source_hash`` it calls. The first five compile
#: references with ``compile_target_surface``; the two quantile maps decode
#: the surface an upstream compile stored.
KERNELS_HASHING_THE_COMPILE_MODULES = (
    (target_kernels.TARGETS_COMPILE, target_kernels),
    (target_kernels.TAKEUP_COMPARE, target_kernels),
    (takeup_kernels.TAKEUP_GAP, takeup_kernels),
    (population_kernels.TRANSPORT_CREATE, population_kernels),
    (population_kernels.TRANSPORT_CREATE_DECLARED, population_kernels),
    (population_kernels.TRANSPORT_QUANTILE_MAP, population_kernels),
    (quantile_kernels.TRANSPORT_QUANTILE_MAP_V2, quantile_kernels),
)
_KERNEL_IDS = [kernel.ref for kernel, _ in KERNELS_HASHING_THE_COMPILE_MODULES]

#: Imports the kernels' modules make only to build the hash. They stay
#: unhashed on purpose, so an edit to the list re-keys only the kernels whose
#: hashed modules it adds or removes.
HASH_INPUTS = MappingProxyType(
    {
        "microcosm.build.transport.compile_modules": (
            "names the modules the compile kernels hash; target_kernels reads "
            "it only to build _COMPILE_MODULES"
        ),
    }
)

#: Every first-party module the walk from a compile kernel's hashed modules
#: meets that the kernel does not hash and no review has classified, with the
#: hashed (or excluded) modules that import it. Recorded as found when the
#: per-kernel check was added; microcosm#1497 classifies them. Hash an entry
#: in the kernels that run it, move what they run into a module they hash, or
#: record why they need not hash it, then delete it here. The check fails on
#: an import missing from this mapping and on an entry the walk no longer
#: meets, so the mapping can only shrink.
UNCLASSIFIED_IMPORTS = MappingProxyType(
    {
        "microcosm.build": ("microcosm.build.trace",),
        "microcosm.build.calibration_columns": (
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.build.calibration_predicates": (
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.build.observation": ("microcosm.build.plan",),
        "microcosm.build.reference_intervals": (
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.build.spec_engine.canonical": (
            "microcosm.build.spec_engine.engine_abi",
            "microcosm.build.spec_engine.loader",
            "microcosm.build.spec_engine.take_up_semantics",
        ),
        "microcosm.build.spec_engine.engine_abi": (
            "microcosm.build.spec_engine.loader",
        ),
        "microcosm.build.spec_engine.errors": (
            "microcosm.build.spec_engine.engine_abi",
            "microcosm.build.spec_engine.loader",
            "microcosm.build.spec_engine.schemas",
            "microcosm.build.spec_engine.take_up_semantics",
            "microcosm.build.spec_engine.yaml12",
        ),
        "microcosm.build.spec_engine.model": (
            "microcosm.build.spec_engine.canonical",
            "microcosm.build.spec_engine.engine_abi",
            "microcosm.build.spec_engine.loader",
        ),
        "microcosm.build.spec_engine.resolver": ("microcosm.build.spec_engine.loader",),
        "microcosm.build.spec_engine.schemas": (
            "microcosm.build.spec_engine.engine_abi",
            "microcosm.build.spec_engine.loader",
        ),
        "microcosm.build.spec_engine.seeds": ("microcosm.build.spec_engine.model",),
        "microcosm.build.spec_engine.take_up_semantics": (
            "microcosm.build.spec_engine.engine_abi",
        ),
        "microcosm.build.spec_engine.yaml12": ("microcosm.build.spec_engine.loader",),
        "microcosm.build.transport.column_kernels": (
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.build.transport.unit_declarations": (
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.build.us_runtime.multispine_pool": (
            "microcosm.build.spec_engine.engine_abi",
        ),
        "microcosm.calibrate.artifacts": ("microcosm.build.transport.target_kernels",),
        "microcosm.calibrate.matrix": (
            "microcosm.build.transport.target_kernels",
            "microcosm.calibrate.artifacts",
        ),
        "microcosm.calibrate.monetary_binding": (
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.calibrate.solve": ("microcosm.calibrate.artifacts",),
        "microcosm.frame.adapters.policyengine_us": (
            "microcosm.build.spec_engine.engine_abi",
        ),
        "microcosm.frame.adapters.policyengine_us_concepts": (
            "microcosm.build.transport.population_kernels",
            "microcosm.frame.transport",
        ),
        "microcosm.frame.bundle": (
            "microcosm.build.plan",
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.target_kernels",
            "microcosm.calibrate.artifacts",
            "microcosm.graph.kernel",
        ),
        "microcosm.frame.concept_mapping": (
            "microcosm.build.transport.population_kernels",
        ),
        "microcosm.frame.concepts": (
            "microcosm.build.reference_intervals",
            "microcosm.build.transport.population_kernels",
            "microcosm.frame.transport",
        ),
        "microcosm.frame.materialize": (
            "microcosm.build.transport.population_kernels",
            "microcosm.frame.transport",
        ),
        "microcosm.frame.scaling": (
            "microcosm.build.transport.population_kernels",
            "microcosm.frame.transport",
        ),
        "microcosm.frame.schema": (
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.target_kernels",
            "microcosm.frame.bundle",
        ),
        "microcosm.frame.unit_construction": (
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.frame.weights": (
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.target_kernels",
            "microcosm.calibrate.artifacts",
            "microcosm.frame.bundle",
            "microcosm.graph.kernel",
        ),
        "microcosm.graph.canonical": (
            "microcosm.build.reference_intervals",
            "microcosm.build.transport.quantile_kernels",
            "microcosm.build.transport.takeup_kernels",
            "microcosm.build.transport.target_kernels",
            "microcosm.calibrate.artifacts",
        ),
        "microcosm.graph.decl": (
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.takeup_kernels",
            "microcosm.build.transport.target_kernels",
            "microcosm.calibrate.artifacts",
            "microcosm.graph.canonical",
            "microcosm.graph.kernel",
        ),
        "microcosm.graph.kernel": (
            "microcosm.build.transport.population_kernels",
            "microcosm.build.transport.quantile_kernels",
            "microcosm.build.transport.takeup_kernels",
            "microcosm.build.transport.target_kernels",
        ),
        "microcosm.graph.population": ("microcosm.build.transport.population_kernels",),
    }
)


def _compile_list() -> ModuleType:
    """The shared list and its exclusions (``compile_modules``, microcosm#1492)."""
    return importlib.import_module("microcosm.build.transport.compile_modules")


@pytest.mark.parametrize(
    ("kernel", "module"), KERNELS_HASHING_THE_COMPILE_MODULES, ids=_KERNEL_IDS
)
def test_an_edit_to_the_geography_labels_rekeys_the_kernel(
    tmp_path, monkeypatch, kernel, module
):
    """ledger_targets labels a target's geography from UK_GEOGRAPHY_ID_TO_LABEL
    when the facts give no name, so an edit there must give a new key."""
    assert_an_edit_rekeys(kernel, geography_constants_module, tmp_path, monkeypatch)


def test_the_compile_list_lives_in_a_module_no_compile_kernel_hashes():
    """target_kernels is hashed by 13 kernels; the list is not, so an edit to
    a name or a reason no longer re-keys all of them."""
    compile_list = _compile_list()
    assert target_kernels._COMPILE_MODULES == compile_list.transport_compile_modules()
    for kernel, module in KERNELS_HASHING_THE_COMPILE_MODULES:
        assert compile_list.__name__ not in hashed_module_names(kernel, module)


def test_every_direct_import_of_a_compile_module_is_listed_or_excluded():
    compile_list = _compile_list()
    uncovered = unlisted_imports(
        compile_list.TRANSPORT_COMPILE_MODULE_NAMES,
        frozenset(compile_list.EXCLUDED_DIRECT_IMPORTS),
    )
    assert not uncovered, sorted(uncovered)


def test_compile_exclusions_are_live_and_never_listed():
    compile_list = _compile_list()
    listed = compile_list.TRANSPORT_COMPILE_MODULE_NAMES
    excluded = set(compile_list.EXCLUDED_DIRECT_IMPORTS)
    imported = set().union(*map(first_party_imports, listed))
    assert excluded <= imported, sorted(excluded - imported)
    assert not excluded & set(listed)


def test_every_import_a_compile_kernel_reaches_is_hashed_or_classified():
    """The walk covers each kernel's own module and every module hashed
    beside the shared list, not only the list (microcosm#1492 review)."""
    compile_list = _compile_list()
    met: set[str] = set()
    for kernel, module in KERNELS_HASHING_THE_COMPILE_MODULES:
        hashed = hashed_module_names(kernel, module)
        assert not set(HASH_INPUTS) & hashed, kernel.ref
        met |= unhashed_imports(
            hashed,
            listed=compile_list.TRANSPORT_COMPILE_MODULE_NAMES,
            excluded=frozenset(compile_list.EXCLUDED_DIRECT_IMPORTS),
        )
    met = {edge for edge in met if edge.split(" -> ")[1] not in HASH_INPUTS}
    recorded = edges(UNCLASSIFIED_IMPORTS)
    assert met == recorded, {
        "unhashed and not classified": sorted(met - recorded),
        "recorded but no longer met (delete them)": sorted(recorded - met),
    }
