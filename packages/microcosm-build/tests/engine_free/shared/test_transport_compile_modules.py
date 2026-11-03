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
  exclusions, or the imports recorded in ``UNCLASSIFIED_IMPORTS`` for that
  kernel's group, which no review has classified yet (microcosm#1497). That
  mapping may only shrink.
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
    HashInput,
    assert_an_edit_rekeys,
    first_party_imports,
    hashed_module_names,
    recorded_import_mismatches,
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

#: The kernels grouped by shared hash: the two CREATE versions share one
#: class body, so one hashed set; every other kernel is its own group.
COMPILE = "targets.compile@1"
COMPARE = "takeup.compare@1"
GAP = "takeup.gap@1"
CREATE = "transport.create@1 and @2"
QUANTILE_MAP_1 = "transport.quantile_map@1"
QUANTILE_MAP_2 = "transport.quantile_map@2"
KERNEL_GROUPS = MappingProxyType(
    {
        COMPILE: ((target_kernels.TARGETS_COMPILE, target_kernels),),
        COMPARE: ((target_kernels.TAKEUP_COMPARE, target_kernels),),
        GAP: ((takeup_kernels.TAKEUP_GAP, takeup_kernels),),
        CREATE: (
            (population_kernels.TRANSPORT_CREATE, population_kernels),
            (population_kernels.TRANSPORT_CREATE_DECLARED, population_kernels),
        ),
        QUANTILE_MAP_1: (
            (population_kernels.TRANSPORT_QUANTILE_MAP, population_kernels),
        ),
        QUANTILE_MAP_2: (
            (quantile_kernels.TRANSPORT_QUANTILE_MAP_V2, quantile_kernels),
        ),
    }
)

#: Imports a kernel's module makes only to build the hash, and the importers
#: the reason covers. They stay unhashed on purpose, so an edit to the list
#: re-keys only the kernels whose hashed modules it adds or removes.
HASH_INPUTS = MappingProxyType(
    {
        "microcosm.build.transport.compile_modules": HashInput(
            importers=("microcosm.build.transport.target_kernels",),
            reason=(
                "names the modules the compile kernels hash; target_kernels "
                "reads it only to build _COMPILE_MODULES"
            ),
        ),
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
        "microcosm.build.trace -> microcosm.build": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.build.calibration_columns": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.build.calibration_predicates": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.plan -> microcosm.build.observation": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.build.reference_intervals": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.build.spec_engine.canonical": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.canonical": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.take_up_semantics -> microcosm.build.spec_engine.canonical": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.engine_abi": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.build.spec_engine.errors": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.errors": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.schemas -> microcosm.build.spec_engine.errors": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.take_up_semantics -> microcosm.build.spec_engine.errors": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.yaml12 -> microcosm.build.spec_engine.errors": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.canonical -> microcosm.build.spec_engine.model": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.build.spec_engine.model": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.model": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.resolver": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.build.spec_engine.schemas": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.schemas": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.model -> microcosm.build.spec_engine.seeds": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.build.spec_engine.take_up_semantics": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.loader -> microcosm.build.spec_engine.yaml12": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.build.transport.column_kernels": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.build.transport.unit_declarations": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.build.transport.unit_declarations": (
            COMPILE,
            COMPARE,
            GAP,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.build.us_runtime.multispine_pool": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.calibrate.artifacts": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.calibrate.matrix": (
            COMPILE,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.calibrate.artifacts -> microcosm.calibrate.matrix": (
            COMPILE,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.calibrate.monetary_binding": (
            COMPILE,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.calibrate.artifacts -> microcosm.calibrate.solve": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.spec_engine.engine_abi -> microcosm.frame.adapters.policyengine_us": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.adapters.policyengine_us_concepts": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.frame.transport -> microcosm.frame.adapters.policyengine_us_concepts": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.plan -> microcosm.frame.bundle": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.bundle": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.frame.bundle": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.calibrate.artifacts -> microcosm.frame.bundle": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.graph.kernel -> microcosm.frame.bundle": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.concept_mapping": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.reference_intervals -> microcosm.frame.concepts": (
            COMPILE,
            COMPARE,
            GAP,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.concepts": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.frame.transport -> microcosm.frame.concepts": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.materialize": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.frame.transport -> microcosm.frame.materialize": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.scaling": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.frame.transport -> microcosm.frame.scaling": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.schema": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.frame.schema": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.frame.bundle -> microcosm.frame.schema": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.unit_construction": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.frame.unit_construction": (
            COMPILE,
            COMPARE,
            GAP,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.frame.weights": (
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.target_kernels -> microcosm.frame.weights": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.calibrate.artifacts -> microcosm.frame.weights": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.frame.bundle -> microcosm.frame.weights": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.graph.kernel -> microcosm.frame.weights": (
            COMPILE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.reference_intervals -> microcosm.graph.canonical": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.quantile_kernels -> microcosm.graph.canonical": (
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.takeup_kernels -> microcosm.graph.canonical": (GAP,),
        "microcosm.build.transport.target_kernels -> microcosm.graph.canonical": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.calibrate.artifacts -> microcosm.graph.canonical": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.graph.decl": (
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.takeup_kernels -> microcosm.graph.decl": (GAP,),
        "microcosm.build.transport.target_kernels -> microcosm.graph.decl": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.calibrate.artifacts -> microcosm.graph.decl": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.graph.canonical -> microcosm.graph.decl": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.graph.kernel -> microcosm.graph.decl": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.graph.kernel": (
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.quantile_kernels -> microcosm.graph.kernel": (
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.takeup_kernels -> microcosm.graph.kernel": (GAP,),
        "microcosm.build.transport.target_kernels -> microcosm.graph.kernel": (
            COMPILE,
            COMPARE,
            GAP,
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
        "microcosm.build.transport.population_kernels -> microcosm.graph.population": (
            CREATE,
            QUANTILE_MAP_1,
            QUANTILE_MAP_2,
        ),
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


def test_every_import_a_compile_kernel_reaches_is_hashed_or_recorded():
    """The walk covers each kernel's own module and every module hashed
    beside the shared list, not only the list (microcosm#1492 review), and
    the record is kept per group of kernels sharing one hash."""
    compile_list = _compile_list()
    assert {kernel for group in KERNEL_GROUPS.values() for kernel, _ in group} == {
        kernel for kernel, _ in KERNELS_HASHING_THE_COMPILE_MODULES
    }
    mismatches = recorded_import_mismatches(
        KERNEL_GROUPS,
        listed=compile_list.TRANSPORT_COMPILE_MODULE_NAMES,
        excluded=frozenset(compile_list.EXCLUDED_DIRECT_IMPORTS),
        hash_inputs=HASH_INPUTS,
        recorded=UNCLASSIFIED_IMPORTS,
    )
    assert not mismatches, mismatches
