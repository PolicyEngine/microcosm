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
- each kernel's: the derived binding rule in test_transport_binding_rule
  holds what each transport kernel's hash covers to what its code computes
  through (``test_every_transport_kernel_hashes_each_helper_it_computes_through``).
  It replaced microcosm#1497's per-kernel import record for these kernels
  (microcosm#931).

Beside a pin of an edit that must re-key the compile kernels, this file keeps
what that rule does not cover: the compile list is a module a kernel's file
imports only to build its hash (``HASH_INPUTS``), so it is still imported
where listed, and no transport kernel hashes it.
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
from test_support.microcosm_build.derived_binding_rule import hashed_modules
from test_support.microcosm_build.first_party_imports import (
    HashInput,
    assert_an_edit_rekeys,
    first_party_imports,
    unlisted_imports,
)
from test_support.microcosm_build.transport_kernels import transport_kernels

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

#: Imports a kernel's module makes only to build an implementation hash, and
#: the importers the reason covers. The derived binding rule in
#: test_transport_binding_rule leaves every implementation hash out of what a
#: kernel's code computes through, so such an import binds nothing. They stay
#: unhashed on purpose, so an edit to the list re-keys only the kernels whose
#: hashed modules it adds or removes
#: (test_no_transport_kernel_hashes_a_hash_input).
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


def test_no_transport_kernel_hashes_a_hash_input():
    """What the derived binding rule does not cover (microcosm#931).

    The rule in test_transport_binding_rule holds each transport kernel's
    hash to what its code computes through, in place of microcosm#1497's
    per-kernel import record, and leaves every implementation hash out of
    that code, so an import made only to build a hash binds nothing; its
    stop at ``target_kernels._COMPILE_MODULES`` rests on only hashes reading
    the list. It does not check the other side, that such a module stays
    out of the hashes. Each hash input is still imported by every importer
    listed for it, the compile kernels hash the list's modules, and no
    transport kernel, of every kernel the registry registers, hashes a hash
    input: many kernels hash target_kernels and none the list, so an edit to
    a name or a reason in it re-keys no kernel, and an edit to the tuple
    re-keys only the kernels whose hashed modules it adds or removes."""
    compile_list = _compile_list()
    assert target_kernels._COMPILE_MODULES == compile_list.transport_compile_modules()
    for name, hash_input in HASH_INPUTS.items():
        for importer in hash_input.importers:
            assert name in first_party_imports(importer), (importer, name)
    hashed = {
        ref: hashed_modules(kernel) for ref, kernel in transport_kernels().items()
    }
    assert {kernel.ref for kernel, _ in KERNELS_HASHING_THE_COMPILE_MODULES} == {
        ref
        for ref, modules in hashed.items()
        if set(compile_list.TRANSPORT_COMPILE_MODULE_NAMES) <= modules
    }
    holding = {ref for ref, modules in hashed.items() if set(HASH_INPUTS) & modules}
    assert not holding, sorted(holding)


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
