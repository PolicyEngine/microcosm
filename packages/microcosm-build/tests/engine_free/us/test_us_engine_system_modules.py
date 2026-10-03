"""A built tax-benefit system's variable modules leave ``sys.modules`` with it.

policyengine-core registers every variable module of a system it builds as
``sys.modules[f"{id(system)}_{hash(path)}_{file name}"]`` and never removes
it. ``build_reform_tax_benefit_system`` evicts those modules once the system
is collected. These engine-free tests pin the eviction's invariants with a
stand-in system that registers its modules the same way; the requires-US test
in ``engine_workflow/us/test_us_fiscal_refresh_memory.py`` pins the naming
against the installed engine.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.engine_lifecycle import (
    build_reform_tax_benefit_system,
    evict_variable_modules_when_freed,
)


def _variable_module(name: str, payload_bytes: int = 0) -> types.ModuleType:
    """A module shaped like an executed variable file: a function whose
    globals are the module's own namespace (the cycle real modules carry)."""
    module = types.ModuleType(name)
    module.payload = bytearray(payload_bytes)
    exec("def formula():\n    return payload\n", module.__dict__)
    return module


class _ModuleRegisteringSystem:
    """Registers ``n_modules`` variable modules the way policyengine-core does,
    and holds a reference cycle, so only a full collection frees it."""

    def __init__(self, reform=None, *, n_modules: int = 4, payload_bytes: int = 0):
        self.reform = reform
        self.formulas = []
        self.module_names = []
        for index in range(n_modules):
            name = f"{id(self)}_{hash(('fixture', index))}_variable_{index}"
            module = _variable_module(name, payload_bytes)
            sys.modules[name] = module
            self.module_names.append(name)
            self.formulas.append(module.formula)
        self.cycle = self


class _System:
    """A bare weakly referenceable object (``SimpleNamespace`` is not)."""


def _engine(**system_kwargs):
    def default_tax_benefit_system(*, reform=None):
        return _ModuleRegisteringSystem(reform, **system_kwargs)

    return SimpleNamespace(default_tax_benefit_system=default_tax_benefit_system)


def test_modules_leave_sys_modules_once_the_system_is_collected() -> None:
    before = set(sys.modules)
    system = build_reform_tax_benefit_system(_engine(n_modules=6), "reform")
    names = list(system.module_names)
    assert system.reform == "reform"
    assert all(name in sys.modules for name in names)
    # The system keeps its modules while it lives.
    gc.collect()
    assert all(name in sys.modules for name in names)
    del system
    gc.collect()
    assert not any(name in sys.modules for name in names)
    assert set(sys.modules) == before


def test_other_new_modules_and_replaced_names_are_left_alone() -> None:
    """A module first imported during the build (no system prefix) stays, and
    a name re-pointed at another module before the system dies keeps it."""
    genuine = "fixture_package_first_imported_during_the_build"

    def default_tax_benefit_system(*, reform=None):
        sys.modules[genuine] = types.ModuleType(genuine)
        return _ModuleRegisteringSystem(reform, n_modules=3)

    engine = SimpleNamespace(default_tax_benefit_system=default_tax_benefit_system)
    try:
        system = build_reform_tax_benefit_system(engine, None)
        evicted, replaced = system.module_names[:2], system.module_names[2]
        replacement = types.ModuleType(replaced)
        sys.modules[replaced] = replacement
        del system
        gc.collect()
        assert not any(name in sys.modules for name in evicted)
        assert sys.modules[replaced] is replacement
        assert genuine in sys.modules
    finally:
        sys.modules.pop(genuine, None)
        sys.modules.pop(replaced, None)


def test_nothing_is_arranged_without_prefixed_modules_or_a_weak_reference() -> None:
    assert evict_variable_modules_when_freed(_System(), set(sys.modules)) == 0
    # ``object()`` cannot be weakly referenced; the helper must not raise.
    plain = object()
    name = f"{id(plain)}_0_variable"
    sys.modules[name] = types.ModuleType(name)
    try:
        assert evict_variable_modules_when_freed(plain, set()) == 0
        assert name in sys.modules
    finally:
        sys.modules.pop(name, None)


def test_a_build_failure_propagates() -> None:
    def default_tax_benefit_system(*, reform=None):
        raise RuntimeError("fixture build failure")

    engine = SimpleNamespace(default_tax_benefit_system=default_tax_benefit_system)
    with pytest.raises(RuntimeError, match="fixture build failure"):
        build_reform_tax_benefit_system(engine, None)


def test_the_finalizer_keeps_no_module_alive() -> None:
    """Eviction holds weak references: once the system is collected and its
    modules evicted, a full collection frees the modules too."""
    system = build_reform_tax_benefit_system(_engine(n_modules=3), None)
    module_refs = [weakref.ref(sys.modules[name]) for name in system.module_names]
    del system
    gc.collect()
    gc.collect()
    assert all(ref() is None for ref in module_refs)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    new_prefixed=st.integers(min_value=0, max_value=8),
    preexisting_prefixed=st.integers(min_value=0, max_value=3),
    other_new=st.integers(min_value=0, max_value=3),
    replaced=st.integers(min_value=0, max_value=8),
)
def test_eviction_removes_exactly_the_systems_own_unreplaced_modules(
    new_prefixed, preexisting_prefixed, other_new, replaced
) -> None:
    """For any mix of names: after the system is collected, exactly its own
    new, still-registered modules are gone and every other entry of
    ``sys.modules`` is as it was."""
    replaced = min(replaced, new_prefixed)
    system = _System()
    system.cycle = system
    prefix = f"{id(system)}_"
    added: list[str] = []
    try:
        preexisting = [f"{prefix}pre_{index}" for index in range(preexisting_prefixed)]
        for name in preexisting:
            sys.modules[name] = types.ModuleType(name)
            added.append(name)
        modules_before = set(sys.modules)
        own = [f"{prefix}{index}_variable" for index in range(new_prefixed)]
        others = [f"fixture_other_{id(system)}_{index}" for index in range(other_new)]
        for name in (*own, *others):
            sys.modules[name] = _variable_module(name)
            added.append(name)

        assert evict_variable_modules_when_freed(system, modules_before) == len(own)

        replacements = {}
        for name in own[:replaced]:
            replacements[name] = types.ModuleType(name)
            sys.modules[name] = replacements[name]
        snapshot = {
            name: module for name, module in sys.modules.items() if name not in own
        }
        del system
        gc.collect()

        for name in own[replaced:]:
            assert name not in sys.modules
        for name, module in replacements.items():
            assert sys.modules[name] is module
        for name, module in snapshot.items():
            assert sys.modules.get(name) is module, name
    finally:
        for name in added:
            sys.modules.pop(name, None)
