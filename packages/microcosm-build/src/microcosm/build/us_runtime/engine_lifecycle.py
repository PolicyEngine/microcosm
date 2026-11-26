"""Engine simulation teardown for bounded build memory (microcosm#456).

policyengine-core keeps a finished simulation's multi-GB object graph
reachable long after the builder drops its last name for it:

- every construction sets ``tax_benefit_system.simulation = <sim>`` — for
  non-reform simulations that pins the sim to the *immortal* shared
  ``default_tax_benefit_system_instance``;
- reform simulations clone a ``baseline`` branch whose
  ``parent_branch``/``branches`` links form cycles, and whose holder storage
  *copies* every input array;
- ``self.calc = self.calculate`` stores a bound method on the instance — a
  guaranteed self-cycle on every simulation.

policyengine-us 2.2.1 adds one more holder: the module-level cache of rules
pinned to earlier federal law that New York's credit formulas fill, whose
entries keep the engine they were made in (see :func:`_drop_pinned_rule_copies`).

Cyclic graphs are invisible to the builder's cheap per-batch ``gc.collect(0)``
once anything promotes them past generation 0, and CPython throttles full
collections against the build's multi-GB long-lived heap — the measured result
was unbounded accumulation (~2.5 GB/min) through the dense target
materializations. :func:`release_engine_simulation` frees the array mass by
*refcount* instead: it drops holder maps, dataset references, and branch
links, and severs the backrefs, so only a small cyclic skeleton is left for
ordinary collection.

Every access is defensive (``getattr``/instance-dict checks): the helper runs
against engine versions that may drift and against test stubs — releasing
less is survivable, raising mid-build is not. It must never be the thing that
kills an 8-hour run.
"""

from __future__ import annotations

import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "EngineVariableModuleScope",
    "release_engine_simulation",
    "temporary_engine_variable_modules",
]


def _loaded_variables_dir() -> str | Path | None:
    engine = sys.modules.get("policyengine_us")
    system_cls = getattr(engine, "CountryTaxBenefitSystem", None)
    return getattr(system_cls, "variables_dir", None)


def _registrations(directory: Path | None) -> dict[str, ModuleType]:
    """Core's per-system variable-module registrations under ``directory``."""
    found = {}
    for name, module in tuple(sys.modules.items()):
        parts = name.split("_", 2)
        if (
            len(parts) != 3
            or not parts[0].isdigit()
            or not parts[1].removeprefix("-").isdigit()
            or not isinstance(module, ModuleType)
        ):
            continue
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            continue
        path = Path(filename).absolute()
        if parts[2] == path.stem and (
            directory is None or path.is_relative_to(directory)
        ):
            found[name] = module
    return found


# Scopes opened and not yet closed, oldest first, as weak references: a scope
# dropped without closing stops counting once it is collected.
_OPEN_SCOPES: list[weakref.ref[EngineVariableModuleScope]] = []
# Reentrant: a garbage collection that frees a suspended
# ``temporary_engine_variable_modules`` scope runs its ``close`` in whichever
# thread is running, possibly one that holds this lock inside ``open``.
_OPEN_SCOPES_LOCK = threading.RLock()


class EngineVariableModuleScope:
    """Remove Core's variable-module registrations made between open and close.

    The explicit form of :func:`temporary_engine_variable_modules`, for a scope
    that spans several calls: post-export scoring builds a reform's system on
    the reform's first key and closes the scope when its consumer moves on. A
    suspended context manager would run its cleanup whenever the garbage
    collector closed it, at an arbitrary later moment, and remove modules a
    later scope registered and still uses. A scope left unclosed here does
    nothing when collected; its modules simply stay registered, as they would
    without it. ``close`` does nothing the second time.

    Scopes may close in any order. A registration belongs to the most recently
    opened scope that is still open when it is made: a scope that closes
    removes what was registered since it opened, except what was registered
    after a later, still-open scope opened, which that scope removes when it
    closes. An entry that existed when the scope opened is put back only if
    the scope is the latest open one and the entry is still registered (Core
    reused a dead system's id and overwrote it); otherwise the later scope,
    whose snapshot holds it too, decides.
    """

    def __init__(self, variables_dir: str | Path | None = None) -> None:
        self._variables_dir = variables_dir
        self._previous: dict[str, ModuleType] | None = None
        self._closed = False

    def open(self) -> EngineVariableModuleScope:
        """Take the snapshot of registrations that ``close`` keeps."""
        if self._previous is not None or self._closed:
            raise RuntimeError("An engine variable module scope opens only once.")
        initial_directory = self._variables_dir or _loaded_variables_dir()
        directory = (
            None if initial_directory is None else Path(initial_directory).absolute()
        )
        with _OPEN_SCOPES_LOCK:
            self._previous = _registrations(directory)
            _OPEN_SCOPES.append(weakref.ref(self))
        return self

    def _first_later_open_scope_snapshot(self) -> dict[str, ModuleType] | None:
        """The snapshot of the oldest scope opened after this one and still open."""
        later = False
        snapshot = None
        # A copy: a close that a garbage collection runs from in here may change
        # the list.
        for reference in tuple(_OPEN_SCOPES):
            scope = reference()
            if scope is self:
                later = True
            elif (
                later
                and snapshot is None
                and scope is not None
                and scope._previous is not None
            ):
                snapshot = scope._previous
        _OPEN_SCOPES[:] = [
            reference
            for reference in _OPEN_SCOPES
            if reference() is not None and reference() is not self
        ]
        return snapshot

    def close(self) -> None:
        """Remove the registrations made since ``open``, except the shared system's."""
        with _OPEN_SCOPES_LOCK:
            previous, self._previous = self._previous, None
            if self._closed or previous is None:
                self._closed = True
                return
            self._closed = True
            later_snapshot = self._first_later_open_scope_snapshot()
            self._remove_since(previous, later_snapshot)

    def _remove_since(
        self,
        previous: dict[str, ModuleType],
        later_snapshot: dict[str, ModuleType] | None,
    ) -> None:
        final_directory = self._variables_dir or _loaded_variables_dir()
        if final_directory is None:
            return
        # The first slice may import the engine lazily. Its newly built shared
        # default system remains live for every later simulation; retain those
        # registrations along with the entry snapshot.
        engine = sys.modules.get("policyengine_us")
        simulation_cls = getattr(engine, "Microsimulation", None)
        shared_system = getattr(
            simulation_cls, "default_tax_benefit_system_instance", None
        )
        shared_prefix = f"{id(shared_system)}_"
        for name in _registrations(Path(final_directory).absolute()):
            if name in previous:
                if later_snapshot is not None:
                    # A later open scope's snapshot holds it too; it decides.
                    continue
                current = sys.modules.get(name)
                # Not if another scope removed it meanwhile (a garbage
                # collection can close one in the middle of this loop).
                if current is not None and current is not previous[name]:
                    sys.modules[name] = previous[name]
            elif later_snapshot is not None and name not in later_snapshot:
                # Registered after a still-open later scope opened.
                continue
            elif shared_system is None or not name.startswith(shared_prefix):
                sys.modules.pop(name, None)


@contextmanager
def temporary_engine_variable_modules(
    variables_dir: str | Path | None = None,
) -> Iterator[None]:
    """Release Core's transient variable-module registrations after scoring.

    Core's ``TaxBenefitSystem.add_variables_from_file`` registers each module
    as ``<system id>_<path hash>_<file stem>`` in ``sys.modules`` and never
    removes it. Rebuilding metadata/reform systems for independent slices
    therefore retains their modules, classes, and formula globals forever.

    Keep registrations available throughout the calculation (including
    engine source inspection), then remove only this scope's registrations
    under the US variable directory. Shared systems' pre-existing modules
    and normal imports remain registered. Restore a pre-existing entry if
    Core reused a dead system's id and overwrote it. No policy or simulation
    cache, formula, or scored value is changed.

    The default discovers an already imported US engine, so importing this
    helper remains engine-free. An explicit directory supports stand-ins.
    Use only around a completed slice whose engine objects do not escape; for
    a scope that spans calls, use :class:`EngineVariableModuleScope`.
    """

    scope = EngineVariableModuleScope(variables_dir).open()
    try:
        yield
    finally:
        scope.close()


#: Instance attributes whose only post-mortem job is to keep big object
#: graphs alive: the (multi-year) dataset tables, memoized short-path results,
#: the tracer's recorded calculations, and the ``calc``/``df`` bound-method
#: self-cycles.
_SEVERED_ATTRIBUTES = ("dataset", "_fast_cache", "tracer", "calc", "df")


def _sever_existing_attribute(target: Any, name: str) -> None:
    """Set ``target.<name> = None`` only when the instance itself carries it."""
    instance_dict = getattr(target, "__dict__", None)
    if not isinstance(instance_dict, dict) or name not in instance_dict:
        return
    try:
        setattr(target, name, None)
    except AttributeError:  # pragma: no cover - read-only stub attribute
        pass


#: The policyengine-us module whose ``_PINNED_TBS_CACHE`` holds the systems
#: that New York's credit formulas pin to earlier federal rules.
_PINNED_RULES_MODULE = "policyengine_us.tools.pinned_tbs"


#: How many ``parent_branch`` links :func:`_drop_pinned_rule_copies` follows
#: from the simulation a pinned copy was made in. policyengine-us formulas
#: branch at most a few levels deep; the bound only guards against a cycle.
_MAXIMUM_BRANCH_DEPTH = 64


def _made_in_released_simulation(clone: Any, simulations: list[Any]) -> bool:
    """Whether ``clone`` was made in one of ``simulations`` or a branch of one.

    Core's ``TaxBenefitSystem.clone`` copies the ``simulation`` attribute, so a
    pinned copy names the simulation its lookup ran in. That may be a branch a
    formula has already removed from ``branches`` (policyengine-us's marginal
    tax rates make a branch per adult, compute household income in it and
    delete it), which the release walk never reaches; its ``parent_branch``
    still leads to the simulation it was made from.

    An attribute read that raises (a dead weak proxy, or a property of another
    engine version) answers ``False``: the entry is left alone.
    """
    try:
        current = getattr(clone, "simulation", None)
        for _ in range(_MAXIMUM_BRANCH_DEPTH):
            if current is None:
                return False
            if any(current is simulation for simulation in simulations):
                return True
            current = getattr(current, "parent_branch", None)
    except Exception:
        return False
    return False


def _drop_pinned_rule_copies(systems: list[Any], simulations: list[Any]) -> None:
    """Remove the pinned-rules cache entries made in released simulations.

    In policyengine-us 2.2.1, New York's pre-2024 child tax credit formulas
    (``ny_ctc_pre_2024_eligible`` and ``ny_ctc_pre_2024``, from 2018 except
    2025-2027) and its 2021 EITC (``ny_eitc``) recompute federal credits under
    earlier rules through ``policyengine_us.tools.pinned_tbs``. A lookup that
    misses clones the simulation's whole tax-benefit system and stores
    ``(weak reference to that system, weak reference to its parameter tree,
    clone)`` under the pin's name in the module-level ``_PINNED_TBS_CACHE``.
    The clone is held strongly, and core's ``TaxBenefitSystem.clone`` copies
    the ``simulation`` attribute into it. So the entry keeps the finished
    engine alive and, through the engine's system, that system's parameter
    tree, variables and rule modules, until another engine's lookup replaces
    the entry: about 1.3 GiB after a reform pass, 0.5 GiB after a baseline
    pass.

    ``systems`` and ``simulations`` are what the release walked. An entry is
    removed when its first item is a weak reference to one of ``systems``, or
    when its clone was made in one of ``simulations`` or in a branch of one
    that a formula removed from ``branches``
    (:func:`_made_in_released_simulation`). A lookup reuses an entry only for
    the same system object it was pinned from, and each policyengine-us
    simulation and branch gets a system object of its own, so no later engine
    can reuse a removed entry and no scored value changes. (Were a system
    shared by a simulation still running, its next lookup would clone again,
    as any miss does.) An entry whose base system is already collected is
    still removed when its clone was made in a released simulation. Entries
    made elsewhere, dead or not, a missing cache, a cache that is not a dict
    and entries without a weak reference first are left alone. The module is
    not imported here.

    policyengine-us 2.88.6 holds every part of an entry weakly
    (PolicyEngine/policyengine-us#10550). Its entries are matched by their
    first item the same way, and removing them frees nothing. Remove this
    when the US engine pin reaches 2.88.6.
    """
    module = sys.modules.get(_PINNED_RULES_MODULE)
    cache = getattr(module, "_PINNED_TBS_CACHE", None)
    if not systems or not isinstance(cache, dict):
        return
    for pin_name, entry in tuple(cache.items()):
        if not isinstance(entry, tuple) or not entry:
            continue
        base_reference = entry[0]
        if not isinstance(base_reference, weakref.ReferenceType):
            continue
        base = base_reference()
        pinned_from_a_released_system = base is not None and any(
            base is system for system in systems
        )
        clone = entry[2] if len(entry) > 2 else None
        if (
            pinned_from_a_released_system
            or _made_in_released_simulation(clone, simulations)
        ) and cache.get(pin_name) is entry:
            del cache[pin_name]


def release_engine_simulation(simulation: Any) -> None:
    """Release a finished engine simulation's retained state.

    Walks the simulation and every branch clone reachable from it
    (``branches`` values, ``baseline``), and for each:

    - clears every population's holder map (the array mass — freed by
      refcount, no cyclic collection needed);
    - severs population → simulation backrefs;
    - severs the tax-benefit system's ``simulation`` backref *if it points at
      the simulation being released* (never unconditionally: the shared
      class-level system instance may already belong to a newer, live
      simulation);
    - drops dataset/tracer/bound-method attributes that only extend the
      graph's lifetime.

    Then it removes the entries policyengine-us's pinned-rules cache holds for
    copies made in those simulations, or in branches of them that a formula
    already removed (:func:`_drop_pinned_rule_copies`).

    Safe to call on stubs and doubles: only attributes the object actually
    carries are touched. Idempotent.
    """
    stack = [simulation]
    seen: set[int] = set()
    released_systems: list[Any] = []
    released_simulations: list[Any] = []
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        released_simulations.append(current)

        branches = getattr(current, "branches", None)
        if isinstance(branches, dict):
            stack.extend(branches.values())
            branches.clear()
        baseline = getattr(current, "baseline", None)
        if baseline is not None and not isinstance(baseline, type):
            stack.append(baseline)
        _sever_existing_attribute(current, "baseline")
        _sever_existing_attribute(current, "parent_branch")

        populations = getattr(current, "populations", None)
        if isinstance(populations, dict):
            for population in populations.values():
                holders = getattr(population, "_holders", None)
                if isinstance(holders, dict):
                    holders.clear()
                _sever_existing_attribute(population, "simulation")
            populations.clear()

        tax_benefit_system = getattr(current, "tax_benefit_system", None)
        if tax_benefit_system is not None:
            released_systems.append(tax_benefit_system)
        if (
            tax_benefit_system is not None
            and getattr(tax_benefit_system, "simulation", None) is current
        ):
            try:
                tax_benefit_system.simulation = None
            except AttributeError:  # pragma: no cover - read-only stub
                pass

        for attribute in _SEVERED_ATTRIBUTES:
            _sever_existing_attribute(current, attribute)

    _drop_pinned_rule_copies(released_systems, released_simulations)
