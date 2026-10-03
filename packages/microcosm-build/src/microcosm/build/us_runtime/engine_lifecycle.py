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

A tax-benefit system outlives itself in ``sys.modules`` as well: building one
executes every variable file as a new module registered under a name unique
to that system, and nothing removes those modules when the system is freed.
:func:`build_reform_tax_benefit_system` builds a reform system whose variable
modules are evicted when it is collected (see
:func:`evict_variable_modules_when_freed`).
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Collection, Mapping
from typing import Any

__all__ = [
    "build_reform_tax_benefit_system",
    "evict_variable_modules_when_freed",
    "release_engine_simulation",
]

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

    Safe to call on stubs and doubles: only attributes the object actually
    carries are touched. Idempotent.
    """
    stack = [simulation]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

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


def _evict_modules(registered: Mapping[str, weakref.ref]) -> None:
    """Remove each still-registered module of a freed system from ``sys.modules``."""
    try:
        for name, module_ref in registered.items():
            module = module_ref()
            if module is not None and sys.modules.get(name) is module:
                sys.modules.pop(name, None)
    except Exception:  # pragma: no cover - a finalizer must not raise
        pass


def evict_variable_modules_when_freed(
    system: Any, modules_before: Collection[str]
) -> int:
    """Evict ``system``'s variable modules from ``sys.modules`` once it is freed.

    policyengine-core's ``TaxBenefitSystem.add_variables_from_file`` executes
    each variable file of a system being built as a new module named
    ``f"{id(system)}_{hash(path)}_{file name}"`` and registers it in
    ``sys.modules``; nothing ever removes it. The modules hold the system's
    ``Variable`` classes, formula functions and code objects, so every system
    built leaves them behind for the rest of the process after the system
    itself is collected (policyengine-us 2.2.1 has 5,990 variable files, and
    microcosm#456 measured a ~55-60 MB resident floor per build).

    This system's modules are the names absent from ``modules_before`` that
    start with ``f"{id(system)}_"``; any other module first imported during
    the build is left alone. A finalizer on ``system`` removes each of them
    when the system is collected, unless ``sys.modules`` by then maps the name
    to a different module. Until then nothing changes, so the system itself
    never runs without its modules. No module is kept alive by this (the
    finalizer holds weak references).

    Returns the number of modules arranged for eviction: 0 when the engine
    registered none under that name, or when ``system`` cannot be weakly
    referenced. Never raises.
    """
    try:
        prefix = f"{id(system)}_"
        registered: dict[str, weakref.ref] = {}
        for name, module in list(sys.modules.items()):
            if (
                isinstance(name, str)
                and name.startswith(prefix)
                and name not in modules_before
            ):
                try:
                    registered[name] = weakref.ref(module)
                except TypeError:  # pragma: no cover - not weak-referenceable
                    continue
        if not registered:
            return 0
        finalizer = weakref.finalize(system, _evict_modules, registered)
        # At interpreter exit the modules go anyway.
        finalizer.atexit = False
        return len(registered)
    except Exception:
        return 0


def build_reform_tax_benefit_system(microsimulation_cls: Any, reform: Any) -> Any:
    """``microsimulation_cls.default_tax_benefit_system(reform=reform)``, with
    the system's variable modules evicted from ``sys.modules`` once the system
    is freed (:func:`evict_variable_modules_when_freed`).

    A build failure propagates unchanged.
    """
    modules_before = set(sys.modules)
    system = microsimulation_cls.default_tax_benefit_system(reform=reform)
    evict_variable_modules_when_freed(system, modules_before)
    return system
