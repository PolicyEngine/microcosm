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
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["release_engine_simulation", "temporary_engine_variable_modules"]


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
    Use only around a completed slice whose engine objects do not escape.
    """

    def loaded_variables_dir():
        engine = sys.modules.get("policyengine_us")
        system_cls = getattr(engine, "CountryTaxBenefitSystem", None)
        return getattr(system_cls, "variables_dir", None)

    initial_directory = variables_dir or loaded_variables_dir()
    directory = (
        None if initial_directory is None else Path(initial_directory).absolute()
    )

    def registrations(directory: Path | None) -> dict[str, ModuleType]:
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

    previous = registrations(directory)
    try:
        yield
    finally:
        final_directory = variables_dir or loaded_variables_dir()
        if final_directory is not None:
            # The first slice may import the engine lazily. Its newly built
            # shared default system remains live for every later simulation;
            # retain those registrations along with the entry snapshot.
            engine = sys.modules.get("policyengine_us")
            simulation_cls = getattr(engine, "Microsimulation", None)
            shared_system = getattr(
                simulation_cls, "default_tax_benefit_system_instance", None
            )
            shared_prefix = f"{id(shared_system)}_"
            for name in registrations(Path(final_directory).absolute()):
                if name in previous:
                    sys.modules[name] = previous[name]
                elif shared_system is None or not name.startswith(shared_prefix):
                    del sys.modules[name]


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
