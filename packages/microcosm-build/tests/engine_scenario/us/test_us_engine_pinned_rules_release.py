"""Releasing an engine frees the rules policyengine-us pins for New York.

New York's pre-2024 CTC formulas look up rules pinned to 2017 federal law
through ``policyengine_us.tools.pinned_tbs``. In policyengine-us 2.2.1 a lookup
that misses clones the simulation's whole rules system into the module-level
``_PINNED_TBS_CACHE``, and the clone's copied ``simulation`` attribute keeps
the engine it was made in alive (``release_engine_simulation`` removes the
entry). The marginal tax rate formula makes that lookup inside a branch it
deletes again, so the release never walks the branch it was made in.
"""

from __future__ import annotations

import gc
import weakref

from microcosm.build.us_runtime.engine_lifecycle import release_engine_simulation


def _new_york_household() -> dict[str, object]:
    year = "2024"
    members = ["parent", "child"]
    return {
        "people": {
            "parent": {"age": {year: 35}, "employment_income": {year: 30_000}},
            "child": {"age": {year: 5}},
        },
        "tax_units": {"tax_unit": {"members": members}},
        "spm_units": {"spm_unit": {"members": members}},
        "families": {"family": {"members": members}},
        "marital_units": {
            "parent_marital_unit": {"members": ["parent"]},
            "child_marital_unit": {"members": ["child"]},
        },
        "households": {"household": {"members": members, "state_code": {year: "NY"}}},
    }


def test_release_frees_rules_pinned_in_a_branch_the_formula_deleted(
    monkeypatch,
) -> None:
    from policyengine_us import Simulation
    from policyengine_us.tools import pinned_tbs

    branch_lookups: list[str] = []
    original_lookup = pinned_tbs._get_pinned_tbs

    def counted_lookup(base_tbs, pin_name, pin_function):
        simulation = getattr(base_tbs, "simulation", None)
        if getattr(simulation, "parent_branch", None) is not None:
            branch_lookups.append(getattr(simulation, "branch_name", ""))
        return original_lookup(base_tbs, pin_name, pin_function)

    monkeypatch.setattr(pinned_tbs, "_get_pinned_tbs", counted_lookup)
    simulation = Simulation(situation=_new_york_household())
    simulation.calculate("marginal_tax_rate", 2024)
    assert any(name.startswith("mtr_for_adult_") for name in branch_lookups)
    assert not any(name.startswith("mtr_for_adult_") for name in simulation.branches)
    released = weakref.ref(simulation)

    release_engine_simulation(simulation)
    del simulation
    gc.collect()

    assert released() is None
