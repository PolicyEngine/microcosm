"""Releasing an engine frees the rules policyengine-us pins for New York.

New York's pre-2024 CTC formulas look up rules pinned to 2017 federal law
through ``policyengine_us.tools.pinned_tbs``. In policyengine-us 2.2.1 a lookup
that misses clones the simulation's whole rules system into the module-level
``_PINNED_TBS_CACHE``, and the clone's copied ``simulation`` attribute keeps
the engine it was made in alive (``release_engine_simulation`` removes the
entry). The marginal tax rate formula makes that lookup inside a branch it
deletes again, so the release never walks the branch it was made in. On
policyengine-us 2.88.6 or later the cache holds nothing strongly, so this
holds there without the removal too.
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

    # Pin name -> where the newest pinned system was made: the branch name
    # and whether that branch is a child of another simulation. The cache
    # keeps only the newest per pin name. Weak, so recording keeps nothing.
    newest: dict[str, tuple[weakref.ref, str, bool]] = {}
    original_lookup = pinned_tbs._get_pinned_tbs

    def recorded_lookup(base_tbs, pin_name, pin_function):
        pinned = original_lookup(base_tbs, pin_name, pin_function)
        previous = newest.get(pin_name)
        if previous is None or previous[0]() is not pinned:
            made_in = getattr(base_tbs, "simulation", None)
            newest[pin_name] = (
                weakref.ref(pinned),
                str(getattr(made_in, "branch_name", "")),
                getattr(made_in, "parent_branch", None) is not None,
            )
        return pinned

    monkeypatch.setattr(pinned_tbs, "_get_pinned_tbs", recorded_lookup)
    simulation = Simulation(situation=_new_york_household())
    simulation.calculate("marginal_tax_rate", 2024)
    _, branch_name, made_in_a_branch = newest["ny_pre_tcja_ctc"]
    # The precondition: the newest pinned system was made in a marginal tax
    # rate branch the formula has already deleted.
    assert made_in_a_branch and branch_name.startswith("mtr_for_adult_")
    assert branch_name not in simulation.branches
    released = weakref.ref(simulation)

    release_engine_simulation(simulation)
    del simulation
    gc.collect()

    assert released() is None
