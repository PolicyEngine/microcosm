"""Every declared implementation contract still describes the tree it names.

``graph_implementation.implementation_manifest`` refuses a stage whose module
carries an import, unbound use or resource access the packaged inventory does
not declare (``graph_implementation.py:432-434``), and a stage kernel's
``implementation_hash`` goes through it -- so a stale contract refuses every
graph run that reaches that stage, not merely a review.

The repository had no test over the whole inventory: the one existing caller
(``test_us_asec_prepared_resources.py``) builds three of the ten stages. A
``Path.read_bytes`` added to ``survey_population_preparation._spill_roster``
on ``native-scale-transport`` (commit ``b6081efcb``) therefore left
``authenticated_survey_population_v1`` refusing with no test red, which is the
defect this file exists to make impossible.

Nothing here reads gated data, runs a graph or needs a country engine: it
hashes the packaged source files the inventory already names.
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime import graph_implementation


def _inventory():
    roots = graph_implementation._package_roots()
    inventory, _ = graph_implementation._inventory(roots)
    return roots, inventory


def _stages():
    return sorted(_inventory()[1]["stages"])


def _contracts():
    return sorted(_inventory()[1]["contracts"])


@pytest.mark.parametrize("name", _contracts())
def test_every_declared_contract_matches_the_packaged_module(name):
    roots, inventory = _inventory()
    package, relative = name.split("/", 1)
    if package not in roots:
        pytest.skip(f"optional dependency package absent: {package}")
    payload = (roots[package] / relative).read_bytes()
    actual = graph_implementation._dependency_contract(
        payload, name, graph_implementation._covered_imports(name, inventory)
    )
    assert actual == inventory["contracts"][name]


@pytest.mark.parametrize("stage", _stages())
def test_every_declared_stage_manifest_builds(stage):
    """A stage kernel's ``implementation_hash`` is this call; it must not raise."""
    manifest = graph_implementation.implementation_manifest(stage)
    assert manifest["stage"] == stage
    assert manifest["schema"] == "microcosm.us.implementation-manifest.v1"
    assert manifest["modules"]


def test_the_inventory_declares_a_contract_for_every_rostered_module():
    _, inventory = _inventory()
    rostered = {
        name for spec in inventory["stages"].values() for name in spec["modules"]
    }
    assert rostered <= set(inventory["contracts"])
