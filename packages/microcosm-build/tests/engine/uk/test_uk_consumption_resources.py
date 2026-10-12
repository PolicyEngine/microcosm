"""Tests split from packages/microcosm-build/tests/test_uk_consumption_resources.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_consumption_resources import *


def test_policy_anchor_values_lockstep_with_engine_parameter_tree() -> None:
    """Every anchor value with a parameter_path equals the installed tree's value.

    This is the Option A drift guard adjudicated on microcosm#682: an engine
    bump that moves one of these historical values fails here and forces a
    reviewed resource diff. Skips where policyengine-uk is absent — PR CI's
    hermetic lanes never import the engine.
    """
    import importlib

    system_module = importlib.import_module("policyengine_uk.system")
    parameters = system_module.system.parameters
    vat = _load("etb_policy_anchors.json")["vat"]
    for name, anchor in vat.items():
        node = parameters
        for part in anchor["parameter_path"].split("."):
            node = getattr(node, part)
        assert float(node(str(anchor["period"]))) == anchor["value"], name

    services = _load("etb_services_anchors.json")
    for key in ("rail_fare_index_2023", "rail_fare_index_2024"):
        anchor = services[key]
        node = parameters
        for part in anchor["parameter_path"].split("."):
            node = getattr(node, part)
        assert float(node(str(anchor["period"]))) == anchor["value"], key
