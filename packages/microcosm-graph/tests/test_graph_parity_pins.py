"""Every pinned H1 platform key is checkable from every platform.

Charter H1 asserts a platform-bitwise kernel's *bytes* only on a platform that
carries a pin, and falls back to identity partitioning elsewhere. That is the
right rule for bytes — this machine cannot produce another architecture's
floats — but it leaves the foreign pins' *keys* unasserted everywhere except on
those machines, so a stale one survives until that CI lane happens to run.

A node key is not a measurement. It is a pure function of the declaration, the
resolved input identities, the implementation hash, the capability projection,
and, for a platform-bitwise kernel (amendment 16), the platform fingerprint
string. So every pinned key is derivable here, and these tests derive all of
them. A kernel edit that re-pins only the local platform fails immediately
rather than on the next Linux run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.graph import graph_from_json, platform_fingerprint
from tools.graph_parity_fixtures import FIXTURES, parity_registry
from tools.graph_parity_repin import derived_node_key

CASES = ("fit.qrf", "calibrate", "simulate")


def _case(name: str) -> tuple[Path, dict]:
    case = FIXTURES / name
    return case, json.loads((case / "pins.json").read_text())


def _platforms(pins: dict) -> dict[str, dict]:
    platforms = dict(pins.get("platforms", {}))
    platforms.setdefault(
        pins["platform"], {"node_key": pins["node_key"], "direct": "direct.csv"}
    )
    return platforms


@pytest.mark.parametrize("name", CASES)
def test_every_pinned_platform_key_is_the_one_that_platform_would_compute(
    name: str,
) -> None:
    case, pins = _case(name)
    graph = graph_from_json((case / "graph.json").read_text())
    for fingerprint, entry in sorted(_platforms(pins).items()):
        derived = derived_node_key(
            graph, pins["node"], case / "inputs.csv", fingerprint
        )
        assert derived == entry["node_key"], (
            f"{name}: the pin for {fingerprint} is stale. Re-pin every platform "
            f"with `uv run python tools/graph_parity_repin.py {name}` — a "
            "kernel edit moves every platform's key, not only this machine's."
        )


@pytest.mark.parametrize("name", CASES)
def test_the_top_level_pin_is_the_authoring_platform_and_its_bytes_exist(
    name: str,
) -> None:
    case, pins = _case(name)
    platforms = _platforms(pins)
    # Read the raw mapping, not the one ``_platforms`` back-fills, or the
    # assertion would be about this helper rather than about the fixture.
    assert pins["platform"] in pins["platforms"]
    assert pins["platforms"][pins["platform"]]["node_key"] == pins["node_key"]
    for entry in platforms.values():
        direct = case / entry["direct"]
        assert direct.is_file(), f"{name}: {entry['direct']} is pinned but absent"
        assert direct.read_bytes(), f"{name}: {entry['direct']} is empty"


@pytest.mark.parametrize("name", CASES)
def test_the_pinned_implementation_hash_is_the_registered_kernel_s(name: str) -> None:
    """A moved kernel must re-pin; H1 checks this too, without the platforms."""
    _, pins = _case(name)
    kernel = parity_registry().get(pins["kernel"])
    assert kernel.implementation_hash() == pins["implementation_hash"]
    assert set(pins["dependencies"]) == set(kernel.capabilities.dependencies)


def test_platform_bitwise_pins_partition_identity_across_platforms() -> None:
    """Amendment 16: a shared store never serves another platform's output."""
    _, pins = _case("fit.qrf")
    assert pins["numeric"] == "platform_bitwise"
    platforms = _platforms(pins)
    assert len(platforms) >= 2, "the whole point is more than one platform"
    keys = [entry["node_key"] for entry in platforms.values()]
    assert len(set(keys)) == len(keys)


def test_a_derived_key_is_the_local_platform_s_own_key() -> None:
    """The derivation is not a second implementation of ``node_key``.

    Deriving every platform's key would be circular if the derivation were only
    ever checked against itself. It is not: on whatever platform this runs,
    ``test_h1_kernel_parity`` executes the graph and asserts that the executor's
    key equals this same local pin, so agreeing with the local pin here anchors
    the derivation to the executor. (The local pin is produced by a run on the
    authoring platform and derived elsewhere, which is exactly why the anchor
    has to come from an executed key rather than from the pin's provenance.)
    """
    case, pins = _case("fit.qrf")
    local = platform_fingerprint()
    platforms = _platforms(pins)
    if local not in platforms:
        pytest.skip(f"{local} carries no fit.qrf pin")
    graph = graph_from_json((case / "graph.json").read_text())
    assert (
        derived_node_key(graph, pins["node"], case / "inputs.csv", local)
        == platforms[local]["node_key"]
    )
