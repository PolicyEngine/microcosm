#!/usr/bin/env python3
"""Re-record one H1 parity case's pins after an additive change to its kernel.

``tools/graph_parity_fixtures.py`` regenerates a fixture from scratch. That is
the wrong instrument for a change that leaves a kernel's outputs alone but moves
its implementation hash, for two reasons:

1. ``generate()`` rewrites ``pins["platforms"]`` as the local platform alone, so
   running it here would silently drop the two pinned ``x86_64/linux`` entries
   and leave charter H1 asserting no bytes on those platforms — its off-platform
   branch asserts only identity partitioning.
2. ``ParityCsvSource`` and ``ParityRulesEngine`` are *defined in* that module, so
   its bytes are inside ``ParityCsvSource.implementation_hash()`` and
   ``SimulateRulesKernel.implementation_hash()``. Editing it to add a re-pin
   path would move every parity node key in all three cases — churn caused by
   the tooling rather than by the kernel under amendment. This module therefore
   sits beside it and imports, so those bytes never move.

What a re-pin may and may not derive: the local platform's node key is
**produced**, by running the graph. Every other pinned platform's key is
**derived**, because a node key is a pure function of the declaration, the
resolved input identities, the implementation hash, the capability projection,
and — for a platform-bitwise kernel (amendment 16) — the platform fingerprint
*string* (``microcosm.graph.keys.node_key``).

The fingerprint is not the *only* platform-dependent input, and this does not
pretend otherwise: ``source_hash`` folds ``f"{distribution}=={version}"`` for
each declared dependency into the implementation hash, so a platform on a
different locked environment keys differently for a reason no fingerprint
string records. A re-pin therefore establishes the condition it needs, in two
checks that close different holes:

1. ``pins["dependencies"]`` — the versions the existing keys were taken under —
   must equal this machine's installed versions. This is the check that sees
   the dependency channel. The reproduction below cannot see it, because
   substituting the recorded implementation hash removes the only input those
   versions reach.
2. Every pinned platform's key must reproduce from
   ``pins["implementation_hash"]``. This catches an entry inconsistent with the
   hash recorded beside it — a hand edit, or a platform pinned while the
   generator's own bytes had moved — which comparing versions would not.

Together they say the pinned platforms and this one shared one locked
environment at pin time, and every pin is the key that environment computed.
That is what licenses deriving the new foreign keys here. The local key is
additionally derived alongside being executed, so the derivation cannot drift
from what the executor computes.

Pinned **bytes** are never derived: each platform's ``direct.csv`` is left
exactly as that platform recorded it, and a re-pin refuses to write at all if
the local direct call's bytes moved from the file **this** platform's pin points
at, because that would mean the change was not additive and the fixture needs a
real regeneration instead.

Usage::

    uv run python tools/graph_parity_repin.py <case>
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Iterator
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from microcosm.graph import ContentStore, Graph, compile_graph, graph_to_json, run_graph
from microcosm.graph import keys as graph_keys
from microcosm.graph.keys import platform_fingerprint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Importable both as ``tools.graph_parity_repin`` and as a bare script path.
    sys.path.insert(0, str(ROOT))

# The pin schema and the case declarations belong to the generator; importing
# them is what keeps a re-pinned fixture indistinguishable from a generated one.
from tools.graph_parity_fixtures import (  # noqa: E402 - after the path bootstrap
    FIXTURES,
    _calibrate_case,
    _fit_case,
    _pins,
    _simulate_case,
    _write_pins,
    parity_registry,
)

__all__ = [
    "BUILDERS",
    "derived_node_key",
    "installed_dependency_versions",
    "repin",
]

BUILDERS = {
    "fit.qrf": _fit_case,
    "calibrate": _calibrate_case,
    "simulate": _simulate_case,
}


@contextlib.contextmanager
def _as_platform(fingerprint: str) -> Iterator[None]:
    """Compute keys as ``fingerprint`` would, without being that platform.

    The fingerprint reaches a key only as the string ``node_key`` appends for a
    platform-bitwise kernel, so the string is the only thing to vary. See the
    module docstring for why deriving a key is sound and deriving bytes is not.
    """
    real = graph_keys.platform_fingerprint
    graph_keys.platform_fingerprint = lambda: fingerprint
    try:
        yield
    finally:
        graph_keys.platform_fingerprint = real


def derived_node_key(
    graph: Graph,
    node_id: str,
    inputs_path: Path,
    fingerprint: str,
    implementation_hash: str | None = None,
) -> str:
    """The key ``fingerprint`` computes for ``node_id`` over ``inputs_path``.

    ``implementation_hash`` substitutes that value for ``node_id``'s own kernel,
    so a caller can ask what a platform keyed under a *previous* implementation
    — the one its existing pin was taken under — and check the answer against
    that pin. Every other node keeps its registered kernel's hash.
    """
    compiled = compile_graph(graph)
    source_keys = {
        source.name: graph_keys.source_content_key(source.name, inputs_path)
        for source in graph.sources
    }
    registry = parity_registry()
    keys: dict[str, str] = {}
    with _as_platform(fingerprint):
        for current in compiled.order:
            kernel = registry.get(compiled.graph.node(current).kernel)
            keys[current] = graph_keys.node_key(
                compiled,
                current,
                keys,
                implementation_hash
                if current == node_id and implementation_hash is not None
                else kernel.implementation_hash(),
                source_keys,
                kernel_capabilities=kernel.capabilities,
            )
    return keys[node_id]


def installed_dependency_versions(kernel: object) -> dict[str, str]:
    """This machine's versions of ``kernel``'s declared dependencies.

    Built exactly as ``graph_parity_fixtures._pins`` builds the mapping it
    records, so the two are comparable without normalising either.
    """
    capabilities = kernel.capabilities  # type: ignore[attr-defined]
    return {
        name: importlib_metadata.version(name)
        for name in sorted(capabilities.dependencies)
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """The exact bytes ``_write_case`` writes for a pinned table."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _executed_node_key(name: str, graph: Graph, inputs_path: Path) -> str:
    with TemporaryDirectory(prefix="microcosm-parity-repin-") as store_path:
        manifest = run_graph(
            compile_graph(graph),
            sources={"fixture": inputs_path},
            store=ContentStore(Path(store_path)),
            kernels=parity_registry(),
            resume="forbid",
            decisions=(),
        )
    return manifest.nodes[graph.nodes[-1].id].key


def repin(name: str) -> dict[str, str]:
    """Re-record ``name``'s pins in place, keeping every platform it carries.

    Returns the recorded node key per platform fingerprint.
    """
    graph, inputs, direct, kernel, seed = BUILDERS[name]()
    destination = FIXTURES / name
    pins = json.loads((destination / "pins.json").read_text(encoding="utf-8"))
    if pins["kernel"] != kernel.ref:  # type: ignore[attr-defined]
        raise SystemExit(
            f"{name}: pins are for {pins['kernel']}, builder gives {kernel.ref}"  # type: ignore[attr-defined]
        )
    if (destination / "graph.json").read_text(encoding="utf-8") != graph_to_json(graph):
        raise SystemExit(f"{name}: the declaration moved; regenerate instead")
    stored_inputs = pd.read_csv(
        destination / "inputs.csv", float_precision="round_trip"
    )
    if not stored_inputs.equals(inputs.reset_index(drop=True)):
        raise SystemExit(
            f"{name}: inputs.csv no longer matches the builder; regenerate instead"
        )
    fingerprint = platform_fingerprint()
    platforms = dict(pins.get("platforms", {}))
    if (
        pins["platform"] in platforms
        and platforms[pins["platform"]]["node_key"] != pins["node_key"]
    ):
        raise SystemExit(
            f"{name}: authoring platform and top-level key are inconsistent"
        )
    platforms.setdefault(
        pins["platform"], {"node_key": pins["node_key"], "direct": "direct.csv"}
    )
    if fingerprint not in platforms:
        raise SystemExit(
            f"{name}: {fingerprint} carries no pin to re-record; "
            "use graph_parity_fixtures.py platform-pin to add one"
        )

    # This platform's own pinned bytes, which for a platform-bitwise kernel are
    # not the authoring platform's; comparing against the wrong file would refuse
    # a legitimate re-pin off the authoring platform, or accept a drifted one.
    local_direct = destination / platforms[fingerprint]["direct"]
    if local_direct.read_bytes() != _csv_bytes(direct):
        raise SystemExit(
            f"{name}: the direct call's bytes moved from "
            f"{platforms[fingerprint]['direct']}, so this is not a re-pin; "
            "regenerate the fixture and say what changed"
        )

    node_id = pins["node"]
    inputs_path = destination / "inputs.csv"
    # Step 1 of the module docstring's argument, and the only step that sees the
    # dependency channel. It has to come before any derivation: deriving first
    # would already have trusted the environment under test.
    installed = installed_dependency_versions(kernel)
    if pins.get("dependencies") != installed:
        drifted = sorted(
            set(pins.get("dependencies", {})) | set(installed),
            key=lambda dependency: dependency,
        )
        detail = ", ".join(
            f"{dependency}: pinned "
            f"{pins.get('dependencies', {}).get(dependency, '(absent)')} vs "
            f"installed {installed.get(dependency, '(absent)')}"
            for dependency in drifted
            if pins.get("dependencies", {}).get(dependency) != installed.get(dependency)
        )
        raise SystemExit(
            f"{name}: this environment is not the one the pins were taken "
            f"under ({detail}). The implementation hash folds those versions "
            "in, so a foreign platform's new key cannot be derived here; "
            "re-pin from an environment matching uv.lock."
        )

    # Step 2: every pin must be internally consistent with the hash beside it.
    for platform, entry in sorted(platforms.items()):
        reproduced = derived_node_key(
            graph, node_id, inputs_path, platform, pins["implementation_hash"]
        )
        if reproduced != entry["node_key"]:
            local = " (this machine)" if platform == fingerprint else ""
            raise SystemExit(
                f"{name}: cannot reproduce {platform}'s existing pin "
                f"{entry['node_key']} from the recorded implementation hash "
                f"(got {reproduced}){local}. The pin does not belong to the "
                "hash recorded beside it, so re-deriving it here would launder "
                "a stale or hand-edited key. Causes, in the order worth "
                "checking: tools/graph_parity_fixtures.py's own bytes moved "
                "(that re-keys every parity node, and this machine fails "
                "first); the entry was edited by hand; or it was pinned on "
                "that platform under a different implementation. Regenerate "
                "the fixture, or re-pin that platform there."
            )

    produced = _executed_node_key(name, graph, inputs_path)
    if derived_node_key(graph, node_id, inputs_path, fingerprint) != produced:
        raise SystemExit(
            f"{name}: the derived key disagrees with the executed one on "
            f"{fingerprint}; a node key now depends on more than the platform "
            "fingerprint string, so foreign platforms must re-pin themselves"
        )

    recorded = {
        platform: {
            "node_key": (
                produced
                if platform == fingerprint
                else derived_node_key(graph, node_id, inputs_path, platform)
            ),
            "direct": entry["direct"],
        }
        for platform, entry in platforms.items()
    }
    # The authoring platform is a fact about the fixture, not about the machine
    # re-pinning it, so it survives ``_pins``; only its node key is re-recorded.
    authoring = pins["platform"]
    pins.update(_pins(node_id, recorded[authoring]["node_key"], kernel, seed))
    pins["platform"] = authoring
    pins["platforms"] = recorded
    _write_pins(destination, pins)
    return {platform: entry["node_key"] for platform, entry in recorded.items()}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in BUILDERS:
        raise SystemExit(
            f"usage: graph_parity_repin.py <{' | '.join(sorted(BUILDERS))}>"
        )
    os.environ["POPULACE_FIT_N_JOBS"] = "1"
    os.environ["POPULACE_FIT_PREDICT_WORKERS"] = "1"
    for platform, key in sorted(repin(args[0]).items()):
        print(f"{args[0]} {platform} {key}")
    return 0


if __name__ == "__main__":
    # source_hash includes module names, and the generator this imports defines
    # two parity kernels; delegate to the canonical import so a pin never
    # depends on how this file was invoked.
    from tools.graph_parity_repin import main as canonical_main

    raise SystemExit(canonical_main())
