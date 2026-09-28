"""Tests split from packages/microcosm-graph/tests/test_acceptance_h_parity.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_graph.acceptance_h_parity import *


def test_h1_kernel_parity(tmp_path: Path) -> None:
    """A wrapped legacy kernel is byte-identical to the direct call.

    Expects ``packages/microcosm-graph/tests/fixtures/parity/kernels/<name>/``
    for each of ``fit.qrf``, ``draw.qrf``, ``calibrate``, and ``simulate``,
    each holding ``graph.json``, ``inputs.csv``, ``direct.csv``, and
    ``pins.json``. The wrappers live in ``microcosm-fit``,
    ``microcosm-calibrate``, and the ``RulesEngine`` adapter; the lane that
    writes them produces these fixtures at the same pinned seed.
    """
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph
    from tools.graph_parity_fixtures import parity_registry

    _require(KERNEL_PARITY, "the kernel-wrapper lane (#378 step 3)")
    registry = parity_registry()
    for name in WRAPPED_KERNELS:
        case = _require(KERNEL_PARITY / name, "the kernel-wrapper lane")
        pins = json.loads((case / "pins.json").read_text())
        assert set(pins) >= {
            "seed",
            "kernel",
            "implementation_hash",
            "dependencies",
            "node_key",
            "numeric",
            "platform",
        }
        kernel = registry.get(pins["kernel"])
        assert kernel.implementation_hash() == pins["implementation_hash"]
        assert set(pins["dependencies"]) == set(kernel.capabilities.dependencies)

        store = ContentStore(tmp_path / name)
        manifest = run_graph(
            compile_graph(graph_from_json((case / "graph.json").read_text())),
            sources={"fixture": case / "inputs.csv"},
            store=store,
            kernels=registry,
            resume="forbid",
            decisions=(),
        )
        node = manifest.nodes[pins["node"]]
        assert node.receipt["capabilities"]["numeric"] == NUMERIC_CLAIMS[name]
        # A structural node re-keys every carried column as an artifact of its
        # own; the direct call produced only what direct.csv holds, so those
        # are the cells compared. A weight transition is compared through the
        # weight artifact under the ``<entity>.weights`` column.
        # A platform-bitwise kernel's bytes are asserted on every platform
        # that carries a pin: the authoring platform and each CI platform
        # (amendment 16). Elsewhere the property that holds is identity
        # partitioning: the node key carries the platform, so the local key
        # differs from every pinned platform's key and a shared store can
        # never serve another platform's artifact.
        platforms = dict(pins.get("platforms", {}))
        platforms.setdefault(
            pins["platform"], {"node_key": pins["node_key"], "direct": "direct.csv"}
        )
        local = platforms.get(platform_fingerprint())
        off_platform = pins["numeric"] == "platform_bitwise" and local is None
        if off_platform:
            assert node.key not in {entry["node_key"] for entry in platforms.values()}
            direct = _direct_table(case)
        else:
            pinned = (
                local
                if pins["numeric"] == "platform_bitwise"
                else platforms[pins["platform"]]
            )
            assert node.key == pinned["node_key"]
            direct = _direct_table(case, pinned["direct"])
        exposed = 0
        for cell, key in node.artifacts.items():
            label = f"{cell[0]}.{cell[1]}"
            if label in direct.columns:
                exposed += 1
                if not off_platform:
                    _assert_same_bytes(store.load_column(key), direct[label])
        if node.weight_key is not None:
            exposed += 1
            entity = (
                graph_from_json((case / "graph.json").read_text())
                .node(pins["node"])
                .weights.entity
            )
            if not off_platform:
                _assert_same_bytes(
                    store.load_column(node.weight_key), direct[f"{entity}.weights"]
                )
        assert exposed, f"{name}: the fixture exposed nothing to compare"


def test_the_parity_fixtures_are_declared_but_not_faked() -> None:
    """Green from the first commit: no parity fixture is invented here.

    If a directory ever appears under ``fixtures/parity/`` in a commit that
    also touches this file, that is the acceptance lane manufacturing its own
    evidence. The suite says so out loud instead.
    """
    if not PARITY.exists():
        return
    for case in sorted(PARITY.iterdir()):
        assert case.is_dir()
        assert (case / "PRODUCED_BY.txt").exists(), (
            f"{case} carries no note saying which lane produced it"
        )
