"""Charter group F: gates and release.

Finding 1 of the architecture review, and the only one rated critical: change
``release_manifest.json.schema_version`` from ``"1-evidence"`` to ``1``, leave
the ``-evidence-`` release id, ``"tier": "evidence"``, a non-empty
``known_failures``, and a failed calibration gate all in place, and
``validate_release_dir`` accepts it. Tier is a string a validator compares
against one expected value, so one field decides it.

Here tier is not a field. It is derived from content-addressed ancestry, and
the manifest is keyed by its own content, so the one-field flip cannot survive
a load. A gate is an ordinary node whose verdict is an ordinary artifact, its
outcomes are a closed set of five, and a publication decision is an input the
release either has or does not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from microcosm.graph import ContentStore, Owned, RunManifest, Slice, StructuralDelta

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: A band no toy tax mean can sit inside, so the gate fails on the evidence.
IMPOSSIBLE_BAND = {"gate_low": 1e11, "gate_high": 1e12}


def _verdict(run, node_id: str = "gate_tax") -> str:
    key = run.manifest.nodes[node_id].artifacts[("release", "gate_verdict")]
    return str(run.store.load_column(key).iloc[0])


def _tier(run) -> str:
    key = run.manifest.nodes["release"].artifacts[("release", "tier")]
    return str(run.store.load_column(key).iloc[0])


@pytest.mark.xfail(strict=True, reason="charter F1: pending")
def test_f1_a_gate_is_a_node(tmp_path: Path) -> None:
    """A verdict is an artifact keyed like any other: it hits, and it moves."""
    first = toy.run_toy(toy.full_graph(), tmp_path / "run")
    assert _verdict(first) == "pass"
    assert first.manifest.nodes["gate_tax"].receipt["outcome"] == "pass"

    registry = toy.toy_registry()
    again = toy.run_toy(
        toy.full_graph(),
        tmp_path / "run",
        sources=first.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert again.keys()["gate_tax"] == first.keys()["gate_tax"]
    assert again.misses() == set()
    assert toy.total_calls(registry) == 0

    moved = toy.run_toy(
        toy.full_graph(**IMPOSSIBLE_BAND),
        tmp_path / "run",
        sources=first.sources,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert moved.keys()["gate_tax"] != first.keys()["gate_tax"]
    assert moved.misses() == {"gate_tax", "release"}
    assert _verdict(moved) == "fail"


@pytest.mark.xfail(strict=True, reason="charter F2: pending")
def test_f2_tier_is_derived(tmp_path: Path) -> None:
    """A failed gate in the ancestry makes a release evidence-tier by
    construction, and the certified loader will not load it."""
    red = toy.run_toy(toy.full_graph(**IMPOSSIBLE_BAND), tmp_path / "red")
    assert _verdict(red) == "fail"
    assert _tier(red) == "evidence"
    assert red.manifest.nodes["release"].receipt["tier"] == "evidence"

    path = tmp_path / "red" / "manifest.json"
    red.manifest.save(path)
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="evidence"):
        RunManifest.load_certified(path, store=red.store)

    green = toy.run_toy(toy.full_graph(), tmp_path / "green")
    green_path = tmp_path / "green" / "manifest.json"
    green.manifest.save(green_path)
    assert _tier(green) == "certified"
    assert RunManifest.load_certified(green_path, store=green.store).key == (
        green.manifest.key
    )


@pytest.mark.xfail(strict=True, reason="charter F3: pending")
def test_f3_the_one_field_flip_is_impossible(tmp_path: Path) -> None:
    """Editing ``tier`` in a serialized manifest is detected on load.

    The review's reproduction, made a regression test: both ``tier`` and
    ``schema_version`` are derived from content-addressed ancestry, and the
    manifest is keyed by its content, so an edit that is not also a
    recomputation of the whole ancestry cannot be loaded.
    """
    red = toy.run_toy(toy.full_graph(**IMPOSSIBLE_BAND), tmp_path / "red")
    path = tmp_path / "red" / "manifest.json"
    red.manifest.save(path)

    document = json.loads(path.read_text())
    assert document["tier"] == "evidence"
    document["tier"] = "certified"
    path.write_text(json.dumps(document))

    from microcosm.graph import StoreCorruptError

    with pytest.raises(StoreCorruptError, match=red.manifest.key):
        RunManifest.load(path, store=red.store)

    document["schema_version"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(StoreCorruptError):
        RunManifest.load(path, store=red.store)


@pytest.mark.xfail(strict=True, reason="charter F4: pending")
def test_f4_five_outcomes_and_no_accidental_pass(tmp_path: Path) -> None:
    """The outcome set is closed, each member is reachable, and an exception
    inside a gate is a failure carrying the exception as its evidence."""
    from microcosm.graph import GATE_OUTCOMES

    assert set(GATE_OUTCOMES) == set(toy.GATE_OUTCOMES)

    passing = toy.run_toy(toy.full_graph(), tmp_path / "pass")
    assert passing.manifest.nodes["gate_tax"].receipt["outcome"] == "pass"

    failing = toy.run_toy(toy.full_graph(**IMPOSSIBLE_BAND), tmp_path / "fail")
    assert failing.manifest.nodes["gate_tax"].receipt["outcome"] == "fail"

    absent = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.absent_node("no_evidence", "unmeasured"),
            toy.gate_node(population="survey", column="unmeasured"),
            toy.release_node(population="survey"),
        )
    )
    run = toy.run_toy(absent, tmp_path / "absent")
    assert run.manifest.nodes["gate_tax"].receipt["outcome"] == "evidence_absent"

    skipped = toy.run_toy(
        toy.replace_node(toy.full_graph(), toy.gate_node(applicable=False)),
        tmp_path / "skipped",
    )
    assert skipped.manifest.nodes["gate_tax"].receipt["outcome"] == "not_applicable"

    unreached = toy.run_toy(
        toy.full_graph(**IMPOSSIBLE_BAND, requires_decisions=("publish",)),
        tmp_path / "unreached",
    )
    assert unreached.manifest.nodes["release"].receipt["outcome"] == "unreached"

    exploding = toy.run_toy(
        toy.replace_node(toy.full_graph(), toy.gate_node(kernel="bad.raise@1")),
        tmp_path / "exploding",
    )
    receipt = exploding.manifest.nodes["gate_tax"].receipt
    assert receipt["outcome"] == "fail"
    assert "exploded on purpose" in str(receipt["evidence"])


@pytest.mark.xfail(strict=True, reason="charter F5: pending")
def test_f5_human_decisions_are_inputs(tmp_path: Path) -> None:
    """A release without its required signed decision is ``unreached``.

    With the decision supplied it certifies; without it, it is never
    certified and never merely defaults to a tier.
    """
    graph = toy.full_graph(requires_decisions=("publish",))
    missing = toy.run_toy(graph, tmp_path / "missing")
    assert missing.manifest.nodes["release"].receipt["outcome"] == "unreached"
    # The owned tier derives from gate ancestry alone (charter A7: a decision
    # changes no key, so it cannot change an artifact). Publishability is a
    # manifest-level fact, and the certified loader is where it bites.
    missing_path = tmp_path / "missing" / "manifest.json"
    missing.manifest.save(missing_path)
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="unreached"):
        RunManifest.load_certified(missing_path, store=missing.store)

    signed = toy.run_toy(graph, tmp_path / "signed", decisions=(toy.PUBLISH_DECISION,))
    assert signed.manifest.nodes["release"].receipt["outcome"] == "pass"
    assert _tier(signed) == "certified"
    assert [dict(record) for record in signed.manifest.decisions] == [
        toy.PUBLISH_DECISION
    ]

    wrong_owner = dict(toy.PUBLISH_DECISION, name="something_else")
    other = toy.run_toy(graph, tmp_path / "other", decisions=(wrong_owner,))
    assert other.manifest.nodes["release"].receipt["outcome"] == "unreached"


def test_the_toy_release_surface_is_shaped_the_way_the_charter_needs() -> None:
    """Green from the first commit: the gate and release nodes really are nodes.

    F1–F5 above only mean something if the toy country expresses a gate as an
    ordinary node owning an ordinary cell, and a release that reads the gate's
    verdict as an ordinary input. This guards that shape.
    """
    gate = toy.gate_node()
    release = toy.release_node(requires_decisions=("publish",))
    assert gate.outputs == (Owned("release", "gate_verdict", "string"),)
    assert release.inputs == (Slice("release", ("gate_verdict",)),)
    assert release.outputs == (Owned("release", "tier", "string"),)
    assert gate.params["gate"] is True
    assert release.params["requires_decisions"] == ("publish",)
    assert release.structural is StructuralDelta.NONE
