"""A run's verification-epoch record: carried, and outside everything computed.

``RunManifest.verification_epoch`` and ``run_graph(_verification_epoch=...)``
are how a country runtime that scopes source verification around a whole run
tells a reader how many validations that run actually performed. Both shipped
with no test in this shard: nothing asserted that attaching a record leaves the
node keys, the manifest's JSON and its content-addressed body exactly where
they were; nothing asserted that the manifest keeps the caller's own mapping
rather than a copy -- the property the whole design leans on, because the
record is finalised when the caller's scope closes, after ``run_graph`` has
returned; and nothing asserted that a malformed record is refused rather than
carried.

``source_identities`` next door has
``test_the_manifest_records_the_run_end_identities_without_moving_anything``;
this file is the same contract for the field beside it.

Synthetic toy graph only: no data file, no country package, no engine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import (
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.executor import run_graph
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
)
from microcosm.graph.manifest import NodeReceipt, RunManifest
from microcosm.graph.store import ContentStore

SOURCE = SourceRef("survey", "csv-tables")


class _Kernel:
    def __init__(self, ref, capabilities, run):
        self._ref = ref
        self._capabilities = capabilities
        self._run = run

    @property
    def ref(self):
        return self._ref

    @property
    def capabilities(self):
        return self._capabilities

    def implementation_hash(self):
        return f"impl-{self._ref}"

    def run(self, context):
        return self._run(context)


def _source_path(root: Path) -> Path:
    root.mkdir()
    (root / "value.txt").write_text("0", encoding="utf-8")
    return root


def _source_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "age": np.asarray([10, 20, 30], dtype=np.int64),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "size": np.asarray([2, 1], dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b"], name="stratum"),
    )


CREATE = Node(
    "survey",
    "source@1",
    structural=StructuralDelta.CREATE,
    sources=("survey",),
    outputs=(Owned("person", "age", "int64"), Owned("household", "size", "int64")),
)
FOLLOWER = Node(
    "follower",
    "follower@1",
    inputs=(Slice("person", ("age",)),),
    outputs=(Owned("person", "doubled", "float64"),),
    population="survey",
)


def _registry() -> KernelRegistry:
    def source(context: KernelContext) -> KernelResult:
        frame = _source_frame()
        return KernelResult(frame=frame, receipt={"rows": frame.n("person")})

    def follower(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "doubled"): pd.Series(
                    table["age"].to_numpy(dtype=np.float64) * 2.0,
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            },
            receipt={"rows": len(table)},
        )

    registry = KernelRegistry()
    registry.register(
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            source,
        )
    )
    registry.register(
        _Kernel("follower@1", Capabilities(Determinism.DETERMINISTIC), follower)
    )
    return registry


def _run(root: Path, record: Mapping[str, object] | None = None) -> RunManifest:
    """One cold run under ``root``, with its own source tree and its own store."""

    root.mkdir()
    return run_graph(
        compile_graph(Graph("toy", (SOURCE,), (CREATE, FOLLOWER))),
        sources={"survey": _source_path(root / "source")},
        store=ContentStore(root / "store"),
        kernels=_registry(),
        **({} if record is None else {"_verification_epoch": record}),
    )


def _keys(manifest: RunManifest) -> dict[str, str]:
    return {node_id: manifest.nodes[node_id].key for node_id in sorted(manifest.nodes)}


def _json_without_run_local_fields(manifest: RunManifest) -> dict[str, object]:
    """This manifest's JSON document, less exactly the fields a run may move.

    ``to_json`` is not identical between two runs of one computation and is not
    meant to be: it carries three run-level manifest fields -- ``started_at``,
    ``finished_at`` and ``host`` -- and, inside every node receipt, the two the
    receipt itself names as run-level, ``NodeReceipt.RUN_LEVEL_FIELDS``. None of
    the five reaches the manifest key, by two different routes: ``key`` hashes
    ``content_addressed``, which is ``nodes`` and ``tier`` alone, so the two
    timestamps and the host are outside it entirely, and each receipt enters it
    through ``NodeReceipt._content_payload``, which drops ``RUN_LEVEL_FIELDS``.
    Removing exactly those, by their own names and by that class attribute
    rather than by a list written here, is what lets the rest of the document
    be compared between two runs; a ``del`` of a field that stopped being
    serialized would raise rather than quietly widen what this hides. Nothing
    is removed from ``decisions``, ``content_addressed`` or any other field.
    """

    payload = json.loads(manifest.to_json())
    for name in ("started_at", "finished_at", "host"):
        del payload[name]
    for receipt in payload["nodes"].values():
        for name in sorted(NodeReceipt.RUN_LEVEL_FIELDS):
            del receipt[name]
    return payload


# --------------------------------------------------------------------------
# What the record must not move.
# --------------------------------------------------------------------------


def test_a_record_is_outside_the_key_the_json_and_the_content_address(
    tmp_path: Path,
) -> None:
    """The same manifest without the field is byte-identical everywhere it counts."""

    manifest = _run(
        tmp_path / "run", {"protocol": "toy/verification-epoch/1", "hits": 4}
    )
    assert dict(manifest.verification_epoch) == {
        "protocol": "toy/verification-epoch/1",
        "hits": 4,
    }

    bare = type(manifest)(
        country=manifest.country,
        nodes=manifest.nodes,
        decisions=manifest.decisions,
        started_at=manifest.started_at,
        finished_at=manifest.finished_at,
        host=manifest.host,
    )
    assert bare.key == manifest.key
    assert bare.to_json() == manifest.to_json()
    assert "verification_epoch" not in manifest.to_json()
    assert dict(manifest.content_addressed) == dict(bare.content_addressed)


def test_two_runs_agree_on_every_node_key_with_and_without_a_record(
    tmp_path: Path,
) -> None:
    """Executed separately, into separate stores: the record buys no reuse and no key."""

    with_record = _run(
        tmp_path / "with", {"protocol": "toy/verification-epoch/1", "hits": 9}
    )
    without = _run(tmp_path / "without")
    assert _keys(with_record) == _keys(without)
    assert with_record.key == without.key
    assert dict(with_record.content_addressed) == dict(without.content_addressed)
    # The serialized document too, and not only the key and the body it
    # hashes: every field of it except the five a run is allowed to move.
    # ``verification_epoch`` is not among them -- it is not serialized at all,
    # which the bare-manifest test above pins directly.
    assert _json_without_run_local_fields(
        with_record
    ) == _json_without_run_local_fields(without)
    # Both ran cold, so the equality above is between two computations rather
    # than between a run and a cache hit replaying the first one's receipts.
    assert [node.hit for node in with_record.nodes.values()] == [False, False]
    assert [node.hit for node in without.nodes.values()] == [False, False]


def test_no_record_is_an_empty_mapping_not_a_missing_attribute(tmp_path: Path) -> None:
    manifest = _run(tmp_path / "run")
    assert dict(manifest.verification_epoch) == {}
    assert "verification_epoch" not in manifest.to_json()


# --------------------------------------------------------------------------
# The live view, which is the whole reason the field exists.
# --------------------------------------------------------------------------


def test_the_manifest_holds_the_caller_s_own_record_not_a_copy(tmp_path: Path) -> None:
    """A caller's scope closes after ``run_graph`` returns, and the counts follow.

    The runner that opens a verification epoch hands ``run_graph`` the record
    that epoch yields, and that epoch's final counts -- how many capsules were
    re-validated as it closed -- are written into the record only when the
    scope leaves, which is after this call has returned its manifest. A copy
    taken at construction would freeze the record mid-run and report a final
    validation count of zero for every run.
    """

    record = {"protocol": "toy/verification-epoch/1", "hits": 0, "misses": 0}
    manifest = _run(tmp_path / "run", record)
    assert manifest.verification_epoch["hits"] == 0

    record["hits"] = 18
    record["final_validations"] = 2
    assert manifest.verification_epoch["hits"] == 18
    assert manifest.verification_epoch["final_validations"] == 2
    assert dict(manifest.verification_epoch) == record

    # A view, so a reader of the manifest cannot write the caller's record.
    assert isinstance(manifest.verification_epoch, MappingProxyType)
    with pytest.raises(TypeError):
        manifest.verification_epoch["hits"] = 0  # type: ignore[index]


def test_a_mapping_proxy_is_kept_as_it_is_given(tmp_path: Path) -> None:
    """A caller that already wrapped its record is not wrapped twice."""

    record = {"protocol": "toy/verification-epoch/1", "capsules": 1}
    view = MappingProxyType(record)
    manifest = _run(tmp_path / "run", view)
    assert manifest.verification_epoch is view
    record["capsules"] = 3
    assert manifest.verification_epoch["capsules"] == 3


# --------------------------------------------------------------------------
# A malformed record is refused, not carried.
# --------------------------------------------------------------------------


def test_a_record_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(TypeError, match="verification_epoch must be a mapping"):
        RunManifest(country="toy", nodes={}, verification_epoch=("hits", 1))


def test_a_record_key_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="verification_epoch keys must be strings"):
        RunManifest(country="toy", nodes={}, verification_epoch={1: 2})


def test_a_record_value_that_is_neither_an_integer_nor_a_string_is_refused() -> None:
    with pytest.raises(
        TypeError, match="verification_epoch values must be integers or strings"
    ):
        RunManifest(country="toy", nodes={}, verification_epoch={"wall_time": 1.5})


def test_counts_and_labels_are_what_a_record_may_hold() -> None:
    """Integers and strings, which is what an epoch record is: no key, digest or path."""

    manifest = RunManifest(
        country="toy",
        nodes={},
        verification_epoch={
            "protocol": "toy/verification-epoch/1",
            "hits": 18,
            "misses": 1,
            "final_validations": 2,
            "capsules": 2,
        },
    )
    assert manifest.verification_epoch["protocol"] == "toy/verification-epoch/1"
    assert manifest.verification_epoch["final_validations"] == 2
    assert "verification_epoch" not in manifest.to_json()
