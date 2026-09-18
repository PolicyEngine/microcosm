"""Invented complete Frame metadata persistence and stale codec refusal."""

from __future__ import annotations

import hashlib
import json
import math
import struct

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.store as store_module
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.store import ContentStore, StoreCorrupt, StoreUnavailable


def _frame(metadata):
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1, 2], "person_household_id": [1, 1]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([100.0]), WeightKind.DESIGN)},
        metadata=metadata,
    )


def test_roundtrip_preserves_complete_nested_source_metadata_and_types(tmp_path):
    original = _frame(
        {
            "us_spine_assembly_manifest": {
                "source_arm": "invented",
                "sources": ({"sha256": "a" * 64, "rows": 2, "design_total": 100.0},),
                "flags": frozenset({"native", "unknown"}),
                "nullable": None,
                "flag": True,
            },
            "signed_zero": -0.0,
            "nan": struct.unpack(">d", bytes.fromhex("7ff8000000000011"))[0],
            "infinity": float("inf"),
        }
    )
    store = ContentStore(tmp_path / "invented-store")
    store.put_frame("a" * 64, original)
    restored = store.load_frame("a" * 64)
    assert store_module._encode_frame_metadata(
        restored.metadata
    ) == store_module._encode_frame_metadata(original.metadata)
    assert isinstance(restored.metadata["us_spine_assembly_manifest"]["sources"], tuple)
    assert isinstance(
        restored.metadata["us_spine_assembly_manifest"]["flags"], frozenset
    )
    assert math.copysign(1.0, restored.metadata["signed_zero"]) == -1.0
    assert struct.pack(">d", restored.metadata["nan"]).hex() == "7ff8000000000011"
    assert (
        restored.weights_for("household").values.tobytes()
        == original.weights_for("household").values.tobytes()
    )
    with pytest.raises(TypeError):
        restored.metadata["us_spine_assembly_manifest"]["source_arm"] = "changed"


def test_same_key_with_changed_metadata_refuses_without_replacing_original(tmp_path):
    store = ContentStore(tmp_path / "invented-store")
    original = _frame({"source_sha256": "a" * 64})
    store.put_frame("b" * 64, original)
    store.put_frame("b" * 64, original)
    with pytest.raises(StoreCorrupt, match="different metadata"):
        store.put_frame("b" * 64, _frame({"source_sha256": "b" * 64}))
    assert store.load_frame("b" * 64).metadata == original.metadata


@pytest.mark.parametrize("wrap_in_tuple", [False, True])
def test_roundtrip_preserves_frozen_sets_of_inherited_mappings(tmp_path, wrap_in_tuple):
    parent = _frame({"sources": ({"id": "invented", "nested": {"rows": 2}},)})
    source = parent.metadata["sources"][0]
    member = (source,) if wrap_in_tuple else source
    original = _frame({"source_set": frozenset({member})})
    store = ContentStore(tmp_path / "invented-store")
    key = "9" * 64
    store.put_frame(key, original)
    restored = store.load_frame(key)
    assert isinstance(restored.metadata["source_set"], frozenset)
    assert store_module._encode_frame_metadata(
        restored.metadata
    ) == store_module._encode_frame_metadata(original.metadata)
    restored_member = next(iter(restored.metadata["source_set"]))
    restored_source = restored_member[0] if wrap_in_tuple else restored_member
    assert hash(restored_member) == hash(member)
    with pytest.raises(TypeError):
        restored_source["nested"]["rows"] = 3
    # A normal same-key cache write must keep loading the admitted metadata.
    store.put_frame(key, restored)
    assert store.load_frame(key).metadata == original.metadata


def test_v1_frame_is_unavailable_instead_of_silently_losing_metadata(
    tmp_path, monkeypatch
):
    store = ContentStore(tmp_path / "invented-store")
    with monkeypatch.context() as old:
        old.setattr(store_module, "_FRAME_FORMAT", "microcosm-graph-frame-v1")
        store.put_frame("c" * 64, _frame({"source": "invented"}))
    with pytest.raises(StoreUnavailable, match="codec"):
        store.load_frame("c" * 64)
    with pytest.raises(StoreUnavailable, match="predates complete metadata"):
        store.put_frame("c" * 64, _frame({"source": "invented"}))


@pytest.mark.parametrize(
    "value",
    [
        ["mapping", [["duplicate", ["scalar", 1]], ["duplicate", ["scalar", 2]]]],
        ["float64", "bad"],
        ["scalar", 1.2],
        ["mapping", [["", ["scalar", None]]]],
        ["unknown", []],
    ],
)
def test_malformed_metadata_refuses(value):
    with pytest.raises(StoreCorrupt, match="metadata"):
        store_module._decode_frame_metadata(value)


def _restated_manifest(store, key, *, literal=None):
    """Rewrite one stored frame manifest, keeping the store's own gate honest.

    The manifest is re-registered with its true size and SHA-256 so
    :func:`_verified_meta` still runs and still passes; the refusal under test
    therefore comes from the decode boundary, not from a skipped checksum.
    Passing ``literal=None`` restates identical content, which is the control
    proving valid bytes are unaffected.
    """
    object_path = store.object_path(key)
    manifest_path = object_path / "frame.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if literal is not None:
        manifest["metadata"] = [
            "mapping",
            [["us_spine_assembly_manifest", ["float64", "@NON_FINITE@"]]],
        ]
    text = json.dumps(manifest, separators=(",", ":"), sort_keys=True)
    manifest_path.write_text(
        text.replace('"@NON_FINITE@"', literal or ""), encoding="utf-8"
    )
    meta_path = object_path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    record = meta["payloads"]["frame.json"]
    record["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    record["size"] = manifest_path.stat().st_size
    meta_path.write_text(
        json.dumps(meta, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    return manifest_path


@pytest.mark.parametrize(
    "literal", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999", "1E9999"]
)
def test_non_finite_stored_manifest_is_store_corrupt_not_type_error(tmp_path, literal):
    """A checksum-consistent but non-finite manifest keeps the store taxonomy."""
    store = ContentStore(tmp_path / "invented-store")
    store.put_frame("d" * 64, _frame({"source": "invented"}))
    manifest_path = _restated_manifest(store, "d" * 64, literal=literal)
    assert literal in manifest_path.read_text(encoding="utf-8")
    with pytest.raises(StoreCorrupt, match="frame manifest") as caught:
        store.load_frame("d" * 64)
    # The escape this pins is a bare TypeError from the canonical re-encode.
    assert type(caught.value) is StoreCorrupt


def test_restating_the_same_manifest_still_loads_the_identical_frame(tmp_path):
    """Control: the stricter decoder does not reject any valid stored bytes."""
    store = ContentStore(tmp_path / "invented-store")
    original = _frame({"source": "invented", "share": 0.25, "count": 3})
    store.put_frame("e" * 64, original)
    _restated_manifest(store, "e" * 64)
    restored = store.load_frame("e" * 64)
    assert store_module._encode_frame_metadata(
        restored.metadata
    ) == store_module._encode_frame_metadata(original.metadata)


@pytest.mark.parametrize("literal", ["NaN", "-Infinity", "1e999"])
def test_non_finite_object_metadata_is_store_corrupt(tmp_path, literal):
    """The same boundary covers meta.json, which no payload checksum guards."""
    store = ContentStore(tmp_path / "invented-store")
    store.put_frame("f" * 64, _frame({"source": "invented"}))
    meta_path = store.object_path("f" * 64) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["invented_non_finite"] = "@NON_FINITE@"
    meta_path.write_text(
        json.dumps(meta, separators=(",", ":"), sort_keys=True).replace(
            '"@NON_FINITE@"', literal
        ),
        encoding="utf-8",
    )
    with pytest.raises(StoreCorrupt, match="canonical JSON"):
        store.load_frame("f" * 64)


def test_non_finite_json_never_reaches_the_canonical_encoder(tmp_path, monkeypatch):
    """The refusal happens at decode, before any canonical re-encode runs."""
    store = ContentStore(tmp_path / "invented-store")
    store.put_frame("0" * 64, _frame({"source": "invented"}))
    _restated_manifest(store, "0" * 64, literal="NaN")
    seen = []
    canonical = store_module._canonical_json

    def recorded(value):
        seen.append(value)
        return canonical(value)

    monkeypatch.setattr(store_module, "_canonical_json", recorded)
    with pytest.raises(StoreCorrupt):
        store.load_frame("0" * 64)
    assert seen == [], "malformed JSON reached the canonical encoder"
