"""Invented complete Frame metadata persistence and stale codec refusal."""

from __future__ import annotations

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
