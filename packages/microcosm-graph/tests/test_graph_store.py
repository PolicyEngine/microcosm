"""Focused contracts for deterministic, validated graph storage."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.store as store_module
from microcosm.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)
from microcosm.graph.store import ContentStore, StoreCorrupt, StoreMiss


def _key(character: str) -> str:
    return character * 64


def _frame() -> Frame:
    schema = EntitySchema(group_entities=("household",))
    person_index = pd.Index([101, 103, 107], name="source_row")
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "flag": pd.Series(
                [True, pd.NA, False], index=person_index, dtype="boolean"
            ),
            "label": pd.Series(
                ["alpha", "nul\0inside", pd.NA],
                index=person_index,
                dtype="string",
            ),
        },
        index=person_index,
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int32),
            "income": np.asarray([-0.0, 12.5], dtype=np.float64),
        },
        index=pd.Index([41, 43], name="household_source_row"),
    )
    return Frame(
        {"person": person, "household": household},
        schema,
        {
            "household": Weights(
                np.asarray([1.5, 2.5]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["urban", "urban", "rural"], index=person_index, dtype=object),
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=2.0,
                new_total=4.0,
                declared_factor=2.0,
                reason="fixture control",
            ),
        ),
    )


def _object_bytes(path: Path) -> dict[str, bytes]:
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def test_column_round_trip_preserves_exact_dtype_masks_and_signed_zero(
    tmp_path: Path,
) -> None:
    store = ContentStore(tmp_path / "store")
    ids = pd.Index([11, 12, 13], name="person_id")
    columns = {
        "a": (pd.Series([True, False, True], index=ids, dtype="bool"), "bool"),
        "b": (
            pd.Series([True, pd.NA, False], index=ids, dtype="boolean"),
            "boolean",
        ),
        "c": (pd.Series([1, pd.NA, -2], index=ids, dtype="Int64"), "Int64"),
        "d": (
            pd.Series([-0.0, 0.0, 1.5], index=ids, dtype="float64"),
            "float64",
        ),
        "e": (
            pd.Series(["", "é\0x", pd.NA], index=ids, dtype="string"),
            "string",
        ),
    }
    for character, (expected, token) in columns.items():
        key = _key(character)
        store.put_column(
            key,
            expected,
            declared_dtype=token,
            node_key=_key("f"),
        )
        actual = store.load_column(
            key,
            declared_dtype=token,
            entity_ids=ids,
            node_key=_key("f"),
        )
        pd.testing.assert_series_equal(actual, expected, check_exact=True)

    floats = store.load_column(_key("d")).to_numpy()
    assert np.signbit(floats[0])
    assert not np.signbit(floats[1])
    assert store.load_column(_key("a")).dtype == np.dtype(np.bool_)
    assert store.load_column(_key("b")).dtype == pd.BooleanDtype()

    string_path = store.object_path(_key("e"))
    for array_path in string_path.rglob("*.npy"):
        with array_path.open("rb") as stream:
            assert not np.load(stream, allow_pickle=False).dtype.hasobject
    assert np.load(string_path / "values.npy", allow_pickle=False).dtype == np.uint8
    assert (string_path / "mask.npy").is_file()

    metadata = json.loads((string_path / "meta.json").read_text())
    assert metadata["declared_dtype"] == "string"
    assert metadata["pandas_dtype"] == "string"
    assert metadata["length"] == 3
    assert len(metadata["entity_id_hash"]) == 64
    assert metadata["node_key"] == _key("f")


def test_frame_round_trip_is_byte_identical_across_stores(tmp_path: Path) -> None:
    frame = _frame()
    key = _key("9")
    first = ContentStore(tmp_path / "first")
    second = ContentStore(tmp_path / "second")

    first_path = first.put_frame(key, frame, node_key=_key("8"))
    second_path = second.put_frame(key, frame, node_key=_key("8"))

    assert _object_bytes(first_path) == _object_bytes(second_path)
    loaded = first.load_frame(key, node_key=_key("8"))
    assert loaded.schema == frame.schema
    assert loaded.mass_log == frame.mass_log
    assert loaded.weighted_entities == frame.weighted_entities
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            loaded.table(entity), frame.table(entity), check_exact=True
        )
    pd.testing.assert_series_equal(loaded.strata, frame.strata, check_exact=True)
    np.testing.assert_array_equal(
        loaded.weights_for("household").values,
        frame.weights_for("household").values,
    )
    assert loaded.weights_for("household").kind is WeightKind.DESIGN
    assert np.signbit(loaded.table("household")["income"].iloc[0])
    assert (first_path / "schema.json").is_file()


def test_corrupt_payload_is_fatal_not_a_miss(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("a")
    store.put_column(
        key,
        pd.Series([1.0, 2.0], dtype="float64"),
        declared_dtype="float64",
    )
    assert store.metadata(key)["node_key"] is None
    payload = store.object_path(key) / "values.npy"
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 0x01
    payload.write_bytes(damaged)

    assert store.has(key)
    with pytest.raises(StoreCorrupt, match="SHA-256"):
        store.load_column(key)


def test_missing_object_raises_store_miss(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    with pytest.raises(StoreMiss):
        store.load_column(_key("0"))


def test_interrupted_atomic_write_leaves_no_visible_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("c")

    def crash(_source: Path, _destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(store_module.os, "replace", crash)
    with pytest.raises(OSError, match="simulated interruption"):
        store.put_column(
            key,
            pd.Series([1, 2], dtype="int64"),
            declared_dtype="int64",
        )

    assert not store.has(key)
    assert list(store.tmp.iterdir()) == []


def test_write_only_collision_serializes_without_reading_incumbent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("7")
    store.put_bytes(key, b"stable")
    real_replace = store_module.os.replace
    replacements: list[tuple[Path, Path]] = []

    def track_replace(source: Path, destination: Path) -> None:
        replacements.append((source, destination))
        real_replace(source, destination)

    def reject_verification(*args: object, **kwargs: object) -> None:
        raise AssertionError("write-only collision verified the incumbent")

    with monkeypatch.context() as write_only:
        write_only.setattr(store_module.os, "replace", track_replace)
        write_only.setattr(store_module, "_verified_meta", reject_verification)
        assert store.put_bytes(
            key, b"recomputed", verify_existing=False
        ) == store.object_path(key)

    assert len(replacements) == 3
    assert list(store.tmp.iterdir()) == []
    assert store.load_bytes(key) == b"recomputed"


def test_interrupted_write_only_collision_preserves_complete_incumbent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("6")
    store.put_bytes(key, b"incumbent")

    def crash(_source: Path, _destination: Path) -> None:
        raise OSError("simulated collision interruption")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(store_module.os, "replace", crash)
        with pytest.raises(OSError, match="collision interruption"):
            store.put_bytes(key, b"recomputed", verify_existing=False)

    assert list(store.tmp.iterdir()) == []
    assert store.load_bytes(key) == b"incumbent"


def test_interrupted_write_only_swap_rolls_back_complete_incumbent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("5")
    store.put_bytes(key, b"incumbent")
    real_replace = store_module.os.replace
    calls = 0

    def crash_after_displacement(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated publication interruption")
        real_replace(source, destination)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(store_module.os, "replace", crash_after_displacement)
        with pytest.raises(OSError, match="publication interruption"):
            store.put_bytes(key, b"recomputed", verify_existing=False)

    assert calls == 4
    assert list(store.tmp.iterdir()) == []
    assert store.load_bytes(key) == b"incumbent"


def test_json_receipt_and_opaque_bytes_are_content_validated(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    store.put_json(_key("d"), {"node": "n", "ok": True}, kind="node-receipt")
    store.put_bytes(_key("e"), b"model bytes")

    assert store.load_json(_key("d"), kind="node-receipt") == {
        "node": "n",
        "ok": True,
    }
    assert store.load_bytes(_key("e")) == b"model bytes"


def test_load_bytes_returns_verified_buffer_after_same_size_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("a")
    path = store.put_bytes(key, b"original")
    original_verify = store_module._verified_meta

    def replace_after_verification(*args, **kwargs):
        metadata = original_verify(*args, **kwargs)
        (path / "payload.bin").write_bytes(b"replaced")
        return metadata

    monkeypatch.setattr(store_module, "_verified_meta", replace_after_verification)
    assert store.load_bytes(key) == b"original"
    assert (path / "payload.bin").read_bytes() == b"replaced"


def test_load_bytes_checks_hash_of_exact_returned_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("b")
    path = store.put_bytes(key, b"original") / "payload.bin"
    original_read = Path.read_bytes

    def replace_before_read(candidate):
        if candidate == path:
            candidate.write_bytes(b"replaced")
        return original_read(candidate)

    monkeypatch.setattr(Path, "read_bytes", replace_before_read)
    with pytest.raises(StoreCorrupt, match="size/SHA-256"):
        store.load_bytes(key)


def test_load_bytes_checks_roster_after_capturing_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("c")
    path = store.put_bytes(key, b"original")
    original_read = Path.read_bytes

    def add_file_after_read(candidate):
        payload = original_read(candidate)
        if candidate == path / "payload.bin":
            (path / "unlisted.bin").write_bytes(b"unexpected")
        return payload

    monkeypatch.setattr(Path, "read_bytes", add_file_after_read)
    with pytest.raises(StoreCorrupt, match="payload table differs"):
        store.load_bytes(key)


@pytest.mark.parametrize("payload", [b"", b"\x00\xffbinary\x00"])
def test_load_bytes_preserves_exact_binary_payload(
    tmp_path: Path, payload: bytes
) -> None:
    store = ContentStore(tmp_path / "store")
    store.put_bytes(_key("d"), payload)
    assert store.load_bytes(_key("d")) == payload


@pytest.mark.parametrize("damage", ["hash", "size", "missing", "roster", "symlink"])
def test_load_bytes_preserves_payload_refusals(tmp_path: Path, damage: str) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("e")
    path = store.put_bytes(key, b"original")
    payload = path / "payload.bin"
    if damage == "hash":
        payload.write_bytes(b"replaced")
    elif damage == "size":
        payload.write_bytes(b"short")
    elif damage == "missing":
        payload.unlink()
    elif damage == "roster":
        (path / "unlisted.bin").write_bytes(b"unexpected")
    else:
        target = tmp_path / "other.bin"
        target.write_bytes(b"original")
        payload.unlink()
        payload.symlink_to(target)
    with pytest.raises(StoreCorrupt):
        store.load_bytes(key)


@pytest.mark.parametrize("field", ["kind", "key"])
def test_load_bytes_preserves_metadata_refusals(tmp_path: Path, field: str) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("f")
    path = store.put_bytes(key, b"original")
    metadata = json.loads((path / "meta.json").read_text())
    metadata[field] = "json" if field == "kind" else _key("0")
    (path / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(StoreCorrupt):
        store.load_bytes(key)


def test_load_bytes_missing_object_is_still_a_miss(tmp_path: Path) -> None:
    with pytest.raises(StoreMiss):
        ContentStore(tmp_path / "store").load_bytes(_key("1"))


def test_load_bytes_missing_undeclared_payload_is_still_corrupt(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("2")
    path = store.put_bytes(key, b"original")
    metadata = json.loads((path / "meta.json").read_text())
    metadata["payloads"] = {}
    (path / "meta.json").write_text(json.dumps(metadata))
    (path / "payload.bin").unlink()
    with pytest.raises(StoreCorrupt, match="disappeared"):
        store.load_bytes(key)


def test_load_bytes_keeps_other_payloads_and_metadata_calls_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("3")
    path = store.put_bytes(key, b"original")
    extra = b"other declared bytes"
    (path / "extra.bin").write_bytes(extra)
    metadata = json.loads((path / "meta.json").read_text())
    metadata["payloads"]["extra.bin"] = {
        "size": len(extra),
        "sha256": sha256(extra).hexdigest(),
    }
    (path / "meta.json").write_text(json.dumps(metadata))
    original_read = Path.read_bytes
    captured = []

    def record_read(candidate):
        assert candidate != path / "extra.bin", "Extra payload must remain streaming"
        if candidate == path / "payload.bin":
            captured.append(candidate)
        return original_read(candidate)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    assert store.metadata(key) == metadata
    assert captured == []
    assert store.load_bytes(key) == b"original"
    assert captured == [path / "payload.bin"]


# --------------------------------------------------------------------------
# The write ledger, and taking one caller's own writes back out.
# --------------------------------------------------------------------------


def test_the_ledger_holds_what_was_published_not_what_was_already_there(
    tmp_path: Path,
) -> None:
    store = ContentStore(tmp_path / "store")
    existing = _key("1")
    store.put_bytes(existing, b"from an earlier run")

    with store.recording_writes() as written:
        store.put_bytes(existing, b"from an earlier run")  # satisfied, not published
        store.put_bytes(_key("2"), b"this run's")
        store.put_json(_key("3"), {"this": "run"})

    assert written == {_key("2"), _key("3")}


def test_a_write_only_replacement_is_this_caller_s_to_take_back(
    tmp_path: Path,
) -> None:
    """``verify_existing=False`` puts this caller's object where the old one was."""

    store = ContentStore(tmp_path / "store")
    key = _key("4")
    store.put_bytes(key, b"incumbent")

    with store.recording_writes() as written:
        store.put_bytes(key, b"replacement", verify_existing=False)

    assert written == {key}


def test_ledgers_nest_and_each_closes_without_disturbing_the_other(
    tmp_path: Path,
) -> None:
    store = ContentStore(tmp_path / "store")

    with store.recording_writes() as outer:
        store.put_bytes(_key("5"), b"outer")
        with store.recording_writes() as inner:
            store.put_bytes(_key("6"), b"inner")
        assert inner == {_key("6")}
        store.put_bytes(_key("7"), b"outer again")

    assert outer == {_key("5"), _key("6"), _key("7")}
    assert store._write_ledgers == []


def test_two_empty_ledgers_close_without_stranding_each_other(tmp_path: Path) -> None:
    """Equal sets are not the same ledger; closing takes the one that opened."""

    store = ContentStore(tmp_path / "store")
    with store.recording_writes():
        with store.recording_writes():
            pass
        store.put_bytes(_key("8"), b"after the inner close")
    assert store._write_ledgers == []


def test_a_ledger_closes_when_the_block_under_it_raises(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    with pytest.raises(RuntimeError, match="refused"):
        with store.recording_writes():
            store.put_bytes(_key("9"), b"written before the refusal")
            raise RuntimeError("refused")
    assert store._write_ledgers == []


def test_eviction_makes_a_key_a_miss_again_and_repeats_harmlessly(
    tmp_path: Path,
) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("a")
    store.put_bytes(key, b"evict me")
    assert store.has(key)

    assert store.evict(key) is True
    assert store.has(key) is False
    assert store.evict(key) is False  # idempotent
    with pytest.raises(StoreMiss):
        store.load_bytes(key)

    store.put_bytes(key, b"written again")
    assert store.load_bytes(key) == b"written again"


def test_eviction_leaves_no_partial_object_and_no_tmp_litter(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    key = _key("b")
    store.put_frame(key, _frame())

    assert store.evict(key) is True
    assert not store.object_path(key).exists()
    assert list(store.tmp.iterdir()) == []


def test_eviction_of_a_key_never_written_is_not_an_error(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    assert store.evict(_key("c")) is False


def test_eviction_rejects_a_malformed_key(tmp_path: Path) -> None:
    store = ContentStore(tmp_path / "store")
    with pytest.raises(ValueError, match="key"):
        store.evict("not-a-key")
