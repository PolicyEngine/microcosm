from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.frame_checkpoint as frame_checkpoint_module
from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChange,
    MassChangeRecord,
    SharedColumnChangedError,
    WeightKind,
    Weights,
)
from microcosm.frame import shared_columns as shared_columns_module


def _checkpoint_frame() -> Frame:
    schema = EntitySchema(
        group_entities=("household", "firm"),
        links=(LinkSpec(name="jobs", left_entity="person", right_entity="firm"),),
    )
    person_index = pd.Index([101, 103, 107], name="source_person_row")
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "person_firm_id": np.asarray([100, 200, 200], dtype=np.int64),
            "small_integer": np.asarray([1, -2, 3], dtype=np.int8),
            "unsigned_code": np.asarray([4, 5, 6], dtype=np.uint32),
            "float_measure": np.asarray(
                [0x3FA00000, 0x7FC01234, 0xC0600000], dtype=np.uint32
            ).view(np.float32),
            "observed": np.asarray([True, False, True], dtype=np.bool_),
            "nullable_flag": pd.Series(
                [True, np.nan, False],
                index=person_index,
                dtype=object,
            ),
            "nullable_label": pd.Series(
                ["first", np.nan, "third"],
                index=person_index,
                dtype=object,
            ),
            "missing_sentinels": pd.Series(
                [None, pd.NA, pd.NaT],
                index=person_index,
                dtype=object,
            ),
            "string_label": pd.Series(
                ["alpha", None, "gamma"],
                index=person_index,
                dtype="str",
            ),
            "event_date": pd.Series(
                ["2024-01-01", None, "2024-12-31"],
                index=person_index,
                dtype="datetime64[ns]",
            ),
        },
        index=person_index,
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int32),
            "household_income": np.asarray([50_000.0, 75_000.0], dtype=np.float64),
        },
        index=pd.Index([41, 43], name="source_household_row"),
    )
    firm = pd.DataFrame(
        {
            "firm_id": np.asarray([100, 200], dtype=np.int16),
            "firm_name": pd.Series(
                ["A", "B"],
                index=pd.Index([51, 53], name="source_firm_row"),
                dtype="str",
            ),
        },
        index=pd.Index([51, 53], name="source_firm_row"),
    )
    jobs = pd.DataFrame(
        {
            "person_id": np.asarray([1, 1, 2, 3], dtype=np.int64),
            "firm_id": np.asarray([100, 200, 200, 200], dtype=np.int16),
            "hours": np.asarray([20.0, 10.0, 40.0, 35.0], dtype=np.float32),
        },
        index=pd.Index([71, 73, 79, 83], name="source_job_row"),
    )
    weights = {
        "person": Weights(
            np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
            WeightKind.CALIBRATED,
        ),
        "household": Weights(
            np.asarray([1.5, 2.5], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
    }
    strata = pd.Series(
        ["asec_2024", "puf_support", "asec_2023"],
        index=person_index,
        name="stratum",
        dtype=object,
    )
    mass_log = (
        MassChangeRecord(
            entity="household",
            old_total=4.0,
            new_total=4.0,
            declared_factor=1.0,
            reason="split source mass across support channels",
        ),
        MassChangeRecord(
            entity="person",
            old_total=3.0,
            new_total=6.0,
            declared_factor=2.0,
            reason="calibrated person mass to a reviewed control",
        ),
    )
    return Frame(
        {
            "person": person,
            "household": household,
            "firm": firm,
            "jobs": jobs,
        },
        schema,
        weights,
        strata,
        mass_log=mass_log,
    )


def _nullable_boolean_checkpoint_frame() -> Frame:
    frame = _checkpoint_frame()
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables.update({name: frame.link(name).copy() for name in frame.links})
    person = tables["person"]
    person["complete_nullable_boolean"] = pd.Series(
        [True, False, True],
        index=person.index,
        dtype="boolean",
    )
    person["missing_nullable_boolean"] = pd.Series(
        [True, pd.NA, False],
        index=person.index,
        dtype="boolean",
    )
    jobs = tables["jobs"]
    jobs["link_nullable_boolean"] = pd.Series(
        [pd.NA, False, True, pd.NA],
        index=jobs.index,
        dtype="boolean",
    )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _checkpoint_series_group(root, *, table: str, column: str):
    metadata = json.loads(np.asarray(root["metadata_json"]).tobytes())
    table_position = next(
        position
        for position, spec in enumerate(metadata["tables"])
        if spec["name"] == table
    )
    table_spec = metadata["tables"][table_position]
    column_position = next(
        position
        for position, spec in enumerate(table_spec["columns"])
        if spec["name"] == column
    )
    return (
        metadata,
        table_spec["columns"][column_position],
        root[f"tables/t{table_position:05d}/columns/c{column_position:05d}"],
    )


def _replace_checkpoint_metadata(root, metadata: dict[str, object]) -> None:
    encoded = frame_checkpoint_module._canonical_json(metadata).encode("utf-8")
    del root["metadata_json"]
    root.create_dataset(
        "metadata_json",
        data=np.frombuffer(encoded, dtype=np.uint8),
        track_times=False,
    )


def test_frame_checkpoint_round_trip_is_byte_identical(tmp_path: Path) -> None:
    frame = _checkpoint_frame()
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    external_metadata = {
        "stage": "b",
        "period": 2024,
        "audit": {
            "passed": True,
            "resolved_weight_kinds": ["design", "importance"],
        },
    }

    write_frame_checkpoint(first_path, frame, metadata=external_metadata)
    loaded = load_frame_checkpoint(first_path)
    # HDF object timestamps have one-second resolution on common platforms.
    # Crossing that boundary ensures determinism is real, not a same-second
    # false positive from the serializer.
    time.sleep(1.1)
    write_frame_checkpoint(second_path, loaded.frame, metadata=loaded.metadata)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert loaded.metadata == external_metadata
    assert loaded.frame.schema == frame.schema
    assert loaded.frame.entities == frame.entities
    assert loaded.frame.links == frame.links
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            loaded.frame.table(entity),
            frame.table(entity),
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_exact=True,
        )
    for link in frame.links:
        pd.testing.assert_frame_equal(
            loaded.frame.link(link),
            frame.link(link),
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_exact=True,
        )
    assert loaded.frame.weighted_entities == frame.weighted_entities
    for entity in frame.weighted_entities:
        actual = loaded.frame.weights_for(entity)
        expected = frame.weights_for(entity)
        assert actual.kind is expected.kind
        np.testing.assert_array_equal(actual.values, expected.values)
    pd.testing.assert_series_equal(
        loaded.frame.strata,
        frame.strata,
        check_dtype=True,
        check_index_type=True,
        check_exact=True,
    )
    assert loaded.frame.mass_log == frame.mass_log
    sentinels = loaded.frame.person["missing_sentinels"].tolist()
    assert sentinels[0] is None
    assert sentinels[1] is pd.NA
    assert sentinels[2] is pd.NaT
    np.testing.assert_array_equal(
        loaded.frame.person["float_measure"].to_numpy().view(np.uint32),
        frame.person["float_measure"].to_numpy().view(np.uint32),
    )


def test_frame_without_nullable_booleans_keeps_schema_2_byte_golden(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-schema-2.h5"

    write_frame_checkpoint(path, _checkpoint_frame())

    # No cross-platform byte constant: HDF5 wheels differ between macOS and
    # Linux, so whole-file hashes are platform-dependent (run-stable only).
    # The contract is determinism plus staying on schema 2 for frames with
    # no nullable booleans — assert exactly that.
    rewrite = tmp_path / "legacy-schema-2-rewrite.h5"
    write_frame_checkpoint(rewrite, _checkpoint_frame())
    assert path.read_bytes() == rewrite.read_bytes()
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r") as h5:
        raw = np.asarray(h5["_populace_frame_checkpoint/metadata_json"]).tobytes()
    assert json.loads(raw.decode("utf-8"))["schema_version"] == 2


def test_nullable_boolean_round_trip_preserves_dtype_values_and_null_masks(
    tmp_path: Path,
) -> None:
    frame = _nullable_boolean_checkpoint_frame()
    first_path = tmp_path / "nullable-first.h5"
    second_path = tmp_path / "nullable-second.h5"

    write_frame_checkpoint(first_path, frame)
    loaded = load_frame_checkpoint(first_path)
    write_frame_checkpoint(second_path, loaded.frame)

    assert first_path.read_bytes() == second_path.read_bytes()
    for table, column in (
        ("person", "complete_nullable_boolean"),
        ("person", "missing_nullable_boolean"),
        ("jobs", "link_nullable_boolean"),
    ):
        loaded_table = (
            loaded.frame.table(table)
            if table in loaded.frame.entities
            else loaded.frame.link(table)
        )
        expected_table = (
            frame.table(table) if table in frame.entities else frame.link(table)
        )
        pd.testing.assert_series_equal(
            loaded_table[column],
            expected_table[column],
            check_dtype=True,
            check_exact=True,
        )
        assert loaded_table[column].dtype == pd.BooleanDtype()

    h5py = pytest.importorskip("h5py")
    with h5py.File(first_path, mode="r") as h5:
        root = h5["_populace_frame_checkpoint"]
        metadata = json.loads(np.asarray(root["metadata_json"]).tobytes())
        assert metadata["schema_version"] == 3
        for table, column, expected_mask in (
            ("person", "complete_nullable_boolean", None),
            ("person", "missing_nullable_boolean", [0, 1, 0]),
            ("jobs", "link_nullable_boolean", [1, 0, 0, 1]),
        ):
            _metadata, spec, group = _checkpoint_series_group(
                root,
                table=table,
                column=column,
            )
            assert spec == {
                "name": column,
                "dtype": "boolean",
                "encoding": "nullable_boolean_v1",
                "has_null_mask": expected_mask is not None,
            }
            assert np.asarray(group["values"]).dtype == np.dtype(np.bool_)
            if expected_mask is None:
                assert "null_mask" not in group
            else:
                mask = np.asarray(group["null_mask"])
                assert mask.dtype == np.dtype(np.uint8)
                assert mask.tolist() == expected_mask


def test_nullable_boolean_masked_storage_bits_are_canonical(
    tmp_path: Path,
) -> None:
    false_hidden = _nullable_boolean_checkpoint_frame()
    true_hidden = _nullable_boolean_checkpoint_frame()
    false_person = false_hidden.person
    true_person = true_hidden.person
    false_person["missing_nullable_boolean"] = pd.Series(
        pd.arrays.BooleanArray(
            np.asarray([True, False, False], dtype=np.bool_),
            np.asarray([False, True, False], dtype=np.bool_),
        ),
        index=false_person.index,
    )
    true_person["missing_nullable_boolean"] = pd.Series(
        pd.arrays.BooleanArray(
            np.asarray([True, True, False], dtype=np.bool_),
            np.asarray([False, True, False], dtype=np.bool_),
        ),
        index=true_person.index,
    )
    first_path = tmp_path / "hidden-false.h5"
    second_path = tmp_path / "hidden-true.h5"

    write_frame_checkpoint(first_path, false_hidden)
    write_frame_checkpoint(second_path, true_hidden)

    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "all_zero",
        "nonbinary",
        "wrong_length",
        "wrong_dtype",
        "wrong_rank",
    ],
)
def test_nullable_boolean_checkpoint_rejects_malformed_null_mask(
    tmp_path: Path,
    damage: str,
) -> None:
    path = tmp_path / f"malformed-{damage}.h5"
    write_frame_checkpoint(path, _nullable_boolean_checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        _metadata, _spec, group = _checkpoint_series_group(
            root,
            table="person",
            column="missing_nullable_boolean",
        )
        del group["null_mask"]
        if damage == "all_zero":
            group.create_dataset(
                "null_mask",
                data=np.asarray([0, 0, 0], dtype=np.uint8),
                track_times=False,
            )
        elif damage == "nonbinary":
            group.create_dataset(
                "null_mask",
                data=np.asarray([0, 2, 0], dtype=np.uint8),
                track_times=False,
            )
        elif damage == "wrong_length":
            group.create_dataset(
                "null_mask",
                data=np.asarray([0, 1], dtype=np.uint8),
                track_times=False,
            )
        elif damage == "wrong_dtype":
            group.create_dataset(
                "null_mask",
                data=np.asarray([0, 1, 0], dtype=np.int16),
                track_times=False,
            )
        elif damage == "wrong_rank":
            group.create_dataset(
                "null_mask",
                data=np.asarray([[0, 1, 0]], dtype=np.uint8),
                track_times=False,
            )

    with pytest.raises(ValueError, match="null mask"):
        load_frame_checkpoint(path)


@pytest.mark.parametrize("damage", ["wrong_dtype", "wrong_rank"])
def test_nullable_boolean_checkpoint_rejects_malformed_values(
    tmp_path: Path,
    damage: str,
) -> None:
    path = tmp_path / f"malformed-values-{damage}.h5"
    write_frame_checkpoint(path, _nullable_boolean_checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        _metadata, _spec, group = _checkpoint_series_group(
            root,
            table="person",
            column="missing_nullable_boolean",
        )
        del group["values"]
        values = (
            np.asarray([1, 0, 0], dtype=np.uint8)
            if damage == "wrong_dtype"
            else np.asarray([[True, False, False]], dtype=np.bool_)
        )
        group.create_dataset("values", data=values, track_times=False)

    with pytest.raises(ValueError, match="nullable boolean values"):
        load_frame_checkpoint(path)


def test_maskless_nullable_boolean_rejects_unexpected_null_mask(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unexpected-mask.h5"
    write_frame_checkpoint(path, _nullable_boolean_checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        _metadata, _spec, group = _checkpoint_series_group(
            root,
            table="person",
            column="complete_nullable_boolean",
        )
        group.create_dataset(
            "null_mask",
            data=np.zeros(3, dtype=np.uint8),
            track_times=False,
        )

    with pytest.raises(ValueError, match="unexpected null mask"):
        load_frame_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("has_null_mask", "yes", "has_null_mask"),
        ("dtype", "bool", "declared dtype"),
    ],
)
def test_nullable_boolean_checkpoint_rejects_malformed_spec(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / f"malformed-spec-{field}.h5"
    write_frame_checkpoint(path, _nullable_boolean_checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        metadata, spec, _group = _checkpoint_series_group(
            root,
            table="person",
            column="missing_nullable_boolean",
        )
        spec[field] = value
        _replace_checkpoint_metadata(root, metadata)

    with pytest.raises(ValueError, match=message):
        load_frame_checkpoint(path)


def test_schema_2_checkpoint_rejects_boolean_dtype_under_legacy_encoding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged-schema-2-object-boolean.h5"
    write_frame_checkpoint(path, _checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        metadata, spec, _group = _checkpoint_series_group(
            root,
            table="person",
            column="nullable_flag",
        )
        spec["dtype"] = "boolean"
        _replace_checkpoint_metadata(root, metadata)

    with pytest.raises(ValueError, match="schema version 2.*boolean"):
        load_frame_checkpoint(path)


def test_schema_2_checkpoint_cannot_smuggle_nullable_boolean_encoding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged-schema-2.h5"
    write_frame_checkpoint(path, _nullable_boolean_checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        metadata = json.loads(np.asarray(root["metadata_json"]).tobytes())
        metadata["schema_version"] = 2
        _replace_checkpoint_metadata(root, metadata)

    with pytest.raises(ValueError, match="schema version 2.*nullable"):
        load_frame_checkpoint(path)


def test_schema_3_checkpoint_requires_nullable_boolean_encoding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged-schema-3.h5"
    write_frame_checkpoint(path, _checkpoint_frame())
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, mode="r+") as h5:
        root = h5["_populace_frame_checkpoint"]
        metadata = json.loads(np.asarray(root["metadata_json"]).tobytes())
        metadata["schema_version"] = 3
        _replace_checkpoint_metadata(root, metadata)

    with pytest.raises(ValueError, match="schema version 3.*nullable"):
        load_frame_checkpoint(path)


def test_frame_checkpoint_fsyncs_parent_directory_after_rename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(frame_checkpoint_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(frame_checkpoint_module.os, "replace", tracked_replace)
    output = tmp_path / "durable.frame.h5"

    write_frame_checkpoint(output, _checkpoint_frame())

    assert events == [
        ("replace", output.name),
        ("fsync", "directory"),
    ]


def test_frame_checkpoint_restores_caller_bound_frame_metadata_without_rewrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.h5"
    frame_metadata = {
        "assembly": {"channels": ["asec", "acs"]},
        "checkpoint_identity": "a" * 64,
    }
    write_frame_checkpoint(path, _checkpoint_frame(), metadata={"stage": "assembled"})

    loaded = load_frame_checkpoint(path, frame_metadata=frame_metadata)

    assert loaded.metadata == {"stage": "assembled"}
    assert loaded.frame.metadata["checkpoint_identity"] == "a" * 64
    assert tuple(loaded.frame.metadata["assembly"]["channels"]) == ("asec", "acs")


def test_frame_checkpoint_preserves_range_indexes_and_column_axis_name(
    tmp_path: Path,
) -> None:
    original = _checkpoint_frame()
    tables = {
        entity: original.table(entity).reset_index(drop=True)
        for entity in original.entities
    }
    tables.update(
        {link: original.link(link).reset_index(drop=True) for link in original.links}
    )
    tables["person"].columns.name = ("person", "variables")
    frame = Frame(
        tables,
        original.schema,
        {entity: original.weights_for(entity) for entity in original.weighted_entities},
        original.strata.reset_index(drop=True),
        mass_log=original.mass_log,
    )

    path = tmp_path / "range-index.h5"
    write_frame_checkpoint(path, frame)
    loaded = load_frame_checkpoint(path).frame

    for entity in frame.entities:
        assert isinstance(loaded.table(entity).index, pd.RangeIndex)
    for link in frame.links:
        assert isinstance(loaded.link(link).index, pd.RangeIndex)
    assert loaded.person.columns.name == ("person", "variables")


@pytest.mark.parametrize(
    "unsupported",
    [
        pd.Series([1, pd.NA, 3], dtype="Int64"),
        pd.Series(["a", "b", "a"], dtype="category"),
    ],
    ids=["nullable_integer", "categorical"],
)
def test_frame_checkpoint_rejects_unsupported_dtype_without_replacing_destination(
    tmp_path: Path,
    unsupported: pd.Series,
) -> None:
    frame = _checkpoint_frame()
    frame.person["unsupported"] = unsupported.set_axis(frame.person.index)
    destination = tmp_path / "checkpoint.h5"
    destination.write_bytes(b"previous-good-checkpoint")

    with pytest.raises(TypeError, match="does not support"):
        write_frame_checkpoint(destination, frame)

    assert destination.read_bytes() == b"previous-good-checkpoint"
    assert not (tmp_path / ".checkpoint.h5.tmp").exists()


def test_frame_checkpoint_rejects_nonfinite_external_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite and JSON-compatible"):
        write_frame_checkpoint(
            tmp_path / "checkpoint.h5",
            _checkpoint_frame(),
            metadata={"peak_rss": np.nan},
        )


def test_string_dtype_restores_python_storage_under_pyarrow_default(
    tmp_path: Path,
) -> None:
    """A checkpoint restores one physical string dtype in every environment.

    ``str(StringDtype)`` collapses to "str"/"string", which pandas resolves
    to the environment-default storage — pyarrow wherever pyarrow is
    installed. The spec records storage explicitly, so a checkpoint written
    with canonical python-storage strings restores python storage even when
    the reading environment defaults to pyarrow.
    """
    pytest.importorskip("pyarrow")
    frame = _checkpoint_frame()
    canonical = pd.StringDtype(storage="python", na_value=np.nan)
    person = frame.table("person")
    person["canonical_label"] = pd.Series(
        ["alpha", "beta", "gamma"], index=person.index, dtype=canonical
    )
    path = tmp_path / "strings.h5"
    write_frame_checkpoint(path, frame)

    with pd.option_context("mode.string_storage", "pyarrow"):
        restored = load_frame_checkpoint(path).frame
        dtype = restored.table("person")["canonical_label"].dtype
        assert isinstance(dtype, pd.StringDtype)
        assert dtype.storage == "python"
        assert dtype == canonical


def test_legacy_string_specs_restore_python_storage() -> None:
    """Specs written before storage was recorded restore deterministically."""
    resolved = frame_checkpoint_module._declared_dtype(
        {"dtype": "str", "encoding": "object_scalars_v1"},
        path=Path("legacy.h5"),
        label="person.label",
    )
    assert isinstance(resolved, pd.StringDtype)
    assert resolved.storage == "python"
    assert resolved.na_value is not pd.NA

    resolved_na = frame_checkpoint_module._declared_dtype(
        {"dtype": "string", "encoding": "object_scalars_v1"},
        path=Path("legacy.h5"),
        label="person.label",
    )
    assert isinstance(resolved_na, pd.StringDtype)
    assert resolved_na.storage == "python"
    assert resolved_na.na_value is pd.NA

    with pytest.raises(ValueError, match="unknown string storage"):
        frame_checkpoint_module._declared_dtype(
            {"dtype": "str", "string_storage": "bogus"},
            path=Path("legacy.h5"),
            label="person.label",
        )


@pytest.fixture
def copy_on_write_on() -> Iterator[None]:
    """Turn copy-on-write on under pandas 2.x; pandas 3 always has it on."""

    if int(pd.__version__.split(".", 1)[0]) >= 3:
        yield
        return
    with pd.option_context("mode.copy_on_write", True):
        yield


def test_loader_keeps_the_decoded_columns_without_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, copy_on_write_on: None
) -> None:
    """The loaded frame holds the columns the loader decoded, not a copy.

    The loader builds its one frame through ``Frame.adopt``, so reading a
    checkpoint holds one copy of the data instead of the decoded columns plus
    the constructor's copy of them.
    """
    path = tmp_path / "frame.h5"
    write_frame_checkpoint(path, _checkpoint_frame())
    decoded: dict[str, pd.Series] = {}
    read_series = frame_checkpoint_module._read_series

    def recording_read_series(group, spec, input_path, *, label):
        series = read_series(group, spec, input_path, label=label)
        decoded[label] = series
        return series

    monkeypatch.setattr(frame_checkpoint_module, "_read_series", recording_read_series)

    loaded = load_frame_checkpoint(path).frame

    compared = []
    for name, table in frame_checkpoint_module._frame_tables(loaded):
        for column in table.select_dtypes("number").columns:
            assert np.shares_memory(
                table[column].to_numpy(), decoded[f"{name}.{column}"].to_numpy()
            ), f"{name}.{column}"
            compared.append(f"{name}.{column}")
    assert {"person.float_measure", "household.household_income"} <= set(compared)


def _named_axes_frame() -> Frame:
    """The checkpoint fixture with named column axes on two of its tables."""

    frame = _checkpoint_frame()
    tables = dict(frame_checkpoint_module._frame_tables(frame))
    for name in ("person", "household"):
        tables[name] = tables[name].copy()
        tables[name].columns.name = f"{name}_fields"
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


#: Source frames for the checkpoint comparisons below.
_SOURCE_FRAMES = {
    "fixture": _checkpoint_frame,
    "named column axes": _named_axes_frame,
}


def _without_links(frame: Frame) -> Frame:
    """The frame with its link tables dropped, which ``Frame.select`` requires."""

    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        EntitySchema(
            person_entity=frame.schema.person_entity,
            group_entities=frame.schema.group_entities,
        ),
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _doubled_household_weights(frame: Frame) -> Frame:
    current = frame.weights_for("household")
    return frame.with_weights(
        "household",
        Weights(current.values * 2.0, WeightKind.CALIBRATED),
        mass=MassChange(factor=2.0, reason="test calibration"),
    )


#: Operations that build their result through ``Frame.adopt``.
_ADOPTING_OPERATIONS = {
    "with_weights": _doubled_household_weights,
    "place person to household": lambda frame: frame.place(
        "small_integer", "household", how="sum"
    ),
    "place household to person": lambda frame: frame.place(
        "household_income", "person", how="broadcast"
    ),
    "place a string to firm": lambda frame: frame.place(
        "string_label", "firm", how="first"
    ),
    "select every person": lambda frame: _without_links(frame).select(
        np.ones(frame.n("person"), dtype=bool)
    ),
    "select two persons": lambda frame: _without_links(frame).select(
        np.asarray([True, False, True])
    ),
    "no operation": lambda frame: frame,
}


def _derived_frame(
    directory: Path, source_frame: str, source: str, operation: str
) -> Frame:
    """Run ``operation`` on a source frame, built directly or loaded from disk."""

    frame = _SOURCE_FRAMES[source_frame]()
    if source == "loaded":
        write_frame_checkpoint(directory / "source.h5", frame)
        frame = load_frame_checkpoint(directory / "source.h5").frame
    return _ADOPTING_OPERATIONS[operation](frame)


def _checkpoint_content_sha256(path: Path) -> str:
    """SHA-256 of every dataset in a checkpoint: its name, dtype, shape and bytes.

    The metadata JSON is one of the datasets. Hashing what the datasets hold,
    rather than the file, keeps the digest independent of the HDF5 build,
    whose file layout differs between platforms.
    """

    h5py = pytest.importorskip("h5py")
    digest = hashlib.sha256()
    with h5py.File(path, mode="r") as h5:
        dataset_names: list[str] = []
        h5.visititems(
            lambda name, item: (
                dataset_names.append(name) if isinstance(item, h5py.Dataset) else None
            )
        )
        for name in sorted(dataset_names):
            values = np.asarray(h5[name])
            digest.update(f"{name}\0{values.dtype.str}\0{values.shape!r}\0".encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


#: Checkpoint content digests of each operation's result, recorded with the
#: copying implementation at commit 8afea44e9, before ``with_weights``,
#: ``place``, ``select`` and the loader moved to ``Frame.adopt``.
_COPYING_IMPLEMENTATION_CONTENT_SHA256: dict[tuple[str, str, str], str] = {
    (
        "fixture",
        "constructed",
        "with_weights",
    ): "a3f0e69dd495e6f9e45caa4d13851a9356c7810df828fbe2bdba38a94272db72",
    (
        "fixture",
        "constructed",
        "place person to household",
    ): "9f5e81635306d47e73ae0fb4b593edcf1f47b6b40c865197e641a9b1adfb89c5",
    (
        "fixture",
        "constructed",
        "place household to person",
    ): "92eafb157899735517c357a02459528b5f8b9d08a5f6c070a79d4cb043bd314b",
    (
        "fixture",
        "constructed",
        "place a string to firm",
    ): "a7314048203f36611c2ff9011150b5bda8476407b618949756f681df5c82cb7b",
    (
        "fixture",
        "constructed",
        "select every person",
    ): "05ef02724683caa4e5815702b3a2d291573c8e39fba9022e4b3360da4054d0e8",
    (
        "fixture",
        "constructed",
        "select two persons",
    ): "4a4dc266b99efa4075a3bbf45163040e21445a62da8d0c2eee7a4fd06abc0672",
    (
        "fixture",
        "constructed",
        "no operation",
    ): "ad3d5745332e9725b941f2db6e1eb60e5d93dfdd5204fefe3497e7c702b5e543",
    (
        "fixture",
        "loaded",
        "with_weights",
    ): "a3f0e69dd495e6f9e45caa4d13851a9356c7810df828fbe2bdba38a94272db72",
    (
        "fixture",
        "loaded",
        "place person to household",
    ): "9f5e81635306d47e73ae0fb4b593edcf1f47b6b40c865197e641a9b1adfb89c5",
    (
        "fixture",
        "loaded",
        "place household to person",
    ): "92eafb157899735517c357a02459528b5f8b9d08a5f6c070a79d4cb043bd314b",
    (
        "fixture",
        "loaded",
        "place a string to firm",
    ): "a7314048203f36611c2ff9011150b5bda8476407b618949756f681df5c82cb7b",
    (
        "fixture",
        "loaded",
        "select every person",
    ): "05ef02724683caa4e5815702b3a2d291573c8e39fba9022e4b3360da4054d0e8",
    (
        "fixture",
        "loaded",
        "select two persons",
    ): "4a4dc266b99efa4075a3bbf45163040e21445a62da8d0c2eee7a4fd06abc0672",
    (
        "fixture",
        "loaded",
        "no operation",
    ): "ad3d5745332e9725b941f2db6e1eb60e5d93dfdd5204fefe3497e7c702b5e543",
    (
        "named column axes",
        "constructed",
        "with_weights",
    ): "f29f6e2ff48a87b5ea477b673c1d9947b1a58cb235dbee2eb83aaf72f15e37f9",
    (
        "named column axes",
        "constructed",
        "place person to household",
    ): "dc49c663387ebf528fd5586eab0067981d972bc02f8ab7e2f15615c25481ed27",
    (
        "named column axes",
        "constructed",
        "place household to person",
    ): "e6d89ab3373d5057399ef4983bad424bbab31399bd25b95c1743951474c3a83e",
    (
        "named column axes",
        "constructed",
        "place a string to firm",
    ): "995b70f552de6b61271ef88b6ea6f2ed3912e01c9e585ce9c11fa18e7e0358b6",
    (
        "named column axes",
        "constructed",
        "select every person",
    ): "8cb1bcfff2e6c27fb8067c5a9bc3745704aab1f5e67073093d2ca581e108c4b9",
    (
        "named column axes",
        "constructed",
        "select two persons",
    ): "d7e010959ae5b3f231bd64ace603b4b529bbe66fda946da0750df0b2f5ea8c30",
    (
        "named column axes",
        "constructed",
        "no operation",
    ): "56c83a3998b88b9a950bf1da9ffb506a79f6d6666669de04e8ee76fd4b3ba53c",
    (
        "named column axes",
        "loaded",
        "with_weights",
    ): "f29f6e2ff48a87b5ea477b673c1d9947b1a58cb235dbee2eb83aaf72f15e37f9",
    (
        "named column axes",
        "loaded",
        "place person to household",
    ): "dc49c663387ebf528fd5586eab0067981d972bc02f8ab7e2f15615c25481ed27",
    (
        "named column axes",
        "loaded",
        "place household to person",
    ): "e6d89ab3373d5057399ef4983bad424bbab31399bd25b95c1743951474c3a83e",
    (
        "named column axes",
        "loaded",
        "place a string to firm",
    ): "995b70f552de6b61271ef88b6ea6f2ed3912e01c9e585ce9c11fa18e7e0358b6",
    (
        "named column axes",
        "loaded",
        "select every person",
    ): "8cb1bcfff2e6c27fb8067c5a9bc3745704aab1f5e67073093d2ca581e108c4b9",
    (
        "named column axes",
        "loaded",
        "select two persons",
    ): "d7e010959ae5b3f231bd64ace603b4b529bbe66fda946da0750df0b2f5ea8c30",
    (
        "named column axes",
        "loaded",
        "no operation",
    ): "56c83a3998b88b9a950bf1da9ffb506a79f6d6666669de04e8ee76fd4b3ba53c",
}


_COMPARISON_CASES = [
    (source_frame, source, operation)
    for source_frame in _SOURCE_FRAMES
    for source in ("constructed", "loaded")
    for operation in _ADOPTING_OPERATIONS
]


@pytest.mark.parametrize(("source_frame", "source", "operation"), _COMPARISON_CASES)
def test_adopted_frames_write_the_checkpoint_content_of_the_copying_code(
    tmp_path: Path, source_frame: str, source: str, operation: str
) -> None:
    """Each operation's result writes what the copying implementation wrote."""

    derived = _derived_frame(tmp_path, source_frame, source, operation)
    write_frame_checkpoint(tmp_path / "derived.h5", derived)

    assert (
        _checkpoint_content_sha256(tmp_path / "derived.h5")
        == _COPYING_IMPLEMENTATION_CONTENT_SHA256[(source_frame, source, operation)]
    )


@pytest.mark.parametrize(("source_frame", "source", "operation"), _COMPARISON_CASES)
def test_adopted_frames_write_the_bytes_of_copied_frames(
    tmp_path: Path, source_frame: str, source: str, operation: str
) -> None:
    """A shared table layout writes the same file as a copied one.

    The reference rebuilds the derived frame through the copying constructor,
    which merges each table's internal blocks. Both files are written in this
    process, so the comparison does not depend on HDF5 library bytes that
    differ between platforms.
    """

    derived = _derived_frame(tmp_path, source_frame, source, operation)
    copied = Frame(
        dict(frame_checkpoint_module._frame_tables(derived)),
        derived.schema,
        {entity: derived.weights_for(entity) for entity in derived.weighted_entities},
        derived.strata,
        mass_log=derived.mass_log,
        metadata=derived.metadata,
    )

    write_frame_checkpoint(tmp_path / "derived.h5", derived)
    write_frame_checkpoint(tmp_path / "copied.h5", copied)

    assert (tmp_path / "derived.h5").read_bytes() == (
        tmp_path / "copied.h5"
    ).read_bytes()


def test_writer_refuses_a_frame_whose_shared_column_changed(tmp_path: Path) -> None:
    source = _checkpoint_frame()
    shared = source.replace_tables()
    np.add.at(source.table("person")["float_measure"].to_numpy(), [0], 1.0)

    with pytest.raises(SharedColumnChangedError) as raised:
        write_frame_checkpoint(tmp_path / "frame.h5", shared)

    assert raised.value.changed_columns == (("person", "float_measure"),)
    assert list(tmp_path.iterdir()) == []


def test_loading_hashes_nothing_until_a_loaded_column_is_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "frame.h5"
    write_frame_checkpoint(path, _checkpoint_frame())
    hashed_lengths: list[int] = []
    original_digest = shared_columns_module._buffer_digest

    def counting_digest(stored: shared_columns_module._StoredArray) -> bytes:
        hashed_lengths.append(len(stored.values))
        return original_digest(stored)

    monkeypatch.setattr(shared_columns_module, "_buffer_digest", counting_digest)

    loaded = load_frame_checkpoint(path).frame
    assert hashed_lengths == []

    shared = loaded.replace_tables()
    assert hashed_lengths != []
    np.add.at(loaded.table("person")["float_measure"].to_numpy(), [0], 1.0)
    with pytest.raises(SharedColumnChangedError):
        shared.check_shared_columns()
