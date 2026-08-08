from __future__ import annotations

import os
import stat
import time
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
    MassChangeRecord,
    WeightKind,
    Weights,
)


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
        pd.Series([True, pd.NA, False], dtype="boolean"),
        pd.Series(["a", "b", "a"], dtype="category"),
    ],
    ids=["nullable_integer", "nullable_boolean", "categorical"],
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
