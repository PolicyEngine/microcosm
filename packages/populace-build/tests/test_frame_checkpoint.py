from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from populace.frame import (
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
            "float_measure": np.asarray([1.25, np.nan, -3.5], dtype=np.float32),
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
