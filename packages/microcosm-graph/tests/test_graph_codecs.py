"""Source-codec registry and shipped codec contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.codecs import SOURCE_CODECS, SourceCodecRegistry, load_source
from microcosm.graph.store import ContentStore, StoreUnavailable


def _frame() -> Frame:
    schema = EntitySchema(group_entities=("household",))
    person = pd.DataFrame(
        {
            "person_id": pd.Series([1, 2, 3], dtype="int64"),
            "person_household_id": pd.Series([10, 10, 20], dtype="int64"),
            "flag": pd.Series([True, pd.NA, False], dtype="boolean"),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": pd.Series([10, 20], dtype="int64"),
            "size": pd.Series([2, 1], dtype="int64"),
        }
    )
    return Frame(
        {"person": person, "household": household},
        schema,
        {"household": Weights(np.asarray([2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["urban", "urban", "rural"], dtype="string"),
    )


def test_csv_tables_loads_assignment_weights_json_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    frame = _frame()
    frame.table("person").assign(stratum=frame.strata).to_csv(
        source / "person.csv", index=False
    )
    frame.table("household").to_csv(source / "household.csv", index=False)
    (source / "schema.json").write_text(
        json.dumps(
            {
                "person_entity": "person",
                "group_entities": ["household"],
                "strata_column": "stratum",
                "dtypes": {
                    "person": {
                        "person_id": "int64",
                        "person_household_id": "int64",
                        "flag": "boolean",
                        "stratum": "string",
                    },
                    "household": {"household_id": "int64", "size": "int64"},
                },
            }
        )
    )
    (source / "weights.json").write_text(
        json.dumps({"household": {"kind": "design", "values": [2.0, 3.0]}})
    )

    loaded = load_source("csv-tables", source)

    assert loaded.schema == frame.schema
    for entity in frame.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), frame.table(entity))
    pd.testing.assert_series_equal(loaded.strata, frame.strata, check_names=False)
    assert loaded.weights_for("household").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        loaded.weights_for("household").values,
        frame.weights_for("household").values,
    )


def test_csv_tables_loads_tables_mapping_and_weights_csv_layout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    frame = _frame()
    frame.table("person").assign(stratum=frame.strata).to_csv(
        source / "people.csv", index=False
    )
    frame.table("household").to_csv(source / "homes.csv", index=False)
    pd.DataFrame({"household_id": [10, 20], "design_weight": [2.0, 3.0]}).to_csv(
        source / "weights.csv", index=False
    )
    (source / "schema.json").write_text(
        json.dumps(
            {
                "person_entity": "person",
                "group_entities": ["household"],
                "tables": {"person": "people.csv", "household": "homes.csv"},
                "dtypes": {
                    "person_id": "int64",
                    "person_household_id": "int64",
                    "household_id": "int64",
                    "size": "int64",
                    "flag": "boolean",
                    "stratum": "string",
                    "design_weight": "float64",
                },
                "strata_column": "stratum",
                "weights": {
                    "entity": "household",
                    "column": "design_weight",
                    "kind": "design",
                },
                "weights_table": "weights.csv",
            }
        )
    )

    loaded = SOURCE_CODECS.load("csv-tables", source)

    for entity in frame.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), frame.table(entity))
    pd.testing.assert_series_equal(loaded.strata, frame.strata, check_names=False)
    np.testing.assert_array_equal(
        loaded.weights_for("household").values,
        np.asarray([2.0, 3.0]),
    )


def test_frame_store_codec_loads_verified_object_directory(tmp_path: Path) -> None:
    frame = _frame()
    store = ContentStore(tmp_path / "store")
    path = store.put_frame("a" * 64, frame)

    loaded = SOURCE_CODECS.load("frame-store", path, store=store)

    assert loaded.schema == frame.schema
    for entity in frame.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), frame.table(entity))


def test_missing_codec_and_codec_dependency_are_store_unavailable(
    tmp_path: Path,
) -> None:
    registry = SourceCodecRegistry()
    with pytest.raises(StoreUnavailable, match="not installed"):
        registry.load("missing", tmp_path)

    def missing_dependency(_path: Path, *, store: ContentStore | None = None) -> Frame:
        del store
        raise ImportError("optional engine")

    registry.register("engine", missing_dependency)
    with pytest.raises(StoreUnavailable, match="dependency"):
        registry.load("engine", tmp_path)
