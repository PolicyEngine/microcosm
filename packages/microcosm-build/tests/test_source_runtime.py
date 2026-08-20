from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeContext,
    SourceRuntimeError,
    UnsupportedSourceOperationError,
    run_source_stage,
)


def test_source_runtime_reads_table_as_defensive_copy() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "mini",
            "survey": "Synthetic",
            "source": "https://example.test/source",
            "grain": "person",
            "operations": [{"kind": "read_table", "table": "people"}],
            "outputs": ["income"],
        }
    )
    source = pd.DataFrame({"income": [1.0, 2.0]})

    result = run_source_stage(stage, tables={"people": source})

    result.loc[0, "income"] = 99.0
    assert source.loc[0, "income"] == 1.0


def test_source_runtime_refuses_missing_table() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "mini",
            "survey": "Synthetic",
            "source": "https://example.test/source",
            "grain": "person",
            "operations": [{"kind": "read_table", "table": "missing"}],
            "outputs": ["income"],
        }
    )

    with pytest.raises(SourceRuntimeError, match="was not provided"):
        run_source_stage(stage, tables={})


def test_source_runtime_refuses_unsupported_operation_after_read() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "mini",
            "survey": "Synthetic",
            "source": "https://example.test/source",
            "grain": "person",
            "operations": [
                {"kind": "read_table", "table": "people"},
                {"kind": "uprate", "from_year": 2020, "to_year": 2024},
            ],
            "outputs": ["income"],
        }
    )

    with pytest.raises(UnsupportedSourceOperationError, match="uprate"):
        run_source_stage(stage, tables={"people": pd.DataFrame({"income": [1.0]})})


def test_source_runtime_can_stop_after_manifest_prefix() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "mini",
            "survey": "Synthetic",
            "source": "https://example.test/source",
            "grain": "person",
            "operations": [
                {"kind": "read_table", "table": "people"},
                {"kind": "uprate", "from_year": 2020, "to_year": 2024},
            ],
            "outputs": ["income"],
        }
    )

    result = run_source_stage(
        stage,
        tables={"people": pd.DataFrame({"income": [1.0]})},
        stop_after="read_table",
    )

    assert result["income"].tolist() == [1.0]


def test_source_runtime_allows_injected_first_frame_handlers() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "mini",
            "survey": "Synthetic",
            "source": "https://example.test/source",
            "grain": "person",
            "operations": [
                {
                    "kind": "read_tables",
                    "tables": ["people", "income"],
                }
            ],
            "outputs": ["income"],
        }
    )

    def read_tables_handler(
        current: pd.DataFrame | None,
        _operation,
        context: SourceRuntimeContext,
    ) -> pd.DataFrame:
        assert current is None
        people = context.read_table("people")
        income = context.read_table("income")
        return people.merge(income, on="person_id")

    result = run_source_stage(
        stage,
        tables={
            "people": pd.DataFrame({"person_id": [1, 2]}),
            "income": pd.DataFrame({"person_id": [1, 2], "income": [10.0, 20.0]}),
        },
        operation_handlers={"read_tables": read_tables_handler},
    )

    assert result["income"].tolist() == [10.0, 20.0]


def test_source_runtime_threads_optional_narrow_rng_capability() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "mini",
            "survey": "Synthetic",
            "source": "https://example.test/source",
            "grain": "person",
            "operations": [{"kind": "derive"}],
            "outputs": ["income"],
        }
    )
    sentinel = object()

    def derive(
        current: pd.DataFrame | None,
        _operation,
        context: SourceRuntimeContext,
    ) -> pd.DataFrame:
        assert current is None
        assert context.rng is sentinel
        return pd.DataFrame({"income": [1.0]})

    result = run_source_stage(
        stage,
        tables={},
        operation_handlers={"derive": derive},
        rng=sentinel,  # type: ignore[arg-type]
    )

    assert result["income"].tolist() == [1.0]
