from __future__ import annotations

import ast
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.frame_serializer_registry import (
    FRAME_TABLE_SERIALIZERS,
    HDF_WRITE_EXCLUSIONS,
    FrameSerializerSpec,
)
from microcosm.build.uk_runtime.national_build import (
    _read_uk_national_tables,
    _write_uk_single_year_tables,
)
from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
from microcosm.frame import (
    US_SCHEMA,
    EntitySchema,
    Frame,
    WeightKind,
    Weights,
    read_frame_table,
)
from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOTS = (
    REPOSITORY_ROOT / "packages",
    REPOSITORY_ROOT / "tools",
)
COMPLETE_COLUMN = "fixture_complete_nullable_boolean"
MISSING_COLUMN = "fixture_missing_nullable_boolean"
NATIVE_COLUMN = "fixture_native_boolean"


@dataclass(frozen=True)
class BooleanRoundTrip:
    source: pd.DataFrame
    source_before_write: pd.DataFrame
    loaded: pd.DataFrame
    stored_complete_values: np.ndarray
    stored_missing_values: np.ndarray
    stored_missing_mask: np.ndarray
    stored_missing_mask_dtype: np.dtype | None


RoundTripAdapter = Callable[[Path, str], BooleanRoundTrip]


def _dtype_family_table(
    nullable_case: str,
    *,
    id_column: str = "person_id",
) -> pd.DataFrame:
    index = pd.RangeIndex(3)
    missing_values = {
        "mixed": [True, pd.NA, False],
        "all_missing": [pd.NA, pd.NA, pd.NA],
    }[nullable_case]
    return pd.DataFrame(
        {
            id_column: np.asarray([1, 2, 3], dtype=np.int64),
            NATIVE_COLUMN: np.asarray([False, True, False], dtype=np.bool_),
            COMPLETE_COLUMN: pd.Series(
                [True, False, True], index=index, dtype="boolean"
            ),
            MISSING_COLUMN: pd.Series(missing_values, index=index, dtype="boolean"),
        },
        index=index,
    )


def _semantic_observation(
    source: pd.DataFrame,
    source_before_write: pd.DataFrame,
    loaded: pd.DataFrame,
) -> BooleanRoundTrip:
    missing = loaded[MISSING_COLUMN]
    return BooleanRoundTrip(
        source=source,
        source_before_write=source_before_write,
        loaded=loaded,
        stored_complete_values=loaded[COMPLETE_COLUMN].to_numpy(
            dtype=np.bool_, copy=False
        ),
        stored_missing_values=missing.to_numpy(
            dtype=np.bool_, na_value=False, copy=False
        ),
        stored_missing_mask=missing.isna().to_numpy(dtype=np.bool_, copy=False),
        stored_missing_mask_dtype=None,
    )


def _small_frame(source: pd.DataFrame) -> Frame:
    household = pd.DataFrame(
        {"household_id": np.asarray([1, 2, 3], dtype=np.int64)},
        index=source.index,
    )
    person = source.copy(deep=False)
    person.insert(
        1,
        "person_household_id",
        np.asarray([1, 2, 3], dtype=np.int64),
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                np.asarray([1.0, 2.0, 3.0]),
                WeightKind.DESIGN,
            )
        },
    )


def _us_frame(source: pd.DataFrame) -> Frame:
    ids = np.asarray([1, 2, 3], dtype=np.int64)
    person = source.copy(deep=False)
    for position, entity in enumerate(US_SCHEMA.group_entities, start=1):
        person.insert(position, US_SCHEMA.membership_column(entity), ids)
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids}, index=source.index)
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
    )


def _load_tool(relative_path: str, module_name: str):
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_column_group(root, *, table: str, column: str):
    metadata = json.loads(np.asarray(root["metadata_json"]).tobytes())
    table_index, table_spec = next(
        (index, spec)
        for index, spec in enumerate(metadata["tables"])
        if spec["name"] == table
    )
    column_index = next(
        index
        for index, spec in enumerate(table_spec["columns"])
        if spec["name"] == column
    )
    return root["tables"][f"t{table_index:05d}"]["columns"][f"c{column_index:05d}"]


def _round_trip_frame_checkpoint(
    tmp_path: Path, nullable_case: str
) -> BooleanRoundTrip:
    h5py = pytest.importorskip("h5py")
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    frame = _small_frame(source)
    path = tmp_path / "frame-checkpoint.h5"
    write_frame_checkpoint(path, frame)
    loaded = load_frame_checkpoint(path).frame.table("person")
    with h5py.File(path, mode="r") as h5:
        root = h5["_populace_frame_checkpoint"]
        complete = _checkpoint_column_group(
            root, table="person", column=COMPLETE_COLUMN
        )
        missing = _checkpoint_column_group(root, table="person", column=MISSING_COLUMN)
        return BooleanRoundTrip(
            source=source,
            source_before_write=before,
            loaded=loaded,
            stored_complete_values=np.asarray(complete["values"]),
            stored_missing_values=np.asarray(missing["values"]),
            stored_missing_mask=np.asarray(missing["null_mask"], dtype=np.bool_),
            stored_missing_mask_dtype=np.asarray(missing["null_mask"]).dtype,
        )


def _round_trip_nullable_us_h5(tmp_path: Path, nullable_case: str) -> BooleanRoundTrip:
    pytest.importorskip("tables")
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    frame = _us_frame(source)
    path = tmp_path / "nullable-us.h5"
    write_nullable_us_h5(
        frame,
        path,
        period=2024,
        artifact_kind="registry_dtype_family_fixture",
    )
    with pd.HDFStore(path, mode="r") as store:
        loaded = read_frame_table(store, "person")
    return _semantic_observation(source, before, loaded)


def _round_trip_uk_single_year(tmp_path: Path, nullable_case: str) -> BooleanRoundTrip:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    path = tmp_path / "uk-single-year.h5"
    _write_uk_single_year_tables(
        person=source,
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3]}, index=source.index),
        household=pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "household_weight": [1.0, 2.0, 3.0],
            },
            index=source.index,
        ),
        time_period="2023",
        weight_kind=WeightKind.DESIGN,
        mass_log=(),
        path=path,
    )
    loaded = _read_uk_national_tables(path)[0]["person"]
    return _semantic_observation(source, before, loaded)


def _round_trip_axiom(tmp_path: Path, nullable_case: str) -> BooleanRoundTrip:
    pytest.importorskip("tables")
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    path = tmp_path / "axiom.h5"
    AxiomEntityTableDataset(tables={"person": source}, time_period=2025).save(path)
    loaded = AxiomEntityTableDataset(file_path=path).person
    return _semantic_observation(source, before, loaded)


def _round_trip_policyengine_us(tmp_path: Path, nullable_case: str) -> BooleanRoundTrip:
    pytest.importorskip("tables")
    pytest.importorskip("policyengine_us")
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    frame = _us_frame(source)
    tables = {entity: frame.table(entity) for entity in frame.entities}
    path = tmp_path / "policyengine-us.h5"
    PolicyEngineUSEngine()._write_and_verify(tables, period=2024, output_path=path)
    with pd.HDFStore(path, mode="r") as store:
        loaded = read_frame_table(store, "person")
    return _semantic_observation(source, before, loaded)


def _round_trip_legacy_us(tmp_path: Path, nullable_case: str) -> BooleanRoundTrip:
    pytest.importorskip("tables")
    legacy = _load_tool(
        "tools/_legacy/build_us_acs_multispine_base.py",
        "registry_legacy_us_builder",
    )
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    path = tmp_path / "legacy-us.h5"
    legacy._write_dataset(_us_frame(source), path, period=2024)
    with pd.HDFStore(path, mode="r") as store:
        loaded = read_frame_table(store, "person")
    return _semantic_observation(source, before, loaded)


def _round_trip_acs_lean(tmp_path: Path, nullable_case: str) -> BooleanRoundTrip:
    pytest.importorskip("tables")
    tool = _load_tool(
        "tools/build_us_acs_local_release.py",
        "registry_acs_local_release",
    )
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    ids = np.asarray([1, 2, 3], dtype=np.int64)
    person = source.copy(deep=False)
    for position, entity in enumerate(US_SCHEMA.group_entities, start=1):
        person.insert(position, US_SCHEMA.membership_column(entity), ids)
    struct = {
        "household_struct": pd.DataFrame({"household_id": ids}),
        "person": person,
        "groups": {
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
            for entity in tool.GROUP_IDS
        },
        "weights": np.asarray([1.0, 2.0, 3.0]),
    }
    path, _targets = tool.write_lean_checkpoint(
        struct,
        np.empty((3, 0), dtype=np.float64),
        [],
        [],
        [],
        [],
        [],
        tmp_path / "acs-lean",
    )
    loaded = tool.load_lean_frame(path)[0].table("person")
    return _semantic_observation(source, before, loaded)


def _round_trip_fiscal_checkpoint(
    tmp_path: Path, nullable_case: str
) -> BooleanRoundTrip:
    h5py = pytest.importorskip("h5py")
    tool = _load_tool(
        "tools/build_us_fiscal_refresh_release.py",
        "registry_fiscal_refresh",
    )
    source = _dtype_family_table(nullable_case)
    before = source.copy(deep=True)
    frame = _small_frame(source)
    path = tmp_path / "fiscal-target-frame.h5"
    tool._write_target_frame_checkpoint(
        path,
        frame=frame,
        identity={"registry_fixture": True},
        compilation={},
    )
    with h5py.File(path, mode="r") as h5:
        person = h5["tables"]["person"]
        loaded = tool._read_checkpoint_dataframe(person)
        columns = json.loads(str(person.attrs["columns_json"]))
        complete_index = columns.index(COMPLETE_COLUMN)
        missing_index = columns.index(MISSING_COLUMN)
        complete = person["columns"][f"{complete_index:05d}"]
        missing = person["columns"][f"{missing_index:05d}"]
        return BooleanRoundTrip(
            source=source,
            source_before_write=before,
            loaded=loaded,
            stored_complete_values=np.asarray(complete["values"]),
            stored_missing_values=np.asarray(missing["values"]),
            stored_missing_mask=np.asarray(missing["null_mask"], dtype=np.bool_),
            stored_missing_mask_dtype=np.asarray(missing["null_mask"]).dtype,
        )


ROUND_TRIP_ADAPTERS: dict[str, RoundTripAdapter] = {
    "frame_checkpoint": _round_trip_frame_checkpoint,
    "nullable_us_h5": _round_trip_nullable_us_h5,
    "uk_single_year_h5": _round_trip_uk_single_year,
    "axiom_entity_tables": _round_trip_axiom,
    "policyengine_us_single_year": _round_trip_policyengine_us,
    "legacy_us_two_spine": _round_trip_legacy_us,
    "acs_local_lean_checkpoint": _round_trip_acs_lean,
    "fiscal_target_frame_checkpoint": _round_trip_fiscal_checkpoint,
}


def _qualified_call_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    if not isinstance(call.func.value, ast.Name):
        return None
    return f"{call.func.value.id}.{call.func.attr}"


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def _literal_hdf_mode(call: ast.Call) -> str:
    mode_node: ast.AST | None = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
            break
    if mode_node is None:
        return "a"
    value = ast.literal_eval(mode_node)
    if not isinstance(value, str):
        raise AssertionError("Production HDF modes must be literal strings.")
    return value


def _discover_writable_hdf_sites() -> set[str]:
    discovered: set[str] = set()
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _qualified_call_name(node) not in {"h5py.File", "pd.HDFStore"}:
                    continue
                if _literal_hdf_mode(node) == "r":
                    continue
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                function = _enclosing_function(node, parents)
                discovered.add(f"{relative}::{function}")
    return discovered


def test_registry_classifies_every_writable_production_hdf_site() -> None:
    classified = {
        spec.writer.key for spec in FRAME_TABLE_SERIALIZERS if spec.direct_hdf_open
    }
    classified.update(exclusion.writer.key for exclusion in HDF_WRITE_EXCLUSIONS)
    assert _discover_writable_hdf_sites() == classified


def test_registry_has_exactly_eight_unique_frame_table_serializers() -> None:
    assert len(FRAME_TABLE_SERIALIZERS) == 8
    assert len({spec.serializer_id for spec in FRAME_TABLE_SERIALIZERS}) == 8
    assert len({spec.writer.key for spec in FRAME_TABLE_SERIALIZERS}) == 8


def test_round_trip_adapter_registry_exactly_matches_serializer_registry() -> None:
    assert set(ROUND_TRIP_ADAPTERS) == {
        spec.serializer_id for spec in FRAME_TABLE_SERIALIZERS
    }


@pytest.mark.parametrize(
    "serializer",
    FRAME_TABLE_SERIALIZERS,
    ids=lambda serializer: serializer.serializer_id,
)
@pytest.mark.parametrize("nullable_case", ("mixed", "all_missing"))
def test_registered_serializer_round_trips_nullable_boolean_dtype_family(
    serializer: FrameSerializerSpec,
    nullable_case: str,
    tmp_path: Path,
) -> None:
    observation = ROUND_TRIP_ADAPTERS[serializer.serializer_id](tmp_path, nullable_case)

    # Serializers may materialize a boundary copy, never rewrite the source.
    pd.testing.assert_frame_equal(
        observation.source,
        observation.source_before_write,
        check_exact=True,
        check_dtype=True,
    )

    loaded = observation.loaded
    assert loaded[NATIVE_COLUMN].dtype == np.dtype(np.bool_)
    assert loaded[NATIVE_COLUMN].to_numpy(copy=False).tobytes() == (
        observation.source[NATIVE_COLUMN].to_numpy(copy=False).tobytes()
    )

    # Complete BooleanDtype columns have exactly the same physical bool bytes
    # as their logical values; PyTables-facing codecs reload them as native
    # bool, while the explicit h5py codecs retain semantic dtype metadata.
    complete_expected = observation.source[COMPLETE_COLUMN].to_numpy(
        dtype=np.bool_, copy=False
    )
    assert observation.stored_complete_values.dtype == np.dtype(np.bool_)
    assert observation.stored_complete_values.tobytes() == complete_expected.tobytes()

    # Missing booleans are canonical false value bits plus an explicit null
    # representation. The semantic reload must recover both observations and
    # the exact NA positions.
    missing_expected = observation.source[MISSING_COLUMN]
    expected_values = missing_expected.to_numpy(
        dtype=np.bool_, na_value=False, copy=False
    )
    expected_mask = missing_expected.isna().to_numpy(dtype=np.bool_, copy=False)
    assert observation.stored_missing_values.dtype == np.dtype(np.bool_)
    assert observation.stored_missing_values.tobytes() == expected_values.tobytes()
    np.testing.assert_array_equal(observation.stored_missing_mask, expected_mask)
    pd.testing.assert_series_equal(
        loaded[MISSING_COLUMN].astype("boolean"),
        missing_expected,
        check_names=True,
    )

    if serializer.nullable_boolean_storage == "bool_values_optional_uint8_mask":
        assert observation.stored_missing_mask_dtype == np.dtype(np.uint8)
    else:
        assert serializer.nullable_boolean_storage == "numpy_bool_or_object_pd_na_v1"
        assert loaded[COMPLETE_COLUMN].dtype == np.dtype(np.bool_)
        assert loaded[MISSING_COLUMN].dtype == np.dtype(object)
        missing_scalars = loaded.loc[loaded[MISSING_COLUMN].isna(), MISSING_COLUMN]
        assert all(value is pd.NA for value in missing_scalars)


def test_policyengine_us_adapter_owns_its_registered_hdf_boundary() -> None:
    (spec,) = (
        candidate
        for candidate in FRAME_TABLE_SERIALIZERS
        if candidate.serializer_id == "policyengine_us_single_year"
    )
    assert spec.direct_hdf_open is True
    assert spec.writer.key in _discover_writable_hdf_sites()


def test_no_production_dataframe_to_hdf_sink_bypasses_registry() -> None:
    sites: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to_hdf"
                ):
                    sites.append(
                        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}:{node.lineno}"
                    )
    assert sites == []
