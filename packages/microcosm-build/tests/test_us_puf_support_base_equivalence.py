"""Bit-exact equivalence gates for the staged US PUF-support base builder."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.outer_stage_runtime import StageRuntime
from microcosm.build.us_runtime import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.frame import Frame

pytest.importorskip("tables")  # pandas HDF backend for equivalence H5 comparisons

_INPUT_DIR_ENV = "POPULACE_US_PUF_EQUIVALENCE_INPUT_DIR"
_WEEKS_SOURCE_ENV = "POPULACE_US_PUF_EQUIVALENCE_WEEKS_SOURCE"
_MAX_HOUSEHOLDS_ENV = "POPULACE_US_PUF_EQUIVALENCE_MAX_HOUSEHOLDS"
_MAX_PUF_TAX_UNITS_ENV = "POPULACE_US_PUF_EQUIVALENCE_MAX_PUF_TAX_UNITS"
_PRODUCTION_THREAD_STRESS_ENV = "POPULACE_US_PUF_PRODUCTION_THREAD_STRESS"
_DATASET_FILENAME = "base_populace_us_2024_puf_support.h5"
_LOCKED_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "POPULACE_FIT_N_JOBS": "1",
    "POPULACE_FIT_PREDICT_WORKERS": "1",
    "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class _BuilderFixture:
    root: Path
    tool: Path
    common_command: tuple[str, ...]
    builder: ModuleType


@pytest.mark.slow
def test_monolith_and_staged_builds_are_bit_exact_at_every_boundary(
    tmp_path: Path,
) -> None:
    """Compare locked single-thread builds at every lossless cut point."""

    fixture = _prepare_builder_fixture(tmp_path)
    monolith_out = tmp_path / "monolith"
    staged_out = tmp_path / "staged"
    monolith_boundaries = tmp_path / "monolith_boundaries"
    staged_checkpoints = tmp_path / "staged_checkpoints"
    environment = {**os.environ, **_LOCKED_ENVIRONMENT}

    _run_builder(
        [
            *fixture.common_command,
            "--out",
            str(monolith_out),
            "--equivalence-boundary-dir",
            str(monolith_boundaries),
        ],
        cwd=fixture.root,
        environment=environment,
    )
    _run_builder(
        [
            *fixture.common_command,
            "--out",
            str(staged_out),
            "--checkpoint-dir",
            str(staged_checkpoints),
        ],
        cwd=fixture.root,
        environment=environment,
    )

    runtime = StageRuntime(
        staged_checkpoints,
        fixture.builder.OUTER_STAGE_PIPELINE,
    )
    assert runtime.context.completed == fixture.builder.PIPELINE_STEPS
    expected_boundary_names = {
        f"{index:03d}_{stage}.frame.h5"
        for index, stage in enumerate(fixture.builder.PIPELINE_STEPS)
    }
    assert {path.name for path in monolith_boundaries.glob("*.frame.h5")} == (
        expected_boundary_names
    )
    for index, stage in enumerate(fixture.builder.PIPELINE_STEPS):
        monolith_checkpoint = monolith_boundaries / f"{index:03d}_{stage}.frame.h5"
        monolith = load_frame_checkpoint(monolith_checkpoint).frame
        staged = runtime.load(stage).frame
        _assert_frames_logically_identical(
            monolith,
            staged,
            label=f"boundary {index:03d} {stage}",
        )

    _assert_primary_qrf_raw_bits_identical(
        monolith_boundaries / "primary_qrf",
        staged_checkpoints / "primary_qrf",
    )
    monolith_h5 = monolith_out / _DATASET_FILENAME
    staged_h5 = staged_out / _DATASET_FILENAME
    _assert_h5_logically_identical(monolith_h5, staged_h5)
    assert monolith_h5.read_bytes() == staged_h5.read_bytes()
    _assert_successful_stage_profiles(
        staged_checkpoints,
        fixture.builder.PIPELINE_STEPS,
    )


@pytest.mark.slow
def test_staged_build_completes_with_production_threading(tmp_path: Path) -> None:
    """Opt-in stress run: validate production-width execution, not equality."""

    if os.environ.get(_PRODUCTION_THREAD_STRESS_ENV) != "1":
        pytest.skip(f"set {_PRODUCTION_THREAD_STRESS_ENV}=1 to run the stress test")
    fixture = _prepare_builder_fixture(tmp_path)
    output_dir = tmp_path / "production_thread_output"
    checkpoint_dir = tmp_path / "production_thread_checkpoints"
    environment = os.environ.copy()
    for variable in _LOCKED_ENVIRONMENT:
        environment.pop(variable, None)
    # Named stage children require a stable hash seed; numerical workers retain
    # their normal production defaults because every thread-width override is
    # deliberately absent.
    environment["PYTHONHASHSEED"] = "0"

    _run_builder(
        [
            *fixture.common_command,
            "--out",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        cwd=fixture.root,
        environment=environment,
    )

    runtime = StageRuntime(checkpoint_dir, fixture.builder.OUTER_STAGE_PIPELINE)
    assert runtime.context.completed == fixture.builder.PIPELINE_STEPS
    for stage in fixture.builder.PIPELINE_STEPS:
        loaded = runtime.load(stage)
        assert loaded.frame.entities == loaded.frame.schema.entities
        for entity in loaded.frame.entities:
            table = loaded.frame.table(entity)
            assert len(table) > 0
            assert table[loaded.frame.schema.entity_id_column(entity)].is_unique
        for entity in loaded.frame.weighted_entities:
            weights = loaded.frame.weights_for(entity)
            assert len(weights.values) == len(loaded.frame.table(entity))
            assert np.isfinite(weights.values).all()
            assert (weights.values >= 0).all()
            assert (weights.values > 0).any()
    _assert_complete_primary_qrf(checkpoint_dir / "primary_qrf")
    _assert_valid_final_h5(output_dir / _DATASET_FILENAME)
    _assert_successful_stage_profiles(
        checkpoint_dir,
        fixture.builder.PIPELINE_STEPS,
    )


def _prepare_builder_fixture(tmp_path: Path) -> _BuilderFixture:
    """Resolve opt-in artifacts and construct one deterministic smoke command."""

    if importlib.util.find_spec("policyengine_us") is None:
        pytest.skip("requires the policyengine-us [us] extra")
    input_dir_value = os.environ.get(_INPUT_DIR_ENV)
    weeks_source_value = os.environ.get(_WEEKS_SOURCE_ENV)
    if input_dir_value is None or weeks_source_value is None:
        pytest.skip(
            f"set {_INPUT_DIR_ENV} and {_WEEKS_SOURCE_ENV} to run the slow gate"
        )

    input_dir = Path(input_dir_value)
    weeks_source = Path(weeks_source_value)
    inputs = {
        "asec_2024": input_dir / "census_cps_2024.h5",
        "asec_2023": input_dir / "census_cps_2023.h5",
        "asec_2022": input_dir / "census_cps_2022.h5",
        "puf": input_dir / "puf_2024.h5",
        "acs": input_dir / "acs_2022.h5",
        "weeks_source": weeks_source,
    }
    missing = [f"{name}={path}" for name, path in inputs.items() if not path.is_file()]
    assert not missing, "Missing equivalence input(s): " + ", ".join(missing)

    root = Path(__file__).resolve().parents[3]
    tool = root / "tools" / "build_us_puf_support_base.py"
    max_households = int(os.environ.get(_MAX_HOUSEHOLDS_ENV, "5000"))
    max_puf_tax_units = int(os.environ.get(_MAX_PUF_TAX_UNITS_ENV, "10000"))
    assert max_households > 0
    assert max_puf_tax_units > 0
    sampled_asec = _write_representative_asec_samples(
        {
            2024: inputs["asec_2024"],
            2023: inputs["asec_2023"],
            2022: inputs["asec_2022"],
        },
        destination=tmp_path / "asec_samples",
        max_households=max_households,
    )
    sampled_puf = _write_representative_puf_sample(
        inputs["puf"],
        destination=tmp_path / "puf_sample.h5",
        max_tax_units=max_puf_tax_units,
    )
    common = (
        sys.executable,
        str(tool),
        "--asec-h5",
        f"2024={sampled_asec[2024]}",
        "--asec-h5",
        f"2023={sampled_asec[2023]}",
        "--asec-h5",
        f"2022={sampled_asec[2022]}",
        "--asec-max-households",
        str(max_households),
        "--puf-h5",
        str(sampled_puf),
        "--acs-h5",
        str(inputs["acs"]),
        "--asec-2023-weeks-unemployed-source",
        str(inputs["weeks_source"]),
        "--target-year",
        "2024",
        "--seed",
        "0",
        "--n-estimators",
        "32",
        "--equivalence-deterministic-h5-metadata",
        "--without-block-ladder",
        "--stage",
        "all",
    )
    return _BuilderFixture(
        root=root,
        tool=tool,
        common_command=common,
        builder=_load_builder_module(tool),
    )


def _load_builder_module(tool: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_equivalence_build_us_puf_support_base",
        tool,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_frames_logically_identical(
    expected: Frame,
    actual: Frame,
    *,
    label: str,
) -> None:
    """Assert the complete ordered, typed, bit-level Frame contract."""

    assert actual.schema == expected.schema, f"{label}: schema"
    assert actual.entities == expected.entities, f"{label}: entities"
    assert actual.links == expected.links, f"{label}: links"
    assert actual.weighted_entities == expected.weighted_entities, (
        f"{label}: weighted entities"
    )
    for entity in expected.entities:
        _assert_tables_logically_identical(
            expected.table(entity),
            actual.table(entity),
            label=f"{label}: entity {entity}",
        )
    for link in expected.links:
        _assert_tables_logically_identical(
            expected.link(link),
            actual.link(link),
            label=f"{label}: link {link}",
        )
    for entity in expected.weighted_entities:
        expected_weights = expected.weights_for(entity)
        actual_weights = actual.weights_for(entity)
        assert actual_weights.kind is expected_weights.kind, (
            f"{label}: {entity} WeightKind"
        )
        _assert_array_values_identical(
            expected_weights.values,
            actual_weights.values,
            label=f"{label}: {entity} weight bits",
        )
    _assert_series_logically_identical(
        expected.strata,
        actual.strata,
        label=f"{label}: strata",
    )
    assert len(actual.mass_log) == len(expected.mass_log), f"{label}: mass_log length"
    for index, (expected_record, actual_record) in enumerate(
        zip(expected.mass_log, actual.mass_log, strict=True)
    ):
        record_label = f"{label}: mass_log[{index}]"
        assert actual_record.entity == expected_record.entity, record_label
        assert actual_record.reason == expected_record.reason, record_label
        for field in ("old_total", "new_total"):
            _assert_array_values_identical(
                np.asarray([getattr(expected_record, field)], dtype=np.float64),
                np.asarray([getattr(actual_record, field)], dtype=np.float64),
                label=f"{record_label}.{field}",
            )
        assert (actual_record.declared_factor is None) == (
            expected_record.declared_factor is None
        ), f"{record_label}.declared_factor presence"
        if expected_record.declared_factor is not None:
            _assert_array_values_identical(
                np.asarray([expected_record.declared_factor], dtype=np.float64),
                np.asarray([actual_record.declared_factor], dtype=np.float64),
                label=f"{record_label}.declared_factor",
            )


def _assert_tables_logically_identical(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    label: str,
) -> None:
    assert type(actual.index) is type(expected.index), f"{label}: index type"
    pd.testing.assert_index_equal(
        actual.index,
        expected.index,
        exact=True,
        check_names=True,
        check_order=True,
        obj=f"{label} index",
    )
    _assert_index_float_bits(expected.index, actual.index, label=f"{label}: index")
    assert type(actual.columns) is type(expected.columns), f"{label}: column type"
    pd.testing.assert_index_equal(
        actual.columns,
        expected.columns,
        exact=True,
        check_names=True,
        check_order=True,
        obj=f"{label} columns",
    )
    assert actual.dtypes.tolist() == expected.dtypes.tolist(), f"{label}: dtypes"
    for column in expected.columns:
        _assert_series_logically_identical(
            expected[column],
            actual[column],
            label=f"{label}.{column}",
        )


def _assert_series_logically_identical(
    expected: pd.Series,
    actual: pd.Series,
    *,
    label: str,
) -> None:
    assert actual.name == expected.name, f"{label}: name"
    assert actual.dtype == expected.dtype, f"{label}: dtype"
    assert type(actual.index) is type(expected.index), f"{label}: index type"
    pd.testing.assert_index_equal(
        actual.index,
        expected.index,
        exact=True,
        check_names=True,
        check_order=True,
        obj=f"{label} index",
    )
    _assert_index_float_bits(expected.index, actual.index, label=f"{label}: index")
    _assert_array_values_identical(
        expected.to_numpy(copy=False),
        actual.to_numpy(copy=False),
        label=label,
    )


def _assert_index_float_bits(
    expected: pd.Index,
    actual: pd.Index,
    *,
    label: str,
) -> None:
    expected_values = np.asarray(expected)
    if expected_values.dtype.kind == "f":
        _assert_array_values_identical(
            expected_values,
            np.asarray(actual),
            label=f"{label} float bits",
        )


def _assert_array_values_identical(
    expected: Any,
    actual: Any,
    *,
    label: str,
) -> None:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    assert actual_array.shape == expected_array.shape, f"{label}: shape"
    assert actual_array.dtype == expected_array.dtype, f"{label}: dtype"
    if expected_array.dtype.names is not None:
        for field in expected_array.dtype.names:
            _assert_array_values_identical(
                expected_array[field],
                actual_array[field],
                label=f"{label}.{field}",
            )
        return
    if expected_array.dtype.kind == "O":
        expected_tokens = [
            _object_scalar_token(value) for value in expected_array.reshape(-1)
        ]
        actual_tokens = [
            _object_scalar_token(value) for value in actual_array.reshape(-1)
        ]
        assert actual_tokens == expected_tokens, f"{label}: object/null values"
        return
    if expected_array.dtype.kind == "f":
        expected_bits = _float_bits(expected_array)
        actual_bits = _float_bits(actual_array)
        np.testing.assert_array_equal(actual_bits, expected_bits, err_msg=label)
        return
    np.testing.assert_array_equal(actual_array, expected_array, err_msg=label)


def _float_bits(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    unsigned = {
        2: np.dtype("u2"),
        4: np.dtype("u4"),
        8: np.dtype("u8"),
    }.get(contiguous.dtype.itemsize)
    if unsigned is None:
        return contiguous.view(np.uint8).reshape(-1, contiguous.dtype.itemsize)
    return contiguous.reshape(-1).view(unsigned)


def _object_scalar_token(value: Any) -> tuple[Any, ...]:
    """Distinguish every supported object scalar, including null sentinels."""

    if value is None:
        return ("none",)
    if value is pd.NA:
        return ("pd.NA",)
    if value is pd.NaT:
        return ("pd.NaT",)
    if isinstance(value, np.datetime64):
        return ("numpy.datetime64", str(value.dtype), value.view("i8").item())
    if isinstance(value, np.timedelta64):
        return ("numpy.timedelta64", str(value.dtype), value.view("i8").item())
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (float, np.floating)):
        scalar = np.asarray([value])
        return (
            "float",
            str(scalar.dtype),
            bytes(np.ascontiguousarray(scalar).view(np.uint8)),
        )
    if isinstance(value, (int, np.integer)):
        return ("integer", type(value).__name__, int(value))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (bytes, np.bytes_)):
        return ("bytes", bytes(value))
    if isinstance(value, pd.Timestamp):
        return ("pandas.Timestamp", value.value, str(value.tz))
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _assert_primary_qrf_raw_bits_identical(
    monolith_root: Path,
    staged_root: Path,
) -> None:
    target_order = (
        *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )
    assert len(target_order) == 64
    manifest = json.loads((staged_root / "manifest.json").read_text())
    assert tuple(manifest["target_order"]) == target_order
    expected_names = [
        _target_checkpoint_name(index, target)
        for index, target in enumerate(target_order)
    ]
    monolith_target_dir = monolith_root / "targets"
    staged_target_dir = staged_root / "targets"
    assert sorted(path.name for path in monolith_target_dir.glob("*.h5")) == sorted(
        expected_names
    )
    assert sorted(path.name for path in staged_target_dir.glob("*.h5")) == sorted(
        expected_names
    )
    for name in expected_names:
        with (
            h5py.File(monolith_target_dir / name, mode="r") as monolith_h5,
            h5py.File(staged_target_dir / name, mode="r") as staged_h5,
        ):
            assert "raw_draw_bits" in monolith_h5
            assert "raw_draw_bits" in staged_h5
            monolith_bits = np.asarray(monolith_h5["raw_draw_bits"])
            staged_bits = np.asarray(staged_h5["raw_draw_bits"])
        assert monolith_bits.dtype == np.dtype("<u8")
        assert staged_bits.dtype == np.dtype("<u8")
        np.testing.assert_array_equal(staged_bits, monolith_bits, err_msg=name)


def _target_checkpoint_name(index: int, target: str) -> str:
    safe_target = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in target
    )
    return f"{index:03d}__{safe_target}.h5"


def _assert_h5_logically_identical(expected_path: Path, actual_path: Path) -> None:
    assert expected_path.is_file()
    assert actual_path.is_file()
    with (
        h5py.File(expected_path, mode="r") as expected,
        h5py.File(actual_path, mode="r") as actual,
    ):
        expected_objects = _h5_object_kinds(expected)
        actual_objects = _h5_object_kinds(actual)
        assert actual_objects == expected_objects
        _assert_h5_attrs_identical(expected, actual, label="/")
        for path, kind in expected_objects.items():
            expected_object = expected[path]
            actual_object = actual[path]
            _assert_h5_attrs_identical(
                expected_object,
                actual_object,
                label=f"/{path}",
            )
            if kind == "dataset":
                assert isinstance(expected_object, h5py.Dataset)
                assert isinstance(actual_object, h5py.Dataset)
                assert actual_object.shape == expected_object.shape
                assert actual_object.dtype == expected_object.dtype
                _assert_array_values_identical(
                    expected_object[...],
                    actual_object[...],
                    label=f"H5 dataset /{path}",
                )


def _h5_object_kinds(h5: h5py.File) -> dict[str, str]:
    objects: dict[str, str] = {}

    def collect(name: str, value: h5py.Group | h5py.Dataset) -> None:
        objects[name] = "dataset" if isinstance(value, h5py.Dataset) else "group"

    h5.visititems(collect)
    return objects


def _assert_h5_attrs_identical(
    expected: h5py.Group | h5py.Dataset | h5py.File,
    actual: h5py.Group | h5py.Dataset | h5py.File,
    *,
    label: str,
) -> None:
    assert set(actual.attrs) == set(expected.attrs), f"{label}: H5 attrs"
    for name in expected.attrs:
        _assert_array_values_identical(
            expected.attrs[name],
            actual.attrs[name],
            label=f"{label} attr {name}",
        )


def _assert_complete_primary_qrf(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    target_order = tuple(manifest["target_order"])
    assert target_order == (
        *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )
    for index, target in enumerate(target_order):
        path = root / "targets" / _target_checkpoint_name(index, target)
        with h5py.File(path, mode="r") as h5:
            bits = np.asarray(h5["raw_draw_bits"])
        assert bits.dtype == np.dtype("<u8")
        assert bits.ndim == 1
        assert len(bits) == manifest["recipient_rows"]


def _assert_valid_final_h5(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0
    with h5py.File(path, mode="r") as h5:
        objects = _h5_object_kinds(h5)
        datasets = [name for name, kind in objects.items() if kind == "dataset"]
        assert datasets
        for name in datasets:
            dataset = h5[name]
            assert isinstance(dataset, h5py.Dataset)
            assert dataset.ndim == 1
            assert len(dataset) > 0


def _assert_successful_stage_profiles(
    checkpoint_dir: Path,
    outer_stages: tuple[str, ...],
) -> None:
    profile = json.loads((checkpoint_dir / "stage_profile.json").read_text())
    records = profile["stages"]
    target_order = (
        *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )
    assert len(target_order) == 64
    target_stages = tuple(
        f"primary_qrf_{index:03d}_{target}" for index, target in enumerate(target_order)
    )
    expected_stages = (*outer_stages, *target_stages)
    assert set(records) == set(expected_stages)
    for stage in expected_stages:
        record = records[stage]
        assert record["status"] == "succeeded"
        assert record["entry_rss_bytes"] > 0
        assert record["peak_rss_bytes"] >= record["entry_rss_bytes"]
        assert record["peak_rss_bytes"] >= record["exit_rss_bytes"]
        assert record["wall_seconds"] > 0


def _write_representative_asec_samples(
    sources: dict[int, Path],
    *,
    destination: Path,
    max_households: int,
) -> dict[int, Path]:
    """Write representative samples while preserving adjacent-year panels."""

    destination.mkdir(parents=True)
    sampled: dict[int, Path] = {}
    carry_person_ids: set[object] = set()
    for year in sorted(sources, reverse=True):
        source = sources[year]
        with pd.HDFStore(source, mode="r") as store:
            household = store["household"]
            person = store["person"]
        sample_size = min(max_households, len(household))
        rng = np.random.default_rng(year)
        carry_households = set(
            person.loc[person["PERIDNUM"].isin(carry_person_ids), "PH_SEQ"].tolist()
        )
        signal_households: set[object] = set()
        signal_target = max(5, sample_size // 50)
        for column, predicate in (
            ("CSP_VAL", np.greater),
            ("CHSP_VAL", np.greater),
            ("SEMP_VAL", np.less),
        ):
            values = pd.to_numeric(person[column], errors="coerce").to_numpy()
            mask = predicate(values, 0)
            candidates = np.asarray(sorted(set(person.loc[mask, "PH_SEQ"].tolist())))
            selected = candidates
            if len(candidates) > signal_target:
                selected = rng.choice(
                    candidates,
                    size=signal_target,
                    replace=False,
                )
            signal_households.update(selected.tolist())
        retirement_code_target = max(2, sample_size // 1000)
        retirement_code_columns = (
            "DST_SC1",
            "DST_SC2",
            "DST_SC1_YNG",
            "DST_SC2_YNG",
        )
        for account_code in range(1, 7):
            mask = np.zeros(len(person), dtype=bool)
            for column in retirement_code_columns:
                values = pd.to_numeric(person[column], errors="coerce").to_numpy()
                mask |= values == account_code
            candidates = np.asarray(sorted(set(person.loc[mask, "PH_SEQ"].tolist())))
            selected = candidates
            if len(candidates) > retirement_code_target:
                selected = rng.choice(
                    candidates,
                    size=retirement_code_target,
                    replace=False,
                )
            signal_households.update(selected.tolist())
        signal_positions = np.flatnonzero(
            household["H_SEQ"].isin(signal_households).to_numpy()
        )
        assert len(signal_positions) <= sample_size
        carry_positions = np.flatnonzero(
            household["H_SEQ"].isin(carry_households - signal_households).to_numpy()
        )
        carry_capacity = sample_size - len(signal_positions)
        if len(carry_positions) > carry_capacity:
            carry_positions = rng.choice(
                carry_positions,
                size=carry_capacity,
                replace=False,
            )
        required_positions = np.concatenate([signal_positions, carry_positions])
        remaining_positions = np.setdiff1d(
            np.arange(len(household)), required_positions, assume_unique=True
        )
        fill_size = sample_size - len(required_positions)
        fill_positions = rng.choice(remaining_positions, size=fill_size, replace=False)
        positions = np.sort(np.concatenate([required_positions, fill_positions]))
        household = household.iloc[positions].reset_index(drop=True)
        household_ids = set(household["H_SEQ"].tolist())
        person = person[person["PH_SEQ"].isin(household_ids)].reset_index(drop=True)
        carry_person_ids = set(person["PERIDNUM"].tolist())
        if "ED_VAL" not in person:
            # The currently pinned processed ASEC artifacts omit this raw
            # source column even though the production education stage
            # requires it.  Supply a bounded deterministic fixture surface so
            # this refactor gate can reach final export in both modes.
            education_assistance = np.zeros(len(person), dtype=np.float64)
            positive_rows = rng.choice(
                len(person), size=max(1, len(person) // 50), replace=False
            )
            education_assistance[positive_rows] = 1_000.0
            person["ED_VAL"] = education_assistance
        output = destination / f"census_cps_{year}.h5"
        household.to_hdf(output, key="household", mode="w", format="fixed")
        person.to_hdf(output, key="person", mode="a", format="fixed")
        sampled[year] = output
    return sampled


def _write_representative_puf_sample(
    source: Path,
    *,
    destination: Path,
    max_tax_units: int,
) -> Path:
    """Apply one deterministic entity-consistent sample to the PUF arrays."""

    rng = np.random.default_rng(2024)
    with h5py.File(source, mode="r") as input_h5:
        group_ids = np.asarray(input_h5["tax_unit_id"])
        sample_size = min(max_tax_units, len(group_ids))
        person_tax_unit_ids = np.asarray(input_h5["person_tax_unit_id"])
        required_group_ids: set[object] = set()
        signal_columns = (
            set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
            | set(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS)
        ) & set(input_h5)
        for column in sorted(signal_columns):
            values = np.asarray(input_h5[column])
            if len(values) == len(person_tax_unit_ids):
                entity_tax_unit_ids = person_tax_unit_ids
            elif len(values) == len(group_ids):
                entity_tax_unit_ids = group_ids
            else:
                continue
            for mask in (values > 0, values < 0):
                signal_ids = np.unique(entity_tax_unit_ids[mask])
                signal_target = min(
                    len(signal_ids),
                    max(10, sample_size // 100),
                )
                if len(signal_ids) > signal_target:
                    signal_ids = rng.choice(
                        signal_ids, size=signal_target, replace=False
                    )
                required_group_ids.update(signal_ids.tolist())
        # These gated outputs are sparse (and the QBI pair is jointly
        # constrained during reconciliation).  Ten generic tail examples are
        # not enough for a small QRF donor pool to draw them reliably, so
        # reserve deterministic positive strata (ten percent for the joint
        # SSTB signal and four percent for the other targets).
        for column, denominator in (
            ("qualified_bdc_income", 25),
            ("qualified_tuition_expenses", 25),
            ("sstb_w2_wages_from_qualified_business", 10),
            ("unreimbursed_business_employee_expenses", 25),
        ):
            values = np.asarray(input_h5[column])
            assert len(values) == len(person_tax_unit_ids)
            signal_ids = np.unique(person_tax_unit_ids[values > 0])
            signal_target = min(
                len(signal_ids),
                max(10, sample_size // denominator),
            )
            if len(signal_ids) > signal_target:
                signal_ids = rng.choice(
                    signal_ids,
                    size=signal_target,
                    replace=False,
                )
            required_group_ids.update(signal_ids.tolist())
        required_positions = np.flatnonzero(
            np.fromiter(
                (value in required_group_ids for value in group_ids),
                dtype=bool,
                count=len(group_ids),
            )
        )
        if len(required_positions) > sample_size:
            required_positions = rng.choice(
                required_positions,
                size=sample_size,
                replace=False,
            )
        remaining_positions = np.setdiff1d(
            np.arange(len(group_ids)), required_positions, assume_unique=True
        )
        fill_positions = rng.choice(
            remaining_positions,
            size=sample_size - len(required_positions),
            replace=False,
        )
        group_positions = np.sort(np.concatenate([required_positions, fill_positions]))
        selected_group_ids = set(group_ids[group_positions].tolist())
        person_mask = np.fromiter(
            (value in selected_group_ids for value in person_tax_unit_ids),
            dtype=bool,
            count=len(person_tax_unit_ids),
        )
        selected_marital_ids = set(
            np.asarray(input_h5["person_marital_unit_id"])[person_mask].tolist()
        )
        marital_ids = np.asarray(input_h5["marital_unit_id"])
        marital_mask = np.fromiter(
            (value in selected_marital_ids for value in marital_ids),
            dtype=bool,
            count=len(marital_ids),
        )
        lengths_to_selection = {
            len(group_ids): group_positions,
            len(person_tax_unit_ids): person_mask,
            len(marital_ids): marital_mask,
        }
        with h5py.File(destination, mode="w") as output_h5:
            for name, value in input_h5.attrs.items():
                output_h5.attrs[name] = value
            for name, dataset in input_h5.items():
                assert len(dataset.shape) == 1
                selection = lengths_to_selection.get(len(dataset))
                assert selection is not None, (
                    f"Unexpected PUF entity length for {name}: {len(dataset)}"
                )
                output_h5.create_dataset(
                    name,
                    data=np.asarray(dataset)[selection],
                    track_times=False,
                )
    return destination


def _run_builder(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Builder failed with exit {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
