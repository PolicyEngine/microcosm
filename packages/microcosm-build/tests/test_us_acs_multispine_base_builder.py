"""Compatibility tests for the retired ACS builder shim and shared US H5 I/O."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import h5_io
from microcosm.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    load_legacy_calibrated_us_h5,
    write_nullable_us_h5,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _shim_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "tools"
        / ("build_us_acs_multispine_base.py")
    )


def _load_shim_module():
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_multispine_base",
        _shim_path(),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame(*, weight_kind: WeightKind = WeightKind.IMPORTANCE) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 2],
            "person_tax_unit_id": [1, 2],
            "person_spm_unit_id": [1, 2],
            "person_family_id": [1, 2],
            "person_marital_unit_id": [1, 2],
            "age": [35, 67],
            "is_snap_abawd_discretionary_exempt": pd.Series(
                [True, np.nan],
                dtype=object,
            ),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": [6, 36],
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2]}),
        "family": pd.DataFrame({"family_id": [1, 2]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([40.0, 60.0]),
                weight_kind,
            )
        },
    )


def _stored_metadata(path: Path) -> dict[str, object]:
    with pd.HDFStore(path, mode="r") as store:
        return json.loads(store["_populace_staging_metadata"].iloc[0])


def test_nullable_writer_round_trips_fixed_tables_and_caller_artifact_kind(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _frame()
    output = tmp_path / "pool.h5"

    write_nullable_us_h5(
        frame,
        output,
        period=2024,
        artifact_kind="nullable_multispine_pool_h5",
    )

    with pd.HDFStore(output, mode="r") as store:
        assert store.get_storer("person").is_table is False
        assert store["person"]["is_snap_abawd_discretionary_exempt"].tolist()[0] is True
        assert pd.isna(store["person"]["is_snap_abawd_discretionary_exempt"].iloc[1])
        assert store["household"]["household_weight"].tolist() == [40.0, 60.0]
        assert store["_time_period"].tolist() == [2024]
    assert _stored_metadata(output) == {
        "artifact_kind": "nullable_multispine_pool_h5",
        "entity_hdf_format": "fixed_nullable",
        "household_weight_kind": "importance",
    }


def test_nullable_writer_keeps_existing_bytes_when_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    output = tmp_path / "existing.h5"
    output.write_bytes(b"existing-artifact")

    def fail_verification(*_args, **_kwargs):
        raise RuntimeError("injected verification failure")

    monkeypatch.setattr(h5_io, "_verify_nullable_us_h5", fail_verification)

    with pytest.raises(RuntimeError, match="injected verification failure"):
        write_nullable_us_h5(
            _frame(),
            output,
            period=2024,
            artifact_kind="fixture",
        )

    assert output.read_bytes() == b"existing-artifact"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_shim_preserves_legacy_write_signature_and_default(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    shim = _load_shim_module()
    signature = inspect.signature(shim._write_dataset)

    assert list(signature.parameters) == [
        "frame",
        "path",
        "period",
        "artifact_kind",
    ]
    assert signature.parameters["period"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["artifact_kind"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["artifact_kind"].default == (
        LEGACY_NULLABLE_STAGING_ARTIFACT_KIND
    )

    output = tmp_path / "legacy-staging.h5"
    shim._write_dataset(_frame(), output, period=2024)

    assert _stored_metadata(output)["artifact_kind"] == (
        LEGACY_NULLABLE_STAGING_ARTIFACT_KIND
    )


def test_legacy_loader_and_shim_keep_calibrated_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _frame(weight_kind=WeightKind.DESIGN)
    captured: dict[str, str] = {}

    class FakeDataset:
        def __init__(self, *, file_path: str) -> None:
            captured["file_path"] = file_path
            for entity in source.entities:
                table = source.table(entity).copy()
                if entity == "household":
                    table["household_weight"] = [7.0, 11.0]
                setattr(self, entity, table)

    package = ModuleType("policyengine_us")
    package.__path__ = []  # type: ignore[attr-defined]
    data = ModuleType("policyengine_us.data")
    data.USSingleYearDataset = FakeDataset  # type: ignore[attr-defined]
    package.data = data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "policyengine_us", package)
    monkeypatch.setitem(sys.modules, "policyengine_us.data", data)

    path = Path("legacy.h5")
    direct = load_legacy_calibrated_us_h5(path)
    via_shim = _load_shim_module()._load_base_frame(path)

    assert captured["file_path"] == str(path)
    for loaded in (direct, via_shim):
        assert loaded.weights_for("household").kind is WeightKind.CALIBRATED
        assert loaded.weights_for("household").values.tolist() == [7.0, 11.0]
        assert "household_weight" not in loaded.table("household")


def test_deprecated_cli_keeps_legacy_staging_recipe_available() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_shim_path()),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--base-h5 BASE_H5" in result.stdout
    assert "--out-h5 OUT_H5" in result.stdout
    assert "--donor-release-manifest DONOR_RELEASE_MANIFEST" in result.stdout
    assert "microcosm#578 increment 4" in result.stderr
    assert "tools/build_us_multispine_pool.py" in result.stderr


def test_deprecated_shim_warns_and_delegates_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = _load_shim_module()
    argv = ["--base-h5", "dense.h5", "--out-h5", "staging.h5"]
    captured: list[list[str] | None] = []

    def fake_main(actual_argv: list[str] | None = None) -> int:
        captured.append(actual_argv)
        return 0

    monkeypatch.setattr(shim._legacy, "main", fake_main)

    with pytest.warns(DeprecationWarning, match=r"microcosm#578 increment 4"):
        assert shim.main(argv) == 0

    assert captured == [argv]
