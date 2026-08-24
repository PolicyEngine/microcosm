"""Compatibility tests for the retired ACS builder shim and shared US H5 I/O."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import h5_io
from microcosm.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    load_legacy_calibrated_us_h5,
    write_nullable_us_h5,
)
from microcosm.frame import (
    US_SCHEMA,
    Frame,
    WeightKind,
    Weights,
    put_frame_table,
)


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
    h5py = pytest.importorskip("h5py")
    frame = _frame()
    output = tmp_path / "pool.h5"
    root_attributes = {
        "populace_congressional_district_vintage_crosswalk_sha256": "a" * 64,
        "populace_congressional_district_vintage_target": "119th_congress",
    }

    write_nullable_us_h5(
        frame,
        output,
        period=2024,
        artifact_kind="nullable_multispine_pool_h5",
        root_attributes=root_attributes,
    )

    with pd.HDFStore(output, mode="r") as store:
        assert store.get_storer("person").is_table is False
        assert store["person"]["is_snap_abawd_discretionary_exempt"].tolist()[0] is True
        assert pd.isna(store["person"]["is_snap_abawd_discretionary_exempt"].iloc[1])
        assert store["household"]["household_weight"].tolist() == [40.0, 60.0]
        assert store["_time_period"].tolist() == [2024]
        stored_root_attributes = store.get_node("/")._v_attrs
        assert {
            key: str(stored_root_attributes[key]) for key in root_attributes
        } == root_attributes
    with h5py.File(output, mode="r") as h5:
        assert {
            key: h5.attrs[key].decode() for key in root_attributes
        } == root_attributes
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
    original_write = h5_io._write_nullable_us_h5_file

    def omit_root_attributes(*args, **kwargs):
        kwargs["root_attributes"] = ()
        original_write(*args, **kwargs)

    monkeypatch.setattr(
        h5_io,
        "_write_nullable_us_h5_file",
        omit_root_attributes,
    )

    with pytest.raises(RuntimeError, match="omitted root attribute"):
        write_nullable_us_h5(
            _frame(),
            output,
            period=2024,
            artifact_kind="fixture",
            root_attributes={"fixture_provenance": "verified-before-replace"},
        )

    assert output.read_bytes() == b"existing-artifact"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("root_attributes", "expected_error"),
    [
        ([("fixture", "value")], TypeError),
        ({1: "value"}, TypeError),
        ({"bad-name": "value"}, ValueError),
        ({"CLASS": "value"}, ValueError),
        ({"fixture": 1}, TypeError),
        ({"fixture": ""}, ValueError),
    ],
)
def test_nullable_writer_rejects_invalid_root_attributes_before_writing(
    root_attributes,
    expected_error,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    output = tmp_path / "invalid-root-attrs.h5"

    with pytest.raises(expected_error):
        write_nullable_us_h5(
            _frame(),
            output,
            period=2024,
            artifact_kind="fixture",
            root_attributes=root_attributes,
        )

    assert not output.exists()


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
    tmp_path: Path,
) -> None:
    # The loader reads legacy artifacts directly via HDFStore — no
    # policyengine_us dependency remains — so the contract is exercised
    # against a real legacy-shaped file.
    pytest.importorskip("tables")
    source = _frame(weight_kind=WeightKind.DESIGN)
    path = tmp_path / "legacy.h5"
    with pd.HDFStore(path, mode="w") as store:
        for entity in source.entities:
            table = source.table(entity).copy()
            if entity == "household":
                table["household_weight"] = [7.0, 11.0]
            put_frame_table(store, entity, table, preferred_format="fixed")

    direct = load_legacy_calibrated_us_h5(path)
    via_shim = _load_shim_module()._load_base_frame(path)

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
