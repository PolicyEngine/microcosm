"""Small-fixture checks for annual candidate serialization and evidence safety."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build import us_annual_static_aging as annual
from microcosm.frame import US_SCHEMA, SignedScale, WeightKind, Weights
from microcosm.frame.materialize import put_frame_table, read_frame_table

pytestmark = pytest.mark.requires_us


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): [1, 2, 3]})
        for entity in US_SCHEMA.entities
    }
    for group in US_SCHEMA.group_entities:
        tables["person"][US_SCHEMA.membership_column(group)] = [1, 2, 3]
    tables["person"]["age"] = [40, 40, 40]
    tables["person"]["is_female"] = [False, True, False]
    tables["person"]["employment_income_before_lsr"] = [100.0, 200.0, 300.0]
    tables["person"]["miscellaneous_income"] = [10.0, -5.0, 0.0]
    tables["person"]["input_flag"] = pd.Series([True, False, True], dtype="boolean")
    tables["household"]["household_weight"] = [1.0, 2.0, 3.0]
    tables["household"]["state_fips"] = [6, 6, 6]
    base = tmp_path / "base.h5"
    with pd.HDFStore(base, "w") as store:
        for entity, table in tables.items():
            put_frame_table(
                store, entity, table, preferred_format="table", data_columns=True
            )
        store.put("_time_period", pd.Series([2024]), format="table")
    ssa = tmp_path / "ssa.csv"
    pd.DataFrame(
        {
            "Year": [2024, 2025, 2026],
            "Age": [40] * 3,
            "M Tot": [4, 5, 6],
            "F Tot": [2, 3, 4],
        }
    ).to_csv(ssa, index=False)
    monkeypatch.setattr(
        annual,
        "_model_provenance",
        lambda **kwargs: {"version": "test", "commit": "a" * 40},
    )
    return dict(
        base_h5=base,
        base_sha256=annual._sha256(base),
        parent_release="pinned-parent",
        ssa_csv=ssa,
        ssa_sha256=annual._sha256(ssa),
        model_source=tmp_path,
        model_commit="a" * 40,
        model_version="test",
        output_dir=tmp_path / "candidate",
        base_year=2024,
        end_year=2026,
        epochs=2,
    )


def test_annual_files_preserve_base_shape_values_and_year(inputs, monkeypatch):
    calls = []

    def project(frame, **kwargs):
        (year,) = kwargs["years"]
        calls.append(year)
        projection = SimpleNamespace(
            weights=Weights(np.array([2.0, 3.0, 4.0]), WeightKind.CALIBRATED),
            factors={
                "employment_income_before_lsr": 2.0,
                "miscellaneous_income": SignedScale(3.0, 4.0),
            },
            demographic_fit=pd.DataFrame({"target": [1.0], "achieved": [1.0]}),
            fraction_within_10pct=1.0,
        )
        return SimpleNamespace(year=lambda requested: projection)

    monkeypatch.setattr(annual, "static_aging", project)
    manifest = annual.build_annual_static_aging(**inputs)
    assert calls == [2025, 2026]  # Every projection starts from the original frame.
    output = inputs["output_dir"]
    assert (output / "populace_us_2024.h5").read_bytes() == inputs[
        "base_h5"
    ].read_bytes()
    assert manifest["metadata"]["dataset_years"] == {
        "populace_us_2024": {str(y): f"populace_us_{y}" for y in (2024, 2025, 2026)}
    }
    for year in (2025, 2026):
        path = output / f"populace_us_{year}.h5"
        import h5py

        with (
            h5py.File(inputs["base_h5"], "r") as base,
            h5py.File(path, "r") as projected,
        ):
            for entity in US_SCHEMA.entities:
                assert (
                    projected[f"{entity}/table"].dtype.names
                    == base[f"{entity}/table"].dtype.names
                )
        with pd.HDFStore(path, "r") as store:
            assert store["_time_period"].iloc[0] == year
            person = read_frame_table(store, "person")
            assert person["employment_income_before_lsr"].tolist() == [200, 400, 600]
            assert person["miscellaneous_income"].tolist() == [30, -20, 0]
            assert person["person_id"].tolist() == [1, 2, 3]
            assert person["age"].tolist() == [40, 40, 40]
        artifact = manifest["artifacts"][f"populace_us_{year}"]
        assert artifact["sha256"] == annual._sha256(path)
        assert artifact["rows"] == {entity: 3 for entity in US_SCHEMA.entities}
        assert artifact["round_trip"]["time_period"] == year
        from policyengine_us.data import USSingleYearDataset
        from policyengine_us.data.economic_assumptions import extend_single_year_dataset

        reloaded = USSingleYearDataset(file_path=str(path))
        extended = extend_single_year_dataset(reloaded)
        # Native model loading may extend other years, but the selected annual
        # inputs are already projected and must not receive a second factor.
        assert extended.datasets[year].person[
            "employment_income_before_lsr"
        ].tolist() == [200, 400, 600]
        assert extended.datasets[year].household["household_weight"].tolist() == [
            2,
            3,
            4,
        ]


def test_small_actual_calibration_exports_independent_years(inputs):
    manifest = annual.build_annual_static_aging(**inputs)
    assert manifest["status"] == "complete"
    receipt = json.loads((inputs["output_dir"] / "projection_2025.json").read_text())
    assert receipt["demographic_fit"]
    assert receipt["column_series"]["employment_income_before_lsr"] in receipt["totals"]


def test_base_h5_symlink_to_extensionless_cache_blob(inputs):
    base = inputs["base_h5"]
    blob = base.with_name("cached-blob-without-extension")
    base.rename(blob)
    base.symlink_to(blob)
    manifest = annual.build_annual_static_aging(**inputs)
    assert manifest["base"]["path"] == str(base)
    assert (
        inputs["output_dir"] / "populace_us_2024.h5"
    ).read_bytes() == blob.read_bytes()


def test_existing_output_is_rejected_before_work(inputs, monkeypatch):
    output = inputs["output_dir"]
    output.mkdir()
    evidence = output / "prior.json"
    evidence.write_text("untouched")
    monkeypatch.setattr(
        annual, "_model_provenance", lambda **kwargs: pytest.fail("must reject first")
    )
    with pytest.raises(FileExistsError):
        annual.build_annual_static_aging(**inputs)
    assert evidence.read_text() == "untouched"


@pytest.mark.parametrize("corruption", ["hash", "year", "missing_year", "shape"])
def test_invalid_base_is_rejected_without_completed_manifest(inputs, corruption):
    if corruption == "hash":
        inputs["base_sha256"] = "0" * 64
    else:
        with pd.HDFStore(inputs["base_h5"], "a") as store:
            if corruption == "year":
                store.put("_time_period", pd.Series([2023]))
            elif corruption == "missing_year":
                del store["_time_period"]
            else:
                del store["family"]
        inputs["base_sha256"] = annual._sha256(inputs["base_h5"])
    with pytest.raises((ValueError, KeyError)):
        annual.build_annual_static_aging(**inputs)
    assert not (inputs["output_dir"] / "annual_manifest.json").exists()


def test_failed_projection_retains_evidence_and_cannot_be_retried_in_place(
    inputs, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture projection failure")

    monkeypatch.setattr(annual, "static_aging", fail)
    with pytest.raises(RuntimeError, match="fixture projection failure"):
        annual.build_annual_static_aging(**inputs)
    output = inputs["output_dir"]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert json.loads(before["build_status.json"])["status"] == "failed"
    assert "annual_manifest.json" not in before
    with pytest.raises(FileExistsError):
        annual.build_annual_static_aging(**inputs)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_source_drift_prevents_completion(inputs, monkeypatch):
    snapshots = iter([{"sha": "before"}, {"sha": "after"}])
    monkeypatch.setattr(annual, "_runtime_provenance", lambda: next(snapshots))
    with pytest.raises(ValueError, match="source changed"):
        annual.build_annual_static_aging(**inputs)
    output = inputs["output_dir"]
    assert not (output / "annual_manifest.json").exists()
    assert json.loads((output / "build_status.json").read_text())["status"] == "failed"


def test_fixed_storage_base_requires_a_separate_layout_contract(inputs):
    with pd.HDFStore(inputs["base_h5"], "a") as store:
        person = read_frame_table(store, "person")
        del store["person"]
        person["input_flag"] = pd.Series([True, None, False], dtype="boolean")
        put_frame_table(
            store, "person", person, preferred_format="table", data_columns=True
        )
    inputs["base_sha256"] = annual._sha256(inputs["base_h5"])
    with pytest.raises(ValueError, match="table-format storage"):
        annual.build_annual_static_aging(**inputs)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("packed", ["person", "_time_period"])
def test_non_native_table_storage_rejected_before_build(inputs, packed):
    with pd.HDFStore(inputs["base_h5"], "a") as store:
        value = store[packed]
        del store[packed]
        store.put(
            packed, value, format="fixed" if packed == "_time_period" else "table"
        )
    inputs["base_sha256"] = annual._sha256(inputs["base_h5"])
    with pytest.raises(ValueError, match="direct HDF fields|table-format _time_period"):
        annual.build_annual_static_aging(**inputs)
    assert not inputs["output_dir"].exists()


def test_round_trip_rejects_changed_values_and_wrong_year(inputs):
    with pd.HDFStore(inputs["base_h5"], "r") as store:
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
    tables["person"].loc[0, "employment_income_before_lsr"] += 1
    with pytest.raises(AssertionError):
        annual._verify(inputs["base_h5"], tables, 2024)
    with pytest.raises(ValueError, match="year|time_period"):
        annual._verify(inputs["base_h5"], tables, 2025)


def test_model_provenance_requires_imported_clean_pinned_source(tmp_path, monkeypatch):
    package = tmp_path / "policyengine_us"
    package.mkdir()
    source_file = package / "__init__.py"
    source_file.write_text("# fixture model\n")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git("add", "policyengine_us")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    commit = git("rev-parse", "HEAD")
    monkeypatch.setitem(
        sys.modules, "policyengine_us", SimpleNamespace(__file__=str(source_file))
    )
    monkeypatch.setattr(annual.importlib.metadata, "version", lambda name: "test")
    kwargs = dict(source=tmp_path, commit=commit, version="test")
    receipt = annual._model_provenance(**kwargs)
    assert receipt["commit"] == commit
    assert receipt["source_files"] == {
        "policyengine_us/__init__.py": annual._sha256(source_file)
    }
    with pytest.raises(ValueError, match="commit mismatch"):
        annual._model_provenance(**{**kwargs, "commit": "0" * 40})
    with pytest.raises(ValueError, match="version mismatch"):
        annual._model_provenance(**{**kwargs, "version": "other"})
    source_file.write_text("# edited\n")
    with pytest.raises(ValueError, match="clean, committed"):
        annual._model_provenance(**kwargs)
