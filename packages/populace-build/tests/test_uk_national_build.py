from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from populace.build.gates import GateResult
from populace.build.uk_runtime.national_build import (
    UKNationalDataset,
    UKNationalStage,
    build_uk_national_dataset,
    load_uk_national_dataset,
)
from populace.frame import MassChangeRecord, WeightKind


@pytest.fixture(autouse=True)
def _isolate_generic_seam_from_shipped_family_contract(monkeypatch) -> None:
    """These seam tests use toy stages; family enforcement has its own tests."""

    from populace.build.uk_runtime import national_build

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_build_stages",
        lambda _stage_names: None,
    )


def _write_toy_h5(path: Path, *, employment_income: float = 0.0) -> None:
    with pd.HDFStore(path) as store:
        store.put(
            "person",
            pd.DataFrame(
                {
                    "person_id": [10],
                    "person_household_id": [1],
                    "person_benunit_id": [100],
                    "employment_income": [employment_income],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "benunit",
            pd.DataFrame({"benunit_id": [100]}),
            format="table",
            data_columns=True,
        )
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": [1],
                    "household_weight": [2.0],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format="table",
            data_columns=True,
        )


def _passing_gate() -> GateResult:
    return GateResult(
        name="uk_release_input_coverage",
        passed=True,
        failures=(),
        details={"required_columns": 1, "missing": [], "degenerate": []},
    )


def _failing_gate() -> GateResult:
    return GateResult(
        name="uk_release_input_coverage",
        passed=False,
        failures=("required column employment_income is default-only",),
        details={
            "required_columns": 1,
            "missing": [],
            "degenerate": ["employment_income"],
        },
    )


def test_national_build_runs_preflight_stages_gate_then_staging_write(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    coverage_json = tmp_path / "input_coverage.json"
    _write_toy_h5(input_h5)
    events: list[str] = []

    def stage_transform(dataset: UKNationalDataset) -> UKNationalDataset:
        events.append("stage:income")
        person = dataset.person.copy()
        person["employment_income"] = 50_000.0
        return dataset.with_tables(person=person)

    def assert_current(**_kwargs) -> None:
        events.append("manifest_preflight")

    def coverage_gate(dataset, _engine):
        events.append("final_coverage_gate")
        assert dataset.person["employment_income"].tolist() == [50_000.0]
        return _passing_gate()

    real_writer = national_build.write_uk_national_dataset

    def recording_writer(dataset, path):
        events.append("staging_write")
        return real_writer(dataset, path)

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        assert_current,
    )
    monkeypatch.setattr(
        national_build,
        "uk_release_input_coverage_gate",
        coverage_gate,
    )
    monkeypatch.setattr(
        national_build,
        "write_uk_national_dataset",
        recording_writer,
    )

    result = build_uk_national_dataset(
        input_h5=input_h5,
        staging_h5=staging_h5,
        stages=(UKNationalStage("income", stage_transform),),
        coverage_engine=object(),
        input_coverage_path=coverage_json,
    )

    assert events == [
        "manifest_preflight",
        "stage:income",
        "final_coverage_gate",
        "staging_write",
    ]
    assert result.stage_names == ("income",)
    assert result.input_coverage.passed is True
    assert staging_h5.exists()
    staged = load_uk_national_dataset(staging_h5)
    assert staged.person["employment_income"].tolist() == [50_000.0]
    assert staged.household["household_weight"].tolist() == [2.0]
    diagnostic = json.loads(coverage_json.read_text())
    assert diagnostic["enforced"] is True
    assert diagnostic["input_coverage"]["passed"] is True


def test_national_build_gate_failure_writes_diagnostic_not_h5(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    coverage_json = tmp_path / "input_coverage.json"
    _write_toy_h5(input_h5)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        national_build,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _failing_gate(),
    )

    with pytest.raises(RuntimeError, match="Input coverage failed"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            input_coverage_path=coverage_json,
        )

    assert not staging_h5.exists()
    diagnostic = json.loads(coverage_json.read_text())
    assert diagnostic["input_coverage"]["passed"] is False
    assert diagnostic["input_coverage"]["details"]["degenerate"] == [
        "employment_income"
    ]


def test_national_build_rejects_duplicate_stage_names_before_running(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    called = False

    def transform(dataset: UKNationalDataset) -> UKNationalDataset:
        nonlocal called
        called = True
        return dataset

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match="Duplicate UK national stage"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            stages=(
                UKNationalStage("income", transform),
                UKNationalStage("income", transform),
            ),
            coverage_engine=object(),
        )

    assert called is False


def test_national_build_manifest_failure_removes_stale_outputs_before_stages(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    coverage_json = tmp_path / "input_coverage.json"
    _write_toy_h5(input_h5)
    staging_h5.write_bytes(b"stale-success")
    coverage_json.write_text('{"stale_success": true}\n')
    stage_called = False

    def stage_transform(dataset: UKNationalDataset) -> UKNationalDataset:
        nonlocal stage_called
        stage_called = True
        return dataset

    def reject_manifest(**_kwargs) -> None:
        raise ValueError("manifest drift")

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        reject_manifest,
    )

    with pytest.raises(ValueError, match="manifest drift"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=staging_h5,
            stages=(UKNationalStage("should_not_run", stage_transform),),
            coverage_engine=object(),
            input_coverage_path=coverage_json,
        )

    assert stage_called is False
    assert not staging_h5.exists()
    assert not coverage_json.exists()


def test_national_build_rejects_stage_that_breaks_entity_links(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )

    def break_links(dataset: UKNationalDataset) -> UKNationalDataset:
        person = dataset.person.copy()
        person["person_household_id"] = 999
        return dataset.with_tables(person=person)

    with pytest.raises(ValueError, match="absent from household"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            stages=(UKNationalStage("bad", break_links),),
            coverage_engine=object(),
        )


@pytest.mark.parametrize(
    ("stage_name", "transform", "message"),
    [
        (
            "missing_period",
            lambda dataset: UKNationalDataset(
                person=dataset.person,
                benunit=dataset.benunit,
                household=dataset.household,
                time_period=None,
            ),
            "time_period must be a non-empty string",
        ),
        (
            "zero_population",
            lambda dataset: dataset.with_tables(
                household=dataset.household.assign(household_weight=0.0)
            ),
            "retain at least one positive value",
        ),
    ],
)
def test_national_build_rejects_invalid_stage_population_metadata(
    monkeypatch, tmp_path, stage_name, transform, message
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match=message):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            stages=(UKNationalStage(stage_name, transform),),
            coverage_engine=object(),
        )


def test_national_build_refuses_to_overwrite_its_input(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match="must differ"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=input_h5,
            coverage_engine=object(),
        )


def test_national_staging_h5_loads_through_policyengine_uk(tmp_path) -> None:
    pytest.importorskip("tables")
    policyengine_data = pytest.importorskip("policyengine_uk.data")
    from populace.build.uk_runtime.national_build import write_uk_national_dataset

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    dataset = load_uk_national_dataset(input_h5)
    dataset = dataset.with_tables(
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=2.0,
                new_total=2.0,
                declared_factor=1.0,
                reason="test reviewed support-channel mass allocation",
            ),
        ),
    )

    write_uk_national_dataset(dataset, staging_h5)

    round_tripped = load_uk_national_dataset(staging_h5)
    assert round_tripped.household_weight_kind is WeightKind.IMPORTANCE
    assert round_tripped.mass_log == dataset.mass_log

    loaded = policyengine_data.UKSingleYearDataset(file_path=str(staging_h5))
    assert loaded.time_period == "2023"
    assert loaded.person["employment_income"].tolist() == [40_000.0]
    assert loaded.household["household_weight"].tolist() == [2.0]


def test_atomic_writer_cleans_temporary_h5_after_write_failure(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    dataset = load_uk_national_dataset(input_h5)
    staging_h5.write_bytes(b"previous-good-artifact")

    def fail_store(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated HDF write failure")

    monkeypatch.setattr(national_build.pd, "HDFStore", fail_store)

    with pytest.raises(OSError, match="simulated HDF write failure"):
        national_build.write_uk_national_dataset(dataset, staging_h5)

    assert staging_h5.read_bytes() == b"previous-good-artifact"
    assert list(tmp_path.glob(".staging.h5.*.tmp.h5")) == []
