from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from populace.build.gates import FitWeightRecord, GateReport, GateResult
from populace.build.uk_runtime.national_build import (
    UKNationalDataset,
    UKNationalStage,
    build_uk_national_dataset,
    load_uk_national_dataset,
)
from populace.build.uk_runtime.terminal_gates import (
    UKReleaseParityEvidence,
)
from populace.build.uk_runtime.terminal_gates import (
    uk_terminal_gate_report as real_uk_terminal_gate_report,
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
    monkeypatch.setattr(
        national_build,
        "uk_terminal_gate_report",
        lambda dataset, engine, **_kwargs: GateReport(
            (national_build.uk_release_input_coverage_gate(dataset, engine),)
        ),
    )

    def write_generic_seam_report(report, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {"schema_version": 2, "enforced": True, **report.to_manifest()},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return output

    monkeypatch.setattr(
        national_build,
        "write_uk_terminal_gate_report",
        write_generic_seam_report,
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


def _write_two_row_h5(
    path: Path,
    *,
    employment_income: tuple[float, float] = (40_000.0, 55_000.0),
) -> None:
    with pd.HDFStore(path) as store:
        store.put(
            "person",
            pd.DataFrame(
                {
                    "person_id": [10, 20],
                    "person_household_id": [1, 2],
                    "person_benunit_id": [100, 200],
                    "employment_income": employment_income,
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "benunit",
            pd.DataFrame({"benunit_id": [100, 200]}),
            format="table",
            data_columns=True,
        )
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "household_weight": [1.0, 2.0],
                    "household_is_spi_synthetic": [False, True],
                    "household_is_capital_gains_clone": [False, True],
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


class _RecordedFitStage:
    fit_weight_records = (
        FitWeightRecord("uk_spi_2022_23_income", "design"),
        FitWeightRecord("uk_frs_only_spi_fill", "importance"),
    )

    def __call__(self, dataset: UKNationalDataset) -> UKNationalDataset:
        return dataset


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
    assert result.terminal_gates.passed is True
    assert result.terminal_gate_path == coverage_json.resolve()
    assert result.input_coverage_path == result.terminal_gate_path
    assert result.dataset.source_h5 == input_h5.resolve()
    assert staging_h5.exists()
    staged = load_uk_national_dataset(staging_h5)
    assert staged.source_h5 == staging_h5.resolve()
    assert staged.person["employment_income"].tolist() == [50_000.0]
    assert staged.household["household_weight"].tolist() == [2.0]
    diagnostic = json.loads(coverage_json.read_text())
    assert diagnostic["enforced"] is True
    assert diagnostic["input_coverage"]["passed"] is True


def test_legacy_input_coverage_alias_is_byte_compatible_with_origin_main(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    legacy_json = tmp_path / "input_coverage.json"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        national_build,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _passing_gate(),
    )
    build_uk_national_dataset(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
        input_coverage_path=legacy_json,
    )

    # Pinned from origin/main's schema-1 serializer for this exact GateResult.
    expected = (
        b'{\n  "enforced": true,\n  "input_coverage": {\n'
        b'    "details": {\n      "degenerate": [],\n      "missing": [],\n'
        b'      "required_columns": 1\n    },\n    "failures": [],\n'
        b'    "passed": true\n  },\n  "schema_version": 1\n}\n'
    )
    assert legacy_json.read_bytes() == expected


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
    with pytest.raises(RuntimeError, match="Release gates failed"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            input_coverage_path=coverage_json,
        )

    assert not staging_h5.exists()
    diagnostic = json.loads(coverage_json.read_text())
    coverage = diagnostic["input_coverage"]
    assert coverage["passed"] is False
    assert coverage["details"]["degenerate"] == ["employment_income"]


def test_default_terminal_report_write_precedes_gate_failure_raise(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    default_terminal_json = staging_h5.with_suffix(".terminal_gates.json")
    _write_toy_h5(input_h5)
    events: list[str] = []
    real_loader = national_build.load_uk_national_dataset
    real_report_writer = national_build.write_uk_terminal_gate_report

    def preflight(**_kwargs) -> None:
        events.append("preflight")

    def stage_contract(_stage_names) -> None:
        events.append("stage contract")

    def load(path):
        events.append("load")
        return real_loader(path)

    def evaluate(_dataset, _engine):
        events.append("evaluate")
        return _failing_gate()

    def write_report(report, path):
        events.append("write report")
        assert Path(path) == default_terminal_json.resolve()
        return real_report_writer(report, path)

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        preflight,
    )
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_build_stages",
        stage_contract,
    )
    monkeypatch.setattr(national_build, "load_uk_national_dataset", load)
    monkeypatch.setattr(national_build, "uk_release_input_coverage_gate", evaluate)
    monkeypatch.setattr(
        national_build,
        "write_uk_terminal_gate_report",
        write_report,
    )

    with pytest.raises(RuntimeError, match="Release gates failed"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            terminal_gate_path=None,
        )
    events.append("raise")

    assert events == [
        "preflight",
        "stage contract",
        "load",
        "evaluate",
        "write report",
        "raise",
    ]
    assert default_terminal_json.is_file()
    assert json.loads(default_terminal_json.read_text())["passed"] is False
    assert not staging_h5.exists()


def test_national_build_real_terminal_batch_passes_before_staging(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build, terminal_gates

    input_h5 = tmp_path / "healthy.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_two_row_h5(input_h5)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        national_build,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _passing_gate(),
    )
    monkeypatch.setattr(
        terminal_gates,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _passing_gate(),
    )
    monkeypatch.setattr(
        national_build,
        "uk_terminal_gate_report",
        real_uk_terminal_gate_report,
    )

    result = build_uk_national_dataset(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
        terminal_gate_path=terminal_json,
    )

    assert result.terminal_gates.passed
    assert [gate.name for gate in result.terminal_gates.results] == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
    ]
    assert result.input_coverage is result.terminal_gates.results[0]
    assert result.terminal_gate_path == terminal_json.resolve()
    assert staging_h5.is_file()
    payload = json.loads(terminal_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert set(payload["gates"]) == {
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
    }
    assert "weights_audit" not in payload["gates"]
    assert "export_surface" not in payload["gates"]
    assert "target_surface" not in payload["gates"]
    assert "target_fit" not in payload["gates"]


def test_national_build_real_terminal_batch_writes_all_findings_before_raise(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build, terminal_gates

    input_h5 = tmp_path / "defective.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_two_row_h5(input_h5, employment_income=(0.0, 0.0))
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
    monkeypatch.setattr(
        terminal_gates,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _failing_gate(),
    )
    monkeypatch.setattr(
        national_build,
        "uk_terminal_gate_report",
        real_uk_terminal_gate_report,
    )

    with pytest.raises(RuntimeError, match="Release gates failed") as error:
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=staging_h5,
            stages=(UKNationalStage("hmrc_spi_income", lambda dataset: dataset),),
            coverage_engine=object(),
            terminal_gate_path=terminal_json,
        )

    assert "[uk_release_input_coverage]" in str(error.value)
    assert "[degenerate_release_surface]" in str(error.value)
    assert "[weights_audit]" in str(error.value)
    assert terminal_json.is_file()
    payload = json.loads(terminal_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["gates"]["uk_release_input_coverage"]["passed"] is False
    assert payload["gates"]["degenerate_release_surface"]["passed"] is False
    assert payload["gates"]["weights_audit"] == {
        "details": {"evidence_missing": True, "fits_checked": 0},
        "failures": [
            "A production fit stage ran but emitted no FitWeightRecord evidence; "
            "an absent audit is not a passing audit."
        ],
        "passed": False,
    }
    assert not staging_h5.exists()


def test_national_build_includes_parity_trio_only_with_real_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build, terminal_gates

    input_h5 = tmp_path / "healthy.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_two_row_h5(input_h5)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        national_build,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _passing_gate(),
    )
    monkeypatch.setattr(
        terminal_gates,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _passing_gate(),
    )
    monkeypatch.setattr(
        national_build,
        "uk_terminal_gate_report",
        real_uk_terminal_gate_report,
    )
    parity = UKReleaseParityEvidence(
        candidate_columns=("person.employment_income",),
        reference_columns=("person.employment_income",),
        candidate_targets=("population",),
        reference_targets=("population",),
        target_relative_errors={"population": 0.0},
    )

    result = build_uk_national_dataset(
        input_h5=input_h5,
        staging_h5=staging_h5,
        stages=(UKNationalStage("hmrc_spi_income", _RecordedFitStage()),),
        coverage_engine=object(),
        parity_evidence=parity,
        terminal_gate_path=terminal_json,
    )

    assert result.terminal_gates.passed
    weights_audit = next(
        gate for gate in result.terminal_gates.results if gate.name == "weights_audit"
    )
    assert weights_audit.details["resolved_weight_kinds"] == {
        "uk_frs_only_spi_fill": "importance",
        "uk_spi_2022_23_income": "design",
    }
    assert [gate.name for gate in result.terminal_gates.results][-3:] == [
        "export_surface",
        "target_surface",
        "target_fit",
    ]


def test_national_build_rejects_both_gate_path_names_and_h5_collisions(
    tmp_path,
) -> None:
    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            coverage_engine=object(),
            terminal_gate_path=tmp_path / "terminal.json",
            input_coverage_path=tmp_path / "coverage.json",
        )

    with pytest.raises(ValueError, match="must differ"):
        build_uk_national_dataset(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            coverage_engine=object(),
            terminal_gate_path=input_h5,
        )


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


def test_national_build_accepts_hugging_face_style_h5_symlink(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build

    cached_blob = tmp_path / "content-addressed-blob"
    input_h5 = tmp_path / "populace_uk_2023.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(cached_blob, employment_income=40_000.0)
    input_h5.symlink_to(cached_blob)
    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        national_build,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _passing_gate(),
    )

    result = build_uk_national_dataset(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
    )

    assert result.input_h5 == cached_blob.resolve()
    assert result.dataset.source_h5 == cached_blob.resolve()
    assert staging_h5.is_file()


def test_national_staging_h5_loads_through_policyengine_uk(tmp_path) -> None:
    pytest.importorskip("tables")
    policyengine_data = pytest.importorskip("policyengine_uk.data")
    from populace.build.uk_runtime.national_build import write_uk_national_dataset

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    dataset = load_uk_national_dataset(input_h5)
    assert dataset.source_h5 == input_h5.resolve()
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
    assert dataset.source_h5 == input_h5.resolve()

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
