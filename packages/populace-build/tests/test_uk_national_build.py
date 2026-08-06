from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from populace.build.gates import FitWeightRecord, GateReport, GateResult
from populace.build.uk_runtime.national_build import (
    UKNationalStage,
    build_uk_national_dataset,
    load_uk_national_frame,
)
from populace.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from populace.build.uk_runtime.terminal_gates import (
    UK_TERMINAL_GATE_SIGNING_KEY_ENV,
    UKReleaseParityEvidence,
)
from populace.build.uk_runtime.terminal_gates import (
    uk_terminal_gate_report as real_uk_terminal_gate_report,
)
from populace.build.uk_runtime.terminal_gates import (
    write_uk_terminal_gate_report as real_write_uk_terminal_gate_report,
)
from populace.frame import Frame, MassChangeRecord, WeightKind

TEST_UK_RELEASE_ID = "populace-uk-2023-frs-k535080"
TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256 = "c" * 64
TEST_UK_TERMINAL_GATE_SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


def _run_national_build(**kwargs):
    return build_uk_national_dataset(
        release_id=TEST_UK_RELEASE_ID,
        calibration_diagnostics_sha256=TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256,
        **kwargs,
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    """Rebuild the frame with the person table replaced (mass untouched)."""

    return uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )


@pytest.fixture(autouse=True)
def _trusted_terminal_gate_signing_key(monkeypatch) -> None:
    monkeypatch.setenv(
        UK_TERMINAL_GATE_SIGNING_KEY_ENV,
        TEST_UK_TERMINAL_GATE_SIGNING_KEY,
    )


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


def test_gate_evidence_reproduces_the_legacy_attr_surface() -> None:
    """_uk_gate_evidence exposes exactly what the duck-typed gates read.

    The gate modules stay deliberately duck-typed until #611 types them on
    Frame; the evidence adapter must therefore carry the metadata attrs
    (kind, period, mass log) the coverage gate's hmrc family getattr-reads,
    with the typed weights materialized authoritatively into the tables.
    """

    from populace.build.uk_runtime import national_build

    mass_log = (
        MassChangeRecord(
            entity="household",
            old_total=2.0,
            new_total=2.0,
            declared_factor=1.0,
            reason="Toy reviewed record.",
        ),
    )
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [10],
                "person_benunit_id": [100],
                "person_household_id": [1],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [100]}),
        household=pd.DataFrame({"household_id": [1], "household_weight": [2.0]}),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=mass_log,
    )

    evidence = national_build._uk_gate_evidence(frame)

    assert evidence.household_weight_kind is WeightKind.IMPORTANCE
    assert evidence.time_period == "2023"
    assert evidence.mass_log == mass_log
    assert evidence.household["household_weight"].tolist() == [2.0]
    pd.testing.assert_frame_equal(evidence.person, frame.person)
    # The same getattr surface the gates use resolves to real values, never
    # the silent fallbacks a plain table mapping produced.
    assert getattr(evidence, "household_weight_kind", None) is not None
    assert str(getattr(evidence, "time_period", "")) == "2023"
    assert tuple(getattr(evidence, "mass_log", ())) == mass_log


class _RecordedFitStage:
    fit_weight_records = (
        FitWeightRecord("uk_spi_2022_23_income", "design"),
        FitWeightRecord("uk_frs_only_spi_fill", "importance"),
    )

    def __call__(self, frame: Frame) -> Frame:
        return frame


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

    def stage_transform(frame: Frame) -> Frame:
        events.append("stage:income")
        person = frame.table("person").copy()
        person["employment_income"] = 50_000.0
        return _replace_person(frame, person)

    def assert_current(**_kwargs) -> None:
        events.append("manifest_preflight")

    def coverage_gate(evidence, _engine):
        events.append("final_coverage_gate")
        assert evidence.person["employment_income"].tolist() == [50_000.0]
        # The gate battery's evidence carries the frame's metadata surface —
        # the coverage gate's hmrc family reads these attrs, and a bare table
        # mapping silently fails them to ''/() (caught by the first
        # credentialed acceptance build, not by CI's toy stages).
        assert evidence.time_period == "2023"
        assert evidence.household_weight_kind is WeightKind.DESIGN
        assert evidence.mass_log == ()
        return _passing_gate()

    real_writer = national_build.write_uk_national_frame

    def recording_writer(frame, path):
        events.append("staging_write")
        return real_writer(frame, path)

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
        "write_uk_national_frame",
        recording_writer,
    )

    result = _run_national_build(
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
    assert result.provenance.source_h5 == input_h5.resolve()
    assert staging_h5.exists()
    staged, staged_provenance = load_uk_national_frame(staging_h5)
    assert staged_provenance.source_h5 == staging_h5.resolve()
    assert staged.person["employment_income"].tolist() == [50_000.0]
    assert staged.table("household")["household_weight"].tolist() == [2.0]
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
    _run_national_build(
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


def test_legacy_input_coverage_alias_fails_closed_without_signing_key(
    monkeypatch,
    tmp_path,
) -> None:
    """The compatibility output cannot bypass the signed terminal writer."""

    pytest.importorskip("tables")
    from populace.build.uk_runtime import national_build, terminal_gates

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    legacy_json = tmp_path / "input_coverage.json"
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
    monkeypatch.setattr(
        national_build,
        "write_uk_terminal_gate_report",
        real_write_uk_terminal_gate_report,
    )
    monkeypatch.delenv(UK_TERMINAL_GATE_SIGNING_KEY_ENV)

    with pytest.raises(RuntimeError, match="Unsigned failed report was written"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            input_coverage_path=legacy_json,
        )

    assert not staging_h5.exists()
    payload = json.loads(legacy_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["passed"] is False
    assert payload["attestation"]["signature"] is None
    assert payload["attestation"]["signing_key_sha256"] is None


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
        _run_national_build(
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
    real_loader = national_build.load_uk_national_frame
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
    monkeypatch.setattr(national_build, "load_uk_national_frame", load)
    monkeypatch.setattr(national_build, "uk_release_input_coverage_gate", evaluate)
    monkeypatch.setattr(
        national_build,
        "write_uk_terminal_gate_report",
        write_report,
    )

    with pytest.raises(RuntimeError, match="Release gates failed"):
        _run_national_build(
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

    result = _run_national_build(
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
        _run_national_build(
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

    result = _run_national_build(
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
    pytest.importorskip("tables")
    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)

    with pytest.raises(ValueError, match="mutually exclusive"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            coverage_engine=object(),
            terminal_gate_path=tmp_path / "terminal.json",
            input_coverage_path=tmp_path / "coverage.json",
        )

    with pytest.raises(ValueError, match="must differ"):
        _run_national_build(
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

    def transform(frame: Frame) -> Frame:
        nonlocal called
        called = True
        return frame

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match="Duplicate UK national stage"):
        _run_national_build(
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

    def stage_transform(frame: Frame) -> Frame:
        nonlocal stage_called
        stage_called = True
        return frame

    def reject_manifest(**_kwargs) -> None:
        raise ValueError("manifest drift")

    monkeypatch.setattr(
        national_build,
        "assert_uk_release_input_coverage_manifest_current",
        reject_manifest,
    )

    with pytest.raises(ValueError, match="manifest drift"):
        _run_national_build(
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

    def break_links(frame: Frame) -> Frame:
        person = frame.table("person").copy()
        person["person_household_id"] = 999
        return _replace_person(frame, person)

    # Frame construction inside the stage is where the invariant now lives.
    with pytest.raises(ValueError, match="absent from the table"):
        _run_national_build(
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
            lambda frame: uk_national_frame(
                person=frame.table("person"),
                benunit=frame.table("benunit"),
                household=frame.table("household"),
                time_period=None,
            ),
            "time_period must be a non-empty string",
        ),
        (
            "zero_population",
            lambda frame: uk_national_frame(
                person=frame.table("person"),
                benunit=frame.table("benunit"),
                household=frame.table("household").assign(household_weight=0.0),
                time_period=uk_time_period(frame),
            ),
            "Weights cannot be all zero",
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
        _run_national_build(
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
        _run_national_build(
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

    result = _run_national_build(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
    )

    assert result.input_h5 == cached_blob.resolve()
    assert result.provenance.source_h5 == cached_blob.resolve()
    assert staging_h5.is_file()


def test_national_staging_h5_loads_through_policyengine_uk(tmp_path) -> None:
    pytest.importorskip("tables")
    policyengine_data = pytest.importorskip("policyengine_uk.data")
    from populace.build.uk_runtime.national_build import write_uk_national_frame

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    frame, provenance = load_uk_national_frame(input_h5)
    assert provenance.source_h5 == input_h5.resolve()
    frame = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
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

    write_uk_national_frame(frame, staging_h5)

    round_tripped, _staging_provenance = load_uk_national_frame(staging_h5)
    assert uk_household_weight_kind(round_tripped) is WeightKind.IMPORTANCE
    assert round_tripped.mass_log == frame.mass_log

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
    frame, _provenance = load_uk_national_frame(input_h5)
    staging_h5.write_bytes(b"previous-good-artifact")

    def fail_store(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated HDF write failure")

    monkeypatch.setattr(national_build.pd, "HDFStore", fail_store)

    with pytest.raises(OSError, match="simulated HDF write failure"):
        national_build.write_uk_national_frame(frame, staging_h5)

    assert staging_h5.read_bytes() == b"previous-good-artifact"
    assert list(tmp_path.glob(".staging.h5.*.tmp.h5")) == []
