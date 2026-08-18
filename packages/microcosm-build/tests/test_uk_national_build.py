from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.gate_battery import (
    GateBatteryBlockedError,
    gate_signing_key_env,
)
from microcosm.build.gates import FitWeightRecord, GateResult
from microcosm.build.plan import Stage, StagePlan
from microcosm.build.uk_runtime.battery_bindings import UKGateBinding
from microcosm.build.uk_runtime.national_build import (
    UKNationalStage,
    build_uk_national_dataset,
    load_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    _uk_gate_surface,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    uk_release_input_coverage_gate,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind

TEST_UK_RELEASE_ID = "populace-uk-2023-frs-k535080"
TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256 = "c" * 64
TEST_UK_TERMINAL_GATE_SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="
#: A fixed exclusion clock inside the committed register's validity window
#: keeps toy builds deterministic across the suite's lifetime.
TEST_UK_EXCLUSION_CLOCK = date(2026, 9, 1)


def _toy_coverage_evaluator(context, parameters):
    """Pass the manifest preflight, run the real coverage gate at terminal.

    The manifest-currency check needs the shipped coverage machinery these
    toy builds do not carry; the terminal verdict stays the real gate over
    the real frame surface, as the legacy seam fixture ran it.
    """

    if parameters.get("check") == "manifest_current":
        return GateResult(
            name="release_input_coverage",
            passed=True,
            details={"check": "manifest_current", "toy_preflight": True},
        )
    return uk_release_input_coverage_gate(
        _uk_gate_surface(context.frame), context.artifacts["coverage_engine"]
    )


def _toy_gate_registry() -> dict[str, UKGateBinding]:
    """The seam-test registry: real terminal coverage, pass-through roster.

    These seam tests use toy stages, so the family-roster gate and the
    manifest preflight are pass-throughs (both have their own tests) and
    every gate without a binding is a named ``evidence_absent`` gap —
    non-blocking off the release-candidate posture, exactly the legacy
    fixture's effect of reporting only the coverage verdict. The one
    exception is the weights audit: its manifest entry declares
    ``evidence_absent_blocks`` (an absent audit is not a passing audit,
    in every posture), so the seam registry binds it as a pass-through —
    the strict-absence behavior has its own tests.
    """

    return {
        "weights_audit": UKGateBinding(
            name="weights_audit",
            evaluator=lambda context, parameters: GateResult(
                name="weights_audit",
                passed=True,
                details={"toy_audit": True},
            ),
            needs_frame=False,
        ),
        "release_input_coverage": UKGateBinding(
            name="release_input_coverage",
            evaluator=_toy_coverage_evaluator,
            parameter_keys=frozenset({"check"}),
            artifact_keys=frozenset({"coverage_engine"}),
            frame_predicate=(
                lambda parameters: parameters.get("check") != "manifest_current"
            ),
            legacy_name="uk_release_input_coverage",
        ),
        "source_coverage": UKGateBinding(
            name="source_coverage",
            evaluator=lambda context, parameters: GateResult(
                name="source_coverage",
                passed=True,
                details={"toy_stage_roster": True},
            ),
            needs_frame=False,
        ),
    }


def _run_national_build(**kwargs):
    kwargs.setdefault("gate_registry", _toy_gate_registry())
    kwargs.setdefault("now", TEST_UK_EXCLUSION_CLOCK)
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
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _assert_same_frame_payload(left: Frame, right: Frame) -> None:
    assert left.schema == right.schema
    assert left.entities == right.entities
    for entity in left.entities:
        pd.testing.assert_frame_equal(
            left.table(entity),
            right.table(entity),
            check_exact=True,
            check_dtype=True,
        )
    assert left.weighted_entities == right.weighted_entities
    for entity in left.weighted_entities:
        assert left.weights_for(entity).kind is right.weights_for(entity).kind
        pd.testing.assert_series_equal(
            pd.Series(left.weights_for(entity).values),
            pd.Series(right.weights_for(entity).values),
            check_exact=True,
            check_dtype=True,
        )
    pd.testing.assert_series_equal(left.strata, right.strata, check_exact=True)
    assert left.mass_log == right.mass_log
    assert left.metadata == right.metadata


@pytest.fixture(autouse=True)
def _trusted_terminal_gate_signing_key(monkeypatch) -> None:
    monkeypatch.setenv(
        gate_signing_key_env("uk"),
        TEST_UK_TERMINAL_GATE_SIGNING_KEY,
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
    n = 100
    household_ids = np.arange(1, n + 1)
    person_ids = np.arange(10, 10 + n)
    benunit_ids = np.arange(100, 100 + n)
    employment = np.resize(np.asarray(employment_income, dtype=float), n)

    def flags(true_count: int) -> list[bool]:
        return [index < true_count for index in range(n)]

    with pd.HDFStore(path) as store:
        store.put(
            "person",
            pd.DataFrame(
                {
                    "person_id": person_ids,
                    "person_household_id": household_ids,
                    "person_benunit_id": benunit_ids,
                    "employment_income": employment,
                    "age": [6 + index % 3 for index in range(n)],
                    "would_claim_marriage_allowance": flags(50),
                    "would_claim_scp": flags(85),
                    "attends_private_school_random_draw": np.linspace(0.01, 0.99, n),
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "benunit",
            pd.DataFrame(
                {
                    "benunit_id": benunit_ids,
                    "would_claim_child_benefit": flags(89),
                    "child_benefit_opts_out": flags(23),
                    "would_claim_pc": flags(70),
                    "would_claim_uc": flags(55),
                    "would_claim_tfc": flags(59),
                    "would_claim_extended_childcare": flags(81),
                    "would_claim_universal_childcare": flags(56),
                    "would_claim_targeted_childcare": flags(60),
                    "maximum_extended_childcare_hours_usage": np.linspace(1.0, 30.0, n),
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": household_ids,
                    "household_weight": np.ones(n),
                    "household_is_spi_synthetic": [
                        index % 2 == 1 for index in range(n)
                    ],
                    "household_is_capital_gains_clone": [
                        index % 4 >= 2 for index in range(n)
                    ],
                    "household_owns_tv": flags(95),
                    "would_evade_tv_licence_fee": flags(11),
                    "main_residential_property_purchased_is_first_home": flags(38),
                    "property_purchased": flags(4),
                    "brma": [
                        "ABERDEEN_AND_SHIRE" if index % 2 == 0 else "ARGYLL_AND_BUTE"
                        for index in range(n)
                    ],
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


def _registry_with_coverage(gate_result_factory) -> dict[str, UKGateBinding]:
    """The toy registry with the terminal coverage verdict stubbed."""

    def evaluator(context, parameters):
        if parameters.get("check") == "manifest_current":
            return GateResult(
                name="release_input_coverage",
                passed=True,
                details={"check": "manifest_current", "toy_preflight": True},
            )
        return gate_result_factory()

    registry = _toy_gate_registry()
    registry["release_input_coverage"] = UKGateBinding(
        name="release_input_coverage",
        evaluator=evaluator,
        parameter_keys=frozenset({"check"}),
        artifact_keys=frozenset({"coverage_engine"}),
        frame_predicate=(
            lambda parameters: parameters.get("check") != "manifest_current"
        ),
        legacy_name="uk_release_input_coverage",
    )
    return registry


def test_driver_validates_the_uk_residue_after_each_stage(
    monkeypatch, tmp_path
) -> None:
    """The driver's post-stage validate is load-bearing, not decorative.

    A stage can directly construct a kernel-valid Frame carrying the exported
    ``household_weight`` column. Only the driver's
    ``validate_uk_national_frame`` call can stop that column from returning
    to the in-build carrier.
    """

    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_two_row_h5(input_h5)

    def return_export_column(frame: Frame) -> Frame:
        return Frame(
            {
                "person": frame.table("person"),
                "benunit": frame.table("benunit"),
                "household": frame.table("household").assign(household_weight=999.0),
            },
            frame.schema,
            {"household": frame.weights_for("household")},
            metadata=frame.metadata,
            mass_log=frame.mass_log,
        )

    with pytest.raises(ValueError, match="must not persist exported weight"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            stages=(UKNationalStage("export_column", return_export_column),),
            coverage_engine=object(),
        )


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
    from microcosm.build.uk_runtime import national_build

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

    def recording_coverage(context, parameters):
        if parameters.get("check") == "manifest_current":
            events.append("manifest_preflight")
            return GateResult(
                name="release_input_coverage",
                passed=True,
                details={"check": "manifest_current"},
            )
        events.append("final_coverage_gate")
        surface = _uk_gate_surface(context.frame)
        assert surface.person["employment_income"].tolist() == [50_000.0]
        # The battery's evidence surface carries the frame's metadata — the
        # coverage gate's hmrc family reads these attrs, and a bare table
        # mapping silently fails them to ''/() (caught by the first
        # credentialed acceptance build, not by CI's toy stages).
        assert surface.time_period == "2023"
        assert surface.household_weight_kind is WeightKind.DESIGN
        assert surface.mass_log == ()
        return _passing_gate()

    registry = _toy_gate_registry()
    registry["release_input_coverage"] = UKGateBinding(
        name="release_input_coverage",
        evaluator=recording_coverage,
        parameter_keys=frozenset({"check"}),
        artifact_keys=frozenset({"coverage_engine"}),
        frame_predicate=(
            lambda parameters: parameters.get("check") != "manifest_current"
        ),
        legacy_name="uk_release_input_coverage",
    )

    real_writer = national_build.write_uk_national_frame

    def recording_writer(frame, path):
        events.append("staging_write")
        return real_writer(frame, path)

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
        gate_registry=registry,
    )

    assert events == [
        "manifest_preflight",
        "stage:income",
        "final_coverage_gate",
        "staging_write",
    ]
    assert result.sampling_receipt is None
    assert result.stage_names == ("income",)
    assert result.input_coverage.passed is True
    assert result.gate_report["blocked_at_phase"] is None
    assert result.gate_report["phases_evaluated"] == ["preflight", "terminal"]
    gates = result.gate_report["gates"]
    assert gates["uk_release_input_coverage"]["status"] == "passed"
    assert result.gate_report["release_evidence"] == {
        "calibration_diagnostics_sha256": TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256
    }
    assert result.terminal_gate_path == coverage_json.resolve()
    assert result.input_coverage_path == result.terminal_gate_path
    assert result.provenance.source_h5 == input_h5.resolve()
    assert staging_h5.exists()
    staged, staged_provenance = load_uk_national_frame(staging_h5)
    assert staged_provenance.source_h5 == staging_h5.resolve()
    assert staged.person["employment_income"].tolist() == [50_000.0]
    assert staged.weights_for("household").values.tolist() == [2.0]
    diagnostic = json.loads(coverage_json.read_text())
    assert diagnostic["enforced"] is True
    assert diagnostic["input_coverage"]["passed"] is True


def test_national_build_accepts_stage_plan_and_records_stage_evidence(
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(input_h5)

    def add_bonus(frame: Frame) -> Frame:
        person = frame.table("person").copy()
        person["bonus_income"] = [125.0]
        return _replace_person(frame, person)

    result = _run_national_build(
        input_h5=input_h5,
        staging_h5=staging_h5,
        stages=StagePlan(
            (
                Stage(
                    name="income",
                    transform=add_bonus,
                    produces=("bonus_income",),
                ),
            )
        ),
        coverage_engine=object(),
        gate_registry=_registry_with_coverage(_passing_gate),
    )

    assert result.stage_names == ("income",)
    assert [record.stage for record in result.stage_records] == ["income"]
    assert result.stage_records[0].produced == ("bonus_income",)
    assert result.stage_records[0].nonzero_share == {"bonus_income": 1.0}


def test_deprecated_shim_and_country_stage_plan_paths_are_payload_identical(
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    spec = load_country_spec("uk")
    # Select by name, not position: the manifest's stage order changed when
    # frs_spine became the pipeline root, and this test only exercises the
    # two national staging stages.
    stages_by_name = {stage.stage: stage for stage in spec.sources.stages}
    retained_outputs = stages_by_name["frs_hmrc_retained_leaves"].outputs
    hmrc_outputs = stages_by_name["hmrc_spi_income"].outputs

    def retained(frame: Frame) -> Frame:
        person = frame.table("person").copy()
        for index, column in enumerate(retained_outputs, start=1):
            person[column] = float(index)
        return _replace_person(frame, person)

    def hmrc(frame: Frame) -> Frame:
        person = frame.table("person").copy()
        for index, column in enumerate(hmrc_outputs, start=1):
            person[column] = float(index * 10)
        return _replace_person(frame, person)

    legacy = _run_national_build(
        input_h5=input_h5,
        staging_h5=tmp_path / "legacy.h5",
        stages=(
            UKNationalStage("frs_hmrc_retained_leaves", retained),
            UKNationalStage("hmrc_spi_income", hmrc),
        ),
        coverage_engine=object(),
        gate_registry=_registry_with_coverage(_passing_gate),
    )
    shared = _run_national_build(
        input_h5=input_h5,
        staging_h5=tmp_path / "shared.h5",
        stages=country_stage_plan(
            spec,
            {
                "frs_hmrc_retained_leaves": retained,
                "hmrc_spi_income": hmrc,
            },
            stage_names=("frs_hmrc_retained_leaves", "hmrc_spi_income"),
        ),
        coverage_engine=object(),
        gate_registry=_registry_with_coverage(_passing_gate),
    )

    _assert_same_frame_payload(legacy.frame, shared.frame)


def _write_clone_family_h5(path: Path) -> None:
    """Four base clone families (canonical + one geography clone each).

    Persons are ``household_id + 200``, so the canonical max is 214 and the
    clone multiplier is 1000 — clone ids reverse onto the canonical surface
    exactly as the stage fence re-derives them.
    """

    canonical = [11, 12, 13, 14]
    regions = ["london", "north", "london", "north"]
    rows = []
    for household_id, region in zip(canonical, regions, strict=True):
        rows.append((household_id, 0, region))
        rows.append((household_id + 1_000, 1, "scotland"))
    rows.sort()
    ids = [row[0] for row in rows]
    with pd.HDFStore(path) as store:
        store.put(
            "person",
            pd.DataFrame(
                {
                    "person_id": [value + 200 for value in ids],
                    "person_household_id": ids,
                    "person_benunit_id": [value + 5_000_000 for value in ids],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "benunit",
            pd.DataFrame({"benunit_id": [value + 5_000_000 for value in ids]}),
            format="table",
            data_columns=True,
        )
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": ids,
                    "household_weight": [2.0] * len(ids),
                    "clone_index": [row[1] for row in rows],
                    "region": [row[2] for row in rows],
                    "household_is_spi_synthetic": [False] * len(ids),
                    "household_is_capital_gains_clone": [False] * len(ids),
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


def test_national_build_samples_the_loaded_frame_before_stages(tmp_path) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    coverage_json = tmp_path / "gates.json"
    _write_clone_family_h5(input_h5)
    stage_household_counts: list[int] = []

    def stage_transform(frame: Frame) -> Frame:
        stage_household_counts.append(len(frame.table("household")))
        return frame

    result = _run_national_build(
        input_h5=input_h5,
        staging_h5=staging_h5,
        stages=(UKNationalStage("income", stage_transform),),
        coverage_engine=object(),
        input_coverage_path=coverage_json,
        sample_fraction=0.5,
        sample_seed=3,
        gate_registry=_registry_with_coverage(_passing_gate),
    )

    receipt = result.sampling_receipt
    assert receipt is not None
    # The stages saw the sampled frame — the rung is upstream of stage one.
    assert stage_household_counts == [receipt["realized_household_count"]]
    assert receipt["realized_household_count"] < 8
    assert receipt["uk_policy"]["sampling_unit"] == "source_frs_family"
    # Renormalization: the staged artifact carries the full input mass.
    staged, _staged_provenance = load_uk_national_frame(staging_h5)
    assert float(staged.weights_for("household").total) == pytest.approx(8 * 2.0)
    assert result.gate_report["blocked_at_phase"] is None


def test_legacy_input_coverage_alias_is_byte_compatible_with_origin_main(
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    legacy_json = tmp_path / "input_coverage.json"
    _write_toy_h5(input_h5, employment_income=40_000.0)
    _run_national_build(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
        input_coverage_path=legacy_json,
        gate_registry=_registry_with_coverage(_passing_gate),
    )

    # Pinned from origin/main's schema-1 serializer for this exact GateResult.
    expected = (
        b'{\n  "enforced": true,\n  "input_coverage": {\n'
        b'    "details": {\n      "degenerate": [],\n      "missing": [],\n'
        b'      "required_columns": 1\n    },\n    "failures": [],\n'
        b'    "passed": true\n  },\n  "schema_version": 1\n}\n'
    )
    assert legacy_json.read_bytes() == expected


def test_full_scale_build_refuses_to_stage_unsigned(monkeypatch, tmp_path) -> None:
    """No full-scale staging artifact without an attested report.

    The battery core records a missing key as ``signing_error`` and carries
    on; the national build restores the legacy guarantee for full-scale
    builds — the unsigned report is on disk, the H5 is not.
    """

    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_two_row_h5(input_h5)
    monkeypatch.delenv(gate_signing_key_env("uk"))

    with pytest.raises(RuntimeError, match="unsigned and this is a full-scale"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            terminal_gate_path=terminal_json,
            gate_registry=_registry_with_coverage(_passing_gate),
        )

    assert not staging_h5.exists()
    payload = json.loads(terminal_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["shippable"] is False
    assert payload["attestation"]["signature"] is None
    assert payload["attestation"]["signing_key_sha256"] is None
    assert "signing_error" in payload["attestation"]


def test_rung_build_proceeds_unsigned_with_an_honest_report(
    monkeypatch, tmp_path
) -> None:
    """A rung is structurally non-releasable, so it may run without the key;
    its report says so instead of pretending."""

    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_clone_family_h5(input_h5)
    monkeypatch.delenv(gate_signing_key_env("uk"))

    result = _run_national_build(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
        terminal_gate_path=terminal_json,
        sample_fraction=0.5,
        sample_seed=3,
        gate_registry=_registry_with_coverage(_passing_gate),
    )

    assert staging_h5.exists()
    assert result.gate_report["shippable"] is False
    assert "signing_error" in result.gate_report["attestation"]


def test_national_build_gate_failure_writes_diagnostic_not_h5(tmp_path) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    coverage_json = tmp_path / "input_coverage.json"
    _write_toy_h5(input_h5)
    with pytest.raises(GateBatteryBlockedError, match="Gate battery blocked"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            input_coverage_path=coverage_json,
            gate_registry=_registry_with_coverage(_failing_gate),
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
    from microcosm.build.uk_runtime import national_build

    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    default_terminal_json = staging_h5.with_suffix(".terminal_gates.json")
    _write_toy_h5(input_h5)
    events: list[str] = []
    real_loader = national_build.load_uk_national_frame

    def load(path):
        events.append("load")
        return real_loader(path)

    def recording_coverage(context, parameters):
        if parameters.get("check") == "manifest_current":
            events.append("preflight")
            return GateResult(
                name="release_input_coverage",
                passed=True,
                details={"check": "manifest_current"},
            )
        events.append("evaluate")
        # The preflight report is already on disk before the frame loads —
        # the write-then-block ordering holds per phase, not just at the end.
        assert json.loads(default_terminal_json.read_text())["phases_evaluated"] == [
            "preflight"
        ]
        return _failing_gate()

    def recording_roster(context, parameters):
        events.append("stage contract")
        return GateResult(name="source_coverage", passed=True, details={})

    registry = _toy_gate_registry()
    registry["release_input_coverage"] = UKGateBinding(
        name="release_input_coverage",
        evaluator=recording_coverage,
        parameter_keys=frozenset({"check"}),
        artifact_keys=frozenset({"coverage_engine"}),
        frame_predicate=(
            lambda parameters: parameters.get("check") != "manifest_current"
        ),
        legacy_name="uk_release_input_coverage",
    )
    registry["source_coverage"] = UKGateBinding(
        name="source_coverage",
        evaluator=recording_roster,
        needs_frame=False,
    )
    monkeypatch.setattr(national_build, "load_uk_national_frame", load)

    with pytest.raises(GateBatteryBlockedError, match="Gate battery blocked"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            coverage_engine=object(),
            terminal_gate_path=None,
            gate_registry=registry,
        )
    events.append("raise")

    assert events == [
        "preflight",
        "stage contract",
        "load",
        "evaluate",
        "raise",
    ]
    assert default_terminal_json.is_file()
    payload = json.loads(default_terminal_json.read_text())
    assert payload["schema_version"] == 4
    assert payload["blocked_at_phase"] == "terminal"
    assert payload["gates"]["uk_release_input_coverage"]["status"] == "failed"
    assert not staging_h5.exists()


def _stub_real_coverage(monkeypatch, gate_result_factory) -> None:
    """Point the real registry's coverage binding at a stubbed verdict.

    The bindings resolve the manifest assert and the coverage gate as
    module globals at call time, so patching them where the bindings look
    them up leaves every other real binding untouched.
    """

    from microcosm.build.uk_runtime import battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "assert_uk_release_input_coverage_manifest_current",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        battery_bindings,
        "assert_uk_release_input_coverage_build_stages",
        lambda _stage_names, manifest=None: None,
    )
    monkeypatch.setattr(
        battery_bindings,
        "uk_release_input_coverage_gate",
        lambda _surface, _engine, manifest=None: gate_result_factory(),
    )


def test_national_build_real_terminal_batch_blocks_incomplete_qrf_before_staging(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "healthy.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_two_row_h5(input_h5)
    _stub_real_coverage(monkeypatch, _passing_gate)

    with pytest.raises(GateBatteryBlockedError) as error:
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            # The audit's absence blocks every posture (evidence_absent_blocks),
            # so this supplies the HMRC stage's audit evidence. The real QRF
            # gate is spec-armed now and correctly blocks this tiny synthetic
            # frame because it lacks the declared QRF output surface.
            stages=(UKNationalStage("hmrc_spi_income", _RecordedFitStage()),),
            coverage_engine=object(),
            terminal_gate_path=terminal_json,
            gate_registry=None,  # the real UK registry
        )

    assert "[uk_qrf_tail_concentration]" in str(error.value)
    assert error.value.phase == "terminal"
    assert not staging_h5.exists()
    payload = json.loads(terminal_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["blocked_at_phase"] == "terminal"
    statuses = {entry_id: gate["status"] for entry_id, gate in payload["gates"].items()}
    assert statuses == {
        "uk_release_input_coverage_manifest_current": "passed",
        "uk_release_family_build_stages": "passed",
        "uk_release_input_coverage": "passed",
        "uk_degenerate_release_surface": "passed",
        "uk_zero_weight_strata": "passed",
        "uk_weight_ess": "passed",
        "uk_weight_ratio": "passed",
        "uk_weights_audit": "passed",
        "uk_nonnegative_columns": "passed",
        "uk_take_up_signal": "passed",
        "uk_brma_enum_domain": "passed",
        # The legacy report omitted unevidenced gates; the battery names
        # every gap — non-blocking off the release-candidate posture.
        "uk_export_surface": "evidence_absent",
        "uk_target_surface": "evidence_absent",
        "uk_target_fit": "evidence_absent",
        "uk_input_mass_parity": "evidence_absent",
        "uk_qrf_tail_concentration": "failed",
    }
    # One exclusion clock: the evaluated exclusion gate stamps the injected
    # date, never a per-gate default.
    degenerate = payload["gates"]["uk_degenerate_release_surface"]
    assert (
        degenerate["details"]["exclusions_evaluated_on"]
        == TEST_UK_EXCLUSION_CLOCK.isoformat()
    )
    assert payload["release_evidence"] == {
        "calibration_diagnostics_sha256": TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256
    }


def test_national_build_real_terminal_batch_writes_all_findings_before_raise(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "defective.h5"
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    _write_two_row_h5(input_h5, employment_income=(0.0, 0.0))
    _stub_real_coverage(monkeypatch, _failing_gate)

    with pytest.raises(GateBatteryBlockedError) as error:
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            stages=(UKNationalStage("hmrc_spi_income", lambda dataset: dataset),),
            coverage_engine=object(),
            terminal_gate_path=terminal_json,
            gate_registry=None,  # the real UK registry
        )

    assert "[uk_release_input_coverage]" in str(error.value)
    assert "[uk_degenerate_release_surface]" in str(error.value)
    assert "[uk_weights_audit]" in str(error.value)
    assert error.value.phase == "terminal"
    assert terminal_json.is_file()
    payload = json.loads(terminal_json.read_text(encoding="utf-8"))
    assert payload["blocked_at_phase"] == "terminal"
    assert payload["shippable"] is False
    assert payload["gates"]["uk_release_input_coverage"]["status"] == "failed"
    assert payload["gates"]["uk_degenerate_release_surface"]["status"] == "failed"
    weights_audit = payload["gates"]["uk_weights_audit"]
    assert weights_audit["status"] == "failed"
    assert weights_audit["details"] == {
        "evidence_missing": True,
        "fits_checked": 0,
    }
    assert weights_audit["failures"] == [
        "A production fit stage ran but emitted no FitWeightRecord evidence; "
        "an absent audit is not a passing audit."
    ]
    assert not staging_h5.exists()


def test_national_build_parity_trio_is_evidence_absent(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "healthy.h5"
    _write_two_row_h5(input_h5)
    _stub_real_coverage(monkeypatch, _passing_gate)

    terminal_json = tmp_path / "terminal_gates.json"
    with pytest.raises(GateBatteryBlockedError):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            stages=(UKNationalStage("hmrc_spi_income", _RecordedFitStage()),),
            coverage_engine=object(),
            terminal_gate_path=terminal_json,
            gate_registry=None,
        )

    gates = json.loads(terminal_json.read_text(encoding="utf-8"))["gates"]
    assert gates["uk_weights_audit"]["status"] == "passed"
    assert gates["uk_weights_audit"]["details"]["resolved_weight_kinds"] == {
        "uk_frs_only_spi_fill": "importance",
        "uk_spi_2022_23_income": "design",
    }
    for entry_id in ("uk_export_surface", "uk_target_surface", "uk_target_fit"):
        assert gates[entry_id]["status"] == "evidence_absent", entry_id
        assert gates[entry_id]["reason"] == "missing evidence: parity_evidence"
    assert gates["uk_input_mass_parity"]["status"] == "evidence_absent"
    assert gates["uk_qrf_tail_concentration"]["status"] == "failed"


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
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    called = False

    def transform(frame: Frame) -> Frame:
        nonlocal called
        called = True
        return frame

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


def test_national_build_manifest_failure_blocks_before_stages_with_a_report(
    tmp_path,
) -> None:
    """Preflight drift blocks before any stage — and now leaves a report.

    The legacy assertions raised bare, deleting the stale outputs and
    writing nothing; the battery persists the refusal as a schema-4 report
    with the terminal entries honestly ``unreached``.
    """

    pytest.importorskip("tables")

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

    def drifting_coverage(context, parameters):
        if parameters.get("check") == "manifest_current":
            raise ValueError("manifest drift")
        return _passing_gate()

    registry = _toy_gate_registry()
    registry["release_input_coverage"] = UKGateBinding(
        name="release_input_coverage",
        evaluator=drifting_coverage,
        parameter_keys=frozenset({"check"}),
        artifact_keys=frozenset({"coverage_engine"}),
        frame_predicate=(
            lambda parameters: parameters.get("check") != "manifest_current"
        ),
        legacy_name="uk_release_input_coverage",
    )

    with pytest.raises(GateBatteryBlockedError, match="manifest drift") as error:
        _run_national_build(
            input_h5=input_h5,
            staging_h5=staging_h5,
            stages=(UKNationalStage("should_not_run", stage_transform),),
            coverage_engine=object(),
            input_coverage_path=coverage_json,
            gate_registry=registry,
        )

    assert error.value.phase == "preflight"
    assert stage_called is False
    assert not staging_h5.exists()
    payload = json.loads(coverage_json.read_text())
    assert payload["schema_version"] == 4
    assert payload["blocked_at_phase"] == "preflight"
    assert (
        payload["gates"]["uk_release_input_coverage_manifest_current"]["status"]
        == "failed"
    )
    assert payload["gates"]["uk_release_input_coverage"]["status"] == "unreached"
    assert payload["gates"]["uk_weight_ratio"]["status"] == "unreached"


def test_national_build_rejects_stage_that_breaks_entity_links(tmp_path) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)

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
                household_weights=frame.weights_for("household").values,
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
    tmp_path, stage_name, transform, message
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)

    with pytest.raises(ValueError, match=message):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            stages=(UKNationalStage(stage_name, transform),),
            coverage_engine=object(),
        )


def test_national_build_refuses_to_overwrite_its_input(tmp_path) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)

    with pytest.raises(ValueError, match="must differ"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=input_h5,
            coverage_engine=object(),
        )


def test_national_build_accepts_hugging_face_style_h5_symlink(tmp_path) -> None:
    pytest.importorskip("tables")

    cached_blob = tmp_path / "content-addressed-blob"
    input_h5 = tmp_path / "populace_uk_2023.h5"
    staging_h5 = tmp_path / "staging.h5"
    _write_toy_h5(cached_blob, employment_income=40_000.0)
    input_h5.symlink_to(cached_blob)

    result = _run_national_build(
        input_h5=input_h5,
        staging_h5=staging_h5,
        coverage_engine=object(),
        gate_registry=_registry_with_coverage(_passing_gate),
    )

    assert result.input_h5 == cached_blob.resolve()
    assert result.provenance.source_h5 == cached_blob.resolve()
    assert staging_h5.is_file()


def test_national_staging_h5_loads_through_policyengine_uk(tmp_path) -> None:
    pytest.importorskip("tables")
    policyengine_data = pytest.importorskip("policyengine_uk.data")
    from microcosm.build.uk_runtime.national_build import write_uk_national_frame

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
        household_weights=frame.weights_for("household").values,
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
    from microcosm.build.uk_runtime import national_build

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


def _counting_stage(name: str, calls: list[str] | None = None) -> UKNationalStage:
    def transform(frame: Frame) -> Frame:
        if calls is not None:
            calls.append(name)
        person = frame.table("person").copy()
        person["employment_income"] = person["employment_income"] + 1.0
        return _replace_person(frame, person)

    return UKNationalStage(name=name, transform=transform)


def _assert_same_staging_payload(left: Path, right: Path) -> None:
    left_frame, _ = load_uk_national_frame(left)
    right_frame, _ = load_uk_national_frame(right)
    from microcosm.build.uk_runtime import uk_frame_content_identity

    assert uk_frame_content_identity(left_frame) == uk_frame_content_identity(
        right_frame
    )


def test_checkpointed_build_matches_the_monolith(monkeypatch, tmp_path) -> None:
    """The checkpointed mode is the monolith plus receipts, not a variant.

    Same input, same stages: the staged build's output is content-identical
    to the monolith's, and each stage boundary leaves a resumable checkpoint.
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")

    registry = _registry_with_coverage(_passing_gate)
    input_h5 = tmp_path / "base.h5"
    _write_two_row_h5(input_h5)
    run_config = {"input_sha256": "a" * 64, "seed": 42}

    _run_national_build(
        coverage_engine=object(),
        input_h5=input_h5,
        staging_h5=tmp_path / "mono.h5",
        stages=(_counting_stage("one"), _counting_stage("two")),
        gate_registry=registry,
    )
    calls: list[str] = []
    _run_national_build(
        coverage_engine=object(),
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        stages=(_counting_stage("one", calls), _counting_stage("two", calls)),
        checkpoint_dir=tmp_path / "checkpoints",
        run_config=run_config,
        gate_registry=registry,
    )
    assert calls == ["one", "two"]
    _assert_same_staging_payload(tmp_path / "mono.h5", tmp_path / "staged.h5")
    context = json.loads(
        (tmp_path / "checkpoints" / "stage_run_context.json").read_text()
    )
    assert context["completed"] == ["one", "two"]

    # A full resume re-runs no transform and reproduces the same payload.
    resumed_calls: list[str] = []
    _run_national_build(
        coverage_engine=object(),
        input_h5=input_h5,
        staging_h5=tmp_path / "resumed.h5",
        stages=(
            _counting_stage("one", resumed_calls),
            _counting_stage("two", resumed_calls),
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        run_config=run_config,
        gate_registry=registry,
    )
    assert resumed_calls == []
    _assert_same_staging_payload(tmp_path / "mono.h5", tmp_path / "resumed.h5")


def test_checkpointed_build_resumes_past_a_crash(monkeypatch, tmp_path) -> None:
    """A stage crash leaves the completed prefix; the rerun picks up after it."""

    pytest.importorskip("tables")
    pytest.importorskip("h5py")

    registry = _registry_with_coverage(_passing_gate)
    input_h5 = tmp_path / "base.h5"
    _write_two_row_h5(input_h5)
    run_config = {"input_sha256": "a" * 64, "seed": 42}

    def exploding(frame: Frame) -> Frame:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_national_build(
            coverage_engine=object(),
            input_h5=input_h5,
            staging_h5=tmp_path / "crashed.h5",
            stages=(
                _counting_stage("one"),
                UKNationalStage(name="two", transform=exploding),
            ),
            checkpoint_dir=tmp_path / "checkpoints",
            run_config=run_config,
            gate_registry=registry,
        )

    calls: list[str] = []
    _run_national_build(
        coverage_engine=object(),
        input_h5=input_h5,
        staging_h5=tmp_path / "recovered.h5",
        stages=(_counting_stage("one", calls), _counting_stage("two", calls)),
        checkpoint_dir=tmp_path / "checkpoints",
        run_config=run_config,
        gate_registry=registry,
    )
    assert calls == ["two"]


def test_checkpointed_build_pins_the_run_config(tmp_path) -> None:
    """Resuming under a different configuration is refused, never blended."""

    pytest.importorskip("tables")
    pytest.importorskip("h5py")

    registry = _registry_with_coverage(_passing_gate)
    input_h5 = tmp_path / "base.h5"
    _write_two_row_h5(input_h5)

    with pytest.raises(ValueError, match="requires run_config"):
        _run_national_build(
            coverage_engine=object(),
            input_h5=input_h5,
            staging_h5=tmp_path / "unpinned.h5",
            stages=(_counting_stage("one"),),
            checkpoint_dir=tmp_path / "checkpoints",
            gate_registry=registry,
        )

    _run_national_build(
        coverage_engine=object(),
        input_h5=input_h5,
        staging_h5=tmp_path / "first.h5",
        stages=(_counting_stage("one"),),
        checkpoint_dir=tmp_path / "checkpoints",
        run_config={"input_sha256": "a" * 64, "seed": 42},
        gate_registry=registry,
    )
    with pytest.raises(ValueError, match="new checkpoint directory"):
        _run_national_build(
            coverage_engine=object(),
            input_h5=input_h5,
            staging_h5=tmp_path / "drifted.h5",
            stages=(_counting_stage("one"),),
            checkpoint_dir=tmp_path / "checkpoints",
            run_config={"input_sha256": "b" * 64, "seed": 42},
            gate_registry=registry,
        )


def test_release_candidate_blocks_on_named_evidence_gaps(tmp_path) -> None:
    """The chartered semantics live: a candidate cannot excuse absent
    evidence, a dev build records the same gaps and continues."""

    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    registry = _registry_with_coverage(_passing_gate)

    dev = _run_national_build(
        input_h5=input_h5,
        staging_h5=tmp_path / "dev.h5",
        coverage_engine=object(),
        terminal_gate_path=tmp_path / "dev_gates.json",
        gate_registry=registry,
    )
    assert dev.gate_report["blocked_at_phase"] is None
    absent = {
        entry_id
        for entry_id, gate in dev.gate_report["gates"].items()
        if gate["status"] == "evidence_absent"
    }
    assert "uk_weight_ratio" in absent  # unbound in the toy registry

    with pytest.raises(GateBatteryBlockedError) as error:
        _run_national_build(
            input_h5=input_h5,
            staging_h5=tmp_path / "candidate.h5",
            coverage_engine=object(),
            terminal_gate_path=tmp_path / "candidate_gates.json",
            gate_registry=registry,
            release_candidate=True,
        )
    assert error.value.phase == "terminal"
    assert "[uk_weight_ratio]" in str(error.value)
    assert not (tmp_path / "candidate.h5").exists()


def test_release_candidate_is_refused_on_a_rung_before_any_unlink(
    tmp_path,
) -> None:
    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    terminal_json = tmp_path / "terminal_gates.json"
    terminal_json.write_text('{"previous_report": true}\n')

    with pytest.raises(ValueError, match="structurally non-releasable"):
        _run_national_build(
            input_h5=input_h5,
            staging_h5=tmp_path / "staging.h5",
            coverage_engine=object(),
            terminal_gate_path=terminal_json,
            sample_fraction=0.5,
            release_candidate=True,
        )

    # Configuration refusals precede the sidecar unlinks: the contradictory
    # request must not destroy the previous run's report.
    assert terminal_json.read_text() == '{"previous_report": true}\n'


@pytest.mark.parametrize(
    ("bad_arguments", "match"),
    [
        ({"release_id": ""}, "release_id"),
        ({"calibration_diagnostics_sha256": ""}, "release_evidence"),
        ({"now": datetime(2026, 9, 1, 12, 0)}, "date"),
        (
            {"release_candidate": True, "use_alias_path": True},
            "mutually exclusive",
        ),
    ],
    ids=["empty-release-id", "empty-diagnostics-sha", "datetime-clock", "alias"],
)
def test_every_identity_refusal_precedes_the_sidecar_unlinks(
    tmp_path, bad_arguments, match
) -> None:
    """No destructive step precedes argument validation — for every
    validation, including the ones the battery construction owns."""

    pytest.importorskip("tables")

    input_h5 = tmp_path / "base.h5"
    _write_toy_h5(input_h5)
    staging_h5 = tmp_path / "staging.h5"
    terminal_json = tmp_path / "terminal_gates.json"
    staging_h5.write_bytes(b"previous-artifact")
    terminal_json.write_text('{"previous_report": true}\n')
    arguments: dict = {
        "input_h5": input_h5,
        "staging_h5": staging_h5,
        "release_id": TEST_UK_RELEASE_ID,
        "calibration_diagnostics_sha256": TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256,
        "coverage_engine": object(),
        "now": TEST_UK_EXCLUSION_CLOCK,
        "gate_registry": _toy_gate_registry(),
        "terminal_gate_path": terminal_json,
    }
    arguments.update(bad_arguments)
    if arguments.pop("use_alias_path", False):
        arguments["input_coverage_path"] = arguments.pop("terminal_gate_path")

    with pytest.raises((ValueError, TypeError), match=match):
        build_uk_national_dataset(**arguments)

    assert staging_h5.read_bytes() == b"previous-artifact"
    assert terminal_json.read_text() == '{"previous_report": true}\n'
