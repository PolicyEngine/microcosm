from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.uk_runtime import calibration_run
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
    UKCalibrationRunPaths,
    run_uk_calibration,
)
from microcosm.build.uk_runtime.national_doctrine import UKNationalSolveDoctrine
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind

SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


@pytest.fixture(autouse=True)
def _signing_key(monkeypatch):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", SIGNING_KEY)


def _frame():
    ids = np.arange(4, dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "nhs_spending": [50.0, 50.0, 50.0, 50.0],
            }
        ),
        benunit=pd.DataFrame(
            {"benunit_id": ids, "universal_credit": [1.0, 1.0, 0.0, 0.0]}
        ),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [10.0, 10.0, 10.0, 10.0],
                "household_is_spi_synthetic": [False, False, False, False],
                "household_is_capital_gains_clone": [False, False, False, False],
                "electricity_consumption": [1.0, 1.0, 1.0, 1.0],
                "gas_consumption": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.DESIGN,
    )


def _registry():
    return TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=20.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _admin_anchor_values():
    values = {}
    for entry in load_country_spec("uk").gates.gates:
        if entry.id == "uk_aggregate_admin":
            for anchor in entry.parameters["anchors"]:
                values[str(anchor["name"])] = float(anchor["value"])
    return values


def test_gate_scope_classifies_every_uk_gate():
    all_ids = {entry.id for entry in load_country_spec("uk").gates.gates}
    assert set(UK_CALIBRATION_GATE_SCOPE) | set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS) == all_ids
    assert set(UK_CALIBRATION_GATE_SCOPE).isdisjoint(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
    assert all(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS.values())


def test_import_hygiene_does_not_load_national_build_in_fresh_subprocess():
    source = Path(calibration_run.__file__).read_text(encoding="utf-8")
    assert "microcosm.build.uk_runtime.national_build" not in source
    assert "from microcosm.build.uk_runtime.national_build" not in source


def test_run_uk_calibration_writes_cross_pinned_outputs(monkeypatch, tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )
    source_pins = {
        "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size},
        "ledger_facts": {"sha256": "a" * 64, "size_bytes": 1},
    }

    result = run_uk_calibration(
        paths=paths,
        input_sha256=_sha(input_h5),
        ledger_artifact=object(),
        register_registry=_registry(),
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=5),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins=source_pins,
        run_config_extra={"calibration_year": 2025},
        release_candidate=False,
        release_id="test-run",
    )

    assert paths.staging_h5.exists()
    assert paths.diagnostics_json.exists()
    assert paths.build_record_json.exists()
    assert paths.terminal_gate_json.exists()
    assert result.build_record["artifacts"]["staging_h5"]["sha256"] == _sha(paths.staging_h5)
    assert result.build_record["artifacts"]["diagnostics_json"]["sha256"] == _sha(paths.diagnostics_json)
    assert result.build_record["artifacts"]["terminal_gate_json"]["sha256"] == _sha(paths.terminal_gate_json)
    staged, _ = load_uk_national_frame(paths.staging_h5)
    assert staged.weights_for("household").kind is WeightKind.CALIBRATED
    report = json.loads(paths.terminal_gate_json.read_text())
    assert report["posture"] == "calibration_seam"
    assert set(report["scope_exclusions"]) == set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
    attestation = report["attestation"]
    signature = attestation["signature"]
    attestation["signature"] = None
    key = b"0123456789abcdef0123456789abcdef"
    assert hmac.new(key, canonical_json_bytes(report), hashlib.sha256).hexdigest() == signature
    assert result.logbook_spool.exists()


def test_run_uk_calibration_refuses_input_sha_before_outputs(tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )
    with pytest.raises(ValueError, match="sha mismatch"):
        run_uk_calibration(
            paths=paths,
            input_sha256="0" * 64,
            ledger_artifact=object(),
            register_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={
                "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size}
            },
            run_config_extra={"calibration_year": 2025},
            release_candidate=False,
            release_id="bad-sha",
        )
    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()


def test_seam_never_modifies_data_variables(monkeypatch, tmp_path: Path):
    """The seam's defining invariant: weights move, data never does.

    Every data column of every entity table in the staged H5 must be
    byte-identical to the input; only the household weights, the weight
    kind, and exactly one appended mass record may differ.
    """

    pytest.importorskip("tables")  # pandas HDF backend

    monkeypatch.setattr(
        calibration_run,
        "_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )

    # Target 30 against an initial weighted UC count of 20, so the solve
    # genuinely has to move weights while the data stays untouched.
    pulling_registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=30.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )
    run_uk_calibration(
        paths=paths,
        input_sha256=_sha(input_h5),
        ledger_artifact=object(),
        register_registry=pulling_registry,
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=50),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size}
        },
        run_config_extra={},
        release_candidate=False,
        release_id="invariant-run",
    )

    source, _ = load_uk_national_frame(input_h5)
    staged, _ = load_uk_national_frame(paths.staging_h5)
    for entity in ("person", "benunit", "household"):
        left = source.table(entity)
        right = staged.table(entity)
        assert list(left.columns) == list(right.columns), entity
        for column in left.columns:
            pd.testing.assert_series_equal(left[column], right[column])
    assert staged.weights_for("household").kind is WeightKind.CALIBRATED
    assert not np.allclose(
        staged.weights_for("household").values,
        source.weights_for("household").values,
    )
    assert len(staged.mass_log) == len(source.mass_log) + 1


def test_aggregate_admin_measurement_convention_and_refusals():
    frame = _frame()
    manifest = calibration_run._calibration_gate_manifest()

    totals, receipt = calibration_run._aggregate_admin_totals(frame, manifest)

    # Small anchors (NEED means) measure as the weighted mean over carriers;
    # the NHS total measures as the person total under mapped household
    # weights: 4 persons x 50.0 x weight 10.0.
    assert totals["need_electricity_mean_spending"] == pytest.approx(1.0)
    assert totals["need_gas_mean_spending"] == pytest.approx(1.0)
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["entity"] == "person"
    assert (
        by_anchor["need_electricity_mean_spending"]["statistic_convention"]
        == "assessed_by_anchor_magnitude"
    )

    stripped = _frame()
    stripped.table("household").drop(columns=["electricity_consumption"], inplace=True)
    with pytest.raises(ValueError, match="household.electricity_consumption"):
        calibration_run._aggregate_admin_totals(stripped, manifest)


def test_seam_pipeline_derives_a_ratified_logbook_scope():
    """The seam appends to the FRS line's chain, not a new unratified one."""

    logbook_tool = _load_logbook_tool()

    scope = logbook_tool._chain_scope(calibration_run._PIPELINE)

    assert scope == "uk/frs"
    assert scope in logbook_tool.DECLARED_SCOPES


def _load_logbook_tool():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "tools" / "logbook.py"
    spec = importlib.util.spec_from_file_location("_logbook_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refusal_records_a_failed_attempt_and_stages_nothing(tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )

    with pytest.raises(ValueError, match="sha mismatch"):
        run_uk_calibration(
            paths=paths,
            input_sha256="0" * 64,
            ledger_artifact=object(),
            register_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={
                "input_h5": {
                    "sha256": _sha(input_h5),
                    "size_bytes": input_h5.stat().st_size,
                }
            },
            run_config_extra={"calibration_year": 2025},
            release_candidate=False,
            release_id="refused-run",
        )

    # Every terminal disposition is a row; a refusal that left the chain
    # silent would hide the attempt entirely.
    spooled = sorted((tmp_path / "logbook-spool").rglob("*.json"))
    assert spooled, "refusal recorded no Logbook row"
    rows = [json.loads(path.read_text()) for path in spooled]
    assert [row["disposition"] for row in rows] == ["failed"]
    assert rows[0]["pipeline"] == calibration_run._PIPELINE
    # The refusal is explained on disk, not only in the row.
    receipts = sorted((tmp_path / "logbook-receipts").rglob("error.json"))
    assert len(receipts) == 1
    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.terminal_gate_json.exists()


def test_attempt_ids_are_unique_across_reruns_of_one_release(
    monkeypatch, tmp_path: Path
):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    source_pins = {
        "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size}
    }
    build_ids = []
    for attempt in ("a", "b"):
        run_dir = tmp_path / attempt
        run_dir.mkdir()
        result = run_uk_calibration(
            paths=UKCalibrationRunPaths(
                input_h5=input_h5,
                staging_h5=run_dir / "staged.h5",
                diagnostics_json=run_dir / "diagnostics.json",
                build_record_json=run_dir / "build_record.json",
                terminal_gate_json=run_dir / "terminal_gates.json",
            ),
            input_sha256=_sha(input_h5),
            ledger_artifact=object(),
            register_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=5),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins=source_pins,
            run_config_extra={"calibration_year": 2025},
            release_candidate=False,
            release_id="one-release-id",
        )
        build_ids.append(result.build_record["build_id"])

    # Both the local chain and the store reject a duplicate build id, so one
    # release re-run twice must not collide.
    assert build_ids[0] != build_ids[1]
    assert all(value.startswith("uk-frs-calibration-attempt-") for value in build_ids)


def test_verified_ledger_identity_reaches_the_run_evidence(monkeypatch, tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    artifact = SimpleNamespace(
        facts_sha256="d" * 64,
        fact_row_count=107_550,
        manifest_sha256="e" * 64,
        manifest={
            "artifact_id": "chronicle-uk-artifact-1cab809",
            "profile": "uk-national",
            "schema_version": 1,
            "unrelated": "not carried",
        },
    )

    result = run_uk_calibration(
        paths=UKCalibrationRunPaths(
            input_h5=input_h5,
            staging_h5=tmp_path / "staged.h5",
            diagnostics_json=tmp_path / "diagnostics.json",
            build_record_json=tmp_path / "build_record.json",
            terminal_gate_json=tmp_path / "terminal_gates.json",
        ),
        input_sha256=_sha(input_h5),
        ledger_artifact=artifact,
        register_registry=_registry(),
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=5),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size}
        },
        run_config_extra={"calibration_year": 2025},
        release_candidate=False,
        release_id="ledger-identity",
    )

    ledger = result.build_record["run_config"]["ledger"]
    assert ledger["facts_sha256"] == "d" * 64
    assert ledger["manifest_sha256"] == "e" * 64
    assert ledger["fact_row_count"] == 107_550
    assert ledger["manifest"] == {
        "artifact_id": "chronicle-uk-artifact-1cab809",
        "profile": "uk-national",
        "schema_version": 1,
    }
    # A bare feed carries no manifest, and that absence is recorded rather
    # than invented.
    assert calibration_run._ledger_provenance(object())["manifest_sha256"] is None
