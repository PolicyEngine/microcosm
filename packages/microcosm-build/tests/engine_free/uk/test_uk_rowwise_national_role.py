"""The UK build driver's national release role (microcosm#823) on the graph.

The national line solves the bound spine checkpoint as bound through
``uk_runtime.graph_national`` and materialises the seam-shaped evidence
(``build_record.json``, the signed calibration-seam gate report, the seam
diagnostics, the frozen registries, the schema-4 national manifest). These
tests run the driver end to end on the calibration seam suite's synthetic
frame and register (loaded from that module): the register materialises from
the frame's own columns (no engine), the Chronicle artifact is stood in for
on ``full_targets``, and the Hub is the candidate suite's fake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.build.logbook import load_spool_rows
from microcosm.build.staging_v2 import validate_v2_bundle
from microcosm.build.uk_runtime import (
    calibration_run,
    full_targets,
    graph_national,
    national_role,
    rowwise_staging,
    spine_build,
)
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
)
from microcosm.build.uk_runtime.chronicle_feed import (
    load_uk_chronicle_feed,
    require_committed_uk_chronicle_feed_pin,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.national_calibration import (
    CalibrationFrameAdapter,
    national_calibration_mass_reason,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UKNationalSolveDoctrine,
    uk_national_target_loss_weights,
)
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.release_identity import UK_NATIONAL_RELEASE_ID
from microcosm.calibrate import TargetRegistry, calibrate
from microcosm.frame import WeightKind
from test_support.microcosm_build import uk_calibration_run as seam
from test_support.microcosm_build import uk_rowwise_candidate as candidate
from test_support.microcosm_build.uk_calibration_run import (  # noqa: F401
    _cgt_projection,
    _signing_key,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_spine(tmp_path: Path, frame) -> Path:
    """The seam fixture frame as a bound checkpoint: H5, sidecar, gate report."""
    input_h5 = tmp_path / "spine.h5"
    write_uk_national_frame(frame, input_h5)
    report = {
        **calibration_run.uk_spine_checkpoint_gate_digests(),
        "blocked_at_phase": None,
        "gates": {
            gate_id: {"status": "passed", "criticality": "release_blocking"}
            for gate_id in calibration_run.UK_SPINE_GATE_SCOPE
        },
    }
    report_path = input_h5.with_suffix(".spine_gates.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sidecar = {
        "schema_version": 2,
        "pipeline": "uk-frs-spine",
        "stages": ["frs_spine", "was_wealth"],
        "stage_records": [
            {
                "stage": "was_wealth",
                "produced": ["property_wealth"],
                "nonzero_share": {"property_wealth": 1.0},
                "seconds": 0.1,
            }
        ],
        "stage_evidence": {
            "was_wealth": {"stage": "was_wealth", "support_clip": {"columns": {}}}
        },
        "artifact_pins": {"person": "a" * 64},
        "input_artifact_pins": {"was_qrf_donor": {"sha256": "b" * 64}},
        "resource_pins": {"wealth.json": "c" * 64},
        "stage_artifact_pins": {"was_wealth": {"was_qrf_donor": "d" * 64}},
        "declared_seeds": {"was_wealth": {"was_wealth": 0}},
        "rules_engine": {"package": "policyengine-uk", "version": "unavailable"},
        "source_vintages": {"frs": "2024_25"},
        "stochastic_contract_sha256": "e" * 64,
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
        "household_weight_kind": frame.weights_for("household").kind.value,
        "household_weight_total": float(frame.weights_for("household").values.sum()),
        "uk_frame_content_identity": uk_frame_content_identity(frame),
        "spine_gate_report": {"sha256": _sha(report_path)},
        "fit_weight_records": {},
    }
    input_h5.with_suffix(".build.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    )
    return input_h5


def _national_inputs(monkeypatch, tmp_path: Path, *, registry=None):
    """The synthetic national build: bound frame, stood-in Chronicle artifact,
    no engine (the register materialises from the frame's own columns)."""
    frame = seam._frame()
    input_h5 = _bound_spine(tmp_path, frame)
    registry = seam._registry() if registry is None else registry
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "consumer_facts.jsonl").write_text("{}\n")
    artifact = SimpleNamespace(
        facts=None,
        facts_sha256="1" * 64,
        manifest_sha256="2" * 64,
        path=ledger_dir,
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-national-fixture",
        },
    )
    pin = SimpleNamespace(
        to_dict=lambda: {"facts_sha256": "1" * 64, "manifest_sha256": "2" * 64}
    )
    monkeypatch.setattr(
        full_targets, "load_ledger_consumer_artifact", lambda path, **kwargs: artifact
    )
    monkeypatch.setattr(
        full_targets,
        "require_committed_uk_chronicle_feed_pin",
        lambda facts_sha256, **kwargs: pin,
    )
    monkeypatch.setattr(
        full_targets,
        "compile_uk_target_registry",
        lambda facts, target_period: SimpleNamespace(registry=registry, unsupported=()),
    )
    monkeypatch.setattr(
        full_targets, "load_uk_calibration_measure_exclusions", lambda p: ()
    )
    monkeypatch.setattr(
        full_targets,
        "apply_uk_calibration_measure_exclusions",
        lambda reg, exclusions, now=None: (reg, {}),
    )
    monkeypatch.setattr(
        full_targets,
        "_ledger_provenance",
        lambda artifact: {"artifact_id": "synthetic-national-fixture"},
    )
    # No engine: the national problem node takes the seam's frame-only route.
    monkeypatch.setattr(graph_national, "UKMeasureResolver", None)
    monkeypatch.setattr(spine_build, "_rules_engine", lambda: None)
    monkeypatch.setattr(
        spine_build,
        "_rules_engine_provenance",
        lambda: {"package": "policyengine-uk", "version": "test"},
    )
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (seam._admin_anchor_values(), []),
    )
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", seam.SIGNING_KEY)
    return input_h5, registry, artifact, pin


def _argv(input_h5: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--input-h5",
        str(input_h5),
        "--release-role",
        "national",
        "--input-sha256",
        _sha(input_h5),
        "--ledger-facts",
        str(input_h5.parent / "ledger"),
        "--ledger-facts-sha256",
        "1" * 64,
        "--ledger-manifest-sha256",
        "2" * 64,
        "--out",
        str(out),
        *extra,
    ]


def _sums_verify(directory: Path) -> None:
    for line in (directory / "sha256sums.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert _sha(directory / name) == digest, name


def _seam_weights(frame, registry, doctrine: UKNationalSolveDoctrine) -> np.ndarray:
    """An in-process calibrate call under the national doctrine: the seam's solve."""
    from microcosm.build.uk_runtime.ledger_targets import materialize_uk_ledger_targets

    adapter = CalibrationFrameAdapter(frame)
    materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=registry
    )
    families = [spec.family for spec in registry.specs]
    result = calibrate(
        adapter.prepared_frame(),
        registry.to_target_set(),
        weight_entity="household",
        epochs=doctrine.epochs,
        learning_rate=doctrine.learning_rate,
        mass=doctrine.mass_rule,
        mass_reason=national_calibration_mass_reason(families),
        max_weight_ratio=doctrine.max_weight_ratio,
        seed=doctrine.seed,
        l0_lambda=doctrine.l0_lambda,
        target_loss_cap=doctrine.target_loss_cap,
        target_loss_weights=uk_national_target_loss_weights(
            families, rule=doctrine.target_weight_rule
        ),
    )
    return result.weights


def test_uk_national_role_builds_through_the_graph_and_stages_locally(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, registry, _artifact, pin = _national_inputs(monkeypatch, tmp_path)
    out = tmp_path / "national"

    assert (
        builder.main(_argv(input_h5, out, "--staging-local-only", "--epochs", "5")) == 0
    )

    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert json.loads(capsys.readouterr().out) == manifest
    run_id = manifest["build_id"]
    assert run_id.startswith("uk-frs-calibration-attempt-")
    # The attempt records how it ran its graph and names itself in its request
    # evidence, so a later attempt on the same store can link back to it.
    execution = manifest["execution"]
    assert execution["nodes_total"] > 0 and execution["nodes_reused"] == 0
    assert execution["earlier_attempts"] == []
    request = json.loads(
        (Path(execution["attempt_directory"]) / "request.json").read_text()
    )
    assert request["attempt"]["build_id"] == run_id
    # The seam's output names, from the posture.
    dataset = out / "microcosm_uk_2024_25.h5"
    gates = out / "microcosm_uk_2024_25.terminal_gates.json"
    diagnostics = out / "calibration_diagnostics.json"
    for path in (
        dataset,
        gates,
        diagnostics,
        out / "build_record.json",
        out / "national_target_registry.json",
        out / "national_contract_registry.json",
        out / "target_support_manifest.json",
        out / "build.json",
        out / "graph.json",
        out / "operations.json",
        out / "build.graph.json",
    ):
        assert path.is_file(), path.name
    assert not (out / "failure.json").exists()

    # The build record the release-cut certifier reads: the seam's schema.
    record = json.loads((out / "build_record.json").read_text())
    assert record["schema_version"] == 1
    assert record["build_id"] == run_id
    assert record["pipeline"] == "uk-frs-calibration"
    run_config = record["run_config"]
    assert run_config["pipeline"] == "uk-frs-calibration"
    assert run_config["release_id"] == UK_NATIONAL_RELEASE_ID
    assert run_config["release_role"] == "national"
    assert run_config["register_sha256"] == registry.version
    assert run_config["band_edge_register_sha256"] == registry.version
    assert run_config["calibration_year"] == 2025
    assert run_config["doctrine_overrides"] == {
        "epochs": {"default": 1500, "effective": 5}
    }
    assert run_config["doctrine"]["target_weight_rule"] == "family_equal"
    assert run_config["doctrine"]["seed"] == 0
    assert run_config["allow_unpinned_feed"] is False
    assert run_config["chronicle_feed_pin"] == pin.to_dict()
    assert run_config["rowwise_driver_parameters"]["n_clones"] is None
    assert record["source_pins"]["input_h5"]["sha256"] == _sha(input_h5)
    assert record["input_posture"] == {
        "tier": "staging_candidate",
        "sha256": _sha(input_h5),
        "size_bytes": input_h5.stat().st_size,
    }
    assert record["register"] == {
        "country": "uk",
        "version": registry.version,
        "compiled_count": 1,
        "excluded_count": 0,
        "calibrated_count": 1,
    }
    assert record["artifacts"]["staging_h5"]["sha256"] == _sha(dataset)
    assert record["artifacts"]["staging_h5"]["path"] == str(dataset)
    assert record["artifacts"]["diagnostics_json"]["sha256"] == _sha(diagnostics)
    assert record["artifacts"]["terminal_gate_json"]["sha256"] == _sha(gates)
    assert record["certification"] == {
        "expected_artifact": str(dataset.with_suffix(".release_certification.json")),
        "producer": "tools/certify_uk_release_cut.py",
    }
    provenance = record["spine_provenance"]
    assert provenance["spine_gate_report"]["sha256"] == _sha(
        input_h5.with_suffix(".spine_gates.json")
    )
    assert provenance["stages"] == ["frs_spine", "was_wealth"]
    assert provenance["artifact_pins"] == {"person": "a" * 64}
    calibration = record["calibration"]
    assert calibration["solve"]["n_targets"] == 1
    assert calibration["solve"]["n_households"] == 4
    assert calibration["weights"]["household_weight_kind"] == "calibrated"
    assert calibration["parameters"]["doctrine"]["epochs"] == 5
    assert record["gate_summary"] == {
        gate_id: "passed" for gate_id in UK_CALIBRATION_GATE_SCOPE
    }
    assert record["staging_delivery"]["mode"] == "local_only"
    assert set(record["graph"]["artifacts"]) >= {
        "uk.full.spine_checkpoint",
        "uk.full.national_targets",
        "uk.full.national_problem",
        "uk.full.dense",
        "uk.full.gates.calibrated",
        "uk.full.national.readback",
    }
    assert record["graph"]["readback"]["passed"] is True

    # The seam diagnostics with the build block, digest measured from the file.
    written = json.loads(diagnostics.read_text())
    assert written["build"]["build_id"] == run_id
    assert written["build"]["spine_provenance"] == provenance
    assert written["build"]["doctrine"]["epochs"] == 5

    # The signed calibration-seam gate report.
    report = json.loads(gates.read_text())
    assert report["posture"] == "calibration_seam"
    assert set(report["scope_exclusions"]) == set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
    assert set(report["gates"]) == set(UK_CALIBRATION_GATE_SCOPE)
    assert report["blocked_at_phase"] is None
    assert report["release_candidate"] is False
    assert report["release_evidence"] == {
        "calibration_diagnostics_sha256": _sha(diagnostics)
    }
    attestation = report["attestation"]
    signature = attestation["signature"]
    attestation["signature"] = None
    key = b"0123456789abcdef0123456789abcdef"
    import hmac

    from microcosm.build.gate_battery import _canonical_json_bytes

    assert (
        hmac.new(key, _canonical_json_bytes(report), hashlib.sha256).hexdigest()
        == signature
    )

    # The H5: calibrated weights, the seam's mass record.
    staged, _ = load_uk_national_frame(dataset)
    assert staged.weights_for("household").kind is WeightKind.CALIBRATED
    assert len(staged.mass_log) == 1
    assert staged.mass_log[0].reason == national_calibration_mass_reason(
        [spec.family for spec in registry.specs]
    )
    frozen = TargetRegistry.from_json(out / "national_target_registry.json")
    assert frozen.version == registry.version

    # The schema-4 national manifest.
    assert manifest["schema_version"] == 4
    assert manifest["build_kind"] == "uk_national_calibrated_candidate"
    assert manifest["release_role"] == "national"
    assert manifest["release_id"] == UK_NATIONAL_RELEASE_ID
    assert set(manifest["outputs"]) == {
        "dataset",
        "calibration_diagnostics",
        "build_record",
        "terminal_gate_report",
        "national_target_registry",
        "national_contract_registry",
    }
    assert manifest["outputs"]["dataset"]["sha256"] == _sha(dataset)
    assert manifest["outputs"]["dataset"]["path"] == str(dataset)
    assert manifest["outputs"]["build_record"]["sha256"] == _sha(
        out / "build_record.json"
    )
    assert manifest["build_record"]["sha256"] == _sha(out / "build_record.json")
    assert manifest["outputs"]["national_contract_registry"]["sha256"] == _sha(
        out / "national_contract_registry.json"
    )
    assert manifest["evaluation"]["status"] == "not_requested"
    assert manifest["solve"]["n_targets_by_kind"] == {
        "national": 1,
        "local": 0,
        "ladder": 0,
    }
    assert (
        manifest["solve"]["pool_households"] == manifest["solve"]["n_households"] == 4
    )
    assert manifest["parameters"]["n_clones"] is None
    assert manifest["parameters"]["doctrine"]["clone_count"] is None
    assert manifest["parameters"]["doctrine"]["target_weight_rule"] == "family_equal"
    assert manifest["fit"]["national_by_family"][0]["family"] == "dwp_universal_credit"
    assert manifest["gate"]["scope"] == list(UK_CALIBRATION_GATE_SCOPE)
    assert manifest["gate"]["posture"] == "calibration_seam"
    assert set(manifest["gate"]["statuses"]) == set(UK_CALIBRATION_GATE_SCOPE)
    assert manifest["failing_gate_ids"] == []
    assert manifest["releasable"] is False
    assert manifest["release_posture"]["calibration_seam_gates_passed"] is True
    assert (
        manifest["release_posture"]["shippable_by"] == "tools/certify_uk_release_cut.py"
    )
    assert manifest["staged_dataset"]["status"] == "skipped"
    assert manifest["staging_delivery"] == record["staging_delivery"]
    assert manifest["graph"]["readback"]["passed"] is True
    _sums_verify(out)

    bundle = validate_v2_bundle(out / "staging", run_id)
    run_manifest = bundle["run_manifest"]
    assert run_manifest["operation_id"] == "uk_national_calibration"
    assert run_manifest["pipeline"]["id"] == "uk-frs-calibration"
    assert run_manifest["run_kind"] == "calibration"
    assert bundle["progress"]["status"] == "completed"
    stage_ids = {event["stage_id"] for event in bundle["events"]}
    assert {
        "input_pinning",
        "input_loading",
        "target_compilation",
        "calibration",
        "diagnostics",
        "target_support_sidecars",
        "release_check_evaluation",
        "candidate_h5_creation",
        "build_record_creation",
        "dataset_staging",
        "complete",
    } <= stage_ids
    artifacts = out / "staging" / "runs" / run_id / "artifacts"
    fit_summary = json.loads((artifacts / "fit_summary.json").read_text())
    assert fit_summary["build_kind"] == "uk_national_calibrated_candidate"
    assert fit_summary["fit_by_family"]["local"] == {}
    assert "dwp_universal_credit" in fit_summary["fit_by_family"]["national"]

    rows = load_spool_rows(out / "logbook-spool")
    assert len(rows) == 1
    assert rows[0].build_id == run_id
    assert rows[0].pipeline == "uk-frs-calibration"
    assert rows[0].disposition == "iterating"
    phases = list(rows[0].phases_reached)
    for phase in (
        "input_sidecar_bound",
        "targets_bound",
        "national_calibration_solved",
        "diagnostics_written",
        "calibration_gates_evaluated",
        "staging_h5_written",
        "build_record_written",
        "published",
    ):
        assert phase in phases, phase


def test_uk_national_role_solve_is_the_seam_solve(monkeypatch, tmp_path):
    """The graph's H5 carries the weights an in-process calibrate call under
    the national doctrine produces on the same frame and register."""
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    out = tmp_path / "national"

    assert builder.main(_argv(input_h5, out, "--no-staging", "--epochs", "7")) == 0

    staged, _ = load_uk_national_frame(out / "microcosm_uk_2024_25.h5")
    expected = _seam_weights(seam._frame(), registry, UKNationalSolveDoctrine(epochs=7))
    assert np.array_equal(staged.weights_for("household").values, expected)


def test_uk_national_role_records_doctrine_overrides(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    out = tmp_path / "override"

    assert (
        builder.main(
            _argv(
                input_h5,
                out,
                "--no-staging",
                "--epochs",
                "5",
                "--target-weight-rule",
                "uniform",
                "--target-loss-cap",
                "4",
            )
        )
        == 0
    )

    record = json.loads((out / "build_record.json").read_text())
    assert record["run_config"]["doctrine_overrides"] == {
        "epochs": {"default": 1500, "effective": 5},
        "target_loss_cap": {"default": 10.0, "effective": 4.0},
        "target_weight_rule": {"default": "family_equal", "effective": "uniform"},
    }
    assert record["run_config"]["doctrine"]["target_weight_rule"] == "uniform"
    assert record["run_config"]["doctrine"]["target_loss_cap"] == 4.0
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["solve"]["target_weight_rule"] == "uniform"
    assert manifest["solve"]["target_weight_rule_override"] == {
        "default": "family_equal",
        "effective": "uniform",
    }
    operations = json.loads((out / "operations.json").read_text())
    problem = next(
        node for node in operations["nodes"] if node["id"] == "uk.full.national_problem"
    )
    assert problem["kernel"] == "uk.full.national_problem@1"
    assert operations["configuration"]["doctrine"]["target_loss_cap"] == 4.0


def test_uk_national_role_publishes_telemetry_and_the_bundle_to_the_hub(
    monkeypatch, tmp_path
):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    hub = candidate._FakeHub()
    monkeypatch.setattr(rowwise_staging, "_hub_api", lambda: hub)
    monkeypatch.setattr(rowwise_staging, "_hub_token", lambda: "hf_test_token")
    out = tmp_path / "national"

    assert (
        builder.main(
            _argv(
                input_h5, out, "--epochs", "5", "--staging-upload-interval-seconds", "0"
            )
        )
        == 0
    )

    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    run_id = manifest["build_id"]
    assert run_id.startswith("uk-frs-calibration-attempt-")
    telemetry_paths = hub.paths("policyengine/populace-uk-staging")
    assert telemetry_paths == sorted(
        f"runs/{run_id}/{name}"
        for name in (
            "run_manifest.json",
            "progress.json",
            "events.ndjson",
            "calibration_progress.json",
            "artifacts/fit_summary.json",
            "artifacts/staged_dataset.json",
        )
    )
    remote_progress = json.loads(
        hub.files[("policyengine/populace-uk-staging", f"runs/{run_id}/progress.json")]
    )
    assert remote_progress["status"] == "completed"
    remote_manifest = json.loads(
        hub.files[
            ("policyengine/populace-uk-staging", f"runs/{run_id}/run_manifest.json")
        ]
    )
    assert remote_manifest["operation_id"] == "uk_national_calibration"
    delivery = manifest["staging_delivery"]
    assert delivery["mode"] == "local_and_remote"
    assert delivery["upload_successes"] == delivery["upload_attempts"] > 0
    assert delivery["last_error_code"] is None

    assert len(hub.commits) == 1
    commit = hub.commits[0]
    assert commit["repo_id"] == "policyengine/populace-uk-private"
    expected = {Path(entry["path"]).name for entry in manifest["outputs"].values()} | {
        builder.MANIFEST_FILENAME,
        "staged_manifest.json",
        "sha256sums.txt",
    }
    assert commit["paths"] == sorted(f"staged/{run_id}/{name}" for name in expected)
    assert "microcosm_uk_2024_25.h5" in expected and "build_record.json" in expected
    staged = manifest["staged_dataset"]
    assert staged["status"] == "uploaded"
    assert staged["repository"] == "policyengine/populace-uk-private"
    assert staged["prefix"] == f"staged/{run_id}"
    remote_h5 = hub.files[
        ("policyengine/populace-uk-private", f"staged/{run_id}/microcosm_uk_2024_25.h5")
    ]
    assert remote_h5 == (out / "microcosm_uk_2024_25.h5").read_bytes()
    record = json.loads((out / "build_record.json").read_text())
    assert record["staging_delivery"] == delivery


def test_uk_national_dry_run_prints_the_plan_and_writes_nothing(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    out = tmp_path / "national"

    assert builder.main(_argv(input_h5, out, "--dry-run")) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["build_kind"] == "uk_national_calibrated_candidate_plan"
    assert plan["release_role"] == "national"
    assert plan["release_id"] == UK_NATIONAL_RELEASE_ID
    assert plan["dry_run"] is True
    assert plan["targets"]["active"] == plan["targets"]["compiled"] == 1
    assert plan["targets"]["register_sha256"] == registry.version
    assert plan["doctrine"]["epochs"] == 1500
    assert plan["doctrine"]["target_weight_rule"] == "family_equal"
    assert plan["doctrine_overrides"] == {}
    assert plan["parameters"]["release_role"] == "national"
    assert plan["engine"] == "not_run"
    assert plan["releasable"] is False
    # The compiled graph's inventory rides the plan.
    assert [node["id"] for node in plan["graph"]["nodes"]] == [
        "uk.full.spine_checkpoint",
        "uk.full.national_targets",
        "uk.full.national_problem",
        "uk.full.dense",
        "uk.full.calibrated",
        "uk.full.gates.calibrated",
    ]
    assert not out.exists()


def test_uk_national_role_marks_the_staging_run_failed_on_a_refusal(
    monkeypatch, tmp_path
):
    """A register that does not compile refuses inside the targets node; the
    driver records the failure sidecar, the failed row and the failed run."""
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        full_targets,
        "compile_uk_target_registry",
        lambda facts, target_period: SimpleNamespace(
            registry=None, unsupported=("dwp.uc.households",)
        ),
    )
    out = tmp_path / "national"
    assert builder.main(_argv(input_h5, out, "--staging-local-only")) == 1
    failure = json.loads((out / "failure.json").read_text())
    assert "failed to compile" in failure["message"]
    runs = sorted(path.name for path in (out / "staging" / "runs").iterdir())
    assert len(runs) == 1
    bundle = validate_v2_bundle(out / "staging", runs[0])
    assert bundle["progress"]["status"] == "failed"
    assert not (out / "build_record.json").exists()
    rows = load_spool_rows(out / "logbook-spool")
    assert len(rows) == 1
    assert rows[0].disposition == "failed"
    assert rows[0].pipeline == "uk-frs-calibration"


def test_uk_national_role_refuses_an_occupied_output_directory(monkeypatch, tmp_path):
    """The seam refused to overwrite an existing candidate artifact; the graph
    driver refuses it with the other argument refusals, before the attempt
    opens, so a second run into the same ``--out`` leaves the first candidate's
    directory exactly as it was: no failure sidecar, no second Logbook row
    (microcosm#1057 review round 2, item 9)."""
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    out = tmp_path / "national"
    assert builder.main(_argv(input_h5, out, "--no-staging", "--epochs", "5")) == 0
    before = {
        str(path.relative_to(out)): _sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    head = load_spool_rows(out / "logbook-spool")[0].row_digest

    with pytest.raises(
        FileExistsError, match="refusing to overwrite existing candidate artifact"
    ):
        builder.main(
            _argv(
                input_h5,
                out,
                "--no-staging",
                "--epochs",
                "5",
                "--logbook-prev-row-digest",
                head,
            )
        )

    after = {
        str(path.relative_to(out)): _sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert not (out / "failure.json").exists()
    rows = load_spool_rows(out / "logbook-spool")
    assert [row.disposition for row in rows] == ["iterating"]


def test_uk_national_role_blocks_on_a_failed_seam_gate(monkeypatch, tmp_path):
    """A blocking calibration-seam outcome stops the artifact after the
    battery, block included, is on disk: no H5, no build record, a failed row."""
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (
            {name: value * 5.0 for name, value in seam._admin_anchor_values().items()},
            [],
        ),
    )
    out = tmp_path / "blocked"

    assert builder.main(_argv(input_h5, out, "--no-staging", "--epochs", "5")) == 1

    gates = json.loads((out / "microcosm_uk_2024_25.terminal_gates.json").read_text())
    assert gates["blocked_at_phase"] == "terminal"
    assert gates["gates"]["uk_aggregate_admin"]["status"] == "failed"
    assert not (out / "microcosm_uk_2024_25.h5").exists()
    assert not (out / "build_record.json").exists()
    assert (out / "calibration_diagnostics.json").is_file()
    assert (out / "target_support_manifest.json").is_file()
    completion = json.loads((out / "build.json").read_text())
    assert completion["kind"] == "uk_national_build_refused"
    rows = load_spool_rows(out / "logbook-spool")
    assert rows[0].disposition == "failed"
    assert "candidate_blocked" in rows[0].phases_reached
    assert rows[0].gate_verdicts["uk_aggregate_admin"]["verdict"] == "failed"


def test_uk_national_role_closes_a_seam_block_as_a_blocked_run(monkeypatch, tmp_path):
    """The seam battery's refusal closes the staging run ``blocked`` at the
    terminal phase, naming the refusing gate, while the row stays ``failed``."""
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (
            {name: value * 5.0 for name, value in seam._admin_anchor_values().items()},
            [],
        ),
    )
    out = tmp_path / "blocked"

    assert (
        builder.main(_argv(input_h5, out, "--staging-local-only", "--epochs", "5")) == 1
    )

    runs = sorted(path.name for path in (out / "staging" / "runs").iterdir())
    bundle = validate_v2_bundle(out / "staging", runs[0])
    assert bundle["progress"]["status"] == "blocked"
    assert bundle["progress"]["block"]["phase"] == "terminal"
    assert "uk_aggregate_admin" in bundle["progress"]["block"]["blocking_gate_ids"]
    checks = [
        e for e in bundle["events"] if e["stage_id"] == "release_check_evaluation"
    ]
    assert checks[-1]["details"]["gate_statuses"]["uk_aggregate_admin"] == "failed"
    assert load_spool_rows(out / "logbook-spool")[0].disposition == "failed"


# --- Behaviours re-anchored from the retired seam command's tests (B3) ---


def test_uk_national_role_has_no_release_id_flag(tmp_path) -> None:
    """Canonical ids are the role's own; the seam's --release-id is gone."""

    builder = candidate._load_builder_module()
    (tmp_path / "spine.h5").write_bytes(b"spine")
    with pytest.raises(SystemExit):
        builder._parse_args(
            _argv(tmp_path / "spine.h5", tmp_path / "out", "--release-id", "dev-x")
        )


def test_uk_national_role_accepts_operator_exclusions_and_refuses_a_bad_sha(
    tmp_path,
) -> None:
    builder = candidate._load_builder_module()
    (tmp_path / "spine.h5").write_bytes(b"spine")
    exclusions = tmp_path / "operator.json"
    exclusions.write_text("{}", encoding="utf-8")
    argv = _argv(tmp_path / "spine.h5", tmp_path / "out")
    argv[argv.index("--input-sha256") + 1] = "0" * 64
    parsed = builder._parse_args([*argv, "--measure-exclusions", str(exclusions)])
    builder._validate_cli_args(parsed)
    assert parsed.measure_exclusions == exclusions
    argv[argv.index("--input-sha256") + 1] = "not-a-sha"
    with pytest.raises(SystemExit):
        builder._parse_args(argv)


def test_uk_national_role_exposes_the_shared_staging_modes(tmp_path) -> None:
    builder = candidate._load_builder_module()
    (tmp_path / "spine.h5").write_bytes(b"spine")
    argv = _argv(tmp_path / "spine.h5", tmp_path / "out")
    argv[argv.index("--input-sha256") + 1] = "0" * 64
    remote = builder._parse_args(argv)
    local = builder._parse_args([*argv, "--staging-local-only"])
    disabled = builder._parse_args([*argv, "--no-staging"])
    assert remote.staging_repo_id == "policyengine/populace-uk-staging"
    assert not remote.staging_local_only and not remote.no_staging
    assert local.staging_local_only and not local.no_staging
    assert disabled.no_staging
    with pytest.raises(SystemExit):
        builder._parse_args([*argv, "--staging-repo-id", ""])
    with pytest.raises(SystemExit):
        builder._parse_args([*argv, "--staging-local-only", "--staging-read-back"])


def _foreign_artifact(ledger_dir: Path, *, facts_sha256: str, manifest_sha256):
    return SimpleNamespace(
        facts=None,
        facts_sha256=facts_sha256,
        manifest_sha256=manifest_sha256,
        path=ledger_dir,
        provenance=lambda: {
            "facts_sha256": facts_sha256,
            "manifest_sha256": manifest_sha256,
            "artifact_id": "foreign",
        },
    )


@pytest.mark.parametrize("manifest_sha256", ["c" * 64, None])
def test_uk_national_role_refuses_a_feed_outside_the_committed_pin(
    monkeypatch, tmp_path, manifest_sha256
) -> None:
    """The committed pin is checked before the register compiles."""

    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    # The fixture stubs the check; this test wants the real one.
    monkeypatch.setattr(
        full_targets,
        "require_committed_uk_chronicle_feed_pin",
        require_committed_uk_chronicle_feed_pin,
    )
    pin = load_uk_chronicle_feed()
    artifact = _foreign_artifact(
        tmp_path / "ledger",
        facts_sha256=pin.facts_sha256,
        manifest_sha256=manifest_sha256,
    )
    monkeypatch.setattr(
        full_targets, "load_ledger_consumer_artifact", lambda path, **kwargs: artifact
    )

    def compile_must_not_run(facts, target_period):
        raise AssertionError("the register compiled before the feed pin was checked")

    monkeypatch.setattr(
        full_targets, "compile_uk_target_registry", compile_must_not_run
    )
    out = tmp_path / "national"
    assert builder.main(_argv(input_h5, out, "--staging-local-only")) == 1
    failure = json.loads((out / "failure.json").read_text())
    assert "manifest: loaded" in failure["message"]
    # The refusal happened after telemetry opened: the local run reads failed.
    runs = sorted(path.name for path in (out / "staging" / "runs").iterdir())
    assert (
        validate_v2_bundle(out / "staging", runs[0])["progress"]["status"] == "failed"
    )


def test_uk_national_role_records_an_unpinned_feed_override(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        full_targets,
        "require_committed_uk_chronicle_feed_pin",
        require_committed_uk_chronicle_feed_pin,
    )
    artifact = _foreign_artifact(
        tmp_path / "ledger", facts_sha256="0" * 64, manifest_sha256="c" * 64
    )
    monkeypatch.setattr(
        full_targets, "load_ledger_consumer_artifact", lambda path, **kwargs: artifact
    )
    out = tmp_path / "national"
    argv = _argv(
        input_h5, out, "--no-staging", "--epochs", "5", "--allow-unpinned-feed"
    )
    argv[argv.index("--ledger-facts-sha256") + 1] = "0" * 64
    argv[argv.index("--ledger-manifest-sha256") + 1] = "c" * 64
    assert builder.main(argv) == 0
    record = json.loads((out / "build_record.json").read_text())
    assert record["run_config"]["allow_unpinned_feed"] is True
    # The pin the artifact was checked against is the committed one.
    assert record["run_config"]["chronicle_feed_pin"]["facts_sha256"] != "0" * 64
    operations = json.loads((out / "operations.json").read_text())
    assert operations["configuration"]["allow_unpinned_feed"] is True


# --- the end-of-build incumbent evaluation (microcosm#823 E2) -----------------


def _fake_evaluator(receipt_of, calls: list[dict]):
    """A stand-in for microcosm.build.uk_runtime.candidate_score (#967)."""

    class Module:
        @staticmethod
        def evaluate_uk_candidate_against_incumbent(**kwargs):
            calls.append(kwargs)
            return receipt_of(kwargs)

        @staticmethod
        def uk_default_measure_resolver_factory(scratch_dir, year):
            return lambda path, frame: None

        @staticmethod
        def pruned_warning(score):
            pruned = score.get("incumbent_unresolvable_pruned") or {}
            if not pruned.get("n_pruned"):
                return None
            return "warning: pruned from BOTH arms: " + ", ".join(pruned["measures"])

    return Module


def _receipt(kwargs, *, verdict="passed", pruned_measures=()):
    n_pruned = len(pruned_measures)
    return {
        "artifacts": {
            "candidate": {"sha256": kwargs["candidate_sha256"]},
            "incumbent": {
                "sha256": kwargs["incumbent_sha256"],
                "label": kwargs["incumbent_label"],
            },
        },
        "candidate_full_loss": 0.01,
        "incumbent_full_loss": 0.2,
        "candidate_target_wins": 1,
        "incumbent_target_wins": 0,
        "register": {"n_specs": 1},
        # Record arrays the reviewed telemetry artifact must not carry.
        "target_drift": [
            {
                "target": "dwp.uc.households@2025",
                "family": "dwp_universal_credit",
                "candidate_relative_error": 0.01,
                "incumbent_relative_error": 0.2,
                "winner": "candidate",
            }
        ],
        "signed_asymmetries": [{"id": "incumbent_own_registry", "description": "…"}],
        "measure_resolution": {
            "candidate": {"rounds": [{"round": 0}]},
            "incumbent": None,
        },
        "incumbent_unresolvable_pruned": {
            "n_pruned": n_pruned,
            "n_scored": 1,
            "n_surface": 1 + n_pruned,
            "pruned_targets": {},
            "measures": list(pruned_measures),
            "families": {"dwp_universal_credit": n_pruned} if n_pruned else {},
            "note": "pruned",
        },
        "evaluation": {
            "schema_version": 1,
            "rule": "rule 1",
            "scored_surface": {
                "n_scored": 1,
                "n_pruned": n_pruned,
                "n_surface": 1 + n_pruned,
            },
            "rule_1": {
                "passed": verdict == "passed",
                "candidate_full_loss": 0.01,
                "incumbent_full_loss": 0.2,
                "candidate_target_wins": 1,
                "incumbent_target_wins": 0,
            },
            "verdict": verdict,
        },
    }


def _incumbent_argv(input_h5: Path, tmp_path: Path, *extra: str) -> list[str]:
    incumbent = tmp_path / "incumbent.h5"
    shutil.copyfile(input_h5, incumbent)
    return [
        "--incumbent-h5",
        str(incumbent),
        "--incumbent-sha256",
        _sha(incumbent),
        "--incumbent-label",
        "efrs_fixture",
        *extra,
    ]


def test_uk_dense_role_refuses_the_incumbent_flags(tmp_path) -> None:
    builder = candidate._load_builder_module()
    args = argparse.Namespace(
        target_loss_cap=None,
        allow_unpinned_feed=False,
        target_weight_rule="grain_equal",
        incumbent_h5=tmp_path / "incumbent.h5",
        incumbent_sha256=None,
    )
    with pytest.raises(ValueError, match="--incumbent-h5/--incumbent-sha256"):
        builder._refuse_national_role_arguments(
            args, builder.uk_rowwise_posture("dense")
        )


def test_uk_national_role_requires_the_incumbent_pair(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    out = tmp_path / "out"
    assert (
        builder.main(
            _argv(input_h5, out, "--no-staging", "--incumbent-h5", str(input_h5))
        )
        == 1
    )
    failure = json.loads((out / "failure.json").read_text())
    assert "must be given together" in failure["message"]
    assert not (out / "microcosm_uk_2024_25.h5").exists()


def test_uk_national_role_refuses_the_incumbent_without_the_scorer() -> None:
    def missing(name: str):
        raise ImportError(name)

    with pytest.raises(ValueError, match="microcosm#967"):
        national_role._load_candidate_evaluator(importer=missing)


def test_uk_national_role_evaluates_against_the_incumbent(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    calls: list[dict] = []
    module = _fake_evaluator(
        lambda kwargs: _receipt(
            kwargs, pruned_measures=("benunit.uc_calibration_child_count",)
        ),
        calls,
    )
    monkeypatch.setattr(
        national_role, "_load_candidate_evaluator", lambda importer=None: module
    )
    out = tmp_path / "national"

    assert (
        builder.main(
            _argv(
                input_h5,
                out,
                "--no-staging",
                "--epochs",
                "5",
                *_incumbent_argv(input_h5, tmp_path),
            )
        )
        == 0
    )

    call = calls[0]
    assert call["candidate_h5"] == out / "microcosm_uk_2024_25.h5"
    assert call["candidate_sha256"] == _sha(out / "microcosm_uk_2024_25.h5")
    assert call["incumbent_h5"] == tmp_path / "incumbent.h5"
    assert call["incumbent_sha256"] == _sha(tmp_path / "incumbent.h5")
    assert call["incumbent_label"] == "efrs_fixture"
    # The registries the scorer receives are the ones the graph solved and
    # froze, by content hash.
    assert call["target_registry"].version == registry.version
    assert call["band_edge_registry"].version == registry.version
    assert call["calibration_year"] == 2025
    receipt_path = out / "score_vs_incumbent.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["evaluation"]["verdict"] == "passed"
    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    evaluation = manifest["evaluation"]
    assert evaluation["status"] == "completed"
    assert evaluation["verdict"] == "passed"
    assert evaluation["rule_1"]["passed"] is True
    assert evaluation["scored_surface"] == {
        "n_scored": 1,
        "n_pruned": 1,
        "n_surface": 2,
    }
    assert evaluation["pruned_measures"] == ["benunit.uc_calibration_child_count"]
    assert evaluation["pruned_families"] == {"dwp_universal_credit": 1}
    assert evaluation["receipt"]["sha256"] == _sha(receipt_path)
    assert evaluation["incumbent"]["label"] == "efrs_fixture"
    assert evaluation["incumbent"]["sha256"] == _sha(tmp_path / "incumbent.h5")
    assert manifest["outputs"]["score_receipt"]["sha256"] == _sha(receipt_path)
    # The absent measure is named loudly on stderr, never hidden.
    assert "benunit.uc_calibration_child_count" in capsys.readouterr().err


def test_uk_national_role_records_an_evaluation_error_without_failing(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)

    def explode(kwargs):
        raise RuntimeError("the incumbent frame refused to load")

    module = _fake_evaluator(explode, [])
    monkeypatch.setattr(
        national_role, "_load_candidate_evaluator", lambda importer=None: module
    )
    out = tmp_path / "national"

    assert (
        builder.main(
            _argv(
                input_h5,
                out,
                "--no-staging",
                "--epochs",
                "5",
                *_incumbent_argv(input_h5, tmp_path),
            )
        )
        == 0
    )

    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    assert manifest["evaluation"]["status"] == "error"
    assert "refused to load" in manifest["evaluation"]["error"]
    assert manifest["evaluation"]["receipt"] is None
    assert "score_receipt" not in manifest["outputs"]
    assert not (out / "score_vs_incumbent.json").exists()
    assert "the incumbent evaluation failed" in capsys.readouterr().err


def test_uk_national_role_evaluates_after_staging_the_bundle(
    monkeypatch, tmp_path, capsys
):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    calls: list[dict] = []
    module = _fake_evaluator(lambda kwargs: _receipt(kwargs), calls)
    monkeypatch.setattr(
        national_role, "_load_candidate_evaluator", lambda importer=None: module
    )
    out = tmp_path / "national"

    assert (
        builder.main(
            _argv(
                input_h5,
                out,
                "--staging-local-only",
                "--epochs",
                "5",
                *_incumbent_argv(input_h5, tmp_path),
            )
        )
        == 0
    )

    manifest = json.loads((out / builder.MANIFEST_FILENAME).read_text())
    run_id = manifest["build_id"]
    assert manifest["evaluation"]["status"] == "completed"
    assert manifest["evaluation"]["verdict"] == "passed"
    assert manifest["outputs"]["score_receipt"]["sha256"] == _sha(
        out / "score_vs_incumbent.json"
    )
    # The receipt is listed in the local sums beside the record and manifest.
    _sums_verify(out)
    assert any(
        line.endswith("  score_vs_incumbent.json")
        for line in (out / "sha256sums.txt").read_text().splitlines()
    )
    bundle = validate_v2_bundle(out / "staging", run_id)
    assert bundle["progress"]["status"] == "completed"
    events = [
        (event["stage_id"], event["status"])
        for event in bundle["events"]
        if event["stage_id"] == "incumbent_evaluation"
    ]
    assert events == [
        ("incumbent_evaluation", "started"),
        ("incumbent_evaluation", "completed"),
    ]
    # The evaluation ran after the bundle was staged, before completion.
    stage_ids = [event["stage_id"] for event in bundle["events"]]
    assert stage_ids.index("dataset_staging") < stage_ids.index("incumbent_evaluation")
    assert stage_ids.index("incumbent_evaluation") < stage_ids.index("complete")
    artifacts = out / "staging" / "runs" / run_id / "artifacts"
    staged_receipt = json.loads((artifacts / "score_vs_incumbent.json").read_text())
    assert staged_receipt["evaluation"]["verdict"] == "passed"
    # The staged copy keeps the verdict, the pruned block and the aggregates
    # and drops the record arrays the reviewed-artifact policy refuses; the
    # full receipt is beside the outputs.
    assert "incumbent_unresolvable_pruned" in staged_receipt
    assert staged_receipt["candidate_target_wins"] == 1
    for key in ("target_drift", "signed_asymmetries", "measure_resolution"):
        assert key not in staged_receipt
    full_receipt = json.loads((out / "score_vs_incumbent.json").read_text())
    assert full_receipt["target_drift"][0]["winner"] == "candidate"
    assert calls[0]["candidate_sha256"] == manifest["outputs"]["dataset"]["sha256"]
    # The completion marker binds the final bytes: the build record gained
    # the delivery summary and the manifest its receipts after publication,
    # and ``build.json`` was re-issued over both.
    completion = json.loads((out / "build.json").read_text())
    for role, name in (
        ("build_record", "build_record.json"),
        ("rowwise_candidate_manifest", builder.MANIFEST_FILENAME),
    ):
        assert completion[role]["sha256"] == _sha(out / name), role
        assert completion[role]["size_bytes"] == (out / name).stat().st_size
        assert "note" not in completion[role]
    assert completion["dataset"]["sha256"] == _sha(out / "microcosm_uk_2024_25.h5")


def test_uk_national_dry_run_records_the_incumbent(monkeypatch, tmp_path, capsys):
    pytest.importorskip("tables")
    builder = candidate._load_builder_module()
    input_h5, _registry, _artifact, _pin = _national_inputs(monkeypatch, tmp_path)
    module = _fake_evaluator(lambda kwargs: _receipt(kwargs), [])
    monkeypatch.setattr(
        national_role, "_load_candidate_evaluator", lambda importer=None: module
    )

    assert (
        builder.main(
            _argv(
                input_h5,
                tmp_path / "plan",
                "--dry-run",
                "--no-staging",
                *_incumbent_argv(input_h5, tmp_path),
            )
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["incumbent"]["label"] == "efrs_fixture"
    assert plan["incumbent"]["sha256"] == _sha(tmp_path / "incumbent.h5")
    assert not (tmp_path / "plan").exists()


def test_uk_national_role_loads_the_real_scorer_by_its_public_name() -> None:
    """The driver calls the scorer's public factory name; loading the real
    module (no stand-in) proves the name exists (Vahid's #965 note 1)."""
    module = national_role._load_candidate_evaluator()
    assert module.__name__ == "microcosm.build.uk_runtime.candidate_score"
    assert callable(module.uk_default_measure_resolver_factory)
    assert callable(module.evaluate_uk_candidate_against_incumbent)


def test_uk_score_receipt_telemetry_summary_keeps_the_pruned_block() -> None:
    """The staged copy drops the three per-target arrays and keeps the
    pruned block whole, including a populated pruned_targets mapping (the
    block a real run fills with 120 rows; Vahid's #965 note 2)."""
    pruned_targets = {
        f"dwp/uc/family_{i}": {
            "name": f"dwp/uc/family_{i}",
            "family": "dwp_universal_credit",
            "unresolvable_measure": "benunit.uc_calibration_child_count",
            "reason": "provider failed computing benunit.uc_calibration_child_count",
            "adjudication": "microcosm#823",
        }
        for i in range(3)
    }
    score = {
        "artifacts": {"candidate": {"sha256": "1" * 64}},
        "target_drift": [{"target": "a@0", "error": 0.1}],
        "signed_asymmetries": [{"id": "x"}],
        "measure_resolution": {"candidate": {"rounds": []}},
        "incumbent_unresolvable_pruned": {
            "n_pruned": 3,
            "n_scored": 1,
            "n_surface": 4,
            "pruned_targets": pruned_targets,
            "measures": ["benunit.uc_calibration_child_count"],
            "families": {"dwp_universal_credit": 3},
            "reviewed_register": {"resource": "r.json", "sha256": "2" * 64},
            "note": "pruned",
        },
        "evaluation": {"verdict": "passed"},
    }
    summary = national_role._score_receipt_telemetry_summary(score)
    assert set(summary) == {
        "artifacts",
        "incumbent_unresolvable_pruned",
        "evaluation",
    }
    assert summary["incumbent_unresolvable_pruned"]["pruned_targets"] == pruned_targets

    # No list of mappings survives (the staging content policy's record
    # array refusal); the pruned rows are a mapping keyed by target name.
    def _record_arrays(value):
        if isinstance(value, list):
            return any(isinstance(item, dict) for item in value) or any(
                _record_arrays(item) for item in value
            )
        if isinstance(value, dict):
            return any(_record_arrays(item) for item in value.values())
        return False

    assert not _record_arrays(summary)
