"""US builds write their staging run files with the version 2 writer (#1135).

The build and its certification each open a staging run through
``_BuildProgress``; these tests read the run's local files and what the
in-memory emitter committed, and check them against the version 2 contract.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from importlib import metadata
from types import SimpleNamespace

import pytest

from microcosm.build.staging_v2 import (
    EVENT_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    STAGING_DOCUMENT_VERSION,
    validate_staging_bundle,
    validate_staging_delivery,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
US_STAGING = "policyengine/populace-us-staging"
BUILD = "rel-1"


def _load_builder_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


def _hosted_telemetry(client=None):
    from microcosm.build.telemetry_emitter import TelemetryRun
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    return FakeTelemetryEmitter(
        TelemetryRun(BUILD, "US", "us_fiscal_refresh"), client=client
    )


def _staging_arguments(builder, *, repository=US_STAGING):
    """The build's staging options with their defaults: no local directory."""
    return SimpleNamespace(
        no_staging=False,
        staging_dir=None,
        staging_repo_id=repository,
        staging_run_id=None,
        epochs=builder.DEFAULT_US_FISCAL_CALIBRATION_EPOCHS,
        dense_default_dataset=False,
    )


def _remote_build_progress(builder, tmp_path, client=None):
    emitter = _hosted_telemetry(client)
    progress = builder._build_progress(
        _staging_arguments(builder),
        release_root=tmp_path,
        release_id=BUILD,
        emitter=emitter,
    )
    return progress, emitter


def _run_dir(tmp_path, run_id=BUILD):
    """Where a run's files are written when no ``--staging-dir`` is given."""
    return tmp_path / "staging" / "runs" / run_id


def _read(run_dir, name):
    return json.loads((run_dir / name).read_text())


def _events(run_dir):
    return [
        json.loads(text)
        for text in (run_dir / "events.ndjson").read_text().splitlines()
        if text
    ]


def _run_events(emitter):
    return [event for event in emitter.events if event["event_type"] == "run"]


def test_a_us_build_writes_version_2_documents_with_its_identity(
    builder, tmp_path
) -> None:
    progress, _ = _remote_build_progress(builder, tmp_path)
    progress.stage("target_registry", message="Compiled targets.", n_targets=3)

    progress.complete()

    run_dir = _run_dir(tmp_path)
    manifest = _read(run_dir, "run_manifest.json")
    assert (manifest.get("schema_name"), manifest.get("schema_version")) == (
        RUN_MANIFEST_SCHEMA,
        STAGING_DOCUMENT_VERSION,
    )
    identity = (
        "run_id",
        "country_code",
        "operation_id",
        "pipeline",
        "candidate_id",
        "release_id",
        "run_kind",
        "non_release",
        "status",
        "current_stage",
        "sample",
    )
    assert {field: manifest.get(field) for field in identity} == {
        "run_id": BUILD,
        "country_code": "US",
        "operation_id": "us_fiscal_refresh_build",
        "pipeline": {
            "id": "us_fiscal_refresh",
            "version": metadata.version("microcosm-build"),
        },
        "candidate_id": BUILD,
        "release_id": BUILD,
        "run_kind": "release",
        "non_release": False,
        "status": "completed",
        "current_stage": "complete",
        "sample": {"mode": "full"},
    }
    documents = validate_staging_bundle(tmp_path / "staging", BUILD)
    delivery = documents["progress"]["delivery"]
    assert (delivery["mode"], delivery["configured_repository"]) == (
        "local_and_remote",
        US_STAGING,
    )
    assert delivery["upload_successes"] >= 1


def test_us_stage_statuses_become_version_2_event_statuses(builder, tmp_path) -> None:
    """A failed check is an event, not the run's end; other outcomes stay in the details."""
    progress, emitter = _remote_build_progress(builder, tmp_path)

    progress.stage("load_base_frame", message="Loading base population H5.")
    progress.stage(
        "qbi_input_gate",
        status="failed",
        message="QBI input signal check failed.",
        failures=["qbi share is zero"],
    )
    progress.stage(
        "base_frame_spm_composition",
        status="skipped",
        message="Base pool carries no readable SPM composition inputs.",
        error="no SPM columns",
    )
    progress.stage(
        "base_frame_spm_composition",
        status="retained",
        message="Classified the base pool's SPM measurement composition.",
    )

    run_dir = _run_dir(tmp_path)
    events = _events(run_dir)
    assert [
        (event.get("schema_name"), event.get("stage_id"), event.get("status"))
        for event in events[1:]
    ] == [
        (EVENT_SCHEMA, "load_base_frame", "started"),
        (EVENT_SCHEMA, "qbi_input_gate", "failed"),
        (EVENT_SCHEMA, "base_frame_spm_composition", "completed"),
        (EVENT_SCHEMA, "base_frame_spm_composition", "completed"),
    ]
    assert events[2]["details"]["failures"] == ["qbi share is zero"]
    assert [event["details"].get("stage_status") for event in events[3:]] == [
        "skipped",
        "retained",
    ]
    assert _read(run_dir, "progress.json").get("status") == "running"
    # Hosted telemetry keeps the statuses the build reported.
    assert [
        event["status"]
        for event in emitter.events
        if event["stage_id"] == "qbi_input_gate"
    ][0] == "failed"


def test_a_failed_us_build_records_a_version_2_failure_without_its_text(
    builder, tmp_path
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    progress.stage("export_dataset", message="Exporting the dataset.")

    progress.fail(RuntimeError("export failed at /restricted/asec.h5 token=abc"))

    run_dir = _run_dir(tmp_path)
    document = _read(run_dir, "progress.json")
    assert document.get("status") == "failed"
    assert document.get("failure") == {
        "error_code": "BUILD_FAILED",
        "error_type": "RuntimeError",
        "message": "The build failed during export_dataset.",
        "local_diagnostic_reference": None,
        "failure_class": "error",
    }
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert "token=abc" not in path.read_text(), path.name
    validate_staging_bundle(tmp_path / "staging", BUILD)
    assert [event["status"] for event in _run_events(emitter)] == ["failed"]


def test_a_terminated_us_build_records_a_termination(builder, tmp_path) -> None:
    from microcosm.build.termination import BuildTerminatedError

    progress, _ = _remote_build_progress(builder, tmp_path)

    progress.fail(BuildTerminatedError())

    failure = _read(_run_dir(tmp_path), "progress.json").get("failure") or {}
    assert (failure.get("error_code"), failure.get("failure_class")) == (
        "TERMINATED",
        "terminated",
    )


def test_a_us_run_writes_no_latest_pointer_or_runs_index(builder, tmp_path) -> None:
    """Version 2 runs are found by their folders, as UK runs are."""
    progress, emitter = _remote_build_progress(builder, tmp_path)

    progress.complete()

    requests = emitter.client.run_file_requests
    assert requests
    assert [
        (request.latest_pointer, request.run_index_entry) for request in requests
    ] == [(None, None)] * len(requests)
    run_dir = _run_dir(tmp_path)
    assert not (run_dir / "latest_staging.json").exists()
    assert not (run_dir / "runs.json").exists()
    assert {"latest_staging.json", "runs.json"}.isdisjoint(
        emitter.client.paths(US_STAGING)
    )


def test_a_hand_over_the_emitter_refuses_is_recorded_and_the_build_goes_on(
    builder, tmp_path
) -> None:
    from test_support.microcosm_build.emitter import FakeEmitterClient

    progress, emitter = _remote_build_progress(
        builder, tmp_path, client=FakeEmitterClient(accept_run_files=False)
    )
    progress.stage("target_registry", message="Compiled targets.")

    progress.complete()

    delivery = _read(_run_dir(tmp_path), "progress.json").get("delivery") or {}
    assert (delivery.get("upload_successes"), delivery.get("last_error_code")) == (
        0,
        "UPLOAD_FAILED",
    )
    assert [event["status"] for event in _run_events(emitter)] == ["completed"]


def test_a_passing_result_file_is_staged_as_a_version_2_artifact(
    builder, tmp_path
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    result_file = tmp_path / "stored_inputs.json"
    result_file.write_text(json.dumps({"status": "passed", "checked_columns": 12}))

    progress.attach_artifact("stored_inputs", result_file)

    manifest = _read(_run_dir(tmp_path), "run_manifest.json")
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    assert [
        (artifact["logical_name"], artifact["contract_relative_path"])
        for artifact in artifacts
    ] == [("stored_inputs", "artifacts/stored_inputs.json")]
    assert (
        emitter.client.hub_files[
            (US_STAGING, f"runs/{BUILD}/artifacts/stored_inputs.json")
        ]
        == result_file.read_bytes()
    )


def test_a_result_file_the_content_checks_refuse_stays_out_and_the_build_goes_on(
    builder, tmp_path, capsys
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    refused = tmp_path / "input_coverage.json"
    refused.write_text(json.dumps({"token": "not for the staging repository"}))
    passing = tmp_path / "stored_inputs.json"
    passing.write_text(json.dumps({"status": "passed"}))

    progress.attach_artifact("input_coverage", refused)
    progress.attach_artifact("stored_inputs", passing)
    progress.complete()

    artifacts = _read(_run_dir(tmp_path), "run_manifest.json").get("artifacts")
    assert isinstance(artifacts, list)
    assert [artifact["logical_name"] for artifact in artifacts] == ["stored_inputs"]
    assert not any(
        "not for the staging repository" in content.decode(errors="replace")
        for content in emitter.client.hub_files.values()
    )
    assert "input_coverage" in capsys.readouterr().err
    assert [event["status"] for event in _run_events(emitter)] == ["completed"]


def test_a_staging_error_fails_neither_the_build_nor_its_hosted_telemetry(
    builder, capsys
) -> None:
    emitter = _hosted_telemetry()

    class BrokenBundle:
        remote_staging = False

        def stage(self, *args, **kwargs):
            raise OSError("staging disk unavailable")

        attach_artifact = calibration_progress = complete = stage

    progress = builder._BuildProgress(
        run_id=BUILD, emitter=emitter, staging_bundle=BrokenBundle()
    )

    try:
        progress.stage("target_compilation", message="Compiling targets.")
        progress.complete()
    except OSError:
        pytest.fail("a staging error stopped the build")

    assert [
        (event["stage_id"], event["status"])
        for event in emitter.events
        if event["stage_id"] == "target_compilation"
    ][0] == ("target_compilation", "started")
    assert [event["status"] for event in _run_events(emitter)] == ["completed"]
    assert "staging disk unavailable" in capsys.readouterr().err


def test_the_build_manifest_records_the_version_2_delivery_summary(
    builder, tmp_path
) -> None:
    progress, _ = _remote_build_progress(builder, tmp_path)
    progress.complete()

    block = builder._staging_manifest_block(progress)

    assert block.get("contract_version") == 2
    assert validate_staging_delivery(block) == block
    assert (block["mode"], block["run_id"]) == ("local_and_remote", BUILD)
    assert block["upload_successes"] >= 1
    opted_out = builder._staging_manifest_block(None)
    assert validate_staging_delivery(opted_out) == opted_out
    assert (opted_out["enabled"], opted_out["opt_out_reason"]) == (
        False,
        "--no-staging",
    )


def test_certification_opens_a_version_2_run_of_its_own(builder, tmp_path) -> None:
    emitter = _hosted_telemetry()
    record = SimpleNamespace(
        build=BUILD,
        release_root=tmp_path,
        staging={"run_id": BUILD},
        staging_settings={"no_staging": False, "repo_id": US_STAGING, "prefix": "runs"},
    )

    progress = builder._certify_progress(
        SimpleNamespace(no_staging=False, staging_dir=None),
        record=record,
        build=f"{BUILD}-evidence",
        run_id=f"{BUILD}-certify",
        emitter=emitter,
    )
    progress.complete()

    manifest = _read(_run_dir(tmp_path, f"{BUILD}-certify"), "run_manifest.json")
    assert {
        field: manifest.get(field)
        for field in (
            "run_id",
            "operation_id",
            "candidate_id",
            "release_id",
            "run_kind",
        )
    } == {
        "run_id": f"{BUILD}-certify",
        "operation_id": "us_fiscal_refresh_certification",
        "candidate_id": BUILD,
        "release_id": f"{BUILD}-evidence",
        "run_kind": "certification",
    }
    validate_staging_bundle(tmp_path / "staging", f"{BUILD}-certify")


def _write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_us_result_files_of_aggregate_rows_are_staged(builder, tmp_path) -> None:
    """Rows of aggregates by state, age band or program pass the reviewed list."""
    progress, emitter = _remote_build_progress(builder, tmp_path)
    demographics = _write_json(
        tmp_path / "demographics.json",
        {"age_bands": [{"band": "0-4", "weighted_people": 19.5e6}]},
    )
    take_up = _write_json(
        tmp_path / "us_snap_state_take_up.json",
        {"states": [{"state": "CA", "take_up_rate": 0.71}]},
    )

    progress.attach_artifact("demographics", demographics)
    progress.attach_artifact("us_snap_state_take_up", take_up)

    artifacts = _read(_run_dir(tmp_path), "run_manifest.json").get("artifacts")
    assert isinstance(artifacts, list)
    assert [artifact["logical_name"] for artifact in artifacts] == [
        "demographics",
        "us_snap_state_take_up",
    ]
    assert (
        US_STAGING,
        f"runs/{BUILD}/artifacts/demographics.json",
    ) in emitter.client.hub_files


def test_calibration_diagnostics_larger_than_5_mib_are_staged(
    builder, tmp_path
) -> None:
    """Per-target rows the dashboard's calibration view reads, up to 32 MiB."""
    progress, emitter = _remote_build_progress(builder, tmp_path)
    targets = [
        {"name": f"target_{index}", "target": 1.0, "final_estimate": 1.01}
        for index in range(90_000)
    ]
    diagnostics = _write_json(
        tmp_path / "calibration_diagnostics.json", {"targets": targets}
    )
    assert diagnostics.stat().st_size > 5 * 1024 * 1024

    progress.attach_artifact("calibration_diagnostics", diagnostics)

    artifacts = _read(_run_dir(tmp_path), "run_manifest.json").get("artifacts")
    assert isinstance(artifacts, list)
    assert [artifact["logical_name"] for artifact in artifacts] == [
        "calibration_diagnostics"
    ]
    assert (
        emitter.client.hub_files[
            (US_STAGING, f"runs/{BUILD}/artifacts/calibration_diagnostics.json")
        ]
        == diagnostics.read_bytes()
    )


def test_a_result_file_not_on_the_reviewed_list_may_not_hold_rows(
    builder, tmp_path, capsys
) -> None:
    progress, _ = _remote_build_progress(builder, tmp_path)
    rows = _write_json(
        tmp_path / "input_coverage.json",
        {"columns": [{"column": "age", "covered": True}]},
    )

    progress.attach_artifact("input_coverage", rows)

    artifacts = _read(_run_dir(tmp_path), "run_manifest.json").get("artifacts")
    assert artifacts == []
    assert "input_coverage" in capsys.readouterr().err


def _attached_result_file_names(builder) -> set[str]:
    """Every name the builder attaches a result file under, read from its source."""
    source = (
        _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    ).read_text()
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "attach_artifact"
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)
            elif isinstance(first, ast.Name) and first.id.isupper():
                names.add(getattr(builder, first.id))
    return names


def test_every_result_file_the_build_attaches_is_reviewed(builder) -> None:
    attached = _attached_result_file_names(builder)
    reviewed = getattr(builder, "US_STAGING_RESULT_FILES", {})

    assert len(attached) == 20
    assert attached == set(reviewed)


def test_a_build_the_release_checks_refuse_closes_its_staging_run_blocked(
    builder, tmp_path
) -> None:
    progress, emitter = _remote_build_progress(builder, tmp_path)
    progress.stage("release_gates", message="Checking the release.")
    try:
        refusal = builder._BuildFailedReleaseGatesError(
            ["Input coverage failed: x", "Target fit failed: y"]
        )
    except TypeError:
        pytest.fail("the release-check refusal does not carry its failures")

    progress.fail(refusal)

    document = _read(_run_dir(tmp_path), "progress.json")
    assert (document.get("status"), document.get("failure")) == ("blocked", None)
    assert document.get("block") == {
        "phase": "release_gates",
        "blocking_failure_count": 2,
        "blocking_gate_ids": [],
    }
    validate_staging_bundle(tmp_path / "staging", BUILD)
    # The final commit is still confirmed, and hosted telemetry is unchanged.
    assert emitter.client.read_back_waits == [120.0]
    assert [event["status"] for event in _run_events(emitter)] == ["failed"]


def test_a_local_only_run_writes_below_the_staging_dir(builder, tmp_path) -> None:
    """--staging-dir is the local root: the run's files go to runs/<run_id>/."""
    emitter = _hosted_telemetry()
    arguments = _staging_arguments(builder, repository=None)
    arguments.staging_dir = tmp_path / "local-staging"

    progress = builder._build_progress(
        arguments, release_root=tmp_path / "out", release_id=BUILD, emitter=emitter
    )
    progress.complete()

    assert (tmp_path / "local-staging" / "runs" / BUILD / "run_manifest.json").is_file()
    documents = validate_staging_bundle(tmp_path / "local-staging", BUILD)
    delivery = documents["progress"]["delivery"]
    assert (delivery["mode"], delivery["configured_repository"]) == (
        "local_only",
        None,
    )
    assert emitter.client.run_file_requests == []


def test_the_staging_prefix_option_is_gone(builder, monkeypatch) -> None:
    """Version 2 runs are always under runs/, so there is no prefix to set."""
    monkeypatch.setenv("POPULACE_STAGING_PREFIX", "candidate-runs")
    argv = ["--ledger-facts", "facts.jsonl", "--out", "release"]

    assert not hasattr(builder._parse_args(argv), "staging_prefix")
    with pytest.raises(SystemExit):
        builder._parse_args([*argv, "--staging-prefix", "runs"])


#: An exact-count build's arguments, as the launcher writes them, without
#: ``--no-staging``.
_EXACT_COUNT_ARGV = [
    "--pool-manifest",
    "pool.manifest.json",
    "--pool-manifest-sha256",
    "a" * 64,
    "--pool-release-id",
    "fixture-publication",
    "--exact-k",
    "20000",
    "--exact-k-pi-hi",
    "0.95",
    "--ledger-facts",
    "facts",
    "--ledger-facts-sha256",
    "b" * 64,
    "--ledger-manifest-sha256",
    "c" * 64,
    "--incumbent-diagnostics",
    "incumbent.json",
    "--incumbent-diagnostics-sha256",
    "d" * 64,
    "--frozen-target-surface-sha256",
    "e" * 64,
    "--out",
    "out",
    "--release-id",
    "populace-us-2024-k20000-fixture",
    "--seed",
    "17",
]


def test_an_exact_count_build_stages_under_its_build_id(builder, tmp_path) -> None:
    try:
        arguments = builder._parse_args(_EXACT_COUNT_ARGV)
    except SystemExit:
        pytest.fail("an exact-count build still requires --no-staging")
    build = "populace-us-2024-k20000-fixture"
    run_id = builder._telemetry_run_id(arguments, timestamp=builder.datetime.now())

    progress = builder._build_progress(
        arguments,
        release_root=tmp_path,
        release_id=build,
        run_id=run_id,
        emitter=_hosted_telemetry(),
    )
    progress.complete()

    assert run_id == build
    manifest = validate_staging_bundle(tmp_path / "staging", build)["run_manifest"]
    assert (manifest["candidate_id"], manifest["status"]) == (build, "completed")


def test_microcosm_has_no_version_1_staging_writer() -> None:
    import microcosm.build as build_package
    from microcosm.build.staging_storage import RunFileHandOver

    assert importlib.util.find_spec("microcosm.build.staging") is None
    for name in (
        "StagingRunBundleWriter",
        "LATEST_STAGING_POINTER",
        "RUNS_INDEX",
        "STAGING_SCHEMA_VERSION",
    ):
        assert not hasattr(build_package, name), name
    parameters = inspect.signature(RunFileHandOver.hand_over).parameters
    assert {"latest_pointer", "run_index_entry"}.isdisjoint(parameters)


def test_a_failure_after_the_run_closed_leaves_it_as_it_was(
    builder, tmp_path, capsys
) -> None:
    """An error after the run completed records the build's failure with the
    hosted telemetry only: the run keeps its state, and its final commit is
    not checked again."""
    progress, emitter = _remote_build_progress(builder, tmp_path)
    progress.complete()
    capsys.readouterr()

    progress.fail(RuntimeError("printing the result failed"))

    assert _read(_run_dir(tmp_path), "progress.json")["status"] == "completed"
    assert emitter.client.read_back_waits == [120.0]
    assert "could not" not in capsys.readouterr().err


def test_an_interrupt_while_the_run_closes_still_hands_its_final_state_over(
    builder, tmp_path
) -> None:
    """A SIGTERM during the final hand-over of a completed run (review round 1
    of microcosm#1771): the run has stopped running, but its final state has
    not reached the emitter, so the build's failure path hands it over and
    checks it by receipt, as microcosm#1749 does for any interrupted run."""
    from microcosm.build.termination import BuildTerminatedError
    from test_support.microcosm_build.emitter import FakeEmitterClient

    class InterruptedHandOver(FakeEmitterClient):
        interrupt_next = False

        def stage_run_files(self, request):
            if self.interrupt_next:
                self.interrupt_next = False
                raise BuildTerminatedError()
            return super().stage_run_files(request)

    client = InterruptedHandOver()
    progress, emitter = _remote_build_progress(builder, tmp_path, client=client)
    progress.stage("manifests", message="Writing the manifests.")
    client.interrupt_next = True
    with pytest.raises(BuildTerminatedError):
        progress.complete()

    progress.fail(BuildTerminatedError())

    remote = client.hub_json(US_STAGING, f"runs/{BUILD}/progress.json")
    assert remote["status"] != "running"
    checks = [
        event["details"]
        for event in emitter.events
        if event["details"].get("check") == "staging_final_commit"
    ]
    assert [check["method"] for check in checks] == ["receipt"]
    assert "read_back_run_files" not in client.calls
