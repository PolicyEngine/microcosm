"""The registered US Modal build programs start the emitter and report to it.

``tools/build_us_acs_local_release.py``, ``tools/build_us_puf_support_base.py``
and the preserved ACS staging builder behind
``tools/build_us_acs_multispine_base.py`` each open one telemetry run
(``microcosm.build.program_telemetry``) after parsing their arguments. Their
work is replaced here by stand-ins that record what ran; the shared fixture's
in-memory emitter records what they report. The tests of an emitter that
cannot be used put the adapter's own start back and name a missing emitter
command, so no test runs a real emitter.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import signal
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from microcosm_emitter_interface import client as interface_client
from microcosm_emitter_interface import launcher as interface_launcher

from microcosm.build.telemetry_emitter import (
    LocalTelemetryEmitter,
    TelemetryRequiredError,
)
from test_support.microcosm_build.telemetry import ADAPTER_START
from test_support.paths import paths_for

_TOOLS = paths_for("microcosm-build").repository / "tools"
#: The interface's own startup, captured before the shared fixture forbids it.
INTERFACE_START_EMITTER = interface_launcher.start_emitter
RUN_ID_VARIABLE = "MICROCOSM_TELEMETRY_RUN_ID"
PARENT_PID_VARIABLE = "MICROCOSM_TELEMETRY_PARENT_PID"
ACS_RELEASE_STAGES = ("materialize", "calibrate", "qa", "finalize", "package")


@functools.cache
def _tool(file_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        file_name.removesuffix(".py"), _TOOLS / file_name
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _acs_release() -> ModuleType:
    return _tool("build_us_acs_local_release.py")


def _puf_support_base() -> ModuleType:
    return _tool("build_us_puf_support_base.py")


def _acs_staging_builder() -> ModuleType:
    """The preserved builder the multispine command's shim runs."""
    return _tool("build_us_acs_multispine_base.py")._legacy


def _event_summary(emitter) -> list[tuple[str, str | None, str]]:
    return [
        (event["event_type"], event["stage_id"], event["status"])
        for event in emitter.events
    ]


def _arrange_unusable_emitter(monkeypatch, tmp_path: Path) -> None:
    """The adapter's own start, looking for an emitter command that is missing."""

    monkeypatch.setattr(LocalTelemetryEmitter, "start", classmethod(ADAPTER_START))
    monkeypatch.setattr(interface_client, "start_emitter", INTERFACE_START_EMITTER)
    monkeypatch.setenv(
        "MICROCOSM_EMITTER_COMMAND", str(tmp_path / "missing" / "microcosm-emitter")
    )


# --------------------------------------------------------------------------- #
# Each program, with its work replaced by stand-ins that record what ran       #
# --------------------------------------------------------------------------- #


def _passes(*arguments, **options) -> None:
    """A stand-in for a check or setting that passes."""


def _run_acs_release(
    monkeypatch, tmp_path: Path, stages: tuple[str, ...] = ACS_RELEASE_STAGES
) -> tuple[Callable[[], object], list]:
    module = _acs_release()
    ran: list[str] = []
    args = SimpleNamespace(stages=list(stages), checkpoint_dir=tmp_path / "checkpoints")
    monkeypatch.setattr(module, "_parse_args", lambda argv: args)
    for stage in ACS_RELEASE_STAGES:
        monkeypatch.setattr(
            module,
            f"do_{stage}",
            lambda parsed, stage=stage: ran.append(stage),
        )
    return (lambda: module.main([])), ran


def _run_puf_support_base_compatibility_path(
    monkeypatch, tmp_path: Path
) -> tuple[Callable[[], object], list]:
    module = _puf_support_base()
    ran: list[str] = []
    args = SimpleNamespace(stage="all", checkpoint_dir=None)
    monkeypatch.setattr(module, "_parse_args", lambda argv: args)
    monkeypatch.setattr(module, "_run_all", lambda parsed: ran.append("all"))
    return (lambda: module.main([])), ran


def _run_acs_staging_builder(
    monkeypatch, tmp_path: Path
) -> tuple[Callable[[], object], list]:
    module = _acs_staging_builder()
    ran: list[str] = []
    args = SimpleNamespace()
    monkeypatch.setattr(module, "_parse_args", lambda argv: args)

    def build(parsed, telemetry) -> int:
        ran.append("build")
        return 0

    monkeypatch.setattr(module, "_build", build)
    return (lambda: module.main([])), ran


PROGRAMS = {
    "us_acs_local_release": _run_acs_release,
    "us_puf_support_base": _run_puf_support_base_compatibility_path,
    "us_acs_multispine_staging": _run_acs_staging_builder,
}


@pytest.mark.parametrize("pipeline", PROGRAMS)
def test_each_program_reports_under_the_run_its_starter_names(
    monkeypatch, tmp_path, fake_telemetry_emitters, pipeline: str
) -> None:
    run, ran = PROGRAMS[pipeline](monkeypatch, tmp_path)
    monkeypatch.setenv(RUN_ID_VARIABLE, "plan-run.all")
    run()
    assert ran
    [emitter] = fake_telemetry_emitters
    assert (
        emitter.run.run_id,
        emitter.run.candidate_id,
        emitter.run.country_code,
        emitter.run.pipeline,
        emitter.run.run_kind,
    ) == ("plan-run.all", "plan-run", "US", pipeline, "build")
    assert _event_summary(emitter)[-1] == ("run", "complete", "completed")
    assert not emitter.available


@pytest.mark.parametrize("pipeline", PROGRAMS)
def test_each_program_started_by_hand_makes_its_own_attempt_id(
    monkeypatch, tmp_path, fake_telemetry_emitters, pipeline: str
) -> None:
    run, _ran = PROGRAMS[pipeline](monkeypatch, tmp_path)
    run()
    [emitter] = fake_telemetry_emitters
    assert emitter.run.run_id.startswith(pipeline.replace("_", "-") + "-")
    assert emitter.run.candidate_id is None


@pytest.mark.parametrize("pipeline", PROGRAMS)
def test_each_program_that_must_send_telemetry_stops_before_any_work_without_an_emitter(
    monkeypatch, tmp_path, pipeline: str
) -> None:
    run, ran = PROGRAMS[pipeline](monkeypatch, tmp_path)
    _arrange_unusable_emitter(monkeypatch, tmp_path)
    monkeypatch.setenv("MICROCOSM_TELEMETRY_REQUIRED", "1")
    with pytest.raises(TelemetryRequiredError):
        run()
    assert ran == []
    # Not even the release's checkpoint directory was made.
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("required", ["0", None], ids=["opted-out", "unset"])
@pytest.mark.parametrize("pipeline", PROGRAMS)
def test_each_program_whose_plan_opts_out_runs_its_work_without_an_emitter(
    monkeypatch, capfd, tmp_path, pipeline: str, required: str | None
) -> None:
    # "0" is a plan's opt-out; unset is a workstation's default.
    run, ran = PROGRAMS[pipeline](monkeypatch, tmp_path)
    _arrange_unusable_emitter(monkeypatch, tmp_path)
    if required is None:
        monkeypatch.delenv("MICROCOSM_TELEMETRY_REQUIRED", raising=False)
    else:
        monkeypatch.setenv("MICROCOSM_TELEMETRY_REQUIRED", required)
    try:
        run()
    except TelemetryRequiredError as error:
        pytest.fail(f"the program stopped without an emitter: {error}")
    assert ran
    assert "command_not_found" in capfd.readouterr().err


# --------------------------------------------------------------------------- #
# The ACS local release: one stage per stage it runs                          #
# --------------------------------------------------------------------------- #


def test_the_acs_local_release_records_each_stage_it_runs(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    run, ran = _run_acs_release(monkeypatch, tmp_path)
    assert run() == 0
    assert ran == list(ACS_RELEASE_STAGES)
    [emitter] = fake_telemetry_emitters
    assert _event_summary(emitter) == [
        *(
            (event_type, stage, status)
            for stage in ACS_RELEASE_STAGES
            for event_type, status in (("stage", "started"), ("stage", "completed"))
        ),
        ("run", "complete", "completed"),
    ]


def test_the_acs_local_release_records_the_stage_that_failed(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    module = _acs_release()
    run, ran = _run_acs_release(monkeypatch, tmp_path)

    def diverge(parsed) -> None:
        raise RuntimeError("calibration diverged")

    monkeypatch.setattr(module, "do_calibrate", diverge)
    with pytest.raises(RuntimeError, match="calibration diverged"):
        run()
    assert ran == ["materialize"]
    [emitter] = fake_telemetry_emitters
    assert _event_summary(emitter) == [
        ("stage", "materialize", "started"),
        ("stage", "materialize", "completed"),
        ("stage", "calibrate", "started"),
        ("stage", "calibrate", "failed"),
        ("run", "failed", "failed"),
    ]
    assert emitter.events[-1]["details"]["error_type"] == "RuntimeError"
    # The memory samples are still written when a stage fails.
    assert (tmp_path / "checkpoints" / "rss_samples.json").is_file()


class UnhandledSigtermError(Exception):
    """The test's own SIGTERM handler ran: nothing in the program took the signal."""


def test_the_acs_local_release_records_a_budget_stop_as_a_termination(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    module = _acs_release()
    run, ran = _run_acs_release(monkeypatch, tmp_path)

    def stopped_at_the_budget(parsed) -> None:
        # The runner's budget sends SIGTERM to the tool's process group.
        os.kill(os.getpid(), signal.SIGTERM)

    def reached_the_default(signum, frame) -> None:
        raise UnhandledSigtermError()

    monkeypatch.setattr(module, "do_calibrate", stopped_at_the_budget)
    previous = signal.signal(signal.SIGTERM, reached_the_default)
    try:
        with pytest.raises(BaseException) as raised:
            run()
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert isinstance(raised.value, SystemExit), repr(raised.value)
    assert raised.value.code == 128 + signal.SIGTERM
    assert ran == ["materialize"]
    [emitter] = fake_telemetry_emitters
    failure = emitter.events[-1]["details"]
    assert (failure["failed_during"], failure["failure_class"]) == (
        "calibrate",
        "terminated",
    )
    # The memory samples are still written.
    assert (tmp_path / "checkpoints" / "rss_samples.json").is_file()


def test_the_acs_local_release_records_only_the_stage_a_plan_names(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    run, ran = _run_acs_release(monkeypatch, tmp_path, stages=("qa",))
    run()
    assert ran == ["qa"]
    assert _event_summary(fake_telemetry_emitters[0]) == [
        ("stage", "qa", "started"),
        ("stage", "qa", "completed"),
        ("run", "complete", "completed"),
    ]


# --------------------------------------------------------------------------- #
# The PUF support base: the --stage all parent records its children           #
# --------------------------------------------------------------------------- #


def _arrange_staged_parent(
    monkeypatch, tmp_path: Path, *, failing_stage: str | None = None
) -> tuple[ModuleType, list[tuple[str, dict[str, str]]]]:
    """``--stage all`` with a checkpoint directory and stand-in children."""

    module = _puf_support_base()
    children: list[tuple[str, dict[str, str]]] = []
    args = SimpleNamespace(stage="all", checkpoint_dir=tmp_path / "checkpoints")
    monkeypatch.setattr(module, "_parse_args", lambda argv: args)

    class FreshRuntime:
        def __init__(self, *runtime_args, **runtime_options) -> None:
            self.context = SimpleNamespace(completed=())

    monkeypatch.setattr(module, "StageRuntime", FreshRuntime)
    monkeypatch.setattr(module, "_stage_run_config", lambda parsed: {})
    monkeypatch.setattr(
        module, "_stage_cli_args", lambda parsed, stage: ["--stage", stage]
    )

    def run_child(command, *, check, env):
        stage = command[-1]
        children.append((stage, env))
        return SimpleNamespace(returncode=3 if stage == failing_stage else 0)

    monkeypatch.setattr(module.subprocess, "run", run_child)
    return module, children


def test_the_staged_puf_support_base_parent_records_each_child_stage(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    module, children = _arrange_staged_parent(monkeypatch, tmp_path)
    monkeypatch.setenv(RUN_ID_VARIABLE, "route-a-base.all")
    module.main([])

    assert [stage for stage, _env in children] == list(module.PIPELINE_STEPS)
    # Each child is told its parent records its telemetry, and keeps the
    # parent's run ID and hash seed default.
    for _stage, env in children:
        assert env[PARENT_PID_VARIABLE] == str(os.getpid())
        assert env["PYTHONHASHSEED"] == os.environ.get("PYTHONHASHSEED", "0")
    [emitter] = fake_telemetry_emitters
    assert emitter.run.run_id == "route-a-base.all"
    assert _event_summary(emitter) == [
        *(
            ("stage", stage, status)
            for stage in module.PIPELINE_STEPS
            for status in ("started", "completed")
        ),
        ("run", "complete", "completed"),
    ]


def test_the_staged_puf_support_base_parent_records_a_failing_child(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    module = _puf_support_base()
    first, failing = module.PIPELINE_STEPS[:2]
    module, children = _arrange_staged_parent(
        monkeypatch, tmp_path, failing_stage=failing
    )
    with pytest.raises(SystemExit, match="failed with exit code 3"):
        module.main([])

    assert [stage for stage, _env in children] == [first, failing]
    [emitter] = fake_telemetry_emitters
    assert _event_summary(emitter) == [
        ("stage", first, "started"),
        ("stage", first, "completed"),
        ("stage", failing, "started"),
        ("stage", failing, "failed"),
        ("run", "failed", "failed"),
    ]
    assert emitter.events[-1]["details"]["failed_during"] == failing


def _arrange_named_stage(monkeypatch) -> tuple[ModuleType, list[str]]:
    module = _puf_support_base()
    ran: list[str] = []
    args = SimpleNamespace(stage="source_construction", checkpoint_dir=Path("cp"))
    monkeypatch.setattr(module, "_parse_args", lambda argv: args)
    monkeypatch.setattr(
        module, "_run_configured_stage", lambda parsed: ran.append(parsed.stage)
    )
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    return module, ran


def test_a_child_stage_of_the_puf_support_base_starts_no_emitter(
    monkeypatch, fake_telemetry_emitters
) -> None:
    module, ran = _arrange_named_stage(monkeypatch)
    monkeypatch.setenv(PARENT_PID_VARIABLE, str(os.getppid()))
    module.main([])
    assert ran == ["source_construction"]
    assert fake_telemetry_emitters == []


def test_a_puf_support_base_stage_run_by_hand_records_its_own_telemetry(
    monkeypatch, fake_telemetry_emitters
) -> None:
    module, ran = _arrange_named_stage(monkeypatch)
    # A value left in the shell names no parent of this process.
    monkeypatch.setenv(PARENT_PID_VARIABLE, str(os.getpid()))
    module.main([])
    assert ran == ["source_construction"]
    [emitter] = fake_telemetry_emitters
    assert _event_summary(emitter) == [
        ("stage", "source_construction", "started"),
        ("stage", "source_construction", "completed"),
        ("run", "complete", "completed"),
    ]


def test_the_puf_support_base_compatibility_path_records_one_stage(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    run, ran = _run_puf_support_base_compatibility_path(monkeypatch, tmp_path)
    run()
    assert ran == ["all"]
    assert _event_summary(fake_telemetry_emitters[0]) == [
        ("stage", "all", "started"),
        ("stage", "all", "completed"),
        ("run", "complete", "completed"),
    ]


def test_a_puf_support_base_input_refusal_is_recorded_as_a_failure(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    module = _puf_support_base()
    run, ran = _run_puf_support_base_compatibility_path(monkeypatch, tmp_path)

    def refuse(parsed) -> None:
        raise SystemExit("ASEC 2024 source failed digest verification.")

    monkeypatch.setattr(module, "_verify_asec_source_digests", refuse)
    with pytest.raises(SystemExit, match="digest verification"):
        run()
    assert ran == []
    [emitter] = fake_telemetry_emitters
    assert _event_summary(emitter)[-1] == ("run", "failed", "failed")


# --------------------------------------------------------------------------- #
# The ACS staging builder: one stage per main phase                           #
# --------------------------------------------------------------------------- #


def test_the_acs_staging_builder_records_its_phases_up_to_a_failure(
    monkeypatch, tmp_path, fake_telemetry_emitters
) -> None:
    module = _acs_staging_builder()
    args = SimpleNamespace(
        summary=tmp_path / "staging.summary.json",
        out_h5=tmp_path / "staging.h5",
        source_manifest=None,
        base_h5=tmp_path / "base.h5",
        donor_release_manifest=tmp_path / "donor.json",
        donor_channel="asec",
        sipp_donor=None,
        period=2024,
        puma_ladder=tmp_path / "ladder.npz",
        inputs_dir=tmp_path / "acs-pums",
    )
    base = SimpleNamespace(weights_for=lambda unit: SimpleNamespace(total=1.0))
    monkeypatch.setattr(module, "_parse_args", lambda argv: args)
    monkeypatch.setattr(module, "_validate_artifact_paths", _passes)
    monkeypatch.setattr(
        module.acs_sources, "load_acs_source_manifest", lambda path: object()
    )
    monkeypatch.setattr(module, "_manifest_file", lambda override: tmp_path / "m")
    monkeypatch.setattr(module, "_sha256", lambda path: "0" * 64)
    monkeypatch.setattr(
        module, "_donor_release_identity", lambda manifest, base_sha256: {}
    )
    monkeypatch.setattr(module, "_load_base_frame", lambda path: base)
    for check in (
        "_require_benefit_participation_inputs",
        "_require_donor_spm_independence_role",
        "_require_immigration_donor",
        "_require_income_donor",
        "_require_ssi_disability_donor",
        "_require_donor_work_experience",
        "_require_dense_donor_coverage",
    ):
        monkeypatch.setattr(module, check, _passes)
    monkeypatch.setattr(module, "declared_acs_transfer_target_families", tuple)
    monkeypatch.setattr(
        module, "_load_ssi_disability_donor", lambda path, *, period: (None, {})
    )
    monkeypatch.setattr(module, "_row_counts", lambda frame: {})
    monkeypatch.setattr(module, "load_us_puma_ladder", lambda path: None)

    def fetch(*fetch_args, **fetch_options):
        raise OSError("archive unavailable")

    monkeypatch.setattr(module.acs_sources, "fetch_acs_pums_sources", fetch)
    with pytest.raises(OSError, match="archive unavailable"):
        module.main([])

    [emitter] = fake_telemetry_emitters
    assert emitter.run.pipeline == "us_acs_multispine_staging"
    assert _event_summary(emitter) == [
        ("stage", "load_donor", "started"),
        ("stage", "load_donor", "completed"),
        ("stage", "fetch_acs_sources", "started"),
        ("stage", "fetch_acs_sources", "failed"),
        ("run", "failed", "failed"),
    ]
