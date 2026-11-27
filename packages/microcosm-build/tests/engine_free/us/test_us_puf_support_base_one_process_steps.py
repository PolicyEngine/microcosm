"""Under ``--stage all``, one step process runs 16 steps (decision 1.5 · 6, A).

``tools/build_us_puf_support_base.py --stage all`` starts one child process
for the 16 steps from ``capital_gains_tail_transfer`` to
``education_inputs_post_clone`` instead of one per step. The child runs them
in order. Only the first step reads its predecessor's checkpoint from disk;
each later step takes the frame the step before it completed. Each step still
writes its checkpoint, completion record, profile record and saved check
observations, so a stopped run resumes after the last finished step.

Before each step the child writes the step's name to a pipe its parent
opened, and the parent records one telemetry stage change per step under the
step's own name. The child learns the pipe's file descriptor from the
environment variable ``MICROCOSM_US_BASE_STEP_REPORT``, and uses it only
when the process that set it is its parent; a named ``--stage`` run by hand
runs one step, as before.

The child-side tests call ``_run_outer_stage`` in this process, with the
real checkpoint runtime and writer. Step functions that need real inputs are
replaced with deterministic ones that read every numeric and text value, the
dtypes, the frame metadata and the weights, so a frame passed in memory that
differed from the one a checkpoint read gives would write different bytes.
``capital_gain_distributions`` runs its real code. The parent-side tests
replace ``subprocess.run`` and ``subprocess.Popen``; one starts a real child.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.outer_stage_runtime import (
    OUTER_STAGE_CONTEXT_FILENAME,
    frame_identity,
)
from microcosm.build.stage_profile import STAGE_PROFILE_FILENAME
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_gate_shadow import arm_us_tool_gate_shadows
from test_support.paths import paths_for

_TOOL = (
    paths_for("microcosm-build").repository / "tools" / "build_us_puf_support_base.py"
)
_REPORT_VARIABLE = "MICROCOSM_US_BASE_STEP_REPORT"
_ONE_PROCESS_STEPS = (
    "capital_gains_tail_transfer",
    "capital_gain_distributions",
    "qbi_reconciliation",
    "wic_post_clone",
    "housing_assistance",
    "prior_year_income_post_clone",
    "child_support_post_clone",
    "disability_benefits_post_clone",
    "workers_compensation_post_clone",
    "weeks_unemployed_post_clone",
    "childcare_post_clone",
    "adult_care_post_clone",
    "energy_subsidy_post_clone",
    "retirement_contributions_post_clone",
    "retirement_distributions_post_clone",
    "education_inputs_post_clone",
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", _TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder() -> ModuleType:
    return _load_builder()


def test_the_one_process_steps_are_the_16_from_the_tail_to_education_inputs(
    builder: ModuleType,
) -> None:
    assert builder.ONE_PROCESS_STEPS == _ONE_PROCESS_STEPS
    # The same 16 follow each other under both location rules.
    for pipeline_steps in (builder.PIPELINE_STEPS, builder.BLOCK_V1_PIPELINE_STEPS):
        first = pipeline_steps.index(_ONE_PROCESS_STEPS[0])
        assert pipeline_steps[first : first + 16] == _ONE_PROCESS_STEPS
        assert pipeline_steps[first - 1] == "qrf_finalization"


# ---------------------------------------------------------------------------
# The child: one process runs the 16 steps
# ---------------------------------------------------------------------------


def _qrf_finalization_frame(builder: ModuleType) -> Frame:
    """A small cloned US frame, standing in for what qrf_finalization writes.

    It has float, integer, pyarrow- and python-backed ``str`` and object text
    columns, and the two capital-gains inputs the real
    ``capital_gain_distributions`` step reads.
    """

    unit_ids = np.asarray([1, 2, 3, 4, 5], dtype="int64")
    person_unit_ids = np.asarray([1, 1, 2, 3, 4, 5], dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype="int64"),
            "person_household_id": person_unit_ids,
            "person_tax_unit_id": person_unit_ids,
            "person_spm_unit_id": person_unit_ids,
            "person_family_id": person_unit_ids,
            "person_marital_unit_id": person_unit_ids,
            "long_term_capital_gains_before_response": np.asarray(
                [6_000.25, 4_000.5, 0.0, -4_000.75, 50_000.125, 8_000.0]
            ),
            "non_sch_d_capital_gains": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 3_000.5]),
            "employment_income_before_lsr": np.asarray(
                [51_234.567, 0.1, 19_999.99, 7.0, 125_000.333, 3.3]
            ),
            "ssn_card_type": pd.Series(
                ["CITIZEN", "NONE", "CITIZEN", "OTHER_NON_CITIZEN", "CITIZEN", "NONE"],
                dtype="str",
            ),
            "immigration_status_str": pd.Series(
                ["CITIZEN"] * 5 + ["LEGAL_PERMANENT_RESIDENT"],
                dtype=pd.StringDtype("python", na_value=np.nan),
            ),
            "tax_unit_role_input": pd.Series(
                ["HEAD", "SPOUSE", "HEAD", "HEAD", "HEAD", "HEAD"], dtype=object
            ),
        }
    )
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({f"{entity}_id": unit_ids})
    tables["tax_unit"]["filing_status_input"] = pd.Series(
        ["JOINT", "SINGLE", "SINGLE", "HEAD_OF_HOUSEHOLD", "SINGLE"], dtype=object
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([100.5, 200.25, 300.0, 50.75, 75.0]), WeightKind.DESIGN
            )
        },
        pd.Series(["asec_2024"] * 6, name="stratum", dtype=object),
    )
    return builder.clone_us_frame_for_puf_support(frame)


def _with_person_columns(frame: Frame, columns: dict[str, object]) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for name, values in columns.items():
        tables["person"][name] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _value_read_from_everything(frame: Frame, extra: float) -> np.ndarray:
    """One float per person, read from every value a step could depend on."""

    person = frame.table("person")
    numeric_sum = person.select_dtypes("number").sum(axis=1).to_numpy(np.float64)
    text_lengths = np.zeros(len(person))
    for name in person.columns:
        if not pd.api.types.is_numeric_dtype(person[name].dtype):
            text_lengths += np.asarray(
                [
                    len(value) if isinstance(value, str) else -1
                    for value in person[name]
                ],
                dtype=np.float64,
            )
    dtype_text = "|".join(f"{name}:{dtype}" for name, dtype in person.dtypes.items())
    dtype_fingerprint = int(hashlib.sha256(dtype_text.encode()).hexdigest()[:6], 16)
    filing_status_lengths = sum(
        len(value) for value in frame.table("tax_unit")["filing_status_input"]
    )
    return (
        numeric_sum * 1.000001
        + text_lengths
        + dtype_fingerprint
        + 7.0 * len(frame.metadata or {})
        + frame.weights_for("household").values.sum() / 1e6
        + filing_status_lengths
        + extra
    )


class _StandInSteps:
    """Deterministic stand-ins for the step functions that need real inputs.

    Each records the frame it was given, by step, and adds one person
    column computed from every value in it.
    """

    def __init__(self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch):
        self.inputs: dict[str, Frame] = {}
        self.outputs: dict[str, Frame] = {}
        self.fail_at: str | None = None

        def tail_transfer(_args, frame):
            self._record_input(builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME, frame)
            value = _value_read_from_everything(frame, 0.0)
            result = _with_person_columns(
                frame,
                {
                    "tail_value": value,
                    "tail_label": pd.Series(
                        np.where(value > np.median(value), "high", "low"),
                        index=frame.table("person").index,
                        dtype="str",
                    ),
                },
            )
            self.outputs[builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME] = result
            return result, {"note": "tail"}

        def post_qrf_step(stage, _args, frame, stage_metadata):
            self._record_input(stage, frame)
            result = _with_person_columns(
                frame,
                {
                    f"{stage}_value": _value_read_from_everything(
                        frame, float(len(stage_metadata))
                    )
                },
            )
            self.outputs[stage] = result
            return result, {"signals": {"stage": stage}}

        distributions = builder._capital_gain_distributions_stage

        def recorded_distributions(args, frame):
            self._record_input(builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME, frame)
            result = distributions(args, frame)
            self.outputs[builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME] = result[0]
            return result

        monkeypatch.setattr(
            builder, "_capital_gains_tail_transfer_stage", tail_transfer
        )
        monkeypatch.setattr(
            builder, "_capital_gain_distributions_stage", recorded_distributions
        )
        monkeypatch.setattr(builder, "_post_qrf_frame_stage", post_qrf_step)

    def _record_input(self, stage: str, frame: Frame) -> None:
        if stage == self.fail_at:
            raise RuntimeError(f"stand-in failure in {stage}")
        self.inputs[stage] = frame


def _step_args(builder: ModuleType, checkpoint_dir: Path, stage: str):
    return SimpleNamespace(
        stage=stage,
        checkpoint_dir=checkpoint_dir,
        location_rule=builder.US_LOCATION_RULE_LEGACY,
        seed=0,
        target_year=2024,
        out=checkpoint_dir.parent / "out",
        allow_geography_ladder_gate_failures=False,
    )


def _complete_steps_before_the_16(builder: ModuleType, checkpoint_dir: Path) -> None:
    """Record the five steps before the 16, as the staged build does."""

    runtime = builder.StageRuntime(
        checkpoint_dir, builder.OUTER_STAGE_PIPELINE, run_config={}
    )
    frame = _qrf_finalization_frame(builder)
    for stage in builder.PIPELINE_STEPS[
        : builder.PIPELINE_STEPS.index("qrf_finalization") + 1
    ]:
        if stage == "primary_qrf_chain":
            runtime.complete_without_frame(stage)
        else:
            runtime.complete(stage, frame)


def _completed_steps(builder: ModuleType, checkpoint_dir: Path) -> tuple[str, ...]:
    return builder.StageRuntime(
        checkpoint_dir, builder.OUTER_STAGE_PIPELINE
    ).context.completed


@contextmanager
def _report_pipe(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], list[str]]]:
    """Set the report variable as a ``--stage all`` parent would, with this
    process as the step process's parent; yield the call that returns the
    step names written to the pipe once its write end is closed.
    """

    read_fd, write_fd = os.pipe()
    open_fds = {read_fd, write_fd}
    with monkeypatch.context() as child:
        child.setenv(
            _REPORT_VARIABLE,
            builder._step_report_environment(write_fd)[_REPORT_VARIABLE],
        )
        child.setattr(os, "getppid", os.getpid)

        def reported() -> list[str]:
            # A step process's write end closes when the process exits; here
            # the test closes it once the steps returned.
            os.close(write_fd)
            open_fds.discard(write_fd)
            with os.fdopen(read_fd, "rb") as reports:
                open_fds.discard(read_fd)
                return reports.read().decode("utf-8").splitlines()

        try:
            yield reported
        finally:
            for fd in open_fds:
                os.close(fd)


def _files_and_digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_frames_identical(in_memory: Frame, read_back: Frame) -> None:
    assert in_memory.entities == read_back.entities
    assert in_memory.links == read_back.links
    for name in in_memory.entities:
        pd.testing.assert_frame_equal(
            in_memory.table(name), read_back.table(name), check_exact=True
        )
        assert [str(dtype) for dtype in in_memory.table(name).dtypes] == [
            str(dtype) for dtype in read_back.table(name).dtypes
        ]
    for entity in in_memory.weighted_entities:
        assert in_memory.weights_for(entity).kind == read_back.weights_for(entity).kind
        np.testing.assert_array_equal(
            in_memory.weights_for(entity).values, read_back.weights_for(entity).values
        )
    pd.testing.assert_series_equal(in_memory.strata, read_back.strata, check_exact=True)
    assert in_memory.mass_log == read_back.mass_log
    assert dict(in_memory.metadata or {}) == dict(read_back.metadata or {})
    assert frame_identity(in_memory) == frame_identity(read_back)


def test_one_child_writes_the_same_files_as_one_process_per_step(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Both runs use the same directory, so the paths in the run context and
    # the saved observations are the same. The saved check observations are
    # written (the shadow evaluation is on) and compared too.
    arm_us_tool_gate_shadows(monkeypatch)
    monkeypatch.delenv(_REPORT_VARIABLE, raising=False)
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "run" / "checkpoints"

    # Today's staged build: one process per step, each reading the previous
    # step's checkpoint. Each _run_outer_stage call does what one step
    # process does.
    _complete_steps_before_the_16(builder, checkpoint_dir)
    with monkeypatch.context() as first_run:
        one_per_step = _StandInSteps(builder, first_run)
        for stage in _ONE_PROCESS_STEPS:
            builder._run_outer_stage(_step_args(builder, checkpoint_dir, stage))
    files_one_per_step = _files_and_digests(tmp_path / "run")
    shutil.rmtree(tmp_path / "run")

    _complete_steps_before_the_16(builder, checkpoint_dir)
    with monkeypatch.context() as second_run:
        one_child = _StandInSteps(builder, second_run)
        with _report_pipe(builder, second_run) as reported:
            builder._run_outer_stage(
                _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
            )
            assert reported() == list(_ONE_PROCESS_STEPS)
    files_one_child = _files_and_digests(tmp_path / "run")

    pipeline = builder.OUTER_STAGE_PIPELINE
    checkpoint_names = [
        f"checkpoints/{pipeline.index(stage):03d}_{stage}.frame.h5"
        for stage in _ONE_PROCESS_STEPS
    ]
    observation_names = [
        f"checkpoints/{builder.GATE_BATTERY_SHADOW_DIRECTORY}/{stage}.observations.json"
        for stage in _ONE_PROCESS_STEPS
    ]
    assert set(checkpoint_names) | set(observation_names) <= set(files_one_child)
    assert f"checkpoints/{OUTER_STAGE_CONTEXT_FILENAME}" in files_one_child
    assert files_one_child == files_one_per_step
    assert (
        _completed_steps(builder, checkpoint_dir)
        == pipeline.names[: pipeline.index(_ONE_PROCESS_STEPS[-1]) + 1]
    )
    # Each step after the first was given the frame the step before it
    # completed, not a frame read from its checkpoint; that frame equals
    # what reading the checkpoint gives.
    for previous, stage in zip(
        _ONE_PROCESS_STEPS[:-1], _ONE_PROCESS_STEPS[1:], strict=True
    ):
        assert one_child.inputs[stage] is one_child.outputs[previous], stage
        _assert_frames_identical(one_child.inputs[stage], one_per_step.inputs[stage])
        read_back = load_frame_checkpoint(
            checkpoint_dir / f"{pipeline.index(previous):03d}_{previous}.frame.h5"
        ).frame
        _assert_frames_identical(one_child.inputs[stage], read_back)


def test_the_child_reports_profiles_saves_and_records_each_step_in_order(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch)
    events: list[tuple[str, str]] = []

    @contextmanager
    def recording_profiler(stage_name: str, directory: Path):
        assert directory == checkpoint_dir
        events.append(("start profiling", stage_name))
        yield
        events.append(("finish profiling", stage_name))

    @contextmanager
    def recording_gate_shadow(step_args):
        yield lambda: events.append(("save observations", step_args.stage))

    load_predecessor = builder.StageRuntime.load_predecessor
    complete = builder.StageRuntime.complete
    report_step_start = builder._report_step_start

    def recording_load(self, stage_name, **kwargs):
        events.append(("read the previous checkpoint", stage_name))
        return load_predecessor(self, stage_name, **kwargs)

    def recording_complete(self, stage_name, frame, **kwargs):
        events.append(("record complete", stage_name))
        return complete(self, stage_name, frame, **kwargs)

    def recording_report(report_fd: int, stage_name: str) -> None:
        events.append(("report start", stage_name))
        report_step_start(report_fd, stage_name)

    monkeypatch.setattr(builder, "profile_stage", recording_profiler)
    monkeypatch.setattr(builder, "_armed_stage_gate_shadow", recording_gate_shadow)
    monkeypatch.setattr(builder.StageRuntime, "load_predecessor", recording_load)
    monkeypatch.setattr(builder.StageRuntime, "complete", recording_complete)
    monkeypatch.setattr(builder, "_report_step_start", recording_report)

    with _report_pipe(builder, monkeypatch) as reported:
        builder._run_outer_stage(
            _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
        )
        assert reported() == list(_ONE_PROCESS_STEPS)

    expected: list[tuple[str, str]] = []
    for stage in _ONE_PROCESS_STEPS:
        expected += [("report start", stage), ("start profiling", stage)]
        if stage == _ONE_PROCESS_STEPS[0]:
            # Only the first step reads its predecessor from disk.
            expected.append(("read the previous checkpoint", stage))
        expected += [
            ("save observations", stage),
            ("record complete", stage),
            ("finish profiling", stage),
        ]
    assert events == expected


def test_the_child_writes_one_profile_record_per_step(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch)

    with _report_pipe(builder, monkeypatch) as reported:
        builder._run_outer_stage(
            _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
        )
        reported()

    profile = json.loads((checkpoint_dir / STAGE_PROFILE_FILENAME).read_text())
    assert sorted(profile["stages"]) == sorted(_ONE_PROCESS_STEPS)
    assert all(
        profile["stages"][stage]["status"] == "succeeded"
        for stage in _ONE_PROCESS_STEPS
    )


def test_a_step_that_fails_leaves_the_earlier_steps_recorded(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arm_us_tool_gate_shadows(monkeypatch)
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    steps = _StandInSteps(builder, monkeypatch)
    steps.fail_at = "wic_post_clone"
    completed_before = _completed_steps(builder, checkpoint_dir)

    with _report_pipe(builder, monkeypatch) as reported:
        with pytest.raises(RuntimeError, match="stand-in failure in wic_post_clone"):
            builder._run_outer_stage(
                _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
            )
        # The failing step was reported, so the parent records the failure
        # against it; no later step started.
        assert reported() == list(_ONE_PROCESS_STEPS[:4])

    assert _completed_steps(builder, checkpoint_dir) == (
        *completed_before,
        *_ONE_PROCESS_STEPS[:3],
    )
    observations = checkpoint_dir / builder.GATE_BATTERY_SHADOW_DIRECTORY
    # The failing step saved what it observed on the way out, as a step
    # process does today.
    assert sorted(
        path.name for path in observations.glob("*.observations.json")
    ) == sorted(f"{stage}.observations.json" for stage in _ONE_PROCESS_STEPS[:4])


@pytest.mark.parametrize(
    "case", ("no variable", "another process's variable", "a step after the 16")
)
def test_a_step_without_its_parent_s_report_pipe_runs_one_step(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    # A named --stage run by hand, a variable copied from a process that is
    # not this one's parent, and a step outside the 16 each run one step and
    # report nothing.
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.delenv(_REPORT_VARIABLE, raising=False)
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch)
    stage = _ONE_PROCESS_STEPS[0]
    if case == "a step after the 16":
        for earlier in _ONE_PROCESS_STEPS:
            builder._run_outer_stage(_step_args(builder, checkpoint_dir, earlier))
        stage = "congressional_district_assignment"
    completed_before = _completed_steps(builder, checkpoint_dir)
    read_fd, write_fd = os.pipe()
    if case != "no variable":
        monkeypatch.setenv(
            _REPORT_VARIABLE,
            builder._step_report_environment(write_fd)[_REPORT_VARIABLE],
        )
    if case == "a step after the 16":
        monkeypatch.setattr(os, "getppid", os.getpid)

    builder._run_outer_stage(_step_args(builder, checkpoint_dir, stage))

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as reports:
        assert reports.read() == b""
    assert _completed_steps(builder, checkpoint_dir) == (*completed_before, stage)


@pytest.mark.parametrize(
    "value",
    (
        "not json",
        "[]",
        json.dumps({"fd": 3, "parent_pid": 1}),
        json.dumps({"fd": 3, "parent_pid": 1, "schema_version": 2}),
        json.dumps({"fd": "3", "parent_pid": 1, "schema_version": 1}),
        json.dumps({"fd": True, "parent_pid": 1, "schema_version": 1}),
        json.dumps({"fd": -1, "parent_pid": 1, "schema_version": 1}),
        json.dumps({"fd": 3, "parent_pid": 0, "schema_version": 1}),
        json.dumps({"fd": 3, "parent_pid": 1, "schema_version": 1, "steps": []}),
    ),
)
def test_a_malformed_report_variable_stops_the_step_process(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_REPORT_VARIABLE, value)
    with pytest.raises(SystemExit, match=_REPORT_VARIABLE):
        builder._step_report_fd(_ONE_PROCESS_STEPS[0])


def test_the_report_variable_names_this_process_and_the_pipe(
    builder: ModuleType,
) -> None:
    environment = builder._step_report_environment(7)
    assert json.loads(environment[_REPORT_VARIABLE]) == {
        "fd": 7,
        "parent_pid": os.getpid(),
        "schema_version": 1,
    }


# ---------------------------------------------------------------------------
# The parent: one child for the 16 steps, one stage change per step
# ---------------------------------------------------------------------------


class _Telemetry:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def transition_stage(self, stage_id: str, **_details: object) -> None:
        self.stages.append(stage_id)


class _ChildProcesses:
    """Stands in for ``subprocess.run`` and ``subprocess.Popen``.

    Records each process the parent starts, as ``(first step, environment,
    started with Popen)``. A Popen child writes ``reports`` (by default every
    step from its first to the last of the 16) to the pipe its report
    variable names, then exits with ``popen_returncode``.
    """

    def __init__(self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch):
        self.started: list[tuple[str, dict[str, str], bool]] = []
        self.reports: list[str] | None = None
        self.popen_returncode = 0
        self.killed = False
        processes = self

        def run(command, *, check, env):
            assert not check
            processes.started.append((_first_step(command), dict(env), False))
            return SimpleNamespace(returncode=0)

        class Popen:
            def __init__(self, command, *, env, pass_fds):
                first = _first_step(command)
                processes.started.append((first, dict(env), True))
                payload = json.loads(env[_REPORT_VARIABLE])
                assert payload["parent_pid"] == os.getpid()
                assert tuple(pass_fds) == (payload["fd"],)
                reports = processes.reports
                if reports is None:
                    reports = list(
                        _ONE_PROCESS_STEPS[_ONE_PROCESS_STEPS.index(first) :]
                    )
                # The parent closes its copy of the write end once Popen
                # returns, which ends the reports.
                for name in reports:
                    os.write(payload["fd"], f"{name}\n".encode())
                self.returncode: int | None = None

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info) -> bool:
                return False

            def wait(self) -> int:
                self.returncode = processes.popen_returncode
                return self.returncode

            def kill(self) -> None:
                processes.killed = True

        monkeypatch.setattr(builder.subprocess, "run", run)
        monkeypatch.setattr(builder.subprocess, "Popen", Popen)

    def first_steps(self) -> list[str]:
        return [first for first, _environment, _popen in self.started]


def _first_step(command: list[str]) -> str:
    return command[command.index("--stage") + 1]


def _stand_in_parent(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    completed: tuple[str, ...],
    stages_run: list[list[str]] | None = None,
) -> None:
    class CompletedPrefix:
        def __init__(self, *_args, **_kwargs) -> None:
            self.context = SimpleNamespace(completed=completed)

    monkeypatch.setattr(builder, "StageRuntime", CompletedPrefix)
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(
        builder, "_stage_cli_args", lambda _args, stage: ["--stage", stage]
    )
    if stages_run is not None:
        shadow = SimpleNamespace(stop=lambda _error: None, finish=lambda: None)
        monkeypatch.setattr(
            builder, "_final_gate_shadow_directory", lambda _args: Path(".")
        )
        monkeypatch.setattr(
            builder, "_base_gate_shadow", lambda _args, _directory: shadow
        )
        monkeypatch.setattr(
            builder,
            "_load_stage_gate_shadows",
            lambda _shadow, _args, stages: stages_run.append(list(stages)),
        )


@pytest.mark.parametrize(
    "completed_count",
    (0, 5, 5 + 11, 5 + 15),
    ids=(
        "fresh",
        "after qrf_finalization",
        "inside the 16",
        "only the last of the 16 left",
    ),
)
def test_the_parent_starts_one_child_for_the_16_and_records_each_step(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    completed_count: int,
) -> None:
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, steps[:completed_count])
    processes = _ChildProcesses(builder, monkeypatch)
    telemetry = _Telemetry()

    builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), telemetry)

    remaining = steps[completed_count:]
    last_of_16 = steps.index(_ONE_PROCESS_STEPS[-1])
    first_remaining_of_16 = max(completed_count, steps.index(_ONE_PROCESS_STEPS[0]))
    one_child = last_of_16 > first_remaining_of_16
    expected_first_steps = [
        stage
        for index, stage in enumerate(steps)
        if index >= completed_count
        and not (one_child and first_remaining_of_16 < index <= last_of_16)
    ]
    assert processes.first_steps() == expected_first_steps
    for first, environment, popen in processes.started:
        # Only a child that runs more than one step gets a report pipe.
        assert popen == (one_child and first == steps[first_remaining_of_16])
        assert (_REPORT_VARIABLE in environment) == popen
        assert environment["PYTHONHASHSEED"] == "0"
    # One stage change per step, under the step's own name.
    assert telemetry.stages == list(remaining)


def test_a_report_variable_the_parent_inherited_reaches_no_child(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, ())
    processes = _ChildProcesses(builder, monkeypatch)
    inherited = json.dumps({"fd": 9, "parent_pid": os.getppid(), "schema_version": 1})
    monkeypatch.setenv(_REPORT_VARIABLE, inherited)

    builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), _Telemetry())

    assert len(processes.started) == len(steps) - 15
    for first, environment, popen in processes.started:
        if popen:
            assert environment[_REPORT_VARIABLE] != inherited
        else:
            assert _REPORT_VARIABLE not in environment, first


def test_a_failed_child_is_recorded_against_the_step_that_failed(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    steps = builder.PIPELINE_STEPS
    completed = steps[: steps.index(_ONE_PROCESS_STEPS[0])]
    stages_run: list[list[str]] = []
    _stand_in_parent(builder, monkeypatch, completed, stages_run)
    processes = _ChildProcesses(builder, monkeypatch)
    processes.reports = list(_ONE_PROCESS_STEPS[:3])
    processes.popen_returncode = 1
    telemetry = _Telemetry()

    with pytest.raises(
        SystemExit, match=r"stage 'qbi_reconciliation' failed with exit code 1"
    ):
        builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), telemetry)

    assert telemetry.stages == list(_ONE_PROCESS_STEPS[:3])
    assert processes.first_steps() == [_ONE_PROCESS_STEPS[0]]
    # The parent loads the observations of the steps that started, and only
    # those: the failing step's included.
    assert stages_run == [[*completed, *_ONE_PROCESS_STEPS[:3]]]


def test_a_finished_child_has_every_step_s_observations_loaded(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    steps = builder.PIPELINE_STEPS
    stages_run: list[list[str]] = []
    _stand_in_parent(builder, monkeypatch, (), stages_run)
    _ChildProcesses(builder, monkeypatch)

    builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), _Telemetry())

    assert stages_run == [list(steps)]


@pytest.mark.parametrize(
    ("reports", "problem"),
    (
        (
            ["qbi_reconciliation"],
            r"reported 'qbi_reconciliation'.*expected 'capital_gains_tail_transfer'",
        ),
        (
            ["capital_gains_tail_transfer", "capital_gains_tail_transfer"],
            r"reported 'capital_gains_tail_transfer'.*expected 'capital_gain_distributions'",
        ),
        (
            [*_ONE_PROCESS_STEPS, "congressional_district_assignment"],
            r"reported 'congressional_district_assignment' after",
        ),
        (
            ["capital_gains_tail_transfer\x00"],
            r"reported 'capital_gains_tail_transfer\\x00'",
        ),
    ),
)
def test_a_child_that_reports_another_step_is_stopped(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reports: list[str],
    problem: str,
) -> None:
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, steps[: steps.index(_ONE_PROCESS_STEPS[0])])
    processes = _ChildProcesses(builder, monkeypatch)
    processes.reports = reports

    with pytest.raises(RuntimeError, match=problem):
        builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), _Telemetry())

    assert processes.killed
    assert processes.first_steps() == [_ONE_PROCESS_STEPS[0]]


def test_a_child_that_exits_cleanly_before_its_last_step_is_refused(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, steps[: steps.index(_ONE_PROCESS_STEPS[0])])
    processes = _ChildProcesses(builder, monkeypatch)
    processes.reports = list(_ONE_PROCESS_STEPS[:5])

    with pytest.raises(
        RuntimeError, match=r"exited cleanly before step 'prior_year_income_post_clone'"
    ):
        builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), _Telemetry())

    # No later step's process starts.
    assert processes.first_steps() == [_ONE_PROCESS_STEPS[0]]


#: Run in a real child process: load the builder, find the report pipe the
#: parent opened, and report each of the 16 steps from the first.
_REPORTING_CHILD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("build_us_puf_support_base", sys.argv[1])
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
report_fd = builder._step_report_fd(sys.argv[2])
assert report_fd is not None, "no report pipe"
for step in builder.ONE_PROCESS_STEPS:
    builder._report_step_start(report_fd, step)
"""


def test_a_real_child_finds_the_pipe_its_parent_opened(
    builder: ModuleType, tmp_path: Path
) -> None:
    telemetry = _Telemetry()
    stages_run: list[str] = []
    command = [
        sys.executable,
        "-c",
        _REPORTING_CHILD,
        str(_TOOL),
        _ONE_PROCESS_STEPS[0],
    ]

    builder._run_one_process_steps(
        command,
        dict(os.environ),
        _ONE_PROCESS_STEPS,
        telemetry=telemetry,
        stages_run=stages_run,
    )

    # The first step's stage change is recorded before the child starts.
    assert telemetry.stages == list(_ONE_PROCESS_STEPS)
    assert stages_run == list(_ONE_PROCESS_STEPS)


def test_the_parent_stops_its_child_when_it_is_interrupted(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # As subprocess.run does: an exception in the parent while the child
    # runs (a SIGTERM the build turns into one, say) stops the child.
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, steps[: steps.index(_ONE_PROCESS_STEPS[0])])
    processes = _ChildProcesses(builder, monkeypatch)

    class ParentInterruptedError(Exception):
        pass

    class InterruptedTelemetry(_Telemetry):
        def transition_stage(self, stage_id: str, **details: object) -> None:
            super().transition_stage(stage_id, **details)
            if stage_id == _ONE_PROCESS_STEPS[2]:
                raise ParentInterruptedError

    with pytest.raises(ParentInterruptedError):
        builder._run_staged_all(
            SimpleNamespace(checkpoint_dir=tmp_path), InterruptedTelemetry()
        )
    assert processes.killed
    assert processes.first_steps() == [_ONE_PROCESS_STEPS[0]]
