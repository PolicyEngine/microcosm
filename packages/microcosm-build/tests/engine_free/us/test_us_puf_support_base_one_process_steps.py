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
replaced with deterministic ones that write a digest of every value with
its type, every dtype, the indexes, the column labels, the strata, the
weights, the mass log and the frame metadata, and a floating-point sum of
each person's numeric values, so a frame passed in memory that differed
from the one a checkpoint read gives would write different bytes.
``capital_gain_distributions`` runs its real code, and the stand-ins
evaluate each step's ``us/gates.json`` check through the shadow evaluation,
so the saved observations hold entries. One test runs the 16 steps in 16
real step processes and in one real child, and compares every file they
write. The parent-side tests replace ``subprocess.run`` and
``subprocess.Popen``; one starts a real child.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build import GateResult, outer_stage_runtime
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


def _row_values(table: pd.DataFrame) -> np.ndarray:
    """Per row, the sum of its numeric and boolean values, computed with
    floating-point arithmetic across the columns."""

    return (
        table.select_dtypes(["number", "bool"])
        .astype(np.float64)
        .sum(axis=1)
        .to_numpy(np.float64)
    )


def _frame_content_digest(frame: Frame) -> int:
    """A digest of everything a step could read from ``frame``.

    Every value of every column, index and the strata with its type and
    ``repr``; the number of distinct objects in each (``set`` counts two
    NaN objects twice); every dtype, including the column labels'; the
    weights; the mass log; and the frame metadata.
    """

    digest = hashlib.sha256()

    def add(text: str) -> None:
        digest.update(text.encode("utf-8") + b"\0")

    def add_values(label: str, values: pd.Series | pd.Index) -> None:
        objects = list(values.to_numpy(dtype=object))
        add(
            f"{label}: {values.dtype}, {len(objects)} values, {len(set(objects))} distinct"
        )
        for value in objects:
            add(f"{type(value).__module__}.{type(value).__qualname__}:{value!r}")

    tables = {
        **{entity: frame.table(entity) for entity in frame.entities},
        **{link: frame.link(link) for link in frame.links},
    }
    for name, table in tables.items():
        add(f"{name} column labels: {table.columns.dtype}, {list(table.columns)!r}")
        add_values(f"{name} index", table.index)
        for column in table.columns:
            add_values(f"{name}.{column}", table[column])
    add(f"strata name: {frame.strata.name!r}")
    add_values("strata", frame.strata)
    add_values("strata index", frame.strata.index)
    for entity in frame.weighted_entities:
        weights = frame.weights_for(entity)
        add(f"{entity} weights: {weights.kind}, {weights.values.dtype}")
        digest.update(np.ascontiguousarray(weights.values).tobytes())
    add(f"mass log: {frame.mass_log!r}")
    add(f"metadata: {sorted(dict(frame.metadata or {}).items())!r}")
    return int.from_bytes(digest.digest()[:7], "big")


def _read_from_everything(frame: Frame, extra: float) -> dict[str, np.ndarray]:
    """Two person columns computed from ``frame``: its person rows' sums,
    and a digest of everything in it (``_frame_content_digest``)."""

    person = frame.table("person")
    return {
        "row_values": _row_values(person) * 1.000001 + extra,
        "content": np.full(len(person), _frame_content_digest(frame), dtype=np.int64),
    }


#: The ``us/gates.json`` entry each stand-in step evaluates, one per step that
#: evaluates any in ``_post_qrf_frame_stage``.
_CHECK_ENTRY_BY_STEP = {
    "wic_post_clone": "us_base_wic_claim_post_clone",
    "prior_year_income_post_clone": "us_base_prior_year_income",
    "child_support_post_clone": "us_base_child_support",
    "disability_benefits_post_clone": "us_base_disability_benefits",
    "workers_compensation_post_clone": "us_base_workers_compensation",
    "weeks_unemployed_post_clone": "us_base_weeks_unemployed",
    "childcare_post_clone": "us_base_childcare",
    "adult_care_post_clone": "us_base_adult_care",
    "energy_subsidy_post_clone": "us_base_energy_subsidy",
    "retirement_contributions_post_clone": "us_base_retirement_contributions",
    "retirement_distributions_post_clone": "us_base_retirement_distributions",
    "education_inputs_post_clone": "us_base_education_inputs",
}


class _StandInSteps:
    """Deterministic stand-ins for the step functions that need real inputs.

    Each adds one person column computed from every value in the frame it
    was given. With ``record``, each also keeps the frames it was given and
    returned, by step.
    """

    def __init__(
        self,
        builder: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        *,
        record: bool = True,
    ):
        self.inputs: dict[str, Frame] = {}
        self.outputs: dict[str, Frame] = {}
        self.fail_at: str | None = None
        self.record = record

        def tail_transfer(_args, frame):
            self._record_input(builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME, frame)
            read = _read_from_everything(frame, 0.0)
            value = read["row_values"]
            result = _with_person_columns(
                frame,
                {
                    "tail_value": value,
                    "tail_content": read["content"],
                    "tail_label": pd.Series(
                        np.where(value > np.median(value), "high", "low"),
                        index=frame.table("person").index,
                        dtype="str",
                    ),
                },
            )
            self._record_output(builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME, result)
            return result, {"note": "tail"}

        def post_qrf_step(stage, _args, frame, stage_metadata):
            self._record_input(stage, frame)
            entry_id = _CHECK_ENTRY_BY_STEP.get(stage)
            if entry_id is not None:
                # Today's verdict is a stand-in; the shadow evaluates the
                # entry's own binding on the frame and saves what it saw.
                builder._shadow_frame_gate(
                    entry_id,
                    lambda: GateResult(
                        name=entry_id, passed=True, failures=(), details={}
                    ),
                    frame,
                )
            read = _read_from_everything(frame, float(len(stage_metadata)))
            result = _with_person_columns(
                frame,
                {
                    f"{stage}_value": read["row_values"],
                    f"{stage}_content": read["content"],
                },
            )
            self._record_output(stage, result)
            return result, {"signals": {"stage": stage}}

        distributions = builder._capital_gain_distributions_stage

        def recorded_distributions(args, frame):
            self._record_input(builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME, frame)
            result = distributions(args, frame)
            self._record_output(
                builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME, result[0]
            )
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
        if self.record:
            self.inputs[stage] = frame

    def _record_output(self, stage: str, frame: Frame) -> None:
        if self.record:
            self.outputs[stage] = frame


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
    # As pass_fds leaves it in a step process.
    os.set_inheritable(write_fd, True)
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

        reported.write_fd = write_fd  # type: ignore[attr-defined]
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


def _observation_names(builder: ModuleType) -> list[str]:
    return [
        f"checkpoints/{builder.GATE_BATTERY_SHADOW_DIRECTORY}/{stage}.observations.json"
        for stage in _ONE_PROCESS_STEPS
    ]


def _checkpoint_names(builder: ModuleType) -> list[str]:
    pipeline = builder.OUTER_STAGE_PIPELINE
    return [
        f"checkpoints/{pipeline.index(stage):03d}_{stage}.frame.h5"
        for stage in _ONE_PROCESS_STEPS
    ]


def _assert_observations_saved(run_dir: Path, builder: ModuleType) -> None:
    """Each stand-in step that evaluates a check saved what the shadow saw."""

    for stage, name in zip(
        _ONE_PROCESS_STEPS, _observation_names(builder), strict=True
    ):
        payload = json.loads((run_dir / name).read_text())
        assert bool(payload["observations"]) == (stage in _CHECK_ENTRY_BY_STEP), stage


def test_one_child_hands_each_step_the_frame_its_checkpoint_holds(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # In this process, so the frames can be compared: the 16 steps run once
    # one at a time, each reading its input from disk as a step process
    # does, and once as the one child runs them. Both runs use the same
    # directory, so the paths in the run context and the saved observations
    # are the same.
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
    assert set(_checkpoint_names(builder)) | set(_observation_names(builder)) <= set(
        files_one_child
    )
    assert f"checkpoints/{OUTER_STAGE_CONTEXT_FILENAME}" in files_one_child
    _assert_observations_saved(tmp_path / "run", builder)
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


#: Run in a real step process: load the builder and this test module, put in
#: the stand-in steps, and run ``_run_outer_stage`` for one ``--stage``, as
#: the tool's command does after parsing its arguments.
_STAND_IN_STEP_PROCESS = """
import importlib.util, sys
from contextlib import nullcontext
from pathlib import Path

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load("build_us_puf_support_base", sys.argv[1])
tests = load("one_process_steps_test_module", sys.argv[2])
patch = pytest.MonkeyPatch()
tests._StandInSteps(builder, patch, record=False)
patch.setattr(builder, "_stage_run_config", lambda _args: {})
patch.setattr(builder, "profile_stage", lambda *_args, **_kwargs: nullcontext())
builder._run_outer_stage(tests._step_args(builder, Path(sys.argv[3]), sys.argv[4]))
"""


def test_one_child_writes_the_same_files_as_16_separate_step_processes(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The finish condition of task 1.5 · 7: the 16 steps in 16 real step
    # processes, each reading its input checkpoint from disk, and in one
    # real child started through the parent's own code, which reads only the
    # first step's input from disk. Every file they write (the 16
    # checkpoints, the run context and the 16 saved observation files) has
    # the same SHA-256. Each process gets this process's policyengine-us
    # lock-file check record, as a --stage all parent's step processes do,
    # so it starts without repeating the check.
    arm_us_tool_gate_shadows(monkeypatch)
    monkeypatch.delenv(_REPORT_VARIABLE, raising=False)
    repository = str(paths_for("microcosm-build").repository)
    environment = builder._staged_subprocess_environment()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (repository, environment.get("PYTHONPATH")) if path
    )
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"

    def step_process(stage: str) -> list[str]:
        return [
            sys.executable,
            "-c",
            _STAND_IN_STEP_PROCESS,
            str(_TOOL),
            str(Path(__file__).resolve()),
            str(checkpoint_dir),
            stage,
        ]

    _complete_steps_before_the_16(builder, checkpoint_dir)
    for stage in _ONE_PROCESS_STEPS:
        subprocess.run(step_process(stage), env=environment, check=True)
    files_16_processes = _files_and_digests(run_dir)
    shutil.rmtree(run_dir)

    _complete_steps_before_the_16(builder, checkpoint_dir)
    telemetry = _Telemetry()
    stages_run: list[str] = []
    builder._run_one_process_steps(
        step_process(_ONE_PROCESS_STEPS[0]),
        environment,
        _ONE_PROCESS_STEPS,
        telemetry=telemetry,
        stages_run=stages_run,
    )
    files_one_child = _files_and_digests(run_dir)

    assert telemetry.stages == stages_run == list(_ONE_PROCESS_STEPS)
    assert set(_checkpoint_names(builder)) | set(_observation_names(builder)) <= set(
        files_one_child
    )
    assert f"checkpoints/{OUTER_STAGE_CONTEXT_FILENAME}" in files_one_child
    _assert_observations_saved(run_dir, builder)
    assert files_one_child == files_16_processes


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


@pytest.mark.parametrize("sources", ("unchanged", "changed after three steps"))
def test_the_child_compares_the_run_config_with_the_locked_one_before_each_step(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sources: str
) -> None:
    # As each step process does when it starts: a change to the builder's
    # source files (a git pull during the run, say), its dependencies or its
    # thread settings after the run started stops the run before the next
    # step, which is the step the parent records the failure against.
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch, record=False)
    completed_before = _completed_steps(builder, checkpoint_dir)
    computed_for: list[str] = []

    def run_config(step_args) -> dict[str, object]:
        computed_for.append(step_args.stage)
        if sources == "changed after three steps" and len(computed_for) > 3:
            return {"builder_code_identity": {"source_sha256": "e" * 64}}
        return {}

    monkeypatch.setattr(builder, "_stage_run_config", run_config)
    first = _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])

    with _report_pipe(builder, monkeypatch) as reported:
        if sources == "unchanged":
            builder._run_outer_stage(first)
            ran = _ONE_PROCESS_STEPS
        else:
            with pytest.raises(ValueError, match="run_config differs"):
                builder._run_outer_stage(first)
            ran = _ONE_PROCESS_STEPS[:3]
        started = _ONE_PROCESS_STEPS[: len(computed_for)]
        assert reported() == list(started)

    assert computed_for == list(_ONE_PROCESS_STEPS[: len(ran) + (len(ran) < 16)])
    assert _completed_steps(builder, checkpoint_dir) == (*completed_before, *ran)


def _stored_objects(frame: Frame) -> list[object]:
    """The tables and strata ``frame`` stores. ``Frame`` has no
    ``__weakref__`` slot, so these stand for it: once they are gone, so is
    the frame."""

    return [
        *(frame.table(entity) for entity in frame.entities),
        *(frame.link(name) for name in frame.links),
        frame.strata,
    ]


def _column_buffers(frame: Frame) -> list[np.ndarray]:
    """The NumPy arrays that own the memory of ``frame``'s NumPy-typed columns."""

    buffers: list[np.ndarray] = []
    for table in (
        *(frame.table(entity) for entity in frame.entities),
        *(frame.link(name) for name in frame.links),
    ):
        for column in table.columns:
            series = table[column]
            if not isinstance(series.dtype, np.dtype):
                continue
            owner = series.to_numpy()
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            buffers.append(owner)
    return buffers


def test_each_step_frees_the_frame_it_was_handed_before_its_checkpoint_write(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The frame a step takes from the step before it is freed before the
    # step writes its own checkpoint, as a frame read from disk is (1.5 · 5).
    # Garbage collection is off while the steps run, so only reference
    # counting can free it; the shadow check run is on.
    arm_us_tool_gate_shadows(monkeypatch)
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch, record=False)
    inputs: dict[str, tuple[list[weakref.ref], list[weakref.ref]]] = {}
    predecessor_checkpoint = builder._predecessor_checkpoint

    def recording_predecessor(step_args, runtime, carried):
        checkpoint = predecessor_checkpoint(step_args, runtime, carried)
        inputs[step_args.stage] = (
            [weakref.ref(item) for item in _stored_objects(checkpoint.frame)],
            [weakref.ref(buffer) for buffer in _column_buffers(checkpoint.frame)],
        )
        return checkpoint

    alive_at_write: dict[str, tuple[int, int]] = {}
    write_frame_checkpoint = outer_stage_runtime.write_frame_checkpoint

    def recording_write(path, frame, **kwargs):
        stage = kwargs["metadata"]["stage"]
        objects, buffers = inputs[stage]
        result_buffers = {id(buffer) for buffer in _column_buffers(frame)}
        live_buffers = [reference() for reference in buffers]
        alive_at_write[stage] = (
            sum(reference() is not None for reference in objects),
            sum(
                buffer is not None and id(buffer) not in result_buffers
                for buffer in live_buffers
            ),
        )
        del live_buffers
        return write_frame_checkpoint(path, frame, **kwargs)

    monkeypatch.setattr(builder, "_predecessor_checkpoint", recording_predecessor)
    monkeypatch.setattr(outer_stage_runtime, "write_frame_checkpoint", recording_write)

    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with _report_pipe(builder, monkeypatch) as reported:
            builder._run_outer_stage(
                _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
            )
            reported()
    finally:
        if collection_was_enabled:
            gc.enable()

    # Stored objects alive, and column arrays alive that the result does not
    # hold, at each step's checkpoint write: none.
    assert alive_at_write == {stage: (0, 0) for stage in _ONE_PROCESS_STEPS}


def test_the_child_keeps_the_report_pipe_from_processes_it_starts(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A process the child started that inherited the write end, and outlived
    # the child, would keep the parent waiting for the end of the reports.
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch, record=False)

    with _report_pipe(builder, monkeypatch) as reported:
        assert os.get_inheritable(reported.write_fd)
        builder._run_outer_stage(
            _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
        )
        assert not os.get_inheritable(reported.write_fd)
        reported()


def _with_numpy_scalar(frame: Frame, where: str, value: object) -> Frame:
    """``frame`` with NumPy scalars of ``value``'s type in a new person object
    column, a table's index or the strata, as a step's own code might leave
    them. Every value there has that type, as the checkpoint writer refuses
    an object column that mixes value families."""

    scalar_type = type(value)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    strata = frame.strata
    if where == "column":
        person = tables["person"]
        person["numpy_scalars"] = pd.Series(
            [value] * len(person), index=person.index, dtype=object
        )
    elif where == "python-backed str column":
        person = tables["person"]
        person["numpy_scalars"] = pd.Series(
            [value] * len(person),
            index=person.index,
            dtype=pd.StringDtype("python", na_value=np.nan),
        )
    elif where == "strata index":
        strata = pd.Series(
            strata.to_numpy(),
            index=pd.Index(
                [scalar_type(position) for position in range(len(strata))],
                dtype=object,
            ),
            name=strata.name,
            dtype=strata.dtype,
        )
    elif where == "index":
        household = tables["household"]
        household.index = pd.Index(
            [scalar_type(position) for position in range(len(household))],
            dtype=object,
        )
    else:
        strata = pd.Series(
            [scalar_type(stratum) for stratum in strata],
            index=strata.index,
            name=strata.name,
            dtype=object,
        )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


@pytest.mark.parametrize(
    ("where", "value", "type_name"),
    (
        ("column", np.float32(0.5), "numpy.float32"),
        ("column", np.int64(3), "numpy.int64"),
        ("column", np.str_("HEAD"), "numpy.str_"),
        ("column", np.bool_(True), "numpy.bool"),
        ("index", np.int64(0), "numpy.int64"),
        ("strata", np.str_("asec_2024"), "numpy.str_"),
        ("python-backed str column", np.str_("HEAD"), "numpy.str_"),
        ("strata index", np.int64(0), "numpy.int64"),
    ),
)
def test_the_child_stops_before_handing_over_values_a_checkpoint_read_changes(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    where: str,
    value: object,
    type_name: str,
) -> None:
    # Reading a checkpoint turns NumPy scalars in object columns, an object
    # index or object strata into Python values. A frame handed over in
    # memory keeps them, so the next step would compute on other values
    # than in a run with one process per step. The child stops instead, once
    # the step that left them has completed: a resume reads its checkpoint.
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch, record=False)
    tail_transfer = builder._capital_gains_tail_transfer_stage

    def tail_leaving_numpy_scalars(args, frame):
        result, metadata = tail_transfer(args, frame)
        return _with_numpy_scalar(result, where, value), metadata

    monkeypatch.setattr(
        builder, "_capital_gains_tail_transfer_stage", tail_leaving_numpy_scalars
    )
    completed_before = _completed_steps(builder, checkpoint_dir)
    location = {
        "column": r"person column 'numpy_scalars'",
        "python-backed str column": r"person column 'numpy_scalars'",
        "index": r"household index",
        "strata": r"strata",
        "strata index": r"strata index",
    }[where]

    with _report_pipe(builder, monkeypatch) as reported:
        with pytest.raises(
            RuntimeError,
            match=rf"'capital_gains_tail_transfer' left {type_name} values in the {location}",
        ):
            builder._run_outer_stage(
                _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
            )
        assert reported() == list(_ONE_PROCESS_STEPS[:2])

    assert _completed_steps(builder, checkpoint_dir) == (
        *completed_before,
        _ONE_PROCESS_STEPS[0],
    )


def test_the_last_of_the_16_may_leave_numpy_scalars(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Its frame is not handed over: the next step's process reads its
    # checkpoint, as with one process per step.
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _complete_steps_before_the_16(builder, checkpoint_dir)
    _StandInSteps(builder, monkeypatch, record=False)
    post_qrf_step = builder._post_qrf_frame_stage

    def last_step_leaving_numpy_scalars(stage, args, frame, stage_metadata):
        result, metadata = post_qrf_step(stage, args, frame, stage_metadata)
        if stage == _ONE_PROCESS_STEPS[-1]:
            result = _with_numpy_scalar(result, "column", np.float32(0.5))
        return result, metadata

    monkeypatch.setattr(
        builder, "_post_qrf_frame_stage", last_step_leaving_numpy_scalars
    )

    with _report_pipe(builder, monkeypatch) as reported:
        builder._run_outer_stage(
            _step_args(builder, checkpoint_dir, _ONE_PROCESS_STEPS[0])
        )
        assert reported() == list(_ONE_PROCESS_STEPS)

    assert _completed_steps(builder, checkpoint_dir)[-1] == _ONE_PROCESS_STEPS[-1]


@pytest.mark.parametrize("descriptor", ("a file", "a closed descriptor"))
def test_a_report_variable_that_names_no_pipe_stops_the_step_process(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    descriptor: str,
) -> None:
    # A variable copied into a process whose parent happens to have the
    # recorded process ID (process IDs repeat in fresh containers) must not
    # make it write step names into an unrelated file.
    monkeypatch.setattr(os, "getppid", os.getpid)
    with open(tmp_path / "not-a-pipe", "wb") as not_a_pipe:
        fd = not_a_pipe.fileno()
        if descriptor == "a file":
            monkeypatch.setenv(
                _REPORT_VARIABLE, builder._step_report_environment(fd)[_REPORT_VARIABLE]
            )
            with pytest.raises(SystemExit, match=rf"{_REPORT_VARIABLE} .*not a pipe"):
                builder._step_report_fd(_ONE_PROCESS_STEPS[0])
    if descriptor == "a closed descriptor":
        monkeypatch.setenv(
            _REPORT_VARIABLE, builder._step_report_environment(fd)[_REPORT_VARIABLE]
        )
        with pytest.raises(SystemExit, match=rf"{_REPORT_VARIABLE} .*not open"):
            builder._step_report_fd(_ONE_PROCESS_STEPS[0])
    assert (tmp_path / "not-a-pipe").read_bytes() == b""


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
        self.events: list[tuple[str, float | None]] = []
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

            def wait(self, timeout: float | None = None) -> int:
                processes.events.append(("wait", timeout))
                self.returncode = processes.popen_returncode
                return self.returncode

            def kill(self) -> None:
                processes.events.append(("kill", None))
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


def test_a_child_that_reports_no_step_is_refused(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A child that found no report pipe runs one step and reports nothing.
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, steps[: steps.index(_ONE_PROCESS_STEPS[0])])
    processes = _ChildProcesses(builder, monkeypatch)
    processes.reports = []

    with pytest.raises(
        RuntimeError,
        match=rf"exited cleanly without reporting any step.*{_REPORT_VARIABLE}",
    ):
        builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path), _Telemetry())

    assert processes.first_steps() == [_ONE_PROCESS_STEPS[0]]


def test_after_a_ctrl_c_the_parent_gives_its_child_a_moment_before_stopping_it(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # As subprocess.run does: the child got the same SIGINT, so the parent
    # waits briefly for it to finish its own handling (the failing step
    # saving its observations and profile record) before stopping it.
    steps = builder.PIPELINE_STEPS
    _stand_in_parent(builder, monkeypatch, steps[: steps.index(_ONE_PROCESS_STEPS[0])])
    processes = _ChildProcesses(builder, monkeypatch)

    class InterruptedTelemetry(_Telemetry):
        def transition_stage(self, stage_id: str, **details: object) -> None:
            super().transition_stage(stage_id, **details)
            if stage_id == _ONE_PROCESS_STEPS[2]:
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        builder._run_staged_all(
            SimpleNamespace(checkpoint_dir=tmp_path), InterruptedTelemetry()
        )
    assert processes.events == [("wait", 0.25), ("kill", None)]


def test_the_report_pipe_never_takes_a_standard_stream_descriptor(
    builder: ModuleType,
) -> None:
    # os.pipe returns the lowest free descriptors. In a parent started with
    # standard output and error closed, those are 1 and 2, and a write end
    # at 2 would become the child's standard error.
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    try:
        os.close(1)
        os.close(2)
        read_fd, write_fd = builder._report_pipe_fds()
    finally:
        for fd, copy_of_fd in saved.items():
            os.dup2(copy_of_fd, fd)
            os.close(copy_of_fd)
    try:
        assert min(read_fd, write_fd) > 2
        assert not os.get_inheritable(read_fd)
        assert not os.get_inheritable(write_fd)
        os.write(write_fd, b"step\n")
        assert os.read(read_fd, 5) == b"step\n"
    finally:
        os.close(read_fd)
        os.close(write_fd)


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
