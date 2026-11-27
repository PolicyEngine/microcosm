"""A checkpointed base run hashes its input files once.

``tools/build_us_puf_support_base.py --stage all`` starts one step process
per pipeline step, except for the 16 steps from the capital-gains tail
transfer to the education inputs, which share one step process. Each step
process opens the checkpoint directory with
its own run config, which holds the SHA-256 of the PUF H5, the PUF CSV and
the builder's source files. The parent computes that run config once,
compares it with the one locked in the checkpoint directory before any step
process starts, and passes each input file's digest, under its resolved
path, to each step process in the environment variable
``MICROCOSM_US_BASE_PARENT_DIGESTS``. A step process uses a digest only for
the same path, and only when the process that set the variable is its
parent; it hashes every other file itself. It always hashes the builder's
source files itself, so a change to them after the parent started stops the
next step.

Most of these tests replace ``subprocess.run`` and ``subprocess.Popen`` with
stand-ins that record each step process's command and environment, then
compute each step process's run config in
this process, as that step process would, while recording what gets hashed.
One starts a real step process to show that it finds this process as its
parent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from microcosm.build import code_identity
from microcosm.build.outer_stage_runtime import OUTER_STAGE_CONTEXT_FILENAME
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_TOOL = _TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"
_DIGESTS_VARIABLE = "MICROCOSM_US_BASE_PARENT_DIGESTS"
_REPORT_VARIABLE = "MICROCOSM_US_BASE_STEP_REPORT"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", _TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder() -> ModuleType:
    return _load_builder()


class _Hashing:
    """Records each file the tool's ``_sha256`` hashes, still hashing it, and
    counts ``builder_code_identity`` calls.

    Hashing inside other modules (the ladder loader, for one) is not
    recorded. ``builder_code_identity`` is a stand-in with the real one's
    arguments that returns ``source_sha256`` as the builder's source digest,
    so a test can change the builder's code by changing it.
    """

    def __init__(self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch):
        self.files: list[Path] = []
        self.code_identity_calls = 0
        self.source_sha256 = "c" * 64
        hash_file = builder._sha256

        def recording_sha256(path: Path) -> str:
            self.files.append(Path(path).resolve())
            return hash_file(path)

        def stand_in_code_identity(
            repo_root: Path,
            *,
            tool_path: Path,
            distributions: tuple[str, ...],
        ) -> dict[str, object]:
            self.code_identity_calls += 1
            return {
                "dependency_versions": {name: "1.0" for name in distributions},
                "python": "stand-in",
                "source_sha256": self.source_sha256,
            }

        monkeypatch.setattr(builder, "_sha256", recording_sha256)
        monkeypatch.setattr(
            code_identity, "builder_code_identity", stand_in_code_identity
        )

    def clear(self) -> None:
        self.files.clear()
        self.code_identity_calls = 0


class _StepProcesses:
    """Stands in for ``subprocess.run`` and ``subprocess.Popen``: records each
    step process the parent would start.

    The parent starts the step process for the 16 steps from the
    capital-gains tail transfer to the education inputs with ``Popen``; the
    stand-in reports each of its steps on the pipe the parent opened, as
    that step process would.
    """

    def __init__(self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch):
        self.started: list[tuple[list[str], dict[str, str]]] = []
        processes = self

        def record_step(command, *, check, env):
            assert not check
            self.started.append((list(command), dict(env)))
            return SimpleNamespace(returncode=0)

        class RecordedPopen:
            def __init__(self, command, *, env, pass_fds):
                processes.started.append((list(command), dict(env)))
                report_fd = json.loads(env[_REPORT_VARIABLE])["fd"]
                assert tuple(pass_fds) == (report_fd,)
                first = builder.ONE_PROCESS_STEPS.index(_step_name(command))
                for step in builder.ONE_PROCESS_STEPS[first:]:
                    os.write(report_fd, f"{step}\n".encode())

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info) -> bool:
                return False

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                raise AssertionError("the parent stopped a step process")

        monkeypatch.setattr(builder.subprocess, "run", record_step)
        monkeypatch.setattr(builder.subprocess, "Popen", RecordedPopen)


def _first_steps_of_step_processes(
    builder: ModuleType, pipeline_steps: tuple[str, ...]
) -> list[str]:
    """The step each step process of a fresh ``--stage all`` run starts at.

    The 16 steps from the capital-gains tail transfer to the education
    inputs share one step process, which starts at the first of them.
    """

    return [
        step for step in pipeline_steps if step not in builder.ONE_PROCESS_STEPS[1:]
    ]


@contextmanager
def _as_step_process(
    monkeypatch: pytest.MonkeyPatch, environment: Mapping[str, str]
) -> Iterator[None]:
    """Give the body the digests variable a step process started with
    ``environment`` would see, and make this process its parent.

    Only that variable and ``os.getppid`` change; the rest of this process's
    environment stays as it is.
    """

    with monkeypatch.context() as step:
        if _DIGESTS_VARIABLE in environment:
            step.setenv(_DIGESTS_VARIABLE, environment[_DIGESTS_VARIABLE])
        else:
            step.delenv(_DIGESTS_VARIABLE, raising=False)
        step.setattr(os, "getppid", os.getpid)
        yield


def _write_inputs(tmp_path: Path) -> SimpleNamespace:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    puf_h5 = inputs / "puf_2024.h5"
    puf_h5.write_bytes(b"processed puf")
    puf_csv = inputs / "puf_2015.csv"
    puf_csv.write_bytes(b"RECID,E00100\n1,2\n")
    return SimpleNamespace(puf_h5=puf_h5, puf_csv=puf_csv)


def _parent_argv(tmp_path: Path, inputs: SimpleNamespace) -> list[str]:
    return [
        "--base-h5",
        str(tmp_path / "base.h5"),
        "--puf-h5",
        str(inputs.puf_h5),
        "--puf-source-year-csv",
        str(inputs.puf_csv),
        "--out",
        str(tmp_path / "out"),
        "--without-block-ladder",
        "--stage",
        "all",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
    ]


def _step_argv(command: list[str]) -> list[str]:
    """The arguments of a step command: what follows the interpreter and tool."""

    return command[2:]


def _step_name(command: list[str]) -> str:
    return command[command.index("--stage") + 1]


def _locked_run_config(builder: ModuleType, args) -> dict[str, object]:
    return builder.StageRuntime(
        args.checkpoint_dir, builder._pipeline_for(args)
    ).context.run_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_each_step_process_uses_the_parent_s_digests_instead_of_hashing(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    parent = builder._parse_args(_parent_argv(tmp_path, inputs))

    builder._run_staged_all(parent)

    # The parent hashed each input and the builder's sources once.
    assert sorted(hashing.files) == sorted(
        [inputs.puf_h5.resolve(), inputs.puf_csv.resolve()]
    )
    assert hashing.code_identity_calls == 1
    locked = _locked_run_config(builder, parent)
    assert locked["puf_h5_sha256"] == _sha256(inputs.puf_h5)
    assert locked["puf_source_year_csv_sha256"] == _sha256(inputs.puf_csv)
    assert locked["builder_code_identity"]["source_sha256"] == "c" * 64
    assert [
        _step_name(command) for command, _environment in steps.started
    ] == _first_steps_of_step_processes(builder, builder.PIPELINE_STEPS)
    for command, environment in steps.started:
        hashing.clear()
        with _as_step_process(monkeypatch, environment):
            step = builder._parse_args(_step_argv(command))
            config = builder._stage_run_config(step)
            # The step's own resume check accepts the parent's digests.
            builder._open_stage_runtime(step)
        # Every value of the run config is what the parent locked, and no
        # input file was hashed again. The builder's sources were, once for
        # each of the two run configs.
        assert config == locked, step.stage
        assert hashing.files == [], step.stage
        assert hashing.code_identity_calls == 2, step.stage


def test_a_step_process_started_without_digests_hashes_its_inputs(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A named --stage run by hand.
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    argv = _parent_argv(tmp_path, inputs)
    argv[argv.index("all")] = "pre_clone_enrichment"
    step = builder._parse_args(argv)

    with _as_step_process(monkeypatch, {}):
        config = builder._stage_run_config(step)

    assert sorted(hashing.files) == sorted(
        [inputs.puf_h5.resolve(), inputs.puf_csv.resolve()]
    )
    assert hashing.code_identity_calls == 1
    assert config["puf_h5_sha256"] == _sha256(inputs.puf_h5)
    assert config["puf_source_year_csv_sha256"] == _sha256(inputs.puf_csv)


def test_digests_set_by_a_process_other_than_the_parent_are_not_used(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every process a step starts inherits the variable, and it could be
    # left exported in a shell; only the parent's own step processes use it.
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    builder._run_staged_all(builder._parse_args(_parent_argv(tmp_path, inputs)))
    command, environment = steps.started[1]
    hashing.clear()

    with monkeypatch.context() as other_parent:
        other_parent.setenv(_DIGESTS_VARIABLE, environment[_DIGESTS_VARIABLE])
        other_parent.setattr(os, "getppid", lambda: os.getpid() + 1)
        builder._stage_run_config(builder._parse_args(_step_argv(command)))

    assert sorted(hashing.files) == sorted(
        [inputs.puf_h5.resolve(), inputs.puf_csv.resolve()]
    )


def test_a_step_process_hashes_a_file_the_parent_gave_no_digest_for(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    # Same file name, another directory: only the full path matches.
    other_puf = tmp_path / "elsewhere" / "puf_2024.h5"
    other_puf.parent.mkdir()
    other_puf.write_bytes(b"another processed puf")
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    builder._run_staged_all(builder._parse_args(_parent_argv(tmp_path, inputs)))
    command, environment = steps.started[1]
    step_argv = _step_argv(command)
    step_argv[step_argv.index(str(inputs.puf_h5))] = str(other_puf)
    hashing.clear()

    with _as_step_process(monkeypatch, environment):
        config = builder._stage_run_config(builder._parse_args(step_argv))

    # The PUF H5's digest belongs to another path, so the step hashed the
    # file it was given; the CSV's digest was the parent's.
    assert hashing.files == [other_puf.resolve()]
    assert config["puf_h5_sha256"] == _sha256(other_puf)
    assert config["puf_source_year_csv_sha256"] == _sha256(inputs.puf_csv)


@pytest.mark.parametrize("changed", ("puf_h5", "puf_csv", "builder_source"))
def test_a_resume_after_a_changed_input_or_builder_source_is_refused(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    argv = _parent_argv(tmp_path, inputs)
    builder._run_staged_all(builder._parse_args(argv))
    first_run_steps = list(steps.started)
    steps.started.clear()

    if changed == "puf_h5":
        inputs.puf_h5.write_bytes(b"changed processed puf")
    elif changed == "puf_csv":
        inputs.puf_csv.write_bytes(b"RECID,E00100\n1,3\n")
    else:
        hashing.source_sha256 = "d" * 64

    # The parent recomputes its digests and refuses before any step starts.
    with pytest.raises(ValueError, match="run_config differs"):
        builder._run_staged_all(builder._parse_args(argv))
    assert steps.started == []
    # So does a step started by hand, without the parent's digests.
    command, _environment = first_run_steps[1]
    with _as_step_process(monkeypatch, {}):
        with pytest.raises(ValueError, match="run_config differs"):
            builder._open_stage_runtime(builder._parse_args(_step_argv(command)))


def test_a_builder_source_change_during_the_run_stops_the_next_step(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A step process hashes the builder's sources itself, so a git pull or
    # an edit in the checkout after the parent started is refused by the
    # next step rather than run under the code identity locked at the start.
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    builder._run_staged_all(builder._parse_args(_parent_argv(tmp_path, inputs)))
    command, environment = steps.started[1]
    hashing.source_sha256 = "d" * 64

    with _as_step_process(monkeypatch, environment):
        step = builder._parse_args(_step_argv(command))
        config = builder._stage_run_config(step)
        with pytest.raises(ValueError, match="run_config differs"):
            builder._open_stage_runtime(step)

    assert config["builder_code_identity"]["source_sha256"] == "d" * 64


def test_the_parent_hashes_its_inputs_whatever_its_own_environment_holds(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Digests left in the parent's environment, from an earlier run, are
    # ignored: the parent computes its own at every start.
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    argv = _parent_argv(tmp_path, inputs)
    builder._run_staged_all(builder._parse_args(argv))
    _command, environment = steps.started[0]
    (tmp_path / "checkpoints" / OUTER_STAGE_CONTEXT_FILENAME).unlink()
    inputs.puf_h5.write_bytes(b"changed processed puf")
    hashing.clear()

    with _as_step_process(monkeypatch, environment):
        builder._run_staged_all(builder._parse_args(argv))

    assert sorted(hashing.files) == sorted(
        [inputs.puf_h5.resolve(), inputs.puf_csv.resolve()]
    )
    assert hashing.code_identity_calls == 1
    assert _locked_run_config(builder, builder._parse_args(argv))[
        "puf_h5_sha256"
    ] == _sha256(inputs.puf_h5)
    # Each step process gets the new digests, not the ones left over.
    puf_path = str(inputs.puf_h5.resolve())
    first_run_processes = len(
        _first_steps_of_step_processes(builder, builder.PIPELINE_STEPS)
    )
    for _command, step_environment in steps.started[first_run_processes:]:
        passed = json.loads(step_environment[_DIGESTS_VARIABLE])
        assert passed["file_sha256"][puf_path] == _sha256(inputs.puf_h5)
        assert passed["parent_pid"] == os.getpid()


#: Run in a real step process: load the builder, refuse any input file
#: hashing, and print the step's run config. The builder's source files are
#: still hashed, by the real builder_code_identity.
_STEP_PROBE = """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("base_tool_step_probe", sys.argv[1])
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def refuse_hashing(*_args, **_kwargs):
    raise AssertionError("the step process hashed an input file")


builder._sha256 = refuse_hashing
config = builder._stage_run_config(builder._parse_args(sys.argv[2:]))
print("RUN_CONFIG " + json.dumps(config))
"""


def test_a_real_step_process_finds_its_parent_and_reuses_the_digests(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No stand-in for the process ID or the code identity: the first step
    # is a real process started from this one with the environment
    # _run_staged_all gives it.
    inputs = _write_inputs(tmp_path)
    start_process = builder.subprocess.run
    step_runs = []

    def probe_the_first_step(command, *, check, env):
        if not step_runs:
            step_runs.append(
                start_process(
                    [command[0], "-c", _STEP_PROBE, str(_TOOL), *_step_argv(command)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", probe_the_first_step)
    parent = builder._parse_args(_parent_argv(tmp_path, inputs))

    builder._run_staged_all(parent)

    [step] = step_runs
    assert step.returncode == 0, step.stderr
    [printed] = [
        line for line in step.stdout.splitlines() if line.startswith("RUN_CONFIG ")
    ]
    assert json.loads(printed.removeprefix("RUN_CONFIG ")) == _locked_run_config(
        builder, parent
    )


def _relative_file_path(payload: dict, puf_path: str) -> object:
    payload["file_sha256"] = {"puf_2024.h5": payload["file_sha256"][puf_path]}
    return payload


def _short_file_digest(payload: dict, puf_path: str) -> object:
    payload["file_sha256"][puf_path] = "a" * 63
    return payload


def _uppercase_file_digest(payload: dict, puf_path: str) -> object:
    payload["file_sha256"][puf_path] = "A" * 64
    return payload


def _unknown_key(payload: dict, _puf_path: str) -> object:
    payload["block_ladder_sha256"] = {}
    return payload


def _later_schema_version(payload: dict, _puf_path: str) -> object:
    payload["schema_version"] = 2
    return payload


def _parent_pid_as_text(payload: dict, _puf_path: str) -> object:
    payload["parent_pid"] = str(payload["parent_pid"])
    return payload


def _a_list(_payload: dict, _puf_path: str) -> object:
    return []


@pytest.mark.parametrize(
    "malform",
    (
        _relative_file_path,
        _short_file_digest,
        _uppercase_file_digest,
        _unknown_key,
        _later_schema_version,
        _parent_pid_as_text,
        _a_list,
        None,
    ),
)
def test_a_malformed_digest_variable_is_refused(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    malform,
) -> None:
    inputs = _write_inputs(tmp_path)
    _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    builder._run_staged_all(builder._parse_args(_parent_argv(tmp_path, inputs)))
    command, environment = steps.started[1]
    if malform is None:
        value = "not json"
    else:
        payload = json.loads(environment[_DIGESTS_VARIABLE])
        value = json.dumps(malform(payload, str(inputs.puf_h5.resolve())))

    with _as_step_process(monkeypatch, {_DIGESTS_VARIABLE: value}):
        with pytest.raises(SystemExit, match=_DIGESTS_VARIABLE):
            builder._stage_run_config(builder._parse_args(_step_argv(command)))


# --------------------------------------------------------------------------
# The pinned ASEC sources and the block_v1 location ladder
# --------------------------------------------------------------------------
#
# With --asec-h5-sha256 pins, every process compares each pinned ASEC file
# with its pin before any stage runs (_verify_asec_source_digests). The parent
# still hashes them; a step process compares each pin with the parent's
# digest for the same resolved path instead, except household_location_
# assignment (block_v1), which reads the files and so hashes them itself.
# source_construction, the other step that reads them, hashes each file
# again once it has read it and compares that digest with the pin. Under block_v1 the run config also
# holds the location ladder's digest, which every step takes from the parent.

#: Income years without a canonical pin, so a fixture file's own digest is
#: an admissible pin.
_FIXTURE_ASEC_YEARS = (2019, 2020)


def _write_pooled_inputs(tmp_path: Path) -> SimpleNamespace:
    inputs = _write_inputs(tmp_path)
    inputs.asec = {}
    for year in _FIXTURE_ASEC_YEARS:
        path = tmp_path / "inputs" / f"census_cps_{year}.h5"
        path.write_bytes(f"asec income year {year}".encode())
        inputs.asec[year] = path
    inputs.ladder = tmp_path / "inputs" / "ladder.npz"
    inputs.ladder.write_bytes(b"block ladder")
    inputs.crosswalk = tmp_path / "inputs" / "crosswalk.csv"
    inputs.crosswalk.write_bytes(b"source_geography_id,target_geography_id,weight\n")
    return inputs


def _pooled_parent_argv(
    tmp_path: Path, inputs: SimpleNamespace, *, block_v1: bool
) -> list[str]:
    argv: list[str] = []
    for year, path in inputs.asec.items():
        argv += ["--asec-h5", f"{year}={path}"]
        argv += ["--asec-h5-sha256", f"{year}={_sha256(path)}"]
    argv += [
        "--puf-h5",
        str(inputs.puf_h5),
        "--puf-source-year-csv",
        str(inputs.puf_csv),
        "--out",
        str(tmp_path / "out"),
        "--stage",
        "all",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
    ]
    if block_v1:
        argv += [
            "--location-rule",
            "block_v1",
            "--block-ladder-artifact",
            str(inputs.ladder),
            "--congressional-district-vintage-crosswalk",
            str(inputs.crosswalk),
        ]
    else:
        argv.append("--without-block-ladder")
    return argv


def _asec_paths(inputs: SimpleNamespace) -> list[Path]:
    return sorted(path.resolve() for path in inputs.asec.values())


def _step_hashes(builder, monkeypatch, hashing, command, environment) -> list[Path]:
    """What one step process hashes in its ASEC pin check and run config."""

    hashing.clear()
    with _as_step_process(monkeypatch, environment):
        step = builder._parse_args(_step_argv(command))
        builder._verify_asec_source_digests(step)
        builder._stage_run_config(step)
    return sorted(hashing.files)


def test_step_processes_take_the_asec_digests_from_the_parent_not_the_files(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_pooled_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    parent = builder._parse_args(_pooled_parent_argv(tmp_path, inputs, block_v1=False))

    builder._dispatch(parent, None)

    # The parent hashed each pinned ASEC file and each PUF file once.
    assert sorted(hashing.files) == sorted(
        [*_asec_paths(inputs), inputs.puf_h5.resolve(), inputs.puf_csv.resolve()]
    )
    assert [
        _step_name(command) for command, _env in steps.started
    ] == _first_steps_of_step_processes(builder, builder.PIPELINE_STEPS)
    for command, environment in steps.started:
        assert (
            _step_hashes(builder, monkeypatch, hashing, command, environment) == []
        ), _step_name(command)


def test_under_block_v1_only_the_location_step_hashes_the_asec_files(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_pooled_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    parent = builder._parse_args(_pooled_parent_argv(tmp_path, inputs, block_v1=True))

    builder._dispatch(parent, None)

    assert sorted(hashing.files) == sorted(
        [
            *_asec_paths(inputs),
            inputs.puf_h5.resolve(),
            inputs.puf_csv.resolve(),
            inputs.ladder.resolve(),
        ]
    )
    locked = _locked_run_config(builder, parent)
    assert locked["block_ladder_artifact_sha256"] == _sha256(inputs.ladder)
    assert [
        _step_name(command) for command, _env in steps.started
    ] == _first_steps_of_step_processes(builder, builder.BLOCK_V1_PIPELINE_STEPS)
    for command, environment in steps.started:
        hashed = _step_hashes(builder, monkeypatch, hashing, command, environment)
        if _step_name(command) == builder.HOUSEHOLD_LOCATION_STAGE_NAME:
            # It reads the ASEC files, so it hashes them; the ladder's and
            # the PUF files' digests are the parent's.
            assert hashed == _asec_paths(inputs)
        else:
            assert hashed == [], _step_name(command)


def test_a_step_refuses_a_parent_asec_digest_that_differs_from_its_pin(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_pooled_inputs(tmp_path)
    _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    builder._dispatch(
        builder._parse_args(_pooled_parent_argv(tmp_path, inputs, block_v1=False)),
        None,
    )
    command, environment = steps.started[1]
    payload = json.loads(environment[_DIGESTS_VARIABLE])
    payload["file_sha256"][str(inputs.asec[2019].resolve())] = "e" * 64

    with _as_step_process(monkeypatch, {_DIGESTS_VARIABLE: json.dumps(payload)}):
        with pytest.raises(
            SystemExit, match=r"ASEC 2019 source .* failed digest verification"
        ):
            builder._verify_asec_source_digests(
                builder._parse_args(_step_argv(command))
            )


@pytest.mark.parametrize("changed", ("asec", "ladder"))
def test_a_resume_after_a_changed_asec_file_or_ladder_is_refused(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: str,
) -> None:
    inputs = _write_pooled_inputs(tmp_path)
    _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    argv = _pooled_parent_argv(tmp_path, inputs, block_v1=True)
    builder._dispatch(builder._parse_args(argv), None)
    steps.started.clear()

    if changed == "asec":
        inputs.asec[2020].write_bytes(b"changed asec income year 2020")
        refusal = (SystemExit, r"ASEC 2020 source .* failed digest verification")
    else:
        inputs.ladder.write_bytes(b"changed block ladder")
        refusal = (ValueError, "run_config differs")

    # The parent hashes both again and refuses before any step starts.
    with pytest.raises(refusal[0], match=refusal[1]):
        builder._dispatch(builder._parse_args(argv), None)
    assert steps.started == []


def test_relative_and_symlinked_paths_find_the_parent_s_digests(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A step process gets the parent's own argument strings
    # (_stage_cli_args), so its lookup must use each path's resolved form.
    inputs = _write_pooled_inputs(tmp_path)
    csv_link = tmp_path / "puf_csv_link.csv"
    csv_link.symlink_to(inputs.puf_csv)
    monkeypatch.chdir(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    argv = _pooled_parent_argv(tmp_path, inputs, block_v1=False)
    argv[argv.index(str(inputs.puf_h5))] = "inputs/puf_2024.h5"
    argv[argv.index(str(inputs.puf_csv))] = csv_link.name
    for year, path in inputs.asec.items():
        argv[argv.index(f"{year}={path}")] = f"{year}=inputs/{path.name}"

    builder._dispatch(builder._parse_args(argv), None)

    assert len(steps.started) == len(
        _first_steps_of_step_processes(builder, builder.PIPELINE_STEPS)
    )
    for command, environment in steps.started:
        assert "inputs/puf_2024.h5" in command
        assert csv_link.name in command
        assert (
            _step_hashes(builder, monkeypatch, hashing, command, environment) == []
        ), _step_name(command)
