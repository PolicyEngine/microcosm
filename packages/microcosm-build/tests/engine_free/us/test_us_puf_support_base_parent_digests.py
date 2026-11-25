"""A checkpointed base run hashes its inputs and the builder's code once.

``tools/build_us_puf_support_base.py --stage all`` starts one step process
per pipeline step, and each step process opens the checkpoint directory with
its own run config, which holds the SHA-256 of the PUF H5, the PUF CSV and
the builder's source files. The parent computes that run config once,
compares it with the one locked in the checkpoint directory before any step
process starts, and passes its digests to each step process in the
environment variable ``MICROCOSM_US_BASE_PARENT_DIGESTS``: each file's digest
under its resolved path, and the source digest with the repository root and
tool file it covers. A step process uses a digest only for the same path,
and only when the process that set the variable is its parent; it hashes
everything else itself.

These tests replace ``subprocess.run`` with a stand-in that records each step
command and environment, then compute each step's run config in this
process, as that step process would, while recording what gets hashed.
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
_DIGESTS_VARIABLE = "MICROCOSM_US_BASE_PARENT_DIGESTS"


def _load_builder() -> ModuleType:
    path = _TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder() -> ModuleType:
    return _load_builder()


class _Hashing:
    """Records what the builder hashes; the files are still hashed for real.

    The builder's source digest is a stand-in, ``source_sha256``, so a test
    can change the builder's code by changing it; ``code_identity_calls``
    holds the digest each call was given, or ``None`` when it hashed.
    """

    def __init__(self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch):
        self.files: list[Path] = []
        self.code_identity_calls: list[str | None] = []
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
            source_sha256: str | None = None,
        ) -> dict[str, object]:
            self.code_identity_calls.append(source_sha256)
            return {
                "dependency_versions": {name: "1.0" for name in distributions},
                "python": "stand-in",
                "source_sha256": (
                    self.source_sha256 if source_sha256 is None else source_sha256
                ),
            }

        monkeypatch.setattr(builder, "_sha256", recording_sha256)
        monkeypatch.setattr(
            code_identity, "builder_code_identity", stand_in_code_identity
        )

    def clear(self) -> None:
        self.files.clear()
        self.code_identity_calls.clear()


class _StepProcesses:
    """Stands in for ``subprocess.run``: records each step it would start."""

    def __init__(self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch):
        self.started: list[tuple[list[str], dict[str, str]]] = []

        def record_step(command, *, check, env):
            assert not check
            self.started.append((list(command), dict(env)))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(builder.subprocess, "run", record_step)


@contextmanager
def _as_step_process(
    monkeypatch: pytest.MonkeyPatch, environment: Mapping[str, str]
) -> Iterator[None]:
    """Run the body as a step process this process started with ``environment``."""

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


def _parent_argv(
    tmp_path: Path, inputs: SimpleNamespace, *, puf_h5: Path | None = None
) -> list[str]:
    return [
        "--base-h5",
        str(tmp_path / "base.h5"),
        "--puf-h5",
        str(inputs.puf_h5 if puf_h5 is None else puf_h5),
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
    assert hashing.code_identity_calls == [None]
    locked = _locked_run_config(builder, parent)
    assert locked["puf_h5_sha256"] == _sha256(inputs.puf_h5)
    assert locked["puf_source_year_csv_sha256"] == _sha256(inputs.puf_csv)
    assert locked["builder_code_identity"]["source_sha256"] == "c" * 64
    assert [_step_name(command) for command, _environment in steps.started] == list(
        builder.PIPELINE_STEPS
    )
    for command, environment in steps.started:
        hashing.clear()
        with _as_step_process(monkeypatch, environment):
            step = builder._parse_args(_step_argv(command))
            config = builder._stage_run_config(step)
            # The step's own resume check accepts the parent's digests.
            builder._open_stage_runtime(step)
        # Every value of the run config is what the parent locked, and
        # nothing was hashed again.
        assert config == locked, step.stage
        assert hashing.files == [], step.stage
        assert hashing.code_identity_calls == ["c" * 64, "c" * 64], step.stage


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
    assert hashing.code_identity_calls == [None]
    assert config["puf_h5_sha256"] == _sha256(inputs.puf_h5)
    assert config["puf_source_year_csv_sha256"] == _sha256(inputs.puf_csv)


def test_digests_set_by_a_process_other_than_the_parent_are_not_used(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The variable reaches the QRF worker processes a step starts, and it
    # could be left exported in a shell; only the parent's own step
    # processes use it.
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
    assert hashing.code_identity_calls == [None]


def test_a_step_process_hashes_a_file_the_parent_gave_no_digest_for(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    other_puf = tmp_path / "other_puf.h5"
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


def test_a_source_digest_for_another_builder_file_is_not_used(
    builder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    hashing = _Hashing(builder, monkeypatch)
    steps = _StepProcesses(builder, monkeypatch)
    builder._run_staged_all(builder._parse_args(_parent_argv(tmp_path, inputs)))
    command, environment = steps.started[1]
    hashing.clear()

    for key, elsewhere in (
        ("tool_path", str(tmp_path / "elsewhere" / "build.py")),
        ("repo_root", str(tmp_path / "elsewhere")),
    ):
        payload = json.loads(environment[_DIGESTS_VARIABLE])
        payload["builder_source_sha256"][key] = elsewhere
        with _as_step_process(monkeypatch, {_DIGESTS_VARIABLE: json.dumps(payload)}):
            builder._stage_run_config(builder._parse_args(_step_argv(command)))

    assert hashing.code_identity_calls == [None, None]
    assert hashing.files == []


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
    assert hashing.code_identity_calls == [None]
    assert _locked_run_config(builder, builder._parse_args(argv))[
        "puf_h5_sha256"
    ] == _sha256(inputs.puf_h5)


def _relative_file_path(payload: dict, puf_path: str) -> object:
    payload["file_sha256"] = {"puf_2024.h5": payload["file_sha256"][puf_path]}
    return payload


def _short_file_digest(payload: dict, puf_path: str) -> object:
    payload["file_sha256"][puf_path] = "a" * 63
    return payload


def _uppercase_file_digest(payload: dict, puf_path: str) -> object:
    payload["file_sha256"][puf_path] = "A" * 64
    return payload


def _short_source_digest(payload: dict, _puf_path: str) -> object:
    payload["builder_source_sha256"]["sha256"] = "a" * 63
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
        _short_source_digest,
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
