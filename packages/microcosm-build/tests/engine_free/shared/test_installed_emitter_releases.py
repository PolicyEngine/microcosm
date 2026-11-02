"""The releases the installed-emitter job installs, and its runner's control flow.

``tools/installed_emitter_releases.py`` names the releases: the minimum builds
accept, then the pinned newest. ``tools/run_installed_emitter_tests.sh`` runs
the tests once per release; here it runs against a stand-in ``uv`` command
that records each call instead of installing anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.telemetry_emitter import MINIMUM_EMITTER_VERSION
from test_support.paths import paths_for
from tools import installed_emitter_releases

_ROOT = paths_for("microcosm-build").repository
_RUNNER = _ROOT / "tools/run_installed_emitter_tests.sh"
_STAND_IN_UV = """\
import json
import os
import subprocess
import sys

arguments = sys.argv[1:]
with open(os.environ["STAND_IN_UV_LOG"], "a", encoding="utf-8") as log:
    record = {
        "arguments": arguments,
        "release": os.environ.get("INSTALLED_EMITTER_VERSION"),
        "command": os.environ.get("MICROCOSM_EMITTER_COMMAND"),
        "tool_directory": os.environ.get("UV_TOOL_DIR"),
        "tool_bin_directory": os.environ.get("UV_TOOL_BIN_DIR"),
    }
    log.write(json.dumps(record) + "\\n")
if arguments[:4] == ["run", "--no-sync", "python", "tools/installed_emitter_releases.py"]:
    print(os.environ["STAND_IN_UV_RELEASES"].replace(" ", "\\n"))
    sys.exit(0)
if arguments[:5] == ["run", "--no-sync", "python", "tools/ci_test_plan.py", "list-job"]:
    print(os.environ["STAND_IN_UV_TEST_FILES"].replace(" ", "\\n"))
    sys.exit(0)
if arguments[:3] == ["run", "--no-sync", "pytest"]:
    failing = os.environ.get("STAND_IN_UV_FAILING_RELEASES", "").split()
    sys.exit(1 if os.environ["INSTALLED_EMITTER_VERSION"] in failing else 0)
if arguments[:2] == ["tool", "install"]:
    sys.exit(0)
sys.exit(2)
"""


def _pin(release: str = "0.11.11") -> str:
    return (
        "# The newest microcosm-emitter release the job tests.\n"
        f"microcosm-emitter=={release}\n"
    )


@pytest.mark.parametrize(
    ("minimum", "pinned", "expected"),
    [
        ("0.6.0", "0.11.11", ["0.6.0", "0.11.11"]),
        ("0.11.9", "0.11.11", ["0.11.9", "0.11.11"]),
        ("0.11.11", "0.11.11", ["0.11.11"]),
        ("0.9.0", "0.10.0", ["0.9.0", "0.10.0"]),
    ],
)
def test_the_minimum_and_the_pinned_release_are_installed_in_order(
    minimum: str, pinned: str, expected: list[str]
) -> None:
    assert installed_emitter_releases.releases_to_install(minimum, pinned) == expected


@pytest.mark.parametrize(
    ("minimum", "pinned"), [("0.11.9", "0.11.8"), ("1.0.0", "0.99.0")]
)
def test_a_pin_older_than_the_minimum_is_refused(minimum: str, pinned: str) -> None:
    with pytest.raises(SystemExit) as refusal:
        installed_emitter_releases.releases_to_install(minimum, pinned)

    message = str(refusal.value)
    assert f"microcosm-emitter {pinned}" in message
    assert f"MINIMUM_EMITTER_VERSION {minimum}" in message


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# only a comment\n",
        "microcosm-emitter>=0.11.11\n",
        "microcosm-emitter==0.11\n",
        "microcosm-emitter==0.11.11rc1\n",
        "microcosm-emitter==0.11.10\nmicrocosm-emitter==0.11.11\n",
        "microcosm-emitter==0.11.11\nhttpx==0.28.1\n",
    ],
)
def test_a_malformed_pin_is_refused(text: str) -> None:
    with pytest.raises(SystemExit) as refusal:
        installed_emitter_releases.pinned_release(text)

    assert "requirements.txt" in str(refusal.value)


def test_the_repository_pin_names_one_release_no_older_than_the_minimum() -> None:
    pinned = installed_emitter_releases.pinned_release(
        installed_emitter_releases.PIN_FILE.read_text(encoding="utf-8")
    )

    assert (
        installed_emitter_releases.releases_to_install(MINIMUM_EMITTER_VERSION, pinned)[
            -1
        ]
        == pinned
    )


@pytest.mark.parametrize(
    ("pinned", "published", "warned"),
    [
        ("0.11.10", "0.11.11", True),
        ("0.11.11", "0.11.11", False),
        ("0.11.11", None, False),
    ],
)
def test_a_pin_behind_the_package_index_is_reported(
    pinned: str, published: str | None, warned: bool
) -> None:
    warning = installed_emitter_releases.lag_warning(pinned, published)

    if warned:
        assert warning is not None
        assert warning.startswith("::warning::")
        assert f"pins microcosm-emitter {pinned}" in warning
        assert f"newest release is {published}" in warning
    else:
        assert warning is None


def _answer(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["uv"], returncode, stdout=stdout, stderr="")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (_answer(0, "microcosm-emitter==0.11.11\n"), "0.11.11"),
        (_answer(2, ""), None),
        (_answer(0, "something unexpected\n"), None),
        (OSError("uv not found"), None),
        (subprocess.TimeoutExpired(["uv"], 60), None),
    ],
)
def test_the_package_index_is_asked_quietly(
    answer: subprocess.CompletedProcess[str] | Exception, expected: str | None
) -> None:
    calls: list[list[str]] = []

    def run(
        arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert installed_emitter_releases.published_release(run=run) == expected
    assert calls[0][:3] == ["uv", "pip", "compile"]


def test_the_command_prints_the_releases_and_warns_about_a_lagging_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pin_file = tmp_path / "requirements.txt"
    pin_file.write_text(_pin("0.11.10"), encoding="utf-8")
    monkeypatch.setattr(
        installed_emitter_releases, "published_release", lambda: "0.11.11"
    )

    status = installed_emitter_releases.main(
        ["--minimum", "0.6.0", "--pin-file", str(pin_file)]
    )

    output = capsys.readouterr()
    assert status == 0
    assert output.out.splitlines() == ["0.6.0", "0.11.10"]
    assert "::warning::" in output.err
    assert "newest release is 0.11.11" in output.err


def test_the_command_can_skip_the_package_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pin_file = tmp_path / "requirements.txt"
    pin_file.write_text(_pin("0.11.11"), encoding="utf-8")

    def unexpected() -> str:
        raise AssertionError("the package index was asked")

    monkeypatch.setattr(installed_emitter_releases, "published_release", unexpected)

    status = installed_emitter_releases.main(
        ["--minimum", "0.11.11", "--pin-file", str(pin_file), "--no-index-check"]
    )

    output = capsys.readouterr()
    assert status == 0
    assert output.out.splitlines() == ["0.11.11"]
    assert output.err == ""


def _run_runner(
    tmp_path: Path,
    *,
    releases: str,
    test_files: str,
    failing: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    """Run the runner script with a stand-in ``uv`` first on ``PATH``."""
    commands = tmp_path / "commands"
    commands.mkdir()
    stand_in_uv = commands / "uv"
    stand_in_uv.write_text(f"#!{sys.executable}\n{_STAND_IN_UV}", encoding="utf-8")
    stand_in_uv.chmod(0o755)
    log = tmp_path / "uv-calls.jsonl"
    tools = tmp_path / "tools"
    environment = {
        **os.environ,
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "STAND_IN_UV_LOG": str(log),
        "STAND_IN_UV_RELEASES": releases,
        "STAND_IN_UV_TEST_FILES": test_files,
        "STAND_IN_UV_FAILING_RELEASES": failing,
    }
    result = subprocess.run(
        ["bash", str(_RUNNER), str(tools)],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    calls = (
        [json.loads(entry) for entry in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return result, calls


def _test_runs(calls: list[dict]) -> list[str]:
    return [
        call["release"]
        for call in calls
        if call["arguments"][:3] == ["run", "--no-sync", "pytest"]
    ]


def test_the_runner_tests_each_release_with_its_own_tool_directory(
    tmp_path: Path,
) -> None:
    result, calls = _run_runner(
        tmp_path, releases="0.6.0 0.11.11", test_files="tests/test_one.py"
    )

    assert result.returncode == 0, result.stderr
    installs = [call for call in calls if call["arguments"][:2] == ["tool", "install"]]
    assert [call["arguments"][2] for call in installs] == [
        "microcosm-emitter==0.6.0",
        "microcosm-emitter==0.11.11",
    ]
    for call, release in zip(installs, ("0.6.0", "0.11.11"), strict=True):
        directory = tmp_path / "tools" / f"microcosm-emitter-{release}"
        assert call["tool_directory"] == str(directory / "tools")
        assert call["tool_bin_directory"] == str(directory / "bin")
        assert "--constraints" in call["arguments"]
    runs = [
        call
        for call in calls
        if call["arguments"][:3] == ["run", "--no-sync", "pytest"]
    ]
    assert [call["release"] for call in runs] == ["0.6.0", "0.11.11"]
    for call in runs:
        assert call["command"] == str(
            tmp_path
            / "tools"
            / f"microcosm-emitter-{call['release']}"
            / "bin"
            / "microcosm-emitter"
        )
        assert call["arguments"][3] == "tests/test_one.py"
        assert "--run-integration" in call["arguments"]


def test_the_runner_tests_one_release_once(tmp_path: Path) -> None:
    result, calls = _run_runner(
        tmp_path, releases="0.11.11", test_files="tests/test_one.py"
    )

    assert result.returncode == 0, result.stderr
    assert _test_runs(calls) == ["0.11.11"]


def test_a_failure_at_the_first_release_still_tests_the_second(tmp_path: Path) -> None:
    result, calls = _run_runner(
        tmp_path,
        releases="0.6.0 0.11.11",
        test_files="tests/test_one.py",
        failing="0.6.0",
    )

    errors = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("::error::")
    ]
    assert result.returncode == 1
    assert _test_runs(calls) == ["0.6.0", "0.11.11"]
    assert len(errors) == 1
    assert "microcosm-emitter 0.6.0" in errors[0]
    assert "0.11.11" not in errors[0]


def test_the_runner_refuses_an_empty_test_list(tmp_path: Path) -> None:
    result, calls = _run_runner(tmp_path, releases="0.6.0 0.11.11", test_files="")

    assert result.returncode != 0
    assert _test_runs(calls) == []
    assert not [call for call in calls if call["arguments"][:2] == ["tool", "install"]]
    assert "no test files" in result.stdout + result.stderr


def test_the_runner_refuses_an_empty_release_list(tmp_path: Path) -> None:
    result, calls = _run_runner(tmp_path, releases="", test_files="tests/test_one.py")

    assert result.returncode != 0
    assert _test_runs(calls) == []
    assert "no microcosm-emitter release" in result.stdout + result.stderr
