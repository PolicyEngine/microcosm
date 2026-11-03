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


class _IndexAnswers:
    """A package-index lookup that answers from ``answers`` and records each
    requirement it was asked about."""

    def __init__(self, answers: dict[str, str | None]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, requirement: str) -> str | None:
        self.asked.append(requirement)
        return self.answers[requirement]


@pytest.mark.parametrize(
    ("published", "level", "phrases"),
    [
        ("0.11.12", "::warning::", ["pins microcosm-emitter 0.11.11", "is 0.11.12"]),
        ("0.12.0", "::warning::", ["pins microcosm-emitter 0.11.11", "is 0.12.0"]),
        ("0.11.10", "::notice::", ["is 0.11.10, older than the pin", "yanked"]),
        (None, "::notice::", ["could not ask", "microcosm-emitter"]),
    ],
)
def test_the_emitter_pin_is_compared_with_the_package_index(
    published: str | None, level: str, phrases: list[str]
) -> None:
    newest = _IndexAnswers(
        {"microcosm-emitter": published, "huggingface-hub<2": "1.33.0"}
    )

    messages = installed_emitter_releases.index_messages("0.11.11", "1.33.0", newest)

    assert len(messages) == 1
    assert messages[0].startswith(level)
    for phrase in phrases:
        assert phrase in messages[0]
    if level == "::notice::":
        assert "raise" not in messages[0]
    assert newest.asked == ["microcosm-emitter", "huggingface-hub<2"]


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        ("1.34.0", ["::warning::", "huggingface-hub 1.33.0", "1.x release is 1.34.0"]),
        ("1.33.0", None),
        ("1.32.0", None),
        (None, ["::notice::", "could not ask", "huggingface-hub<2"]),
    ],
)
def test_the_hub_constraint_is_compared_with_the_package_index(
    published: str | None, expected: list[str] | None
) -> None:
    newest = _IndexAnswers(
        {"microcosm-emitter": "0.11.11", "huggingface-hub<2": published}
    )

    messages = installed_emitter_releases.index_messages("0.11.11", "1.33.0", newest)

    if expected is None:
        assert messages == []
    else:
        assert len(messages) == 1
        assert messages[0].startswith(expected[0])
        for phrase in expected[1:]:
            assert phrase in messages[0]


def test_an_up_to_date_pin_and_constraint_print_nothing() -> None:
    newest = _IndexAnswers(
        {"microcosm-emitter": "0.11.11", "huggingface-hub<2": "1.33.0"}
    )

    assert installed_emitter_releases.index_messages("0.11.11", "1.33.0", newest) == []


def _answer(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["uv"], returncode, stdout=stdout, stderr="")


@pytest.mark.parametrize(
    ("requirement", "answer", "expected"),
    [
        ("microcosm-emitter", _answer(0, "microcosm-emitter==0.11.11\n"), "0.11.11"),
        ("huggingface-hub<2", _answer(0, "huggingface-hub==1.34.0\n"), "1.34.0"),
        ("huggingface-hub<2", _answer(0, "huggingface-hub==1.34.0.post1\n"), None),
        ("microcosm-emitter", _answer(2, ""), None),
        ("microcosm-emitter", _answer(0, "something unexpected\n"), None),
        ("microcosm-emitter", OSError("uv not found"), None),
        ("microcosm-emitter", subprocess.TimeoutExpired(["uv"], 60), None),
    ],
)
def test_the_package_index_is_asked_without_failing(
    requirement: str,
    answer: subprocess.CompletedProcess[str] | Exception,
    expected: str | None,
) -> None:
    calls: list[tuple[list[str], object]] = []

    def run(
        arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, options.get("input")))
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert installed_emitter_releases.newest_release(requirement, run=run) == expected
    arguments, sent = calls[0]
    assert arguments[:3] == ["uv", "pip", "compile"]
    assert sent == f"{requirement}\n"


@pytest.mark.parametrize(
    "text",
    ["", "httpx==0.28.1\n", "huggingface-hub>=1.33.0\n", "huggingface-hub==1.33\n"],
)
def test_a_malformed_hub_constraint_is_refused(text: str) -> None:
    with pytest.raises(SystemExit) as refusal:
        installed_emitter_releases.constrained_release(text, "huggingface-hub")

    assert "constraints.txt" in str(refusal.value)


def test_the_repository_constraint_names_one_hub_release() -> None:
    constraint = installed_emitter_releases.constrained_release(
        installed_emitter_releases.CONSTRAINTS_FILE.read_text(encoding="utf-8"),
        "huggingface-hub",
    )

    assert installed_emitter_releases.release_number(constraint)


def _command_files(tmp_path: Path, *, pinned: str) -> list[str]:
    pin_file = tmp_path / "requirements.txt"
    pin_file.write_text(_pin(pinned), encoding="utf-8")
    constraints_file = tmp_path / "constraints.txt"
    constraints_file.write_text("huggingface-hub==1.33.0\n", encoding="utf-8")
    return ["--pin-file", str(pin_file), "--constraints-file", str(constraints_file)]


def test_the_command_prints_the_releases_and_warns_about_lagging_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    newest = _IndexAnswers(
        {"microcosm-emitter": "0.11.11", "huggingface-hub<2": "1.34.0"}
    )
    monkeypatch.setattr(installed_emitter_releases, "newest_release", newest)

    status = installed_emitter_releases.main(
        ["--minimum", "0.6.0", *_command_files(tmp_path, pinned="0.11.10")]
    )

    output = capsys.readouterr()
    warnings = output.err.splitlines()
    assert status == 0
    assert output.out.splitlines() == ["0.6.0", "0.11.10"]
    assert len(warnings) == 2
    assert all(line.startswith("::warning::") for line in warnings)
    assert "is 0.11.11" in warnings[0]
    assert "1.x release is 1.34.0" in warnings[1]


def test_the_command_notes_a_comparison_it_could_not_make(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    newest = _IndexAnswers({"microcosm-emitter": None, "huggingface-hub<2": None})
    monkeypatch.setattr(installed_emitter_releases, "newest_release", newest)

    status = installed_emitter_releases.main(
        ["--minimum", "0.6.0", *_command_files(tmp_path, pinned="0.11.11")]
    )

    output = capsys.readouterr()
    notes = output.err.splitlines()
    assert status == 0
    assert output.out.splitlines() == ["0.6.0", "0.11.11"]
    assert len(notes) == 2
    assert all(line.startswith("::notice::") for line in notes)


def test_the_command_can_skip_the_package_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected(requirement: str) -> str:
        raise AssertionError("the package index was asked")

    monkeypatch.setattr(installed_emitter_releases, "newest_release", unexpected)

    status = installed_emitter_releases.main(
        [
            "--minimum",
            "0.11.11",
            *_command_files(tmp_path, pinned="0.11.11"),
            "--no-index-check",
        ]
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
