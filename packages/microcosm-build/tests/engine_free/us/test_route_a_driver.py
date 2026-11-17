"""Exercise Route A orchestration without loading a country engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_support.microcosm_build.route_a_driver import (
    ROUTE_A_TOOLS,
    SECRET_ALIASES,
    run_driver_settings,
    write_executable,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _function(name: str) -> str:
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{name}\(\)\s*\{{.*?^\}}", source)
    assert match is not None, f"driver has no {name} function"
    return match.group(0)


def _release_result(
    staging: str | None, *, tail: str = "", extras: str = ""
) -> subprocess.CompletedProcess[bytes]:
    settings = {
        "W": "/fixture/worktree",
        "PY": sys.executable,
        "BASE_H5": "/fixture/base with spaces.h5",
        "FEED": "/fixture/feed.jsonl",
        "FEED_SHA": "a" * 64,
        "TAIL": tail,
        "SSI": "/fixture/ssi.json",
        "SSI_SHA": "b" * 64,
        "SCF": "/fixture/scf.dta",
        "REL_OUT": "/fixture/release-out",
        "RID": "fixture-release",
        "REL_CKPT": "/fixture/checkpoints",
        "RELEASE_EXTRA_ARGS": extras,
        "TOKEN_WRAPPER": str(ROUTE_A_TOOLS / "with_hf_token.sh"),
        "AGENT_SECRET": "/fixture/agent-secret",
    }
    script = "set -eu\nfail() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
    script += "\n".join(
        f"{key}={shlex.quote(value)}" for key, value in settings.items()
    )
    script += "\nunset ROUTE_A_STAGING\n"
    if staging is not None:
        script += f"ROUTE_A_STAGING={shlex.quote(staging)}\n"
    script += _function("release_command")
    script += '\nrelease_command\nprintf "%s\\0" "${RELEASE_ARGV[@]}"\n'
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, timeout=20
    )


def _release_argv(staging: str | None, **kwargs: str) -> list[str]:
    result = _release_result(staging, **kwargs)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().rstrip("\0").split("\0")


def _flag_checker():
    spec = importlib.util.spec_from_file_location(
        "route_a_check_flags", ROUTE_A_TOOLS / "check_flags.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shell_sources() -> list[Path]:
    return sorted(ROUTE_A_TOOLS.glob("*.sh")) + [ROUTE_A_TOOLS / "route_a.env.example"]


def _example_extra_args() -> str:
    source = shlex.quote(str(ROUTE_A_TOOLS / "route_a.env.example"))
    result = subprocess.run(
        [
            "bash",
            "-c",
            "PE=/fixture; EXPORT_MASS_REF_H5=/fixture/mass.h5; unset RELEASE_EXTRA_ARGS; "
            f". {source}; "
            'printf "%s" "$RELEASE_EXTRA_ARGS"',
        ],
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode()


@pytest.mark.parametrize("override", [False, True])
def test_feed_and_crosswalk_pins_keep_defaults_or_settings_overrides(
    tmp_path: Path, override: bool
) -> None:
    historical_feed = "b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801"
    historical_crosswalk = (
        "c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec"
    )
    overrides = (
        {
            "FEED_SHA": "a" * 64,
            "FEED_BYTES": "123",
            "CROSSWALK_SHA": "b" * 64,
            "BASE_LEDGER_FACTS": str(tmp_path / "historical-base-feed.jsonl"),
            "BASE_LEDGER_FACTS_SHA": historical_feed,
        }
        if override
        else {}
    )
    result = run_driver_settings(
        tmp_path,
        overrides,
        '\ninput_rows\nprintf "crosswalk|%s\\n" "$CROSSWALK_SHA"\n',
    )
    assert result.returncode == 0, result.stderr.decode()
    rows = {
        row[0]: row[1:]
        for row in map(lambda s: s.split("|"), result.stdout.decode().splitlines())
    }
    release = rows["release_ledger_facts"]
    base = rows["base_ledger_facts"]
    assert release[1:3] == (
        ["a" * 64, "123"] if override else [historical_feed, "164603204"]
    )
    assert rows["crosswalk"] == ["b" * 64 if override else historical_crosswalk]
    assert base[1] == historical_feed
    assert base[0] == (overrides["BASE_LEDGER_FACTS"] if override else release[0])


@pytest.mark.parametrize("mismatch", [None, "sha", "size"])
def test_overridden_feed_pin_still_verifies_digest_and_size(
    tmp_path: Path, mismatch: str | None
) -> None:
    feed = tmp_path / "new-feed.jsonl"
    payload = b'{"fixture": "new release feed"}\n'
    feed.write_bytes(payload)
    result = run_driver_settings(
        tmp_path,
        {
            "FEED": str(feed),
            "FEED_SHA": "0" * 64
            if mismatch == "sha"
            else hashlib.sha256(payload).hexdigest(),
            "FEED_BYTES": str(len(payload) + (mismatch == "size")),
        },
        """
# Exercise the driver's registered release row with real tiny bytes; no
# production input or git repository is read. Adapt its BSD stat call for CI.
registered_rows=$(input_rows)
input_rows() { printf '%s\\n' "$registered_rows" | awk -F '|' '$1 == "release_ledger_facts"'; }
stat() { "$PYTHON" -c 'import os,sys; print(os.stat(sys.argv[1]).st_size)' "$3"; }
verify_inputs
printf '%s\\n' "${#INPUT_PROBLEMS[@]}"
""",
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == ("0" if mismatch is None else "1")
    report = json.loads((tmp_path / "run_root" / "inputs.resolved.json").read_text())
    assert len(report["inputs"]) == 1
    assert report["inputs"][0]["status"] == (
        "ok" if mismatch is None else f"{mismatch}_mismatch"
    )


def _tiny_main(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    """A throwaway local repository standing in for ``$MAIN`` at one commit."""

    main = tmp_path / "main-repo"
    for relative, text in files.items():
        path = main / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git = [
        "git",
        "-C",
        str(main),
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
    ]
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["commit", "-q", "-m", "fixture"],
        # check_commit looks for the commit on origin's branches.
        ["update-ref", "refs/remotes/origin/main", "HEAD"],
    ):
        subprocess.run([*git, *args], check=True, capture_output=True, timeout=20)
    sha = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    return main, sha


_FEED_PIN_REL = "packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json"
_CROSSWALK_REL = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/data/"
    "congressional_district_vintage_crosswalk.csv"
)
_EMITTER_FLOOR_REL = (
    "packages/microcosm-build/src/microcosm/build/minimum_emitter_version.json"
)


def _report_lines(commands: str) -> str:
    return (
        commands
        + """
for x in ${COMMIT_NOTES[@]+"${COMMIT_NOTES[@]}"}; do printf 'NOTE %s\\n' "$x"; done
for x in ${COMMIT_BLOCKERS[@]+"${COMMIT_BLOCKERS[@]}"}; do printf 'BLOCKER %s\\n' "$x"; done
"""
    )


def test_stale_feed_and_crosswalk_blockers_name_the_commits_pins(
    tmp_path: Path,
) -> None:
    """The driver keeps the published run's pins by default; a commit that
    moved them gets blockers naming the commit's own values, so a rerun's
    settings can copy them."""

    crosswalk = "state,district\nCT,1\n"
    main, sha = _tiny_main(
        tmp_path,
        {
            _FEED_PIN_REL: json.dumps({"facts_sha256": "c" * 64}, indent=2) + "\n",
            _CROSSWALK_REL: crosswalk,
        },
    )
    result = _input_settings_result(
        tmp_path,
        {"MAIN": str(main), "FEED_SHA": "a" * 64, "CROSSWALK_SHA": "b" * 64},
        _report_lines(f"\ncheck_commit {sha} {sha}\n"),
    )
    assert result.returncode == 0, result.stderr.decode()
    lines = result.stdout.decode().splitlines()
    digest = hashlib.sha256(crosswalk.encode()).hexdigest()
    assert any(
        line.startswith("BLOCKER ")
        and f"does not pin facts_sha256 {'a' * 64}" in line
        and f"it pins {'c' * 64}" in line
        for line in lines
    ), lines
    assert any(
        line.startswith("BLOCKER ")
        and f"crosswalk at {sha} is {digest}, not {'b' * 64}" in line
        for line in lines
    ), lines
    # check_commit ends with the emitter check (no floor file here).
    assert f"NOTE no {_EMITTER_FLOOR_REL} at {sha}; emitter not checked" in lines


def test_a_missing_crosswalk_blocks_without_naming_a_digest(tmp_path: Path) -> None:
    main, sha = _tiny_main(
        tmp_path,
        {_FEED_PIN_REL: json.dumps({"facts_sha256": "c" * 64}) + "\n"},
    )
    result = _input_settings_result(
        tmp_path,
        {"MAIN": str(main), "FEED_SHA": "c" * 64},
        _report_lines(f"\ncheck_commit {sha} {sha}\n"),
    )
    assert result.returncode == 0, result.stderr.decode()
    lines = result.stdout.decode().splitlines()
    assert (
        f"BLOCKER no packaged CD vintage crosswalk at {sha} ({_CROSSWALK_REL})" in lines
    )
    assert not [line for line in lines if "set CROSSWALK_SHA" in line]


def test_matching_feed_and_crosswalk_pins_are_notes_not_blockers(
    tmp_path: Path,
) -> None:
    crosswalk = "state,district\nCT,1\n"
    digest = hashlib.sha256(crosswalk.encode()).hexdigest()
    main, sha = _tiny_main(
        tmp_path,
        {
            _FEED_PIN_REL: json.dumps({"facts_sha256": "c" * 64}, indent=2) + "\n",
            _CROSSWALK_REL: crosswalk,
        },
    )
    result = _input_settings_result(
        tmp_path,
        {"MAIN": str(main), "FEED_SHA": "c" * 64, "CROSSWALK_SHA": digest},
        _report_lines(f"\ncheck_commit {sha} {sha}\n"),
    )
    assert result.returncode == 0, result.stderr.decode()
    lines = result.stdout.decode().splitlines()
    assert f"NOTE us/chronicle_feed.json at {sha} pins facts_sha256 {'c' * 64}" in lines
    assert f"NOTE packaged CD vintage crosswalk at {sha} = {digest}" in lines
    blockers = [line for line in lines if line.startswith("BLOCKER ")]
    assert not [line for line in blockers if "chronicle_feed.json" in line], blockers
    assert not [line for line in blockers if "CD vintage crosswalk" in line], blockers


def _emitter_stub(
    where: Path, answer: str, status: int = 0, name: str = "emitter-stub"
) -> Path:
    return write_executable(
        where / name,
        "#!/bin/bash\n"
        '[ "$#" = 1 ] && [ "$1" = version ] || exit 64\n'
        f"printf '%s\\n' {shlex.quote(answer)}\n"
        f"exit {status}\n",
    )


_FLOOR = '{"minimum_emitter_version": "0.16.0"}\n'
_GOOD_EMITTER = '{"emitter_version": "0.16.0", "protocol_versions": [1]}'


def _emitter_line(floor: str, env: dict[str, str], cwd: Path | None = None) -> str:
    """The one line ``tools/route_a/check_emitter.py`` prints."""

    result = subprocess.run(
        [sys.executable, "-I", str(ROUTE_A_TOOLS / "check_emitter.py")],
        input=floor,
        env={"PATH": "/usr/bin:/bin", **env},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1, lines
    return lines[0]


def _assert_problem(line: str, problem: str, required: bool) -> None:
    assert problem in line, line
    if required:
        assert line.startswith("BLOCKER "), line
        assert "would stop before any work" in line
    else:
        assert line.startswith("NOTE "), line
        assert "without collector telemetry" in line


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize(
    ("answer", "status", "problem"),
    [
        (_GOOD_EMITTER, 0, None),
        ('{"emitter_version": "0.17.2rc1", "protocol_versions": [1, 2]}', 0, None),
        (
            '{"emitter_version": "0.13.0", "protocol_versions": [1]}',
            0,
            "older than the commit's floor 0.16.0",
        ),
        (
            '{"emitter_version": "0.9.10", "protocol_versions": [1]}',
            0,
            "older than the commit's floor 0.16.0",
        ),
        ("not json", 0, "printed no JSON object"),
        ('["0.16.0"]', 0, "printed no JSON object"),
        ('{"emitter_version": "dev"}', 0, "reported no release version"),
        (
            '{"emitter_version": "v0.16.0", "protocol_versions": [1]}',
            0,
            "reported no release version",
        ),
        (_GOOD_EMITTER, 7, "exited 7"),
    ],
)
def test_emitter_check_follows_the_releases_version_rules(
    tmp_path: Path, answer: str, status: int, problem: str | None, required: bool
) -> None:
    """``check_emitter.py`` repeats ``telemetry_emitter._start_emitter``: the
    version command must exit 0 with a JSON object whose ``emitter_version``
    starts with a release at or above the floor."""

    env = {"MICROCOSM_EMITTER_COMMAND": str(_emitter_stub(tmp_path, answer, status))}
    if required:
        env["MICROCOSM_TELEMETRY_REQUIRED"] = "1"
    line = _emitter_line(_FLOOR, env)
    if problem is None:
        assert line.startswith("NOTE microcosm-emitter 0."), line
        assert line.endswith("meets the commit's floor 0.16.0")
        return
    _assert_problem(line, problem, required)


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ('{"emitter_version": "0.17.2"}', "printed no list of protocol versions"),
        (
            '{"emitter_version": "0.17.2", "protocol_versions": ["1"]}',
            "printed no list of protocol versions",
        ),
        (
            '{"emitter_version": "0.17.2", "protocol_versions": [2]}',
            "not protocol version 1",
        ),
    ],
)
def test_a_required_build_also_needs_protocol_version_1(
    tmp_path: Path, answer: str, problem: str
) -> None:
    command = str(_emitter_stub(tmp_path, answer))
    required = _emitter_line(
        _FLOOR,
        {"MICROCOSM_EMITTER_COMMAND": command, "MICROCOSM_TELEMETRY_REQUIRED": "1"},
    )
    _assert_problem(required, problem, True)
    # The release reads the protocol list only when telemetry is required.
    optional = _emitter_line(_FLOOR, {"MICROCOSM_EMITTER_COMMAND": command})
    assert optional.startswith("NOTE microcosm-emitter 0.17.2 at "), optional


def test_emitter_command_lookup_follows_the_release(tmp_path: Path) -> None:
    stub = _emitter_stub(tmp_path, _GOOD_EMITTER)
    # Padding, Unicode whitespace included, is stripped as str.strip() does.
    padded = _emitter_line(_FLOOR, {"MICROCOSM_EMITTER_COMMAND": f"\u0085 {stub}\t"})
    assert (
        padded
        == f"NOTE microcosm-emitter 0.16.0 at {stub} meets the commit's floor 0.16.0"
    )
    # Unset, the default name on PATH.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    on_path = _emitter_stub(bin_dir, _GOOD_EMITTER, name="microcosm-emitter")
    found = _emitter_line(_FLOOR, {"PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert found.startswith(f"NOTE microcosm-emitter 0.16.0 at {on_path} "), found
    missing = _emitter_line(_FLOOR, {})
    _assert_problem(
        missing, "no microcosm-emitter found (microcosm-emitter on PATH)", False
    )
    # A relative path resolves against the working directory, and the stages
    # run in the build worktree, so it is refused even where it resolves.
    relative = _emitter_line(
        _FLOOR,
        {
            "MICROCOSM_EMITTER_COMMAND": "bin/microcosm-emitter",
            "MICROCOSM_TELEMETRY_REQUIRED": "1",
        },
        cwd=tmp_path,
    )
    _assert_problem(relative, "is the relative path 'bin/microcosm-emitter'", True)


def test_a_relative_path_entry_before_the_emitter_is_refused(
    tmp_path: Path,
) -> None:
    """``shutil.which`` searches relative and empty PATH entries against the
    working directory, and the stages run in the build worktree."""

    local = tmp_path / "bin"
    local.mkdir()
    _emitter_stub(local, _GOOD_EMITTER, name="microcosm-emitter")
    required = {"MICROCOSM_TELEMETRY_REQUIRED": "1"}
    relative = _emitter_line(
        _FLOOR, {**required, "PATH": "bin:/usr/bin:/bin"}, cwd=tmp_path
    )
    _assert_problem(relative, "the relative PATH entry 'bin' comes before", True)
    empty = _emitter_line(_FLOOR, {**required, "PATH": ":/usr/bin:/bin"}, cwd=local)
    _assert_problem(empty, "an empty PATH entry (the working directory)", True)
    # Found in an absolute entry first: the relative one after it is never
    # searched.
    absolute = tmp_path / "abs"
    absolute.mkdir()
    found = _emitter_stub(absolute, _GOOD_EMITTER, name="microcosm-emitter")
    line = _emitter_line(
        _FLOOR, {**required, "PATH": f"{absolute}:bin:/usr/bin"}, cwd=tmp_path
    )
    assert line.startswith(f"NOTE microcosm-emitter 0.16.0 at {found} "), line
    # An absolute MICROCOSM_EMITTER_COMMAND does not search PATH.
    line = _emitter_line(
        _FLOOR,
        {**required, "PATH": "bin:/usr/bin", "MICROCOSM_EMITTER_COMMAND": str(found)},
        cwd=tmp_path,
    )
    assert line.startswith(f"NOTE microcosm-emitter 0.16.0 at {found} "), line


@pytest.mark.parametrize("required", [False, True])
def test_undecodable_version_output_is_a_failed_run_as_in_the_release(
    tmp_path: Path, required: bool
) -> None:
    """The release catches any exception from ``version`` (a decoding error
    included) and goes on without an emitter unless telemetry is required."""

    stub = write_executable(
        tmp_path / "bytes-emitter",
        "#!/bin/bash\n"
        f"printf '%s\\n' {shlex.quote(_GOOD_EMITTER)}\n"
        "printf '\\377\\n' >&2\n",
    )
    env = {"MICROCOSM_EMITTER_COMMAND": str(stub)}
    if required:
        env["MICROCOSM_TELEMETRY_REQUIRED"] = "1"
    line = _emitter_line(_FLOOR, env)
    _assert_problem(line, "could not be run (UnicodeDecodeError)", required)


def test_a_version_command_that_hangs_is_cut_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "route_a_check_emitter", ROUTE_A_TOOLS / "check_emitter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TIMEOUT_SECONDS == 15.0  # the release's STARTUP_TIMEOUT_SECONDS
    monkeypatch.setattr(module, "TIMEOUT_SECONDS", 0.5)
    hang = write_executable(tmp_path / "hang", "#!/bin/bash\nexec sleep 30\n")
    monkeypatch.setenv("MICROCOSM_EMITTER_COMMAND", str(hang))
    monkeypatch.setenv("MICROCOSM_TELEMETRY_REQUIRED", "1")
    _assert_problem(module.check(_FLOOR), "did not answer within 0.5 s", True)


@pytest.mark.parametrize(
    "floor",
    [
        '{"minimum_emitter_version": "0.16.0junk"}',
        '{"minimum_emitter_version": "0.16"}',
        '{"minimum_emitter_version": ""}',
        "{}",
        "not json",
    ],
)
def test_a_floor_the_release_cannot_read_blocks(tmp_path: Path, floor: str) -> None:
    command = str(_emitter_stub(tmp_path, _GOOD_EMITTER))
    line = _emitter_line(floor, {"MICROCOSM_EMITTER_COMMAND": command})
    assert line.startswith("BLOCKER the commit's minimum_emitter_version.json"), line


@pytest.mark.parametrize("setting", ["yes", "true", " 1", ""])
def test_a_telemetry_setting_the_release_refuses_blocks(
    tmp_path: Path, setting: str
) -> None:
    command = str(_emitter_stub(tmp_path, _GOOD_EMITTER))
    line = _emitter_line(
        _FLOOR,
        {"MICROCOSM_EMITTER_COMMAND": command, "MICROCOSM_TELEMETRY_REQUIRED": setting},
    )
    assert line == (
        f"BLOCKER MICROCOSM_TELEMETRY_REQUIRED is {setting!r}; the release "
        "accepts only 1, 0 or unset and refuses anything else"
    )


def _driver_emitter(tmp_path: Path, overrides: dict[str, str], sha: str) -> list[str]:
    result = _input_settings_result(
        tmp_path,
        overrides,
        _report_lines(
            f"\nSCRIPT_DIR={shlex.quote(str(ROUTE_A_TOOLS))}\n"
            f"COMMIT_NOTES=(); COMMIT_BLOCKERS=()\ncheck_emitter {sha}\n"
        ),
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().splitlines()


def test_driver_settings_reach_the_emitter_check_and_the_stages(tmp_path: Path) -> None:
    """Settings-file values are exported, so the check and the stages (which
    inherit the driver's environment) see them."""

    main, sha = _tiny_main(tmp_path, {_EMITTER_FLOOR_REL: _FLOOR})
    old = str(_emitter_stub(tmp_path, '{"emitter_version": "0.13.0"}'))
    good = str(_emitter_stub(tmp_path, _GOOD_EMITTER, name="good-emitter"))
    required = {"MAIN": str(main), "MICROCOSM_TELEMETRY_REQUIRED": "1"}
    assert _driver_emitter(
        tmp_path, {**required, "MICROCOSM_EMITTER_COMMAND": old}, sha
    ) == [
        f"BLOCKER microcosm-emitter 0.13.0 at {old} is older than the commit's floor "
        "0.16.0 (uv tool install --upgrade 'microcosm-emitter>=0.16.0', or point "
        "MICROCOSM_EMITTER_COMMAND at a newer one); MICROCOSM_TELEMETRY_REQUIRED=1, "
        "so the release would stop before any work"
    ]
    assert _driver_emitter(
        tmp_path, {**required, "MICROCOSM_EMITTER_COMMAND": good}, sha
    ) == [f"NOTE microcosm-emitter 0.16.0 at {good} meets the commit's floor 0.16.0"]
    exported = _input_settings_result(
        tmp_path,
        {**required, "MICROCOSM_EMITTER_COMMAND": good},
        "\nprintenv MICROCOSM_TELEMETRY_REQUIRED\nprintenv MICROCOSM_EMITTER_COMMAND\n",
    )
    assert exported.stdout.decode().splitlines() == ["1", good]


def test_the_emitter_floor_is_read_at_the_requested_commit(tmp_path: Path) -> None:
    """``--resolve`` reads the floor of the commit it checks, not of the
    checkout: an older commit with a lower floor accepts an older emitter."""

    main, older = _tiny_main(
        tmp_path, {_EMITTER_FLOOR_REL: '{"minimum_emitter_version": "0.13.0"}\n'}
    )
    (main / _EMITTER_FLOOR_REL).write_text(_FLOOR, encoding="utf-8")
    git = ["git", "-C", str(main), "-c", "user.name=fixture"]
    git += ["-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false"]
    git += ["-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "commit", "-qam", "raise"], check=True, timeout=20)
    old = str(
        _emitter_stub(
            tmp_path, '{"emitter_version": "0.13.0", "protocol_versions": [1]}'
        )
    )
    settings = {
        "MAIN": str(main),
        "MICROCOSM_TELEMETRY_REQUIRED": "1",
        "MICROCOSM_EMITTER_COMMAND": old,
    }
    assert _driver_emitter(tmp_path, settings, older) == [
        f"NOTE microcosm-emitter 0.13.0 at {old} meets the commit's floor 0.13.0"
    ]
    line = _driver_emitter(tmp_path, settings, "HEAD")
    assert len(line) == 1 and "older than the commit's floor 0.16.0" in line[0]


def test_emitter_check_skips_a_commit_without_a_floor(tmp_path: Path) -> None:
    main, sha = _tiny_main(tmp_path, {"README.md": "fixture\n"})
    assert _driver_emitter(
        tmp_path,
        {"MAIN": str(main), "MICROCOSM_EMITTER_COMMAND": str(tmp_path / "never-run")},
        sha,
    ) == [f"NOTE no {_EMITTER_FLOOR_REL} at {sha}; emitter not checked"]


_SYNC_SHA = "d" * 40


def _ensure_worktree(
    tmp_path: Path,
    overrides: dict[str, str],
    marker: str | None = None,
    environment: str = "",
) -> tuple[subprocess.CompletedProcess[bytes], Path]:
    """Run ``ensure_worktree`` with stub ``git`` and ``uv``; ``marker`` is an
    earlier sync's ``.route_a_synced`` in a worktree that already exists, and
    ``environment`` is shell run first (exports a caller's shell would hand
    down). The stub refuses ``--locked`` whenever ``UV_FROZEN`` is set; uv
    0.11 refuses it beside a true value and fails to parse an empty one."""

    recorded = tmp_path / "uv-args"
    uv = write_executable(
        tmp_path / "uv-stub",
        "#!/bin/bash\n"
        'if [ -n "${UV_FROZEN+set}" ]; then\n'
        '  for a in "$@"; do\n'
        '    [ "$a" != --locked ] || {\n'
        "      echo 'error: the argument `--locked` cannot be used with `UV_FROZEN`' >&2\n"
        "      exit 2\n"
        "    }\n"
        "  done\n"
        "fi\n"
        f'printf "%s\\n" "$@" > {shlex.quote(str(recorded))}\n'
        "mkdir -p .venv/bin\n"
        "printf '#!/bin/bash\\nexit 0\\n' > .venv/bin/python\n"
        "chmod +x .venv/bin/python\n",
    )
    if marker is not None:
        venv = tmp_path / "wt_root" / f"microcosm-route-a-{_SYNC_SHA[:12]}" / ".venv"
        venv.mkdir(parents=True)
        (venv / ".route_a_synced").write_text(marker, encoding="utf-8")
    commands = (
        "\ngit() {\n"
        '  case "$3" in\n'
        '    worktree) mkdir -p "$6" ;;\n'
        "    status) : ;;\n"
        '    rev-parse) printf "%s\\n" "$FIXTURE_SHA" ;;\n'
        "  esac\n"
        "}\n"
        f"FIXTURE_SHA={_SYNC_SHA}\n"
        + environment
        + "\n"
        + _function("ensure_worktree")
        + '\nensure_worktree "$FIXTURE_SHA"\n'
    )
    run_root = tmp_path / "run-root"
    run_root.mkdir()
    result = _input_settings_result(
        tmp_path, {"UV": str(uv), "RUN_ROOT": str(run_root), **overrides}, commands
    )
    return result, recorded


@pytest.mark.parametrize("setting", [None, "3.13"])
def test_worktree_sync_names_its_interpreter(
    tmp_path: Path, setting: str | None
) -> None:
    """A fresh worktree's uv sync names an interpreter the lock has wheels
    for; without one uv takes the newest installed CPython."""

    overrides = {} if setting is None else {"UV_SYNC_PYTHON": setting}
    result, recorded = _ensure_worktree(tmp_path, overrides)
    assert result.returncode == 0, result.stderr.decode()
    assert recorded.read_text(encoding="utf-8").splitlines() == [
        "sync",
        "--python",
        setting or "3.14",
        "--all-packages",
        "--locked",
        "--extra",
        "us",
    ]
    marker = tmp_path / "wt_root" / f"microcosm-route-a-{_SYNC_SHA[:12]}"
    synced = (marker / ".venv" / ".route_a_synced").read_text(encoding="utf-8")
    assert synced.splitlines()[1] == f"python={setting or '3.14'}"


@pytest.mark.parametrize("frozen", ["1", ""])
def test_worktree_sync_drops_an_inherited_uv_frozen(
    tmp_path: Path, frozen: str
) -> None:
    """A driver started from a shell that exports UV_FROZEN=1 (as a profile
    may) still syncs: uv refuses --locked beside a true UV_FROZEN and fails on
    an empty one, so the sync drops it."""

    result, recorded = _ensure_worktree(
        tmp_path, {}, environment=f"export UV_FROZEN={shlex.quote(frozen)}"
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    assert "--locked" in recorded.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("marker", "setting", "outcome"),
    [
        ("2026-11-16 10:00:00\npython=3.14\n", None, "kept"),
        ("2026-11-16 10:00:00\npython=3.13\n", "3.13", "kept"),
        ("2026-11-16 10:00:00\n", None, "unrecorded"),
        ("2026-11-16 10:00:00\npython=3.13\n", None, "refused"),
    ],
)
def test_a_resumed_worktree_keeps_its_venv_only_for_the_same_interpreter(
    tmp_path: Path, marker: str, setting: str | None, outcome: str
) -> None:
    overrides = {} if setting is None else {"UV_SYNC_PYTHON": setting}
    result, recorded = _ensure_worktree(tmp_path, overrides, marker)
    out = result.stdout.decode()
    assert not recorded.exists(), "a synced worktree is not synced again"
    if outcome == "refused":
        assert result.returncode == 1
        assert "was synced with --python 3.13, not UV_SYNC_PYTHON=3.14" in out
        assert (tmp_path / "run-root" / "ROUTE_A_FAILED").exists()
        return
    assert result.returncode == 0, result.stderr.decode()
    assert ("before its interpreter was recorded" in out) == (outcome == "unrecorded")


def test_shell_scripts_are_valid_bash() -> None:
    scripts = _shell_sources()
    assert scripts
    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, check=False, timeout=20
        )
        assert result.returncode == 0, result.stderr.decode()


def test_shell_scripts_pass_shellcheck() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not installed")
    scripts = _shell_sources()
    assert scripts
    result = subprocess.run(
        [shellcheck, *map(str, scripts)], capture_output=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()


@pytest.mark.parametrize("staging", [None, "1", "0"])
def test_every_release_flag_is_declared(staging: str | None) -> None:
    checker = _flag_checker()
    source = (
        _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    ).read_text(encoding="utf-8")
    argv = _release_argv(
        staging,
        tail="/fixture/tail register.json",
        extras=_example_extra_args() + " --epochs 6000",
    )
    flags = [item.partition("=")[0] for item in argv if item.startswith("--")]
    assert flags
    assert not checker.missing_flags(source, flags)
    assert checker.missing_flags(source, ["--route-a-undeclared-flag"])


@pytest.mark.parametrize("staging", [None, "1", "0"])
def test_staging_controls_only_release_wrapper_and_no_staging_flag(
    staging: str | None,
) -> None:
    argv = _release_argv(staging)
    enabled = staging != "0"
    assert ("--no-staging" in argv) is not enabled
    assert argv.count("--no-staging") == (0 if enabled else 1)
    if enabled:
        assert argv[:2] == [
            str(ROUTE_A_TOOLS / "with_hf_token.sh"),
            "/fixture/agent-secret",
        ]
    else:
        assert argv[:9] == [
            "/usr/bin/env",
            "-u",
            "HF_TOKEN",
            "-u",
            "HUGGING_FACE_HUB_TOKEN",
            "-u",
            "HUGGINGFACE_HUB_TOKEN",
            "-u",
            "HUGGING_FACE_TOKEN_MAX",
        ]
        assert argv[9] == sys.executable
        assert str(ROUTE_A_TOOLS / "with_hf_token.sh") not in argv
    assert argv[argv.index("--base-h5") + 1] == "/fixture/base with spaces.h5"


@pytest.mark.parametrize("staging", ["", "2", "false", "yes"])
def test_release_rejects_invalid_staging_setting(tmp_path: Path, staging: str) -> None:
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    prefix, delimiter, _ = source.partition('mkdir -p "$RUN_ROOT"')
    assert delimiter
    # Execute only settings validation. The remainder can fetch refs and run
    # stages, so it must never be executed by an engine-free test.
    script = tmp_path / "settings-check.sh"
    script.write_text(prefix, encoding="utf-8")
    required = (
        "PE MAIN WT_ROOT RUN_ROOT CHAIN_ROOT CHAIN_LOG DISK_PATH UV PYTHON "
        "STORAGE EDU FEED LADDER SSI SCF AGENT_SECRET"
    ).split()
    settings = tmp_path / "route_a.env"
    settings.write_text(
        "\n".join(f"{key}=/fixture/{key.lower()}" for key in required)
        + f"\nROUTE_A_STAGING={shlex.quote(staging)}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "ROUTE_A_ENV": str(settings)},
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert "ROUTE_A_STAGING" in result.stderr.decode()


@pytest.mark.parametrize(
    "extra",
    [
        "--no-staging",
        "--no-staging=0",
        "--out=/fixture",
        "--base-h5=/fixture",
        "--allow-unpinned-feed",
        "--skip-reform-validation",
        "--dense-default-dataset",
    ],
)
def test_release_extra_arguments_preserve_refusals(extra: str) -> None:
    script = "set -eu\nfail() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
    script += f"RELEASE_EXTRA_ARGS={shlex.quote(extra)}\nDENSE_RELEASE_D122=0\n"
    script += _function("refuse_bad_release_args") + "\nrefuse_bad_release_args\n"
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, timeout=20
    )
    assert result.returncode != 0
    assert extra in result.stderr.decode()


def test_flag_checker_reads_argument_declarations() -> None:
    checker = _flag_checker()
    source = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--real", "-r")
group = parser.add_mutually_exclusive_group()
group.add_argument("--other")
unrelated = "--not-declared"
# parser.add_argument("--comment")
"""
    assert checker.declared_flags(source) == {"--real", "-r", "--other"}
    assert not checker.missing_flags(source, ["--real", "--other"])
    assert set(checker.missing_flags(source, ["--comment", "--not-declared"])) == {
        "--comment",
        "--not-declared",
    }


def test_wrapper_secret_reaches_only_child_environment(tmp_path: Path) -> None:
    """The wrapper hands the credential to the exec'd child's environment only.

    Engine-free and psutil-free: it runs the wrapper directly under ``bash -x``.
    The same check through the real supervisor, which needs psutil, lives in
    ``engine_workflow/us/test_route_a_driver.py``.
    """

    wrapper_source = (ROUTE_A_TOOLS / "with_hf_token.sh").read_text(encoding="utf-8")
    assert re.search(r"(?m)^export HF_TOKEN\s*$", wrapper_source)
    assert re.search(r'(?m)^exec "\$@"\s*$', wrapper_source)
    # An intermediate `env HF_TOKEN=...` launcher would expose the credential
    # briefly even when the final child's argv is clean.
    assert not re.search(r"\b(?:exec|env)\b[^\n]*\bHF_TOKEN\b", wrapper_source)
    # This deliberately does not resemble a Hub credential and never invokes
    # the actual agent-secret executable.
    marker = "route-a-dummy-environment-value"
    secret = write_executable(
        tmp_path / "secret-stub",
        """#!/bin/bash
set -eu
[ "$#" = 2 ] && [ "$1" = get ] && [ "$2" = HUGGING_FACE_TOKEN_MAX ]
printf '%s\\n' "$ROUTE_A_TEST_SECRET"
printf '%s\\n' "$ROUTE_A_TEST_SECRET" >&2
""",
    )
    report = tmp_path / "child-report.json"
    child = tmp_path / "child.py"
    child.write_text(
        """import json, os, subprocess, sys
marker = os.environ["ROUTE_A_TEST_SECRET"]
argv = subprocess.run(
    ["ps", "-ww", "-o", "args=", "-p", str(os.getpid())],
    capture_output=True, text=True, check=True,
).stdout
payload = {
    "token_present": os.environ.get("HF_TOKEN") == marker,
    "argv_read": bool(argv.strip()),
    "argv_contains_token": marker in argv,
    "aliases_present": [key for key in (
        "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_TOKEN_MAX"
    ) if key in os.environ],
    "pid": os.getpid(),
}
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream)
print("dummy release child finished")
""",
        encoding="utf-8",
    )
    env = {**os.environ, "ROUTE_A_TEST_SECRET": marker, "HF_TOKEN": "stale-value"}
    env.update(dict.fromkeys(SECRET_ALIASES, "stale-value"))
    process = subprocess.Popen(
        [
            "bash",
            "-x",
            str(ROUTE_A_TOOLS / "with_hf_token.sh"),
            str(secret),
            sys.executable,
            str(child),
            str(report),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, stderr.decode()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["token_present"] is True
    assert payload["argv_read"] is True
    assert payload["argv_contains_token"] is False
    assert payload["aliases_present"] == []
    # exec, not a fork: the release runs as the wrapper's own process.
    assert payload["pid"] == process.pid
    assert marker.encode() not in stdout + stderr
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert marker.encode() not in path.read_bytes(), path.name


@pytest.mark.parametrize("secret_result", ["empty", "failed"])
def test_wrapper_refuses_missing_secret(tmp_path: Path, secret_result: str) -> None:
    secret = write_executable(
        tmp_path / "secret-stub",
        "#!/bin/bash\n"
        + (
            "exit 0\n"
            if secret_result == "empty"
            else "printf 'dummy-lookup-output\\n'\nexit 9\n"
        ),
    )
    sentinel = tmp_path / "child-started"
    child = write_executable(tmp_path / "child-stub", '#!/bin/bash\ntouch "$1"\n')
    result = subprocess.run(
        [
            "bash",
            str(ROUTE_A_TOOLS / "with_hf_token.sh"),
            str(secret),
            str(child),
            str(sentinel),
        ],
        env={**os.environ, "HF_TOKEN": "inherited-value-must-not-be-used"},
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert not sentinel.exists()


def test_route_a_sources_contain_no_machine_paths_or_credentials() -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(_TEST_PATHS.repository),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "tools/route_a/",
        ],
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()
    source_paths = sorted(
        ROUTE_A_TOOLS / Path(path).relative_to("tools/route_a")
        for path in result.stdout.decode().rstrip("\0").split("\0")
        if path
    )
    assert source_paths
    credential = re.compile(
        r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    )
    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        assert "/Users/" not in source, path.name
        assert not credential.search(source), path.name


def test_route_a_preflights_the_build_in_its_release_directory() -> None:
    """A national run writes its build into the national release's directory,
    and the publication command that publishes into build folders runs its
    preflight (microcosm#1570)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")

    assert "REL_DIR=$REL_OUT/releases/microcosm-us-2024-national" in source
    assert '"$W/.venv/bin/microcosm-publish-release-build"' in source
    assert '"$W/.venv/bin/microcosm-publish-release" ' not in source


def test_route_a_hands_over_the_emitter_publication_commands() -> None:
    """The hand-off record's next step publishes the build through the
    emitter and then makes it current, in place of tools/publish_release.sh
    (microcosm#1570)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")

    assert "tools/publish_release.sh" not in source
    # Shell-quoted, as the builders' publication_commands are, so a run root
    # with a space or a shell character still pastes as one argument (review
    # of #1571).
    command = '["uv", "run", "microcosm-publish-release-build", rel_dir'
    assert (
        f'shlex.join({command}, "--repo-id", "policyengine/populace-us", '
        '"--artifact-root", art])'
    ) in source
    assert (
        f'shlex.join({command}, "--repo-id", "policyengine/populace-us", "--promote"])'
    ) in source
