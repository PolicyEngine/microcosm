"""Exercise Route A orchestration without loading a country engine."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
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
# production input or git repository is read. The driver's own size helper
# runs, so this also proves it on the CI runner's (GNU) stat.
registered_rows=$(input_rows)
input_rows() { printf '%s\\n' "$registered_rows" | awk -F '|' '$1 == "release_ledger_facts"'; }
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


@pytest.mark.parametrize("override", [False, True])
def test_ladder_pins_keep_defaults_or_settings_overrides(
    tmp_path: Path, override: bool
) -> None:
    overrides = {"LADDER_SHA": "d" * 64, "LADDER_BYTES": "456"} if override else {}
    result = run_driver_settings(tmp_path, overrides, "\ninput_rows\n")
    assert result.returncode == 0, result.stderr.decode()
    rows = {
        row[0]: row[1:]
        for row in (line.split("|") for line in result.stdout.decode().splitlines())
    }
    # The default is the published run's d713 ladder.
    assert rows["block_ladder_npz"][1:3] == (
        ["d" * 64, "456"]
        if override
        else [
            "6840b990acdfa2003d7723da5595e3cfff7205a4fa3d8d6206ed835c85c3f233",
            "18991218",
        ]
    )


@pytest.mark.parametrize(
    ("rule", "members", "problem"),
    [
        ("block_v1", ("block_geoid.npy", "puma.npy"), None),
        ("block_v1", ("block_geoid.npy",), "has no per-block puma array"),
        ("block_v1", None, "could not be read to list its arrays"),
        ("legacy", ("block_geoid.npy",), None),
    ],
)
def test_block_v1_requires_the_ladder_s_puma_array(
    tmp_path: Path, rule: str, members: tuple[str, ...] | None, problem: str | None
) -> None:
    ladder = tmp_path / "ladder.npz"
    if members is None:
        ladder.write_bytes(b"not a zip archive\n")
    else:
        with zipfile.ZipFile(ladder, "w") as archive:
            for name in members:
                archive.writestr(name, b"fixture")
    payload = ladder.read_bytes()
    result = run_driver_settings(
        tmp_path,
        {
            "LOCATION_RULE": rule,
            "LADDER": str(ladder),
            "LADDER_SHA": hashlib.sha256(payload).hexdigest(),
            "LADDER_BYTES": str(len(payload)),
        },
        """
# Only the ladder row, with its true digest and size, so any problem is the
# PUMA check's.
registered_rows=$(input_rows)
input_rows() { printf '%s\\n' "$registered_rows" | awk -F '|' '$1 == "block_ladder_npz"'; }
verify_inputs
printf '%s\\n' ${INPUT_PROBLEMS[@]+"${INPUT_PROBLEMS[@]}"}
""",
    )
    assert result.returncode == 0, result.stderr.decode()
    problems = [line for line in result.stdout.decode().splitlines() if line]
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1
        assert problem in problems[0]
        assert str(ladder) in problems[0]


@pytest.mark.parametrize(
    "openssl",
    [
        # Nothing on stdout: openssl could not open the file.
        "openssl() { return 1; }",
        # An error on stdout instead of a digest.
        "openssl() { echo 'Operation not permitted'; }",
        # A digest-shaped line that is not 64 hex digits.
        "openssl() { echo '0123 *x'; }",
    ],
)
def test_an_input_openssl_cannot_read_is_unreadable_not_an_empty_digest(
    tmp_path: Path, openssl: str
) -> None:
    # On 2026-11-17 a launchd job's /usr/bin/openssl could not read the
    # archive drive, and sha_of turned that into an empty digest. Now it
    # prints nothing and fails, and the input is reported as unreadable.
    feed = tmp_path / "new-feed.jsonl"
    payload = b'{"fixture": "feed"}\n'
    feed.write_bytes(payload)
    result = run_driver_settings(
        tmp_path,
        {
            "FEED": str(feed),
            "FEED_SHA": hashlib.sha256(payload).hexdigest(),
            "FEED_BYTES": str(len(payload)),
        },
        f"""
registered_rows=$(input_rows)
input_rows() {{ printf '%s\\n' "$registered_rows" | awk -F '|' '$1 == "release_ledger_facts"'; }}
{openssl}
if out=$(sha_of "{feed}"); then echo "sha_of ok [$out]"; else echo "sha_of failed [$out]"; fi
verify_inputs
printf '%s\\n' "${{INPUT_PROBLEMS[@]}}"
""",
    )
    assert result.returncode == 0, result.stderr.decode()
    lines = result.stdout.decode().splitlines()
    assert lines[0] == "sha_of failed []"
    assert lines[1:] == [f"release_ledger_facts could not be read to hash it: {feed}"]
    report = json.loads((tmp_path / "run_root" / "inputs.resolved.json").read_text())
    assert report["inputs"][0]["status"] == "unreadable"
    assert report["inputs"][0]["observed_sha256"] == ""


def test_the_crosswalk_and_base_digests_refuse_an_unreadable_file() -> None:
    # These run after the fetch, where the settings harness stops, so the
    # source is checked: neither compares nor records an empty digest.
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    assert (
        'CROSSWALK_ACTUAL=$(sha_of "$W/$CROSSWALK_REL") || fail "could not read'
        in source
    )
    assert 'BASE_SHA=$(sha_of "$BASE_H5") || fail "could not read $BASE_H5' in source
    assert "[[ $BASE_SHA =~ ^[0-9a-f]{64}$ ]] || fail" in source
    assert '"$(sha_of ' not in source


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
    result = run_driver_settings(
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
    result = run_driver_settings(
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
    result = run_driver_settings(
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
    result = run_driver_settings(
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
    exported = run_driver_settings(
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
    result = run_driver_settings(
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


@pytest.mark.parametrize("rule", ["block_v2", "Block_v1", "block-v1", "legacy "])
def test_an_unknown_location_rule_is_refused(tmp_path: Path, rule: str) -> None:
    result = run_driver_settings(tmp_path, {"LOCATION_RULE": rule}, "\n")
    assert result.returncode == 64
    assert "LOCATION_RULE must be legacy or block_v1" in result.stderr.decode()


@pytest.mark.parametrize("rule", [None, "legacy", "block_v1"])
def test_check_commit_checks_the_location_flags_only_under_block_v1(
    tmp_path: Path, rule: str | None
) -> None:
    overrides = {} if rule is None else {"LOCATION_RULE": rule}
    result = run_driver_settings(tmp_path, overrides, '\nprintf "%s" "$BASE_FLAGS"\n')
    assert result.returncode == 0, result.stderr.decode()
    flags = result.stdout.decode().split()
    # The legacy list is the published run's, so an older commit stays clear.
    for flag in ("--ledger-facts", "--assign-congressional-districts"):
        assert flag in flags
    location = {"--location-rule", "--location-seed"}
    if rule == "block_v1":
        assert location <= set(flags)
    else:
        assert not location & set(flags)


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
    """Certification places the build in its tier's release directory, the
    national release's or the evidence release's (microcosm#1585 item 11), and
    the publication command that publishes into build folders runs its
    preflight there (microcosm#1570)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")

    assert (
        'if [ "$ROUTE_A_CERTIFY_TIER" = evidence ]; then\n'
        "  REL_RELEASE=microcosm-us-2024-evidence\n"
        "else\n"
        "  REL_RELEASE=microcosm-us-2024-national\n"
        "fi\n"
        "REL_DIR=$REL_OUT/releases/$REL_RELEASE\n"
    ) in source
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


def test_size_and_free_disk_helpers_run_on_this_host_s_stat_and_df(
    tmp_path: Path,
) -> None:
    # GNU stat (Linux) takes -c %s and BSD stat (macOS) -f %z; df -P keeps one
    # line per filesystem. Both helpers must work on whichever host runs them.
    sized = tmp_path / "sized.bin"
    sized.write_bytes(b"x" * 1234)
    disk = tmp_path / "disk"
    disk.mkdir()
    result = run_driver_settings(
        tmp_path,
        {"DISK_PATH": str(disk)},
        f"""
file_bytes "{sized}"
free_bytes
""",
    )
    assert result.returncode == 0, result.stderr.decode()
    size, free = result.stdout.decode().split()
    assert size == "1234"
    assert int(free) > 0


@pytest.mark.parametrize(("value", "status"), [("0", 0), ("1", 0), ("yes", 64)])
def test_stop_before_release_accepts_only_zero_or_one(
    tmp_path: Path, value: str, status: int
) -> None:
    result = run_driver_settings(
        tmp_path, {"STOP_BEFORE_RELEASE": value}, '\necho "$STOP_BEFORE_RELEASE"\n'
    )
    assert result.returncode == status, result.stderr.decode()
    if status == 0:
        assert result.stdout.decode().strip() == value
    else:
        assert "STOP_BEFORE_RELEASE must be 0 or 1" in result.stderr.decode()


def test_the_stop_before_release_sits_after_the_release_config_and_before_its_run() -> (
    None
):
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    config = source.index('write_config release 110 "$RELEASE_RAM_GIB"')
    stop = source.index(
        'if [ "$STOP_BEFORE_RELEASE" = 1 ] && ! stage_done release; then'
    )
    # The release stage also accepts a build that failed its release gates
    # (exit 3, microcosm#1585 item 11), which certification can only ship as
    # evidence.
    run = source.index(
        'run_stage release "0 $BUILT_FAILED_GATES_RC" "$RELEASE_RAM_GIB" 45'
    )
    assert config < stop < run


def test_route_a_stops_after_the_build_and_certifies_only_when_asked() -> None:
    """The release tool builds and stops; the driver hands the build over
    unless --certify (or ROUTE_A_CERTIFY=1) asks it to certify it, which reruns
    no accepted stage (microcosm#1585 item 11, PR 4a)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")

    assert "--certify) CERTIFY_FLAG=1 ;;" in source
    assert ': "${ROUTE_A_CERTIFY=0}"' in source
    assert 'case "$ROUTE_A_CERTIFY" in 0|1)' in source
    built = source.index('[ -f "$REL_OUT/build_record.json" ]')
    handoff = source.index('if [ "$ROUTE_A_CERTIFY" != 1 ]; then')
    certify = source.index("run_stage certify 0")
    certified = source.index('[ -f "$REL_DIR/release_manifest.json" ]')
    preflight = source.index("run_stage preflight-release")
    assert built < handoff < certify < certified < preflight
    assert '"$RUN_ROOT/ROUTE_A_BUILT.json"' in source
    # A build-only run whose base preflight is blocked exits 4, as a
    # certified one does; the example settings name the switch.
    built_only = source[handoff:certify]
    assert built_only.index('if [ "$PF_BLOCKED" = 1 ]; then') < built_only.index(
        "exit 4"
    )
    assert built_only.index("exit 4") < built_only.index("exit 0")
    example = (ROUTE_A_TOOLS / "route_a.env.example").read_text(encoding="utf-8")
    assert ': "${ROUTE_A_CERTIFY=0}"' in example
    # The certify stage runs the command certify_command builds, in the
    # release stage's environment.
    assert (
        "  certify_command\n"
        '  write_config certify 110 "$RELEASE_RAM_GIB" 45 '
        '"${CERTIFY_WALL_SECONDS:-345600}" 2000000 "$W" "$BUILD_ENV" -- \\\n'
        '    "${CERTIFY_ARGV[@]}"\n'
    ) in source


def _certify_argv(
    staging: str | None, *, tier: str | None = None, owners: str | None = None
) -> list[str]:
    settings = {
        "W": "/fixture/worktree",
        "REL_OUT": "/fixture/release-out",
        "TOKEN_WRAPPER": str(ROUTE_A_TOOLS / "with_hf_token.sh"),
        "AGENT_SECRET": "/fixture/agent-secret",
    }
    script = "set -eu\nfail() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
    script += "\n".join(
        f"{key}={shlex.quote(value)}" for key, value in settings.items()
    )
    script += (
        "\nunset ROUTE_A_STAGING ROUTE_A_CERTIFY_TIER ROUTE_A_EVIDENCE_FAILURE_OWNERS\n"
    )
    if staging is not None:
        script += f"ROUTE_A_STAGING={shlex.quote(staging)}\n"
    if tier is not None:
        script += f"ROUTE_A_CERTIFY_TIER={shlex.quote(tier)}\n"
    if owners is not None:
        script += f"ROUTE_A_EVIDENCE_FAILURE_OWNERS={shlex.quote(owners)}\n"
    script += _function("certify_command")
    script += '\ncertify_command\nprintf "%s\\0" "${CERTIFY_ARGV[@]}"\n'
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=False, timeout=20
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().rstrip("\0").split("\0")


@pytest.mark.parametrize("staging", [None, "1", "0"])
def test_certification_stages_like_the_release(staging: str | None) -> None:
    """Certification opens a staging run where its build staged, so it gets
    the release stage's staging credential; without staging it runs as the
    release stage does, online and without the HF tokens (microcosm#1585
    item 11, PR 4a)."""
    argv = _certify_argv(staging)
    command = [
        "/fixture/worktree/.venv/bin/python",
        "-B",
        "tools/certify_us_national_build.py",
        "/fixture/release-out",
    ]
    if staging == "0":
        release = _release_argv("0")
        launcher = release[: release.index(sys.executable)]
        assert launcher == [
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
        assert argv == [*launcher, *command, "--no-staging"]
    else:
        assert argv == [
            str(ROUTE_A_TOOLS / "with_hf_token.sh"),
            "/fixture/agent-secret",
            *command,
        ]


@pytest.mark.parametrize("owners", [None, "/fixture/owners with spaces.json"])
def test_the_certify_stage_passes_the_tier_it_was_given(owners: str | None) -> None:
    """A build has no tier: ROUTE_A_CERTIFY_TIER chooses it at certification,
    and the evidence tier passes the per-run owners file (microcosm#1585 item
    11). The certified tier, the default, passes neither flag."""
    command = [
        "/fixture/worktree/.venv/bin/python",
        "-B",
        "tools/certify_us_national_build.py",
        "/fixture/release-out",
    ]
    launcher = [str(ROUTE_A_TOOLS / "with_hf_token.sh"), "/fixture/agent-secret"]
    assert _certify_argv("1", tier="certified") == [*launcher, *command]
    expected = [*launcher, *command, "--evidence-release"]
    if owners is not None:
        expected += ["--evidence-failure-owners", owners]
    assert _certify_argv("1", tier="evidence", owners=owners) == expected


def _settings_file(tmp_path: Path, **overrides: str) -> Path:
    """A settings file that passes the driver's required-settings checks."""
    values = {
        name: str(tmp_path / name.lower())
        for name in (
            "PE",
            "MAIN",
            "WT_ROOT",
            "RUN_ROOT",
            "CHAIN_ROOT",
            "CHAIN_LOG",
            "DISK_PATH",
            "UV",
            "STORAGE",
            "EDU",
            "FEED",
            "LADDER",
            "SSI",
            "SCF",
            "AGENT_SECRET",
        )
    }
    values["PYTHON"] = sys.executable
    values.update(overrides)
    path = tmp_path / "route_a.env"
    path.write_text(
        "".join(f"{name}={shlex.quote(value)}\n" for name, value in values.items()),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ROUTE_A_CERTIFY_TIER": "gold"}, b"must be certified or evidence"),
        (
            {"ROUTE_A_EVIDENCE_FAILURE_OWNERS": "/fixture/owners.json"},
            b"needs ROUTE_A_CERTIFY_TIER=evidence",
        ),
        (
            {
                "ROUTE_A_CERTIFY_TIER": "evidence",
                "ROUTE_A_EVIDENCE_FAILURE_OWNERS": "owners.json",
            },
            b"must be an absolute path",
        ),
    ],
)
def test_route_a_refuses_a_tier_setting_it_cannot_use(
    tmp_path: Path, overrides: dict[str, str], message: bytes
) -> None:
    """The tier settings are checked with the other settings, before the run
    root is made or any ref is fetched."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ROUTE_A_")
    }
    env["ROUTE_A_ENV"] = str(_settings_file(tmp_path, **overrides))
    result = subprocess.run(
        ["bash", str(ROUTE_A_TOOLS / "route_a.sh"), "--certify"],
        capture_output=True,
        env=env,
        timeout=20,
    )
    assert result.returncode == 64, result.stderr.decode()
    assert message in result.stderr
    assert not (tmp_path / "run_root").exists()


def test_route_a_keeps_a_build_that_failed_its_release_gates() -> None:
    """A build that failed its release gates writes its H5 and record and
    exits 3 (the release tool's US_BUILD_FAILED_RELEASE_GATES_EXIT). The
    release stage accepts that, so the build is kept for an evidence-tier
    certification rather than moved aside and built again; a build-only run
    says so and exits 3, and a certify run at the certified tier fails before
    the certify stage starts (microcosm#1585 item 11)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    tool = (ROUTE_A_TOOLS.parent / "build_us_fiscal_refresh_release.py").read_text(
        encoding="utf-8"
    )

    assert "US_BUILD_FAILED_RELEASE_GATES_EXIT = 3\n" in tool
    assert "BUILT_FAILED_GATES_RC=3\n" in source
    assert 'result_accepted "$RUN/release-sup" "0 $BUILT_FAILED_GATES_RC"' in source
    assert 'run_stage release "0 $BUILT_FAILED_GATES_RC"' in source
    assert "run_stage release 0 " not in source
    handoff = source.index('if [ "$ROUTE_A_CERTIFY" != 1 ]; then')
    certify = source.index("run_stage certify 0")
    built_only = source[handoff:certify]
    assert '"release_gates_failed": failed,' in built_only
    assert built_only.index('if [ "$BUILD_FAILED_GATES" = 1 ]; then') < (
        built_only.index("exit 3")
    )
    assert built_only.index("exit 3") < built_only.index("exit 4")
    refusal = built_only.index(
        'if [ "$BUILD_FAILED_GATES" = 1 ] && [ "$ROUTE_A_CERTIFY_TIER" != evidence ]'
    )
    assert built_only.index("exit 0") < refusal
    example = (ROUTE_A_TOOLS / "route_a.env.example").read_text(encoding="utf-8")
    assert ': "${ROUTE_A_CERTIFY_TIER=certified}"' in example
    assert ': "${ROUTE_A_EVIDENCE_FAILURE_OWNERS=}"' in example


def test_route_a_removes_the_build_only_hand_off_once_certified() -> None:
    """Once certification is accepted, the build-only ROUTE_A_BUILT.json an
    earlier run left no longer holds, whatever the preflights after it find,
    so it goes then, not only with a run that succeeds (review of #1681)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    certified = source.index('[ -f "$REL_DIR/release_manifest.json" ] || fail')
    removed = source.index('rm -f "$RUN_ROOT/ROUTE_A_BUILT.json"')
    preflight = source.index("run_stage preflight-release")
    assert certified < removed < preflight
    assert source.count('rm -f "$RUN_ROOT/ROUTE_A_BUILT.json"') == 1
    readme = " ".join((ROUTE_A_TOOLS / "README.md").read_text(encoding="utf-8").split())
    assert (
        "Once certification is accepted the driver removes `ROUTE_A_BUILT.json`"
        in readme
    )


def _after_the_release_stage(
    tmp_path: Path,
    *,
    returncode: int,
    certify: str,
    tier: str = "certified",
    owners: str = "",
    blocked: str = "0",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the driver from the release stage's acceptance to the certify
    stage, on a release stage that exited ``returncode``."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    start = source.index('[ -f "$REL_OUT/build_record.json" ] || fail')
    end = source.index("wait_prior_supervisor certify\n")
    run_root = tmp_path / "run-root"
    run = run_root / "run-abc"
    (run / "release-sup").mkdir(parents=True)
    (run / "release-sup" / "ACCEPTED").write_text(
        f"returncode={returncode} 2026-11-18 20:00:00\n"
    )
    rel_out = run / "release-out" / "populace-us-2024-x"
    rel_out.mkdir(parents=True)
    (rel_out / "build_record.json").write_text("{}")
    settings = {
        "RUN_ROOT": str(run_root),
        "RUN": str(run),
        "REL_OUT": str(rel_out),
        "RID": "populace-us-2024-x",
        "PY": sys.executable,
        "C": "c" * 40,
        "PEUS": "2.145.3",
        "BASE_H5": "/fixture/base.h5",
        "BASE_SHA": "b" * 64,
        "PF_BLOCKED": blocked,
        "BUILT_FAILED_GATES_RC": "3",
        "ROUTE_A_CERTIFY": certify,
        "ROUTE_A_CERTIFY_TIER": tier,
        "ROUTE_A_EVIDENCE_FAILURE_OWNERS": owners,
        "LOG": str(run_root / "route_a.log"),
        "MODE": "run",
    }
    # The driver's own log, fail and sha_of, so the run is the driver's.
    helpers = source[source.index("log() {") : source.index("free_bytes() {")]
    script = "set -u\n"
    script += "".join(
        f"{key}={shlex.quote(value)}\n" for key, value in settings.items()
    )
    script += helpers + source[start:end]
    script += "echo REACHED_THE_CERTIFY_STAGE\n"
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    return result, run_root


def test_route_a_keeps_a_failed_build_and_certifies_it_only_as_evidence(
    tmp_path: Path,
) -> None:
    """Run, not read: a release stage that exited 3 (a build that failed its
    release gates) is kept. A build-only run says so and exits 3; a certify
    run refuses the certified tier before the certify stage, and the evidence
    tier goes on to it (review of #1681)."""
    built, root = _after_the_release_stage(tmp_path / "a", returncode=3, certify="0")
    assert built.returncode == 3, built.stderr
    hand_off = json.loads((root / "ROUTE_A_BUILT.json").read_text())
    assert hand_off["release_gates_failed"] is True
    assert hand_off["certified"] is False
    assert "ROUTE_A_CERTIFY_TIER=evidence" in hand_off["next"]
    assert "--rebuild" in hand_off["next"]
    assert hand_off["release_dry_run"] == "not run"
    assert "REACHED_THE_CERTIFY_STAGE" not in built.stdout

    green, root = _after_the_release_stage(tmp_path / "b", returncode=0, certify="0")
    assert green.returncode == 0, green.stderr
    assert (
        json.loads((root / "ROUTE_A_BUILT.json").read_text())["release_gates_failed"]
        is False
    )
    blocked, _ = _after_the_release_stage(
        tmp_path / "c", returncode=0, certify="0", blocked="1"
    )
    assert blocked.returncode == 4, blocked.stderr

    refused, root = _after_the_release_stage(tmp_path / "d", returncode=3, certify="1")
    assert refused.returncode == 1
    assert "refuses it at the certified tier" in refused.stdout
    assert (root / "ROUTE_A_FAILED").is_file()
    assert "REACHED_THE_CERTIFY_STAGE" not in refused.stdout

    owners = tmp_path / "owners.json"
    owners.write_text("{}")
    for case, given in (("e", ""), ("f", str(owners))):
        evidence, root = _after_the_release_stage(
            tmp_path / case, returncode=3, certify="1", tier="evidence", owners=given
        )
        assert evidence.returncode == 0, evidence.stderr
        assert "REACHED_THE_CERTIFY_STAGE" in evidence.stdout
        assert not (root / "ROUTE_A_FAILED").exists()
        digest = hashlib.sha256(owners.read_bytes()).hexdigest()
        assert (f"sha256 {digest}" in evidence.stdout) is bool(given)
    missing, _ = _after_the_release_stage(
        tmp_path / "g",
        returncode=3,
        certify="1",
        tier="evidence",
        owners=str(tmp_path / "absent.json"),
    )
    assert missing.returncode == 1
    assert "is not a file" in missing.stdout


def test_route_a_checks_the_certification_parser_for_the_tier_flags() -> None:
    """The flags the certify stage passes are checked against the build
    commit's certification parser, not its build parser, which declares
    --evidence-release for its dry run (microcosm#1585 item 11)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    assert (
        '"$PYTHON" "$SCRIPT_DIR/check_flags.py" --in-function _certify_parser '
        "$CERTIFY_FLAGS"
    ) in source
    checker = _flag_checker()
    tool = (ROUTE_A_TOOLS.parent / "build_us_fiscal_refresh_release.py").read_text(
        encoding="utf-8"
    )
    flags = ["--evidence-release", "--evidence-failure-owners", "--no-staging"]
    assert checker.missing_flags(tool, flags, function="_certify_parser") == []
    # The build parser declares them too, so only the narrowed check tells a
    # certification parser without the tier apart.
    assert checker.missing_flags(tool, ["--evidence-release"]) == []
    without_tier = (
        "def _certify_parser():\n"
        "    parser.add_argument('build_dir')\n"
        "    parser.add_argument('--no-staging')\n"
        "def _parse_args():\n"
        "    parser.add_argument('--evidence-release')\n"
    )
    assert checker.missing_flags(
        without_tier, ["--evidence-release"], function="_certify_parser"
    ) == ["--evidence-release"]
    assert checker.missing_flags(without_tier, ["--evidence-release"]) == []
    assert checker.missing_flags("x = 1\n", ["--a"], function="_certify_parser") == [
        "--a"
    ]


def test_route_a_certifies_an_accepted_build_without_hashing_its_inputs() -> None:
    """Certification reads none of the raw inputs, which the build verified
    when it ran, so --certify on an accepted build hashes them neither at the
    start nor after the wait (microcosm#1585 item 11, PR 4a)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    decided = source.index(
        'if [ "$ROUTE_A_CERTIFY" = 1 ] && '
        '[ -f "$RUN_ROOT/run-${C:0:12}/release-sup/ACCEPTED" ]; then'
    )
    first = source.index("verify_inputs\nfi\ncheck_commit")
    waited = source.index('if [ "$CERTIFY_ONLY" != 1 ]; then\n  verify_inputs\n')
    assert decided < first < waited
    # The run directory the decision reads is the one the stages use.
    assert "RUN=$RUN_ROOT/run-${C:0:12}\n" in source
    # A real run calls it only inside those two guards; --resolve, which
    # exits before the real run, hashes them as before.
    real_run = source[source.index("# Real run.") :]
    calls = [line for line in real_run.splitlines() if line.strip() == "verify_inputs"]
    assert calls == ["  verify_inputs", "  verify_inputs"]


@pytest.mark.parametrize("flag", ["--certify", "--resolve", "--rebuild", "--probe"])
def test_route_a_parses_its_modes_before_reading_settings(
    tmp_path: Path, flag: str
) -> None:
    script = ROUTE_A_TOOLS / "route_a.sh"
    env = {**os.environ, "ROUTE_A_ENV": str(tmp_path / "absent.env")}
    known = subprocess.run(
        ["bash", str(script), flag], capture_output=True, env=env, timeout=20
    )
    assert known.returncode == 64
    assert b"missing settings" in known.stderr
    unknown = subprocess.run(
        ["bash", str(script), "--bogus"], capture_output=True, env=env, timeout=20
    )
    assert unknown.returncode == 64
    assert b"[--resolve|--rebuild|--probe|--certify]" in unknown.stderr
    both = subprocess.run(
        ["bash", str(script), flag, "--certify"],
        capture_output=True,
        env=env,
        timeout=20,
    )
    assert both.returncode == 64
    assert b"one mode at a time" in both.stderr


def _run_functions(names: list[str], settings: dict[str, str], commands: str):
    """Run driver functions on their own, with ``settings`` as shell variables."""
    script = "set -u\nfail() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
    script += "".join(
        f"{key}={shlex.quote(value)}\n" for key, value in settings.items()
    )
    script += "".join(_function(name) + "\n" for name in names)
    script += commands
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False, timeout=30
    )


@pytest.mark.parametrize(("value", "status"), [("0", 0), ("1", 0), ("yes", 64)])
def test_the_dry_run_setting_accepts_only_zero_or_one(
    tmp_path: Path, value: str, status: int
) -> None:
    result = run_driver_settings(
        tmp_path, {"ROUTE_A_DRY_RUN": value}, '\necho "$ROUTE_A_DRY_RUN"\n'
    )
    assert result.returncode == status, result.stderr.decode()
    if status == 0:
        assert result.stdout.decode().strip() == value
    else:
        assert "ROUTE_A_DRY_RUN must be 0 or 1" in result.stderr.decode()


def test_the_release_gates_are_dry_run_before_the_release_starts() -> None:
    """The dry run is a stage between the release config and the release: 0
    and 2 let the release start, a certain failure (1) stops the run, and a
    dry run counts only for the settings it checked (microcosm#1042; the
    build is one command)."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    config = source.index('write_config release 110 "$RELEASE_RAM_GIB"')
    guard = source.index(
        'if [ "$RELEASE_PENDING" = 1 ] && [ "$ROUTE_A_DRY_RUN" = 1 ]; then'
    )
    dry_run = source.index('( run_stage release-dry-run "0 2" "$RELEASE_RAM_GIB" 22 )')
    key = source.index('echo "$DRY_KEY" > "$RUN/release-dry-run.key"')
    stop = source.index(
        'if [ "$STOP_BEFORE_RELEASE" = 1 ] && ! stage_done release; then'
    )
    run = source.index(
        'run_stage release "0 $BUILT_FAILED_GATES_RC" "$RELEASE_RAM_GIB" 45'
    )
    assert config < guard < dry_run < key < stop < run
    assert "stop point S1" in source[dry_run:key]
    # A stale dry run is moved aside, never deleted.
    assert (
        'mv "$RUN/release-dry-run-sup" "$RUN/release-dry-run-sup.stale-$(date +%s)"'
        in source[guard:dry_run]
    )


@pytest.mark.parametrize("staging", ["1", "0"])
def test_the_dry_run_starts_the_release_tool_as_the_release_stage_does(
    staging: str,
) -> None:
    settings = {
        "ROUTE_A_STAGING": staging,
        "TOKEN_WRAPPER": "/fixture/with_hf_token.sh",
        "AGENT_SECRET": "/fixture/agent-secret",
        "PY": "/fixture/python",
        "RUN": "/fixture/run dir",
    }
    result = _run_functions(
        ["dry_run_command"],
        settings,
        'dry_run_command\nprintf "%s\\0" "${DRY_RUN_ARGV[@]}"\n',
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.rstrip("\0").split("\0")
    tail = [
        "/fixture/python",
        "-B",
        "tools/dry_run_us_release_gates.py",
        "--release-config",
        "/fixture/run dir/release-config.json",
        "--json-out",
        "/fixture/run dir/release-dry-run.json",
    ]
    assert argv[-len(tail) :] == tail
    launcher = argv[: -len(tail)]
    if staging == "1":
        assert launcher == ["/fixture/with_hf_token.sh", "/fixture/agent-secret"]
    else:
        assert launcher[0] == "/usr/bin/env"
        assert "/fixture/with_hf_token.sh" not in launcher
        for alias in ("HF_TOKEN", *SECRET_ALIASES):
            assert alias in launcher


def test_the_release_settings_key_ignores_only_the_build_id() -> None:
    def key(build: str, *argv: str) -> str:
        result = _run_functions(
            ["release_settings_key"],
            {"PY": sys.executable},
            "release_settings_key " + " ".join(map(shlex.quote, (build, *argv))) + "\n",
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    first = key(
        "b-1", "--out", "/run/release-out/b-1", "--release-id", "b-1", "--seed", "0"
    )
    restart = key(
        "b-2", "--out", "/run/release-out/b-2", "--release-id", "b-2", "--seed", "0"
    )
    changed = key(
        "b-2", "--out", "/run/release-out/b-2", "--release-id", "b-2", "--seed", "1"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first == restart != changed


def test_release_ran_compares_the_command_the_release_ran(tmp_path: Path) -> None:
    (tmp_path / "release-sup").mkdir()
    argv = [
        "/w/python",
        "-B",
        "tools/build_us_fiscal_refresh_release.py",
        "--seed",
        "0",
    ]
    (tmp_path / "release-sup" / "COMMAND.json").write_text(json.dumps({"argv": argv}))
    settings = {"PY": sys.executable, "RUN": str(tmp_path)}
    same = _run_functions(
        ["release_ran"], settings, "release_ran " + shlex.join(argv) + "\n"
    )
    assert same.returncode == 0, same.stderr
    other = _run_functions(
        ["release_ran"], settings, "release_ran " + shlex.join([*argv[:-1], "1"]) + "\n"
    )
    assert other.returncode == 1


def test_a_build_rerun_with_other_settings_is_refused_until_rebuild() -> None:
    """A rerun to build keeps an accepted build only for the settings it was
    built with; with others it says how to build again, rather than keep the
    build without a word (review of #1681). Certification and the probe read
    the build, so they skip the check."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    rid = source.index('RID=$("$PY" -c \'import json,sys; a=json.load(')
    guard = source.index(
        'if [ "$MODE" = run ] && [ "$ROUTE_A_CERTIFY" != 1 ] && '
        '[ "$RELEASE_PENDING" = 0 ]; then'
    )
    check = source.index('release_ran "${RELEASE_ARGV[@]}"')
    inputs = source.index("inputs_now=$(release_inputs_key)")
    handoff = source.index('if [ "$ROUTE_A_CERTIFY" != 1 ]; then\n  "$PY" -')
    assert rid < guard < check < inputs < handoff
    assert "rerun with --rebuild" in source[check:inputs]
    assert "Rerun with --rebuild" in source[inputs:handoff]
    # The key the guard compares is the one the release was launched with,
    # and the dry run's key carries it too.
    launched = source.index('echo "$INPUTS_KEY" > "$RUN/release-inputs.key"')
    config = source.index('write_config release 110 "$RELEASE_RAM_GIB"')
    assert config < launched < rid
    assert '"inputs=$INPUTS_KEY"' in source


def _inputs_key(tmp_path: Path, **changes: str) -> str:
    """release_inputs_key over a fixture's inputs, with ``changes`` applied."""
    reference = tmp_path / "reference.h5"
    if not reference.exists():
        reference.write_bytes(b"reference")
    settings = {
        "STORAGE": "/fixture/storage",
        "EDU": "/fixture/edu",
        "BASE_LEDGER_FACTS": "/fixture/feed.jsonl",
        "BASE_LEDGER_FACTS_SHA": "a" * 64,
        "LADDER": "/fixture/ladder.npz",
        "LADDER_SHA": "b" * 64,
        "LADDER_BYTES": "1",
        "FEED": "/fixture/feed.jsonl",
        "FEED_SHA": "a" * 64,
        "FEED_BYTES": "1",
        "TAIL": "/fixture/tail.json",
        "QRF_TAIL_EXCLUSIONS_SHA": "c" * 64,
        "SSI": "/fixture/ssi.json",
        "SSI_SHA": "d" * 64,
        "SCF": "/fixture/scf.dta",
        "BASE_SHA": "e" * 64,
        "EXPORT_MASS_REF_H5": str(reference),
    }
    for name in (
        "ASEC_2025_SHA",
        "ASEC_2024_SHA",
        "ASEC_2023_SHA",
        "ASEC_EDU_2023_SHA",
        "ASEC_EDU_2024_SHA",
        "ASEC_EDU_2025_SHA",
    ):
        settings[name] = "f" * 64
    settings.update(changes)
    result = _run_functions(
        ["input_rows", "sha_of", "release_inputs_key"],
        settings,
        "release_inputs_key\n",
    )
    assert result.returncode == 0, result.stderr
    key = result.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", key), key
    return key


def test_the_rerun_guard_keys_the_files_the_release_reads(tmp_path: Path) -> None:
    """The rerun guard compares the digests of what the release reads by
    path beside its command: a QRF tail register edited in place with a new
    QRF_TAIL_EXCLUSIONS_SHA, an export reference replaced in place or a
    rebuilt base changes the key; a base-only input does not (review of
    #1676)."""
    same = _inputs_key(tmp_path)
    assert _inputs_key(tmp_path) == same
    assert _inputs_key(tmp_path, QRF_TAIL_EXCLUSIONS_SHA="9" * 64) != same
    assert _inputs_key(tmp_path, BASE_SHA="8" * 64) != same
    assert _inputs_key(tmp_path, SSI_SHA="7" * 64) != same
    # The base's own inputs belong to the base, which --rebuild keeps.
    assert _inputs_key(tmp_path, LADDER_SHA="6" * 64) == same
    (tmp_path / "reference.h5").write_bytes(b"replaced in place")
    assert _inputs_key(tmp_path) != same


def _rebuild(
    tmp_path: Path, *, certified: bool
) -> tuple[subprocess.CompletedProcess, Path]:
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    start = source.index('if [ "$REBUILD" = 1 ]; then')
    end = source.index("\nfi\n", start) + len("\nfi\n")
    run_root = tmp_path / "run-root"
    run = run_root / "run-abc"
    for stage in ("release", "release-dry-run", "probe-sample", "probe", "base"):
        (run / f"{stage}-sup").mkdir(parents=True)
        (run / f"{stage}-sup" / "ACCEPTED").write_text("returncode=0\n")
    (run / "probe" / "report").mkdir(parents=True)
    (run / "release-dry-run.key").write_text("k\n")
    (run_root / "ROUTE_A_BUILT.json").write_text("{}")
    if certified:
        (run / "certify-sup").mkdir()
        (run / "certify-sup" / "ACCEPTED").write_text("returncode=0\n")
    settings = {
        "REBUILD": "1",
        "RUN": str(run),
        "RUN_ROOT": str(run_root),
        "C": "c" * 40,
        "LOG": str(run_root / "route_a.log"),
    }
    helpers = source[source.index("log() {") : source.index("free_bytes() {")]
    script = "set -u\n" + "".join(
        f"{key}={shlex.quote(value)}\n" for key, value in settings.items()
    )
    script += helpers + "wait_prior_supervisor() { :; }\n" + source[start:end]
    script += "echo REBUILT\n"
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    return result, run


def test_rebuild_sets_the_accepted_build_aside_and_keeps_it(tmp_path: Path) -> None:
    """--rebuild moves the accepted release and its probe aside, so the run
    builds the release again on the accepted base under a new build ID;
    nothing is deleted, and the dry run's settings key decides whether it
    runs again (review of #1681)."""
    result, run = _rebuild(tmp_path / "a", certified=False)
    assert result.returncode == 0, result.stderr
    assert "REBUILT" in result.stdout
    for stage in ("release", "probe-sample", "probe"):
        assert not (run / f"{stage}-sup").exists()
        [aside] = run.glob(f"{stage}-sup.set-aside-*")
        assert (aside / "ACCEPTED").is_file()
    [probe] = run.glob("probe.set-aside-*")
    assert (probe / "report").is_dir()
    assert (run / "base-sup" / "ACCEPTED").is_file()
    assert (run / "release-dry-run-sup" / "ACCEPTED").is_file()
    assert (run / "release-dry-run.key").is_file()
    assert (
        "set aside release-sup probe-sample-sup probe-sup probe ROUTE_A_BUILT.json"
        in result.stdout
    )
    assert not (run.parent / "ROUTE_A_BUILT.json").exists()
    [hand_off] = run.glob("ROUTE_A_BUILT.json.set-aside-*")
    assert hand_off.read_text() == "{}"


def test_rebuild_never_sets_a_certified_build_aside(tmp_path: Path) -> None:
    result, run = _rebuild(tmp_path / "b", certified=True)
    assert result.returncode == 1
    assert "is certified" in result.stdout
    assert (run / "release-sup" / "ACCEPTED").is_file()
    assert not list(run.glob("*.set-aside-*"))


def _probe_record(tmp_path: Path) -> Path:
    rel_out = tmp_path / "release-out" / "populace-us-2024-x"
    rel_out.mkdir(parents=True)
    (rel_out / "build_record.json").write_text(
        json.dumps(
            {
                "layout": {
                    "build_dir": "build",
                    "artifact_root": "artifacts",
                    "dataset_filename": "microcosm_us_2024.h5",
                },
                "manifest": {
                    "target_surface_selection": {
                        "mode": "national_state",
                        "dropped_congressional_district_targets": 436,
                    }
                },
            }
        )
    )
    return rel_out


@pytest.mark.parametrize("failed", ["0", "1"])
def test_the_probe_reads_its_export_and_reference_from_the_build_record(
    tmp_path: Path, failed: str
) -> None:
    """A build that passed its gates is the probe's reference; a failed build
    is none, so its own fit and target surface are passed instead, as the
    probe requires (review of #1681)."""
    rel_out = _probe_record(tmp_path)
    run = tmp_path / "run dir"
    result = _run_functions(
        ["probe_record_arguments", "probe_commands"],
        {
            "PY": sys.executable,
            "REL_OUT": str(rel_out),
            "RUN": str(run),
            "PROBE_STAGES": "take_up_participation,demographics",
        },
        "PROBE_LAUNCHER=(/usr/bin/env -u HF_TOKEN)\n"
        f"probe_commands {failed}\n"
        'printf "%s\\0" "${PROBE_SAMPLE_ARGV[@]}"; printf "\\n"\n'
        'printf "%s\\0" "${PROBE_ARGV[@]}"\n',
    )
    assert result.returncode == 0, result.stderr
    sample_line, probe_line = result.stdout.split("\n", 1)
    sample = sample_line.rstrip("\0").split("\0")
    probe = probe_line.rstrip("\0").split("\0")
    export = str(rel_out / "artifacts" / "microcosm_us_2024.h5")
    assert sample == [
        "/usr/bin/env",
        "-u",
        "HF_TOKEN",
        sys.executable,
        "-B",
        "tools/sample_us_export_households.py",
        "--export",
        export,
        "--fraction",
        "0.05",
        "--seed",
        "0",
        "--out",
        f"{run}/probe/subsample",
    ]
    assert probe[:9] == [
        "/usr/bin/env",
        "-u",
        "HF_TOKEN",
        sys.executable,
        "-B",
        "tools/probe_us_post_export.py",
        "--export",
        f"{run}/probe/subsample/microcosm_us_2024.h5",
        "--out",
    ]
    assert probe[9] == f"{run}/probe/report"
    assert probe[-2:] == ["--stages", "take_up_participation,demographics"]
    reference = probe[10:-2]
    if failed == "0":
        assert reference == ["--reference-release-dir", str(rel_out / "build")]
    else:
        assert reference == [
            "--calibration-diagnostics",
            str(rel_out / "build" / "calibration_diagnostics.json"),
            "--target-surface",
            "national_state",
            "--dropped-congressional-district-targets",
            "436",
        ]


@pytest.mark.parametrize(
    ("register", "reference"),
    [
        (None, True),
        ("this build", False),
        ("no record", False),
        ("another build", True),
        ("unreadable", False),
    ],
)
def test_a_build_whose_certification_failed_is_no_reference_to_the_probe(
    tmp_path: Path, register: str | None, reference: bool
) -> None:
    """A build that passed its gates and then failed certification is no
    reference (certification_failures.json), so the probe gets its fit and
    target surface instead; only attempts on this build's record count
    (review of #1676)."""
    rel_out = _probe_record(tmp_path)
    record_path = rel_out / "build_record.json"
    record = json.loads(record_path.read_text())
    record["record_sha256"] = "r" * 64
    record_path.write_text(json.dumps(record))
    attempt = {"failures": ["Reform coverage smoke failed: x"]}
    contents = {
        "this build": json.dumps(
            {"attempts": [{**attempt, "record_sha256": "r" * 64}]}
        ),
        "no record": json.dumps({"attempts": [attempt]}),
        "another build": json.dumps(
            {"attempts": [{**attempt, "record_sha256": "o" * 64}]}
        ),
        "unreadable": "{",
    }
    if register is not None:
        (rel_out / "certification_failures.json").write_text(contents[register])
    result = _run_functions(
        ["probe_record_arguments"],
        {"PY": sys.executable, "REL_OUT": str(rel_out)},
        "probe_record_arguments 0\n",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert ("--reference-release-dir" in lines) is reference
    assert ("--calibration-diagnostics" in lines) is not reference


def test_the_probe_runs_only_on_an_accepted_build_and_stops_after_it() -> None:
    """--probe needs an accepted build, hashes no input, runs its two stages
    after the build is identified and before any certification, and stops;
    its stages leave the QRF tail stage out."""
    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    needs = source.index(
        '  [ -f "$RUN_ROOT/run-${C:0:12}/release-sup/ACCEPTED" ] \\\n'
        '    || fail "--probe: no accepted build'
    )
    hashed = source.index('if [ "$CERTIFY_ONLY" = 1 ]; then\n  log "inputs: not hashed')
    identified = source.index('[ -f "$REL_OUT/build_record.json" ] || fail')
    probe = source.index('if [ "$MODE" = probe ]; then\n  probe_commands')
    certify = source.index("run_stage certify 0")
    assert needs < hashed < identified < probe < certify
    block = source[probe : source.index("\nfi\n", probe)]
    assert block.index("run_stage probe-sample 0") < block.index("run_stage probe 0")
    assert "exit 0" in block
    stages = re.search(r"^PROBE_STAGES=(\S+)$", source, re.M)
    assert stages is not None
    assert "qrf_tail_concentration" not in stages.group(1).split(",")
    assert {"reform_coverage_smoke", "reform_validation", "demographics"} <= set(
        stages.group(1).split(",")
    )


def test_check_commit_refuses_a_commit_that_cannot_certify_at_any_tier(
    tmp_path: Path,
) -> None:
    """Every build is certified at its commit later, so check_commit checks
    the certification tool and parser whatever the tier, and with no flag to
    check (review of #1681), and the dry-run and probe tools when they run."""

    # run_driver_settings runs a copy of the driver from tmp_path, its
    # SCRIPT_DIR, so the flag checker the checks call goes beside it.
    shutil.copy(ROUTE_A_TOOLS / "check_flags.py", tmp_path / "check_flags.py")
    cases = itertools.count()

    def lines(files: dict[str, str], prefix: str = "", **overrides: str) -> list[str]:
        main, sha = _tiny_main(tmp_path / f"case-{next(cases)}", files)
        result = run_driver_settings(
            tmp_path,
            {"MAIN": str(main), **overrides},
            _report_lines(f"\n{prefix}check_commit {sha} {sha}\n"),
        )
        assert result.returncode == 0, result.stderr.decode()
        return result.stdout.decode().splitlines()

    release = "def _certify_parser():\n    pass\n"
    complete = {
        "tools/certify_us_national_build.py": "",
        "tools/build_us_fiscal_refresh_release.py": release,
        "tools/dry_run_us_release_gates.py": "",
    }
    ok = lines(complete)
    assert not [line for line in ok if "certif" in line and line.startswith("BLOCKER")]
    assert any(
        "certification parser at" in line for line in ok if line.startswith("NOTE")
    )
    no_tool = lines(
        {
            key: value
            for key, value in complete.items()
            if key != "tools/certify_us_national_build.py"
        }
    )
    assert any(
        line.startswith("BLOCKER") and "certify_us_national_build.py is absent" in line
        for line in no_tool
    )
    no_parser = lines(
        {**complete, "tools/build_us_fiscal_refresh_release.py": "x = 1\n"}
    )
    assert any(
        line.startswith("BLOCKER") and "certification parser at" in line
        for line in no_parser
    )
    without_dry_run = {
        key: value
        for key, value in complete.items()
        if key != "tools/dry_run_us_release_gates.py"
    }
    assert any(
        line.startswith("BLOCKER") and "ROUTE_A_DRY_RUN=0" in line
        for line in lines(without_dry_run)
    )
    assert not any(
        "dry_run_us_release_gates.py" in line
        for line in lines(without_dry_run, ROUTE_A_DRY_RUN="0")
    )
    probe_blockers = [
        line
        for line in lines(complete, prefix="MODE=probe\n")
        if line.startswith("BLOCKER") and "--probe cannot run" in line
    ]
    assert len(probe_blockers) == 2
    assert not [line for line in ok if "--probe cannot run" in line]


def test_the_flag_checker_refuses_a_source_without_the_named_parser() -> None:
    checker = ROUTE_A_TOOLS / "check_flags.py"

    def run(source: str, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(checker), *flags],
            input=source,
            capture_output=True,
            text=True,
            timeout=20,
        )

    missing = run("x = 1\n", "--in-function", "_certify_parser")
    assert missing.returncode == 1
    assert "defines no function _certify_parser" in missing.stderr
    present = run(
        "def _certify_parser():\n    pass\n", "--in-function", "_certify_parser"
    )
    assert present.returncode == 0, present.stderr
    assert run("x = 1\n").returncode == 0
