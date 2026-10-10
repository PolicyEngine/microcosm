"""Exercise Route A orchestration without loading a country engine."""

from __future__ import annotations

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
