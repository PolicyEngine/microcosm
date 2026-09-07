"""Bounded invented-only tests and actual executed Git-source proof; no source input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from decimal import localcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python"
)
PYTHON_SHA256 = "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6"
CPU_SECONDS = 120
WALL_SECONDS = 180
RSS_BYTES = 1024**3
TESTS = (
    "test_us_sipp_annual_panel.py",
    "test_us_sipp_annual_records.py",
    "test_us_sipp_annual_housing_source.py",
    "test_us_sipp_housing_projection.py",
    "test_us_sipp_measurements.py",
    "test_us_sipp_december_source.py",
    "test_us_sipp_december_bank_assets.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def runtime():
    prefix = sys.pycache_prefix
    if (
        sys.version_info[:3] != (3, 14, 4)
        or sha(Path(sys.executable).read_bytes()) != PYTHON_SHA256
        or not sys.dont_write_bytecode
        or sys.flags.optimize
        or not prefix
        or not Path(prefix).is_absolute()
        or os.path.lexists(prefix)
        or "-B" not in sys.orig_argv
        or "-X" not in sys.orig_argv
        or "pycache_prefix=" + prefix not in sys.orig_argv
    ):
        raise ValueError("PINNED_RUNTIME_AND_FRESH_EXPLICIT_BYTECODE_FLAGS_REQUIRED")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "executable_sha256": PYTHON_SHA256,
        "isolated": sys.flags.isolated,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": prefix,
        "prefix_absent": True,
        "orig_argv": sys.orig_argv,
    }


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
    path.chmod(0o600)


def frozen_annual_parity(fixture, output_dir):
    """Execute exact frozen Git source against invented panels; compare all diagnostics."""
    commit = "02e7b2933035ae609187d1c715531ca300ed50d7"
    relative = "packages/microcosm-build/src/microcosm/build/sipp_annual_panel.py"
    contract = relative.replace(".py", "_contract.json")
    source = git("show", commit + ":" + relative)
    contract_bytes = git("show", commit + ":" + contract)
    assert sha(source) == fixture["annual"].VERSION_1_MODULE_SHA256
    assert sha(contract_bytes) == fixture["annual"].CONTRACT_SHA256
    directory = output_dir / "frozen-annual-reference"
    directory.mkdir(mode=0o700)
    for name, data in (
        ("sipp_annual_panel.py", source),
        ("sipp_annual_panel_contract.json", contract_bytes),
    ):
        with (directory / name).open("xb") as stream:
            stream.write(data)
    path = directory / "sipp_annual_panel.py"
    name = "microcosm.build._sipp_annual_frozen_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    try:
        spec.loader.exec_module(reference)
        cases = {"mover_child_unavailable": fixture["invented_rows"]()}
        for label, field, value in (
            ("signed", "TPTOTINC", "-5"),
            ("genuine_zero", "TPTOTINC", "0"),
            ("missing", "TPTOTINC", ""),
            ("flag_zero", "APTOTINC", "0"),
            ("flag_unknown", "APTOTINC", ""),
            ("flag_nine", "APTOTINC", "9"),
            ("allocated", "APTOTINC", "2"),
            ("zero_weight", "WPFINWGT", "0"),
            ("unknown_weight", "WPFINWGT", ""),
        ):
            rows = fixture["invented_rows"]()
            for row in rows:
                if row["SSUID"] == fixture["SSUID1"] and row["PNUM"] == "101":
                    row[field] = value
            cases[label] = rows
        compared = []
        for label, rows in cases.items():
            supplied = fixture["supplied_fixture"](rows)
            with localcontext(fixture["measurements"]._ARITHMETIC_CONTEXT):
                old = reference.diagnostics(
                    supplied.panel, supplied.source.declared.data_sha256
                )
                new = fixture["annual"].diagnostics(
                    supplied.panel, supplied.source.declared.data_sha256
                )
            assert old == new, "FROZEN_ANNUAL_DIAGNOSTIC_PARITY"
            compared.append(label)
        return {
            "all_diagnostics_equal": True,
            "cases": compared,
            "reference_commit": commit,
            "reference_git_path": relative,
            "executed_reference_path": str(path),
            "executed_reference_sha256": sha(path.read_bytes()),
            "reference_git_sha256": sha(source),
            "contract_sha256": sha(contract_bytes),
            "amounts_reported_here": False,
        }
    finally:
        del sys.modules[name]


def worker(args):
    flags = runtime()
    if not sys.flags.isolated:
        raise ValueError("ISOLATED_CHILD_REQUIRED")
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    signal.alarm(WALL_SECONDS)
    started = time.monotonic()
    executed = {}

    def record_exec(event, arguments):
        if event != "exec":
            return
        path = Path(arguments[0].co_filename)
        if path.is_absolute() and path.is_relative_to(REPO) and path.suffix == ".py":
            relative = str(path.relative_to(REPO))
            executed.setdefault(relative, sha(path.read_bytes()))

    sys.addaudithook(record_exec)
    # This entry point executed before the audit hook was attached.
    executed[str(Path(__file__).relative_to(REPO))] = sha(Path(__file__).read_bytes())
    for directory in sorted((REPO / "packages").glob("microcosm-*/src")):
        sys.path.insert(0, str(directory))
    sys.path.insert(0, str(REPO))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["SIPP_INVENTED_PROOF_DIR"] = str(args.output_dir)
    os.environ["SIPP_REQUIRE_GIT_BYTES"] = "1" if args.require_git_bytes else "0"
    stop = threading.Event()

    def watch_rss():
        while not stop.wait(0.05):
            used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            used *= 1 if sys.platform == "darwin" else 1024
            if used > RSS_BYTES:
                os._exit(74)

    threading.Thread(target=watch_rss, daemon=True).start()
    import pytest

    code = pytest.main(
        [
            "-q",
            "--tb=no",
            "--no-header",
            "--no-summary",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "--confcutdir=" + str(REPO),
            *[str(REPO / "packages/microcosm-build/tests" / name) for name in TESTS],
            "--junitxml=" + str(args.output_dir / "tests.xml"),
        ]
    )
    # Execute the real creator and decoder once more, writing only invented
    # private bytes plus their count receipt. Test helpers contain no data IO.
    artifact_proof = {}
    annual_parity = {}
    if code == 0:
        import runpy

        fixture = runpy.run_path(
            str(
                REPO
                / "packages/microcosm-build/tests/test_us_sipp_annual_housing_source.py"
            )
        )
        artifact_proof = fixture["write_invented_artifacts"](args.output_dir)
        annual_parity = frozen_annual_parity(fixture, args.output_dir)
    head = git("rev-parse", "HEAD").decode().strip()
    loaded = {}
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if name.startswith("microcosm.") and not path.is_relative_to(REPO):
            raise ValueError("NONLOCAL_MICROCOSM_MODULE")
        if path.is_relative_to(REPO) and path.suffix == ".py":
            loaded[name] = str(path.relative_to(REPO))
    proof = {}
    for relative, digest in sorted(executed.items()):
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", head + ":" + relative], capture_output=True
        )
        committed = sha(blob.stdout) if blob.returncode == 0 else None
        proof[relative] = {
            "executed_source_sha256": digest,
            "git_blob_sha256": committed,
            "current_file_sha256": sha((REPO / relative).read_bytes()),
            "match": digest == committed == sha((REPO / relative).read_bytes()),
        }
    contract = (
        "packages/microcosm-build/src/microcosm/build/sipp_annual_panel_contract.json"
    )
    contract_proof = {
        "sha256": sha((REPO / contract).read_bytes()),
        "git_sha256": sha(git("show", head + ":" + contract)),
    }
    suites = ET.parse(args.output_dir / "tests.xml").getroot()
    counts = {
        key: sum(int(s.get(key, "0")) for s in suites.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    stop.set()
    flags_after = runtime()
    result = {
        "phase": "final_committed" if args.require_git_bytes else "precommit",
        "head": head,
        "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "status": git("status", "--short").decode(),
        "runtime": flags,
        "runtime_rechecked_after_tests": flags_after == flags,
        "tests": counts,
        "pytest_exit": int(code),
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "caps": {
            "cpu_seconds": CPU_SECONDS,
            "wall_seconds": WALL_SECONDS,
            "rss_bytes": RSS_BYTES,
        },
        "loaded_modules": loaded,
        "executed_source_files": proof,
        "compiled_inventory_is_execution_proof": False,
        "all_executed_source_matches_git": all(v["match"] for v in proof.values()),
        "contract": contract_proof,
        "invented_artifacts": artifact_proof,
        "frozen_annual_diagnostic_parity": annual_parity,
        "genuine_source_access": False,
    }
    write(args.output_dir / "verification.json", result)
    if code or (
        args.require_git_bytes
        and (not result["all_executed_source_matches_git"] or result["status"])
    ):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-git-bytes", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        worker(args)
        return
    flags = runtime()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(mode=0o700)
    prefix = args.output_dir / ("absent-child-pycache-" + uuid.uuid4().hex)
    command = [
        str(PYTHON),
        "-I",
        "-B",
        "-X",
        "pycache_prefix=" + str(prefix),
        str(Path(__file__).resolve()),
        "--worker",
        "--output-dir",
        str(args.output_dir),
    ]
    if args.require_git_bytes:
        command.append("--require-git-bytes")
    before = git("status", "--short").decode()
    with (args.output_dir / "worker.log").open("xb") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=WALL_SECONDS + 15,
            cwd=REPO,
        )
    write(
        args.output_dir / "parent.json",
        {
            "runtime": flags,
            "child_command": command,
            "child_exit": completed.returncode,
            "status_before": before,
            "status_after": git("status", "--short").decode(),
            "head": git("rev-parse", "HEAD").decode().strip(),
            "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
            "parent_source_sha256": sha(Path(__file__).read_bytes()),
            "parent_git_source_sha256": sha(
                git("show", "HEAD:tools/verify_sipp_annual_housing_invented.py")
            )
            if args.require_git_bytes
            else None,
            "runtime_rechecked_after_child": runtime() == flags,
        },
    )
    proof_path = args.output_dir / "verification.json"
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        print(
            json.dumps(
                {
                    k: proof[k]
                    for k in (
                        "phase",
                        "head",
                        "tests",
                        "all_executed_source_matches_git",
                        "peak_rss_bytes",
                        "wall_seconds",
                    )
                }
            )
        )
    else:
        print(
            json.dumps(
                {"child_exit": completed.returncode, "verification_written": False}
            )
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
