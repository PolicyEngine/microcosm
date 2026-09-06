"""Fixed-ceiling aggregate-only pilot runner. Unbound declarations refuse before source access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WALL_SECONDS = 900
CPU_SECONDS = 840
RSS_BYTES = 1024**3


def bytecode_flags():
    prefix = sys.pycache_prefix
    if (
        not sys.dont_write_bytecode
        or not prefix
        or not Path(prefix).is_absolute()
        or os.path.lexists(prefix)
        or sys.flags.optimize
        or "-B" not in sys.orig_argv
        or "-X" not in sys.orig_argv
        or "pycache_prefix=" + prefix not in sys.orig_argv
    ):
        raise ValueError("EXPLICIT_B_AND_FRESH_ABSOLUTE_PYCACHE_PREFIX_REQUIRED")
    return {
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": prefix,
        "isolated": sys.flags.isolated,
        "optimize": sys.flags.optimize,
        "xoptions": sys._xoptions,
        "orig_argv": sys.orig_argv,
    }


def binding(path):
    value = json.loads(path.read_text())
    # Only metadata declarations are inspected here, never source paths.
    if (
        value.get("schema") != "sipp-annual-panel-binding/1"
        or value.get("source_kind")
        not in ("invented", "root-approved-source-only-pilot")
        or value.get("survey_year") != 2023
        or value.get("income_year") != 2022
        or not value.get("source_path")
        or not value.get("source_sha256")
        or not value.get("source_size")
    ):
        raise ValueError("UNBOUND_OR_INVALID_DECLARATION")
    return value


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def loaded_code():
    result = {}
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location)
        if name.startswith("microcosm.") and not path.is_relative_to(REPO):
            raise ValueError("NONLOCAL_MICROCOSM_CODE")
        if path.is_relative_to(REPO) and path.suffix in (".py", ".json"):
            result[str(path.relative_to(REPO))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    contract = (
        REPO
        / "packages/microcosm-build/src/microcosm/build/sipp_annual_panel_contract.json"
    )
    result[str(contract.relative_to(REPO))] = hashlib.sha256(
        contract.read_bytes()
    ).hexdigest()
    return result


def check_rss():
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except (OSError, ValueError):
        raise ValueError("RSS_MONITOR_UNAVAILABLE") from None
    if rss <= 0:
        raise ValueError("RSS_MONITOR_UNAVAILABLE")
    if rss > RSS_BYTES:
        raise ValueError("RSS_CEILING")
    return rss


def rss_watchdog(stop):
    while not stop.wait(0.05):
        try:
            check_rss()
        except ValueError as error:
            print("REFUSED:" + str(error), file=sys.stderr, flush=True)
            os._exit(2)


def worker(args):
    flags = bytecode_flags()
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    value = binding(args.binding)
    check_rss()
    stop = threading.Event()
    threading.Thread(target=rss_watchdog, args=(stop,), daemon=True).start()
    for source_root in sorted((REPO / "packages").glob("microcosm-*/src")):
        sys.path.insert(0, str(source_root))
    from microcosm.build import sipp_annual_panel as panel

    if (
        value.get("selection") != panel.SELECTION
        or value.get("contract_sha256") != panel.CONTRACT_SHA256
    ):
        raise ValueError("CONTRACT_OR_SELECTION_MISMATCH")
    started = time.monotonic()
    result = panel.extract(
        Path(value["source_path"]), value["source_sha256"], value["source_size"]
    )
    result["execution"] = {
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": check_rss(),
        "bytecode": flags,
        "python_version": sys.version,
        "loaded_code_sha256": loaded_code(),
        "source_kind": value["source_kind"],
    }
    write_new(args.output_dir / "aggregate-diagnostics.json", result)
    check_rss()
    stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    flags = bytecode_flags()
    binding(args.binding)  # Unbound declarations create no output and open no source.
    args.output_dir.mkdir()  # Exclusive: refuse existing directories, files and links.
    child_prefix = str(
        Path(sys.pycache_prefix).with_name("annual-panel-worker-" + uuid.uuid4().hex)
    )
    command = [
        sys.executable,
        "-B",
        "-X",
        "pycache_prefix=" + child_prefix,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        "--binding",
        str(args.binding.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    started = time.monotonic()
    refused = None
    # Child diagnostics never print rows or source values; suppress unexpected
    # tracebacks too. Failure output contains only an exception type/code.
    with (args.output_dir / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            code = process.wait(timeout=WALL_SECONDS)
        except subprocess.TimeoutExpired:
            refused = "WALL_CEILING"
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    status = "complete" if code == 0 and not refused else "refused"
    write_new(
        args.output_dir / "execution-receipt.json",
        {
            "status": status,
            "refusal": refused,
            "worker_exit": code,
            "wall_seconds": time.monotonic() - started,
            "ceilings": {
                "wall_seconds": WALL_SECONDS,
                "cpu_seconds": CPU_SECONDS,
                "worker_rss_bytes": RSS_BYTES,
                "rss_monitor_interval_seconds": 0.05,
                "rss_monitor": "worker_daemon_watchdog_getrusage_peak_RSS_before_import_and_before_success",
                "rss_is_sampled_not_a_kernel_allocation_limit": True,
            },
            "parent_bytecode": flags,
            "child_command": command,
            "output_contract": "new directory only; aggregate-diagnostics.json on worker success, worker.log, execution-receipt.json; a failed or interrupted directory remains and is never overwritten",
        },
    )
    if status != "complete":
        raise SystemExit(2)
    print("Completed bounded source-only pilot; aggregate diagnostics saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Preserve only known static refusal codes, never arbitrary error strings
        # that could contain a path or field value.
        code = str(error)
        if not code or any(
            c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in code
        ):
            code = type(error).__name__
        print("REFUSED:" + code, file=sys.stderr)
        raise SystemExit(2) from None
