"""Bounded actual-body ACS checks; never import a model or build a Frame.

The production leaf cannot be imported normally without its country package
initializer. This runner compiles only its exact declared loader/relationship
bodies, constants and stdlib/NumPy/pandas imports. The source-owned tests import
the public module in normal CI; this scoped runner is not end-to-end evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

BASE = "669edf780d4b45e09eb4e38a9e458798b04352d2"
SOURCE = "packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py"
TEST = "packages/microcosm-build/tests/test_us_acs_presampling_performance.py"
SITE = Path(
    "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/lib/python3.14/site-packages"
)
FUNCTIONS = {
    "load_acs_pums_tables",
    "_read_archive",
    "_occupied_households",
    "_smoke_household_selection",
    "_validate_person_counts",
    "_with_structural_columns",
    "_required_integer",
}
PYTHON_SHA = "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--expect-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    assert sys.version_info[:3] == (3, 14, 4)
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    prefix = Path(sys.pycache_prefix)
    assert prefix.is_absolute() and not prefix.exists()
    assert hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == PYTHON_SHA
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.alarm(180)
    stop = threading.Event()

    def watch():
        while not stop.wait(0.02):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 2 * 1024**3:
                os._exit(86)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    start = time.monotonic()
    w, output = args.checkout.resolve(), args.output.absolute()
    assert not output.exists()
    output.mkdir(mode=0o700)

    def git(*parts):
        return subprocess.check_output(["git", "-C", str(w), *parts])

    assert git("rev-parse", "HEAD").decode().strip() == args.expect_head
    assert git("status", "--porcelain") == b""
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path.insert(0, str(SITE))
    denials = []

    def guard_path(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        name = os.fsdecode(path)
        if any(
            s in name
            for s in (
                "/_buildo-runtime/",
                "/.cache/microcosm/",
                "/storage/",
                "/Downloads/",
            )
        ):
            denials.append("GENUINE_PATH")
            raise RuntimeError("GENUINE_PATH")

    stat, lstat = os.stat, os.lstat

    def guarded_stat(path, *a, **kw):
        guard_path(path)
        return stat(path, *a, **kw)

    def guarded_lstat(path, *a, **kw):
        guard_path(path)
        return lstat(path, *a, **kw)

    os.stat, os.lstat = guarded_stat, guarded_lstat

    def audit(event, values):
        if event in ("open", "os.listdir", "os.scandir"):
            guard_path(values[0])
        if event in ("socket.connect", "socket.getaddrinfo") or (
            event == "import"
            and values[0].split(".")[0]
            in {
                "policyengine_us",
                "policyengine_uk",
                "policyengine_core",
                "microunit",
                "torch",
            }
        ):
            denials.append(event)
            raise RuntimeError("MODEL_NETWORK_FORBIDDEN")

    sys.addaudithook(audit)

    def scoped(raw, name):
        tree = ast.parse(raw, filename=str(w / SOURCE))
        selected = []
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                selected.append(node)
            elif isinstance(node, ast.Import) or (
                isinstance(node, ast.ImportFrom)
                and not node.module.startswith("microcosm")
            ):
                selected.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                selected.append(node)
            elif isinstance(node, ast.ClassDef) and node.name == "AcsPumsSource":
                selected.append(node)
        assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == FUNCTIONS
        module = types.ModuleType(name)
        module.__file__ = str(w / SOURCE)
        sys.modules[name] = module
        exec(
            compile(
                ast.Module(body=selected, type_ignores=[]), str(w / SOURCE), "exec"
            ),
            module.__dict__,
        )
        return module

    current_raw = (w / SOURCE).read_bytes()
    assert current_raw == git("show", f"{args.expect_head}:{SOURCE}")
    old_raw = git("show", f"{BASE}:{SOURCE}")
    current = scoped(current_raw, "_acs_performance_scope")
    old = scoped(old_raw, "_acs_performance_baseline")
    import pytest

    code = pytest.main(
        [
            str(w / TEST),
            "-q",
            "--noconftest",
            "-c",
            "/dev/null",
            f"--rootdir={w}",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            f"--basetemp={output / 'pytest'}",
            f"--junitxml={output / 'junit.xml'}",
        ]
    )
    parity = []
    # Compare the same actual bodies independently of the expected-value tests.
    import numpy as np
    import pandas as pd

    for roles in (
        [20],
        *([20, r] for r in range(21, 37)),
        [37],
        [38],
        [37, 38],
        [20, 21, 25],
        [25],
        [20, 20],
        [20, 21, 23],
        [20, 99],
    ):
        n = len(roles)
        frame = pd.DataFrame(
            {
                "household_id": np.ones(n, dtype=np.int64),
                "SPORDER": np.arange(1, n + 1),
                "AGEP": np.full(n, 40),
                "MAR": np.ones(n, dtype=np.int64),
                "RELSHIPP": roles,
                "SEX": np.ones(n, dtype=np.int64),
            }
        )
        results = []
        for module in (old, current):
            try:
                results.append(module._with_structural_columns(frame))
            except ValueError as error:
                results.append((type(error).__name__, str(error)))
        if isinstance(results[0], tuple):
            assert results[0] == results[1]
        else:
            pd.testing.assert_frame_equal(*results, check_exact=True)
        parity.append(
            {
                "roles": roles,
                "outcome": "equal_refusal"
                if isinstance(results[0], tuple)
                else "exact_frame_equal",
            }
        )

    timings = []
    if args.timing and code == 0:
        for households in (1000, 10000):
            n = households * 3
            frame = pd.DataFrame(
                {
                    "household_id": np.repeat(np.arange(households), 3),
                    "SPORDER": np.tile([1, 2, 3], households),
                    "AGEP": np.tile([40, 38, 10], households),
                    "MAR": np.tile([1, 1, 5], households),
                    "RELSHIPP": np.tile([20, 21, 25], households),
                    "SEX": np.tile([1, 2, 1], households),
                }
            )
            outputs = []
            row = {"households": households, "people": n}
            for label, module in (("old", old), ("new", current)):
                at = time.perf_counter()
                outputs.append(module._with_structural_columns(frame))
                row[label + "_wall_seconds"] = time.perf_counter() - at
            pd.testing.assert_frame_equal(*outputs, check_exact=True)
            timings.append(row)

    loaded = []
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and Path(path).is_relative_to(w):
            p = Path(path)
            raw = p.read_bytes()
            rel = p.relative_to(w).as_posix()
            assert raw == git("show", f"{args.expect_head}:{rel}")
            loaded.append(
                {
                    "module": name,
                    "path": str(p),
                    "head": args.expect_head,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            )
    assert (
        (w / SOURCE).read_bytes() == current_raw and not denials and not prefix.exists()
    )
    stop.set()
    thread.join()
    result = {
        "head": args.expect_head,
        "baseline_head": BASE,
        "exit_code": int(code),
        "loading_scope": "Exact AST-selected seven loader/relationship bodies, AcsPumsSource, constants and real imports; not whole module/package/Frame/graph execution.",
        "source_sha256": hashlib.sha256(current_raw).hexdigest(),
        "baseline_source_sha256": hashlib.sha256(old_raw).hexdigest(),
        "functions": sorted(FUNCTIONS),
        "loaded_source": loaded,
        "parity": parity,
        "timings": timings,
        "denials": denials,
        "runtime": {
            "argv": sys.orig_argv,
            "pid": os.getpid(),
            "python": sys.version,
            "python_sha256": PYTHON_SHA,
            "pycache_prefix": sys.pycache_prefix,
            "prefix_absent_before_after": not prefix.exists(),
            "dont_write_bytecode": sys.dont_write_bytecode,
            "cpu_cap_seconds": 120,
            "wall_alarm_seconds": 180,
            "rss_watchdog_bytes": 2 * 1024**3,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_seconds": time.monotonic() - start,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "proof.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "exit_code": int(code),
                "proof": str(output / "proof.json"),
                "timings": timings,
            }
        )
    )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
