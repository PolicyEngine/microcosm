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
import importlib.util
import inspect
import itertools
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

    def scoped(raw, name, head):
        filename = f"git:{head}:{SOURCE}" if head == BASE else str(w / SOURCE)
        tree = ast.parse(raw, filename=filename)
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
        module.__file__ = filename
        sys.modules[name] = module
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"),
            module.__dict__,
        )
        return module

    current_raw = (w / SOURCE).read_bytes()
    assert current_raw == git("show", f"{args.expect_head}:{SOURCE}")
    old_raw = git("show", f"{BASE}:{SOURCE}")
    current = scoped(current_raw, "_acs_performance_scope", args.expect_head)
    old = scoped(old_raw, "_acs_performance_baseline", BASE)
    identity_path = w / SOURCE.replace("acs_pums.py", "graph_implementation.py")
    identity_spec = importlib.util.spec_from_file_location(
        "_acs_implementation_inventory", identity_path
    )
    identity = importlib.util.module_from_spec(identity_spec)
    sys.modules[identity_spec.name] = identity
    identity_spec.loader.exec_module(identity)
    inventory_path = identity_path.with_name("graph_implementation_inventory.json")
    inventory_raw = inventory_path.read_bytes()
    assert inventory_raw == git("show", f"{BASE}:{inventory_path.relative_to(w)}")
    inventory = json.loads(inventory_raw)
    inventory_name = "microcosm.build/us_runtime/acs_pums.py"
    covered = identity._covered_imports(inventory_name, inventory)
    contract = identity._dependency_contract(current_raw, inventory_name, covered)
    assert contract == inventory["contracts"][inventory_name]
    affected_scopes = sorted(
        name
        for name, spec in inventory["stages"].items()
        if inventory_name in spec["modules"]
    )
    import pytest

    # Directly execute the named source-owned pytest bodies/parametrizations.
    # Pytest's path discovery otherwise probes unrelated ancestor directories.
    spec = importlib.util.spec_from_file_location("_acs_performance_tests", w / TEST)
    tests = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tests
    spec.loader.exec_module(tests)
    checks = []
    for name, test in vars(tests).items():
        if not name.startswith("test_") or not callable(test):
            continue
        groups = []
        for mark in getattr(test, "pytestmark", []):
            assert mark.name == "parametrize"
            keys, values = mark.args
            keys = (
                tuple(k.strip() for k in keys.split(","))
                if isinstance(keys, str)
                else keys
            )
            groups.append(
                [
                    dict(zip(keys, (v,) if len(keys) == 1 else v, strict=True))
                    for v in values
                ]
            )
        for variant, combination in enumerate(itertools.product(*groups)):
            params = {k: v for group in combination for k, v in group.items()}
            tmp = output / f"case-{len(checks)}"
            tmp.mkdir()
            with pytest.MonkeyPatch.context() as monkeypatch:
                available = {
                    "acs": current,
                    "tmp_path": tmp,
                    "monkeypatch": monkeypatch,
                    **params,
                }
                try:
                    test(
                        **{k: available[k] for k in inspect.signature(test).parameters}
                    )
                    checks.append({"name": name, "variant": variant, "passed": True})
                except Exception as error:
                    checks.append(
                        {
                            "name": name,
                            "variant": variant,
                            "passed": False,
                            "error": type(error).__name__,
                            "message": str(error),
                        }
                    )
    code = int(any(not c["passed"] for c in checks))
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

    # Check actual old/new loader results and failures against the same ZIP bytes.
    loader_parity = []
    mutations = {
        "none": None,
        "orphan": lambda h, p: (
            h,
            p.assign(SERIALNO=["orphan"] + p.SERIALNO.tolist()[1:]),
        ),
        "duplicate_household": lambda h, p: (
            pd.concat([h, h.iloc[:1]], ignore_index=True),
            p,
        ),
        "duplicate_person": lambda h, p: (
            h,
            pd.concat([p, p.iloc[:1]], ignore_index=True),
        ),
        "count_mismatch": lambda h, p: (h.assign(NP=[4, 2, 1, 1, 1, 0]), p),
        "gq_weight": lambda h, p: (h.assign(WGTP=[10, 20, 30, 1, 0, 40]), p),
        "gq_count": lambda h, p: (h.assign(NP=[3, 2, 1, 2, 1, 0]), p),
        "missing_state": lambda h, p: (h.drop(columns="ST"), p),
        "missing_column": lambda h, p: (h, p.drop(columns="SEX")),
        "empty_people": lambda h, p: (h, p.iloc[:0]),
    }
    for label, mutation in mutations.items():
        h, p = tests.survey_archives(
            current, output / f"loader-{label}", mutation=mutation
        )
        for limit in (None, 2):
            for chunksize in (1, 3, 100):
                outcomes = []
                for module in (old, current):
                    try:
                        outcomes.append(
                            module.load_acs_pums_tables(
                                module.AcsPumsSource(h, p, max_households=limit),
                                chunksize=chunksize,
                            )
                        )
                    except (ValueError, KeyError) as error:
                        outcomes.append((type(error).__name__, str(error)))
                if isinstance(outcomes[0][0], str):
                    assert outcomes[0] == outcomes[1], (label, limit, chunksize)
                    outcome = "equal_refusal"
                else:
                    assert outcomes[0][1] == outcomes[1][1]
                    for entity in ("household", "person"):
                        pd.testing.assert_frame_equal(
                            outcomes[0][0][entity],
                            outcomes[1][0][entity],
                            check_exact=True,
                        )
                    outcome = "exact_tables_and_metadata"
                loader_parity.append(
                    dict(
                        case=label,
                        max_households=limit,
                        chunksize=chunksize,
                        outcome=outcome,
                    )
                )

    extra_relationship_parity = []
    for label, frame in (
        (
            "reordered_index",
            tests.people([20, 21, 25]).set_axis(pd.Index([2, 0, 1], name="source_row")),
        ),
        (
            "float_index",
            tests.people([20, 21, 25]).set_axis(
                pd.Index([0.0, 1.0, 2.0], name="source_row")
            ),
        ),
        (
            "unsorted_households",
            tests.people([20, 20, 21, 25], households=[9, 3, 9, 3]),
        ),
        ("marital_before_relationship", tests.people([20, 99], marital=[6, 1])),
        ("household_error_order", tests.people([25, 20, 20], households=[9, 3, 3])),
        (
            "preexisting_outputs",
            tests.people([20, 21, 25]).assign(
                A_MARITL=-1, A_SPOUSE=-1, PEPAR1=-1, PEPAR2=-1, A_EXPRRP=-1
            ),
        ),
        *(
            (
                "missing_" + field,
                tests.people([20]).assign(**{field: pd.array([None], dtype="Int64")}),
            )
            for field in ("SPORDER", "AGEP", "MAR", "RELSHIPP", "SEX")
        ),
    ):
        before = frame.copy(deep=True)
        outcomes = []
        for module in (old, current):
            try:
                outcomes.append(module._with_structural_columns(frame))
            except (ValueError, IndexError, KeyError) as error:
                outcomes.append((type(error).__name__, str(error)))
        if isinstance(outcomes[0], tuple):
            assert outcomes[0] == outcomes[1]
            outcome = "equal_refusal"
        else:
            pd.testing.assert_frame_equal(*outcomes, check_exact=True)
            outcome = "exact_frame_equal"
        pd.testing.assert_frame_equal(frame, before, check_exact=True)
        extra_relationship_parity.append(dict(case=label, outcome=outcome))

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

        # Modest multichunk inventory witness, separate from relationship timing.
        serials = [f"HU{i:05d}" for i in range(10000)]
        path = tests.archive(output / "timing-persons.zip", serials * 3)
        valid, retained = frozenset(serials), frozenset(serials[::2])
        row = dict(
            kind="archive_inventory", households=10000, people=30000, chunksize=1000
        )
        outcomes = []
        for label, module in (("old", old), ("new", current)):
            at = time.perf_counter()
            outcomes.append(tests.read(module, path, valid, retained, chunksize=1000))
            row[label + "_wall_seconds"] = time.perf_counter() - at
        assert outcomes[0][1] == outcomes[1][1]
        pd.testing.assert_frame_equal(outcomes[0][0], outcomes[1][0], check_exact=True)
        timings.append(row)

    loaded = []
    runtime_imports = []
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
        elif path and Path(path).is_absolute() and Path(path).is_relative_to(SITE):
            p = Path(path)
            raw = p.read_bytes()
            runtime_imports.append(
                dict(
                    module=name,
                    path=str(p),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw),
                )
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
        "checks": checks,
        "loading_scope": "Exact AST-selected seven loader/relationship bodies, AcsPumsSource, constants and real imports; not whole module/package/Frame/graph execution.",
        "source_sha256": hashlib.sha256(current_raw).hexdigest(),
        "baseline_source_sha256": hashlib.sha256(old_raw).hexdigest(),
        "functions": sorted(FUNCTIONS),
        "inventory": dict(
            sha256=hashlib.sha256(inventory_raw).hexdigest(),
            unchanged_from_baseline=True,
            actual_dependency_contract=contract,
            direct_whole_module_scopes=affected_scopes,
            actual_graph_keys_measured=False,
        ),
        "loaded_source": loaded,
        "ast_executions": [
            dict(
                module=name,
                head=head,
                path=SOURCE,
                sha256=hashlib.sha256(raw).hexdigest(),
                functions=sorted(FUNCTIONS),
            )
            for name, head, raw in (
                ("_acs_performance_scope", args.expect_head, current_raw),
                ("_acs_performance_baseline", BASE, old_raw),
            )
        ],
        "runtime_imports": runtime_imports,
        "parity": parity,
        "loader_parity": loader_parity,
        "extra_relationship_parity": extra_relationship_parity,
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
