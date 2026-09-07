"""Run explicitly selected invented checks under the frozen local runtime.

Invoke with the pinned Python and explicit ``-B -X pycache_prefix=/abs/absent``.
The child gets a different absent prefix, ``-I``, and this checkout's shard
sources. This is a local execution audit, not a data admission mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import runpy
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PINNED_PYTHON = Path(
    "/Users/maxghenis/PolicyEngine/_worktrees/"
    "microcosm-us-launch-current-main-20260905/.venv/bin/python"
)
PINNED_SHA256 = "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6"
ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True
    ).strip()


def _runtime(*, child: bool) -> dict:
    prefix = sys.pycache_prefix
    arguments = sys.orig_argv
    explicit_prefix = any(
        argument == "-X"
        and index + 1 < len(arguments)
        and arguments[index + 1] == f"pycache_prefix={prefix}"
        for index, argument in enumerate(arguments)
    )
    if (
        "-B" not in arguments
        or not sys.dont_write_bytecode
        or not explicit_prefix
        or not prefix
        or not Path(prefix).is_absolute()
        or Path(prefix).exists()
    ):
        raise RuntimeError("Require explicit -B -X pycache_prefix=/absolute/absent")
    if child and ("-I" not in arguments or not sys.flags.isolated):
        raise RuntimeError("The check child must use explicit -I")
    binary_sha = _sha(Path(sys.executable).read_bytes())
    if binary_sha != PINNED_SHA256 or sys.version_info[:3] != (3, 14, 4):
        raise RuntimeError("Runtime does not match the pinned Python 3.14.4 binary")
    if not Path(sys.executable).samefile(PINNED_PYTHON):
        raise RuntimeError("Use the exact prescribed Python executable")
    return {
        "executable": sys.executable,
        "binary_sha256": binary_sha,
        "version": sys.version,
        "orig_argv": arguments,
        "isolated": bool(sys.flags.isolated),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": prefix,
        "pycache_prefix_absent_at_start": True,
    }


def _source_inventory() -> tuple[str, dict[str, dict]]:
    """Read Python source only; never enumerate data/checkpoint/store trees."""
    head = _git("rev-parse", "HEAD")
    entries = _git("ls-tree", "-r", "--full-tree", head).splitlines()
    git_blobs = {}
    for entry in entries:
        metadata, name = entry.split("\t", 1)
        if name.endswith(".py"):
            git_blobs[name] = metadata.split()[2]
    names = _git(
        "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"
    ).splitlines()
    snapshot = {}
    for name in sorted(set(names)):
        if not (
            name == "conftest.py"
            or name.startswith("tools/")
            or (name.startswith("packages/") and "/src/" in name)
            or (name.startswith("packages/") and "/tests/test_" in name)
        ):
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(
            f"blob {len(data)}\0".encode() + data, usedforsecurity=False
        ).hexdigest()
        snapshot[name] = {
            "sha256": _sha(data),
            "git_blob": git_blobs.get(name),
            "materialized_git_blob": blob,
            "matches_git": blob == git_blobs.get(name),
        }
    return head, snapshot


def _loaded_sources(before: dict, after: dict, executed: list[Path]) -> list[dict]:
    paths = {Path(__file__).resolve(), *executed}
    paths.update(
        Path(module.__file__).resolve()
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and str(module.__file__).endswith(".py")
        and Path(module.__file__).is_relative_to(ROOT)
    )
    result = []
    for path in sorted(paths):
        name = str(path.relative_to(ROOT))
        initial, final = before.get(name), after.get(name)
        result.append(
            {
                "path": name,
                "before": initial,
                "after": final,
                "unchanged": initial is not None
                and final is not None
                and initial["sha256"] == final["sha256"],
            }
        )
    return result


def _child(output: Path, command: list[str], seconds: int, interop_threads: int) -> int:
    audit: dict = {"runtime": _runtime(child=True), "command": command}
    cpu_limit = (seconds, seconds + 1)
    resource.setrlimit(resource.RLIMIT_CPU, cpu_limit)
    # macOS treats RSS as advisory; the wall/CPU limits and one-thread settings
    # are enforced. Do not pretend RSS is an enforced allocator limit.
    rss_limits = resource.getrlimit(resource.RLIMIT_RSS)
    audit["resources"] = {
        "cpu_seconds": cpu_limit,
        "inherited_rss_limits_advisory": rss_limits,
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
    }
    sources = sorted((ROOT / "packages").glob("*/src"))
    sys.path[:0] = [str(path) for path in sources] + [str(ROOT)]
    audit["source_roots"] = [str(path) for path in sources]
    audit["head_before"], before = _source_inventory()
    executed: list[Path] = []
    started = time.monotonic()
    result = 1
    try:
        import numpy
        import torch

        from microcosm.graph import platform_fingerprint

        torch.set_num_threads(1)
        torch.set_num_interop_threads(interop_threads)
        audit["numerical_runtime"] = {
            "numpy_version": numpy.__version__,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "platform_fingerprint": platform_fingerprint(),
        }
        if command[0] == "pytest":
            import pytest

            if not any(
                argument.endswith(".py") or ".py::" in argument
                for argument in command[1:]
            ):
                raise ValueError(
                    "Specify explicit invented test files; no full-suite runs"
                )
            result = int(pytest.main(["-p", "no:cacheprovider", *command[1:]]))
        elif command[0] == "script":
            path = (ROOT / command[1]).resolve()
            if not path.is_relative_to(ROOT) or path.suffix != ".py":
                raise ValueError(
                    "Check scripts must be Python source inside this checkout"
                )
            executed.append(path)
            sys.argv = [str(path), *command[2:]]
            runpy.run_path(str(path), run_name="__main__")
            result = 0
        elif command[:2] == ["module", "ruff"]:
            sys.argv = ["ruff", *command[2:]]
            runpy.run_module("ruff", run_name="__main__")
            result = 0
        else:
            raise ValueError(
                "Use pytest <explicit files>, script <source.py>, or module ruff"
            )
    except SystemExit as error:
        result = (
            error.code if isinstance(error.code, int) else int(error.code is not None)
        )
    except BaseException:
        audit["exception"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        audit["elapsed_seconds"] = time.monotonic() - started
        audit["head_after"], after = _source_inventory()
        audit["loaded_sources"] = _loaded_sources(before, after, executed)
        audit["foreign_microcosm_modules"] = {
            name: str(module.__file__)
            for name, module in tuple(sys.modules.items())
            if name.startswith("microcosm.")
            and getattr(module, "__file__", None)
            and not Path(module.__file__).resolve().is_relative_to(ROOT)
        }
        audit["pycache_prefix_absent_at_end"] = not Path(sys.pycache_prefix).exists()
        audit["source_integrity_passed"] = (
            all(item["unchanged"] for item in audit["loaded_sources"])
            and not audit["foreign_microcosm_modules"]
            and audit["pycache_prefix_absent_at_end"]
        )
        if not audit["source_integrity_passed"]:
            result = 1
        audit["returncode"] = result
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="Recorded prepatch regression setting only; intra-op/BLAS stay at one",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.seconds <= 600:
        parser.error(
            "Choose a check command and a CPU/wall bound from 1 to 600 seconds"
        )
    if not 1 <= args.torch_interop_threads <= 64:
        parser.error("Torch inter-op threads must be between 1 and 64")
    output = args.output.resolve()
    if args.child:
        return _child(output, command, args.seconds, args.torch_interop_threads)
    runtime = _runtime(child=False)
    if output.exists():
        parser.error(
            "Audit output already exists; preserve prior evidence with a fresh name"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    child_folder = Path(tempfile.mkdtemp(prefix="origin-cap-check-"))
    child_prefix = child_folder / "absent-pycache"
    child_output = child_folder / "child.json"
    argv = [
        str(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={child_prefix}",
        str(Path(__file__).resolve()),
        "--child",
        "--output",
        str(child_output),
        "--seconds",
        str(args.seconds),
        "--torch-interop-threads",
        str(args.torch_interop_threads),
        "--",
        *command,
    ]
    environment = os.environ.copy()
    environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    environment.update({"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONHASHSEED": "0"})
    environment.pop("PYTHONPATH", None)
    log = output.with_suffix(output.suffix + ".log")
    started = time.monotonic()
    timed_out = False
    with log.open("x") as stream:
        try:
            process = subprocess.run(
                argv,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.seconds + 15,
                check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, 124
    evidence = {
        "parent_runtime": runtime,
        "child_argv": argv,
        "wall_limit_seconds": args.seconds + 15,
        "elapsed_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "log_path": str(log),
        "log_sha256": _sha(log.read_bytes()),
        "parent_pycache_prefix_absent_at_end": not Path(sys.pycache_prefix).exists(),
        "child": json.loads(child_output.read_text())
        if child_output.exists()
        else None,
    }
    if not evidence["parent_pycache_prefix_absent_at_end"]:
        returncode = evidence["returncode"] = 1
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    lines = log.read_text().splitlines()
    if len(lines) > 40:
        print(f"Full output saved in {log}; last 40 lines:")
    print("\n".join(lines[-40:]))
    print(f"Audit: {output}\nExit: {returncode}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
