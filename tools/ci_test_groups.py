#!/usr/bin/env python3
"""List and verify the pytest file groups used by GitHub Actions CI."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_GLOB = "packages/*/tests/test_*.py"

FAST_GROUPS = ("trade", "spine-uk", "rest")
US_GROUPS = ("us-p", "us-qs", "us-not", "us-am")
ENGINE_GROUPS = ("shared-spec", *US_GROUPS, "uk")
GROUPS = (*FAST_GROUPS, *ENGINE_GROUPS, "wheels")
MAX_SHARDS = 64
# Initial file partitions, not measured runtime weights. The workflow matrix
# and its stdlib regression controls must cover every one of these subdivisions.
SHARD_COUNTS = {"rest": 4, "us-am": 4, "wheels": 4}

# Files explicitly reserved for the engine tier by the existing fast-lane policy.
# PyTables is available in both the workspace and the engine-free wheels venv;
# `importorskip("tables")` does not establish country-engine availability.
# Engine-dependent scorer HDF tests carry `requires_us` for the wheels suite.
# They run in full in the engine tier, which installs both extras, so nothing is lost.
# Listed explicitly, and printed by --verify, so the exclusion is never silent.
ENGINE_ONLY = (
    "packages/microcosm-build/tests/test_us_multispine_pool_tool.py",
    "packages/microcosm-build/tests/test_us_release_head_to_head_scorer.py",
)

# Shared-engine behavioral contracts that deliberately do not carry a country
# prefix or the spec-engine ``test_spec_*`` prefix. Listing them makes their
# shared-lane placement reviewed rather than a silent classifier default.
EXPLICIT_SHARED_SPEC = (
    "packages/microcosm-build/tests/test_atomic_geography.py",
    "packages/microcosm-build/tests/test_ci_test_groups.py",
    "packages/microcosm-build/tests/test_cross_grain.py",
)

PROCESSES = {
    "trade": ("main",),
    "spine-uk": ("main",),
    "rest": ("main",),
    "shared-spec": ("main",),
    "us-p": ("main",),
    "us-qs": ("build", "frame"),
    "us-not": ("main",),
    "us-am": ("build", "other-shards"),
    "uk": ("build", "frame"),
    "wheels": ("main",),
}


def stray_nested_test_files(flat: tuple[str, ...]) -> tuple[str, ...]:
    """Tracked ``test_*.py`` files nested below a shard's ``tests/`` directory.

    CI runs explicit file lists built from the flat ``TEST_GLOB``, while local
    ``uv run pytest`` discovers recursively. Reject nested tests so that moving
    one below ``fixtures/`` or ``golden/`` cannot silently remove CI coverage.
    Names are matched in Python rather than by the flat inventory pathspec.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "packages"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    known = set(flat)
    nested = {
        line
        for line in result.stdout.splitlines()
        if "/tests/" in line
        and (ROOT / line).is_file()
        and basename(line).startswith("test_")
        and line.endswith(".py")
        and line not in known
    }
    return tuple(sorted(nested))


def tracked_test_files() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--", TEST_GLOB],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return tuple(
        sorted(
            line
            for line in result.stdout.splitlines()
            if line and (ROOT / line).is_file()
        )
    )


def package(path: str) -> str:
    return path.split("/", 3)[1]


def basename(path: str) -> str:
    return Path(path).name


def is_build(path: str) -> bool:
    return path.startswith("packages/microcosm-build/tests/")


def is_frame(path: str) -> bool:
    return path.startswith("packages/microcosm-frame/tests/")


def is_calibrate_data_fit(path: str) -> bool:
    # The engine-free operator shards. microcosm-graph joins them: its suite
    # never needs a country engine, so it runs in the fast tier and in the
    # engine tier's other-shards process rather than defaulting to shared-spec.
    return package(path) in {
        "microcosm-calibrate",
        "microcosm-data",
        "microcosm-fit",
        "microcosm-graph",
    }


def us_bucket(name: str) -> str | None:
    if name == "test_puf_qrf_chain.py":
        return "us-p"
    if not name.startswith("test_us_"):
        return None
    suffix = name.removeprefix("test_us_")
    first = suffix[:1]
    if first == "p":
        return "us-p"
    if first in {"q", "r", "s"}:
        return "us-qs"
    if first in {"n", "o", "t", "u", "v", "w", "x", "y", "z"}:
        return "us-not"
    if "a" <= first <= "m":
        return "us-am"
    return "shared-spec"


def fast_group(path: str) -> str:
    name = basename(path)
    if path in ENGINE_ONLY:
        return "engine-only"
    if is_build(path) and name.startswith("test_us_trade_"):
        return "trade"
    if is_build(path) and (
        name == "test_us_stacked_spine.py" or name.startswith("test_uk_")
    ):
        return "spine-uk"
    if path == "packages/microcosm-frame/tests/test_policyengine_uk_adapter.py":
        return "spine-uk"
    return "rest"


def engine_group(path: str) -> str:
    name = basename(path)
    if is_build(path) and name.startswith("test_uk_"):
        return "uk"
    if path == "packages/microcosm-frame/tests/test_policyengine_uk_adapter.py":
        return "uk"
    if is_frame(path) and (
        name.startswith("test_policyengine_us_")
        or name == "test_rules_engine_contract.py"
        # The graph kernel wrapping RulesEngine carries a requires_us parity
        # test against PolicyEngine-US, so it belongs with the US adapter tests.
        or name == "test_kernels.py"
    ):
        return "us-qs"
    if is_calibrate_data_fit(path):
        return "us-am"
    if is_build(path):
        bucket = us_bucket(name)
        if bucket is not None:
            return bucket
    return "shared-spec"


def split_group_arg(group_arg: str) -> tuple[str, str | None]:
    if ":" not in group_arg:
        return group_arg, None
    group, proc = group_arg.split(":", 1)
    return group, proc


def process_for(group: str, path: str) -> str:
    if group == "us-qs":
        return "frame" if is_frame(path) else "build"
    if group == "us-am":
        return "other-shards" if is_calibrate_data_fit(path) else "build"
    if group == "uk":
        return "frame" if is_frame(path) else "build"
    return "main"


def partition(
    files: Iterable[str],
    groups: Iterable[str],
    classifier: Callable[[str], str],
) -> dict[str, list[str]]:
    grouped = {group: [] for group in groups}
    for path in files:
        grouped[classifier(path)].append(path)
    for paths in grouped.values():
        paths.sort()
    return grouped


def parse_shard(value: str) -> tuple[int, int]:
    """Parse a 1-based INDEX/COUNT, bounded to a finite number of partitions."""
    if len(value) > 2 * len(str(MAX_SHARDS)) + 1 or not re.fullmatch(
        r"[1-9][0-9]*/[1-9][0-9]*", value
    ):
        raise argparse.ArgumentTypeError("shard must be INDEX/COUNT (1-based)")
    index, count = map(int, value.split("/"))
    if not 1 <= index <= count <= MAX_SHARDS:
        raise argparse.ArgumentTypeError(
            f"shard requires 1 <= INDEX <= COUNT <= {MAX_SHARDS}"
        )
    return index, count


def shard_files(files: Iterable[str], index: int, count: int) -> list[str]:
    """Partition sorted whole test files; never divide module-scoped fixtures."""
    if (
        type(index) is not int
        or type(count) is not int
        or not 1 <= index <= count <= MAX_SHARDS
    ):
        raise ValueError(f"shard requires 1 <= INDEX <= COUNT <= {MAX_SHARDS}")
    ordered = sorted(files)
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate test files in shard inventory")
    if len(ordered) < count:
        raise ValueError(f"cannot fill {count} shards from {len(ordered)} test files")
    return ordered[index - 1 :: count]


def selected_files(group_arg: str, *, shard: tuple[int, int] = (1, 1)) -> list[str]:
    group, proc = split_group_arg(group_arg)
    if group not in GROUPS:
        raise SystemExit(f"unknown group: {group}")
    if proc is not None and proc not in PROCESSES[group]:
        raise SystemExit(f"unknown process for {group}: {proc}")
    classifier = fast_group if group in FAST_GROUPS else engine_group
    files = [
        path
        for path in tracked_test_files()
        if (group == "wheels" or classifier(path) == group)
        and (proc is None or process_for(group, path) == proc)
    ]
    if not files:
        detail = group if proc is None else f"{group}:{proc}"
        raise SystemExit(f"group is empty: {detail}")
    try:
        return shard_files(files, *shard)
    except ValueError as error:
        raise SystemExit(f"{group_arg}: {error}") from error


def assert_file_shards(name: str, files: list[str], count: int) -> None:
    """Prove the nonempty, disjoint union of the actual file selector's shards."""
    pieces = [shard_files(files, index, count) for index in range(1, count + 1)]
    combined = [path for piece in pieces for path in piece]
    if sorted(combined) != sorted(files) or len(combined) != len(set(combined)):
        raise SystemExit(f"{name} shard coverage mismatch")


def assert_partition(
    name: str,
    files: tuple[str, ...],
    groups: tuple[str, ...],
    classifier: Callable[[str], str],
) -> dict[str, list[str]]:
    grouped = partition(files, groups, classifier)
    seen: dict[str, str] = {}
    for group, paths in grouped.items():
        if not paths:
            raise SystemExit(f"{name} group is empty: {group}")
        for path in paths:
            if path in seen:
                raise SystemExit(
                    f"{name} overlap: {path} in both {seen[path]} and {group}"
                )
            seen[path] = group
    missing = sorted(set(files) - set(seen))
    extra = sorted(set(seen) - set(files))
    if missing or extra:
        raise SystemExit(
            f"{name} partition mismatch: missing={missing!r} extra={extra!r}"
        )
    return grouped


def defaulted_engine_files(grouped: dict[str, list[str]]) -> list[str]:
    defaulted = []
    for path in grouped["shared-spec"]:
        name = basename(path)
        if path not in EXPLICIT_SHARED_SPEC and not (
            is_build(path) and name.startswith("test_spec_")
        ):
            defaulted.append(path)
    return defaulted


def verify() -> None:
    files = tracked_test_files()
    if not files:
        raise SystemExit(f"no tracked tests matched {TEST_GLOB}")

    strays = stray_nested_test_files(files)
    if strays:
        listed = "\n  ".join(strays)
        raise SystemExit(
            "test files nested below a shard's tests/ directory are invisible to "
            "every CI lane (they run only under local recursive discovery). Move them "
            f"directly into packages/<shard>/tests/:\n  {listed}"
        )

    for path in ENGINE_ONLY:
        if path not in files:
            raise SystemExit(f"ENGINE_ONLY names a file that no longer exists: {path}")
    # The fast tier partitions everything except the engine-only files; the engine
    # tier still partitions the complete inventory, so no file escapes CI entirely.
    fast_files = tuple(path for path in files if path not in ENGINE_ONLY)
    fast = assert_partition("fast", fast_files, FAST_GROUPS, fast_group)
    engine = assert_partition("engine", files, ENGINE_GROUPS, engine_group)
    wheels = {"wheels": list(files)}

    for tier_name, grouped in (("fast", fast), ("engine", engine), ("wheels", wheels)):
        for group, paths in grouped.items():
            processes = assert_partition(
                f"{tier_name}/{group}",
                tuple(paths),
                PROCESSES[group],
                lambda path, group=group: process_for(group, path),
            )
            for proc, process_files in processes.items():
                assert_file_shards(
                    f"{tier_name}/{group}:{proc}",
                    process_files,
                    SHARD_COUNTS.get(group, 1),
                )

    print(f"tracked_test_files={len(files)}")
    for tier_name, grouped in (("fast", fast), ("engine", engine), ("wheels", wheels)):
        print(f"[{tier_name}]")
        for group, paths in grouped.items():
            procs = ", ".join(PROCESSES[group])
            print(
                f"{group}: files={len(paths)} procs={procs} "
                f"shards_per_process={SHARD_COUNTS.get(group, 1)}"
            )
            for path in paths:
                print(f"  {path}")

    print("[engine-only] excluded from the fast tier, run in the engine tier")
    for path in ENGINE_ONLY:
        print(f"  {path} -> {engine_group(path)}")

    defaulted = defaulted_engine_files(engine)
    print("[defaulted]")
    if defaulted:
        for path in defaulted:
            print(f"  {path} -> shared-spec")
    else:
        print("  none")
    print("verification=ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", metavar="GROUP")
    action.add_argument("--procs", metavar="GROUP")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--shard", type=parse_shard, metavar="INDEX/COUNT")
    args = parser.parse_args(argv)
    if args.shard is not None and not args.list:
        parser.error("--shard requires --list")

    if args.list:
        for path in selected_files(args.list, shard=args.shard or (1, 1)):
            print(path)
        return 0
    if args.procs:
        group = args.procs
        if group not in GROUPS:
            raise SystemExit(f"unknown group: {group}")
        for proc in PROCESSES[group]:
            print(proc)
        return 0
    verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
