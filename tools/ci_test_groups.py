#!/usr/bin/env python3
"""List and validate the directory-defined pytest execution groups."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"

GROUP_DIRECTORIES = {
    "engine-free-shared": ("engine_free", "shared"),
    "engine-free-us": ("engine_free", "us"),
    "engine-free-uk": ("engine_free", "uk"),
    "engine-us": ("engine", "us"),
    "engine-uk": ("engine", "uk"),
    "integration-uk": ("integration", "uk"),
}
VALID_CATEGORIES = frozenset(GROUP_DIRECTORIES.values())


def engine_guard_lines(source: str) -> tuple[int, ...]:
    """Return lines containing country-engine availability checks."""

    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            first = node.args[0]
            if not (
                isinstance(first, ast.Constant)
                and first.value in {"policyengine_us", "policyengine_uk"}
            ):
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in {
                "importorskip",
                "find_spec",
            }:
                lines.add(node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                expression = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if isinstance(expression, ast.Attribute) and expression.attr in {
                    "requires_us",
                    "requires_uk",
                }:
                    lines.add(decorator.lineno)
    return tuple(sorted(lines))


def test_files() -> tuple[Path, ...]:
    """Return every test module below a workspace package's test directory."""

    return tuple(
        sorted(
            path
            for package in PACKAGES.iterdir()
            if package.is_dir()
            for path in (package / "tests").rglob("test_*.py")
            if path.is_file()
        )
    )


def category(path: Path) -> tuple[str, str] | None:
    """Return the directory-defined execution environment and country scope."""

    relative = path.relative_to(path.parents[2])
    parts = relative.parts
    if len(parts) != 3:
        return None
    return parts[0], parts[1]


def selected_files(group: str) -> tuple[Path, ...]:
    """Return the files belonging to ``group``."""

    try:
        selected_category = GROUP_DIRECTORIES[group]
    except KeyError as error:
        raise SystemExit(f"unknown group: {group}") from error
    files = tuple(path for path in test_files() if category(path) == selected_category)
    if not files:
        raise SystemExit(f"group is empty: {group}")
    return files


def verify() -> None:
    """Fail if a test's path does not fully specify how CI executes it."""

    files = test_files()
    if not files:
        raise SystemExit("no test files found below packages/*/tests")

    errors: list[str] = []
    counts = {name: 0 for name in GROUP_DIRECTORIES}
    by_category = {value: key for key, value in GROUP_DIRECTORIES.items()}
    for path in files:
        relative = path.relative_to(ROOT)
        file_category = category(path)
        if file_category not in VALID_CATEGORIES:
            errors.append(f"{relative}: expected tests/<environment>/<scope>/test_*.py")
            continue
        group = by_category[file_category]
        counts[group] += 1
        source = path.read_text(encoding="utf-8")
        guards = engine_guard_lines(source)
        if guards:
            errors.append(
                f"{relative}: engine availability must be expressed by its directory, "
                f"not a pytest skip or import guard (lines {guards})"
            )

    support_tests = sorted((ROOT / "test_support").rglob("test_*.py"))
    errors.extend(
        f"{path.relative_to(ROOT)}: shared support modules must not be test modules"
        for path in support_tests
    )
    errors.extend(
        f"{name}: group is empty" for name, count in counts.items() if count == 0
    )
    if errors:
        raise SystemExit("invalid test layout:\n  " + "\n  ".join(errors))

    print(f"test_files={len(files)}")
    for name, count in counts.items():
        print(f"{name}={count}")
    print("verification=ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", metavar="GROUP")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for path in selected_files(args.list):
            print(path.relative_to(ROOT))
        return 0
    verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
