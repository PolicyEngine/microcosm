#!/usr/bin/env python3
"""Flip acceptance properties from red to green once they pass.

Runs the ``microcosm-graph`` acceptance suite, collects every strict ``XPASS``
(a property whose implementation has landed), and removes exactly that test's
``xfail`` marker. This is the sanctioned way a property turns green: the
marker leaves in the same change that makes the test pass, and the burndown
ratchet (``tools/graph_acceptance_burndown.py``) records the drop.

Usage::

    uv run python tools/graph_acceptance_flip.py            # flip and report
    uv run python tools/graph_acceptance_flip.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "packages" / "microcosm-graph" / "tests"
MARKER = re.compile(
    r"^@pytest\.mark\.xfail\(strict=True, reason=\"charter [^\"]+\"\)\s*$"
)


def strict_xpasses() -> dict[Path, set[str]]:
    """Test files to the names of their strict-xpass tests.

    A strict xpass is a failure whose message starts with ``[XPASS(strict)]``;
    the JUnit report carries that message per test case, which is sturdier
    than parsing the console summary.
    """
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "junit.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(TESTS),
                "-q",
                "-p",
                "no:warnings",
                "--continue-on-collection-errors",
                f"--junitxml={report}",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if not report.is_file():
            return {}
        tree = ElementTree.parse(report)
    names: set[str] = set()
    for case in tree.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            continue
        message = failure.get("message", "")
        if message.startswith("[XPASS(strict)]"):
            names.add(case.get("name", ""))
    found: dict[Path, set[str]] = {}
    for path in sorted(TESTS.glob("test_acceptance_*.py")):
        defined = set(re.findall(r"^def (test_\w+)\(", path.read_text(), re.M))
        hits = names & defined
        if hits:
            found[path] = hits
    return found


def flip(path: Path, names: set[str], *, dry_run: bool) -> list[str]:
    """Remove the marker in each named test's decorator block."""
    lines = path.read_text().splitlines(keepends=True)
    removed: list[str] = []
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        function_index = index + 1
        while function_index < len(lines) and lines[function_index].startswith("@"):
            function_index += 1
        following = lines[function_index] if function_index < len(lines) else ""
        match = re.match(r"^def (test_\w+)\(", following)
        if MARKER.match(line) and match and match.group(1) in names:
            removed.append(match.group(1))
            index += 1
            continue
        output.append(line)
        index += 1
    if removed and not dry_run:
        path.write_text("".join(output))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    xpasses = strict_xpasses()
    if not xpasses:
        print("no strict xpass: nothing to flip")
        return 0
    total = 0
    for path in sorted(xpasses):
        removed = flip(path, xpasses[path], dry_run=args.dry_run)
        total += len(removed)
        for name in sorted(removed):
            print(
                f"{'would flip' if args.dry_run else 'flipped'} {path.relative_to(ROOT)}::{name}"
            )
        missing = xpasses[path] - set(removed)
        for name in sorted(missing):
            print(
                f"WARNING: {path.relative_to(ROOT)}::{name} xpassed but no marker was found above it"
            )
    print(
        f"{'would flip' if args.dry_run else 'flipped'} {total} propert{'y' if total == 1 else 'ies'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
