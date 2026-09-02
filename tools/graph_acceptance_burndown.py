#!/usr/bin/env python3
"""Count the red properties in the ``microcosm-graph`` acceptance suite.

``docs/graph-acceptance.md`` is the shard's definition of done: every property
there is an executable test committed red under
``pytest.mark.xfail(strict=True)``, and the shard is finished when the suite
carries zero ``xfail`` markers. Process rule 2 is that the count only ever goes
down — "nobody re-reds a property" — so this tool is the ratchet.

Markers are counted from the abstract syntax tree, never by grepping for the
word ``xfail``: a decorator inside a docstring, a commented-out marker, or a
marker on a helper would all fool a text search, and the number this prints is
the number the charter is scored on.

``--verify`` compares against ``origin/main``, file by file, and exits 1
if any file's count rose. A file that does not exist on the baseline is
reported as new and constrains nothing; a file that does is a ratchet. It also
refuses a marker that is not ``strict=True`` (a non-strict marker hides an
``xpass``, so a property could go green without anybody noticing), a marker
whose reason names no charter id, and a charter id with no test at all.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The acceptance files, as a git pathspec. Flat by repo convention.
SUITE_GLOB = "packages/microcosm-graph/tests/test_acceptance_*.py"

#: The charter these markers are scored against.
CHARTER = "docs/graph-acceptance.md"

#: The branch the ratchet compares against, and how to fetch it if absent.
BASELINE_REF = "origin/main"
BASELINE_REFSPEC = "+refs/heads/main:refs/remotes/origin/main"

#: Charter ids this suite owns. Group V (the visuals) belongs to another lane
#: and is not an ``xfail`` in this suite.
CHARTER_ID = re.compile(r"^\|\s*([A-H]\d+)\s*\|")
REASON_ID = re.compile(r"charter\s+([A-H]\d+)\b")
TEST_ID = re.compile(r"^test_([a-h]\d+)_")


@dataclass(frozen=True)
class Marker:
    """One ``xfail`` marker: the test it is on, and what it claims."""

    file: str
    test: str
    charter_id: str
    reason: str
    strict: bool


def dotted(node: ast.expr) -> str:
    """The dotted name of an expression, or "" when it is not a plain name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal(node: ast.expr | None) -> object:
    try:
        return ast.literal_eval(node) if node is not None else None
    except ValueError:
        return None


def markers_in(source: str, file: str = "<memory>") -> tuple[Marker, ...]:
    """Every ``pytest.mark.xfail`` marker on a test function in ``source``."""
    found: list[Marker] = []
    for node in ast.walk(ast.parse(source, filename=file)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            name = dotted(call.func if call else decorator)
            if not name.endswith("mark.xfail") and name != "xfail":
                continue
            keywords = {
                keyword.arg: _literal(keyword.value)
                for keyword in (call.keywords if call else ())
                if keyword.arg
            }
            reason = str(keywords.get("reason") or "")
            match = REASON_ID.search(reason)
            found.append(
                Marker(
                    file=file,
                    test=node.name,
                    charter_id=match.group(1) if match else "",
                    reason=reason,
                    strict=bool(keywords.get("strict")),
                )
            )
    return tuple(found)


def tests_in(source: str, file: str = "<memory>") -> dict[str, str]:
    """Charter id to test name, for every ``test_<id>_...`` in ``source``."""
    named: dict[str, str] = {}
    for node in ast.walk(ast.parse(source, filename=file)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        match = TEST_ID.match(node.name)
        if match:
            named[match.group(1).upper()] = node.name
    return named


def charter_ids(text: str) -> tuple[str, ...]:
    """Every property id the charter's group tables declare, in order."""
    ids = []
    for line in text.splitlines():
        match = CHARTER_ID.match(line.strip())
        if match and match.group(1) not in ids:
            ids.append(match.group(1))
    return tuple(ids)


def suite_files() -> tuple[str, ...]:
    """Tracked acceptance files, plus any not yet added to the index."""
    result = subprocess.run(
        ["git", "ls-files", "--", SUITE_GLOB],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    tracked = {line for line in result.stdout.splitlines() if line}
    on_disk = {
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / SUITE_GLOB).parent.glob(Path(SUITE_GLOB).name))
    }
    return tuple(sorted(tracked | on_disk))


def counts(files: tuple[str, ...]) -> dict[str, tuple[Marker, ...]]:
    """Markers per acceptance file, read from the working tree."""
    return {
        file: markers_in((ROOT / file).read_text(), file)
        for file in files
        if (ROOT / file).is_file()
    }


def baseline_source(ref: str, file: str) -> str | None:
    """``file`` as of ``ref``, or ``None`` when it did not exist there."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{file}"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout if result.returncode == 0 else None


def have_ref(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def fetch_baseline(ref: str) -> bool:
    """Make ``ref`` available, fetching it once if the clone is shallow.

    CI checks out one ref, so the baseline is normally absent and has to be
    fetched. A failure here is a missing network or a deleted branch, not a
    re-redded property, so the caller warns rather than failing the build.
    """
    if have_ref(ref):
        return True
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", BASELINE_REFSPEC],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return have_ref(ref)


def report(current: dict[str, tuple[Marker, ...]]) -> dict[str, object]:
    """The whole burndown as data: by group, by file, and by property."""
    declared = charter_ids((ROOT / CHARTER).read_text())
    markers = [marker for file in sorted(current) for marker in current[file]]
    tests: dict[str, tuple[str, str]] = {}
    for file in sorted(current):
        for identifier, test in tests_in((ROOT / file).read_text(), file).items():
            tests.setdefault(identifier, (test, file))

    red = {marker.charter_id for marker in markers}
    properties = []
    for identifier in declared:
        test, file = tests.get(identifier, ("", ""))
        properties.append(
            {
                "id": identifier,
                "test": test,
                "file": file,
                "state": "red" if identifier in red else "green" if test else "missing",
                "reasons": [m.reason for m in markers if m.charter_id == identifier],
            }
        )

    groups: dict[str, dict[str, object]] = {}
    for marker in markers:
        group = marker.charter_id[:1] or "?"
        entry = groups.setdefault(
            group, {"group": group, "markers": 0, "ids": [], "files": []}
        )
        entry["markers"] = int(entry["markers"]) + 1
        entry["ids"].append(marker.charter_id or "?")
        if marker.file not in entry["files"]:
            entry["files"].append(marker.file)
    for entry in groups.values():
        entry["ids"] = sorted(entry["ids"])

    return {
        "total": len(markers),
        "groups": [groups[key] for key in sorted(groups)],
        "files": [
            {
                "file": file,
                "markers": len(current[file]),
                "ids": sorted(m.charter_id or "?" for m in current[file]),
            }
            for file in sorted(current)
        ],
        "properties": properties,
        "charter_ids": list(declared),
    }


def print_table(data: dict[str, object]) -> None:
    """The table a human reads: by charter group, then by file."""
    print(f"{'group':<7}{'markers':>8}  properties")
    for entry in data["groups"]:
        print(f"{entry['group']:<7}{entry['markers']:>8}  {' '.join(entry['ids'])}")
    print(f"{'total':<7}{data['total']:>8}")
    print()
    print(f"{'markers':>8}  file")
    for entry in data["files"]:
        print(f"{entry['markers']:>8}  {entry['file']}")
    print()
    states: dict[str, list[str]] = {"red": [], "green": [], "missing": []}
    for entry in data["properties"]:
        states[str(entry["state"])].append(str(entry["id"]))
    for state in ("green", "red", "missing"):
        listed = " ".join(states[state]) or "none"
        print(f"{state:<8}{len(states[state]):>4}  {listed}")


def verify(ref: str = BASELINE_REF) -> int:
    """The ratchet, plus the honesty checks the ratchet depends on."""
    files = suite_files()
    current = counts(files)
    data = report(current)
    print_table(data)
    print()

    problems: list[str] = []
    for file in sorted(current):
        for marker in current[file]:
            if not marker.strict:
                problems.append(f"{file}::{marker.test} is not strict")
            if not marker.charter_id:
                problems.append(
                    f"{file}::{marker.test} names no charter id in its reason: "
                    f"{marker.reason!r}"
                )
    for entry in data["properties"]:
        if entry["state"] == "missing":
            problems.append(f"charter {entry['id']} has no test in the suite")

    if not fetch_baseline(ref):
        print(f"baseline={ref} unavailable; the ratchet did not run")
    else:
        print(f"baseline={ref}")
        for file in sorted(current):
            source = baseline_source(ref, file)
            if source is None:
                print(f"  [new]  {file}: {len(current[file])}")
                continue
            was = len(markers_in(source, file))
            now = len(current[file])
            print(f"  {'rose' if now > was else 'ok':<6} {file}: {was} -> {now}")
            if now > was:
                problems.append(
                    f"{file} re-reds {now - was} propert"
                    f"{'y' if now - was == 1 else 'ies'} ({was} -> {now})"
                )

    if problems:
        print()
        for problem in problems:
            print(f"error: {problem}")
        print("verification=failed")
        return 1
    print("verification=ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="fail if any acceptance file carries more markers than the baseline",
    )
    parser.add_argument("--json", action="store_true", help="emit the table as JSON")
    parser.add_argument(
        "--baseline",
        default=BASELINE_REF,
        help=f"the ref the ratchet compares against (default {BASELINE_REF})",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.baseline)
    data = report(counts(suite_files()))
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_table(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
