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

``--verify`` compares against the baseline branch by property identity and
exits 1 if any property that was green there is red now, whichever file the
marker sits in and whatever else went green (counts can offset; identities
cannot). A property the charter gained since the baseline may start red. It
also refuses a marker that is not ``strict=True`` (a non-strict marker hides
an ``xpass``, so a property could go green without anybody noticing), a marker
whose reason names no charter id, an id the charter does not list, or an id
other than the one in its own test name, two markers on one property, and a
charter id with no test at all.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
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
    """Every ``pytest.mark.xfail`` marker on a module-level test function."""
    found: list[Marker] = []
    for node in ast.parse(source, filename=file).body:
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


#: The only marks an acceptance test may carry: the strict charter marker, the
#: engine guards, and a plain parametrize. Anything else that pytest would
#: honour — skip, skipif, a non-strict or unnamed xfail, marks on a class or a
#: module, marks smuggled through ``pytest.param`` — could hide a failing
#: known-green property while the ratchet reports it green.
ALLOWED_MARKS = frozenset({"xfail", "requires_uk", "requires_us", "parametrize"})
#: Runtime calls that suppress a result from inside a test body.
SUPPRESSING_CALLS = frozenset(
    {
        "pytest.xfail",
        "pytest.skip",
        "pytest.importorskip",
        "xfail",
        "skip",
    }
)
PYTEST_SUPPRESSORS = frozenset({"pytest.xfail", "pytest.skip", "pytest.importorskip"})
SAFE_PYTEST_CALLS = frozenset({"pytest.approx", "pytest.fail", "pytest.raises"})
DYNAMIC_NAMESPACE_REFERENCES = frozenset(
    {
        "globals",
        "locals",
        "vars",
        "exec",
        "eval",
        "getattr",
        "setattr",
        "__import__",
        "builtins.globals",
        "builtins.locals",
        "builtins.vars",
        "builtins.exec",
        "builtins.eval",
        "builtins.getattr",
        "builtins.setattr",
        "builtins.__import__",
        "importlib.import_module",
    }
)


def suppressions_in(source: str, file: str = "<memory>") -> tuple[str, ...]:
    """Every way ``source`` could suppress a result that the marker scan misses.

    Returns human-readable problems; an empty tuple means the file uses only
    the forms the ratchet models: module-level ``test_*`` functions carrying
    :data:`ALLOWED_MARKS`, spelled through ``import pytest`` itself. The scan
    fails closed: an alias for pytest, a ``from pytest import ...``, a
    parametrize over a non-literal, or a ``pytest.param`` carrying marks
    anywhere in the module is refused rather than resolved.
    """
    problems: list[str] = []
    tree = ast.parse(source, filename=file)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def mark_name(node: ast.expr) -> str | None:
        target = node.func if isinstance(node, ast.Call) else node
        name = dotted(target)
        parts = name.split(".")
        if len(parts) == 3 and parts[:2] == ["pytest", "mark"]:
            return parts[-1]
        return None

    def rooted_at(node: ast.expr, name: str) -> bool:
        while isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        return isinstance(node, ast.Name) and node.id == name

    def inside_direct_mark_decorator(node: ast.expr) -> bool:
        child: ast.AST = node
        while (parent := parents.get(child)) is not None:
            if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                return child in parent.decorator_list and mark_name(child) is not None
            child = parent
        return False

    def inside_direct_safe_pytest_call(node: ast.expr) -> bool:
        child: ast.AST = node
        parent = parents.get(child)
        while isinstance(parent, ast.Attribute) and parent.value is child:
            child = parent
            parent = parents.get(child)
        return (
            isinstance(parent, ast.Call)
            and parent.func is child
            and dotted(child) in SAFE_PYTEST_CALLS
        )

    class ModuleBindingScan(ast.NodeVisitor):
        """Find bindings executed in the module namespace, not function bodies."""

        pytestmark = False
        pytest_rebound = False

        def _binding(self, name: str | None) -> None:
            if name == "pytestmark":
                self.pytestmark = True
            elif name == "pytest":
                self.pytest_rebound = True

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store | ast.Del):
                self._binding(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.ctx, ast.Store | ast.Del):
                self._binding(node.attr)
                if rooted_at(node.value, "pytest"):
                    self.pytest_rebound = True
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if (
                isinstance(node.ctx, ast.Store | ast.Del)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                self._binding(node.slice.value)
            if isinstance(node.ctx, ast.Store | ast.Del) and rooted_at(
                node.value, "pytest"
            ):
                self.pytest_rebound = True
            self.generic_visit(node)

        def _visit_definition_expressions(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self._binding(node.name)
            for decorator in node.decorator_list:
                self.visit(decorator)
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
            for type_parameter in getattr(node, "type_params", ()):
                self.visit(type_parameter)
            # Inspect deferred bodies conservatively too. A helper invoked at
            # module scope can install pytestmark through ``global`` and a test
            # body can alias a runtime suppressor. Static control-flow analysis
            # cannot prove those bodies harmless, so bindings there fail closed.
            for statement in node.body:
                self.visit(statement)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_definition_expressions(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_definition_expressions(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            # Defaults are evaluated in the surrounding scope; the body is not.
            self.visit(node.args)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._binding(node.name)
            # A class is independently refused below. Only its expressions that
            # execute in the surrounding scope need binding inspection here.
            for expression in (*node.decorator_list, *node.bases):
                self.visit(expression)
            for keyword in node.keywords:
                self.visit(keyword.value)
            for type_parameter in getattr(node, "type_params", ()):
                self.visit(type_parameter)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            self._binding(node.name)
            self.generic_visit(node)

        def visit_MatchAs(self, node: ast.MatchAs) -> None:
            self._binding(node.name)
            self.generic_visit(node)

        def visit_MatchStar(self, node: ast.MatchStar) -> None:
            self._binding(node.name)

        def visit_alias(self, node: ast.alias) -> None:
            bound = node.asname or node.name.split(".", 1)[0]
            if bound == "pytestmark":
                self.pytestmark = True
            if bound == "pytest" and node.name != "pytest":
                self.pytest_rebound = True

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound == "pytestmark":
                    self.pytestmark = True
                if bound == "pytest":
                    self.pytest_rebound = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest" and alias.asname not in (None, "pytest"):
                    problems.append(
                        f"{file}: pytest is imported under an alias ({alias.asname})"
                    )
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "pytest"
        ):
            problems.append(
                f"{file}: 'from {node.module} import ...' hides marker spellings; "
                "import pytest itself"
            )
        if (
            isinstance(node, ast.Call)
            and dotted(node.func).endswith("param")
            and any(keyword.arg == "marks" for keyword in node.keywords)
        ):
            problems.append(
                f"{file}: pytest.param(..., marks=...) is not allowed anywhere in "
                "an acceptance file"
            )
        if isinstance(node, ast.Attribute) and dotted(node) in PYTEST_SUPPRESSORS:
            problems.append(
                f"{file}: references {dotted(node)}, whose result could be aliased"
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "pytest"
            and not inside_direct_mark_decorator(node)
            and not inside_direct_safe_pytest_call(node)
        ):
            problems.append(
                f"{file}: references pytest outside a direct pytest.mark decorator; "
                "the object could be aliased or mutated"
            )
        dynamic_name = dotted(node)
        if (
            isinstance(node, ast.Name | ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and dynamic_name in DYNAMIC_NAMESPACE_REFERENCES
        ):
            problems.append(
                f"{file}: dynamic module namespace access through "
                f"{dynamic_name} is not allowed"
            )
        if isinstance(node, ast.ClassDef):
            problems.append(
                f"{file}: class {node.name} — tests must be module-level functions"
            )
    bindings = ModuleBindingScan()
    bindings.visit(tree)
    if bindings.pytestmark:
        problems.append(f"{file}: module-level pytestmark is not allowed")
    if bindings.pytest_rebound:
        problems.append(f"{file}: module code rebinds pytest, so marks are untrusted")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                name = mark_name(decorator)
                if name is None:
                    # Pytest and unittest both honor arbitrary decorator
                    # aliases on collected tests. If the spelling is not a
                    # direct ``pytest.mark.<name>`` expression, the static
                    # scanner cannot prove that it is non-suppressing, so the
                    # acceptance ratchet must fail closed.
                    if node.name.startswith("test_"):
                        target = (
                            decorator.func
                            if isinstance(decorator, ast.Call)
                            else decorator
                        )
                        spelling = dotted(target) or type(target).__name__
                        problems.append(
                            f"{file}::{node.name} carries unrecognized decorator "
                            f"{spelling!r}, which the ratchet cannot prove safe"
                        )
                    continue
                if name not in ALLOWED_MARKS:
                    problems.append(
                        f"{file}::{node.name} carries mark {name!r}, which is not allowed"
                    )
                if name == "parametrize" and isinstance(decorator, ast.Call):
                    cases = decorator.args[1] if len(decorator.args) > 1 else None
                    if not isinstance(cases, ast.List | ast.Tuple):
                        problems.append(
                            f"{file}::{node.name} parametrizes over a non-literal; "
                            "cases must be written inline"
                        )
                    for argument in ast.walk(decorator):
                        if (
                            isinstance(argument, ast.keyword)
                            and argument.arg == "marks"
                        ):
                            problems.append(
                                f"{file}::{node.name} smuggles marks through pytest.param"
                            )
                            break
            for inner in ast.walk(node):
                if (
                    inner is not node
                    and isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef)
                    and inner.name.startswith("test_")
                ):
                    problems.append(
                        f"{file}::{node.name} nests {inner.name}, which pytest would "
                        "not collect"
                    )
        if isinstance(node, ast.Call) and dotted(node.func) in SUPPRESSING_CALLS:
            problems.append(
                f"{file}: runtime {dotted(node.func)}() suppresses a result"
            )
    return tuple(problems)


def tests_in(source: str, file: str = "<memory>") -> dict[str, str]:
    """Charter id to test name, for every ``test_<id>_...`` in ``source``."""
    named: dict[str, str] = {}
    for node in ast.parse(source, filename=file).body:
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


def baseline_suite_files(ref: str) -> tuple[str, ...]:
    """The acceptance files as of ``ref``, including ones since deleted."""
    # ``git ls-tree`` takes literal paths, not globs: list the directory and
    # match the file pattern here.
    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            ref,
            "--",
            str(Path(SUITE_GLOB).parent),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return ()
    pattern = Path(SUITE_GLOB).name
    return tuple(
        sorted(
            line
            for line in result.stdout.splitlines()
            if line and fnmatch.fnmatch(Path(line).name, pattern)
        )
    )


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
    for file in sorted(current):
        problems.extend(suppressions_in((ROOT / file).read_text(), file))

    declared = set(charter_ids((ROOT / CHARTER).read_text()))
    seen: dict[str, str] = {}
    for file in sorted(current):
        for marker in current[file]:
            if not marker.charter_id:
                continue
            if marker.charter_id not in declared:
                problems.append(
                    f"{file}::{marker.test} names {marker.charter_id}, which "
                    f"{CHARTER} does not list"
                )
            # The reason is free text; the test name is the binding. A marker
            # whose reason names one property while sitting on another's test
            # would let a re-red hide behind a known red, so the two must agree
            # and each property may carry one marker.
            named = TEST_ID.match(marker.test)
            if named is None or named.group(1).upper() != marker.charter_id:
                problems.append(
                    f"{file}::{marker.test} claims charter {marker.charter_id} "
                    "but is not that property's test"
                )
            if marker.charter_id in seen:
                problems.append(
                    f"{marker.charter_id} carries two markers: {seen[marker.charter_id]} "
                    f"and {file}::{marker.test}"
                )
            seen.setdefault(marker.charter_id, f"{file}::{marker.test}")

    if not fetch_baseline(ref):
        print(f"baseline={ref} unavailable; the ratchet did not run")
    else:
        print(f"baseline={ref}")
        # The ratchet is on property identities, not counts: a property that
        # is green on the baseline and red now is a re-red, whatever else
        # went green, whichever file the marker sits in. A property the
        # charter gained since the baseline is committed red first (the
        # charter's meta-TDD rule), so its marker is not a re-red.
        baseline_charter = baseline_source(ref, CHARTER)
        known = set(charter_ids(baseline_charter)) if baseline_charter else set()
        new_to_charter = declared - known
        # A property the baseline charter listed cannot simply vanish: that
        # would drop it from scoring (and let a green id be renamed into a
        # "new" red one). Retiring a property is a charter change of its own.
        for identifier in sorted(known - declared):
            problems.append(
                f"charter {identifier} was listed on {ref} but is gone from "
                f"{CHARTER}; retiring a property needs its own reviewed change"
            )
        was_red: set[str] = set()
        for file in baseline_suite_files(ref):
            source = baseline_source(ref, file)
            if source is not None:
                was_red |= {m.charter_id for m in markers_in(source, file)}
        for file in sorted(current):
            source = baseline_source(ref, file)
            ids = {m.charter_id for m in current[file] if m.charter_id}
            new_reds = sorted(ids & new_to_charter)
            suffix = (
                f" (+{len(new_reds)} new: {', '.join(new_reds)})" if new_reds else ""
            )
            if source is None:
                print(f"  [new]  {file}: {len(current[file])}{suffix}")
                continue
            was = len(markers_in(source, file))
            now = len(current[file]) - len(new_reds)
            print(
                f"  {'rose' if now > was else 'ok':<6} {file}: {was} -> {now}{suffix}"
            )
        now_red = {m.charter_id for file in current for m in current[file]}
        re_reds = sorted((now_red & declared) - was_red - new_to_charter)
        if re_reds:
            where = {
                identifier: sorted(
                    file
                    for file in current
                    if any(m.charter_id == identifier for m in current[file])
                )
                for identifier in re_reds
            }
            problems.append(
                f"re-reds {len(re_reds)} propert"
                f"{'y' if len(re_reds) == 1 else 'ies'}: "
                + ", ".join(
                    f"{identifier} (green on {ref}; red in "
                    f"{', '.join(where[identifier])})"
                    for identifier in re_reds
                )
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
