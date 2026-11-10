#!/usr/bin/env python3
"""Snapshot one country's Axiom RuleSpec input surface from the real engine.

Why
---
The Axiom adapter (``microcosm.frame.adapters.axiom``) binds dataset columns
to RuleSpec *input slots*: ``AxiomEngine.variables()`` unions
``CompiledDenseProgram.from_file(module, rulespec_roots=[root],
entity=E).root_inputs`` over the mapped engine entities. RuleSpec declares
inputs by usage. A referenced name that is neither a parameter nor a derived
rule in the module's import closure becomes an input. So the surface exists
only after compilation, and it depends on the module and the engine entity.
CI installs no Axiom engine, so engine-free tests read this checked-in
snapshot instead: one JSON file per country, generated from the real engine
at pinned rulespec and engine commits.

What it records
---------------
A module is any ``*.yaml`` other than ``*.test.yaml`` under the country's
top-level trees: ``<cc>`` and every ``<cc>-*`` directory, for example ``be``,
``be-bru``, ``be-dg``, ``be-vlg`` and ``be-wal``. For each module:

* ``inputs``: engine entity -> sorted dense ``root_inputs``. These are the
  names the adapter's ``variables()`` sees for that module when a frame
  entity maps to that engine entity. An entity appears iff it has a dense
  program, i.e. at least one derived rule. ``Scalar`` is the engine's
  pseudo-entity for entity-less formula rules.
* ``canonical_inputs``: slot -> the CLI artifact's
  ``metadata.input_catalog[].canonical_request_name`` for every slot in
  ``inputs`` or ``cli_walk_inputs``, or ``null`` if the catalog lacks it.
* ``status``:
  ``compiled``;
  ``compile_failed``, where the CLI refused the module and ``error`` holds
  the tail of its stderr;
  or ``dense_unsupported``, where the module compiles but the dense compiler
  cannot build at least one entity program. Those entities are listed in
  ``dense_unsupported_entities`` and left out of ``inputs``. Their CLI-walk
  inputs are kept in ``cli_walk_inputs``, but the adapter cannot bind them.
* ``sha256`` of the module file bytes. ``path`` is POSIX and relative to the
  rulespec root.

Surfaces
--------
Every module is compiled with the CLI (``compile --program --rulespec-root``).
The artifact supplies compile status, the input catalog and the derived rules.

With ``--dense-python`` (an interpreter where ``axiom_rules_engine`` and its
``axiom_rules_engine_dense`` native extension, built from ``--engine-commit``,
are importable), ``inputs`` comes from the dense surface itself
(``surface: dense_root_inputs``). The CLI walk below then runs as a
differential check. Its result is printed to stderr, and any disagreement
(inputs, dense support, or the input catalog) aborts the run unless
``--allow-walk-mismatch`` is given.

Without it, ``inputs`` comes from the CLI artifact walk alone
(``surface: cli_artifact_walk``). For an entity E, the walk takes every
derived rule on E and follows the Scalar rules they reference, which is the
closure the dense compiler builds. It collects the ``input`` and
``input_or_else`` references in those rules. It skips the ``value`` and
``where`` of ``sum_related`` and ``count_related``: inputs read there are
related-entity (relation) inputs, not root inputs.

Dense-compile blockers are read from the artifact: a reference from E's
closure to a rule on another non-Scalar entity, or a rule in that closure
with more than one version or an end-bounded version. The engine iterates
derived rules in ``HashMap`` order, so when several blockers exist, which
one its error names varies from process to process. The snapshot therefore
records the lexicographically first blocker, in the engine's own message
format, plus a count of the rest. In dense mode every dense error must equal
one of the computed blockers. Anything else aborts the run: a new kind of
dense limitation is a review event, not a silent snapshot change. In CLI-only
mode, only these two blocker kinds are detected.

Regenerate
----------
Archive the pinned commits into a scratch directory (never build inside a
live checkout), build the CLI and the dense extension from the same engine
archive, then run this tool once per country from the repository root.
These are the commands that produced the checked-in snapshots.
``AXIOM`` is the directory holding the three TheAxiomFoundation checkouts,
after a ``git fetch origin`` in each::

    W=<empty scratch dir>
    AXIOM=<dir with axiom-rules-engine, rulespec-nz, rulespec-be checkouts>
    ENGINE=04315d94d04efd61e27665ab3aa300d64fdf3a66
    NZ=6fe181fc4c65763150fe97501ef3443774e64869
    BE=b105e2b3a3086ddd2de447d58a9b951346870dd1
    mkdir -p $W/engine $W/rulespec-nz $W/rulespec-be
    git -C $AXIOM/axiom-rules-engine archive $ENGINE | tar -x -C $W/engine
    git -C $AXIOM/rulespec-nz archive $NZ | tar -x -C $W/rulespec-nz
    git -C $AXIOM/rulespec-be archive $BE | tar -x -C $W/rulespec-be
    (cd $W/engine && cargo build --release --locked)
    uv venv --python <python3.14> $W/dense-venv
    VIRTUAL_ENV=$W/dense-venv uv pip install maturin
    (cd $W/engine && PYO3_PYTHON=$W/dense-venv/bin/python \\
        $W/dense-venv/bin/maturin build --release \\
        --manifest-path python-ext/Cargo.toml \\
        -i $W/dense-venv/bin/python --out $W/wheels)
    VIRTUAL_ENV=$W/dense-venv uv pip install $W/wheels/*.whl $W/engine/python
    uv run --no-sync python tools/refresh_axiom_input_surface.py \\
        --country nz --rulespec-root $W/rulespec-nz --rulespec-commit $NZ \\
        --rulespec-git-dir $AXIOM/rulespec-nz \\
        --engine-bin $W/engine/target/release/axiom-rules-engine \\
        --engine-commit $ENGINE --dense-python $W/dense-venv/bin/python
    uv run --no-sync python tools/refresh_axiom_input_surface.py \\
        --country be --rulespec-root $W/rulespec-be --rulespec-commit $BE \\
        --rulespec-git-dir $AXIOM/rulespec-be \\
        --engine-bin $W/engine/target/release/axiom-rules-engine \\
        --engine-commit $ENGINE --dense-python $W/dense-venv/bin/python

``<python3.14>`` stands for any GIL-enabled CPython 3.14 interpreter.
The wrapper pins Python ``==3.14.*``, and the extension must be built for the
interpreter that loads it: a wheel built for free-threaded 3.14t does not
import on GIL-enabled 3.14. ``maturin build`` runs without ``--locked``
because ``python-ext/Cargo.lock`` at that engine commit still records the
path dependency ``axiom-rules-engine`` as 0.1.0 (the crate is 0.2.2). Cargo
rewrites only that entry.

``--rulespec-git-dir`` checks that every file under ``--rulespec-root`` is
byte-identical to the blob at ``--rulespec-commit``. ``--check`` regenerates
in memory and exits 1, with a summary of the differences, when the result
differs from the checked-in file. Never hand-edit the snapshots. Rerun this
tool and review the diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIRECTORY = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-frame"
    / "tests"
    / "fixtures"
    / "axiom_input_surfaces"
)
GENERATOR = "tools/refresh_axiom_input_surface.py"
SNAPSHOT_FORMAT = "microcosm.axiom_input_surface.v1"
COUNTRIES = ("nz", "be")
ENGINE_REPOSITORY = "TheAxiomFoundation/axiom-rules-engine"
SURFACE_DENSE = "dense_root_inputs"
SURFACE_CLI = "cli_artifact_walk"
#: The CLI artifact layout the walk below was written against.
ARTIFACT_FORMAT_VERSION = 2
SCALAR_ENTITY = "Scalar"
ERROR_TAIL_CHARACTERS = 300
_FULL_SHA = re.compile(r"[0-9a-f]{40}")

#: Every node ``kind`` in artifact v2 expressions (``ScalarExprSpec``,
#: ``JudgmentExprSpec``, ``ScalarValueSpec`` in the engine's ``src/spec.rs``).
#: The walk refuses an unknown kind rather than guessing its evaluation
#: context.
_KNOWN_KINDS = frozenset(
    {
        # Scalar expressions.
        "literal",
        "input",
        "input_or_else",
        "derived",
        "parameter_lookup",
        "add",
        "sub",
        "mul",
        "div",
        "max",
        "min",
        "ceil",
        "floor",
        "period_start",
        "period_end",
        "date_add_days",
        "date_add_months",
        "date_add_years",
        "days_between",
        "count_related",
        "sum_related",
        "if",
        "no_match",
        "over_periods",
        # Judgment expressions ("derived" is shared with scalars).
        "comparison",
        "relation_member",
        "and",
        "or",
        "not",
        "exactly_one",
        # Literal values.
        "bool",
        "integer",
        "decimal",
        "text",
        "date",
    }
)
_INPUT_KINDS = frozenset({"input", "input_or_else"})
#: Aggregations whose ``value`` and ``where`` evaluate on the related entity.
_RELATED_KINDS = frozenset({"count_related", "sum_related"})

_NO_PROGRAM = "dense compilation could not find derived outputs for entity `{}`"
_CROSS_ENTITY = (
    "dense compilation only supports dependencies within the same root "
    "entity; `{dependency}` from `{derived}` crosses into `{entity}`"
)
_VERSIONED = (
    "dense compilation does not yet support versioned derived formulas in "
    "`{name}`; use generic API execution"
)

#: Runs inside ``--dense-python``. It builds each entity program exactly as
#: the adapter's ``_program`` does. It reports a ValueError as data and lets
#: anything else (a missing native extension, say) fail the run.
_DENSE_PROBE = """
import json
import sys

from axiom_rules_engine import CompiledDenseProgram
from axiom_rules_engine.dense import NativeCompiledDenseProgram

if NativeCompiledDenseProgram is None:
    sys.exit("axiom_rules_engine_dense (the native extension) is not importable")
results = []
for job in json.load(sys.stdin):
    entities = {}
    for entity in job["entities"]:
        try:
            program = CompiledDenseProgram.from_file(
                job["module"], rulespec_roots=[job["root"]], entity=entity
            )
        except ValueError as exc:
            entities[entity] = {"error": str(exc)}
            continue
        entities[entity] = {
            "root_entity": program.root_entity,
            "root_inputs": list(program.root_inputs),
            "input_catalog": dict(program.input_catalog),
        }
    results.append({"module": job["module"], "entities": entities})
json.dump(results, sys.stdout)
"""


class SurfaceError(RuntimeError):
    """The engine output cannot be turned into a trustworthy snapshot."""


# ----------------------------------------------------------------------
# Module discovery and provenance
# ----------------------------------------------------------------------


def country_trees(root: Path, country: str) -> list[Path]:
    """The ``<cc>`` and ``<cc>-*`` top-level directories of a rulespec root."""
    trees = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path.name == country or path.name.startswith(f"{country}-"))
    )
    if not any(tree.name == country for tree in trees):
        raise SurfaceError(f"{root} has no top-level {country!r} tree")
    return trees


def module_paths(root: Path, country: str) -> list[Path]:
    """Every non-test RuleSpec YAML module in the country's trees, sorted."""
    modules = [
        path
        for tree in country_trees(root, country)
        for path in tree.rglob("*.yaml")
        if path.is_file() and not path.name.endswith(".test.yaml")
    ]
    return sorted(modules, key=lambda path: path.relative_to(root).as_posix())


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_rulespec_tree(root: Path, git_dir: Path, commit: str) -> int:
    """Check every file under ``root`` against ``commit``'s tree, byte for byte.

    Returns the number of files checked, and raises if any file is missing,
    extra or different.
    """
    listing = subprocess.run(
        ["git", "-C", str(git_dir), "ls-tree", "-r", "-z", "--full-tree", commit],
        capture_output=True,
        check=True,
    ).stdout
    expected: dict[str, str] = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, _, name = record.partition(b"\t")
        _, kind, sha = meta.decode().split()
        if kind == "blob":
            expected[name.decode()] = sha
    actual: dict[str, str] = {}
    for directory, subdirectories, files in os.walk(root):
        # os.walk lists a symlink to a directory among the subdirectories
        # (without following it); git stores it as a blob like any symlink.
        linked = [name for name in subdirectories if Path(directory, name).is_symlink()]
        for name in [*files, *linked]:
            path = Path(directory) / name
            data = (
                os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            )
            actual[path.relative_to(root).as_posix()] = _git_blob_sha1(data)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    )
    if missing or extra or changed:
        raise SurfaceError(
            f"{root} is not the tree of {commit}: missing {missing[:5]} "
            f"({len(missing)}), extra {extra[:5]} ({len(extra)}), "
            f"changed {changed[:5]} ({len(changed)})"
        )
    return len(expected)


def engine_version(engine_bin: Path) -> str:
    """The CLI's ``--version`` line, after checking its artifact format."""
    version = subprocess.run(
        [str(engine_bin), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    capabilities = json.loads(
        subprocess.run(
            [str(engine_bin), "capabilities"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if capabilities.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise SurfaceError(
            f"{engine_bin} emits artifact format "
            f"{capabilities.get('artifact_format_version')!r}; this walk reads "
            f"format {ARTIFACT_FORMAT_VERSION}"
        )
    return version


# ----------------------------------------------------------------------
# CLI compile and the artifact walk
# ----------------------------------------------------------------------


def _sanitize(text: str, replacements: Mapping[str, str]) -> str:
    for needle, placeholder in replacements.items():
        text = text.replace(needle, placeholder)
    return text


def compile_module(
    engine_bin: Path, module: Path, root: Path, artifact: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """Compile one module with the CLI: (artifact, None) or (None, stderr)."""
    process = subprocess.run(
        [
            str(engine_bin),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(artifact),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return None, (process.stderr.strip() or process.stdout.strip())
    compiled = json.loads(artifact.read_text(encoding="utf-8"))
    if compiled.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise SurfaceError(
            f"{artifact} has artifact format "
            f"{compiled.get('artifact_format_version')!r}"
        )
    return compiled, None


def _collect(node: Any, inputs: set[str], derived: set[str]) -> None:
    """Current-entity input slots and derived references under ``node``."""
    if isinstance(node, list):
        for item in node:
            _collect(item, inputs, derived)
        return
    if not isinstance(node, dict):
        return
    kind = node.get("kind")
    if kind is not None:
        if kind not in _KNOWN_KINDS:
            raise SurfaceError(f"unknown artifact expression kind {kind!r}")
        if kind in _INPUT_KINDS:
            inputs.add(node["name"])
            return
        if kind == "derived":
            derived.add(node["name"])
            return
        if kind in _RELATED_KINDS:
            return
    for key, value in node.items():
        if key != "kind":
            _collect(value, inputs, derived)


def walk_program(
    program: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Per-entity root inputs and dense blockers from a CLI artifact program.

    Returns ``(inputs, blockers)``. Each is keyed by every entity that has a
    derived rule. ``blockers[E]`` is the sorted list of the engine's
    dense-compile messages that entity E's program can raise (empty when it
    compiles).
    """
    rules = {rule["name"]: rule for rule in program["derived"]}
    references: dict[str, tuple[set[str], set[str]]] = {}
    for name, rule in rules.items():
        inputs: set[str] = set()
        derived: set[str] = set()
        _collect(rule["expr"], inputs, derived)
        for version in rule.get("versions") or ():
            _collect(version.get("expr"), inputs, derived)
        references[name] = (inputs, derived)

    entity_inputs: dict[str, set[str]] = {}
    entity_blockers: dict[str, list[str]] = {}
    for entity in sorted({rule["entity"] for rule in rules.values()}):
        pending = [name for name, rule in rules.items() if rule["entity"] == entity]
        closure: set[str] = set()
        inputs: set[str] = set()
        blockers: set[str] = set()
        while pending:
            name = pending.pop()
            if name in closure:
                continue
            closure.add(name)
            rule_inputs, rule_derived = references[name]
            inputs |= rule_inputs
            versions = rules[name].get("versions") or []
            if len(versions) > 1 or any(
                version.get("effective_to") is not None for version in versions
            ):
                blockers.add(_VERSIONED.format(name=name))
            for dependency in rule_derived:
                if dependency not in rules:
                    raise SurfaceError(
                        f"`{name}` references `{dependency}`, which the "
                        "artifact does not define"
                    )
                dependency_entity = rules[dependency]["entity"]
                if dependency_entity in (entity, SCALAR_ENTITY):
                    pending.append(dependency)
                else:
                    blockers.add(
                        _CROSS_ENTITY.format(
                            dependency=dependency,
                            derived=name,
                            entity=dependency_entity,
                        )
                    )
        entity_inputs[entity] = inputs
        entity_blockers[entity] = sorted(blockers)
    return entity_inputs, entity_blockers


def _blocker_error(entity: str, blockers: Sequence[str]) -> str:
    message = f"{entity}: {blockers[0]}"
    if len(blockers) > 1:
        message += f" (+{len(blockers) - 1} more blockers)"
    return message


# ----------------------------------------------------------------------
# Dense surface
# ----------------------------------------------------------------------


def dense_surface(
    dense_python: Path, jobs: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Run the dense probe once for every (module, entity) job."""
    process = subprocess.run(
        [str(dense_python), "-c", _DENSE_PROBE],
        input=json.dumps(list(jobs)),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise SurfaceError(
            f"dense probe failed under {dense_python}:\n{process.stderr[-2000:]}"
        )
    return {item["module"]: item["entities"] for item in json.loads(process.stdout)}


# ----------------------------------------------------------------------
# Snapshot
# ----------------------------------------------------------------------


def build_snapshot(
    *,
    country: str,
    rulespec_root: Path,
    rulespec_commit: str,
    engine_bin: Path,
    engine_commit: str,
    dense_python: Path | None,
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build the snapshot and, in dense mode, the CLI-walk differential."""
    version = engine_version(engine_bin)
    replacements = {
        str(artifact_dir): "<artifact-dir>",
        str(rulespec_root): "<rulespec-root>",
    }
    compiled: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for module in module_paths(rulespec_root, country):
        relative = module.relative_to(rulespec_root).as_posix()
        record: dict[str, Any] = {
            "path": relative,
            "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        }
        artifact_path = artifact_dir / (relative.replace("/", "__") + ".json")
        artifact, error = compile_module(
            engine_bin, module, rulespec_root, artifact_path
        )
        if artifact is None:
            record.update(
                status="compile_failed",
                error=_sanitize(error or "", replacements)[-ERROR_TAIL_CHARACTERS:],
                inputs={},
                canonical_inputs={},
            )
        else:
            compiled[relative] = artifact
        records[relative] = record

    walks = {
        relative: walk_program(artifact["program"])
        for relative, artifact in compiled.items()
    }
    dense: dict[str, dict[str, Any]] | None = None
    if dense_python is not None:
        dense = dense_surface(
            dense_python,
            [
                {
                    "module": str(rulespec_root / relative),
                    "root": str(rulespec_root),
                    "entities": sorted(walks[relative][0]),
                }
                for relative in compiled
            ],
        )

    differential = (
        None
        if dense is None
        else {
            "entity_programs": 0,
            "dense_compiled": 0,
            "input_mismatches": [],
            "status_mismatches": [],
            "catalog_mismatches": [],
        }
    )
    for relative, artifact in compiled.items():
        walk_inputs, walk_blockers = walks[relative]
        catalog = {
            entry["slot"]: entry["canonical_request_name"]
            for entry in artifact["metadata"]["input_catalog"]
        }
        inputs: dict[str, list[str]] = {}
        unsupported: dict[str, str] = {}
        if dense is None:
            for entity, names in walk_inputs.items():
                if walk_blockers[entity]:
                    unsupported[entity] = _blocker_error(entity, walk_blockers[entity])
                else:
                    inputs[entity] = sorted(names)
        else:
            by_entity = dense[str(rulespec_root / relative)]
            for entity in sorted(walk_inputs):
                differential["entity_programs"] += 1
                outcome = by_entity[entity]
                if "error" in outcome:
                    message = outcome["error"]
                    if message == _NO_PROGRAM.format(entity):
                        raise SurfaceError(
                            f"{relative}: dense reports no {entity} program "
                            "although the artifact has derived rules on it"
                        )
                    if message not in walk_blockers[entity]:
                        raise SurfaceError(
                            f"{relative} [{entity}]: dense error not explained "
                            f"by the artifact walk: {message}"
                        )
                    unsupported[entity] = _blocker_error(entity, walk_blockers[entity])
                    continue
                differential["dense_compiled"] += 1
                if outcome["root_entity"] != entity:
                    raise SurfaceError(
                        f"{relative}: asked for {entity}, dense rooted at "
                        f"{outcome['root_entity']}"
                    )
                names = sorted(outcome["root_inputs"])
                if len(set(names)) != len(names):
                    raise SurfaceError(f"{relative} [{entity}]: duplicate root inputs")
                inputs[entity] = names
                if walk_blockers[entity]:
                    differential["status_mismatches"].append(
                        (relative, entity, walk_blockers[entity][0])
                    )
                walked = walk_inputs[entity]
                if set(names) != walked:
                    differential["input_mismatches"].append(
                        (
                            relative,
                            entity,
                            sorted(set(names) - walked),
                            sorted(walked - set(names)),
                        )
                    )
                if outcome["input_catalog"] != catalog:
                    differential["catalog_mismatches"].append((relative, entity))
        record = records[relative]
        cli_walk_inputs = {
            entity: sorted(walk_inputs[entity]) for entity in sorted(unsupported)
        }
        slots = {name for names in inputs.values() for name in names}
        slots.update(name for names in cli_walk_inputs.values() for name in names)
        record.update(
            status="dense_unsupported" if unsupported else "compiled",
            inputs=inputs,
            canonical_inputs={slot: catalog.get(slot) for slot in sorted(slots)},
        )
        if unsupported:
            record.update(
                error="; ".join(unsupported[entity] for entity in sorted(unsupported)),
                dense_unsupported_entities=sorted(unsupported),
                cli_walk_inputs=cli_walk_inputs,
            )

    snapshot = {
        "format": SNAPSHOT_FORMAT,
        "country": country,
        "generator": GENERATOR,
        "rulespec": {
            "repository": f"TheAxiomFoundation/rulespec-{country}",
            "commit": rulespec_commit,
        },
        "engine": {
            "repository": ENGINE_REPOSITORY,
            "commit": engine_commit,
            "version": version,
            "surface": SURFACE_CLI if dense is None else SURFACE_DENSE,
        },
        "modules": [records[relative] for relative in sorted(records)],
    }
    return snapshot, differential


def render(snapshot: Mapping[str, Any]) -> str:
    """The canonical file text: sorted keys, ``indent=1``, trailing newline."""
    return json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def summarize(snapshot: Mapping[str, Any]) -> list[str]:
    """Counts for the run log."""
    modules = snapshot["modules"]
    statuses: dict[str, int] = {}
    for module in modules:
        statuses[module["status"]] = statuses.get(module["status"], 0) + 1
    triples = {
        (module["path"], entity, name)
        for module in modules
        for entity, names in module["inputs"].items()
        for name in names
    }
    null_canonical = sorted(
        {
            slot
            for module in modules
            for slot, canonical in module["canonical_inputs"].items()
            if canonical is None
        }
    )
    return [
        f"{snapshot['country']}: {len(modules)} modules "
        + ", ".join(f"{status} {count}" for status, count in sorted(statuses.items())),
        f"{snapshot['country']}: {len(triples)} (module, entity, input) triples, "
        f"{len({name for _, _, name in triples})} distinct input names",
        f"{snapshot['country']}: slots without a canonical request name: "
        f"{null_canonical or 'none'}",
    ]


def _entity_changes(
    path: str, field: str, old: Mapping[str, Any], new: Mapping[str, Any]
) -> Iterable[str]:
    for entity in sorted(old.keys() | new.keys()):
        if entity not in old:
            yield f"{path}: {field}[{entity}] added ({len(new[entity])} inputs)"
            continue
        if entity not in new:
            yield f"{path}: {field}[{entity}] removed ({len(old[entity])} inputs)"
            continue
        added = sorted(set(new[entity]) - set(old[entity]))
        removed = sorted(set(old[entity]) - set(new[entity]))
        if added:
            yield f"{path}: {field}[{entity}] +{len(added)} {added[:5]}"
        if removed:
            yield f"{path}: {field}[{entity}] -{len(removed)} {removed[:5]}"


def describe_difference(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """A human-readable summary of how two snapshots differ."""
    lines = [
        f"{key}: {old.get(key)!r} -> {new.get(key)!r}"
        for key in ("format", "country", "generator", "rulespec", "engine")
        if old.get(key) != new.get(key)
    ]
    old_modules = {module["path"]: module for module in old.get("modules", [])}
    new_modules = {module["path"]: module for module in new.get("modules", [])}
    lines += [
        f"module removed: {path}"
        for path in sorted(old_modules.keys() - new_modules.keys())
    ]
    lines += [
        f"module added: {path}"
        for path in sorted(new_modules.keys() - old_modules.keys())
    ]
    for path in sorted(old_modules.keys() & new_modules.keys()):
        before, after = old_modules[path], new_modules[path]
        if before == after:
            continue
        for field in ("sha256", "status", "error", "dense_unsupported_entities"):
            if before.get(field) != after.get(field):
                lines.append(
                    f"{path}: {field} {before.get(field)!r} -> {after.get(field)!r}"
                )
        for field in ("inputs", "cli_walk_inputs"):
            lines += _entity_changes(
                path, field, before.get(field, {}), after.get(field, {})
            )
        if before.get("canonical_inputs") != after.get("canonical_inputs"):
            lines.append(f"{path}: canonical_inputs changed")
    return lines or ["contents are equal but the bytes differ (formatting)"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _full_sha(value: str) -> str:
    if not _FULL_SHA.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a full 40-hex commit SHA")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--country", required=True, choices=COUNTRIES)
    parser.add_argument(
        "--rulespec-root",
        required=True,
        type=Path,
        help="an archived rulespec-<cc> tree at --rulespec-commit (git archive)",
    )
    parser.add_argument("--rulespec-commit", required=True, type=_full_sha)
    parser.add_argument(
        "--rulespec-git-dir",
        type=Path,
        help="a rulespec-<cc> git repository; verify --rulespec-root against "
        "--rulespec-commit byte for byte before compiling",
    )
    parser.add_argument(
        "--engine-bin",
        required=True,
        type=Path,
        help="axiom-rules-engine CLI built from --engine-commit",
    )
    parser.add_argument("--engine-commit", required=True, type=_full_sha)
    parser.add_argument(
        "--dense-python",
        type=Path,
        help="interpreter with axiom_rules_engine and axiom_rules_engine_dense "
        "built from --engine-commit; omit to snapshot the CLI artifact walk",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: packages/microcosm-frame/tests/fixtures/"
        "axiom_input_surfaces/<country>.json",
    )
    parser.add_argument(
        "--keep-artifacts",
        type=Path,
        help="write the CLI compile artifacts here instead of a temporary directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory; exit 1 if it differs from --output",
    )
    parser.add_argument(
        "--allow-walk-mismatch",
        action="store_true",
        help="in dense mode, write the dense snapshot even when the CLI artifact "
        "walk disagrees with it (by default any disagreement aborts)",
    )
    args = parser.parse_args(argv)

    output = args.output or SNAPSHOT_DIRECTORY / f"{args.country}.json"

    def run(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return build_snapshot(
            country=args.country,
            rulespec_root=rulespec_root,
            rulespec_commit=args.rulespec_commit,
            engine_bin=engine_bin,
            engine_commit=args.engine_commit,
            dense_python=args.dense_python,
            artifact_dir=artifact_dir,
        )

    try:
        rulespec_root = args.rulespec_root.resolve(strict=True)
        engine_bin = args.engine_bin.resolve(strict=True)
        if args.rulespec_git_dir is not None:
            checked = verify_rulespec_tree(
                rulespec_root, args.rulespec_git_dir, args.rulespec_commit
            )
            print(
                f"verified {checked} files under {rulespec_root} against "
                f"{args.rulespec_commit}",
                file=sys.stderr,
            )
        if args.keep_artifacts is not None:
            args.keep_artifacts.mkdir(parents=True, exist_ok=True)
            snapshot, differential = run(args.keep_artifacts.resolve())
        else:
            with tempfile.TemporaryDirectory() as scratch:
                snapshot, differential = run(Path(scratch).resolve())
    except (OSError, SurfaceError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        print(
            f"error: {exc}" + (f"\n{detail.strip()}" if detail else ""), file=sys.stderr
        )
        return 2

    for line in summarize(snapshot):
        print(line, file=sys.stderr)
    if differential is not None:
        kinds = ("input_mismatches", "status_mismatches", "catalog_mismatches")
        print(
            f"{args.country}: differential (dense vs CLI walk): "
            f"{differential['entity_programs']} entity-programs, "
            f"{differential['dense_compiled']} dense-compiled; "
            + ", ".join(f"{len(differential[kind])} {kind}" for kind in kinds),
            file=sys.stderr,
        )
        for kind in kinds:
            for item in differential[kind][:20]:
                print(f"  {kind}: {item}", file=sys.stderr)
        if not args.allow_walk_mismatch and any(differential[k] for k in kinds):
            print(
                "error: the CLI artifact walk disagrees with the dense surface; "
                "fix the walk or rerun with --allow-walk-mismatch",
                file=sys.stderr,
            )
            return 2

    text = render(snapshot)
    if args.check:
        if not output.exists():
            print(f"{output} does not exist", file=sys.stderr)
            return 1
        current = output.read_text(encoding="utf-8")
        if current == text:
            print(f"{output} is up to date", file=sys.stderr)
            return 0
        print(f"{output} is stale:", file=sys.stderr)
        for line in describe_difference(json.loads(current), snapshot)[:60]:
            print(f"  {line}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
