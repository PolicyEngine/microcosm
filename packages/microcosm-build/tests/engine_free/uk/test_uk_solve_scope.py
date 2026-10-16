"""The UK solve never reads a gate-only register (microcosm#1301).

A second local release build signs build 1's failing cells and areas in the
gate-only registers and must reuse build 1's solve. Every node the solve
depends on keys on the *solve fingerprint* (every packaged resource but
``UK_GATE_ONLY_RESOURCES``), so this suite checks the claim that makes that
sound from both sides: the source names no gate-only file outside the gate
machinery, and editing each one leaves every solve kernel's key unchanged
while the calibrated gates re-key.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import types
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from microcosm.build import country_spec
from microcosm.build.uk_runtime import battery_bindings, weighted_integrity
from microcosm.build.uk_runtime.graph_evidence import UKSpineGateKernel
from microcosm.build.uk_runtime.graph_targets import (
    UKFullMeasureKernel,
    UKFullProblemKernel,
    UKFullTargetCompilationKernel,
    UKFullTargetSelectionKernel,
)
from microcosm.build.uk_runtime.graph_terminal import (
    UKFullGateKernel,
    UKFullHoldoutKernel,
    UKFullPreflightGateKernel,
)
from microcosm.build.uk_runtime.solve_scope import (
    UK_GATE_ONLY_RESOURCES,
    uk_solve_fingerprint,
)
from microcosm.build.uk_runtime.target_compile_modules import (
    UK_TARGET_COMPILE_MODULE_NAMES,
)
from test_support.paths import paths_for

_PATHS = paths_for("microcosm-build")
_REPOSITORY = _PATHS.repository
_BUILD_SOURCE = _PATHS.package / "src" / "microcosm" / "build"

#: Names that reach a gate-only register: the loaders and the resource-name
#: constants. A module naming one of these, or a gate-only filename, reads
#: (or can read) a gate-only register.
_GATE_ONLY_READERS = frozenset(
    {
        "uk_default_target_fit_reviewed_exclusions",
        "uk_default_local_target_fit_reviewed_exclusions",
        "uk_default_input_mass_reviewed_exclusions",
        "load_uk_local_area_support_exclusion_register",
        "UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE",
        "UK_LOCAL_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE",
        "UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE",
    }
)

#: Every module allowed to name a gate-only register, with the reason. A new
#: reader fails the scan until it is reviewed here: if it is on the solve's
#: path, the file is not gate-only and must leave ``UK_GATE_ONLY_RESOURCES``.
_REVIEWED_READERS = {
    "packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py": (
        "defines the register loaders and the resource names"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/terminal_gates.py": (
        "the cached loaders of the committed target-fit registers"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/battery_bindings.py": (
        "the terminal gate evaluators, which only the calibrated gates run"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/__init__.py": (
        "re-exports the area-support register loader"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/solve_scope.py": (
        "declares the gate-only set"
    ),
    "packages/microcosm-build/src/microcosm/build/uk_runtime/graph_terminal.py": (
        "names the area-support register as a label in the candidate manifest "
        "projected after the build"
    ),
    "tools/measure_uk_atomic_assignment.py": ("a measurement tool that no build runs"),
    "tools/draft_uk_dense_signoff.py": (
        "the sign-off drafting tool, which writes the registers between builds "
        "(microcosm#1301)"
    ),
}

#: Reviewed readers limited to named functions: graph_terminal also holds the
#: holdout kernel, which is on the solve's path.
_READER_SCOPES = {
    "packages/microcosm-build/src/microcosm/build/uk_runtime/graph_terminal.py": {
        "rowwise_candidate_manifest_from_graph"
    },
}

#: Modules on the solve's path beyond the target-compile modules: the graph
#: composition, the spine and population, the solve and size stages.
_SOLVE_MODULES = (
    "microcosm.build.uk_runtime.graph",
    "microcosm.build.uk_runtime.graph_build",
    "microcosm.build.uk_runtime.graph_calibration",
    "microcosm.build.uk_runtime.graph_evidence",
    "microcosm.build.uk_runtime.graph_kernels",
    "microcosm.build.uk_runtime.graph_population",
    "microcosm.build.uk_runtime.graph_targets",
    "microcosm.build.uk_runtime.full_measure",
    "microcosm.build.uk_runtime.measure_simulation",
    "microcosm.build.uk_runtime.full_problem",
    "microcosm.build.uk_runtime.full_targets",
    "microcosm.build.uk_runtime.support_floor",
    "microcosm.build.uk_runtime.local_doctrine",
    "microcosm.build.uk_runtime.local_rowwise",
    "microcosm.build.uk_runtime.dataset_size",
    "microcosm.build.uk_runtime.national_doctrine",
    "microcosm.build.uk_runtime.target_weights",
    "microcosm.build.uk_runtime.spine_build",
)

#: The offline generators of the committed solve surfaces.
_SOLVE_GENERATORS = (
    "tools/generate_uk_local_target_references.py",
    "tools/generate_uk_target_references.py",
)


def _module_path(name: str) -> str:
    relative = Path(*name.split(".")[2:]).with_suffix(".py")
    return str((_BUILD_SOURCE / relative).relative_to(_REPOSITORY))


def _docstrings(tree: ast.Module) -> set[int]:
    nodes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    found = set()
    for node in nodes:
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return found


def _references(path: Path) -> set[str]:
    """The qualified scopes of ``path`` that name a gate-only register."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstrings(tree)
    scopes: set[str] = set()

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(
                    child,
                    child.name if scope == "<module>" else f"{scope}.{child.name}",
                )
                continue
            if _names_gate_only(child, docstrings):
                scopes.add(scope)
            visit(child, scope)

    visit(tree, "<module>")
    return scopes


def _names_gate_only(node: ast.AST, docstrings: set[int]) -> bool:
    """Whether ``node`` names a gate-only register as code would use it.

    A string is a use when it is a reader's name or a path-like literal
    holding a gate-only filename. Prose that cites a register by name (a
    docstring, or a rationale with spaces in it) reads nothing.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
        if id(node) in docstrings or any(character.isspace() for character in value):
            return False
        return value in _GATE_ONLY_READERS or any(
            name in value for name in UK_GATE_ONLY_RESOURCES
        )
    if isinstance(node, ast.Name):
        return node.id in _GATE_ONLY_READERS
    if isinstance(node, ast.Attribute):
        return node.attr in _GATE_ONLY_READERS
    if isinstance(node, ast.alias):
        return node.name in _GATE_ONLY_READERS or (
            node.asname is not None and node.asname in _GATE_ONLY_READERS
        )
    return False


@pytest.fixture(scope="module")
def readers() -> dict[str, set[str]]:
    paths = [
        *sorted(_BUILD_SOURCE.rglob("*.py")),
        *sorted(_REPOSITORY.glob("tools/*.py")),
    ]
    found = {}
    for path in paths:
        scopes = _references(path)
        if scopes:
            found[str(path.relative_to(_REPOSITORY))] = scopes
    return found


def _perturbed(*resources: str) -> country_spec.ResolvedCountrySpec:
    spec = country_spec.load_country_spec("uk")
    hashes = dict(spec.resource_hashes)
    for resource in resources:
        hashes[resource] = hashlib.sha256(hashes[resource].encode()).hexdigest()
    return replace(spec, resource_hashes=MappingProxyType(hashes))


# ---------------------------------------------------------------------------
# The declared set
# ---------------------------------------------------------------------------


def test_the_gate_only_set_is_the_gate_registers_in_lockstep():
    spec = country_spec.load_country_spec("uk")
    assert UK_GATE_ONLY_RESOURCES <= set(spec.resources)
    area_support = (
        inspect.signature(
            weighted_integrity.load_uk_local_area_support_exclusion_register
        )
        .parameters["resource"]
        .default
    )
    assert UK_GATE_ONLY_RESOURCES == {
        weighted_integrity.UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE,
        weighted_integrity.UK_LOCAL_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE,
        weighted_integrity.UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        area_support,
    }
    named = {}
    for entry in spec.gates.gates:
        for key in ("reviewed_exclusions_resource", "exclusions_resource"):
            resource = (entry.parameters or {}).get(key)
            if resource is not None:
                named.setdefault(resource, set()).add(entry.phase)
    # Each gate-only register is a gate's own register, read at the terminal
    # phase only. The degenerate register is a gate register too, but stays
    # solve-scoped by ruling: the release contract pins its digest.
    assert set(named) - UK_GATE_ONLY_RESOURCES == {
        "degenerate_reviewed_exclusions.json"
    }
    assert UK_GATE_ONLY_RESOURCES <= set(named)
    for resource in UK_GATE_ONLY_RESOURCES:
        assert named[resource] == {"terminal"}, resource


def test_no_spine_stage_pins_a_gate_only_register():
    spec = country_spec.load_country_spec("uk")
    pinned = {
        str(artifact["resource"])
        for stage in spec.sources.stages
        for artifact in stage.artifacts
        if "resource" in artifact
    }
    assert not pinned & UK_GATE_ONLY_RESOURCES


def test_every_gate_reading_a_gate_only_register_is_terminal():
    """Bindings whose evaluator reaches a gate-only loader, however
    indirectly within the bindings module, evaluate at the terminal phase."""

    functions = {
        name: value
        for name, value in vars(battery_bindings).items()
        if inspect.isfunction(value) and value.__module__ == battery_bindings.__name__
    }

    def names(code: types.CodeType) -> set[str]:
        used = set(code.co_names)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                used |= names(constant)
        return used

    used = {name: names(function.__code__) for name, function in functions.items()}
    reading = {name for name, found in used.items() if found & _GATE_ONLY_READERS}
    while True:
        more = {
            name
            for name, found in used.items()
            if name not in reading and found & reading
        }
        if not more:
            break
        reading |= more
    gates = {
        binding.name
        for binding in battery_bindings.UK_GATE_REGISTRY.values()
        for function in (
            getattr(binding, "evaluator", None),
            getattr(binding, "evidence", None),
        )
        if function is not None and getattr(function, "__name__", None) in reading
    }
    # Positive control: the scan finds the three register-reading gates.
    assert {"target_fit", "input_mass_parity", "area_support"} <= gates
    phases = {}
    for entry in country_spec.load_country_spec("uk").gates.gates:
        if entry.gate in gates:
            phases.setdefault(entry.gate, set()).add(entry.phase)
    assert phases and all(found == {"terminal"} for found in phases.values()), phases


# ---------------------------------------------------------------------------
# The solve fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource",
    sorted(country_spec.load_country_spec("uk").resource_hashes),
)
def test_only_a_gate_only_edit_leaves_the_solve_fingerprint(resource):
    spec = country_spec.load_country_spec("uk")
    edited = _perturbed(resource)
    assert edited.fingerprint != spec.fingerprint
    assert (uk_solve_fingerprint(edited) == uk_solve_fingerprint(spec)) is (
        resource in UK_GATE_ONLY_RESOURCES
    )


def test_fingerprint_excluding_refuses_what_it_cannot_honour():
    spec = country_spec.load_country_spec("uk")
    with pytest.raises(ValueError, match="undeclared resource"):
        spec.fingerprint_excluding({"not_a_resource.json"})
    with pytest.raises(ValueError, match="country_package.json"):
        spec.fingerprint_excluding({"country_package.json"})
    with pytest.raises(TypeError, match="collection"):
        spec.fingerprint_excluding("gates.json")
    # The excluded names are bound: another set is another fingerprint, and
    # excluding nothing is not the full fingerprint.
    assert spec.fingerprint_excluding(()) != spec.fingerprint
    assert spec.fingerprint_excluding(UK_GATE_ONLY_RESOURCES) == uk_solve_fingerprint()
    assert (
        spec.fingerprint_excluding(
            UK_GATE_ONLY_RESOURCES | {"degenerate_reviewed_exclusions.json"}
        )
        != uk_solve_fingerprint()
    )


# ---------------------------------------------------------------------------
# Kernel keys
# ---------------------------------------------------------------------------


def _solve_kernels():
    return {
        "target_compilation": UKFullTargetCompilationKernel(),
        "target_selection": UKFullTargetSelectionKernel(),
        "measures": UKFullMeasureKernel(),
        "problem": UKFullProblemKernel(),
        "preflight": UKFullPreflightGateKernel(
            coverage_engine=object(), engine_identity="fixture"
        ),
        "holdout": UKFullHoldoutKernel(),
        "spine_gates": UKSpineGateKernel(
            gates=None, engine=object(), engine_identity="fixture"
        ),
    }


@pytest.fixture
def stand_in_dependency_versions(monkeypatch):
    """The holdout hashes its dependencies' versions; the engine-free
    environment lacks policyengine-uk, so absent ones read as a stand-in."""
    from importlib import metadata

    installed = metadata.version

    def version(distribution):
        try:
            return installed(distribution)
        except metadata.PackageNotFoundError:
            return "0+stand-in"

    monkeypatch.setattr(metadata, "version", version)


def _hashes(monkeypatch, *resources):
    calibrated = UKFullGateKernel(coverage_engine=object(), engine_identity="fixture")
    with monkeypatch.context() as patch:
        if resources:
            edited = _perturbed(*resources)
            patch.setattr(
                country_spec, "_load_packaged_country_spec", lambda _country: edited
            )
        solve = {
            name: kernel.implementation_hash()
            for name, kernel in _solve_kernels().items()
        }
        return solve, calibrated.implementation_hash()


@pytest.mark.parametrize("resource", sorted(UK_GATE_ONLY_RESOURCES))
def test_a_gate_only_edit_rekeys_the_calibrated_gates_and_no_solve_kernel(
    resource, monkeypatch, stand_in_dependency_versions
):
    solve, calibrated = _hashes(monkeypatch)
    edited_solve, edited_calibrated = _hashes(monkeypatch, resource)
    assert edited_solve == solve
    assert edited_calibrated != calibrated


def test_a_solve_scoped_edit_rekeys_every_solve_kernel(
    monkeypatch, stand_in_dependency_versions
):
    solve, calibrated = _hashes(monkeypatch)
    edited_solve, edited_calibrated = _hashes(
        monkeypatch, "calibration_measure_exclusions.json"
    )
    assert {name for name in solve if edited_solve[name] == solve[name]} == set()
    assert edited_calibrated != calibrated


# ---------------------------------------------------------------------------
# The source scan
# ---------------------------------------------------------------------------


def test_only_reviewed_modules_name_a_gate_only_register(readers):
    present = {path for path in _REVIEWED_READERS if (_REPOSITORY / path).is_file()}
    assert set(readers) == present, {
        "unreviewed": sorted(set(readers) - present),
        "stale": sorted(present - set(readers)),
    }
    for path, allowed in _READER_SCOPES.items():
        assert readers[path] <= allowed, (path, readers[path])


def test_no_solve_module_or_generator_names_a_gate_only_register(readers):
    solve = {
        _module_path(name)
        for name in (*UK_TARGET_COMPILE_MODULE_NAMES, *_SOLVE_MODULES)
    }
    assert all((_REPOSITORY / path).is_file() for path in solve), sorted(
        path for path in solve if not (_REPOSITORY / path).is_file()
    )
    assert not solve & set(readers)
    assert not set(_SOLVE_GENERATORS) & set(readers)
    assert all((_REPOSITORY / path).is_file() for path in _SOLVE_GENERATORS)


def test_the_scan_sees_a_reader_that_prose_does_not_make(tmp_path):
    """Control: a loader call, an imported loader and path literals are found;
    the same names in a docstring or a cited rationale are not."""
    module = tmp_path / "module.py"
    module.write_text(
        '"""Reads local_area_support_exclusions.json, in prose."""\n'
        "from somewhere import uk_default_target_fit_reviewed_exclusions\n"
        "def documented():\n"
        '    """uk_default_input_mass_reviewed_exclusions, in prose."""\n'
        '    return "deferred (see uk/local_area_support_exclusions.json)"\n'
        "def reader():\n"
        "    return uk_default_input_mass_reviewed_exclusions()\n"
        "class Holder:\n"
        "    def literal(self):\n"
        '        return "target_fit_reviewed_exclusions.json"\n'
        "    def path(self):\n"
        '        return "uk/local_area_support_exclusions.json"\n'
    )
    assert _references(module) == {
        "<module>",
        "reader",
        "Holder.literal",
        "Holder.path",
    }
