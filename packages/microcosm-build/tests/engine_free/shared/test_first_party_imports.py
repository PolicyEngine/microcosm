"""Controls for the import scan behind the compile kernels' hash checks.

``test_support.microcosm_build.first_party_imports`` reads a module's source
for the first-party modules it imports, and walks a kernel's hashed modules
through them. These tests pin each import form it resolves, that a
submodule resolves the same way before and after it is imported, and how the
walk treats a shared list's exclusions.
"""

from __future__ import annotations

import importlib
import sys
import uuid
from types import SimpleNamespace

import pytest

import microcosm.calibrate
from test_support.microcosm_build.first_party_imports import (
    defining_module,
    first_party_imports,
    recorded_import_mismatches,
    unhashed_imports,
)


@pytest.fixture
def scratch_package(tmp_path, monkeypatch):
    """A throwaway ``microcosm.<name>`` package written under ``tmp_path``.

    ``microcosm`` is a namespace package, so a directory on ``sys.path``
    adds to it. Returns a writer: ``write(module, source)`` creates
    ``microcosm.<name>.<module>`` and returns its dotted name.
    """

    name = f"scan_control_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(module: str, source: str) -> str:
        (package / f"{module}.py").write_text(source)
        return f"microcosm.{name}.{module}"

    yield write
    for loaded in [key for key in sys.modules if key.startswith(f"microcosm.{name}")]:
        del sys.modules[loaded]


def test_the_scan_sees_each_import_form_the_compile_modules_use():
    # A constant imported from its own module.
    assert "microcosm.calibrate.geography_constants" in first_party_imports(
        "microcosm.build.ledger_targets"
    )
    # A class imported from a package resolves to the module defining it.
    assert "microcosm.calibrate.registry" in first_party_imports(
        "microcosm.build.ledger_targets"
    )
    # A constant the package __init__ re-exports resolves through the __init__.
    assert "microcosm.diagnostics.schema" in first_party_imports(
        "microcosm.build.transport.artifact_types"
    )
    # An import inside a function body.
    assert "microcosm.build.reference_intervals" in first_party_imports(
        "microcosm.build.reference_profiles"
    )


def test_a_package_import_resolves_through_the_names_read_from_it(scratch_package):
    module = scratch_package(
        "package_imports",
        "import microcosm.calibrate\n"
        "import microcosm.diagnostics as diagnostics_package\n"
        "def read():\n"
        "    registry = microcosm.calibrate.TargetRegistry\n"
        "    return registry, diagnostics_package.CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION\n",
    )
    assert first_party_imports(module) == {
        "microcosm.calibrate.registry",
        "microcosm.diagnostics.schema",
    }


def test_a_package_import_with_no_name_read_reads_the_package(scratch_package):
    module = scratch_package("bare_import", "import microcosm.calibrate\n")
    assert first_party_imports(module) == {"microcosm.calibrate"}


def test_a_submodule_resolves_the_same_before_it_is_imported(monkeypatch):
    """A function-level ``from package import submodule`` names the submodule
    whether or not an earlier test imported it."""
    name = "microcosm.calibrate.target_selection"
    assert defining_module("microcosm.calibrate", "target_selection") == name
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(microcosm.calibrate, "target_selection", raising=False)
    assert defining_module("microcosm.calibrate", "target_selection") == name


def test_the_walk_follows_an_exclusion_but_not_its_imports(scratch_package):
    """A list's exclusion covers only the listed module's import of it; the
    excluded module's own imports must be hashed."""
    leaf = scratch_package("leaf", "VALUE = 1\n")
    middle = scratch_package("middle", f"from {leaf} import VALUE\n")
    top = scratch_package("top", f"from {middle} import VALUE\n")
    excluded = frozenset({middle})
    assert unhashed_imports({top}, listed=(top,), excluded=excluded) == {
        f"{middle} -> {leaf}"
    }
    # From a module outside the list, the exclusion does not apply.
    assert unhashed_imports({top}, listed=(), excluded=excluded) == {
        f"{top} -> {middle}"
    }
    assert unhashed_imports({top, middle, leaf}, listed=(), excluded=excluded) == set()


class _RecordingKernel:
    """A stand-in kernel whose hash names ``modules`` through ``owner``."""

    def __init__(self, ref, owner, modules):
        self.ref, self.owner, self.modules = ref, owner, modules

    def implementation_hash(self):
        return self.owner.source_hash(*self.modules)


def test_a_group_record_does_not_cover_another_group(scratch_package):
    """Removing a module from one group's hash fails the check even when
    another group records the same edge (round 2 of #1493)."""
    middle = scratch_package("group_middle", "VALUE = 1\n")
    top = scratch_package("group_top", f"from {middle} import VALUE\n")
    owner = SimpleNamespace(source_hash=lambda *objects, **dependencies: "x")
    top_module, middle_module = (
        importlib.import_module(name) for name in (top, middle)
    )
    hashes_both = _RecordingKernel("hashes.both@1", owner, (top_module, middle_module))
    hashes_top = _RecordingKernel("hashes.top@1", owner, (top_module,))
    recorded = {f"{top} -> {middle}": ("top only",)}

    def mismatches(groups):
        return recorded_import_mismatches(
            groups, listed=(), excluded=frozenset(), hash_inputs={}, recorded=recorded
        )

    assert (
        mismatches(
            {"both": ((hashes_both, owner),), "top only": ((hashes_top, owner),)}
        )
        == {}
    )
    # The first group drops the middle module: the edge is recorded for the
    # second group alone, so the first group's walk reports it.
    assert mismatches(
        {"both": ((hashes_top, owner),), "top only": ((hashes_top, owner),)}
    ) == {"both: unhashed and not recorded": [f"{top} -> {middle}"]}
