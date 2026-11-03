"""Controls for the import scan behind the compile kernels' hash checks.

``test_support.microcosm_build.first_party_imports`` reads a module's source
for the first-party modules it imports, and walks a kernel's hashed modules
through them. These tests pin each import form it resolves, that a
submodule resolves the same way before and after it is imported, and how the
walk treats a shared list's exclusions.
"""

from __future__ import annotations

import sys
import uuid

import pytest

import microcosm.calibrate
from test_support.microcosm_build.first_party_imports import (
    defining_module,
    first_party_imports,
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
