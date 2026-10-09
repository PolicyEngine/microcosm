"""The build package loads its public names lazily (PEP 562).

``import microcosm.build`` runs only the frame compatibility gate. Each name in
``__all__`` loads from its defining submodule on first use, so a process that
needs one submodule, such as the telemetry emitter service, does not pay for
torch, pandas and the calibration stack.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest

import microcosm.build
from microcosm.build import _EXPORTS, _EXPORTS_BY_MODULE

#: Third-party and kernel modules that only the re-exported submodules load.
_MODELLING_STACK = (
    "torch",
    "scipy",
    "pandas",
    "numpy",
    "pyarrow",
    "microcosm.frame",
    "microcosm.calibrate",
)


def _loaded(modules: set[str], roots: tuple[str, ...]) -> list[str]:
    return sorted(
        name
        for name in modules
        if any(name == root or name.startswith(f"{root}.") for root in roots)
    )


def test_export_map_names_every_public_name_once() -> None:
    """``__all__`` is the export map plus ``__version__``, with no name twice."""
    names = [name for names in _EXPORTS_BY_MODULE.values() for name in names]
    assert len(names) == len(set(names)) == len(_EXPORTS)
    assert set(_EXPORTS) | {"__version__"} == set(microcosm.build.__all__)
    assert len(microcosm.build.__all__) == len(set(microcosm.build.__all__))


def test_every_public_name_is_its_defining_modules_object() -> None:
    """Lazy resolution returns the same object the eager import used to bind."""
    for name, module in _EXPORTS.items():
        defining = import_module(f"microcosm.build.{module}")
        assert getattr(microcosm.build, name) is getattr(defining, name), name


def test_star_import_binds_every_public_name() -> None:
    namespace: dict[str, object] = {}
    exec("from microcosm.build import *", namespace)
    assert set(microcosm.build.__all__) <= set(namespace)


def test_type_checking_imports_match_the_export_map() -> None:
    """The static import block and the runtime map name the same pairs.

    Gate-binding and worker identities hash the source-import closure, which
    reads the ``TYPE_CHECKING`` block; the runtime reads ``_EXPORTS_BY_MODULE``.
    """
    tree = ast.parse(Path(microcosm.build.__file__).read_text())
    blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ]
    assert len(blocks) == 1
    static = {
        node.module.removeprefix("microcosm.build."): tuple(
            alias.name for alias in node.names
        )
        for node in blocks[0].body
        if isinstance(node, ast.ImportFrom)
    }
    assert static == _EXPORTS_BY_MODULE


def test_logbook_env_attribute_stays_the_module() -> None:
    """Only ``logbook_env_names`` is exported; the name stays the submodule."""
    import microcosm.build.logbook_env

    assert "logbook_env" not in _EXPORTS
    assert isinstance(microcosm.build.logbook_env, types.ModuleType)


def test_unknown_names_raise_attribute_error() -> None:
    """Lookups that miss raise AttributeError, as ``from`` imports require."""
    with pytest.raises(AttributeError, match="no_such_export"):
        microcosm.build.no_such_export  # noqa: B018
    assert not hasattr(microcosm.build, "__wrapped__")
    assert getattr(microcosm.build, "no_such_export", None) is None


def test_dir_lists_every_public_name() -> None:
    assert set(microcosm.build.__all__) <= set(dir(microcosm.build))


_PACKAGE_IMPORT_PROBE = r"""
import json
import sys

import microcosm.build
from microcosm.build import CHRONICLE_EPOCH, chronicle_epoch

assert chronicle_epoch.__name__ == "microcosm.build.chronicle_epoch"
assert CHRONICLE_EPOCH is chronicle_epoch.CHRONICLE_EPOCH
print(json.dumps(sorted(sys.modules)))
"""


def test_package_import_leaves_the_modelling_stack_unloaded() -> None:
    """A fresh interpreter importing the package loads no modelling stack.

    ``chronicle_epoch`` imports only the standard library, so loading one of its
    exports, or the submodule by name, must not pull the other exports in.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PACKAGE_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = set(json.loads(completed.stdout.splitlines()[-1]))
    assert not _loaded(loaded, _MODELLING_STACK)
    assert not _loaded(loaded, ("microcosm.build.gates", "microcosm.build.plan"))
