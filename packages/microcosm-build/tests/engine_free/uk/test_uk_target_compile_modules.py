"""Both UK target kernels hash every module that decides a target value."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from microcosm.build import ledger_targets as shared_ledger_targets
from microcosm.build.uk_runtime import graph_national, graph_targets, hmrc_uprating
from microcosm.build.uk_runtime.target_compile_modules import (
    EXCLUDED_DIRECT_IMPORTS,
    UK_TARGET_COMPILE_MODULE_NAMES,
    uk_target_compile_modules,
)


def _direct_build_imports(module_name: str) -> set[str]:
    """Every ``microcosm.build`` module the named module imports, by AST."""

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    package = module_name.rsplit(".", 1)[0]
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                stem = f"{base}.{node.module}" if node.module else base
            else:
                stem = node.module or ""
            if not stem.startswith("microcosm.build"):
                continue
            candidates.add(stem)
            candidates.update(f"{stem}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            candidates.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("microcosm.build")
            )
    modules: set[str] = set()
    for name in candidates:
        try:
            imported = importlib.import_module(name)
        except ImportError:
            continue
        source = getattr(imported, "__file__", None)
        if source is not None and not source.endswith("__init__.py"):
            modules.add(name)
    return modules


def test_every_direct_import_of_a_listed_module_is_listed_or_excluded():
    listed = set(UK_TARGET_COMPILE_MODULE_NAMES)
    uncovered = {
        f"{name} -> {imported}"
        for name in UK_TARGET_COMPILE_MODULE_NAMES
        for imported in _direct_build_imports(name)
        if imported not in listed and imported not in EXCLUDED_DIRECT_IMPORTS
    }
    assert not uncovered, sorted(uncovered)


def test_exclusions_are_live_and_never_listed():
    imported = set().union(
        *(_direct_build_imports(name) for name in UK_TARGET_COMPILE_MODULE_NAMES)
    )
    assert set(EXCLUDED_DIRECT_IMPORTS) <= imported
    assert not set(EXCLUDED_DIRECT_IMPORTS) & set(UK_TARGET_COMPILE_MODULE_NAMES)


def _hashed_by(kernel, monkeypatch, module) -> list:
    hashed: list = []
    monkeypatch.setattr(
        module, "source_hash", lambda *objects: hashed.extend(objects) or "test"
    )
    kernel.implementation_hash()
    return hashed


def test_full_target_kernel_hashes_the_shared_resolver_and_the_appliers(
    monkeypatch,
):
    hashed = _hashed_by(
        graph_targets.UKFullTargetCompilationKernel(), monkeypatch, graph_targets
    )
    assert shared_ledger_targets in hashed
    assert hmrc_uprating in hashed
    assert set(uk_target_compile_modules()) <= set(hashed)


def test_national_target_kernel_hashes_the_same_compile_modules(monkeypatch):
    hashed = _hashed_by(
        graph_national.UKNationalTargetKernel(), monkeypatch, graph_national
    )
    assert set(uk_target_compile_modules()) <= set(hashed)
