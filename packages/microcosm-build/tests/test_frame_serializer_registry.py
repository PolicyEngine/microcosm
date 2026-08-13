from __future__ import annotations

import ast
from pathlib import Path

from microcosm.build.frame_serializer_registry import (
    FRAME_TABLE_SERIALIZERS,
    HDF_WRITE_EXCLUSIONS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOTS = (
    REPOSITORY_ROOT / "packages",
    REPOSITORY_ROOT / "tools",
)


def _qualified_call_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    if not isinstance(call.func.value, ast.Name):
        return None
    return f"{call.func.value.id}.{call.func.attr}"


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def _literal_hdf_mode(call: ast.Call) -> str:
    mode_node: ast.AST | None = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
            break
    if mode_node is None:
        return "a"
    value = ast.literal_eval(mode_node)
    if not isinstance(value, str):
        raise AssertionError("Production HDF modes must be literal strings.")
    return value


def _discover_writable_hdf_sites() -> set[str]:
    discovered: set[str] = set()
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _qualified_call_name(node) not in {"h5py.File", "pd.HDFStore"}:
                    continue
                if _literal_hdf_mode(node) == "r":
                    continue
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                function = _enclosing_function(node, parents)
                discovered.add(f"{relative}::{function}")
    return discovered


def test_registry_classifies_every_writable_production_hdf_site() -> None:
    classified = {
        spec.writer.key for spec in FRAME_TABLE_SERIALIZERS if spec.direct_hdf_open
    }
    classified.update(exclusion.writer.key for exclusion in HDF_WRITE_EXCLUSIONS)
    assert _discover_writable_hdf_sites() == classified


def test_registry_has_exactly_eight_unique_frame_table_serializers() -> None:
    assert len(FRAME_TABLE_SERIALIZERS) == 8
    assert len({spec.serializer_id for spec in FRAME_TABLE_SERIALIZERS}) == 8
    assert len({spec.writer.key for spec in FRAME_TABLE_SERIALIZERS}) == 8


def test_indirect_policyengine_us_sink_remains_a_dataset_save_call() -> None:
    (spec,) = (
        candidate
        for candidate in FRAME_TABLE_SERIALIZERS
        if candidate.serializer_id == "policyengine_us_dataset"
    )
    path = REPOSITORY_ROOT / spec.writer.path
    tree = ast.parse(path.read_text())
    save_functions: set[str] = set()
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "save":
            continue
        save_functions.add(_enclosing_function(node, parents))
    assert spec.direct_hdf_open is False
    assert spec.writer.function in save_functions


def test_no_production_dataframe_to_hdf_sink_bypasses_registry() -> None:
    sites: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to_hdf"
                ):
                    sites.append(
                        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}:{node.lineno}"
                    )
    assert sites == []
