"""SIPP 2023 donor downloads name the download revision, never the record key.

Each reader keeps ``*_DONOR_REVISION`` as the recorded coordinate the donor
bytes were pinned and measured at, and fetches through
``*_DONOR_DOWNLOAD_REVISION``.  These checks read the reader sources, so a
download that names the record key fails here.  The record key may still name
a local cache entry: a Hugging Face snapshot path or a
``try_to_load_from_cache`` lookup, neither of which reaches the network.  The
head-start and SSI readers alias the voluntary-filing coordinate and fetcher,
so the same checks cover them.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

READERS = (
    ("sipp_financial_assets", "SIPP_2023_FINANCIAL_ASSET_DONOR"),
    ("sipp_tips", "SIPP_2023_TIP_DONOR"),
    ("sipp_vehicles", "SIPP_2023_VEHICLE_DONOR"),
    ("voluntary_filing", "SIPP_2023_VOLUNTARY_FILING_DONOR"),
)
ALIASES = (
    (
        "sipp_head_start",
        "SIPP_2023_HEAD_START_DONOR",
        "fetch_sipp_2023_head_start_donor",
    ),
    (
        "ssi_disability_criteria",
        "SIPP_2023_SSI_DISABILITY_DONOR",
        "fetch_sipp_2023_ssi_disability_donor",
    ),
)
SIPP_DONOR_MODULES = tuple(name for name, *_ in READERS + ALIASES)
DOWNLOAD_CALLS = {"hf_hub_download", "urlopen"}
# Wider net for modules that must not download at all.
ANY_DOWNLOAD_CALLS = DOWNLOAD_CALLS | {
    "build_opener",
    "hf_hub_url",
    "snapshot_download",
    "urlretrieve",
}
RECORDED_REVISION = "21280dca5995e978d706740a8a4b9b7860cfd7b6"
RECORDED_NAME = re.compile(r"SIPP_2023_\w+_DONOR_REVISION")


def _module(module_name: str):
    return importlib.import_module(f"microcosm.build.us_runtime.{module_name}")


def _tree(module_name: str) -> ast.Module:
    source = Path(_module(module_name).__file__).read_text(encoding="utf-8")
    return ast.parse(source)


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _assigned_value(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} is not assigned at module level")


def _is_recorded_name(name: str) -> bool:
    return bool(RECORDED_NAME.fullmatch(name)) and not name.endswith(
        "_DOWNLOAD_REVISION"
    )


def _recorded_names(tree: ast.Module) -> set[str]:
    """Every name a module binds to a recorded revision, import aliases included."""

    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and _is_recorded_name(node.id)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname
                for alias in node.names
                if alias.asname and _is_recorded_name(alias.name)
            )
    return names


def _division_operands(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _division_operands(node.left) + _division_operands(node.right)
    return [node]


def _cache_key_name_ids(tree: ast.Module, recorded: set[str]) -> set[int]:
    """Recorded-revision reads that only name a local cache entry or record.

    Allowed: a module-level ``*_DONOR_REVISION = <recorded name>`` alias, the
    ``revision=`` keyword of a cache lookup, and a direct operand of a ``/``
    path chain that also has ``"snapshots"`` as a direct operand.  Nothing
    nested inside a call in those places is allowed.
    """

    allowed: set[int] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id in recorded
                for target in node.targets
            )
            and isinstance(node.value, ast.Name)
        ):
            allowed.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "try_to_load_from_cache":
            allowed.update(
                id(keyword.value)
                for keyword in node.keywords
                if keyword.arg == "revision" and isinstance(keyword.value, ast.Name)
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            operands = _division_operands(node)
            if any(
                isinstance(operand, ast.Constant) and operand.value == "snapshots"
                for operand in operands
            ):
                allowed.update(
                    id(operand) for operand in operands if isinstance(operand, ast.Name)
                )
    return allowed


def _download_calls(
    tree: ast.Module, call_names: set[str] = DOWNLOAD_CALLS
) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in call_names
    ]


@pytest.mark.parametrize(("module_name", "prefix"), READERS)
def test_download_url_names_the_download_revision_only(
    module_name: str, prefix: str
) -> None:
    assert getattr(_module(module_name), f"{prefix}_DOWNLOAD_REVISION") == "1.115.5"
    url_names = _names(_assigned_value(_tree(module_name), f"{prefix}_URL"))
    assert f"{prefix}_DOWNLOAD_REVISION" in url_names
    assert f"{prefix}_REVISION" not in url_names


@pytest.mark.parametrize(("module_name", "prefix"), READERS)
def test_download_calls_never_name_the_recorded_revision(
    module_name: str, prefix: str
) -> None:
    calls = _download_calls(_tree(module_name))
    assert calls, f"{module_name} has no download call"
    for call in calls:
        assert f"{prefix}_REVISION" not in _names(call)
        if _call_name(call) == "hf_hub_download":
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            revision = keywords["revision"]
            assert isinstance(revision, ast.Name)
            assert revision.id == f"{prefix}_DOWNLOAD_REVISION"
        else:
            assert f"{prefix}_URL" in _names(call)


def test_financial_asset_reader_downloads_through_the_hub_client() -> None:
    calls = _download_calls(_tree("sipp_financial_assets"))
    assert [_call_name(call) for call in calls] == ["hf_hub_download"]


@pytest.mark.parametrize("module_name", SIPP_DONOR_MODULES)
def test_recorded_revision_is_read_only_as_a_cache_key(module_name: str) -> None:
    tree = _tree(module_name)
    recorded = _recorded_names(tree)
    allowed = _cache_key_name_ids(tree, recorded)
    stray = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id in recorded
        and isinstance(node.ctx, ast.Load)
        and id(node) not in allowed
    ]
    assert not stray, f"{module_name} reads a recorded revision at lines {stray}"


@pytest.mark.parametrize("module_name", SIPP_DONOR_MODULES)
def test_recorded_revision_literal_appears_only_in_its_assignment(
    module_name: str,
) -> None:
    tree = _tree(module_name)
    assigned = {
        id(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and all(
            isinstance(target, ast.Name) and _is_recorded_name(target.id)
            for target in node.targets
        )
    }
    stray = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and RECORDED_REVISION[:8] in node.value
        and id(node) not in assigned
    ]
    assert not stray, f"{module_name} spells the recorded revision at lines {stray}"


@pytest.mark.parametrize(("module_name", "prefix", "fetcher"), ALIASES)
def test_alias_readers_fetch_through_the_voluntary_filing_reader(
    module_name: str, prefix: str, fetcher: str
) -> None:
    voluntary_url = _module("voluntary_filing").SIPP_2023_VOLUNTARY_FILING_DONOR_URL
    assert getattr(_module(module_name), f"{prefix}_URL") == voluntary_url
    tree = _tree(module_name)
    assert not _download_calls(tree, ANY_DOWNLOAD_CALLS)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == fetcher
    )
    called = {
        _call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)
    }
    assert "fetch_sipp_2023_voluntary_filing_donor" in called


@pytest.mark.parametrize(("module_name", "prefix"), READERS)
def test_us_runtime_barrel_reexports_both_revisions(
    module_name: str, prefix: str
) -> None:
    from microcosm.build import us_runtime

    module = _module(module_name)
    for suffix in ("REVISION", "DOWNLOAD_REVISION"):
        name = f"{prefix}_{suffix}"
        assert name in us_runtime.__all__, name
        assert getattr(us_runtime, name) == getattr(module, name)
