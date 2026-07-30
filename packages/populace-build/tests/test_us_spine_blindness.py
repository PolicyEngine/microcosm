"""Structural guard for source-spine-blind US population operators."""

from __future__ import annotations

import ast
from pathlib import Path

_US_RUNTIME = (
    Path(__file__).resolve().parents[1] / "src" / "populace" / "build" / "us_runtime"
)

# These modules own source-spine provenance rather than applying population
# treatments. Keep the allowlist exact so adding a new exception requires a
# reviewed contract change.
_SOURCE_SPINE_PROVENANCE_OWNERS = frozenset(
    {
        "base_pool.py",  # Legacy late-spine assembly.
        "spine_agreement.py",  # Pre-calibration distribution comparison.
        "spine_assembly.py",  # New pre-operator assembly seam.
        "support_provenance.py",  # Centralized provenance compatibility.
        "warm_start_selection.py",  # Provenance reporting and recovery.
    }
)

_SOURCE_SPINE_COLUMNS = frozenset(
    {
        "household_spine",
        "household_support_channel",
    }
)
_US_ENTITIES = (
    "family",
    "household",
    "marital_unit",
    "person",
    "spm_unit",
    "tax_unit",
)
_OPERATOR_SOURCE_COLUMNS = frozenset(
    f"{entity}_{suffix}"
    for entity in _US_ENTITIES
    for suffix in ("spine", "support_channel")
)
_SOURCE_SPINE_COLUMN_FACTORIES = frozenset(
    {
        "spine_column",
        "support_channel_column",
    }
)
_PUF_CLONE_OPERATOR_MODULES = (
    "puf_capital_gains_tail.py",
    "puf_qrf_chain.py",
    "puf_support.py",
)
_SPINE_BLIND_OPERATOR_MODULES = (
    "adult_care.py",
    "alimony.py",
    "capital_gain_details.py",
    "child_support.py",
    "childcare.py",
    "disability_benefits.py",
    "domestic_production.py",
    "education_inputs.py",
    "educator_expenses.py",
    "energy_subsidy.py",
    "farm_business_income.py",
    "form_4952.py",
    "housing_inputs.py",
    "medicare_take_up.py",
    "other_health_insurance.py",
    "prior_year_income.py",
    "qbi_inputs.py",
    "retirement_contributions.py",
    "retirement_distributions.py",
    "salt_refund_income.py",
    "sipp_head_start.py",
    "ssi_disability_criteria.py",
    "ssi_take_up.py",
    "voluntary_filing.py",
    "weeks_unemployed.py",
    "wic_claim.py",
    "workers_compensation.py",
)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Return IDs for docstring constants, which are not executable access."""

    owners = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]
    constants: set[int] = set()
    for owner in owners:
        body = getattr(owner, "body", ())
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                constants.add(id(first.value))
    return constants


def _source_spine_accesses(source: str) -> tuple[str, ...]:
    """Describe executable references to household source-spine provenance."""

    tree = ast.parse(source)
    docstrings = _docstring_constant_ids(tree)
    accesses: set[tuple[int, int, str]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and id(node) not in docstrings
            and node.value in _SOURCE_SPINE_COLUMNS
        ):
            accesses.add(
                (
                    node.lineno,
                    node.col_offset,
                    f"literal {node.value!r}",
                )
            )
            continue
        if isinstance(node, ast.Attribute) and node.attr in _SOURCE_SPINE_COLUMNS:
            accesses.add(
                (
                    node.lineno,
                    node.col_offset,
                    f"attribute {node.attr!r}",
                )
            )
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in _SOURCE_SPINE_COLUMN_FACTORIES:
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        if any(_literal_string(argument) == "household" for argument in arguments):
            accesses.add(
                (
                    node.lineno,
                    node.col_offset,
                    f"{name}('household')",
                )
            )
    return tuple(
        f"line {line}:{column + 1}: {description}"
        for line, column, description in sorted(accesses)
    )


def _called_function_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (name := _call_name(node)) is not None
    }


def _operator_source_channel_reads(source: str) -> tuple[str, ...]:
    """Describe direct source-channel table reads in a population operator."""

    tree = ast.parse(source)
    support_column_names: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = _literal_string(node.value)
        if value not in _OPERATOR_SOURCE_COLUMNS:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                support_column_names[target.id] = value

    reads: set[tuple[int, int, str]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _call_name(node) in _SOURCE_SPINE_COLUMN_FACTORIES
        ):
            reads.add(
                (
                    node.lineno,
                    node.col_offset,
                    f"call to {_call_name(node)}",
                )
            )
            continue
        if isinstance(node, ast.Attribute) and node.attr in _OPERATOR_SOURCE_COLUMNS:
            reads.add(
                (
                    node.lineno,
                    node.col_offset,
                    f"attribute {node.attr!r}",
                )
            )
            continue
        if not isinstance(node, ast.Subscript):
            continue
        column = _literal_string(node.slice)
        if isinstance(node.slice, ast.Name):
            column = support_column_names.get(node.slice.id)
        if column in _OPERATOR_SOURCE_COLUMNS:
            reads.add(
                (
                    node.lineno,
                    node.col_offset,
                    f"subscript {column!r}",
                )
            )
    return tuple(
        f"line {line}:{column + 1}: {description}"
        for line, column, description in sorted(reads)
    )


def test_runtime_population_operators_are_source_spine_blind() -> None:
    """Only provenance owners may resolve household source-spine identity.

    The guard parses executable syntax rather than searching raw text, so
    comments and docstrings may explain the invariant without creating an
    exception. Any operator that names the concrete column, accesses it as an
    attribute, or resolves it through the canonical column helper fails.
    """

    missing_owners = sorted(
        name
        for name in _SOURCE_SPINE_PROVENANCE_OWNERS
        if not (_US_RUNTIME / name).is_file()
    )
    assert not missing_owners, (
        "Source-spine provenance-owner allowlist contains missing modules: "
        f"{missing_owners}"
    )

    offenders: dict[str, tuple[str, ...]] = {}
    for path in sorted(_US_RUNTIME.glob("*.py")):
        if path.name in _SOURCE_SPINE_PROVENANCE_OWNERS:
            continue
        accesses = _source_spine_accesses(path.read_text())
        if accesses:
            offenders[path.name] = accesses

    assert not offenders, (
        "US runtime population operators must be source-spine blind. Route "
        "PUF-detail behavior with support clone indices; source-spine "
        "provenance may be inspected only by the reviewed owner modules. "
        f"Found: {offenders}"
    )


def test_registered_population_operators_do_not_read_any_source_channel() -> None:
    """The migrated operator surface may resolve clone roles, never sources."""

    offenders: dict[str, tuple[str, ...]] = {}
    for module_name in _SPINE_BLIND_OPERATOR_MODULES:
        path = _US_RUNTIME / module_name
        assert path.is_file(), f"registered operator module is missing: {module_name}"
        reads = _operator_source_channel_reads(path.read_text())
        if reads:
            offenders[module_name] = reads
    assert not offenders, (
        "Registered population operators must use support_role_series() or "
        "support clone indices instead of source-channel columns. "
        f"Found: {offenders}"
    )


def test_puf_clone_operators_resolve_clone_index_metadata() -> None:
    """The PUF clone, imputation, and tail stages route by clone index."""

    missing_clone_index_calls: list[str] = []
    for module_name in _PUF_CLONE_OPERATOR_MODULES:
        source = (_US_RUNTIME / module_name).read_text()
        if "support_clone_index_column" not in _called_function_names(source):
            missing_clone_index_calls.append(module_name)
    assert not missing_clone_index_calls, (
        "PUF clone operators must resolve support clone-index metadata: "
        f"{missing_clone_index_calls}"
    )


def test_source_spine_ast_guard_detects_direct_and_helper_access() -> None:
    """Pin the detector itself against the two prohibited access forms."""

    direct = """
def operator(frame):
    channel = frame.table("household")["household_support_channel"]
    return channel == "acs"
"""
    via_helper = """
def operator(frame):
    column = support_channel_column("household")
    return frame.table("household")[column].eq("acs")
"""
    clone_index = """
def operator(frame):
    column = support_clone_index_column("household")
    return frame.table("household")[column].eq(1)
"""
    named_person_channel = """
PERSON_CHANNEL = "person_support_channel"
def operator(frame):
    return frame.table("person")[PERSON_CHANNEL].eq("asec")
"""

    assert _source_spine_accesses(direct)
    assert _source_spine_accesses(via_helper)
    assert _source_spine_accesses(clone_index) == ()
    assert _operator_source_channel_reads(named_person_channel)
